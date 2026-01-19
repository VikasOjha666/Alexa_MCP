import asyncio
import json
from typing import List, Dict, Any, Optional,Tuple
from fastapi import FastAPI, HTTPException,Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from llama_cpp import Llama
from llama_index.tools.mcp import BasicMCPClient, McpToolSpec
from utils import  sanitize_for_json,get_prompt, call_mcp_sse,get_server_for_tool,try_parse_function_call
from mcp_tools_ret_utils import index_tools_to_lancedb, fetch_top_k_tools_formatted
import re
import os
MCP_SERVER_URLS = os.environ.get('MCP_SERVER_URLS', '')
if MCP_SERVER_URLS:
    MCP_SERVER_URLS = MCP_SERVER_URLS.split(',')

MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    "gorilla-openfunctions-v2-GGUF/gorilla-openfunctions-v2-q4_K_M.gguf"
)

app = FastAPI(title="MCP")

# Globals populated at startup
server_map_dict: Dict[str, List[Dict[str, Any]]] = {}
llm: Optional[Llama] = None


class QueryRequest(BaseModel):
    query: str
    k: int = 4



async def discover_tools(server_urls: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Discover tools on each MCP server URL and return a mapping:
      { server_url: [ {name, description, parameters}, ... ], ... }
    This is adapted from your provided snippet; it is defensive about missing metadata.
    """
    server_map_dict_local: Dict[str, List[Dict[str, Any]]] = {}

    for server_url in server_urls:
        functions: List[Dict[str, Any]] = []
        try:
            client = BasicMCPClient(server_url)
            tool_spec = McpToolSpec(client=client)

            # Wait for tool list asynchronously with a timeout
            tools = await asyncio.wait_for(tool_spec.to_tool_list_async(), timeout=10)

            for tool in tools:
                meta = getattr(tool, "_metadata", None)
                if meta is None:
                    # fallback: try to access tool metadata properties defensively
                    name = getattr(tool, "name", None) or getattr(tool, "fn", None) or "unknown"
                    description = getattr(tool, "description", "") or ""
                    params_raw = getattr(tool, "fn_schema_str", None)
                else:
                    name = getattr(meta, "name", None) or "unknown"
                    description = getattr(meta, "description", "") or ""
                    params_raw = getattr(meta, "fn_schema_str", None)

                # Try to parse parameters if it's a JSON string; otherwise keep as-is
                params = None
                try:
                    if isinstance(params_raw, str):
                        params = json.loads(params_raw)
                    else:
                        params = params_raw
                except Exception as e:
                    print(f"[discover_tools] Warning: failed to parse fn_schema_str for tool '{name}': {e}")
                    params = params_raw

                func_dict = {
                    "name": name,
                    "description": description,
                    "parameters": params,
                }
                functions.append(func_dict)

        except asyncio.TimeoutError:
            print(f"[discover_tools] Timeout while fetching tools from {server_url}")
        except Exception as e:
            print(f"[discover_tools] Error while contacting {server_url}: {e}")

        # Always set an entry (possibly empty) for this server_url
        server_map_dict_local[server_url] = functions

    return server_map_dict_local


@app.on_event("startup")
async def startup_event():
    """Discover tools and index them at startup. Also create the Llama instance once."""
    global server_map_dict, llm

    # Discover MCP tools (async)
    try:
        server_map_dict = await discover_tools(MCP_SERVER_URLS)
        print("Discovered tools:", {k: len(v) for k, v in server_map_dict.items()})
    except Exception as e:
        print("Warning: discover_tools failed at startup:", e)

    # Index tools into lancedb (safe to call even if server_map_dict is empty)
    try:
        index_tools_to_lancedb(server_map_dict, db_path="./mcp_tools_lancedb", overwrite=True)
        print("Indexed tools to lancedb")
    except Exception as e:
        print("Warning: indexing to lancedb failed:", e)

    # Instantiate Llama in a threadpool to avoid blocking the event loop
    loop = asyncio.get_event_loop()

    def _create_llm():
        print("Creating Llama model with GPU support (RTX 4060)...")

        cpu_threads = min(8, os.cpu_count() or 8)

        llm = Llama(
            model_path=MODEL_PATH,

            # 🔥 GPU settings
            n_gpu_layers=-1,          # IMPORTANT: offload all layers to GPU
            n_batch=512,              # Sweet spot for Ada GPUs
            n_ctx=2048,

            # 🧠 CPU settings
            n_threads=cpu_threads,

            # 🧩 Memory / performance
            use_mmap=True,
            use_mlock=False,

            # 🐛 Debug / verification
            verbose=True,
        )

        print("Llama initialized. If GPU is active, you should see CUDA logs above.")
        return llm

    try:
        llm = await loop.run_in_executor(None, _create_llm)
        print("Llama model created")
    except Exception as e:
        print("Error creating Llama model:", e)
        llm = None

    #Verify GPU.
    if llm is not None:
        try:
            print("GPU info check:")
            test_out = llm("Hello", max_tokens=1)
            print("GPU test token OK")
        except Exception as e:
            print("GPU test failed:", e)
        
    print(f"server_map_dict={server_map_dict}")

async def _call_llm(prompt: str, max_tokens: int = 2048) -> str:
    """Run the synchronous llm(prompt) in a threadpool and return the text string."""
    global llm
    if llm is None:
        raise RuntimeError("LLM not initialized")

    loop = asyncio.get_event_loop()

    def _sync_call():
        out = llm(prompt, max_tokens=max_tokens, echo=False)
        if isinstance(out, dict):
            return out.get("choices", [{}])[0].get("text") or out.get("text") or str(out)
        return str(out)

    return await loop.run_in_executor(None, _sync_call)


async def execute_query_pipeline(query: str, k: int = 2):
    """
    Shared MCP + LLM execution pipeline.
    Returns a dict with final_text, tool_called, tool_result, raw_model_output.
    """
    try:
        # Step 1: retrieve candidate tools
        functions = fetch_top_k_tools_formatted(query, k=k)
    except Exception as e:
        raise RuntimeError(f"Error fetching tools from index: {e}")

    safe_functions = sanitize_for_json(functions)
    prompt = get_prompt(query, safe_functions)

    # Step 3: call LLM
    raw_text = await _call_llm(prompt, max_tokens=2048)

    # Step 4: detect function call
    call = try_parse_function_call(raw_text)

    if call is None:
        return {
            "final_text": raw_text,
            "tool_called": None,
            "tool_result": None,
            "raw_model_output": raw_text,
        }

    func_name, func_args = call

    # Step 5: execute MCP tool via SSE
    url_to_call = get_server_for_tool(func_name, server_map_dict)
    tool_result = await call_mcp_sse(func_name, func_args, url=url_to_call)

    # Step 6: follow-up LLM call
    followup_prompt = (
        prompt
        + "\n\n(The tool call has been executed.)\n"
        + f"Tool name: {func_name}\n"
        + f"Tool output: {tool_result}\n"
        + "Now the result needs to be formatted in a human readable tone.\n"
        + "Assistant:"
    )

    try:
        final_text = await _call_llm(followup_prompt, max_tokens=2048)
    except Exception:
        final_text = None

    return {
        "final_text": final_text,
        "tool_called": func_name,
        "tool_result": sanitize_for_json(tool_result),
        "raw_model_output": raw_text,
    }


@app.post("/query")
async def run_query(req: QueryRequest):
    try:
        result = await execute_query_pipeline(req.query, k=req.k)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/alexa")
async def alexa(req: Request):
    body = await req.json()
    req_type = body["request"]["type"]

    if req_type == "LaunchRequest":
        return {
            "version": "1.0",
            "response": {
                "outputSpeech": {
                    "type": "PlainText",
                    "text": "Hi, you can ask me anything."
                },
                "shouldEndSession": False
            }
        }

    if req_type == "IntentRequest":
        intent = body["request"]["intent"]
        slots = intent.get("slots", {})

        query = (
            slots.get("query", {})
            .get("value")
        )

        if not query:
            return {
                "version": "1.0",
                "response": {
                    "outputSpeech": {
                        "type": "PlainText",
                        "text": "What would you like me to help you with?"
                    },
                    "shouldEndSession": False
                }
            }

        print(f"query={query}")

        try:
            result = await execute_query_pipeline(query, k=4)
            final_text = result.get("final_text") or "Sorry, I could not process that request."
        except Exception as e:
            final_text = "Something went wrong while processing your request."

        return {
            "version": "1.0",
            "response": {
                "outputSpeech": {
                    "type": "PlainText",
                    "text": final_text
                },
                "shouldEndSession": True
            }
        }



if __name__ == '__main__':
    import uvicorn

    # Run with: python fastapi_mcp_endpoint.py  OR: uvicorn fastapi_mcp_endpoint:app --reload
    uvicorn.run(app, host='0.0.0.0', port=3000, log_level='info')
