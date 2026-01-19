import asyncio
import json
import re
import ast
from typing import Dict, List, Optional, Union, Any,Tuple
import aiohttp
from urllib.parse import urljoin
from utils import *
import numpy as np
import copy


def build_tool_index(server_map: Dict[str, List[Dict]]) -> Dict[str, List[str]]:
    """
    Build an index mapping tool name -> list of server URLs that provide that tool.
    """
    index: Dict[str, List[str]] = {}
    for server_url, tools in server_map.items():
        for tool in tools:
            name = tool.get("name")
            if not name:
                continue
            index.setdefault(name, []).append(server_url)
    return index


def try_parse_function_call(text: str) -> Optional[Tuple[str, dict]]:
    """
    Robust function call extractor.
    Handles noisy model outputs like:
      <<<CALL_FUNCTION>>>subtract(a=2, b=3)
      <<function>> add(x=1,y=2)
      function_call: multiply(a=2)
    """
    if not text:
        return None

    txt = text.strip()

    # Flexible patterns
    patterns = [
        
        r"<+ *CALL_FUNCTION *>+\s*([A-Za-z0-9_]+)\s*\((.*)\)",
        
        r"<+ *function *>+\s*([A-Za-z0-9_]+)\s*\((.*)\)",
        
        r"CALL_FUNCTION\s*:?\s*([A-Za-z0-9_]+)\s*\((.*)\)",
        
        r"([A-Za-z0-9_]+)\s*\((.*)\)"
    ]

    for p in patterns:
        m = re.search(p, txt, flags=re.IGNORECASE | re.DOTALL)
        if not m:
            continue

        func_name = m.group(1)
        args_str = m.group(2).strip()
        args = {}

        # No args
        if args_str == "":
            return func_name, args

        # Split arguments by commas NOT inside quotes
        parts = [s.strip() for s in re.split(r',(?=(?:[^"]*"[^"]*")*[^"]*$)', args_str) if s.strip()]

        for part in parts:
            if "=" in part:
                k, v = part.split("=", 1)
                k = k.strip()
                v = v.strip()

                try:
                    parsed_v = ast.literal_eval(v)
                except Exception:
                    parsed_v = v.strip('"').strip("'")

                args[k] = parsed_v

            else:
                # positional args
                args.setdefault("_args", []).append(part)

        return func_name, args

    return None


def get_server_for_tool(
    tool_name: str,
    server_map: Dict[str, List[Dict]],
    *,
    first_only: bool = True,
    case_sensitive: bool = False
) -> Optional[Union[str, List[str]]]:
    """
    Return the server URL(s) that provide the tool `tool_name`.

    Args:
      tool_name: name of the tool to look up (e.g. "add").
      server_map: dict mapping server_url -> list of tool dicts (like your server_map_dict).
      first_only: if True (default) return the first matching URL (string).
                  if False return a list of all matching URLs (could be empty).
      case_sensitive: if False (default) match tool name case-insensitively.

    Returns:
      If first_only=True -> a string (server URL) or None if not found.
      If first_only=False -> list of matching URLs (empty list if none found).
    """
    # Build index (fast for repeated lookups)
    index = build_tool_index(server_map)

    if not case_sensitive:
        # build a case-insensitive mapping
        ci_index = {}
        for name, urls in index.items():
            ci_index[name.lower()] = ci_index.get(name.lower(), []) + urls
        matches = ci_index.get(tool_name.lower(), [])
    else:
        matches = index.get(tool_name, [])

    if first_only:
        return matches[0] if matches else None
    return matches


def sanitize_for_json(obj: Any) -> Any:
    """
    Recursively convert NumPy arrays/scalars and other common non-JSON types
    to JSON-native Python types.
    - numpy.ndarray -> list
    - numpy scalar (np.generic) -> native python via .item()
    - bytes/bytearray -> utf-8 str (or list of ints if decode fails)
    """
    # numpy array -> list
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    # numpy scalar -> native Python (works across numpy versions)
    if isinstance(obj, np.generic):
        try:
            return obj.item()
        except Exception:
            # fallback: convert to Python scalar string
            return str(obj)

    # bytes -> try decode
    if isinstance(obj, (bytes, bytearray)):
        try:
            return obj.decode("utf-8")
        except Exception:
            return list(obj)

    # dict/list/tuple -> recurse
    if isinstance(obj, dict):
        return {sanitize_for_json(k): sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(sanitize_for_json(v) for v in obj)

    # fallback: assume JSON-serializable already
    return obj


def get_prompt(user_query: str, functions: list = []) -> str:
    system = (
        "You are a LLM you can answer questions. If you are not sure about it and you believe you can use a tool then you can call it.\n"
        "If you need to call a tool, output EXACTLY the marker >>>CALL_FUNCTION<<< followed\n"
        "by a single JSON object with keys 'name' and 'arguments' on the next line.\n"
        "Example:\n"
        ">>>CALL_FUNCTION<<<\n"
        "{\"name\": \"get_current_weather\", \"arguments\": {\"location\": \"Boston, MA\", \"unit\": \"celsius\"}}\n"
        "If you do not need to call a function, just answer normally.\n",
        "Also don't give example how to call the function but give the exact marker defined above."
    )
    functions_string = json.dumps(functions, indent=2)
    return f"{system}\nAvailable Functions:\n{functions_string}\nUser: {user_query}\nAssistant:"



def _find_json_block(s: str):
    """
    Locate the first JSON object in the string by finding the first '{' and
    scanning forward until the braces are balanced. Handles quoted strings
    and escaped quotes inside those strings.
    """
    start = s.find('{')
    if start == -1:
        return None

    depth = 0
    in_string = None
    escape = False

    for i in range(start, len(s)):
        ch = s[i]

        if in_string is not None:
            # We're inside a quoted string
            if escape:
                # previous char was backslash, so this char is escaped
                escape = False
            elif ch == "\\":
                # start an escape for the next char
                escape = True
            elif ch == in_string:
                # closing quote
                in_string = None
            # otherwise remain inside the string
        else:
            # not inside a string
            if ch == '"' or ch == "'":
                in_string = ch
            elif ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    return s[start:i+1]

    return None




def sanitize_for_json(obj: Any) -> Any:
    """
    Recursively convert numpy arrays/scalars and basic non-JSON types to JSON-native types.
    """
    # numpy arrays -> lists
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    # numpy scalars -> native Python types
    if isinstance(obj, (np.integer, np.int_)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    if isinstance(obj, (np.bool_)):
        return bool(obj)
    # bytes -> decode if possible
    if isinstance(obj, (bytes, bytearray)):
        try:
            return obj.decode("utf-8")
        except Exception:
            return list(obj)
    # dict/list/tuple -> recurse
    if isinstance(obj, dict):
        return {sanitize_for_json(k): sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(sanitize_for_json(v) for v in obj)
    # fallback: leave as-is
    return obj




def _split_args_top_level(args_str: str):
    # split on commas that are not inside quotes or parentheses
    parts = re.split(r',(?=(?:[^\"\']*[\"\'][^\"\']*[\"\'])*[^\"\']*$)', args_str)
    # the regex above is a heuristic — trim parts
    return [p.strip() for p in parts if p.strip()]


async def call_mcp_sse(name: str, arguments: dict, url: str = "http://localhost:8000/sse",
                                 init_timeout: int = 10, call_timeout: int = 60):
    """
    MCP SSE flow with a properly populated `initialize` payload.
    Steps:
      1) GET /sse and wait for event: endpoint -> data: /messages/?session_id=...
      2) POST initialize (id=0) with required params {protocolVersion, capabilities, clientInfo}
         and wait for its JSON-RPC response on the SSE stream.
      3) POST tools/call (id=1)
      4) Read SSE until JSON-RPC response for id=1 which contains the tool result
    """

    rpc_initialize_id = 0
    rpc_call_id = 1

    # <-- IMPORTANT: supply required fields under params -->
    init_params = {
        "protocolVersion": "1.0",
        # Capabilities: adjust as needed for your client. Keep keys conservative.
        "capabilities": {
            "supportsToolCalls": True,
            "supportsStreaming": True,
            # add other flags your client supports if desired
        },
        "clientInfo": {
            "name": "simple-python-client",
            "version": "0.1",
            "description": "Debug client for MCP SSE calls"
        }
    }

    init_payload = {"jsonrpc": "2.0", "id": rpc_initialize_id, "method": "initialize", "params": init_params}
    call_payload = {"jsonrpc": "2.0", "id": rpc_call_id, "method": "tools/call", "params": {"name": name, "arguments": arguments}}

    async def _sse_lines(resp):
        while True:
            line = await resp.content.readline()
            if not line:
                break
            yield line.decode("utf-8").rstrip("\r\n")

    headers = {"Accept": "text/event-stream"}
    async with aiohttp.ClientSession() as session:
        # 1) Open SSE GET
        async with session.get(url, headers=headers, timeout=init_timeout) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise RuntimeError(f"SSE GET failed: status={resp.status}, body={body}")

            # read until endpoint event
            buf_event = None
            buf_data_lines = []
            messages_path = None
            async for raw in _sse_lines(resp):
                s = raw
                if s == "":
                    if buf_event is not None:
                        data_str = "\n".join(buf_data_lines)
                        if buf_event == "endpoint":
                            messages_path = data_str.strip()
                            break
                    buf_event = None
                    buf_data_lines = []
                    continue
                if s.startswith("event:"):
                    buf_event = s[len("event:"):].strip()
                elif s.startswith("data:"):
                    buf_data_lines.append(s[len("data:"):].lstrip())
                else:
                    # ignore other lines
                    pass

            if not messages_path:
                raise RuntimeError("Did not receive endpoint event from SSE GET (no /messages path).")
            full_messages_url = urljoin(url, messages_path)

            # 2) POST initialize
            async with session.post(full_messages_url, json=init_payload, headers={"Content-Type": "application/json"}, timeout=10) as post_init:
                if post_init.status not in (200, 202):
                    txt = await post_init.text()
                    raise RuntimeError(f"POST initialize failed: status={post_init.status}, body={txt}")
                # server accepted initialize; wait for JSON-RPC response via SSE

            # wait for initialize response (id == rpc_initialize_id)
            init_deadline = asyncio.get_event_loop().time() + init_timeout
            buf_event = None
            buf_data_lines = []
            init_ok = False
            init_response = None
            async for raw in _sse_lines(resp):
                now = asyncio.get_event_loop().time()
                if now > init_deadline:
                    break
                s = raw
                if s == "":
                    if buf_event is not None:
                        data_str = "\n".join(buf_data_lines)
                        try:
                            parsed = json.loads(data_str)
                        except Exception:
                            parsed = None
                        if isinstance(parsed, dict) and parsed.get("id") == rpc_initialize_id:
                            init_response = parsed
                            init_ok = True
                            break
                    buf_event = None
                    buf_data_lines = []
                    continue
                if s.startswith("event:"):
                    buf_event = s[len("event:"):].strip()
                elif s.startswith("data:"):
                    buf_data_lines.append(s[len("data:"):].lstrip())

            if not init_ok:
                # dump a helpful error
                raise RuntimeError(f"Did not receive initialize response before timeout. Last init_response: {init_response}")

            # Optional: inspect the initialize result for diagnostics
            # print("Initialize response:", json.dumps(init_response, indent=2))

            # 3) POST tools/call
            async with session.post(full_messages_url, json=call_payload, headers={"Content-Type": "application/json"}, timeout=10) as call_resp:
                if call_resp.status not in (200, 202):
                    txt = await call_resp.text()
                    raise RuntimeError(f"POST tools/call failed: status={call_resp.status}, body={txt}")

            # 4) wait for tools/call response (id == rpc_call_id)
            call_deadline = asyncio.get_event_loop().time() + call_timeout
            buf_event = None
            buf_data_lines = []
            async for raw in _sse_lines(resp):
                now = asyncio.get_event_loop().time()
                if now > call_deadline:
                    break
                s = raw
                if s == "":
                    if buf_event is not None:
                        data_str = "\n".join(buf_data_lines)
                        try:
                            parsed = json.loads(data_str)
                        except Exception:
                            parsed = None
                        if isinstance(parsed, dict) and parsed.get("id") == rpc_call_id and "result" in parsed:
                            res = parsed["result"]
                            # try to extract text content
                            if isinstance(res, dict) and "content" in res:
                                texts = []
                                for item in res["content"]:
                                    if isinstance(item, dict) and item.get("type") == "text" and "text" in item:
                                        texts.append(item["text"])
                                return "\n".join(texts) if texts else json.dumps(res)
                            if isinstance(res, str):
                                return res
                            return json.dumps(res)
                    buf_event = None
                    buf_data_lines = []
                    continue
                if s.startswith("event:"):
                    buf_event = s[len("event:"):].strip()
                elif s.startswith("data:"):
                    buf_data_lines.append(s[len("data:"):].lstrip())

            raise RuntimeError("Timed out waiting for tools/call JSON-RPC response over SSE.")


