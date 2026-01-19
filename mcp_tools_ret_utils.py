# mcp_tools_ret_utils.py (replacement functions)
import json
import os
from typing import Dict, List, Any

import numpy as np
from sentence_transformers import SentenceTransformer
import lancedb

from lancedb import connect

# Optional imports for typed vector schema fallback
try:
    from lancedb.pydantic import Vector, LanceModel  # newer versions
    HAVE_PYDANTIC = True
except Exception:
    HAVE_PYDANTIC = False

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # change if you use another embedder


def _make_tool_text(tool: Dict[str, Any]) -> str:
    name = tool.get("name", "")
    desc = tool.get("description", "") or ""
    params = tool.get("parameters", None)
    try:
        params_str = json.dumps(params, ensure_ascii=False)
    except Exception:
        params_str = str(params)
    return f"{name}. {desc}. parameters: {params_str}"


def _ensure_json_schema(params: Any) -> Dict[str, Any]:
    if isinstance(params, dict):
        return params
    if isinstance(params, str):
        try:
            parsed = json.loads(params)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
    return {"type": "object", "properties": {}}


def print_lancedb_debug(db):
    print("lancedb version:", getattr(lancedb, "__version__", "unknown"))
    print("db object methods:", [m for m in dir(db) if not m.startswith("_")])


def index_tools_to_lancedb(
    server_map_dict: Dict[str, List[Dict[str, Any]]],
    db_path: str = "./mcp_tools_lancedb",
    table_name: str = "mcp_tools",
    embedding_model_name: str = EMBED_MODEL_NAME,
    overwrite: bool = False,
) -> None:
    """
    Index discovered tools into LanceDB in a portable way that works across versions.
    The vector column will be stored under the key "embedding" (list[float]).
    """
    model = SentenceTransformer(embedding_model_name)

    rows = []
    for server_url, tools in server_map_dict.items():
        for i, tool in enumerate(tools):
            name = tool.get("name") or f"tool_{i}"
            doc_id = f"{server_url}::{name}::{i}"
            params = _ensure_json_schema(tool.get("parameters"))
            text = _make_tool_text({"name": name, "description": tool.get("description", ""), "parameters": params})
            rows.append(
                {
                    "id": doc_id,
                    "server_url": server_url,
                    "tool_name": name,
                    "description": tool.get("description", "") or "",
                    "parameters": params,
                    "text": text,
                    # embedding is added below
                }
            )

    if not rows:
        print("[index_tools_to_lancedb] no tools to index.")
        return

    texts = [r["text"] for r in rows]
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True).astype(np.float32)
    for r, emb in zip(rows, embeddings):
        # Convert to Python list so older lancedb versions accept it
        r["embedding"] = emb.tolist()

    os.makedirs(db_path, exist_ok=True)
    db = connect(db_path)

    # debug info (uncomment if you want to inspect)
    # print_lancedb_debug(db)

    # Try to remove existing table in a portable way
    if overwrite:
        # Some versions expose drop_table or delete_table
        if hasattr(db, "drop_table"):
            try:
                db.drop_table(table_name)
                print(f"[index_tools_to_lancedb] dropped existing table '{table_name}' (overwrite=True).")
            except Exception:
                pass
        elif hasattr(db, "delete_table"):
            try:
                db.delete_table(table_name)
                print(f"[index_tools_to_lancedb] deleted existing table '{table_name}' (overwrite=True).")
            except Exception:
                pass

    # Try several create_table signatures used across releases
    created = False
    create_exceptions = []
    try:
        # common simple form: create_table(name, data)
        db.create_table(table_name, rows)
        created = True
    except Exception as e1:
        create_exceptions.append(e1)
        try:
            # alternate: create_table(name=..., data=...)
            db.create_table(name=table_name, data=rows)
            created = True
        except Exception as e2:
            create_exceptions.append(e2)
            # fallback to explicit schema using pydantic Vector (if available)
            if HAVE_PYDANTIC:
                try:
                    # determine embedding dim from first row
                    dim = len(embeddings[0])
                    class ToolModel(LanceModel):
                        id: str
                        server_url: str
                        tool_name: str
                        description: str
                        parameters: dict
                        text: str
                        embedding: Vector(dim)
                    # create table with schema and mode overwrite to replace if exists
                    db.create_table(table_name, schema=ToolModel, mode="overwrite")
                    tbl = db.open_table(table_name)
                    # many versions expose add/insert or append, try common names
                    if hasattr(tbl, "add"):
                        tbl.add(rows)
                    elif hasattr(tbl, "insert"):
                        tbl.insert(rows)
                    elif hasattr(tbl, "append"):
                        tbl.append(rows)
                    else:
                        # as last resort, try db.create_table with data after schema create
                        try:
                            db.create_table(table_name, rows)
                        except Exception:
                            raise RuntimeError("Could not add rows to newly created typed table.")
                    created = True
                except Exception as e3:
                    create_exceptions.append(e3)

    if not created:
        # If we still didn't create, try opening existing table and inserting rows
        try:
            tbl = db.open_table(table_name)
            # Try common insertion methods
            if hasattr(tbl, "add"):
                tbl.add(rows)
            elif hasattr(tbl, "insert"):
                tbl.insert(rows)
            elif hasattr(tbl, "append"):
                tbl.append(rows)
            else:
                raise RuntimeError("Opened table exists but doesn't support add/insert/append.")
            created = True
            print(f"[index_tools_to_lancedb] appended {len(rows)} rows to existing table '{table_name}'.")
        except Exception as e_open:
            create_exceptions.append(e_open)
            # Give a helpful error message
            msg = "Failed to create or open LanceDB table. Errors:\n" + "\n".join(
                f"- {type(x).__name__}: {x}" for x in create_exceptions
            )
            raise RuntimeError(msg)

    print(f"[index_tools_to_lancedb] table '{table_name}' ready with {len(rows)} rows.")


def fetch_top_k_tools_formatted(
    query: str,
    db_path: str = "./mcp_tools_lancedb",
    table_name: str = "mcp_tools",
    k: int = 5,
    embedding_model_name: str = EMBED_MODEL_NAME
) -> List[Dict[str, Any]]:
    """
    Query LanceDB and return top-k tools in the requested schema.
    Uses the portable search pattern: table.search(vector).limit(k).to_pandas()
    """
    model = SentenceTransformer(embedding_model_name)
    q_emb = model.encode([query], convert_to_numpy=True)[0].astype(np.float32).tolist()

    db = connect(db_path)
    try:
        table = db.open_table(table_name)
    except Exception as e:
        raise RuntimeError(f"Could not open LanceDB table '{table_name}': {e}") from e

    # Portable search: call search(q_emb) then chain limit() and to_pandas() if available
    try:
        search_obj = table.search(q_emb)
        # chain limit if method exists
        if hasattr(search_obj, "limit"):
            search_obj = search_obj.limit(k)
        # prefer to_pandas when available
        if hasattr(search_obj, "to_pandas"):
            df = search_obj.to_pandas()
            hits = []
            for _, row in df.iterrows():
                params = _ensure_json_schema(row.get("parameters", None))
                entry = {"name": row.get("tool_name"), "description": row.get("description") or "", "parameters": params}
                
                entry["server_url"] = row.get("server_url")
                hits.append(entry)
            return hits
        else:
            # try execute() or to_list
            try:
                executed = search_obj.execute()
            except Exception:
                executed = list(search_obj)
            hits = []
            for hit in executed[:k]:
                params = _ensure_json_schema(hit.get("parameters", None))
                entry = {"name": hit.get("tool_name"), "description": hit.get("description") or "", "parameters": params}
                
                entry["server_url"] = hit.get("server_url")
                hits.append(entry)
            return hits
    except Exception as e:
        raise RuntimeError(f"Search failed on table '{table_name}': {e}") from e
