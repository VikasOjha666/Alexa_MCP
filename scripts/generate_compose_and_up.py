#!/usr/bin/env python3
"""
Scan own_MCP_servers/* and create a docker-compose.generated.yml
Then optionally run `docker compose -f docker-compose.generated.yml up -d`.
"""
import os
import yaml
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # repo root
MCP_DIR = ROOT / "own_MCP_servers"
OUT = ROOT / "docker-compose.generated.yml"

BASE_PORT = 8000  # start assigning container host ports from here
fastapi_service_name = "fastapi_app"

def discover_mcp_dirs():
    out = []
    if not MCP_DIR.exists():
        return out
    for child in sorted(MCP_DIR.iterdir()):
        if child.is_dir():
            out.append(child.name)
    return out

def build_compose(mcp_dirs):
    services = {}

    # fastapi service (assumes there's a Dockerfile at repo root)
    services[fastapi_service_name] = {
        "build": {"context": str(ROOT), "dockerfile": "Dockerfile"},
        "container_name": "fastapi_mcp_app",
        "restart": "unless-stopped",
        "ports": ["3000:3000"],
        "environment": [
            # MCP_SERVER_URLS will be set below after we discover services
            "MODEL_PATH=/models/gorilla-openfunctions-v2-q4_K_M.gguf",
        ],
        "volumes": [
            "./gorilla-openfunctions-v2-GGUF:/models:ro",
            "./mcp_tools_lancedb:/home/appuser/app/mcp_tools_lancedb",
            ".:/home/appuser/app:ro",
        ],
        "depends_on": []
    }

    mcp_urls = []
    current_port = BASE_PORT
    for name in mcp_dirs:
        svc_name = f"mcp_{name}"
        container_port = 8000  # assume server listens on 8000 inside container
        # map host port incrementally
        host_port = current_port
        current_port += 1

        services[svc_name] = {
            "build": {"context": f"./own_MCP_servers/{name}", "dockerfile": "Dockerfile"},
            "container_name": svc_name,
            "restart": "unless-stopped",
            "ports": [f"{host_port}:{container_port}"],
            "environment": [f"HOST=0.0.0.0", f"PORT={container_port}"],
            "volumes": [f"./own_MCP_servers/{name}/app:/app:ro"]
        }
        # Use host ports for external testing but use service name for inter-container comms:
        # inside fastapi container we can address by service name and container port:
        mcp_urls.append(f"http://{svc_name}:{container_port}/sse")
        services[fastapi_service_name]["depends_on"].append(svc_name)

    # set MCP_SERVER_URLS env var for fastapi
    services[fastapi_service_name]["environment"].append("MCP_SERVER_URLS=" + ",".join(mcp_urls))

    compose = {"version": "3.8", "services": services}
    return compose

def main(apply_up=True):
    mcp_dirs = discover_mcp_dirs()
    print("Discovered MCP dirs:", mcp_dirs)
    compose = build_compose(mcp_dirs)
    with open(OUT, "w") as f:
        yaml.dump(compose, f)
    print("Wrote:", OUT)
    if apply_up:
        print("Running: docker compose -f docker-compose.generated.yml up -d --build")
        subprocess.check_call(["docker", "compose", "-f", str(OUT), "up", "-d", "--build"])
        print("Started services.")

if __name__ == "__main__":
    main()
