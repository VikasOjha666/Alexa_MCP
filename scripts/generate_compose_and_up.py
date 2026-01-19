#!/usr/bin/env python3
"""
Generate docker-compose.generated.yml by inspecting MCP Dockerfiles.
- MCP internal port inferred from EXPOSE / ENV PORT
- Host ports auto-assigned
- NVIDIA enabled only for fastapi_app (Windows Docker Desktop compatible)
"""

import re
import yaml
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MCP_DIR = ROOT / "own_MCP_servers"
OUT = ROOT / "docker-compose.generated.yml"

FASTAPI_SERVICE = "fastapi_app"
BASE_HOST_PORT = 8000


def parse_dockerfile_ports(dockerfile_path: Path):
    """
    Parse EXPOSE and ENV PORT from a Dockerfile.
    Defaults to 8000 if not found.
    """
    expose_port = None
    env_port = None

    content = dockerfile_path.read_text()

    expose_match = re.search(r"EXPOSE\s+(\d+)", content)
    if expose_match:
        expose_port = int(expose_match.group(1))

    env_match = re.search(r"ENV\s+PORT\s*=\s*(\d+)", content)
    if env_match:
        env_port = int(env_match.group(1))

    return env_port or expose_port or 8000


def discover_mcp_services():
    """
    Discover MCP services and infer their internal ports.
    """
    services = []
    for d in sorted(MCP_DIR.iterdir()):
        dockerfile = d / "Dockerfile"
        if d.is_dir() and dockerfile.exists():
            port = parse_dockerfile_ports(dockerfile)
            services.append((d.name, port))
    return services


def build_compose():
    services = {}

    # -------- FastAPI (GPU-enabled) --------
    services[FASTAPI_SERVICE] = {
        "build": {
            "context": str(ROOT),
            "dockerfile": "Dockerfile",
        },
        "container_name": "fastapi_mcp_app",
        "runtime": "nvidia",
        "environment": [
            "MODEL_PATH=/models/gorilla-openfunctions-v2-q4_K_M.gguf",
            "NVIDIA_VISIBLE_DEVICES=all",
        ],
        "ports": ["3000:3000"],
        "restart": "unless-stopped",
        "volumes": [
            "./gorilla-openfunctions-v2-GGUF:/models:ro",
            "./mcp_tools_lancedb:/home/appuser/app/mcp_tools_lancedb",
            ".:/home/appuser/app:ro",
        ],
        "depends_on": [],
    }

    # -------- MCP services --------
    mcp_urls = []
    host_port = BASE_HOST_PORT

    for name, internal_port in discover_mcp_services():
        svc_name = f"mcp_{name}"

        services[svc_name] = {
            "build": {
                "context": f"./own_MCP_servers/{name}",
                "dockerfile": "Dockerfile",
            },
            "container_name": svc_name,
            "environment": [
                "HOST=0.0.0.0",
                f"PORT={internal_port}",
            ],
            "ports": [f"{host_port}:{internal_port}"],
            "restart": "unless-stopped",
            "volumes": [
                f"./own_MCP_servers/{name}/app:/app:ro",
            ],
        }

        mcp_urls.append(f"http://{svc_name}:{internal_port}/sse")
        services[FASTAPI_SERVICE]["depends_on"].append(svc_name)

        host_port += 1

    # Inject MCP_SERVER_URLS
    services[FASTAPI_SERVICE]["environment"].append(
        "MCP_SERVER_URLS=" + ",".join(mcp_urls)
    )

    return {
        "version": "3.8",
        "services": services,
    }


def main(run_up=True):
    compose = build_compose()
    with open(OUT, "w") as f:
        yaml.safe_dump(compose, f, sort_keys=False)

    print(f"Generated {OUT}")

    if run_up:
        subprocess.check_call(
            ["docker", "compose", "-f", str(OUT), "up", "-d", "--build"]
        )


if __name__ == "__main__":
    main()
