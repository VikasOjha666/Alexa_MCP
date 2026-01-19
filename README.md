# MCP FROM SCRATCH

## How to test
1. Install Python on your system and clone the code.
2. From the base path of the code, run `python scripts/generate_compose_and_up.py`. This will generate Docker Compose YAML while iterating over MCP Servers in the own_MCP_servers path.
3. Now you can build and run the docker containers together with `docker-compose -f docker-compose.generated.yml up --build`.
4. Running command in 3 will run all the docker container and by default the MCP client will be launched at http://localhost:3000.

