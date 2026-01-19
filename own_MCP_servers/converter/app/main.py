# Converter MCP server.py
import os
import logging
from typing import Any

from mcp.server.fastmcp import FastMCP

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp-server")

# Read host/port from environment (with defaults)
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

mcp = FastMCP(host=HOST, port=PORT)

@mcp.tool()
def celsius_to_fahrenheit(celsius: float):
    """Convert temperature from Celsius to Fahrenheit."""
    return (celsius * 9 / 5) + 32

@mcp.tool()
def fahrenheit_to_celsius(fahrenheit: float):
    """Convert temperature from Fahrenheit to Celsius."""
    return (fahrenheit - 32) * 5 / 9

if __name__ == "__main__":
    logger.info("Starting MCP server on %s:%d", HOST, PORT)
    # FastMCP's .run() will start the server. transport "sse" is common for remote clients.
    mcp.run("sse")


