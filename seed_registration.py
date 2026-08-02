#!/usr/bin/env python
"""Seed Registration Script for MCP Orchestrator.

Connects to all configured downstream MCP servers (e.g. Filesystem MCP server),
queries their available tools, and idempotently upserts them into PostgreSQL.
"""

import asyncio
import logging
from orchestrator.registrar import run_registration

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def main():
    print("=== Starting MCP Orchestrator Downstream Server Registration ===")
    stats = asyncio.run(run_registration())
    print("=== Registration Completed Successfully ===")
    print(f"Servers Registered / Processed: {stats['servers_processed']}")
    print(f"Tools Registered / Processed: {stats['tools_registered']}")


if __name__ == "__main__":
    main()
