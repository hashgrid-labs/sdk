"""``hashgrid``: ``hashgrid mcp`` runs the local MCP server (as ``hashgrid-mcp``)."""

from __future__ import annotations

import argparse

from . import __version__


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="hashgrid", description="Hashgrid DNA from text, on your machine."
    )
    parser.add_argument("--version", action="version", version=f"hashgrid {__version__}")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("mcp", help="run the local MCP server over stdio")
    args = parser.parse_args(argv)
    if args.command == "mcp":
        from .mcp_server import main as mcp_main

        mcp_main()
    else:
        parser.print_help()
