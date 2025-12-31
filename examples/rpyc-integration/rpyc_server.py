#!/usr/bin/env python3
"""
RPyC Server for UnifyWeaver Integration Demo

This server provides a basic RPyC endpoint for testing
the UnifyWeaver RPyC transport.

Usage:
    python rpyc_server.py [--port PORT]

Then run the Prolog demo:
    swipl rpyc_demo.pl
    ?- run_demo.
"""

import rpyc
from rpyc.utils.server import ThreadedServer
from rpyc.core.service import ClassicService
import sys
import os
import argparse


class DemoService(ClassicService):
    """Demo RPyC service for UnifyWeaver integration testing."""

    def on_connect(self, conn):
        print(f"Client connected from {conn._config}")
        self._setup_environment()

    def on_disconnect(self, conn):
        print("Client disconnected")

    def _setup_environment(self):
        """Setup demo environment."""
        # Add current directory to path for dict_wrapper if available
        script_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, script_dir)

        # Try to find and add UnifyWeaver's rpyc module
        # Navigate up to find src/unifyweaver/glue/rpyc
        potential_paths = [
            os.path.join(script_dir, '../../src/unifyweaver/glue/rpyc'),
            os.path.join(script_dir, '../../src'),
        ]

        for path in potential_paths:
            abs_path = os.path.abspath(path)
            if os.path.exists(abs_path):
                sys.path.insert(0, abs_path)
                print(f"Added path: {abs_path}")

        # Import dict_wrapper if available
        try:
            import dict_wrapper
            print("+ dict_wrapper loaded")
        except ImportError:
            print("- dict_wrapper not available (optional)")

    # Example exposed methods for testing
    def exposed_echo(self, message):
        """Echo a message back."""
        return f"Echo: {message}"

    def exposed_add(self, a, b):
        """Add two numbers."""
        return a + b

    def exposed_compute_stats(self, numbers):
        """Compute basic statistics for a list of numbers."""
        if not numbers:
            return {"error": "empty list"}
        return {
            "count": len(numbers),
            "sum": sum(numbers),
            "mean": sum(numbers) / len(numbers),
            "min": min(numbers),
            "max": max(numbers),
        }


def start_server(port=18812):
    """Start the RPyC demo server."""
    print("=" * 50)
    print("UnifyWeaver RPyC Demo Server")
    print("=" * 50)
    print()
    print("WARNING: This server uses UNSECURED mode!")
    print("Only use for development and testing.")
    print()
    print(f"Listening on port {port}")
    print("Press Ctrl+C to stop")
    print()
    print("Connect from Prolog with:")
    print("  ?- rpyc_connect('localhost', [")
    print("         security(unsecured),")
    print("         acknowledge_risk(true)")
    print("     ], Proxy).")
    print()
    print("=" * 50)

    server = ThreadedServer(
        DemoService,
        port=port,
        protocol_config={"allow_all_attrs": True}
    )

    try:
        server.start()
    except KeyboardInterrupt:
        print("\nShutting down server...")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='UnifyWeaver RPyC Demo Server')
    parser.add_argument('--port', type=int, default=18812,
                       help='Server port (default: 18812)')

    args = parser.parse_args()
    start_server(args.port)
