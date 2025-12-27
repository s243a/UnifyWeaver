#!/usr/bin/env python3
"""
Python.NET + RPyC Example Client

Demonstrates using Python.NET to embed CPython in .NET and connect to RPyC.

Prerequisites:
    - .NET Core SDK 6.0+ (or .NET 8 recommended)
    - pip install pythonnet rpyc

Usage:
    1. Start RPyC server: python examples/rpyc-integration/rpyc_server.py
    2. Run this script: python examples/python-bridges/pythonnet/rpyc_client.py

Runtime Configuration:
    Python.NET defaults to .NET Core on modern systems.
    Set PYTHONNET_RUNTIME=coreclr explicitly if needed.
"""

import os
import sys


def configure_runtime():
    """Configure Python.NET to use .NET Core (not Mono)."""
    # Explicitly prefer .NET Core over Mono
    os.environ.setdefault("PYTHONNET_RUNTIME", "coreclr")

    # Optional: specify .NET version
    # os.environ["PYTHONNET_DOTNET_VERSION"] = "8.0.0"


def main():
    # Configure runtime before importing clr
    configure_runtime()

    try:
        import clr
        print("Python.NET loaded successfully")
        print(f"Runtime: {os.environ.get('PYTHONNET_RUNTIME', 'default')}")
    except RuntimeError as e:
        print(f"Error loading Python.NET: {e}")
        print("\nEnsure .NET SDK is installed:")
        print("  Ubuntu/Debian: sudo apt install dotnet-sdk-8.0")
        print("  Or download from: https://dotnet.microsoft.com/download")
        sys.exit(1)

    # Import .NET assemblies
    clr.AddReference("System")
    from System import Console, Environment

    print(f".NET Version: {Environment.Version}")
    print(f"OS: {Environment.OSVersion}")

    # Connect to RPyC server
    import rpyc

    print("\nConnecting to RPyC server on localhost:18812...")
    conn = rpyc.classic.connect("localhost", 18812)

    try:
        # Test 1: Basic math
        math = conn.modules.math
        sqrt_result = math.sqrt(16)
        print(f"math.sqrt(16) = {sqrt_result}")

        # Test 2: NumPy (if available)
        try:
            np = conn.modules.numpy
            arr = np.array([1, 2, 3, 4, 5])
            mean = np.mean(arr)
            print(f"numpy.mean([1,2,3,4,5]) = {mean}")
        except Exception as e:
            print(f"NumPy not available: {e}")

        # Test 3: Get server info (if service provides it)
        try:
            info = conn.root.get_info()
            print(f"Server Python: {info['python_version']}")
        except AttributeError:
            # Classic server doesn't have get_info
            sys_mod = conn.modules.sys
            print(f"Server Python: {sys_mod.version.split()[0]}")

        print("\nAll tests passed!")

    finally:
        conn.close()
        print("Connection closed")


if __name__ == "__main__":
    main()
