#!/usr/bin/env python3
"""
JPype + RPyC Example Client

Demonstrates using JPyC to embed CPython in JVM and connect to RPyC.
This Python script shows the pattern that would be called from Java.

Usage:
    1. Start RPyC server: python examples/rpyc-integration/rpyc_server.py
    2. Run this script: python examples/python-bridges/jpype/rpyc_client.py
"""

import jpype
import jpype.imports


def main():
    # Start JVM
    if not jpype.isJVMStarted():
        jpype.startJVM()

    print("JVM started successfully")

    # Get Java version
    from java.lang import System
    java_version = System.getProperty("java.version")
    print(f"Java version: {java_version}")

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

    # Keep JVM running for potential Java code
    # jpype.shutdownJVM()


if __name__ == "__main__":
    main()
