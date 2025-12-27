#!/usr/bin/env python3
"""
jpy + RPyC Example Client

Demonstrates using jpy for bi-directional Java↔Python and RPyC access.

Prerequisites:
    - Java JDK 11+
    - Maven (for jpy build)
    - pip install jpy rpyc  (requires JAVA_HOME set)

Usage:
    1. Start RPyC server: python examples/rpyc-integration/rpyc_server.py
    2. Run this script: python examples/python-bridges/jpy/rpyc_client.py

Note: jpy provides bi-directional calling - Python can call Java AND Java can call Python.
"""

import os
import sys


def check_jpy():
    """Check if jpy is available."""
    try:
        import jpy
        return True
    except ImportError:
        return False


def main():
    if not check_jpy():
        print("jpy not installed.")
        print("\nInstallation requires:")
        print("  1. JAVA_HOME set to JDK directory")
        print("  2. Maven installed (for building Java components)")
        print("  3. pip install jpy")
        print("\nExample:")
        print("  export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64")
        print("  sudo apt install maven")
        print("  pip install jpy")
        sys.exit(1)

    import jpy

    # Initialize JVM
    jpy.create_jvm(["-Xmx512m"])

    print("jpy JVM started successfully")

    # Get Java System class
    System = jpy.get_type("java.lang.System")
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

        # Demonstrate bi-directional: Create Java object and pass to Python
        ArrayList = jpy.get_type("java.util.ArrayList")
        java_list = ArrayList()
        java_list.add("hello")
        java_list.add("from")
        java_list.add("java")
        print(f"Java ArrayList: {list(java_list)}")

        print("\nAll tests passed!")

    finally:
        conn.close()
        print("Connection closed")

    # JVM shutdown handled by jpy


if __name__ == "__main__":
    main()
