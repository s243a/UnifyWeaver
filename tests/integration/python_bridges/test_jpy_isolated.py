#!/usr/bin/env python3
"""
Isolated jpy tests - run in separate process to avoid JVM conflicts.

jpy and JPype cannot both create JVMs in the same process. This file
contains jpy-specific tests that require jpy's own JVM.

Usage:
    # Run in isolation (no JPype interference)
    python -m pytest tests/integration/python_bridges/test_jpy_isolated.py -v

    # Or run directly
    python tests/integration/python_bridges/test_jpy_isolated.py
"""

import subprocess
import sys
import socket
import pytest


def is_rpyc_server_running(host="localhost", port=18812):
    """Check if RPyC server is running."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception:
        return False


@pytest.fixture(scope="module")
def jpy_module():
    """Import and initialize jpy, skip if not available."""
    try:
        import jpy
        return jpy
    except ImportError:
        pytest.skip("jpy not installed")


@pytest.fixture(scope="module")
def jpy_jvm(jpy_module):
    """Start jpy JVM once for all tests in this module."""
    if not jpy_module.has_jvm():
        try:
            jpy_module.create_jvm(["-Xmx512m"])
        except RuntimeError as e:
            pytest.skip(f"Cannot start jpy JVM: {e}")
    return jpy_module


@pytest.fixture(scope="module")
def rpyc_server():
    """Ensure RPyC server is running."""
    if not is_rpyc_server_running():
        pytest.skip("RPyC server not running on localhost:18812")
    yield


class TestJpyIsolated:
    """jpy tests that require jpy's own JVM (no JPype)."""

    def test_jpy_jvm_starts(self, jpy_jvm):
        """Test jpy JVM starts successfully."""
        assert jpy_jvm.has_jvm()
        System = jpy_jvm.get_type("java.lang.System")
        version = System.getProperty("java.version")
        assert version is not None
        print(f"Java version: {version}")

    def test_jpy_java_version(self, jpy_jvm):
        """Test can get Java version and properties."""
        System = jpy_jvm.get_type("java.lang.System")

        version = System.getProperty("java.version")
        vendor = System.getProperty("java.vendor")

        assert version is not None
        assert len(version) > 0
        print(f"Java {version} by {vendor}")

    def test_jpy_java_collections(self, jpy_jvm):
        """Test jpy bi-directional with Java ArrayList."""
        ArrayList = jpy_jvm.get_type("java.util.ArrayList")
        java_list = ArrayList()
        java_list.add("hello")
        java_list.add("from")
        java_list.add("java")

        # Convert using size/get pattern (jpy lists aren't directly iterable)
        py_list = [java_list.get(i) for i in range(java_list.size())]
        assert py_list == ["hello", "from", "java"]

    def test_jpy_hashmap(self, jpy_jvm):
        """Test jpy with Java HashMap."""
        HashMap = jpy_jvm.get_type("java.util.HashMap")
        java_map = HashMap()
        java_map.put("key1", "value1")
        java_map.put("key2", "value2")

        assert java_map.get("key1") == "value1"
        assert java_map.get("key2") == "value2"
        assert java_map.size() == 2

    def test_jpy_rpyc_math(self, jpy_jvm, rpyc_server):
        """Test RPyC math.sqrt via jpy."""
        import rpyc
        conn = rpyc.classic.connect("localhost", 18812)
        try:
            math = conn.modules.math
            result = math.sqrt(16)
            assert result == 4.0
        finally:
            conn.close()

    def test_jpy_rpyc_numpy(self, jpy_jvm, rpyc_server):
        """Test RPyC numpy.mean via jpy."""
        import rpyc
        conn = rpyc.classic.connect("localhost", 18812)
        try:
            np = conn.modules.numpy
            arr = np.array([1, 2, 3, 4, 5])
            mean = float(np.mean(arr))
            assert mean == 3.0
        finally:
            conn.close()

    def test_jpy_java_string_operations(self, jpy_jvm):
        """Test Java String operations via jpy."""
        String = jpy_jvm.get_type("java.lang.String")

        # Create Java string
        java_str = String("Hello, World!")

        assert java_str.length() == 13
        assert java_str.toLowerCase() == "hello, world!"
        assert java_str.toUpperCase() == "HELLO, WORLD!"
        assert java_str.substring(0, 5) == "Hello"

    def test_jpy_java_math(self, jpy_jvm):
        """Test Java Math operations via jpy."""
        Math = jpy_jvm.get_type("java.lang.Math")

        assert Math.sqrt(16) == 4.0
        assert Math.abs(-42) == 42
        assert Math.max(10, 20) == 20
        assert Math.min(10, 20) == 10


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])
