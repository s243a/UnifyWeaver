#!/usr/bin/env python3
"""
Integration tests for Python bridge examples.

Tests JPype, Python.NET (pythonnet), and jpy bridges with RPyC.

Usage:
    1. Start RPyC server: python examples/rpyc-integration/rpyc_server.py
    2. Run tests: python -m pytest tests/integration/python_bridges/test_bridges.py -v
"""

import subprocess
import sys
import time
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
def rpyc_server():
    """Start RPyC server for tests if not running."""
    if is_rpyc_server_running():
        yield  # Server already running
        return

    # Start server
    proc = subprocess.Popen(
        [sys.executable, "examples/rpyc-integration/rpyc_server.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    time.sleep(2)  # Wait for startup

    yield proc

    # Cleanup
    proc.terminate()
    proc.wait()


class TestJPypeBridge:
    """Test JPype bridge with RPyC."""

    @pytest.fixture(autouse=True)
    def check_jpype(self):
        """Skip if JPype not installed."""
        try:
            import jpype
            self.jpype = jpype
        except ImportError:
            pytest.skip("jpype1 not installed")

    def test_jpype_jvm_starts(self, rpyc_server):
        """Test JVM starts successfully."""
        if not self.jpype.isJVMStarted():
            self.jpype.startJVM()
        assert self.jpype.isJVMStarted()

    def test_jpype_java_version(self, rpyc_server):
        """Test can get Java version."""
        if not self.jpype.isJVMStarted():
            self.jpype.startJVM()

        from java.lang import System
        version = System.getProperty("java.version")
        assert version is not None
        assert len(version) > 0

    def test_jpype_rpyc_math(self, rpyc_server):
        """Test RPyC math.sqrt via JPype."""
        if not self.jpype.isJVMStarted():
            self.jpype.startJVM()

        import rpyc
        conn = rpyc.classic.connect("localhost", 18812)
        try:
            math = conn.modules.math
            result = math.sqrt(16)
            assert result == 4.0
        finally:
            conn.close()

    def test_jpype_rpyc_numpy(self, rpyc_server):
        """Test RPyC numpy.mean via JPype."""
        if not self.jpype.isJVMStarted():
            self.jpype.startJVM()

        import rpyc
        conn = rpyc.classic.connect("localhost", 18812)
        try:
            np = conn.modules.numpy
            arr = np.array([1, 2, 3, 4, 5])
            mean = float(np.mean(arr))
            assert mean == 3.0
        finally:
            conn.close()


class TestPythonNetBridge:
    """Test Python.NET bridge with RPyC."""

    @pytest.fixture(autouse=True)
    def check_pythonnet(self):
        """Skip if pythonnet not installed or .NET not available."""
        import os
        os.environ.setdefault("PYTHONNET_RUNTIME", "coreclr")
        try:
            import clr
            self.clr = clr
        except (ImportError, RuntimeError) as e:
            pytest.skip(f"pythonnet not available: {e}")

    def test_pythonnet_dotnet_version(self, rpyc_server):
        """Test can get .NET version."""
        from System import Environment
        version = str(Environment.Version)
        assert version is not None
        assert len(version) > 0

    def test_pythonnet_rpyc_math(self, rpyc_server):
        """Test RPyC math.sqrt via Python.NET."""
        import rpyc
        conn = rpyc.classic.connect("localhost", 18812)
        try:
            math = conn.modules.math
            result = math.sqrt(16)
            assert result == 4.0
        finally:
            conn.close()

    def test_pythonnet_rpyc_numpy(self, rpyc_server):
        """Test RPyC numpy.mean via Python.NET."""
        import rpyc
        conn = rpyc.classic.connect("localhost", 18812)
        try:
            np = conn.modules.numpy
            arr = np.array([1, 2, 3, 4, 5])
            mean = float(np.mean(arr))
            assert mean == 3.0
        finally:
            conn.close()


class TestJpyBridge:
    """Test jpy bridge with RPyC."""

    @pytest.fixture(autouse=True)
    def check_jpy(self):
        """Skip if jpy not installed."""
        try:
            import jpy
            self.jpy = jpy
        except ImportError:
            pytest.skip("jpy not installed")

    def test_jpy_jvm_starts(self, rpyc_server):
        """Test jpy JVM starts."""
        self.jpy.create_jvm(["-Xmx512m"])
        System = self.jpy.get_type("java.lang.System")
        version = System.getProperty("java.version")
        assert version is not None

    def test_jpy_rpyc_math(self, rpyc_server):
        """Test RPyC math.sqrt via jpy."""
        import rpyc
        conn = rpyc.classic.connect("localhost", 18812)
        try:
            math = conn.modules.math
            result = math.sqrt(16)
            assert result == 4.0
        finally:
            conn.close()

    def test_jpy_java_collections(self, rpyc_server):
        """Test jpy bi-directional with Java ArrayList."""
        ArrayList = self.jpy.get_type("java.util.ArrayList")
        java_list = ArrayList()
        java_list.add("hello")
        java_list.add("from")
        java_list.add("java")

        # Convert using size/get pattern
        py_list = [java_list.get(i) for i in range(java_list.size())]
        assert py_list == ["hello", "from", "java"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
