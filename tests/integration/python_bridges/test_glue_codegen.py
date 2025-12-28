#!/usr/bin/env python3
"""
Test Prolog glue module code generation.

Tests that python_bridges_glue.pl generates valid code for all bridges.

Usage:
    python -m pytest tests/integration/python_bridges/test_glue_codegen.py -v
"""

import subprocess
import pytest


def run_prolog(query):
    """Run a Prolog query and return output."""
    cmd = [
        "swipl", "-g", query, "-t", "halt"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    return result.stdout, result.stderr, result.returncode


class TestPrologGlueModule:
    """Test python_bridges_glue.pl code generation."""

    @pytest.fixture(autouse=True)
    def check_swipl(self):
        """Skip if SWI-Prolog not available."""
        try:
            result = subprocess.run(
                ["swipl", "--version"],
                capture_output=True,
                timeout=5
            )
            if result.returncode != 0:
                pytest.skip("SWI-Prolog not available")
        except FileNotFoundError:
            pytest.skip("SWI-Prolog not installed")

    def test_module_loads(self):
        """Test glue module loads without errors."""
        stdout, stderr, code = run_prolog(
            "use_module('src/unifyweaver/glue/python_bridges_glue')"
        )
        assert code == 0, f"Module failed to load: {stderr}"

    def test_generate_pythonnet_client(self):
        """Test Python.NET client code generation."""
        query = """
            use_module('src/unifyweaver/glue/python_bridges_glue'),
            generate_pythonnet_rpyc_client([host(localhost), port(18812)], Code),
            format('~w', [Code])
        """
        stdout, stderr, code = run_prolog(query)
        assert code == 0, f"Code generation failed: {stderr}"
        assert "Python.Runtime" in stdout or "pythonnet" in stdout.lower()

    def test_generate_jpype_client(self):
        """Test JPype client code generation."""
        query = """
            use_module('src/unifyweaver/glue/python_bridges_glue'),
            generate_jpype_rpyc_client([host(localhost), port(18812)], Code),
            format('~w', [Code])
        """
        stdout, stderr, code = run_prolog(query)
        assert code == 0, f"Code generation failed: {stderr}"
        assert "JPype" in stdout or "jpype" in stdout.lower()

    def test_generate_jpy_client(self):
        """Test jpy client code generation."""
        query = """
            use_module('src/unifyweaver/glue/python_bridges_glue'),
            generate_jpy_rpyc_client([host(localhost), port(18812)], Code),
            format('~w', [Code])
        """
        stdout, stderr, code = run_prolog(query)
        assert code == 0, f"Code generation failed: {stderr}"
        assert "jpy" in stdout.lower()

    def test_generate_csnakes_client(self):
        """Test CSnakes client code generation."""
        query = """
            use_module('src/unifyweaver/glue/python_bridges_glue'),
            generate_csnakes_rpyc_client([host(localhost), port(18812)], Code),
            format('~w', [Code])
        """
        stdout, stderr, code = run_prolog(query)
        assert code == 0, f"Code generation failed: {stderr}"
        assert "CSnakes" in stdout or "csnakes" in stdout.lower()

    def test_generic_interface(self):
        """Test generic generate_python_bridge_client/3."""
        for bridge in ["pythonnet", "jpype", "jpy", "csnakes"]:
            query = f"""
                use_module('src/unifyweaver/glue/python_bridges_glue'),
                generate_python_bridge_client({bridge}, [port(18812)], Code),
                format('~w', [Code])
            """
            stdout, stderr, code = run_prolog(query)
            assert code == 0, f"Generic interface failed for {bridge}: {stderr}"
            assert len(stdout) > 100, f"Generated code too short for {bridge}"

    def test_custom_options(self):
        """Test code generation with custom options."""
        query = """
            use_module('src/unifyweaver/glue/python_bridges_glue'),
            generate_jpype_rpyc_client([
                host('myserver.example.com'),
                port(19000),
                package('com.mycompany.rpyc'),
                class_name('CustomRPyCClient')
            ], Code),
            format('~w', [Code])
        """
        stdout, stderr, code = run_prolog(query)
        assert code == 0, f"Custom options failed: {stderr}"
        # Check custom values appear in generated code
        assert "myserver.example.com" in stdout or "19000" in stdout


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
