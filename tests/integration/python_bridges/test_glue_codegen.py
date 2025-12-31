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


class TestAutoDetection:
    """Test auto-detection and bridge selection."""

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

    def test_detect_all_bridges(self):
        """Test detect_all_bridges returns a list."""
        query = """
            use_module('src/unifyweaver/glue/python_bridges_glue'),
            detect_all_bridges(Bridges),
            format('~w', [Bridges])
        """
        stdout, stderr, code = run_prolog(query)
        assert code == 0, f"detect_all_bridges failed: {stderr}"
        # Should return a list (may be empty or have bridges)
        assert stdout.startswith("["), f"Expected list, got: {stdout}"

    def test_auto_select_bridge_any(self):
        """Test auto_select_bridge with 'any' target."""
        query = """
            use_module('src/unifyweaver/glue/python_bridges_glue'),
            auto_select_bridge(any, Bridge),
            format('~w', [Bridge])
        """
        stdout, stderr, code = run_prolog(query)
        assert code == 0, f"auto_select_bridge(any) failed: {stderr}"
        # Should return a bridge name or 'none'
        assert stdout in ["pythonnet", "csnakes", "jpype", "jpy", "none"]

    def test_auto_select_bridge_with_preferences(self):
        """Test auto_select_bridge respects preferences."""
        query = """
            use_module('src/unifyweaver/glue/python_bridges_glue'),
            detect_all_bridges(Available),
            (   member(jpy, Available)
            ->  auto_select_bridge(jvm, [prefer(jpy)], Bridge),
                format('~w', [Bridge])
            ;   format('skip', [])
            )
        """
        stdout, stderr, code = run_prolog(query)
        assert code == 0, f"auto_select_bridge with preferences failed: {stderr}"
        # If jpy is available, should return jpy; otherwise skip
        assert stdout in ["jpy", "skip"]

    def test_bridge_requirements(self):
        """Test bridge_requirements returns requirements list."""
        for bridge in ["pythonnet", "csnakes", "jpype", "jpy"]:
            query = f"""
                use_module('src/unifyweaver/glue/python_bridges_glue'),
                bridge_requirements({bridge}, Reqs),
                length(Reqs, Len),
                format('~d', [Len])
            """
            stdout, stderr, code = run_prolog(query)
            assert code == 0, f"bridge_requirements({bridge}) failed: {stderr}"
            req_count = int(stdout)
            assert req_count >= 2, f"Expected at least 2 requirements for {bridge}"

    def test_check_bridge_ready(self):
        """Test check_bridge_ready returns valid status."""
        for bridge in ["pythonnet", "csnakes", "jpype", "jpy"]:
            query = f"""
                use_module('src/unifyweaver/glue/python_bridges_glue'),
                check_bridge_ready({bridge}, Status),
                format('~w', [Status])
            """
            stdout, stderr, code = run_prolog(query)
            assert code == 0, f"check_bridge_ready({bridge}) failed: {stderr}"
            # Status should be 'ready' or a missing_* term
            assert "ready" in stdout or "missing" in stdout

    def test_validate_bridge_config_valid(self):
        """Test validate_bridge_config with valid options."""
        query = """
            use_module('src/unifyweaver/glue/python_bridges_glue'),
            (   validate_bridge_config(jpype, [host(localhost), port(18812)])
            ->  format('valid', [])
            ;   format('invalid', [])
            )
        """
        stdout, stderr, code = run_prolog(query)
        assert code == 0, f"validate_bridge_config failed: {stderr}"
        assert stdout == "valid"

    def test_validate_bridge_config_invalid_port(self):
        """Test validate_bridge_config rejects invalid port."""
        query = """
            use_module('src/unifyweaver/glue/python_bridges_glue'),
            (   validate_bridge_config(jpype, [port(99999)])
            ->  format('valid', [])
            ;   format('invalid', [])
            )
        """
        stdout, stderr, code = run_prolog(query)
        assert code == 0, f"validate_bridge_config failed: {stderr}"
        assert stdout == "invalid"

    def test_generate_auto_client_jvm(self):
        """Test generate_auto_client for JVM target."""
        query = """
            use_module('src/unifyweaver/glue/python_bridges_glue'),
            detect_all_bridges(Available),
            (   (member(jpype, Available) ; member(jpy, Available))
            ->  generate_auto_client(jvm, [port(18812)], Code),
                atom_length(Code, Len),
                format('~d', [Len])
            ;   format('skip', [])
            )
        """
        stdout, stderr, code = run_prolog(query)
        assert code == 0, f"generate_auto_client(jvm) failed: {stderr}"
        if stdout != "skip":
            code_len = int(stdout)
            assert code_len > 500, f"Generated code too short: {code_len} chars"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
