#!/usr/bin/env python3
"""
C# Target Test Helper

This module provides utilities for testing C# code generation from Prolog.
It can be called from Prolog via the Janus bridge or used standalone.

Usage from Prolog:
    :- use_module(library(janus)).
    ?- py_call(csharp_test_helper:compile_and_run(Code, Output)).

Usage standalone:
    python3 csharp_test_helper.py <code_file> <expected_output>
"""

import subprocess
import tempfile
import shutil
import os
import sys
from pathlib import Path
from typing import Tuple, Optional, List
import json


class CSharpTestResult:
    """Result of compiling and running C# code"""

    def __init__(self, success: bool, stdout: str = "", stderr: str = "",
                 build_output: str = "", error: Optional[str] = None):
        self.success = success
        self.stdout = stdout
        self.stderr = stderr
        self.build_output = build_output
        self.error = error

    def to_dict(self):
        """Convert to dictionary for Janus bridge"""
        return {
            'success': self.success,
            'stdout': self.stdout,
            'stderr': self.stderr,
            'build_output': self.build_output,
            'error': self.error
        }


def find_dotnet() -> Optional[str]:
    """Find dotnet executable in PATH"""
    dotnet_path = shutil.which('dotnet')
    if not dotnet_path:
        # Try common WSL paths
        wsl_paths = ['/usr/bin/dotnet', '/usr/local/bin/dotnet']
        for path in wsl_paths:
            if os.path.exists(path):
                return path
    return dotnet_path


def create_csproj(target_dir: Path, target_framework: str = "net9.0") -> Path:
    """Create a minimal .csproj file"""
    csproj_content = f'''<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>{target_framework}</TargetFramework>
    <Nullable>enable</Nullable>
  </PropertyGroup>
</Project>
'''
    csproj_path = target_dir / "test.csproj"
    csproj_path.write_text(csproj_content)
    return csproj_path


def compile_csharp(code: str, test_name: str = "test") -> CSharpTestResult:
    """
    Compile C# code using dotnet build

    Args:
        code: C# source code as string
        test_name: Name for the test (used in temp directory)

    Returns:
        CSharpTestResult with compilation results
    """
    dotnet = find_dotnet()
    if not dotnet:
        return CSharpTestResult(
            success=False,
            error="dotnet CLI not found in PATH"
        )

    # Create temporary directory for build
    with tempfile.TemporaryDirectory(prefix=f"unifyweaver_csharp_{test_name}_") as tmpdir:
        test_dir = Path(tmpdir)

        # Write source file
        program_cs = test_dir / "Program.cs"
        program_cs.write_text(code)

        # Create .csproj
        csproj = create_csproj(test_dir)

        # Build directory
        bin_dir = test_dir / "bin"
        bin_dir.mkdir(exist_ok=True)

        # Build the project
        build_cmd = [
            dotnet, "build",
            str(csproj),
            "--nologo",
            "-v", "quiet",
            "-o", str(bin_dir)
        ]

        try:
            build_result = subprocess.run(
                build_cmd,
                capture_output=True,
                text=True,
                timeout=30
            )

            if build_result.returncode != 0:
                return CSharpTestResult(
                    success=False,
                    stderr=build_result.stderr,
                    build_output=build_result.stdout,
                    error=f"Build failed with exit code {build_result.returncode}"
                )

            # Find executable
            exe_path = bin_dir / "test"
            dll_path = bin_dir / "test.dll"

            if exe_path.exists():
                # Native executable
                run_cmd = [str(exe_path)]
            elif dll_path.exists():
                # .NET DLL - run with dotnet
                run_cmd = [dotnet, str(dll_path)]
            else:
                return CSharpTestResult(
                    success=False,
                    build_output=build_result.stdout,
                    error="No executable found after build"
                )

            # Run the executable
            run_result = subprocess.run(
                run_cmd,
                capture_output=True,
                text=True,
                timeout=10
            )

            return CSharpTestResult(
                success=run_result.returncode == 0,
                stdout=run_result.stdout,
                stderr=run_result.stderr,
                build_output=build_result.stdout,
                error=None if run_result.returncode == 0 else f"Execution failed with exit code {run_result.returncode}"
            )

        except subprocess.TimeoutExpired:
            return CSharpTestResult(
                success=False,
                error="Build or execution timed out"
            )
        except Exception as e:
            return CSharpTestResult(
                success=False,
                error=f"Unexpected error: {str(e)}"
            )


def compile_and_run(code: str, test_name: str = "test") -> dict:
    """
    Compile and run C# code, returning result as dictionary

    This is the main entry point for Janus bridge usage.

    Args:
        code: C# source code
        test_name: Name for the test

    Returns:
        Dictionary with keys: success, stdout, stderr, build_output, error
    """
    result = compile_csharp(code, test_name)
    return result.to_dict()


def assert_output_contains(code: str, expected_substring: str, test_name: str = "test") -> dict:
    """
    Compile, run, and assert that output contains expected substring

    Args:
        code: C# source code
        expected_substring: Substring that should appear in stdout
        test_name: Name for the test

    Returns:
        Dictionary with test result and assertion status
    """
    result = compile_csharp(code, test_name)
    result_dict = result.to_dict()

    if not result.success:
        result_dict['assertion_passed'] = False
        result_dict['assertion_error'] = result.error or "Compilation/execution failed"
        return result_dict

    if expected_substring in result.stdout:
        result_dict['assertion_passed'] = True
        result_dict['assertion_error'] = None
    else:
        result_dict['assertion_passed'] = False
        result_dict['assertion_error'] = f"Expected '{expected_substring}' not found in output"

    return result_dict


def assert_output_lines(code: str, expected_lines: List[str], test_name: str = "test") -> dict:
    """
    Compile, run, and assert that output contains all expected lines

    Args:
        code: C# source code
        expected_lines: List of lines that should appear in stdout
        test_name: Name for the test

    Returns:
        Dictionary with test result and assertion status
    """
    result = compile_csharp(code, test_name)
    result_dict = result.to_dict()

    if not result.success:
        result_dict['assertion_passed'] = False
        result_dict['assertion_error'] = result.error or "Compilation/execution failed"
        return result_dict

    stdout_lines = set(line.strip() for line in result.stdout.strip().split('\n'))
    missing_lines = []

    for expected_line in expected_lines:
        if expected_line.strip() not in stdout_lines:
            missing_lines.append(expected_line)

    if not missing_lines:
        result_dict['assertion_passed'] = True
        result_dict['assertion_error'] = None
    else:
        result_dict['assertion_passed'] = False
        result_dict['assertion_error'] = f"Missing expected lines: {missing_lines}"

    return result_dict


def get_dotnet_version() -> Optional[str]:
    """Get dotnet CLI version"""
    dotnet = find_dotnet()
    if not dotnet:
        return None

    try:
        result = subprocess.run(
            [dotnet, "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except:
        pass

    return None


def main():
    """Command-line interface"""
    if len(sys.argv) < 2:
        print("Usage: csharp_test_helper.py <code_file> [expected_output]")
        print("       csharp_test_helper.py --version")
        sys.exit(1)

    if sys.argv[1] == "--version":
        version = get_dotnet_version()
        if version:
            print(f"dotnet version: {version}")
        else:
            print("dotnet not found")
        sys.exit(0 if version else 1)

    code_file = sys.argv[1]

    if not os.path.exists(code_file):
        print(f"Error: File not found: {code_file}")
        sys.exit(1)

    with open(code_file, 'r') as f:
        code = f.read()

    if len(sys.argv) >= 3:
        expected = sys.argv[2]
        result = assert_output_contains(code, expected, Path(code_file).stem)

        print(json.dumps(result, indent=2))
        sys.exit(0 if result['assertion_passed'] else 1)
    else:
        result = compile_and_run(code, Path(code_file).stem)

        print(json.dumps(result, indent=2))
        sys.exit(0 if result['success'] else 1)


if __name__ == "__main__":
    main()
