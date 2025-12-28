# Python bridges integration tests
#
# Test files:
#   - test_bridges.py       - JPype, Python.NET, jpy bridge tests
#   - test_glue_codegen.py  - Prolog glue module code generation tests
#   - test_jpy_isolated.py  - jpy tests that must run in isolation
#
# Running tests:
#   # All tests (jpy JVM tests will skip if JPype runs first)
#   python -m pytest tests/integration/python_bridges/ -v
#
#   # jpy tests in isolation (full jpy coverage)
#   python -m pytest tests/integration/python_bridges/test_jpy_isolated.py -v
#
# Note: JPype and jpy cannot both create JVMs in the same process.
# Run test_jpy_isolated.py separately for full jpy test coverage.
