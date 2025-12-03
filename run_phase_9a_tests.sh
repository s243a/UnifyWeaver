#!/bin/bash
# Test runner for Phase 9a: Simple Aggregations

echo "╔════════════════════════════════════════════════════╗"
echo "║   Phase 9a: Simple Aggregations Test Runner      ║"
echo "╚════════════════════════════════════════════════════╝"
echo ""

# Run the test suite
swipl test_phase_9a.pl

# Capture exit code
EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ All Phase 9a tests passed!"
else
    echo "✗ Phase 9a tests failed with exit code: $EXIT_CODE"
fi

exit $EXIT_CODE
