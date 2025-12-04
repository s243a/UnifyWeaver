#!/bin/bash
# Test runner for Phase 9c-1: Multiple Aggregations

echo "╔════════════════════════════════════════════════════╗"
echo "║   Phase 9c-1: Multiple Aggregations Test Runner  ║"
echo "╚════════════════════════════════════════════════════╝"
echo ""

# Run the test suite
swipl test_phase_9c1.pl

# Capture exit code
EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ All Phase 9c-1 multiple aggregation tests passed!"
else
    echo "✗ Phase 9c-1 tests failed with exit code: $EXIT_CODE"
fi

exit $EXIT_CODE
