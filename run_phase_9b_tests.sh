#!/bin/bash
# Test runner for Phase 9b: GROUP BY Aggregations

echo "╔════════════════════════════════════════════════════╗"
echo "║   Phase 9b: GROUP BY Aggregations Test Runner    ║"
echo "╚════════════════════════════════════════════════════╝"
echo ""

# Run the test suite
swipl test_phase_9b.pl

# Capture exit code
EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ All Phase 9b GROUP BY tests passed!"
else
    echo "✗ Phase 9b tests failed with exit code: $EXIT_CODE"
fi

exit $EXIT_CODE
