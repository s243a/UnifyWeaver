#!/bin/bash
# Test runner for Phase 9c-2: HAVING Clause Support

echo "╔════════════════════════════════════════════════════╗"
echo "║   Phase 9c-2: HAVING Clause Test Runner          ║"
echo "╚════════════════════════════════════════════════════╝"
echo ""

# Run the test suite
swipl test_phase_9c2.pl

# Capture exit code
EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ All Phase 9c-2 HAVING clause tests passed!"
else
    echo "✗ Phase 9c-2 tests failed with exit code: $EXIT_CODE"
fi

exit $EXIT_CODE
