#!/bin/bash
# Test runner for Phase 9c-3: Nested Grouping

echo "╔════════════════════════════════════════════════════╗"
echo "║   Phase 9c-3: Nested Grouping Test Runner        ║"
echo "╚════════════════════════════════════════════════════╝"
echo ""

# Run the test suite
swipl test_phase_9c3.pl

# Capture exit code
EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ All Phase 9c-3 nested grouping tests passed!"
else
    echo "✗ Phase 9c-3 tests failed with exit code: $EXIT_CODE"
fi

exit $EXIT_CODE
