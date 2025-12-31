// WASM Integration Test for Node.js
// Tests calling LLVM-compiled Prolog predicates via WebAssembly

const fs = require('fs');
const path = require('path');

async function loadPrologMath(wasmPath) {
    const bytes = fs.readFileSync(wasmPath);
    const { instance } = await WebAssembly.instantiate(bytes);
    return instance.exports;
}

async function runTests() {
    console.log('===========================================');
    console.log('WASM Integration Test - Node.js');
    console.log('===========================================\n');

    const wasmPath = path.join(__dirname, 'prolog_wasm.wasm');
    const math = await loadPrologMath(wasmPath);

    const tests = [
        { name: 'sum(10)', fn: () => math.sum(10), expected: 55 },
        { name: 'sum(100)', fn: () => math.sum(100), expected: 5050 },
        { name: 'factorial(5)', fn: () => math.factorial(5), expected: 120 },
        { name: 'factorial(10)', fn: () => math.factorial(10), expected: 3628800 },
    ];

    let passed = 0;
    let failed = 0;

    for (const test of tests) {
        const result = test.fn();
        if (result === test.expected) {
            console.log(`[PASS] ${test.name} = ${result}`);
            passed++;
        } else {
            console.log(`[FAIL] ${test.name} = ${result} (expected ${test.expected})`);
            failed++;
        }
    }

    console.log(`\nResults: ${passed} passed, ${failed} failed`);
    console.log('===========================================');

    if (failed > 0) {
        console.log('INTEGRATION TEST FAILED');
        process.exit(1);
    } else {
        console.log('INTEGRATION TEST PASSED');
    }
}

runTests().catch(console.error);
