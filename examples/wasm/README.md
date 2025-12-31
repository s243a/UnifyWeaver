# WASM Examples

Demonstrates calling LLVM-compiled Prolog predicates from Node.js and the browser via WebAssembly.

## Prerequisites

```bash
# LLVM with WASM backend
sudo apt install llvm clang

# WASM linker
sudo apt install lld
```

## Building the WASM Module

```bash
# From project root
cd examples/wasm
./build.sh
```

Or manually:

```bash
# Generate WASM-targeted LLVM IR
swipl -g "use_module('src/unifyweaver/targets/llvm_target'), \
  compile_wasm_module([func(sum, 2, tail_recursion), func(factorial, 1, factorial)], \
  [module_name(prolog_wasm)], Code), write(Code)" -t halt > prolog_wasm.ll

# Fix escaping
sed -i 's/%%/%/g' prolog_wasm.ll

# Compile to WASM
llc -march=wasm32 -filetype=obj prolog_wasm.ll -o prolog_wasm.o
wasm-ld-10 --no-entry --export-all prolog_wasm.o -o prolog_wasm.wasm
```

## Run Tests

```bash
node test.js
```

Expected output:
```
[PASS] sum(10) = 55
[PASS] sum(100) = 5050
[PASS] factorial(5) = 120
[PASS] factorial(10) = 3628800
INTEGRATION TEST PASSED
```

## Browser Usage

```html
<script type="module">
async function loadPrologMath() {
    const response = await fetch('prolog_wasm.wasm');
    const bytes = await response.arrayBuffer();
    const { instance } = await WebAssembly.instantiate(bytes);
    return instance.exports;
}

const math = await loadPrologMath();
console.log('sum(10) =', math.sum(10));       // 55
console.log('10! =', math.factorial(10));     // 3628800
</script>
```

## Files

```
examples/wasm/
├── prolog_wasm.ll    # Generated LLVM IR
├── prolog_wasm.wasm  # Compiled WASM module (376 bytes)
├── test.js           # Node.js test
└── build.sh          # Build script
```
