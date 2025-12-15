#!/bin/bash
# Build Haskell integration test
set -e

cd "$(dirname "$0")"

echo "=== Generating Haskell from Prolog ==="
pushd ../.. > /dev/null
swipl -g "use_module('src/unifyweaver/targets/haskell_target'), \
  compile_module_to_haskell([pred(sumTo, 2, tail_recursion), pred(factorial, 1, factorial), pred(fib, 2, linear_recursion)], \
  [module_name('PrologMath')], Code), write(Code)" -t halt > examples/haskell/PrologMath.hs
popd > /dev/null

# Fix Prelude conflict
sed -i 's/sumTo, factorial/sumTo, factorial/' PrologMath.hs
sed -i '5a import Prelude hiding (sum)' PrologMath.hs

echo "=== Compiling with GHC ==="
ghc -O2 Main.hs -o haskell-test

echo ""
echo "=== Running tests ==="
./haskell-test
