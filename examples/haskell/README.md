# Haskell Examples

Working examples of Prolog compiled to Haskell.

## Prerequisites

```bash
sudo apt install ghc
```

## Quick Test

```bash
./build.sh
```

## Files

```
examples/haskell/
├── PrologMath.hs  # Generated Haskell module
├── Main.hs        # Test program
├── haskell-test   # Compiled binary
└── build.sh       # Build script
```

## Generated Code

`PrologMath.hs`:
```haskell
{-# LANGUAGE BangPatterns #-}
module PrologMath (sumTo, factorial, fib) where

sumTo :: Int -> Int -> Int
sumTo 0 !acc = acc
sumTo n !acc = sumTo (n - 1) (acc + n)

factorial :: Int -> Int
factorial 0 = 1
factorial n = n * factorial (n - 1)

fib :: Int -> Int
fib 0 = 0
fib 1 = 1
fib n = fib (n - 1) + fib (n - 2)
```

## Test Results

```
[PASS] sumTo 10 0 = 55
[PASS] sumTo 100 0 = 5050
[PASS] factorial 5 = 120
[PASS] factorial 10 = 3628800
[PASS] fib 10 = 55
[PASS] fib 15 = 610
INTEGRATION TEST PASSED
```
