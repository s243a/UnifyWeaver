// LLVM FFI Integration Test for Go
// Tests calling LLVM-compiled Prolog predicates via cgo
package main

/*
#cgo LDFLAGS: -L.. -lprolog_math -Wl,-rpath,${SRCDIR}/..
#include "../prolog_math.h"
*/
import "C"
import "fmt"

// Sum calls the LLVM-compiled sum function (tail recursion with musttail)
func Sum(n int64) int64 {
	return int64(C.sum(C.int64_t(n)))
}

// Factorial calls the LLVM-compiled factorial function
func Factorial(n int64) int64 {
	return int64(C.factorial(C.int64_t(n)))
}

func main() {
	fmt.Println("===========================================")
	fmt.Println("LLVM FFI Integration Test - Go")
	fmt.Println("===========================================")
	fmt.Println()

	// Test sum (1 + 2 + ... + n)
	tests := []struct {
		name     string
		fn       func(int64) int64
		input    int64
		expected int64
	}{
		{"Sum(10)", Sum, 10, 55},
		{"Sum(100)", Sum, 100, 5050},
		{"Factorial(5)", Factorial, 5, 120},
		{"Factorial(10)", Factorial, 10, 3628800},
	}

	passed := 0
	failed := 0

	for _, test := range tests {
		result := test.fn(test.input)
		if result == test.expected {
			fmt.Printf("[PASS] %s = %d\n", test.name, result)
			passed++
		} else {
			fmt.Printf("[FAIL] %s = %d (expected %d)\n", test.name, result, test.expected)
			failed++
		}
	}

	fmt.Println()
	fmt.Printf("Results: %d passed, %d failed\n", passed, failed)
	fmt.Println("===========================================")

	if failed > 0 {
		fmt.Println("INTEGRATION TEST FAILED")
	} else {
		fmt.Println("INTEGRATION TEST PASSED")
	}
}
