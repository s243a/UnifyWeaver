package main

/*
#cgo LDFLAGS: -L${SRCDIR} -lrpyc_bridge
#include "rpyc_bridge.h"
#include <stdlib.h>
*/
import "C"
import (
	"encoding/json"
	"fmt"
	"os"
	"unsafe"
)

func main() {
	fmt.Println("Go + Rust FFI + RPyC Integration")
	fmt.Println("================================")
	fmt.Println()

	// Initialize Python
	C.rpyc_init()

	// Connect to RPyC server
	fmt.Println("Connecting to RPyC server...")
	host := C.CString("localhost")
	defer C.free(unsafe.Pointer(host))

	if C.rpyc_connect(host, 18812) != 0 {
		fmt.Println("  Failed to connect!")
		fmt.Println()
		fmt.Println("Make sure RPyC server is running:")
		fmt.Println("  python examples/rpyc-integration/rpyc_server.py")
		os.Exit(1)
	}
	fmt.Println("  Connected!")
	defer C.rpyc_disconnect()

	// Test 1: math.sqrt(16)
	fmt.Println()
	fmt.Println("Test 1: math.sqrt(16)")
	result1 := callPython("math", "sqrt", []interface{}{16})
	fmt.Printf("  Result: %v\n", result1)
	if val, ok := result1.(float64); !ok || val != 4.0 {
		fmt.Println("  FAILED!")
		os.Exit(1)
	}
	fmt.Println("  PASSED!")

	// Test 2: numpy.mean([1,2,3,4,5])
	fmt.Println()
	fmt.Println("Test 2: numpy.mean([1,2,3,4,5])")
	result2 := callPython("numpy", "mean", []interface{}{[]int{1, 2, 3, 4, 5}})
	fmt.Printf("  Result: %v\n", result2)
	if val, ok := result2.(float64); !ok || val != 3.0 {
		fmt.Println("  FAILED!")
		os.Exit(1)
	}
	fmt.Println("  PASSED!")

	// Test 3: Get math.pi
	fmt.Println()
	fmt.Println("Test 3: math.pi")
	pi := getAttr("math", "pi")
	fmt.Printf("  Result: %v\n", pi)
	if val, ok := pi.(float64); !ok || val < 3.14 || val > 3.15 {
		fmt.Println("  FAILED!")
		os.Exit(1)
	}
	fmt.Println("  PASSED!")

	fmt.Println()
	fmt.Println("================================")
	fmt.Println("All tests passed!")
}

func callPython(module, function string, args []interface{}) interface{} {
	argsJSON, err := json.Marshal(args)
	if err != nil {
		fmt.Printf("Error marshaling args: %v\n", err)
		return nil
	}

	moduleC := C.CString(module)
	funcC := C.CString(function)
	argsC := C.CString(string(argsJSON))
	defer C.free(unsafe.Pointer(moduleC))
	defer C.free(unsafe.Pointer(funcC))
	defer C.free(unsafe.Pointer(argsC))

	resultC := C.rpyc_call(moduleC, funcC, argsC)
	if resultC == nil {
		fmt.Printf("Error calling %s.%s\n", module, function)
		return nil
	}
	defer C.rpyc_free_string(resultC)

	resultJSON := C.GoString(resultC)

	var result interface{}
	if err := json.Unmarshal([]byte(resultJSON), &result); err != nil {
		return resultJSON
	}
	return result
}

func getAttr(module, attr string) interface{} {
	moduleC := C.CString(module)
	attrC := C.CString(attr)
	defer C.free(unsafe.Pointer(moduleC))
	defer C.free(unsafe.Pointer(attrC))

	resultC := C.rpyc_getattr(moduleC, attrC)
	if resultC == nil {
		fmt.Printf("Error getting %s.%s\n", module, attr)
		return nil
	}
	defer C.rpyc_free_string(resultC)

	resultJSON := C.GoString(resultC)

	var result interface{}
	if err := json.Unmarshal([]byte(resultJSON), &result); err != nil {
		return resultJSON
	}
	return result
}
