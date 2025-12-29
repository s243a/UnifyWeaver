package main

/*
#cgo LDFLAGS: -L${SRCDIR} -lrpyc_bridge -lpython3.11
#include "rpyc_bridge.h"
#include <stdlib.h>
*/
import "C"
import (
    "encoding/json"
    "fmt"
    "unsafe"
)

// RPyCClient wraps the Rust FFI bridge for RPyC access
type RPyCClient struct {
    connected bool
}

// NewRPyCClient creates a new client and connects to the RPyC server
func NewRPyCClient(host string, port int) (*RPyCClient, error) {
    C.rpyc_init()

    hostC := C.CString(host)
    defer C.free(unsafe.Pointer(hostC))

    result := C.rpyc_connect(hostC, C.int(port))
    if result != 0 {
        return nil, fmt.Errorf("failed to connect to RPyC server at %%s:%%d", host, port)
    }

    return &RPyCClient{connected: true}, nil
}

// NewRPyCClientDefault connects to the default RPyC server
func NewRPyCClientDefault() (*RPyCClient, error) {
    return NewRPyCClient("localhost", 18812)
}

// Close disconnects from the RPyC server
func (c *RPyCClient) Close() {
    if c.connected {
        C.rpyc_disconnect()
        c.connected = false
    }
}

// Call invokes a function on a remote Python module
// Returns the result as a Go interface{}
func (c *RPyCClient) Call(module, function string, args ...interface{}) (interface{}, error) {
    if !c.connected {
        return nil, fmt.Errorf("not connected to RPyC server")
    }

    // Serialize args to JSON
    argsJSON, err := json.Marshal(args)
    if err != nil {
        return nil, fmt.Errorf("failed to serialize args: %%v", err)
    }

    moduleC := C.CString(module)
    funcC := C.CString(function)
    argsC := C.CString(string(argsJSON))
    defer C.free(unsafe.Pointer(moduleC))
    defer C.free(unsafe.Pointer(funcC))
    defer C.free(unsafe.Pointer(argsC))

    resultC := C.rpyc_call(moduleC, funcC, argsC)
    if resultC == nil {
        return nil, fmt.Errorf("rpyc_call failed for %%s.%%s", module, function)
    }
    defer C.rpyc_free_string(resultC)

    resultJSON := C.GoString(resultC)

    var result interface{}
    if err := json.Unmarshal([]byte(resultJSON), &result); err != nil {
        // Return as raw string if not valid JSON
        return resultJSON, nil
    }

    return result, nil
}

// GetAttr gets an attribute from a remote Python module
func (c *RPyCClient) GetAttr(module, attr string) (interface{}, error) {
    if !c.connected {
        return nil, fmt.Errorf("not connected to RPyC server")
    }

    moduleC := C.CString(module)
    attrC := C.CString(attr)
    defer C.free(unsafe.Pointer(moduleC))
    defer C.free(unsafe.Pointer(attrC))

    resultC := C.rpyc_getattr(moduleC, attrC)
    if resultC == nil {
        return nil, fmt.Errorf("rpyc_getattr failed for %%s.%%s", module, attr)
    }
    defer C.rpyc_free_string(resultC)

    resultJSON := C.GoString(resultC)

    var result interface{}
    if err := json.Unmarshal([]byte(resultJSON), &result); err != nil {
        return resultJSON, nil
    }

    return result, nil
}

// IsConnected checks if the client is connected
func (c *RPyCClient) IsConnected() bool {
    return c.connected && C.rpyc_is_connected() == 1
}

// Example usage (uncomment for standalone test):
// func main() {
//     client, err := NewRPyCClientDefault()
//     if err != nil {
//         fmt.Println("Error:", err)
//         return
//     }
//     defer client.Close()
//
//     // Call math.sqrt(16)
//     result, err := client.Call("math", "sqrt", 16)
//     if err != nil {
//         fmt.Println("Error:", err)
//         return
//     }
//     fmt.Printf("math.sqrt(16) = %%v\n", result)
//
//     // Get math.pi
//     pi, err := client.GetAttr("math", "pi")
//     if err != nil {
//         fmt.Println("Error:", err)
//         return
//     }
//     fmt.Printf("math.pi = %%v\n", pi)
// }
