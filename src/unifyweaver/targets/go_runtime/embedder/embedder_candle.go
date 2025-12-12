//go:build candle

package embedder

/*
#cgo LDFLAGS: -L${SRCDIR}/../lib -lcandle_semantic_router
#cgo linux LDFLAGS: -lm -ldl -lpthread
#cgo darwin LDFLAGS: -framework Accelerate

#include <stdlib.h>

// FFI declarations for candle_semantic_router library
// These match the Rust library's C API

extern int candle_init_embedding_model(const char* model_path, int use_gpu);
extern int candle_get_embedding(const char* text, float* output, int max_dim);
extern int candle_get_embedding_dim();
extern void candle_cleanup();
*/
import "C"

import (
	"fmt"
	"sync"
	"unsafe"
)

// availableBackend indicates this is the Candle (Rust) backend
var availableBackend = BackendCandle

// candleEmbedder uses the Rust candle library via FFI
type candleEmbedder struct {
	dimensions int
	mu         sync.Mutex
}

var (
	candleOnce     sync.Once
	candleInitErr  error
	candleInstance *candleEmbedder
)

// newEmbedder creates a Candle embedder using Rust FFI
func newEmbedder(config EmbedderConfig) (Embedder, error) {
	candleOnce.Do(func() {
		modelPath := C.CString(config.ModelPath)
		defer C.free(unsafe.Pointer(modelPath))

		useGPU := C.int(0)
		if config.UseGPU {
			useGPU = 1
		}

		ret := C.candle_init_embedding_model(modelPath, useGPU)
		if ret != 0 {
			candleInitErr = fmt.Errorf("candle init failed with code %d", ret)
			return
		}

		dim := int(C.candle_get_embedding_dim())
		if dim <= 0 {
			dim = 384 // default for MiniLM
		}

		candleInstance = &candleEmbedder{
			dimensions: dim,
		}
	})

	if candleInitErr != nil {
		return nil, candleInitErr
	}

	return candleInstance, nil
}

func (e *candleEmbedder) Embed(text string) ([]float32, error) {
	if text == "" {
		return nil, nil
	}

	e.mu.Lock()
	defer e.mu.Unlock()

	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))

	output := make([]float32, e.dimensions)
	ret := C.candle_get_embedding(cText, (*C.float)(unsafe.Pointer(&output[0])), C.int(e.dimensions))
	if ret != 0 {
		return nil, fmt.Errorf("candle embedding failed with code %d", ret)
	}

	return output, nil
}

func (e *candleEmbedder) Close() {
	e.mu.Lock()
	defer e.mu.Unlock()
	C.candle_cleanup()
}
