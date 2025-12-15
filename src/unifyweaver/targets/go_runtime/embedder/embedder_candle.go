//go:build candle

package embedder

/*
#cgo LDFLAGS: -L/usr/local/lib -lcandle_semantic_router
#cgo linux LDFLAGS: -lm -ldl -lpthread
#cgo darwin LDFLAGS: -framework Accelerate

#include <stdlib.h>
#include <stdbool.h>

// EmbeddingResult matches the Rust FFI struct
// Note: Field order must match Rust exactly
typedef struct {
    float* data;
    int length;
    bool error;
    int model_type;
    int sequence_length;
    float processing_time_ms;
} EmbeddingResult;

// FFI declarations for candle_semantic_router library
// These match the Rust library's C API (from src/ffi/init.rs and src/ffi/similarity.rs)

// Initialize BERT similarity model for sentence embeddings
// model_id: HuggingFace model ID (e.g., "sentence-transformers/all-MiniLM-L6-v2")
//           or local path to model directory
// use_cpu: true for CPU, false for GPU (CUDA)
// Returns: true on success, false on failure
extern bool init_similarity_model(const char* model_id, bool use_cpu);

// Check if similarity model is initialized
extern bool is_similarity_model_initialized();

// Get text embedding from initialized model
// text: Input text (null-terminated C string)
// max_length: Maximum sequence length (0 for default)
// Returns: EmbeddingResult struct (caller must free data with free_embedding)
extern EmbeddingResult get_text_embedding(const char* text, int max_length);

// Free embedding data allocated by Rust
extern void free_embedding(float* data, int length);
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
	maxLength  int
	mu         sync.Mutex
}

var (
	candleOnce     sync.Once
	candleInitErr  error
	candleInstance *candleEmbedder
)

// newEmbedder creates a Candle embedder using Rust FFI
// Uses the semantic-router library's BERT similarity model
func newEmbedder(config EmbedderConfig) (Embedder, error) {
	candleOnce.Do(func() {
		modelPath := C.CString(config.ModelPath)
		defer C.free(unsafe.Pointer(modelPath))

		// use_cpu is inverted from UseGPU
		useCPU := C.bool(!config.UseGPU)

		// Initialize the BERT similarity model
		success := C.init_similarity_model(modelPath, useCPU)
		if !success {
			candleInitErr = fmt.Errorf("candle init_similarity_model failed for model: %s", config.ModelPath)
			return
		}

		// Default dimensions for common models
		dim := config.Dimensions
		if dim <= 0 {
			dim = 384 // default for all-MiniLM-L6-v2
		}

		maxLen := config.MaxLength
		if maxLen <= 0 {
			maxLen = 512 // default max sequence length
		}

		candleInstance = &candleEmbedder{
			dimensions: dim,
			maxLength:  maxLen,
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

	// Check model is initialized
	if !C.is_similarity_model_initialized() {
		return nil, fmt.Errorf("candle similarity model not initialized")
	}

	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))

	// Get embedding from Rust library
	result := C.get_text_embedding(cText, C.int(e.maxLength))

	// Check for errors
	if result.error {
		return nil, fmt.Errorf("candle get_text_embedding failed")
	}

	if result.data == nil || result.length <= 0 {
		return nil, fmt.Errorf("candle returned empty embedding")
	}

	// Copy data to Go slice before freeing Rust memory
	length := int(result.length)
	output := make([]float32, length)

	// Convert C float* to Go slice
	cSlice := unsafe.Slice((*float32)(unsafe.Pointer(result.data)), length)
	copy(output, cSlice)

	// Free Rust-allocated memory
	C.free_embedding(result.data, result.length)

	return output, nil
}

func (e *candleEmbedder) Close() {
	// The Rust library handles cleanup internally via OnceLock
	// No explicit cleanup needed - the model lives for the process lifetime
}
