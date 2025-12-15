// Package embedder provides text embedding implementations for semantic search.
//
// # Backend Options
//
// The package supports multiple backends selected via build tags:
//
//   - Default (pure Go): No build tags needed
//     go build ./...
//
//   - Candle (Go+Rust): Use -tags candle
//     go build -tags candle ./...
//
//   - ORT (Go+C): Use -tags ort
//     go build -tags ort ./...
//
//   - XLA (Go+C++): Use -tags xla
//     go build -tags xla ./...
//
// # Portability Ranking (most to least portable)
//
//  1. Pure Go (default) - No external dependencies
//  2. Candle (Rust) - Requires Rust toolchain
//  3. ORT (C) - Requires ONNX Runtime + libtokenizers
//  4. XLA (C++) - Requires PJRT libraries
//
// # GPU Support
//
//   - Pure Go: CPU only
//   - Candle: CUDA support
//   - ORT: CUDA support
//   - XLA: CUDA, Metal, TPU support
package embedder

// Embedder is the common interface for all embedding backends.
// All implementations must be safe for concurrent use.
type Embedder interface {
	// Embed generates a vector embedding for the given text.
	// Returns nil, nil if the text is empty or cannot be embedded.
	Embed(text string) ([]float32, error)

	// Close releases any resources held by the embedder.
	// Must be called when the embedder is no longer needed.
	Close()
}

// EmbedderConfig holds configuration for creating embedders.
type EmbedderConfig struct {
	// ModelPath is the path to the model directory or file
	ModelPath string

	// ModelName is the name/identifier of the model
	ModelName string

	// OnnxFilename is the specific ONNX file to use (for ORT/XLA backends)
	// Required when model directory contains multiple .onnx files
	OnnxFilename string

	// Dimensions is the expected output dimension (0 = auto-detect)
	Dimensions int

	// UseGPU enables GPU acceleration if available
	UseGPU bool

	// MaxLength is the maximum token length (0 = model default)
	MaxLength int
}

// Backend represents the embedding backend type
type Backend string

const (
	BackendPureGo Backend = "purego"  // Pure Go (hugot NewGoSession)
	BackendCandle Backend = "candle"  // Rust candle via FFI
	BackendORT    Backend = "ort"     // ONNX Runtime (C)
	BackendXLA    Backend = "xla"     // OpenXLA/PJRT (C++)
	BackendStub   Backend = "stub"    // Stub for testing
)

// AvailableBackend returns the backend that will be used based on build tags.
// This is determined at compile time.
func AvailableBackend() Backend {
	return availableBackend
}

// NewEmbedder creates an embedder using the available backend.
// The backend is selected at compile time based on build tags.
func NewEmbedder(config EmbedderConfig) (Embedder, error) {
	return newEmbedder(config)
}
