package embedder

import (
	"encoding/binary"
	"math"
	"os"
)

// StubEmbedder provides deterministic embeddings without external dependencies
// Useful for testing the pipeline without real ML models
type StubEmbedder struct {
	dims int
}

// NewStubEmbedder creates a stub embedder that returns zero vectors
// This is useful for testing the pipeline without real embeddings
func NewStubEmbedder(dims int) *StubEmbedder {
	return &StubEmbedder{dims: dims}
}

func (e *StubEmbedder) Embed(text string) ([]float32, error) {
	// Return a simple hash-based vector for testing
	// This won't give meaningful semantic results but allows testing the pipeline
	vec := make([]float32, e.dims)

	// Simple deterministic "embedding" based on text hash
	hash := uint32(0)
	for _, c := range text {
		hash = hash*31 + uint32(c)
	}

	for i := 0; i < e.dims; i++ {
		// Generate pseudo-random values based on hash and position
		seed := hash ^ uint32(i*1000003)
		vec[i] = float32(seed%1000) / 1000.0 * 2.0 - 1.0
	}

	// Normalize
	var norm float32
	for _, v := range vec {
		norm += v * v
	}
	norm = float32(math.Sqrt(float64(norm)))
	if norm > 0 {
		for i := range vec {
			vec[i] /= norm
		}
	}

	return vec, nil
}

func (e *StubEmbedder) Close() {}

// LoadEmbeddingsFromFile loads pre-computed embeddings from a binary file
func LoadEmbeddingsFromFile(path string) (map[string][]float32, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	embeddings := make(map[string][]float32)
	offset := 0

	for offset < len(data) {
		// Read ID length
		if offset+4 > len(data) {
			break
		}
		idLen := int(binary.LittleEndian.Uint32(data[offset:]))
		offset += 4

		// Read ID
		if offset+idLen > len(data) {
			break
		}
		id := string(data[offset : offset+idLen])
		offset += idLen

		// Read vector length
		if offset+4 > len(data) {
			break
		}
		vecLen := int(binary.LittleEndian.Uint32(data[offset:]))
		offset += 4

		// Read vector
		if offset+vecLen*4 > len(data) {
			break
		}
		vec := make([]float32, vecLen)
		for i := 0; i < vecLen; i++ {
			bits := binary.LittleEndian.Uint32(data[offset:])
			vec[i] = math.Float32frombits(bits)
			offset += 4
		}

		embeddings[id] = vec
	}

	return embeddings, nil
}
