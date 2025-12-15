// Package projection provides LDA-based semantic projection for improved RAG retrieval.
//
// Multi-head projection uses per-cluster centroids and answer embeddings to route
// queries, similar to transformer attention heads. Each "head" corresponds to a
// Q-A cluster, with its own centroid (mean of training questions) and answer embedding.
//
// The projection process:
//  1. Compute query similarity to all centroids
//  2. Apply softmax with temperature to get routing weights
//  3. Return weighted combination of answer embeddings as "projected" query
//
// See: docs/proposals/MULTI_HEAD_PROJECTION_THEORY.md
package projection

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
)

// Head represents a single projection head (one Q-A cluster).
type Head struct {
	ClusterID  int
	Centroid   []float32 // Mean of training question embeddings
	AnswerEmb  []float32 // Answer embedding for this cluster
}

// MultiHeadProjection holds the multi-head projection model.
type MultiHeadProjection struct {
	Heads       []Head
	Temperature float32
	Dimension   int
}

// Config for loading multi-head projection.
type Config struct {
	// DataDir is the directory containing centroid and answer embedding files
	DataDir string

	// Temperature for softmax routing (lower = sharper routing)
	// Recommended: 0.1
	Temperature float32

	// HeadFiles maps cluster IDs to their centroid/answer embedding file pairs
	// If nil, auto-discovers from DataDir
	HeadFiles map[int]HeadFilePair
}

// HeadFilePair contains paths for a single head's data.
type HeadFilePair struct {
	CentroidPath  string
	AnswerEmbPath string
}

// LoadMultiHead loads a multi-head projection from numpy files.
func LoadMultiHead(config Config) (*MultiHeadProjection, error) {
	if config.Temperature <= 0 {
		config.Temperature = 0.1 // Default
	}

	mh := &MultiHeadProjection{
		Temperature: config.Temperature,
	}

	// If HeadFiles provided, use them directly
	if len(config.HeadFiles) > 0 {
		for clusterID, files := range config.HeadFiles {
			head, err := loadHead(clusterID, files.CentroidPath, files.AnswerEmbPath)
			if err != nil {
				return nil, fmt.Errorf("loading head %d: %w", clusterID, err)
			}
			mh.Heads = append(mh.Heads, head)
			if mh.Dimension == 0 {
				mh.Dimension = len(head.Centroid)
			}
		}
		return mh, nil
	}

	// Auto-discover from DataDir
	if config.DataDir == "" {
		return nil, fmt.Errorf("either DataDir or HeadFiles must be provided")
	}

	// Look for centroid_*.npy and answer_emb_*.npy pairs
	centroids, err := filepath.Glob(filepath.Join(config.DataDir, "centroid_*.npy"))
	if err != nil {
		return nil, fmt.Errorf("searching for centroids: %w", err)
	}

	for _, centroidPath := range centroids {
		// Extract cluster ID from filename (centroid_1.npy -> 1)
		var clusterID int
		base := filepath.Base(centroidPath)
		_, err := fmt.Sscanf(base, "centroid_%d.npy", &clusterID)
		if err != nil {
			continue // Skip files that don't match pattern
		}

		// Find corresponding answer embedding
		answerPath := filepath.Join(config.DataDir, fmt.Sprintf("answer_emb_%d.npy", clusterID))
		if _, err := os.Stat(answerPath); os.IsNotExist(err) {
			continue // Skip if no answer embedding found
		}

		head, err := loadHead(clusterID, centroidPath, answerPath)
		if err != nil {
			return nil, fmt.Errorf("loading head %d: %w", clusterID, err)
		}
		mh.Heads = append(mh.Heads, head)
		if mh.Dimension == 0 {
			mh.Dimension = len(head.Centroid)
		}
	}

	if len(mh.Heads) == 0 {
		return nil, fmt.Errorf("no heads found in %s", config.DataDir)
	}

	return mh, nil
}

// loadHead loads a single head from numpy files.
func loadHead(clusterID int, centroidPath, answerPath string) (Head, error) {
	centroid, err := loadNpyFloat32(centroidPath)
	if err != nil {
		return Head{}, fmt.Errorf("loading centroid: %w", err)
	}

	answerEmb, err := loadNpyFloat32(answerPath)
	if err != nil {
		return Head{}, fmt.Errorf("loading answer embedding: %w", err)
	}

	return Head{
		ClusterID: clusterID,
		Centroid:  centroid,
		AnswerEmb: answerEmb,
	}, nil
}

// Project applies multi-head projection to a query embedding.
// Returns the projected embedding as a weighted combination of answer embeddings.
func (mh *MultiHeadProjection) Project(queryEmb []float32) ([]float32, error) {
	if len(queryEmb) != mh.Dimension {
		return nil, fmt.Errorf("query dimension %d != model dimension %d", len(queryEmb), mh.Dimension)
	}

	if len(mh.Heads) == 0 {
		return nil, fmt.Errorf("no heads loaded")
	}

	// Normalize query
	queryNormed := normalize(queryEmb)

	// Compute similarity to each centroid
	similarities := make([]float32, len(mh.Heads))
	for i, head := range mh.Heads {
		centroidNormed := normalize(head.Centroid)
		similarities[i] = dotProduct(queryNormed, centroidNormed)
	}

	// Apply softmax with temperature
	weights := softmax(similarities, mh.Temperature)

	// Weighted combination of answer embeddings
	projected := make([]float32, mh.Dimension)
	for i, head := range mh.Heads {
		for j, val := range head.AnswerEmb {
			projected[j] += weights[i] * val
		}
	}

	return projected, nil
}

// ProjectWithWeights applies multi-head projection and returns both the
// projected embedding and the routing weights for each head.
func (mh *MultiHeadProjection) ProjectWithWeights(queryEmb []float32) ([]float32, map[int]float32, error) {
	if len(queryEmb) != mh.Dimension {
		return nil, nil, fmt.Errorf("query dimension %d != model dimension %d", len(queryEmb), mh.Dimension)
	}

	if len(mh.Heads) == 0 {
		return nil, nil, fmt.Errorf("no heads loaded")
	}

	// Normalize query
	queryNormed := normalize(queryEmb)

	// Compute similarity to each centroid
	similarities := make([]float32, len(mh.Heads))
	for i, head := range mh.Heads {
		centroidNormed := normalize(head.Centroid)
		similarities[i] = dotProduct(queryNormed, centroidNormed)
	}

	// Apply softmax with temperature
	weights := softmax(similarities, mh.Temperature)

	// Build weight map
	weightMap := make(map[int]float32)
	for i, head := range mh.Heads {
		weightMap[head.ClusterID] = weights[i]
	}

	// Weighted combination of answer embeddings
	projected := make([]float32, mh.Dimension)
	for i, head := range mh.Heads {
		for j, val := range head.AnswerEmb {
			projected[j] += weights[i] * val
		}
	}

	return projected, weightMap, nil
}

// NumHeads returns the number of projection heads.
func (mh *MultiHeadProjection) NumHeads() int {
	return len(mh.Heads)
}

// GetTemperature returns the softmax temperature.
func (mh *MultiHeadProjection) GetTemperature() float32 {
	return mh.Temperature
}

// SetTemperature updates the softmax temperature.
func (mh *MultiHeadProjection) SetTemperature(t float32) {
	if t > 0 {
		mh.Temperature = t
	}
}

// ============================================================================
// Helper functions
// ============================================================================

// softmax applies softmax with temperature to a slice of values.
func softmax(x []float32, temperature float32) []float32 {
	if len(x) == 0 {
		return nil
	}

	// Scale by temperature
	scaled := make([]float64, len(x))
	maxVal := float64(x[0]) / float64(temperature)
	for i, v := range x {
		scaled[i] = float64(v) / float64(temperature)
		if scaled[i] > maxVal {
			maxVal = scaled[i]
		}
	}

	// Subtract max for numerical stability
	sum := 0.0
	for i := range scaled {
		scaled[i] = math.Exp(scaled[i] - maxVal)
		sum += scaled[i]
	}

	// Normalize
	result := make([]float32, len(x))
	for i := range scaled {
		result[i] = float32(scaled[i] / sum)
	}

	return result
}

// normalize returns a unit vector.
func normalize(v []float32) []float32 {
	norm := float32(0)
	for _, val := range v {
		norm += val * val
	}
	norm = float32(math.Sqrt(float64(norm)))

	if norm == 0 {
		return v
	}

	result := make([]float32, len(v))
	for i, val := range v {
		result[i] = val / norm
	}
	return result
}

// dotProduct computes dot product of two vectors.
func dotProduct(a, b []float32) float32 {
	if len(a) != len(b) {
		return 0
	}
	sum := float32(0)
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}

// loadNpyFloat32 loads a 1D numpy array of float32 values.
// Supports basic .npy format (version 1.0).
func loadNpyFloat32(path string) ([]float32, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	// Read magic number
	magic := make([]byte, 6)
	if _, err := io.ReadFull(f, magic); err != nil {
		return nil, fmt.Errorf("reading magic: %w", err)
	}
	if string(magic) != "\x93NUMPY" {
		return nil, fmt.Errorf("invalid npy magic: %v", magic)
	}

	// Read version
	version := make([]byte, 2)
	if _, err := io.ReadFull(f, version); err != nil {
		return nil, fmt.Errorf("reading version: %w", err)
	}

	// Read header length
	var headerLen uint16
	if version[0] == 1 {
		if err := binary.Read(f, binary.LittleEndian, &headerLen); err != nil {
			return nil, fmt.Errorf("reading header length: %w", err)
		}
	} else if version[0] == 2 || version[0] == 3 {
		var headerLen32 uint32
		if err := binary.Read(f, binary.LittleEndian, &headerLen32); err != nil {
			return nil, fmt.Errorf("reading header length: %w", err)
		}
		headerLen = uint16(headerLen32)
	} else {
		return nil, fmt.Errorf("unsupported npy version: %d.%d", version[0], version[1])
	}

	// Read header (contains dtype and shape info)
	header := make([]byte, headerLen)
	if _, err := io.ReadFull(f, header); err != nil {
		return nil, fmt.Errorf("reading header: %w", err)
	}

	// Parse shape from header string
	// Format: {'descr': '<f4', 'fortran_order': False, 'shape': (384,), }
	headerStr := string(header)
	shape, err := parseNpyShape(headerStr)
	if err != nil {
		return nil, fmt.Errorf("parsing shape: %w", err)
	}

	// Calculate total elements
	totalElements := 1
	for _, dim := range shape {
		totalElements *= dim
	}

	// Read data
	data := make([]float32, totalElements)
	if err := binary.Read(f, binary.LittleEndian, &data); err != nil {
		return nil, fmt.Errorf("reading data: %w", err)
	}

	return data, nil
}

// parseNpyShape extracts shape from numpy header string.
func parseNpyShape(header string) ([]int, error) {
	// Find 'shape': (...)
	start := -1
	for i := 0; i < len(header)-7; i++ {
		if header[i:i+7] == "'shape'" {
			start = i + 7
			break
		}
	}
	if start == -1 {
		return nil, fmt.Errorf("shape not found in header")
	}

	// Find opening paren
	for i := start; i < len(header); i++ {
		if header[i] == '(' {
			start = i + 1
			break
		}
	}

	// Find closing paren
	end := -1
	for i := start; i < len(header); i++ {
		if header[i] == ')' {
			end = i
			break
		}
	}
	if end == -1 {
		return nil, fmt.Errorf("closing paren not found")
	}

	// Parse dimensions
	shapeStr := header[start:end]
	if shapeStr == "" {
		return []int{1}, nil // Scalar
	}

	var shape []int
	var current int
	hasDigit := false
	for _, c := range shapeStr {
		if c >= '0' && c <= '9' {
			current = current*10 + int(c-'0')
			hasDigit = true
		} else if c == ',' || c == ' ' {
			if hasDigit {
				shape = append(shape, current)
				current = 0
				hasDigit = false
			}
		}
	}
	if hasDigit {
		shape = append(shape, current)
	}

	if len(shape) == 0 {
		return []int{1}, nil
	}
	return shape, nil
}
