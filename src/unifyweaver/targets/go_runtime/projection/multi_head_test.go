package projection

import (
	"math"
	"os"
	"path/filepath"
	"testing"
)

func TestSoftmax(t *testing.T) {
	// Test with temperature 1.0
	input := []float32{0.85, 0.70, 0.60}
	result := softmax(input, 1.0)

	// Sum should be 1.0
	var sum float32
	for _, v := range result {
		sum += v
	}
	if math.Abs(float64(sum-1.0)) > 0.001 {
		t.Errorf("softmax sum = %f, want 1.0", sum)
	}

	// First value should be highest
	if result[0] < result[1] || result[0] < result[2] {
		t.Errorf("softmax[0] should be highest, got %v", result)
	}
}

func TestSoftmaxTemperature(t *testing.T) {
	input := []float32{0.85, 0.70, 0.60}

	// Low temperature should produce sharper distribution
	sharpResult := softmax(input, 0.1)
	// High temperature should produce more uniform distribution
	diffuseResult := softmax(input, 1.0)

	// The difference between first and second should be larger with low temp
	sharpDiff := sharpResult[0] - sharpResult[1]
	diffuseDiff := diffuseResult[0] - diffuseResult[1]

	if sharpDiff <= diffuseDiff {
		t.Errorf("low temp should produce sharper distribution: sharp=%v, diffuse=%v",
			sharpResult, diffuseResult)
	}
}

func TestNormalize(t *testing.T) {
	v := []float32{3, 4}
	result := normalize(v)

	// Norm should be 1.0
	var norm float32
	for _, val := range result {
		norm += val * val
	}
	norm = float32(math.Sqrt(float64(norm)))

	if math.Abs(float64(norm-1.0)) > 0.001 {
		t.Errorf("normalized vector norm = %f, want 1.0", norm)
	}
}

func TestDotProduct(t *testing.T) {
	a := []float32{1, 2, 3}
	b := []float32{4, 5, 6}

	result := dotProduct(a, b)
	expected := float32(1*4 + 2*5 + 3*6) // 32

	if result != expected {
		t.Errorf("dotProduct = %f, want %f", result, expected)
	}
}

func TestMultiHeadProjectionWithMockData(t *testing.T) {
	// Create mock heads directly
	mh := &MultiHeadProjection{
		Temperature: 0.1,
		Dimension:   4,
		Heads: []Head{
			{
				ClusterID: 1,
				Centroid:  []float32{1, 0, 0, 0},
				AnswerEmb: []float32{0.5, 0.5, 0, 0},
			},
			{
				ClusterID: 2,
				Centroid:  []float32{0, 1, 0, 0},
				AnswerEmb: []float32{0, 0, 0.5, 0.5},
			},
		},
	}

	// Query close to first centroid
	query := []float32{0.9, 0.1, 0, 0}
	projected, weights, err := mh.ProjectWithWeights(query)
	if err != nil {
		t.Fatalf("ProjectWithWeights failed: %v", err)
	}

	// Should route mostly to first head
	if weights[1] < weights[2] {
		t.Errorf("expected head 1 to dominate: %v", weights)
	}

	// Projected should be closer to first answer embedding
	if projected[0] < 0.4 || projected[1] < 0.4 {
		t.Errorf("projected should be similar to first answer: %v", projected)
	}
}

func TestParseNpyShape(t *testing.T) {
	tests := []struct {
		header   string
		expected []int
	}{
		{
			header:   "{'descr': '<f4', 'fortran_order': False, 'shape': (384,), }",
			expected: []int{384},
		},
		{
			header:   "{'descr': '<f4', 'fortran_order': False, 'shape': (10, 384), }",
			expected: []int{10, 384},
		},
		{
			header:   "{'descr': '<f4', 'fortran_order': False, 'shape': (), }",
			expected: []int{1},
		},
	}

	for _, tc := range tests {
		result, err := parseNpyShape(tc.header)
		if err != nil {
			t.Errorf("parseNpyShape(%q) error: %v", tc.header, err)
			continue
		}

		if len(result) != len(tc.expected) {
			t.Errorf("parseNpyShape(%q) = %v, want %v", tc.header, result, tc.expected)
			continue
		}

		for i := range result {
			if result[i] != tc.expected[i] {
				t.Errorf("parseNpyShape(%q)[%d] = %d, want %d",
					tc.header, i, result[i], tc.expected[i])
			}
		}
	}
}

func TestLoadNpyFloat32(t *testing.T) {
	// Create a temporary test npy file
	tmpDir := t.TempDir()
	npyPath := filepath.Join(tmpDir, "test.npy")

	// Write a simple 1D array: [1.0, 2.0, 3.0]
	// NPY format: magic + version + header_len + header + data
	data := make([]byte, 0)

	// Magic number
	data = append(data, 0x93, 'N', 'U', 'M', 'P', 'Y')

	// Version 1.0
	data = append(data, 1, 0)

	// Header (must be multiple of 64 bytes including 10-byte preamble)
	header := "{'descr': '<f4', 'fortran_order': False, 'shape': (3,), }"
	// Pad to make total (10 + headerLen) divisible by 64
	for len(header)%64 != 54 { // 64 - 10 = 54
		header += " "
	}
	header += "\n"

	// Header length (little-endian uint16)
	headerLen := uint16(len(header))
	data = append(data, byte(headerLen&0xff), byte(headerLen>>8))

	// Header
	data = append(data, []byte(header)...)

	// Data: 3 float32 values in little-endian
	floatData := []float32{1.0, 2.0, 3.0}
	for _, f := range floatData {
		bits := math.Float32bits(f)
		data = append(data, byte(bits), byte(bits>>8), byte(bits>>16), byte(bits>>24))
	}

	if err := os.WriteFile(npyPath, data, 0644); err != nil {
		t.Fatalf("writing test npy file: %v", err)
	}

	// Load and verify
	result, err := loadNpyFloat32(npyPath)
	if err != nil {
		t.Fatalf("loadNpyFloat32 failed: %v", err)
	}

	if len(result) != 3 {
		t.Errorf("loadNpyFloat32 returned %d elements, want 3", len(result))
	}

	for i, expected := range floatData {
		if i < len(result) && result[i] != expected {
			t.Errorf("result[%d] = %f, want %f", i, result[i], expected)
		}
	}
}
