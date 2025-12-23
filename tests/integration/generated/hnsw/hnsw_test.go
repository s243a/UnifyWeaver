package hnsw

import (
	"math/rand"
	"testing"
)

func TestNewHNSWGraph(t *testing.T) {
	g := NewHNSWGraph(16)
	if g == nil {
		t.Fatal("NewHNSWGraph returned nil")
	}
	if g.M != 16 {
		t.Errorf("M = %d, want 16", g.M)
	}
	if g.M0 != 32 {
		t.Errorf("M0 = %d, want 32", g.M0)
	}
}

func TestAddNode(t *testing.T) {
	g := NewHNSWGraph(16)
	rng := rand.New(rand.NewSource(42))

	// Add first node
	n1 := g.AddNode("n1", []float32{1.0, 0.0, 0.0}, rng)
	if n1 == nil {
		t.Fatal("AddNode returned nil")
	}
	if g.EntryPointID != "n1" {
		t.Errorf("EntryPointID = %s, want n1", g.EntryPointID)
	}

	// Add more nodes
	g.AddNode("n2", []float32{0.9, 0.1, 0.0}, rng)
	g.AddNode("n3", []float32{0.8, 0.2, 0.0}, rng)
	g.AddNode("n4", []float32{0.0, 1.0, 0.0}, rng)

	if len(g.Nodes) != 4 {
		t.Errorf("len(Nodes) = %d, want 4", len(g.Nodes))
	}
}

func TestSearch(t *testing.T) {
	g := NewHNSWGraph(16)
	rng := rand.New(rand.NewSource(42))

	// Build index
	vectors := []struct {
		id     string
		vector []float32
	}{
		{"n1", []float32{1.0, 0.0, 0.0}},
		{"n2", []float32{0.9, 0.1, 0.0}},
		{"n3", []float32{0.8, 0.2, 0.0}},
		{"n4", []float32{0.0, 1.0, 0.0}},
		{"n5", []float32{0.1, 0.9, 0.0}},
		{"n6", []float32{0.5, 0.5, 0.0}},
	}

	for _, v := range vectors {
		g.AddNode(v.id, v.vector, rng)
	}

	// Search for vector similar to n1
	query := []float32{1.0, 0.0, 0.0}
	results := g.Search(query, 3, 50)

	if len(results) == 0 {
		t.Fatal("Search returned empty results")
	}

	// First result should be n1 (exact match)
	if results[0].ID != "n1" {
		t.Errorf("First result = %s, want n1", results[0].ID)
	}

	// Distance to exact match should be ~0
	if results[0].Dist > 0.001 {
		t.Errorf("Distance to exact match = %f, want ~0", results[0].Dist)
	}
}

func TestSearchKNN(t *testing.T) {
	g := NewHNSWGraph(8)
	rng := rand.New(rand.NewSource(42))

	// Build index with 20 random vectors
	for i := 0; i < 20; i++ {
		vec := []float32{
			rng.Float32(),
			rng.Float32(),
			rng.Float32(),
		}
		// Normalize
		norm := float32(0)
		for _, v := range vec {
			norm += v * v
		}
		norm = float32(1.0 / float64(norm))
		for j := range vec {
			vec[j] *= norm
		}

		g.AddNode(string(rune('a'+i)), vec, rng)
	}

	// Search for k=5 nearest neighbors
	query := []float32{1.0, 0.0, 0.0}
	results := g.Search(query, 5, 50)

	if len(results) != 5 {
		t.Errorf("len(results) = %d, want 5", len(results))
	}

	// Results should be sorted by distance
	for i := 1; i < len(results); i++ {
		if results[i].Dist < results[i-1].Dist {
			t.Errorf("Results not sorted: [%d].Dist=%f < [%d].Dist=%f",
				i, results[i].Dist, i-1, results[i-1].Dist)
		}
	}
}

func TestRoute(t *testing.T) {
	g := NewHNSWGraph(16)
	rng := rand.New(rand.NewSource(42))

	// Build index
	g.AddNode("n1", []float32{1.0, 0.0, 0.0}, rng)
	g.AddNode("n2", []float32{0.9, 0.1, 0.0}, rng)
	g.AddNode("n3", []float32{0.0, 1.0, 0.0}, rng)

	// Route to find nearest
	query := []float32{1.0, 0.0, 0.0}
	path, count := g.Route(query, true)

	if len(path) == 0 {
		t.Error("Route returned empty path")
	}
	if path[0] != "n1" {
		t.Errorf("Route found %s, want n1", path[0])
	}
	if count == 0 {
		t.Error("Route returned 0 count")
	}
}

func TestCosineDistance(t *testing.T) {
	tests := []struct {
		name string
		a, b []float32
		want float64
	}{
		{
			name: "identical vectors",
			a:    []float32{1.0, 0.0, 0.0},
			b:    []float32{1.0, 0.0, 0.0},
			want: 0.0,
		},
		{
			name: "orthogonal vectors",
			a:    []float32{1.0, 0.0, 0.0},
			b:    []float32{0.0, 1.0, 0.0},
			want: 1.0,
		},
		{
			name: "opposite vectors",
			a:    []float32{1.0, 0.0, 0.0},
			b:    []float32{-1.0, 0.0, 0.0},
			want: 2.0,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := cosineDistance(tc.a, tc.b)
			if got < tc.want-0.001 || got > tc.want+0.001 {
				t.Errorf("cosineDistance = %f, want %f", got, tc.want)
			}
		})
	}
}

func TestTunableM(t *testing.T) {
	// Test with different M values
	for _, m := range []int{4, 8, 16, 32} {
		g := NewHNSWGraph(m)
		if g.M != m {
			t.Errorf("M = %d, want %d", g.M, m)
		}
		if g.M0 != m*2 {
			t.Errorf("M0 = %d, want %d", g.M0, m*2)
		}
	}
}
