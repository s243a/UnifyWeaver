package smallworld

import (
	"testing"
)

func TestNewSmallWorldNetwork(t *testing.T) {
	n := NewSmallWorldNetwork()
	if n == nil {
		t.Fatal("NewSmallWorldNetwork returned nil")
	}
	if n.KLocal != KLocal {
		t.Errorf("KLocal = %d, want %d", n.KLocal, KLocal)
	}
	if n.KLong != KLong {
		t.Errorf("KLong = %d, want %d", n.KLong, KLong)
	}
	if n.Alpha != Alpha {
		t.Errorf("Alpha = %f, want %f", n.Alpha, Alpha)
	}
}

func TestAddNode(t *testing.T) {
	n := NewSmallWorldNetwork()

	// Add first node
	n.AddNode("node1", []float32{1.0, 0.0, 0.0})
	if len(n.Nodes) != 1 {
		t.Errorf("len(Nodes) = %d, want 1", len(n.Nodes))
	}

	// Add second node - should get connections
	n.AddNode("node2", []float32{0.9, 0.1, 0.0})
	if len(n.Nodes) != 2 {
		t.Errorf("len(Nodes) = %d, want 2", len(n.Nodes))
	}

	// Check connections were created
	node2 := n.Nodes["node2"]
	if len(node2.Neighbors) == 0 {
		t.Error("node2 should have at least one neighbor")
	}
}

func TestCosineSimilarity(t *testing.T) {
	tests := []struct {
		name string
		a, b []float32
		want float64
	}{
		{
			name: "identical vectors",
			a:    []float32{1.0, 0.0, 0.0},
			b:    []float32{1.0, 0.0, 0.0},
			want: 1.0,
		},
		{
			name: "orthogonal vectors",
			a:    []float32{1.0, 0.0, 0.0},
			b:    []float32{0.0, 1.0, 0.0},
			want: 0.0,
		},
		{
			name: "opposite vectors",
			a:    []float32{1.0, 0.0, 0.0},
			b:    []float32{-1.0, 0.0, 0.0},
			want: -1.0,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := cosineSimilarity(tc.a, tc.b)
			if got < tc.want-0.001 || got > tc.want+0.001 {
				t.Errorf("cosineSimilarity = %f, want %f", got, tc.want)
			}
		})
	}
}

func TestRouteToTarget(t *testing.T) {
	n := NewSmallWorldNetwork()

	// Build a small network
	vectors := []struct {
		id     string
		vector []float32
	}{
		{"n1", []float32{1.0, 0.0, 0.0}},
		{"n2", []float32{0.9, 0.1, 0.0}},
		{"n3", []float32{0.8, 0.2, 0.0}},
		{"n4", []float32{0.0, 1.0, 0.0}},
		{"n5", []float32{0.1, 0.9, 0.0}},
	}

	for _, v := range vectors {
		n.AddNode(v.id, v.vector)
	}

	// Route towards a target
	target := []float32{1.0, 0.0, 0.0}
	path := n.RouteToTarget(target, 10)

	if len(path) == 0 {
		t.Error("RouteToTarget returned empty path")
	}

	// Final node should be close to target
	finalNode := n.Nodes[path[len(path)-1]]
	finalSim := cosineSimilarity(finalNode.Centroid, target)
	if finalSim < 0.8 {
		t.Errorf("Final node similarity = %f, want >= 0.8", finalSim)
	}
}

func TestComputeCosineAngle(t *testing.T) {
	tests := []struct {
		name string
		a, b []float32
		want float64 // approximate
	}{
		{
			name: "identical vectors (0 angle)",
			a:    []float32{1.0, 0.0, 0.0},
			b:    []float32{1.0, 0.0, 0.0},
			want: 0.0,
		},
		{
			name: "orthogonal vectors (pi/2)",
			a:    []float32{1.0, 0.0, 0.0},
			b:    []float32{0.0, 1.0, 0.0},
			want: 1.5708, // pi/2
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := computeCosineAngle(tc.a, tc.b)
			if got < tc.want-0.01 || got > tc.want+0.01 {
				t.Errorf("computeCosineAngle = %f, want %f", got, tc.want)
			}
		})
	}
}
