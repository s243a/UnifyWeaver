// KG Topology Phase 7: HNSW Hierarchical Navigable Small World
// Generated from Prolog service definition
//
// Implements HNSW for O(log n) approximate nearest neighbor search.
// See: Malkov & Yashunin (2018) "Efficient and robust approximate nearest neighbor search"

package hnsw

import (
	"math"
	"math/rand"
	"sort"
	"sync"
)

// Configuration defaults
const (
	DefaultM        = 16   // Max neighbors per layer
	DefaultM0       = 32   // Max neighbors at layer 0
	DefaultEfSearch = 50   // Search beam width
	DefaultML       = 1.0  // Level multiplier (1/ln(2) ≈ 1.44 in original)
)

// HNSWNode represents a node in the HNSW graph
type HNSWNode struct {
	ID        string
	Vector    []float32
	MaxLayer  int                      // Node exists on layers 0..MaxLayer
	Neighbors map[int]map[string]bool  // layer -> neighbor IDs
	mu        sync.RWMutex
}

// NewHNSWNode creates a new HNSW node
func NewHNSWNode(id string, vector []float32, maxLayer int) *HNSWNode {
	return &HNSWNode{
		ID:        id,
		Vector:    vector,
		MaxLayer:  maxLayer,
		Neighbors: make(map[int]map[string]bool),
	}
}

// GetNeighborsAtLayer returns neighbors at a specific layer
func (n *HNSWNode) GetNeighborsAtLayer(layer int) []string {
	n.mu.RLock()
	defer n.mu.RUnlock()

	neighbors, ok := n.Neighbors[layer]
	if !ok {
		return nil
	}

	result := make([]string, 0, len(neighbors))
	for id := range neighbors {
		result = append(result, id)
	}
	return result
}

// AddNeighbor adds a neighbor at a specific layer
func (n *HNSWNode) AddNeighbor(neighborID string, layer, maxNeighbors int) bool {
	n.mu.Lock()
	defer n.mu.Unlock()

	if neighborID == n.ID {
		return false
	}

	if n.Neighbors[layer] == nil {
		n.Neighbors[layer] = make(map[string]bool)
	}

	if len(n.Neighbors[layer]) >= maxNeighbors {
		return false
	}

	n.Neighbors[layer][neighborID] = true
	return true
}

// HNSWGraph represents the HNSW index
type HNSWGraph struct {
	Nodes          map[string]*HNSWNode
	EntryPointID   string
	MaxLayer       int
	M              int     // Max neighbors per layer
	M0             int     // Max neighbors at layer 0
	ML             float64 // Level multiplier
	EfConstruction int     // Beam width during construction
	mu             sync.RWMutex
}

// NewHNSWGraph creates a new HNSW graph with tunable M parameter
func NewHNSWGraph(m int) *HNSWGraph {
	if m <= 0 {
		m = DefaultM
	}
	return &HNSWGraph{
		Nodes:          make(map[string]*HNSWNode),
		M:              m,
		M0:             m * 2,
		ML:             DefaultML,
		EfConstruction: DefaultEfSearch,
	}
}

// randomLayer assigns a random layer using exponential distribution
func (g *HNSWGraph) randomLayer(rng *rand.Rand) int {
	r := rng.Float64()
	if r == 0 {
		r = 0.0001
	}
	return int(-math.Log(r) * g.ML)
}

// AddNode adds a node to the HNSW graph
func (g *HNSWGraph) AddNode(nodeID string, vector []float32, rng *rand.Rand) *HNSWNode {
	g.mu.Lock()
	defer g.mu.Unlock()

	if rng == nil {
		rng = rand.New(rand.NewSource(42))
	}

	// Assign random layer
	nodeLayer := g.randomLayer(rng)

	node := NewHNSWNode(nodeID, vector, nodeLayer)
	g.Nodes[nodeID] = node

	// First node becomes entry point
	if g.EntryPointID == "" {
		g.EntryPointID = nodeID
		g.MaxLayer = nodeLayer
		return node
	}

	// Greedy descent from top to node's layer + 1
	currentID := g.EntryPointID
	for layer := g.MaxLayer; layer > nodeLayer; layer-- {
		closest := g.greedySearchLayer(vector, currentID, layer, 1)
		if len(closest) > 0 {
			currentID = closest[0].ID
		}
	}

	// Connect at each layer from nodeLayer down to 0
	for layer := min(nodeLayer, g.MaxLayer); layer >= 0; layer-- {
		// Find neighbors at this layer
		candidates := g.searchLayer(vector, currentID, layer, g.EfConstruction)

		// Select and connect neighbors
		maxN := g.M
		if layer == 0 {
			maxN = g.M0
		}

		connected := 0
		for _, cand := range candidates {
			if connected >= maxN {
				break
			}
			if cand.ID == nodeID {
				continue
			}

			// Bidirectional connection
			node.AddNeighbor(cand.ID, layer, maxN)
			neighbor := g.Nodes[cand.ID]
			if neighbor != nil {
				neighbor.AddNeighbor(nodeID, layer, maxN)
			}
			connected++
		}

		// Use closest as entry for next layer
		if len(candidates) > 0 {
			currentID = candidates[0].ID
		}
	}

	// Update entry point if new node is at higher layer
	if nodeLayer > g.MaxLayer {
		g.EntryPointID = nodeID
		g.MaxLayer = nodeLayer
	}

	return node
}

// SearchResult holds search results
type SearchResult struct {
	ID   string
	Dist float64
}

// greedySearchLayer performs greedy search at a single layer
func (g *HNSWGraph) greedySearchLayer(query []float32, entryID string, layer, k int) []SearchResult {
	visited := make(map[string]bool)
	visited[entryID] = true

	entryNode := g.Nodes[entryID]
	if entryNode == nil {
		return nil
	}

	candidates := []SearchResult{{ID: entryID, Dist: cosineDistance(query, entryNode.Vector)}}

	for {
		// Sort by distance
		sort.Slice(candidates, func(i, j int) bool {
			return candidates[i].Dist < candidates[j].Dist
		})

		currentID := candidates[0].ID
		currentDist := candidates[0].Dist
		currentNode := g.Nodes[currentID]

		improved := false
		for _, neighborID := range currentNode.GetNeighborsAtLayer(layer) {
			if visited[neighborID] {
				continue
			}
			visited[neighborID] = true

			neighbor := g.Nodes[neighborID]
			if neighbor == nil {
				continue
			}

			dist := cosineDistance(query, neighbor.Vector)
			if dist < currentDist {
				candidates = append(candidates, SearchResult{ID: neighborID, Dist: dist})
				improved = true
			}
		}

		if !improved {
			break
		}
	}

	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].Dist < candidates[j].Dist
	})

	if len(candidates) > k {
		candidates = candidates[:k]
	}
	return candidates
}

// searchLayer performs beam search at a single layer
func (g *HNSWGraph) searchLayer(query []float32, entryID string, layer, ef int) []SearchResult {
	visited := make(map[string]bool)
	visited[entryID] = true

	entryNode := g.Nodes[entryID]
	if entryNode == nil {
		return nil
	}

	entryDist := cosineDistance(query, entryNode.Vector)
	candidates := []SearchResult{{ID: entryID, Dist: entryDist}}
	results := []SearchResult{{ID: entryID, Dist: entryDist}}

	for len(candidates) > 0 {
		// Pop closest candidate
		sort.Slice(candidates, func(i, j int) bool {
			return candidates[i].Dist < candidates[j].Dist
		})
		current := candidates[0]
		candidates = candidates[1:]

		// Check if we can improve
		sort.Slice(results, func(i, j int) bool {
			return results[i].Dist < results[j].Dist
		})
		if len(results) > 0 && current.Dist > results[len(results)-1].Dist {
			break
		}

		// Explore neighbors
		currentNode := g.Nodes[current.ID]
		for _, neighborID := range currentNode.GetNeighborsAtLayer(layer) {
			if visited[neighborID] {
				continue
			}
			visited[neighborID] = true

			neighbor := g.Nodes[neighborID]
			if neighbor == nil {
				continue
			}

			dist := cosineDistance(query, neighbor.Vector)

			// Add to results if better than worst
			if len(results) < ef || dist < results[len(results)-1].Dist {
				results = append(results, SearchResult{ID: neighborID, Dist: dist})
				candidates = append(candidates, SearchResult{ID: neighborID, Dist: dist})

				// Keep only ef best results
				if len(results) > ef {
					sort.Slice(results, func(i, j int) bool {
						return results[i].Dist < results[j].Dist
					})
					results = results[:ef]
				}
			}
		}
	}

	sort.Slice(results, func(i, j int) bool {
		return results[i].Dist < results[j].Dist
	})
	return results
}

// Search performs k-NN search using HNSW
func (g *HNSWGraph) Search(query []float32, k, efSearch int) []SearchResult {
	g.mu.RLock()
	defer g.mu.RUnlock()

	if len(g.Nodes) == 0 || g.EntryPointID == "" {
		return nil
	}

	if efSearch <= 0 {
		efSearch = DefaultEfSearch
	}

	// Greedy descent from top layer
	currentID := g.EntryPointID
	for layer := g.MaxLayer; layer > 0; layer-- {
		closest := g.greedySearchLayer(query, currentID, layer, 1)
		if len(closest) > 0 {
			currentID = closest[0].ID
		}
	}

	// Beam search at layer 0
	results := g.searchLayer(query, currentID, 0, efSearch)

	if len(results) > k {
		results = results[:k]
	}
	return results
}

// Route performs routing (alias for Search with k=1)
func (g *HNSWGraph) Route(query []float32, useBacktrack bool) ([]string, int) {
	results := g.Search(query, 1, DefaultEfSearch)
	if len(results) == 0 {
		return nil, 0
	}
	return []string{results[0].ID}, 1
}

// cosineDistance computes cosine distance (1 - similarity)
func cosineDistance(a, b []float32) float64 {
	var dot, normA, normB float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		normA += float64(a[i]) * float64(a[i])
		normB += float64(b[i]) * float64(b[i])
	}
	if normA == 0 || normB == 0 {
		return 1.0
	}
	sim := dot / (math.Sqrt(normA) * math.Sqrt(normB))
	return 1.0 - sim
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
