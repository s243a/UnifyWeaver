% Generate Go code for Phase 7-8 features
% Usage: swipl -g 'main' -t halt generate_go_code.pl

%% compile_small_world_proper_go(+Options, -Code)
compile_small_world_proper_go(Options, Code) :-
    ( member(k_local(KLocal), Options) -> true ; KLocal = 10 ),
    ( member(k_long(KLong), Options) -> true ; KLong = 5 ),
    ( member(alpha(Alpha), Options) -> true ; Alpha = 2.0 ),

    format(atom(Code), '// KG Topology Phase 7: Proper Small-World Network
// Generated from Prolog service definition
//
// Network structure enables true Kleinberg routing with O(log^2 n) path length.
// k_local = ~w nearest neighbors, k_long = ~w long-range shortcuts

package smallworld

import (
	"math"
	"math/rand"
	"sort"
	"sync"
)

// Configuration constants
const (
	KLocal = ~w
	KLong  = ~w
	Alpha  = ~w
)

// SmallWorldNode represents a node in the small-world network
type SmallWorldNode struct {
	ID        string
	Centroid  []float32
	Neighbors []*Neighbor // Sorted by angle for binary search
	mu        sync.RWMutex
}

// Neighbor represents a connection with precomputed angle
type Neighbor struct {
	NodeID string
	Angle  float64 // Cosine-based angle for binary search
	IsLong bool    // true = long-range, false = local
}

// SmallWorldNetwork represents the complete network
type SmallWorldNetwork struct {
	Nodes  map[string]*SmallWorldNode
	KLocal int
	KLong  int
	Alpha  float64
	mu     sync.RWMutex
}

// NewSmallWorldNetwork creates a new properly-structured network
func NewSmallWorldNetwork() *SmallWorldNetwork {
	return &SmallWorldNetwork{
		Nodes:  make(map[string]*SmallWorldNode),
		KLocal: KLocal,
		KLong:  KLong,
		Alpha:  Alpha,
	}
}

// AddNode adds a node and establishes connections
func (n *SmallWorldNetwork) AddNode(nodeID string, centroid []float32) {
	n.mu.Lock()
	defer n.mu.Unlock()

	node := &SmallWorldNode{
		ID:       nodeID,
		Centroid: centroid,
	}
	n.Nodes[nodeID] = node

	if len(n.Nodes) > 1 {
		n.connectNode(node)
	}
}

// connectNode establishes k_local + k_long connections
func (n *SmallWorldNetwork) connectNode(node *SmallWorldNode) {
	// Collect all other nodes with their similarities
	type nodeSim struct {
		id  string
		sim float64
	}
	var others []nodeSim
	for id, other := range n.Nodes {
		if id != node.ID {
			sim := cosineSimilarity(node.Centroid, other.Centroid)
			others = append(others, nodeSim{id, sim})
		}
	}

	// Sort by similarity (descending) for k_local selection
	sort.Slice(others, func(i, j int) bool {
		return others[i].sim > others[j].sim
	})

	// Add k_local nearest neighbors
	localCount := minInt(n.KLocal, len(others))
	for i := 0; i < localCount; i++ {
		angle := computeCosineAngle(node.Centroid, n.Nodes[others[i].id].Centroid)
		node.Neighbors = append(node.Neighbors, &Neighbor{
			NodeID: others[i].id,
			Angle:  angle,
			IsLong: false,
		})
	}

	// Add k_long shortcuts using distance-weighted probability
	if len(others) > n.KLocal {
		remaining := others[localCount:]
		longCount := minInt(n.KLong, len(remaining))

		// Compute weights: P(v) ~ 1/distance^alpha
		weights := make([]float64, len(remaining))
		var totalWeight float64
		for i, ns := range remaining {
			distance := 1.0 - ns.sim // Convert similarity to distance
			if distance < 0.001 {
				distance = 0.001
			}
			weights[i] = 1.0 / math.Pow(distance, n.Alpha)
			totalWeight += weights[i]
		}

		// Sample k_long shortcuts
		selected := make(map[int]bool)
		for len(selected) < longCount {
			r := rand.Float64() * totalWeight
			cumulative := 0.0
			for i, w := range weights {
				cumulative += w
				if r <= cumulative && !selected[i] {
					selected[i] = true
					angle := computeCosineAngle(node.Centroid, n.Nodes[remaining[i].id].Centroid)
					node.Neighbors = append(node.Neighbors, &Neighbor{
						NodeID: remaining[i].id,
						Angle:  angle,
						IsLong: true,
					})
					break
				}
			}
		}
	}

	// Sort neighbors by angle for binary search
	sort.Slice(node.Neighbors, func(i, j int) bool {
		return node.Neighbors[i].Angle < node.Neighbors[j].Angle
	})
}

// RouteToTarget performs greedy routing on the small-world network
func (n *SmallWorldNetwork) RouteToTarget(queryEmbedding []float32, maxHops int) []string {
	n.mu.RLock()
	defer n.mu.RUnlock()

	if len(n.Nodes) == 0 {
		return nil
	}

	// Start from random node
	var current *SmallWorldNode
	for _, node := range n.Nodes {
		current = node
		break
	}

	path := []string{current.ID}
	visited := map[string]bool{current.ID: true}

	for hop := 0; hop < maxHops; hop++ {
		// Find best neighbor using binary search
		bestNeighbor := n.findBestNeighbor(current, queryEmbedding, visited)
		if bestNeighbor == "" {
			break // No unvisited neighbors
		}

		// Check if we should stop (neighbor is worse than current)
		currentSim := cosineSimilarity(current.Centroid, queryEmbedding)
		neighborNode := n.Nodes[bestNeighbor]
		neighborSim := cosineSimilarity(neighborNode.Centroid, queryEmbedding)

		if neighborSim <= currentSim {
			break // Reached local optimum
		}

		visited[bestNeighbor] = true
		path = append(path, bestNeighbor)
		current = neighborNode
	}

	return path
}

// findBestNeighbor uses binary search on angle-sorted neighbors
func (n *SmallWorldNetwork) findBestNeighbor(node *SmallWorldNode, query []float32, visited map[string]bool) string {
	if len(node.Neighbors) == 0 {
		return ""
	}

	// Compute query angle relative to node centroid
	queryAngle := computeCosineAngle(node.Centroid, query)

	// Binary search for closest angle
	idx := sort.Search(len(node.Neighbors), func(i int) bool {
		return node.Neighbors[i].Angle >= queryAngle
	})

	// Check neighbors around the found index
	var bestID string
	var bestSim float64 = -1

	for _, checkIdx := range []int{idx - 1, idx, idx + 1} {
		if checkIdx >= 0 && checkIdx < len(node.Neighbors) {
			nb := node.Neighbors[checkIdx]
			if !visited[nb.NodeID] {
				neighbor := n.Nodes[nb.NodeID]
				sim := cosineSimilarity(neighbor.Centroid, query)
				if sim > bestSim {
					bestSim = sim
					bestID = nb.NodeID
				}
			}
		}
	}

	return bestID
}

// computeCosineAngle computes angle using full cosine similarity (not 2D projection)
func computeCosineAngle(a, b []float32) float64 {
	sim := cosineSimilarity(a, b)
	// Clamp for numerical stability
	if sim > 1.0 {
		sim = 1.0
	}
	if sim < -1.0 {
		sim = -1.0
	}
	return math.Acos(float64(sim))
}

func cosineSimilarity(a, b []float32) float64 {
	var dot, normA, normB float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		normA += float64(a[i]) * float64(a[i])
		normB += float64(b[i]) * float64(b[i])
	}
	if normA == 0 || normB == 0 {
		return 0
	}
	return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}
', [KLocal, KLong, KLocal, KLong, Alpha]).


main :-
    format('Generating Go small-world code...~n'),
    Options = [k_local(10), k_long(5), alpha(2.0)],
    compile_small_world_proper_go(Options, Code),
    writeln(Code),
    halt(0).
