// KG Topology Phase 4: Federated Query Engine
// Generated from Prolog service definition
//
// Implements federated search with aggregation strategies:
// - SUM: Boost consensus (exp(z_a) + exp(z_b))
// - MAX: No boost, take max score
// - DIVERSITY: Boost only if sources differ

package federation

import (
	"context"
	"math"
	"sort"
	"sync"
	"time"
)

// AggregationStrategy defines how to merge duplicate results
type AggregationStrategy int

const (
	StrategySUM AggregationStrategy = iota
	StrategyMAX
	StrategyDIVERSITY
)

// SearchResult represents a single search result
type SearchResult struct {
	ID         string
	Score      float64
	AnswerHash string // For deduplication
	SourceNode string
	Metadata   map[string]interface{}
}

// AggregatedResult represents merged results across nodes
type AggregatedResult struct {
	ID            string
	CombinedScore float64
	Sources       []string
	SourceCount   int
	Metadata      map[string]interface{}
}

// NodeClient interface for querying a node
type NodeClient interface {
	Query(ctx context.Context, embedding []float32, k int) ([]SearchResult, error)
	NodeID() string
}

// MockNodeClient for testing
type MockNodeClient struct {
	id      string
	results []SearchResult
}

func NewMockNodeClient(id string, results []SearchResult) *MockNodeClient {
	return &MockNodeClient{id: id, results: results}
}

func (c *MockNodeClient) Query(ctx context.Context, embedding []float32, k int) ([]SearchResult, error) {
	if k > len(c.results) {
		k = len(c.results)
	}
	return c.results[:k], nil
}

func (c *MockNodeClient) NodeID() string {
	return c.id
}

// FederationConfig holds federation settings
type FederationConfig struct {
	Strategy           AggregationStrategy
	FederationK        int           // Number of nodes to query
	TopK               int           // Results per node
	Timeout            time.Duration // Query timeout
	ConsensusThreshold int           // Min node agreement (0 = disabled)
}

// DefaultConfig returns default federation config
func DefaultConfig() FederationConfig {
	return FederationConfig{
		Strategy:           StrategySUM,
		FederationK:        3,
		TopK:               10,
		Timeout:            5 * time.Second,
		ConsensusThreshold: 0,
	}
}

// FederatedQueryEngine executes federated queries
type FederatedQueryEngine struct {
	nodes  []NodeClient
	config FederationConfig
	mu     sync.RWMutex
}

// NewFederatedQueryEngine creates a new federation engine
func NewFederatedQueryEngine(config FederationConfig) *FederatedQueryEngine {
	return &FederatedQueryEngine{
		nodes:  make([]NodeClient, 0),
		config: config,
	}
}

// AddNode adds a node to the federation
func (e *FederatedQueryEngine) AddNode(node NodeClient) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.nodes = append(e.nodes, node)
}

// Query executes a federated query across nodes
func (e *FederatedQueryEngine) Query(ctx context.Context, embedding []float32) ([]AggregatedResult, error) {
	e.mu.RLock()
	nodes := e.nodes
	config := e.config
	e.mu.RUnlock()

	if len(nodes) == 0 {
		return nil, nil
	}

	// Select nodes to query (up to federation_k)
	queryNodes := nodes
	if len(queryNodes) > config.FederationK {
		queryNodes = queryNodes[:config.FederationK]
	}

	// Query nodes in parallel
	type nodeResponse struct {
		nodeID  string
		results []SearchResult
		err     error
	}

	responses := make(chan nodeResponse, len(queryNodes))
	ctx, cancel := context.WithTimeout(ctx, config.Timeout)
	defer cancel()

	for _, node := range queryNodes {
		go func(n NodeClient) {
			results, err := n.Query(ctx, embedding, config.TopK)
			responses <- nodeResponse{
				nodeID:  n.NodeID(),
				results: results,
				err:     err,
			}
		}(node)
	}

	// Collect responses
	allResults := make([]SearchResult, 0)
	for i := 0; i < len(queryNodes); i++ {
		resp := <-responses
		if resp.err == nil {
			for i := range resp.results {
				resp.results[i].SourceNode = resp.nodeID
			}
			allResults = append(allResults, resp.results...)
		}
	}

	// Aggregate results
	aggregated := e.aggregate(allResults, config.Strategy)

	// Apply consensus threshold if set
	if config.ConsensusThreshold > 0 {
		filtered := make([]AggregatedResult, 0)
		for _, r := range aggregated {
			if r.SourceCount >= config.ConsensusThreshold {
				filtered = append(filtered, r)
			}
		}
		aggregated = filtered
	}

	return aggregated, nil
}

// aggregate merges results based on strategy
func (e *FederatedQueryEngine) aggregate(results []SearchResult, strategy AggregationStrategy) []AggregatedResult {
	// Group by answer hash
	groups := make(map[string]*AggregatedResult)
	sourceTracking := make(map[string]map[string]bool) // hash -> sources

	for _, r := range results {
		key := r.AnswerHash
		if key == "" {
			key = r.ID
		}

		if _, ok := sourceTracking[key]; !ok {
			sourceTracking[key] = make(map[string]bool)
		}

		if existing, ok := groups[key]; ok {
			// Merge based on strategy
			switch strategy {
			case StrategySUM:
				// Sum exp scores (consensus boost)
				existing.CombinedScore = existing.CombinedScore + r.Score
			case StrategyMAX:
				// Take max
				if r.Score > existing.CombinedScore {
					existing.CombinedScore = r.Score
				}
			case StrategyDIVERSITY:
				// Boost only if from different source
				if !sourceTracking[key][r.SourceNode] {
					existing.CombinedScore = existing.CombinedScore + r.Score
				} else {
					// Same source - take max
					if r.Score > existing.CombinedScore {
						existing.CombinedScore = r.Score
					}
				}
			}

			// Track sources
			if !sourceTracking[key][r.SourceNode] {
				sourceTracking[key][r.SourceNode] = true
				existing.Sources = append(existing.Sources, r.SourceNode)
				existing.SourceCount++
			}
		} else {
			// New result
			groups[key] = &AggregatedResult{
				ID:            r.ID,
				CombinedScore: r.Score,
				Sources:       []string{r.SourceNode},
				SourceCount:   1,
				Metadata:      r.Metadata,
			}
			sourceTracking[key][r.SourceNode] = true
		}
	}

	// Convert to slice and sort by score
	aggregated := make([]AggregatedResult, 0, len(groups))
	for _, r := range groups {
		aggregated = append(aggregated, *r)
	}

	sort.Slice(aggregated, func(i, j int) bool {
		return aggregated[i].CombinedScore > aggregated[j].CombinedScore
	})

	return aggregated
}

// SetStrategy updates the aggregation strategy
func (e *FederatedQueryEngine) SetStrategy(strategy AggregationStrategy) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.config.Strategy = strategy
}

// SetFederationK updates the number of nodes to query
func (e *FederatedQueryEngine) SetFederationK(k int) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.config.FederationK = k
}

// expScore converts raw score to exp score for aggregation
func expScore(score float64) float64 {
	return math.Exp(score)
}
