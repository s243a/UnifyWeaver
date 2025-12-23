package federation

import (
	"context"
	"math"
	"testing"
	"time"
)

func TestNewFederatedQueryEngine(t *testing.T) {
	config := DefaultConfig()
	engine := NewFederatedQueryEngine(config)

	if engine == nil {
		t.Fatal("NewFederatedQueryEngine returned nil")
	}
	if engine.config.Strategy != StrategySUM {
		t.Errorf("Strategy = %d, want StrategySUM", engine.config.Strategy)
	}
	if engine.config.FederationK != 3 {
		t.Errorf("FederationK = %d, want 3", engine.config.FederationK)
	}
}

func TestAddNode(t *testing.T) {
	engine := NewFederatedQueryEngine(DefaultConfig())

	node1 := NewMockNodeClient("node1", []SearchResult{
		{ID: "doc1", Score: 0.9, AnswerHash: "hash1", SourceNode: "node1"},
	})
	node2 := NewMockNodeClient("node2", []SearchResult{
		{ID: "doc2", Score: 0.8, AnswerHash: "hash2", SourceNode: "node2"},
	})

	engine.AddNode(node1)
	engine.AddNode(node2)

	if len(engine.nodes) != 2 {
		t.Errorf("len(nodes) = %d, want 2", len(engine.nodes))
	}
}

func TestQueryEmptyEngine(t *testing.T) {
	engine := NewFederatedQueryEngine(DefaultConfig())

	results, err := engine.Query(context.Background(), []float32{1.0, 0.0, 0.0})

	if err != nil {
		t.Errorf("Query returned error: %v", err)
	}
	if results != nil && len(results) != 0 {
		t.Errorf("Expected empty results for empty engine")
	}
}

func TestQuerySingleNode(t *testing.T) {
	engine := NewFederatedQueryEngine(DefaultConfig())

	node := NewMockNodeClient("node1", []SearchResult{
		{ID: "doc1", Score: 0.9, AnswerHash: "hash1"},
		{ID: "doc2", Score: 0.8, AnswerHash: "hash2"},
		{ID: "doc3", Score: 0.7, AnswerHash: "hash3"},
	})
	engine.AddNode(node)

	results, err := engine.Query(context.Background(), []float32{1.0, 0.0, 0.0})

	if err != nil {
		t.Fatalf("Query returned error: %v", err)
	}
	if len(results) != 3 {
		t.Errorf("len(results) = %d, want 3", len(results))
	}

	// Results should be sorted by score descending
	if results[0].ID != "doc1" {
		t.Errorf("First result = %s, want doc1", results[0].ID)
	}
}

func TestStrategySUM(t *testing.T) {
	config := DefaultConfig()
	config.Strategy = StrategySUM
	engine := NewFederatedQueryEngine(config)

	// Two nodes return same document with different scores
	node1 := NewMockNodeClient("node1", []SearchResult{
		{ID: "doc1", Score: 0.9, AnswerHash: "hash1"},
	})
	node2 := NewMockNodeClient("node2", []SearchResult{
		{ID: "doc1b", Score: 0.8, AnswerHash: "hash1"}, // Same hash = same answer
	})
	engine.AddNode(node1)
	engine.AddNode(node2)

	results, err := engine.Query(context.Background(), []float32{1.0, 0.0, 0.0})

	if err != nil {
		t.Fatalf("Query returned error: %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("len(results) = %d, want 1 (merged)", len(results))
	}

	// SUM strategy: 0.9 + 0.8 = 1.7
	expectedScore := 0.9 + 0.8
	if math.Abs(results[0].CombinedScore-expectedScore) > 0.001 {
		t.Errorf("CombinedScore = %f, want %f", results[0].CombinedScore, expectedScore)
	}
	if results[0].SourceCount != 2 {
		t.Errorf("SourceCount = %d, want 2", results[0].SourceCount)
	}
}

func TestStrategyMAX(t *testing.T) {
	config := DefaultConfig()
	config.Strategy = StrategyMAX
	engine := NewFederatedQueryEngine(config)

	node1 := NewMockNodeClient("node1", []SearchResult{
		{ID: "doc1", Score: 0.9, AnswerHash: "hash1"},
	})
	node2 := NewMockNodeClient("node2", []SearchResult{
		{ID: "doc1b", Score: 0.8, AnswerHash: "hash1"},
	})
	engine.AddNode(node1)
	engine.AddNode(node2)

	results, err := engine.Query(context.Background(), []float32{1.0, 0.0, 0.0})

	if err != nil {
		t.Fatalf("Query returned error: %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("len(results) = %d, want 1", len(results))
	}

	// MAX strategy: max(0.9, 0.8) = 0.9
	if math.Abs(results[0].CombinedScore-0.9) > 0.001 {
		t.Errorf("CombinedScore = %f, want 0.9", results[0].CombinedScore)
	}
}

func TestStrategyDIVERSITY(t *testing.T) {
	config := DefaultConfig()
	config.Strategy = StrategyDIVERSITY
	engine := NewFederatedQueryEngine(config)

	// Node1 returns same doc twice, Node2 returns it once
	node1 := NewMockNodeClient("node1", []SearchResult{
		{ID: "doc1", Score: 0.9, AnswerHash: "hash1"},
		{ID: "doc1dup", Score: 0.85, AnswerHash: "hash1"}, // Same hash, same source
	})
	node2 := NewMockNodeClient("node2", []SearchResult{
		{ID: "doc1alt", Score: 0.8, AnswerHash: "hash1"}, // Same hash, different source
	})
	engine.AddNode(node1)
	engine.AddNode(node2)

	results, err := engine.Query(context.Background(), []float32{1.0, 0.0, 0.0})

	if err != nil {
		t.Fatalf("Query returned error: %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("len(results) = %d, want 1", len(results))
	}

	// DIVERSITY: First from node1 = 0.9, second from node1 (same source) takes max = 0.9,
	// third from node2 (different source) adds = 0.9 + 0.8 = 1.7
	expectedScore := 0.9 + 0.8
	if math.Abs(results[0].CombinedScore-expectedScore) > 0.001 {
		t.Errorf("CombinedScore = %f, want %f", results[0].CombinedScore, expectedScore)
	}
	if results[0].SourceCount != 2 {
		t.Errorf("SourceCount = %d, want 2", results[0].SourceCount)
	}
}

func TestConsensusThreshold(t *testing.T) {
	config := DefaultConfig()
	config.ConsensusThreshold = 2 // Require at least 2 sources
	engine := NewFederatedQueryEngine(config)

	node1 := NewMockNodeClient("node1", []SearchResult{
		{ID: "doc1", Score: 0.9, AnswerHash: "hash1"},
		{ID: "doc2", Score: 0.7, AnswerHash: "hash2"}, // Only from node1
	})
	node2 := NewMockNodeClient("node2", []SearchResult{
		{ID: "doc1b", Score: 0.8, AnswerHash: "hash1"}, // Same as doc1
		{ID: "doc3", Score: 0.6, AnswerHash: "hash3"},  // Only from node2
	})
	engine.AddNode(node1)
	engine.AddNode(node2)

	results, err := engine.Query(context.Background(), []float32{1.0, 0.0, 0.0})

	if err != nil {
		t.Fatalf("Query returned error: %v", err)
	}

	// Only doc1/hash1 appears in both sources
	if len(results) != 1 {
		t.Errorf("len(results) = %d, want 1 (only consensus)", len(results))
	}
	if len(results) > 0 && results[0].SourceCount < 2 {
		t.Errorf("SourceCount = %d, want >= 2", results[0].SourceCount)
	}
}

func TestFederationK(t *testing.T) {
	config := DefaultConfig()
	config.FederationK = 2 // Only query 2 nodes
	engine := NewFederatedQueryEngine(config)

	// Add 4 nodes but only first 2 should be queried
	for i := 0; i < 4; i++ {
		id := string(rune('a' + i))
		node := NewMockNodeClient("node"+id, []SearchResult{
			{ID: "doc" + id, Score: 0.5, AnswerHash: "hash" + id},
		})
		engine.AddNode(node)
	}

	results, err := engine.Query(context.Background(), []float32{1.0, 0.0, 0.0})

	if err != nil {
		t.Fatalf("Query returned error: %v", err)
	}

	// Should only have results from first 2 nodes
	if len(results) > 2 {
		t.Errorf("len(results) = %d, want <= 2", len(results))
	}
}

func TestSetStrategy(t *testing.T) {
	engine := NewFederatedQueryEngine(DefaultConfig())

	engine.SetStrategy(StrategyMAX)
	if engine.config.Strategy != StrategyMAX {
		t.Errorf("Strategy = %d, want StrategyMAX", engine.config.Strategy)
	}

	engine.SetStrategy(StrategyDIVERSITY)
	if engine.config.Strategy != StrategyDIVERSITY {
		t.Errorf("Strategy = %d, want StrategyDIVERSITY", engine.config.Strategy)
	}
}

func TestSetFederationK(t *testing.T) {
	engine := NewFederatedQueryEngine(DefaultConfig())

	engine.SetFederationK(5)
	if engine.config.FederationK != 5 {
		t.Errorf("FederationK = %d, want 5", engine.config.FederationK)
	}
}

func TestTimeout(t *testing.T) {
	config := DefaultConfig()
	config.Timeout = 100 * time.Millisecond
	engine := NewFederatedQueryEngine(config)

	// Mock node that would timeout (but we just verify config is set)
	node := NewMockNodeClient("node1", []SearchResult{
		{ID: "doc1", Score: 0.9, AnswerHash: "hash1"},
	})
	engine.AddNode(node)

	// Query should complete quickly with mock
	ctx := context.Background()
	_, err := engine.Query(ctx, []float32{1.0, 0.0, 0.0})

	if err != nil {
		t.Errorf("Query returned error: %v", err)
	}
}

func TestResultSorting(t *testing.T) {
	engine := NewFederatedQueryEngine(DefaultConfig())

	node := NewMockNodeClient("node1", []SearchResult{
		{ID: "doc3", Score: 0.3, AnswerHash: "hash3"},
		{ID: "doc1", Score: 0.9, AnswerHash: "hash1"},
		{ID: "doc2", Score: 0.6, AnswerHash: "hash2"},
	})
	engine.AddNode(node)

	results, _ := engine.Query(context.Background(), []float32{1.0, 0.0, 0.0})

	// Should be sorted descending by score
	for i := 1; i < len(results); i++ {
		if results[i].CombinedScore > results[i-1].CombinedScore {
			t.Errorf("Results not sorted: [%d]=%f > [%d]=%f",
				i, results[i].CombinedScore, i-1, results[i-1].CombinedScore)
		}
	}
}
