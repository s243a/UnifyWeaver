package search

import (
	"math"
	"sort"

	"unifyweaver/targets/go_runtime/projection"
)

// Interface for storage to avoid circular dependency if needed,
// but here we just use a callback interface
type VectorIterator interface {
	IterateEmbeddings(func(id string, vector []float32) error) error
}

type Result struct {
	ID             string
	Score          float32
	RoutingWeights map[int]float32 // Multi-head routing weights (optional)
}

// SearchOptions configures search behavior.
type SearchOptions struct {
	// Projection for multi-head LDA projection (optional)
	Projection *projection.MultiHeadProjection

	// UseProjection enables projection if Projection is set
	UseProjection bool

	// IncludeRoutingWeights adds routing weights to results
	IncludeRoutingWeights bool
}

// Search performs basic cosine similarity search without projection.
func Search(store VectorIterator, queryVec []float32, topK int) ([]Result, error) {
	return SearchWithOptions(store, queryVec, topK, SearchOptions{})
}

// SearchWithOptions performs search with optional multi-head LDA projection.
// When projection is enabled, the query is projected through the multi-head
// model before computing similarities.
func SearchWithOptions(store VectorIterator, queryVec []float32, topK int, opts SearchOptions) ([]Result, error) {
	var results []Result
	var routingWeights map[int]float32

	// Apply projection if configured
	searchVec := queryVec
	if opts.UseProjection && opts.Projection != nil {
		var err error
		if opts.IncludeRoutingWeights {
			searchVec, routingWeights, err = opts.Projection.ProjectWithWeights(queryVec)
		} else {
			searchVec, err = opts.Projection.Project(queryVec)
		}
		if err != nil {
			return nil, err
		}
	}

	err := store.IterateEmbeddings(func(id string, vec []float32) error {
		score := CosineSimilarity(searchVec, vec)
		result := Result{ID: id, Score: score}
		if opts.IncludeRoutingWeights && routingWeights != nil {
			result.RoutingWeights = routingWeights
		}
		results = append(results, result)
		return nil
	})

	if err != nil {
		return nil, err
	}

	// Sort descending by score
	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})

	if len(results) > topK {
		return results[:topK], nil
	}
	return results, nil
}

func CosineSimilarity(a, b []float32) float32 {
	if len(a) != len(b) {
		return 0.0
	}

	var dot, normA, normB float32
	for i := 0; i < len(a); i++ {
		dot += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}

	if normA == 0 || normB == 0 {
		return 0.0
	}

	return dot / (float32(math.Sqrt(float64(normA))) * float32(math.Sqrt(float64(normB))))
}
