package search

import (
	"fmt"
	"strings"

	"unifyweaver/targets/go_runtime/utils"
)

// ObjectStore interface for retrieving objects
type ObjectStore interface {
	GetObject(id string) (map[string]interface{}, error)
}

// Embedder interface for generating embeddings
type Embedder interface {
	Embed(text string) ([]float32, error)
}

// BookmarkFiler provides bookmark filing suggestions
type BookmarkFiler struct {
	store    VectorIterator
	objStore ObjectStore
	embedder Embedder
}

// NewBookmarkFiler creates a new bookmark filer
func NewBookmarkFiler(store VectorIterator, objStore ObjectStore, embedder Embedder) *BookmarkFiler {
	return &BookmarkFiler{
		store:    store,
		objStore: objStore,
		embedder: embedder,
	}
}

// Candidate represents a bookmark filing candidate
type Candidate struct {
	ID    string
	Score float32
	Title string
	Data  map[string]interface{}
}

// SuggestBookmarks returns bookmark filing suggestions for a query
func (bf *BookmarkFiler) SuggestBookmarks(query string, topK int) (string, error) {
	// Generate query embedding
	queryVec, err := bf.embedder.Embed(query)
	if err != nil {
		return "", fmt.Errorf("embed query: %w", err)
	}

	// Search for candidates
	results, err := Search(bf.store, queryVec, topK)
	if err != nil {
		return "", fmt.Errorf("search: %w", err)
	}

	if len(results) == 0 {
		return "No suitable placement locations found.", nil
	}

	// Build output
	var sb strings.Builder
	sb.WriteString("=== Bookmark Filing Suggestions ===\n")
	sb.WriteString(fmt.Sprintf("Bookmark: \"%s\"\n", query))
	sb.WriteString(fmt.Sprintf("Found %d candidate location(s):\n\n", len(results)))
	sb.WriteString(strings.Repeat("=", 80) + "\n\n")

	for i, result := range results {
		sb.WriteString(fmt.Sprintf("Option %d:\n\n", i+1))
		sb.WriteString(bf.buildTreeContext(result.ID, result.Score))
		sb.WriteString("\n")
		if i < len(results)-1 {
			sb.WriteString(strings.Repeat("-", 80) + "\n\n")
		}
	}

	return sb.String(), nil
}

// buildTreeContext builds a tree context display for a candidate
func (bf *BookmarkFiler) buildTreeContext(candidateID string, score float32) string {
	data, err := bf.objStore.GetObject(candidateID)
	if err != nil {
		return fmt.Sprintf("Entity %s not found", candidateID)
	}

	title := bf.getTitle(data, candidateID)

	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("Candidate: \"%s\" (similarity: %.3f)\n\n", title, score))

	// Truncate title for display
	displayTitle := title
	if len(displayTitle) > 50 {
		displayTitle = displayTitle[:47] + "..."
	}
	sb.WriteString(fmt.Sprintf("    ├── %s/        ← CANDIDATE (place new bookmark here)\n", displayTitle))

	return sb.String()
}

// getTitle extracts a title from object data using namespace-agnostic lookup
func (bf *BookmarkFiler) getTitle(data map[string]interface{}, fallbackID string) string {
	title := utils.GetLocalString(data, "title")
	if title != "" {
		return title
	}

	about := utils.GetLocalString(data, "about")
	if about != "" {
		return about
	}

	return fallbackID
}
