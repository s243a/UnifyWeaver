package main

import (
	"fmt"
	"os"

	"unifyweaver/targets/go_runtime/crawler"
	"unifyweaver/targets/go_runtime/embedder"
	"unifyweaver/targets/go_runtime/search"
	"unifyweaver/targets/go_runtime/storage"
)

func main() {
	dbPath := "pearltrees.bolt"

	// Check for --search flag
	if len(os.Args) > 1 && os.Args[1] == "--search" {
		if len(os.Args) < 3 {
			fmt.Fprintf(os.Stderr, "Usage: %s --search <query>\n", os.Args[0])
			os.Exit(1)
		}
		query := os.Args[2]
		runSearch(dbPath, query)
		return
	}

	// Default: ingest from stdin
	runIngest(dbPath)
}

func runIngest(dbPath string) {
	fmt.Fprintf(os.Stderr, "=== Go Pearltrees Ingestion ===\n")

	// Initialize store
	store, err := storage.NewStore(dbPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error opening store: %v\n", err)
		os.Exit(1)
	}
	defer store.Close()
	fmt.Fprintf(os.Stderr, "Database: %s\n", dbPath)

	// Initialize embedder - try HugotEmbedder first, fall back to stub
	modelPath := "../../models/all-MiniLM-L6-v2-onnx"
	var emb crawler.Embedder
	hugotEmb, err := embedder.NewHugotEmbedder(modelPath, "all-MiniLM-L6-v2")
	if err != nil {
		fmt.Fprintf(os.Stderr, "Warning: Could not load HugotEmbedder: %v\n", err)
		fmt.Fprintf(os.Stderr, "Falling back to stub embedder...\n")
		emb = embedder.NewStubEmbedder(384)
	} else {
		fmt.Fprintf(os.Stderr, "Embedding model: all-MiniLM-L6-v2 (hugot)\n")
		emb = hugotEmb
		defer hugotEmb.Close()
	}

	// Initialize crawler
	c := crawler.NewCrawler(store, emb)

	// Process fragments from stdin
	fmt.Fprintf(os.Stderr, "Reading XML fragments from stdin...\n")
	fmt.Fprintf(os.Stderr, "(Expecting null-delimited fragments from AWK)\n")

	if err := c.ProcessFragmentsFromStdin(); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}

	// Print stats
	objCount, _ := store.CountObjects()
	embCount, _ := store.CountEmbeddings()
	fmt.Fprintf(os.Stderr, "\n✓ Ingestion complete!\n")
	fmt.Fprintf(os.Stderr, "Objects: %d\n", objCount)
	fmt.Fprintf(os.Stderr, "Embeddings: %d\n", embCount)
}

func runSearch(dbPath, query string) {
	fmt.Fprintf(os.Stderr, "=== Go Bookmark Filing ===\n")

	// Initialize store
	store, err := storage.NewStore(dbPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error opening store: %v\n", err)
		os.Exit(1)
	}
	defer store.Close()

	// Initialize embedder - try HugotEmbedder first, fall back to stub
	modelPath := "../../models/all-MiniLM-L6-v2-onnx"
	var emb search.Embedder
	hugotEmb, err := embedder.NewHugotEmbedder(modelPath, "all-MiniLM-L6-v2")
	if err != nil {
		fmt.Fprintf(os.Stderr, "Warning: Could not load HugotEmbedder: %v\n", err)
		fmt.Fprintf(os.Stderr, "Falling back to stub embedder...\n")
		emb = embedder.NewStubEmbedder(384)
	} else {
		fmt.Fprintf(os.Stderr, "Embedding model: all-MiniLM-L6-v2 (hugot)\n")
		emb = hugotEmb
		defer hugotEmb.Close()
	}

	// Initialize bookmark filer
	filer := search.NewBookmarkFiler(store, store, emb)

	// Get suggestions
	result, err := filer.SuggestBookmarks(query, 3)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}

	fmt.Println(result)
}
