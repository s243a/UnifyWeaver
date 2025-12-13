package main

import (
	"fmt"
	"os"

	"unifyweaver/targets/go_runtime/crawler"
	"unifyweaver/targets/go_runtime/embedder"
	"unifyweaver/targets/go_runtime/search"
	"unifyweaver/targets/go_runtime/storage"
)

// createEmbedder initializes the embedder based on compile-time backend selection.
// Uses MODEL_PATH environment variable or defaults to all-MiniLM-L6-v2.
// Returns the embedder and an optional close function.
func createEmbedder() (embedder.Embedder, func()) {
	// Get model path from environment or use default
	modelPath := os.Getenv("MODEL_PATH")
	if modelPath == "" {
		// Default paths based on backend
		backend := embedder.AvailableBackend()
		switch backend {
		case embedder.BackendCandle:
			// Candle uses HuggingFace model ID or local safetensors path
			modelPath = "sentence-transformers/all-MiniLM-L6-v2"
		default:
			// Pure Go/ORT/XLA use ONNX model path
			modelPath = "../../models/all-MiniLM-L6-v2-onnx"
		}
	}

	backend := embedder.AvailableBackend()
	fmt.Fprintf(os.Stderr, "Backend: %s\n", backend)
	fmt.Fprintf(os.Stderr, "Model: %s\n", modelPath)

	// Check for GPU environment variable
	useGPU := os.Getenv("USE_GPU") == "1" || os.Getenv("USE_GPU") == "true"

	config := embedder.EmbedderConfig{
		ModelPath:  modelPath,
		ModelName:  "all-MiniLM-L6-v2",
		Dimensions: 384,
		UseGPU:     useGPU,
		MaxLength:  512,
	}

	emb, err := embedder.NewEmbedder(config)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Warning: Could not load embedder (%s): %v\n", backend, err)
		fmt.Fprintf(os.Stderr, "Falling back to stub embedder...\n")
		return embedder.NewStubEmbedder(384), nil
	}

	fmt.Fprintf(os.Stderr, "Embedding model loaded successfully\n")
	return emb, func() { emb.Close() }
}

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

	// Initialize embedder using backend selected at compile time
	emb, closeFunc := createEmbedder()
	if closeFunc != nil {
		defer closeFunc()
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

	// Initialize embedder using backend selected at compile time
	emb, closeFunc := createEmbedder()
	if closeFunc != nil {
		defer closeFunc()
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
