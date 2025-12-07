// Demo: Bookmark Filing Assistant using semantic search and tree formatting
//
// This demonstrates Rust's bookmark filing capabilities:
// - Vector similarity search with ONNX embeddings
// - Tree context formatting (ancestors, siblings, children)
// - Semantic placement suggestions

mod importer;
mod crawler;
mod searcher;
mod embedding;

use searcher::PtSearcher;
use embedding::EmbeddingProvider;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    println!("=== Bookmark Filing Assistant Demo ===\n");
    println!("Loading database and embedding model...");

    let model_path = "models/all-MiniLM-L6-v2-safetensors/model.safetensors";
    let tokenizer_path = "models/all-MiniLM-L6-v2-safetensors/tokenizer.json";
    let db_path = "pt_ingest_test.redb";

    let embedder = EmbeddingProvider::new(model_path, tokenizer_path)?;
    let searcher = PtSearcher::new(db_path, embedder)?;

    println!("✓ Ready\n");

    // Test cases for bookmark filing
    let bookmarks = vec![
        "Article about quantum entanglement and its applications in quantum computing",
        "Tutorial on classical mechanics and Newton's laws of motion",
        "Research paper on electromagnetic waves and Maxwell's equations",
    ];

    for bookmark in &bookmarks {
        println!("{}", "=".repeat(100));
        println!();

        // Try with lower threshold to see results
        let result = searcher.find_bookmark_placements(bookmark, 3, 0.1)?;
        println!("{}", result);

        println!();
    }

    println!("{}", "=".repeat(100));
    println!("\n✓ Demo complete");

    Ok(())
}
