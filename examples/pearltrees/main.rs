// Rust Pearltrees Ingestion Tool
// Reads null-delimited XML fragments from stdin (AWK output) and creates redb database

mod importer;
mod crawler;
mod embedding;

use importer::PtImporter;
use crawler::PtCrawler;
use embedding::EmbeddingProvider;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    eprintln!("=== Rust Pearltrees Ingestion ===");
    eprintln!("Loading embedding model...");

    let model_path = "/home/s243a/Projects/UnifyWeaver/models/all-MiniLM-L6-v2-safetensors/model.safetensors";
    let tokenizer_path = "/home/s243a/Projects/UnifyWeaver/models/all-MiniLM-L6-v2-safetensors/tokenizer.json";
    let db_path = "/home/s243a/Projects/UnifyWeaver/pt_ingest_test.redb";

    // Initialize embedding provider
    let embedder = EmbeddingProvider::new(model_path, tokenizer_path)?;
    eprintln!("✓ Model loaded");

    // Initialize importer
    eprintln!("Creating database: {}", db_path);
    let importer = PtImporter::new(db_path)?;

    // Create crawler
    let crawler = PtCrawler::new(importer, embedder);

    // Process fragments from stdin
    eprintln!("Reading XML fragments from stdin...");
    eprintln!("(Expecting null-delimited fragments from AWK)");

    crawler.process_fragments_from_stdin()?;

    eprintln!("\n✓ Ingestion complete!");
    eprintln!("Database created: {}", db_path);

    Ok(())
}
