// Quick test to see vector search results

mod importer;
mod crawler;
mod projection;
mod searcher;
mod embedding;

use searcher::PtSearcher;
use embedding::EmbeddingProvider;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let model_path = "models/all-MiniLM-L6-v2-safetensors/model.safetensors";
    let tokenizer_path = "models/all-MiniLM-L6-v2-safetensors/tokenizer.json";
    let db_path = "pt_ingest_test.redb";

    let embedder = EmbeddingProvider::new(model_path, tokenizer_path)?;
    let searcher = PtSearcher::new(db_path, embedder)?;

    let query = "Article about quantum entanglement and its applications in quantum computing";
    eprintln!("Searching for: {}", query);

    let results = searcher.vector_search(query, 10)?;
    eprintln!("\nTop 10 results:");
    for (i, res) in results.iter().enumerate() {
        let title = res.data.get("title")
            .or(res.data.get("@about"))
            .and_then(|v| v.as_str())
            .unwrap_or("(no title)");
        let entity_type = res.data.get("@type")
            .and_then(|v| v.as_str())
            .unwrap_or("(no type)");
        eprintln!("{:2}. [{:.4}] {} - {} (type: {})", i+1, res.score, res.id, title, entity_type);
    }

    Ok(())
}
