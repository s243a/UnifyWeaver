use redb::{Database, ReadableTable, TableDefinition};
use serde_json::Value;
use std::sync::Arc;
use std::collections::HashSet;
use crate::embedding::EmbeddingProvider;

const OBJECTS: TableDefinition<&str, &str> = TableDefinition::new("objects");
const LINKS: TableDefinition<&str, &str> = TableDefinition::new("links");
const EMBEDDINGS: TableDefinition<&str, &[u8]> = TableDefinition::new("embeddings");

pub struct PtSearcher {
    db: Arc<Database>,
    embedder: EmbeddingProvider,
}

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub id: String,
    pub score: f32,
    pub data: Value,
}

impl PtSearcher {
    pub fn new(path: &str, embedder: EmbeddingProvider) -> Result<Self, Box<dyn std::error::Error>> {
        let db = Database::create(path)?; // Opens or creates
        Ok(Self { db: Arc::new(db), embedder })
    }

    pub fn text_search(&self, query: &str, top_k: usize) -> Result<Vec<SearchResult>, Box<dyn std::error::Error>> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(OBJECTS)?;
        
        let query_lower = query.to_lowercase();
        let mut results = Vec::new();

        // Linear scan for LIKE %query%
        // Optimization: In a real system, use an inverted index.
        for result in table.iter()? {
            let (id, data_str) = result?;
            let id = id.value();
            let data_str = data_str.value();
            
            // Simple heuristic: check if query is in the JSON string
            if data_str.to_lowercase().contains(&query_lower) {
                let data: Value = serde_json::from_str(data_str).unwrap_or(Value::Null);
                results.push(SearchResult {
                    id: id.to_string(),
                    score: 1.0, // Dummy score for text match
                    data,
                });
            }
        }

        // Limit results
        results.truncate(top_k);
        Ok(results)
    }

    pub fn vector_search(&self, query: &str, top_k: usize) -> Result<Vec<SearchResult>, Box<dyn std::error::Error>> {
        // 1. Generate Query Embedding
        let query_vec = self.embedder.get_embedding(query)?;

        // 2. Linear Scan & Compute Cosine Similarity
        // Note: For large datasets, use an index (LanceDB/HNSW). For <10k, scan is fine.
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(EMBEDDINGS)?;
        let objects_table = read_txn.open_table(OBJECTS)?;

        let mut results = Vec::new();

        for result in table.iter()? {
            let (id, blob) = result?;
            let id = id.value();
            let blob = blob.value();

            // Convert blob to f32 vec
            let vec_len = blob.len() / 4;
            let mut vec = Vec::with_capacity(vec_len);
            for chunk in blob.chunks_exact(4) {
                let bytes: [u8; 4] = chunk.try_into()?;
                vec.push(f32::from_ne_bytes(bytes));
            }

            // Cosine Similarity
            let score = cosine_similarity(&query_vec, &vec);
            
            // Optimization: Maintain a min-heap of top-k
            results.push((score, id.to_string()));
        }

        // Sort descending
        results.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);

        // Fetch Objects
        let mut full_results = Vec::new();
        for (score, id) in results {
            if let Ok(Some(data_str)) = objects_table.get(id.as_str()) {
                let data: Value = serde_json::from_str(data_str.value())?;
                full_results.push(SearchResult {
                    id,
                    score,
                    data,
                });
            }
        }

        Ok(full_results)
    }

    pub fn graph_search(&self, query: &str, top_k: usize, _hops: usize, mode: &str) -> Result<Vec<Value>, Box<dyn std::error::Error>> {
        let seeds = if mode == "text" {
            self.text_search(query, top_k)?
        } else {
            self.vector_search(query, top_k)?
        };

        let read_txn = self.db.begin_read()?;
        let links_table = read_txn.open_table(LINKS)?;
        let objects_table = read_txn.open_table(OBJECTS)?;

        let mut final_results = Vec::new();
        
        // 1-hop expansion
        for seed in seeds {
            let mut context = serde_json::Map::new();
            context.insert("focus".to_string(), seed.data.clone());
            
            // Find Children (Source = Seed)
            let mut children = Vec::new();
            let prefix = format!("{}|", seed.id);
            for link_res in links_table.range(prefix.as_str()..)? {
                let (key, _) = link_res?;
                let key_str = key.value();
                if !key_str.starts_with(&prefix) {
                    break;
                }
                // Extract Target
                let parts: Vec<&str> = key_str.split('|').collect();
                if parts.len() == 2 {
                    let target_id = parts[1];
                    if let Ok(Some(obj_val)) = objects_table.get(target_id) {
                         let obj_json: Value = serde_json::from_str(obj_val.value())?;
                         children.push(obj_json);
                    }
                }
            }
            context.insert("children".to_string(), Value::Array(children));
            
            final_results.push(Value::Object(context));
        }

        Ok(final_results)
    }

    pub fn suggest_bookmarks(&self, query: &str, _top_k: usize) -> Result<String, Box<dyn std::error::Error>> {
        // Placeholder for Tree View logic
        // 1. Search
        let results = self.graph_search(query, 5, 1, "text")?;
        
        if results.is_empty() {
            return Ok("No suitable placement locations found.".to_string());
        }

        let mut output = String::new();
        output.push_str("=== Bookmark Filing Suggestions ===\n");
        output.push_str(&format!("Bookmark: \"{}\"\n", query));
        output.push_str(&format!("Found {} candidate location(s):\n\n", results.len()));
        output.push_str("================================================================================\n\n");

        for (i, res) in results.iter().enumerate() {
            output.push_str(&format!("Option {}:\n\n", i + 1));
            // TODO: Build actual tree context string from graph
            // For now, dump JSON
            output.push_str(&serde_json::to_string_pretty(res)?);
            output.push_str("\n\n--------------------------------------------------------------------------------\n\n");
        }

        Ok(output)
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot_product / (norm_a * norm_b)
    }
}
