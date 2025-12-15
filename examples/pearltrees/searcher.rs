use redb::{Database, ReadableTable, TableDefinition};
use serde_json::Value;
use std::sync::Arc;
use std::collections::HashMap;
use crate::embedding::EmbeddingProvider;
use crate::projection::MultiHeadProjection;

const OBJECTS: TableDefinition<&str, &str> = TableDefinition::new("objects");
const LINKS: TableDefinition<&str, &str> = TableDefinition::new("links");
const EMBEDDINGS: TableDefinition<&str, &[u8]> = TableDefinition::new("embeddings");

pub struct PtSearcher {
    db: Arc<Database>,
    embedder: EmbeddingProvider,
    projection: Option<MultiHeadProjection>,
}

/// Options for vector search with projection.
#[derive(Debug, Clone, Default)]
pub struct SearchOptions {
    /// Use projection if available
    pub use_projection: bool,
    /// Include routing weights in results
    pub include_routing_weights: bool,
}

/// Extended search result with optional routing weights.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ExtendedSearchResult {
    pub id: String,
    pub score: f32,
    pub data: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub routing_weights: Option<HashMap<i32, f32>>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct SearchResult {
    pub id: String,
    pub score: f32,
    pub data: Value,
}

impl PtSearcher {
    /// Create a new PtSearcher without projection.
    pub fn new(path: &str, embedder: EmbeddingProvider) -> Result<Self, Box<dyn std::error::Error>> {
        let db = Database::create(path)?; // Opens or creates
        Ok(Self { db: Arc::new(db), embedder, projection: None })
    }

    /// Create a new PtSearcher with multi-head LDA projection.
    pub fn with_projection(path: &str, embedder: EmbeddingProvider, projection: MultiHeadProjection) -> Result<Self, Box<dyn std::error::Error>> {
        let db = Database::create(path)?;
        Ok(Self { db: Arc::new(db), embedder, projection: Some(projection) })
    }

    /// Set the projection model.
    pub fn set_projection(&mut self, projection: MultiHeadProjection) {
        self.projection = Some(projection);
    }

    /// Check if projection is enabled.
    pub fn has_projection(&self) -> bool {
        self.projection.is_some()
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

    /// Vector search with optional multi-head LDA projection.
    /// When projection is enabled, the query embedding is projected through the
    /// multi-head model before computing similarities.
    pub fn vector_search_with_options(&self, query: &str, top_k: usize, options: SearchOptions) -> Result<Vec<ExtendedSearchResult>, Box<dyn std::error::Error>> {
        // 1. Generate Query Embedding
        let query_vec = self.embedder.get_embedding(query)?;

        // 2. Apply projection if configured
        let (search_vec, routing_weights) = if options.use_projection {
            if let Some(ref projection) = self.projection {
                if options.include_routing_weights {
                    let (projected, weights) = projection.project_with_weights(&query_vec)?;
                    (projected, Some(weights))
                } else {
                    let projected = projection.project(&query_vec)?;
                    (projected, None)
                }
            } else {
                (query_vec, None)
            }
        } else {
            (query_vec, None)
        };

        // 3. Linear Scan & Compute Cosine Similarity
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
            let score = cosine_similarity(&search_vec, &vec);
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
                full_results.push(ExtendedSearchResult {
                    id,
                    score,
                    data,
                    routing_weights: routing_weights.clone(),
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

    pub fn find_bookmark_placements(&self, bookmark_description: &str, top_candidates: usize, min_score: f32) -> Result<String, Box<dyn std::error::Error>> {
        // Find semantically similar trees using vector search
        let candidates = self.vector_search(bookmark_description, top_candidates)?;

        // Filter by score and type (only trees)
        let candidates: Vec<_> = candidates.into_iter()
            .filter(|c| c.score >= min_score)
            .filter(|c| {
                c.data.get("@type")
                    .and_then(|v| v.as_str())
                    .map(|t| t == "pt:Tree")
                    .unwrap_or(false)
            })
            .collect();

        if candidates.is_empty() {
            return Ok("No suitable placement locations found. Consider lowering the similarity threshold or creating a new category.".to_string());
        }

        let mut output = String::new();
        output.push_str("=== Bookmark Filing Suggestions ===\n\n");
        output.push_str(&format!("Bookmark: \"{}\"\n", bookmark_description));
        output.push_str(&format!("Found {} candidate location(s):\n\n", candidates.len()));
        output.push_str(&"=".repeat(80));
        output.push_str("\n\n");

        for (i, candidate) in candidates.iter().enumerate() {
            output.push_str(&format!("Option {}:\n\n", i + 1));
            output.push_str(&self.build_tree_context(&candidate.id, candidate.score)?);
            output.push_str("\n");
            if i < candidates.len() - 1 {
                output.push_str(&"-".repeat(80));
                output.push_str("\n\n");
            }
        }

        Ok(output)
    }

    /// Build a tree-formatted context string for a candidate entity.
    /// Shows ancestors, siblings, the candidate itself (marked), and children.
    pub fn build_tree_context(&self, candidate_id: &str, score: f32) -> Result<String, Box<dyn std::error::Error>> {
        let candidate = self.get_entity(candidate_id)?;

        let mut output = String::new();
        let title = candidate.get("title")
            .or(candidate.get("@about"))
            .and_then(|v| v.as_str())
            .unwrap_or(candidate_id);

        // Header with score
        output.push_str(&format!("Candidate: \"{}\" (similarity: {:.3})\n\n", title, score));

        // Build ancestor path (root to immediate parent)
        let ancestors = self.get_ancestors(candidate_id)?;

        // Show ancestor path (reversed to show root first)
        if !ancestors.is_empty() {
            for (i, ancestor) in ancestors.iter().rev().enumerate() {
                let indent = " ".repeat(i * 4);
                let ancestor_title = ancestor.get("title")
                    .or(ancestor.get("@about"))
                    .and_then(|v| v.as_str())
                    .unwrap_or(&ancestor["id"].as_str().unwrap_or(""));
                let ancestor_title = truncate(ancestor_title, 60);

                if i == 0 {
                    output.push_str(&format!("└── {}/\n", ancestor_title));
                } else {
                    output.push_str(&format!("{}└── {}/\n", indent, ancestor_title));
                }
            }
        }

        // Show siblings and candidate at current level
        let siblings = self.get_siblings(candidate_id)?;
        let base_indent = " ".repeat(ancestors.len() * 4);

        // Show some siblings before candidate
        for sibling in siblings.iter().take(3) {
            let sibling_title = sibling.get("title")
                .or(sibling.get("@about"))
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let sibling_title = truncate(sibling_title, 60);
            output.push_str(&format!("{}    ├── {}/\n", base_indent, sibling_title));
        }

        // Show the candidate (marked with arrow)
        let cand_title = truncate(title, 50);
        output.push_str(&format!("{}    ├── {}/        ← CANDIDATE (place new bookmark here)\n", base_indent, cand_title));

        // Show remaining siblings
        if siblings.len() > 3 {
            for sibling in siblings.iter().skip(3).take(2) {
                let sibling_title = sibling.get("title")
                    .or(sibling.get("@about"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let sibling_title = truncate(sibling_title, 60);
                output.push_str(&format!("{}    ├── {}/\n", base_indent, sibling_title));
            }
            if siblings.len() > 5 {
                output.push_str(&format!("{}    ├── ... ({} more siblings)\n", base_indent, siblings.len() - 5));
            }
        }

        // Show children of the candidate
        let children = self.get_children(candidate_id)?;
        if !children.is_empty() {
            let child_indent = format!("{}    │   ", base_indent);
            for (j, child) in children.iter().take(10).enumerate() {
                let child_title = child.get("title")
                    .or(child.get("@about"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let child_title = truncate(child_title, 60);
                let is_last = j == children.len().min(10) - 1;
                let prefix = if is_last && children.len() <= 10 { "└──" } else { "├──" };
                output.push_str(&format!("{}{} {}\n", child_indent, prefix, child_title));
            }
            if children.len() > 10 {
                output.push_str(&format!("{}└── ... ({} more children)\n", child_indent, children.len() - 10));
            }
        }

        Ok(output)
    }

    fn get_entity(&self, id: &str) -> Result<serde_json::Map<String, Value>, Box<dyn std::error::Error>> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(OBJECTS)?;

        if let Some(data_str) = table.get(id)? {
            let data: Value = serde_json::from_str(data_str.value())?;
            if let Value::Object(map) = data {
                return Ok(map);
            }
        }

        Err(format!("Entity {} not found", id).into())
    }

    fn get_children(&self, id: &str) -> Result<Vec<serde_json::Map<String, Value>>, Box<dyn std::error::Error>> {
        let entity = self.get_entity(id)?;
        let children_ids = entity.get("children")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().filter_map(|v| v.as_str()).collect::<Vec<_>>())
            .unwrap_or_default();

        let mut children = Vec::new();
        for child_id in children_ids {
            if let Ok(child) = self.get_entity(child_id) {
                children.push(child);
            }
        }
        Ok(children)
    }

    fn get_parent(&self, id: &str) -> Result<Option<serde_json::Map<String, Value>>, Box<dyn std::error::Error>> {
        let entity = self.get_entity(id)?;
        let parent_id = entity.get("parentTree@resource")
            .or(entity.get("@parentTree"))
            .and_then(|v| v.as_str());

        if let Some(parent_id) = parent_id {
            Ok(Some(self.get_entity(parent_id)?))
        } else {
            Ok(None)
        }
    }

    fn get_ancestors(&self, id: &str) -> Result<Vec<serde_json::Map<String, Value>>, Box<dyn std::error::Error>> {
        let mut ancestors = Vec::new();
        let mut current = self.get_parent(id)?;
        let mut visited = std::collections::HashSet::new();
        visited.insert(id.to_string());

        while let Some(entity) = current {
            let entity_id = entity.get("@id")
                .or(entity.get("@rdf:about"))
                .and_then(|v| v.as_str())
                .unwrap_or("");

            if visited.contains(entity_id) {
                break; // Prevent cycles
            }

            visited.insert(entity_id.to_string());
            current = self.get_parent(entity_id)?;
            ancestors.push(entity);
        }

        Ok(ancestors)
    }

    fn get_siblings(&self, id: &str) -> Result<Vec<serde_json::Map<String, Value>>, Box<dyn std::error::Error>> {
        let entity = self.get_entity(id)?;
        let parent_id = entity.get("parentTree@resource")
            .or(entity.get("@parentTree"))
            .and_then(|v| v.as_str());

        if let Some(parent_id) = parent_id {
            let parent = self.get_entity(parent_id)?;
            let children_ids = parent.get("children")
                .and_then(|v| v.as_array())
                .map(|arr| arr.iter().filter_map(|v| v.as_str()).collect::<Vec<_>>())
                .unwrap_or_default();

            let mut siblings = Vec::new();
            for sibling_id in children_ids {
                if sibling_id != id {
                    if let Ok(sibling) = self.get_entity(sibling_id) {
                        siblings.push(sibling);
                    }
                }
            }
            Ok(siblings)
        } else {
            Ok(Vec::new())
        }
    }

    pub fn suggest_bookmarks(&self, query: &str, top_k: usize) -> Result<String, Box<dyn std::error::Error>> {
        self.find_bookmark_placements(query, top_k, 0.35)
    }
}

fn truncate(s: &str, max_len: usize) -> String {
    if s.len() > max_len {
        format!("{}...", &s[..max_len - 3])
    } else {
        s.to_string()
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
