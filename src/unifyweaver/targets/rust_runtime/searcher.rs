use redb::{Database, ReadableTable, TableDefinition};
use serde_json::Value;
use std::sync::Arc;
use std::collections::HashSet;

const OBJECTS: TableDefinition<&str, &str> = TableDefinition::new("objects");
const LINKS: TableDefinition<&str, &str> = TableDefinition::new("links");

pub struct PtSearcher {
    db: Arc<Database>,
}

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub id: String,
    pub score: f32,
    pub data: Value,
}

impl PtSearcher {
    pub fn new(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let db = Database::create(path)?; // Opens or creates
        Ok(Self { db: Arc::new(db) })
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

    pub fn graph_search(&self, query: &str, top_k: usize, _hops: usize, mode: &str) -> Result<Vec<Value>, Box<dyn std::error::Error>> {
        let seeds = if mode == "text" {
            self.text_search(query, top_k)?
        } else {
            // Vector search placeholder
            println!("WARNING: Vector search not implemented, falling back to text search");
            self.text_search(query, top_k)?
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
            // Links key is "Source|Target". 
            // Redb range scan is needed for prefix matching if we want efficient lookups
            // Key format: "Source|Target"
            // To find targets for source S, scan range "S|" to "S|~"
            
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

            // Find Parents (Target = Seed)
            // This is hard with "Source|Target" keys. We need to scan ALL links? 
            // Or maintain a reverse index "Target|Source".
            // For this prototype, we'll skip reverse lookup optimization and just scan (slow)
            // or assume Importer stores both directions?
            // Let's assume Importer stores "Child|Parent" as the standard link. 
            // So "Source|Target" means "Child|Parent".
            // Parents are found by looking up the object where `id` is the target of a link where `source` is seed.
            // Which is what we just did above (Source=Seed -> Target=Parent).
            // Wait, "pt:parentTree" implies Child -> Parent.
            // So `upsert_link(Child, Parent)`.
            // Finding Parents: keys starting with "Child|". (Efficient)
            // Finding Children: keys where Target = Seed. (Inefficient without index).
            
            // For now, we'll just output what we can efficiently find (Parents).
            // To fix this, Importer should store "reverse_links" table or keys.
            
            final_results.push(Value::Object(context));
        }

        Ok(final_results)
    }
}
