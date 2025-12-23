// KG Topology Phase 7: HNSW Hierarchical Navigable Small World
// Generated from Prolog service definition
//
// Implements HNSW for O(log n) approximate nearest neighbor search.
// See: Malkov & Yashunin (2018) "Efficient and robust approximate nearest neighbor search"

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};
use rand::Rng;

// Configuration defaults
pub const DEFAULT_M: usize = 16;        // Max neighbors per layer
pub const DEFAULT_M0: usize = 32;       // Max neighbors at layer 0
pub const DEFAULT_EF_SEARCH: usize = 50; // Search beam width
pub const DEFAULT_ML: f64 = 1.0;        // Level multiplier

/// HNSW Node
pub struct HNSWNode {
    pub id: String,
    pub vector: Vec<f32>,
    pub max_layer: usize,
    pub neighbors: HashMap<usize, HashSet<String>>, // layer -> neighbor IDs
}

impl HNSWNode {
    pub fn new(id: String, vector: Vec<f32>, max_layer: usize) -> Self {
        Self {
            id,
            vector,
            max_layer,
            neighbors: HashMap::new(),
        }
    }

    pub fn get_neighbors_at_layer(&self, layer: usize) -> Vec<String> {
        self.neighbors
            .get(&layer)
            .map(|s| s.iter().cloned().collect())
            .unwrap_or_default()
    }

    pub fn add_neighbor(&mut self, neighbor_id: &str, layer: usize, max_neighbors: usize) -> bool {
        if neighbor_id == self.id {
            return false;
        }

        let neighbors = self.neighbors.entry(layer).or_insert_with(HashSet::new);
        if neighbors.len() >= max_neighbors {
            return false;
        }

        neighbors.insert(neighbor_id.to_string());
        true
    }
}

/// Search result
#[derive(Clone, Debug)]
pub struct SearchResult {
    pub id: String,
    pub dist: f64,
}

/// HNSW Graph with tunable M parameter
pub struct HNSWGraph {
    pub nodes: RwLock<HashMap<String, Arc<RwLock<HNSWNode>>>>,
    pub entry_point_id: RwLock<Option<String>>,
    pub max_layer: RwLock<usize>,
    pub m: usize,
    pub m0: usize,
    pub ml: f64,
    pub ef_construction: usize,
}

impl HNSWGraph {
    /// Create new HNSW graph with tunable M parameter
    pub fn new(m: usize) -> Self {
        let m = if m == 0 { DEFAULT_M } else { m };
        Self {
            nodes: RwLock::new(HashMap::new()),
            entry_point_id: RwLock::new(None),
            max_layer: RwLock::new(0),
            m,
            m0: m * 2,
            ml: DEFAULT_ML,
            ef_construction: DEFAULT_EF_SEARCH,
        }
    }

    /// Assign random layer using exponential distribution
    fn random_layer(&self) -> usize {
        let mut rng = rand::thread_rng();
        let r: f64 = rng.gen();
        let r = if r == 0.0 { 0.0001 } else { r };
        (-r.ln() * self.ml) as usize
    }

    /// Add node to the HNSW graph
    pub fn add_node(&self, node_id: &str, vector: Vec<f32>) -> Arc<RwLock<HNSWNode>> {
        let node_layer = self.random_layer();
        let node = Arc::new(RwLock::new(HNSWNode::new(
            node_id.to_string(),
            vector.clone(),
            node_layer,
        )));

        // Insert node
        {
            let mut nodes = self.nodes.write().unwrap();
            nodes.insert(node_id.to_string(), node.clone());
        }

        // Check if first node
        let entry_point = self.entry_point_id.read().unwrap().clone();
        if entry_point.is_none() {
            *self.entry_point_id.write().unwrap() = Some(node_id.to_string());
            *self.max_layer.write().unwrap() = node_layer;
            return node;
        }

        let entry_id = entry_point.unwrap();
        let current_max_layer = *self.max_layer.read().unwrap();

        // Greedy descent from top to node's layer + 1
        let mut current_id = entry_id;
        for layer in (node_layer + 1..=current_max_layer).rev() {
            let closest = self.greedy_search_layer(&vector, &current_id, layer, 1);
            if !closest.is_empty() {
                current_id = closest[0].id.clone();
            }
        }

        // Connect at each layer
        let nodes = self.nodes.read().unwrap();
        for layer in (0..=node_layer.min(current_max_layer)).rev() {
            let candidates = self.search_layer(&vector, &current_id, layer, self.ef_construction);

            let max_n = if layer == 0 { self.m0 } else { self.m };
            let mut connected = 0;

            for cand in &candidates {
                if connected >= max_n {
                    break;
                }
                if cand.id == node_id {
                    continue;
                }

                // Bidirectional connection
                {
                    let mut node_lock = node.write().unwrap();
                    node_lock.add_neighbor(&cand.id, layer, max_n);
                }

                if let Some(neighbor) = nodes.get(&cand.id) {
                    let mut neighbor_lock = neighbor.write().unwrap();
                    neighbor_lock.add_neighbor(node_id, layer, max_n);
                }
                connected += 1;
            }

            if !candidates.is_empty() {
                current_id = candidates[0].id.clone();
            }
        }
        drop(nodes);

        // Update entry point if needed
        if node_layer > current_max_layer {
            *self.entry_point_id.write().unwrap() = Some(node_id.to_string());
            *self.max_layer.write().unwrap() = node_layer;
        }

        node
    }

    /// Greedy search at a single layer
    fn greedy_search_layer(&self, query: &[f32], entry_id: &str, layer: usize, k: usize) -> Vec<SearchResult> {
        let nodes = self.nodes.read().unwrap();
        let mut visited: HashSet<String> = HashSet::new();
        visited.insert(entry_id.to_string());

        let entry_node = match nodes.get(entry_id) {
            Some(n) => n.read().unwrap(),
            None => return vec![],
        };
        let entry_dist = cosine_distance(query, &entry_node.vector);
        drop(entry_node);

        let mut candidates = vec![SearchResult {
            id: entry_id.to_string(),
            dist: entry_dist,
        }];

        loop {
            candidates.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap());
            let current = &candidates[0];
            let current_id = current.id.clone();
            let current_dist = current.dist;

            let current_node = nodes.get(&current_id).unwrap().read().unwrap();
            let mut improved = false;

            for neighbor_id in current_node.get_neighbors_at_layer(layer) {
                if visited.contains(&neighbor_id) {
                    continue;
                }
                visited.insert(neighbor_id.clone());

                if let Some(neighbor) = nodes.get(&neighbor_id) {
                    let neighbor = neighbor.read().unwrap();
                    let dist = cosine_distance(query, &neighbor.vector);
                    if dist < current_dist {
                        candidates.push(SearchResult {
                            id: neighbor_id,
                            dist,
                        });
                        improved = true;
                    }
                }
            }

            if !improved {
                break;
            }
        }

        candidates.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap());
        candidates.truncate(k);
        candidates
    }

    /// Beam search at a single layer
    fn search_layer(&self, query: &[f32], entry_id: &str, layer: usize, ef: usize) -> Vec<SearchResult> {
        let nodes = self.nodes.read().unwrap();
        let mut visited: HashSet<String> = HashSet::new();
        visited.insert(entry_id.to_string());

        let entry_node = match nodes.get(entry_id) {
            Some(n) => n.read().unwrap(),
            None => return vec![],
        };
        let entry_dist = cosine_distance(query, &entry_node.vector);
        drop(entry_node);

        let mut candidates = vec![SearchResult {
            id: entry_id.to_string(),
            dist: entry_dist,
        }];
        let mut results = candidates.clone();

        while !candidates.is_empty() {
            candidates.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap());
            let current = candidates.remove(0);

            results.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap());
            if !results.is_empty() && current.dist > results.last().unwrap().dist {
                break;
            }

            if let Some(current_node) = nodes.get(&current.id) {
                let current_node = current_node.read().unwrap();
                for neighbor_id in current_node.get_neighbors_at_layer(layer) {
                    if visited.contains(&neighbor_id) {
                        continue;
                    }
                    visited.insert(neighbor_id.clone());

                    if let Some(neighbor) = nodes.get(&neighbor_id) {
                        let neighbor = neighbor.read().unwrap();
                        let dist = cosine_distance(query, &neighbor.vector);

                        results.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap());
                        if results.len() < ef || dist < results.last().unwrap().dist {
                            let result = SearchResult {
                                id: neighbor_id,
                                dist,
                            };
                            results.push(result.clone());
                            candidates.push(result);

                            if results.len() > ef {
                                results.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap());
                                results.truncate(ef);
                            }
                        }
                    }
                }
            }
        }

        results.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap());
        results
    }

    /// Search for k nearest neighbors
    pub fn search(&self, query: &[f32], k: usize, ef_search: usize) -> Vec<SearchResult> {
        let entry_point = self.entry_point_id.read().unwrap().clone();
        if entry_point.is_none() {
            return vec![];
        }

        let mut current_id = entry_point.unwrap();
        let max_layer = *self.max_layer.read().unwrap();

        // Greedy descent from top layer
        for layer in (1..=max_layer).rev() {
            let closest = self.greedy_search_layer(query, &current_id, layer, 1);
            if !closest.is_empty() {
                current_id = closest[0].id.clone();
            }
        }

        // Beam search at layer 0
        let mut results = self.search_layer(query, &current_id, 0, ef_search);
        results.truncate(k);
        results
    }

    /// Route to find nearest (alias for search with k=1)
    pub fn route(&self, query: &[f32], _use_backtrack: bool) -> (Vec<String>, usize) {
        let results = self.search(query, 1, DEFAULT_EF_SEARCH);
        if results.is_empty() {
            (vec![], 0)
        } else {
            (vec![results[0].id.clone()], 1)
        }
    }
}

/// Compute cosine distance (1 - similarity)
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f64 {
    let mut dot: f64 = 0.0;
    let mut norm_a: f64 = 0.0;
    let mut norm_b: f64 = 0.0;

    for (ai, bi) in a.iter().zip(b.iter()) {
        dot += (*ai as f64) * (*bi as f64);
        norm_a += (*ai as f64) * (*ai as f64);
        norm_b += (*bi as f64) * (*bi as f64);
    }

    if norm_a == 0.0 || norm_b == 0.0 {
        return 1.0;
    }

    1.0 - dot / (norm_a.sqrt() * norm_b.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_hnsw_graph() {
        let g = HNSWGraph::new(16);
        assert_eq!(g.m, 16);
        assert_eq!(g.m0, 32);
    }

    #[test]
    fn test_add_node() {
        let g = HNSWGraph::new(16);

        g.add_node("n1", vec![1.0, 0.0, 0.0]);
        assert_eq!(
            g.entry_point_id.read().unwrap().as_ref().unwrap(),
            "n1"
        );

        g.add_node("n2", vec![0.9, 0.1, 0.0]);
        g.add_node("n3", vec![0.8, 0.2, 0.0]);

        assert_eq!(g.nodes.read().unwrap().len(), 3);
    }

    #[test]
    fn test_search() {
        let g = HNSWGraph::new(16);

        g.add_node("n1", vec![1.0, 0.0, 0.0]);
        g.add_node("n2", vec![0.9, 0.1, 0.0]);
        g.add_node("n3", vec![0.8, 0.2, 0.0]);
        g.add_node("n4", vec![0.0, 1.0, 0.0]);

        let query = vec![1.0, 0.0, 0.0];
        let results = g.search(&query, 3, 50);

        assert!(!results.is_empty());
        assert_eq!(results[0].id, "n1");
        assert!(results[0].dist < 0.001);
    }

    #[test]
    fn test_route() {
        let g = HNSWGraph::new(16);

        g.add_node("n1", vec![1.0, 0.0, 0.0]);
        g.add_node("n2", vec![0.9, 0.1, 0.0]);
        g.add_node("n3", vec![0.0, 1.0, 0.0]);

        let query = vec![1.0, 0.0, 0.0];
        let (path, count) = g.route(&query, true);

        assert!(!path.is_empty());
        assert_eq!(path[0], "n1");
        assert!(count > 0);
    }

    #[test]
    fn test_cosine_distance() {
        // Identical vectors
        let dist = cosine_distance(&[1.0, 0.0, 0.0], &[1.0, 0.0, 0.0]);
        assert!((dist - 0.0).abs() < 0.001);

        // Orthogonal vectors
        let dist = cosine_distance(&[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0]);
        assert!((dist - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_tunable_m() {
        for m in [4, 8, 16, 32] {
            let g = HNSWGraph::new(m);
            assert_eq!(g.m, m);
            assert_eq!(g.m0, m * 2);
        }
    }
}
