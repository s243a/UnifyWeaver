// KG Topology Phase 4: Federated Query Engine
// Generated from Prolog service definition
//
// Implements federated search with aggregation strategies:
// - SUM: Boost consensus (exp(z_a) + exp(z_b))
// - MAX: No boost, take max score
// - DIVERSITY: Boost only if sources differ

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::Duration;

/// Aggregation strategy for merging duplicate results
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AggregationStrategy {
    /// Sum scores (consensus boost)
    Sum,
    /// Take maximum score
    Max,
    /// Boost only if from different sources
    Diversity,
}

/// A single search result from a node
#[derive(Clone, Debug)]
pub struct SearchResult {
    pub id: String,
    pub score: f64,
    pub answer_hash: String,
    pub source_node: String,
    pub metadata: HashMap<String, String>,
}

/// Aggregated result across nodes
#[derive(Clone, Debug)]
pub struct AggregatedResult {
    pub id: String,
    pub combined_score: f64,
    pub sources: Vec<String>,
    pub source_count: usize,
    pub metadata: HashMap<String, String>,
}

/// Node client trait for querying nodes
pub trait NodeClient: Send + Sync {
    fn query(&self, embedding: &[f32], k: usize) -> Result<Vec<SearchResult>, String>;
    fn node_id(&self) -> &str;
}

/// Mock node client for testing
pub struct MockNodeClient {
    id: String,
    results: Vec<SearchResult>,
}

impl MockNodeClient {
    pub fn new(id: &str, results: Vec<SearchResult>) -> Self {
        Self {
            id: id.to_string(),
            results,
        }
    }
}

impl NodeClient for MockNodeClient {
    fn query(&self, _embedding: &[f32], k: usize) -> Result<Vec<SearchResult>, String> {
        let k = k.min(self.results.len());
        Ok(self.results[..k].to_vec())
    }

    fn node_id(&self) -> &str {
        &self.id
    }
}

/// Federation configuration
#[derive(Clone, Debug)]
pub struct FederationConfig {
    pub strategy: AggregationStrategy,
    pub federation_k: usize,
    pub top_k: usize,
    pub timeout: Duration,
    pub consensus_threshold: usize,
}

impl Default for FederationConfig {
    fn default() -> Self {
        Self {
            strategy: AggregationStrategy::Sum,
            federation_k: 3,
            top_k: 10,
            timeout: Duration::from_secs(5),
            consensus_threshold: 0,
        }
    }
}

/// Federated Query Engine
pub struct FederatedQueryEngine {
    nodes: RwLock<Vec<Arc<dyn NodeClient>>>,
    config: RwLock<FederationConfig>,
}

impl FederatedQueryEngine {
    /// Create a new federation engine
    pub fn new(config: FederationConfig) -> Self {
        Self {
            nodes: RwLock::new(Vec::new()),
            config: RwLock::new(config),
        }
    }

    /// Add a node to the federation
    pub fn add_node(&self, node: Arc<dyn NodeClient>) {
        let mut nodes = self.nodes.write().unwrap();
        nodes.push(node);
    }

    /// Execute a federated query across nodes
    pub fn query(&self, embedding: &[f32]) -> Result<Vec<AggregatedResult>, String> {
        let nodes = self.nodes.read().unwrap();
        let config = self.config.read().unwrap().clone();

        if nodes.is_empty() {
            return Ok(Vec::new());
        }

        // Select nodes to query (up to federation_k)
        let query_nodes: Vec<_> = nodes
            .iter()
            .take(config.federation_k)
            .cloned()
            .collect();
        drop(nodes);

        // Query nodes and collect results
        let mut all_results = Vec::new();
        for node in &query_nodes {
            match node.query(embedding, config.top_k) {
                Ok(mut results) => {
                    // Tag results with source node
                    for r in &mut results {
                        r.source_node = node.node_id().to_string();
                    }
                    all_results.extend(results);
                }
                Err(_) => continue, // Skip failed nodes
            }
        }

        // Aggregate results
        let mut aggregated = self.aggregate(&all_results, config.strategy);

        // Apply consensus threshold if set
        if config.consensus_threshold > 0 {
            aggregated.retain(|r| r.source_count >= config.consensus_threshold);
        }

        Ok(aggregated)
    }

    /// Aggregate results based on strategy
    fn aggregate(&self, results: &[SearchResult], strategy: AggregationStrategy) -> Vec<AggregatedResult> {
        let mut groups: HashMap<String, AggregatedResult> = HashMap::new();
        let mut source_tracking: HashMap<String, Vec<String>> = HashMap::new();

        for r in results {
            let key = if r.answer_hash.is_empty() {
                r.id.clone()
            } else {
                r.answer_hash.clone()
            };

            let sources = source_tracking.entry(key.clone()).or_insert_with(Vec::new);

            if let Some(existing) = groups.get_mut(&key) {
                // Merge based on strategy
                match strategy {
                    AggregationStrategy::Sum => {
                        existing.combined_score += r.score;
                    }
                    AggregationStrategy::Max => {
                        if r.score > existing.combined_score {
                            existing.combined_score = r.score;
                        }
                    }
                    AggregationStrategy::Diversity => {
                        if !sources.contains(&r.source_node) {
                            // Different source - add
                            existing.combined_score += r.score;
                        } else {
                            // Same source - take max
                            if r.score > existing.combined_score {
                                existing.combined_score = r.score;
                            }
                        }
                    }
                }

                // Track unique sources
                if !sources.contains(&r.source_node) {
                    sources.push(r.source_node.clone());
                    existing.sources.push(r.source_node.clone());
                    existing.source_count += 1;
                }
            } else {
                // New result
                groups.insert(
                    key.clone(),
                    AggregatedResult {
                        id: r.id.clone(),
                        combined_score: r.score,
                        sources: vec![r.source_node.clone()],
                        source_count: 1,
                        metadata: r.metadata.clone(),
                    },
                );
                sources.push(r.source_node.clone());
            }
        }

        // Convert to vec and sort by score descending
        let mut aggregated: Vec<_> = groups.into_values().collect();
        aggregated.sort_by(|a, b| b.combined_score.partial_cmp(&a.combined_score).unwrap());
        aggregated
    }

    /// Update the aggregation strategy
    pub fn set_strategy(&self, strategy: AggregationStrategy) {
        let mut config = self.config.write().unwrap();
        config.strategy = strategy;
    }

    /// Update the number of nodes to query
    pub fn set_federation_k(&self, k: usize) {
        let mut config = self.config.write().unwrap();
        config.federation_k = k;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_result(id: &str, score: f64, hash: &str) -> SearchResult {
        SearchResult {
            id: id.to_string(),
            score,
            answer_hash: hash.to_string(),
            source_node: String::new(),
            metadata: HashMap::new(),
        }
    }

    #[test]
    fn test_new_federation_engine() {
        let config = FederationConfig::default();
        let engine = FederatedQueryEngine::new(config);

        let cfg = engine.config.read().unwrap();
        assert_eq!(cfg.strategy, AggregationStrategy::Sum);
        assert_eq!(cfg.federation_k, 3);
    }

    #[test]
    fn test_add_node() {
        let engine = FederatedQueryEngine::new(FederationConfig::default());

        let node1: Arc<dyn NodeClient> = Arc::new(MockNodeClient::new("node1", vec![]));
        let node2: Arc<dyn NodeClient> = Arc::new(MockNodeClient::new("node2", vec![]));

        engine.add_node(node1);
        engine.add_node(node2);

        assert_eq!(engine.nodes.read().unwrap().len(), 2);
    }

    #[test]
    fn test_query_empty_engine() {
        let engine = FederatedQueryEngine::new(FederationConfig::default());

        let results = engine.query(&[1.0, 0.0, 0.0]).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_query_single_node() {
        let engine = FederatedQueryEngine::new(FederationConfig::default());

        let node: Arc<dyn NodeClient> = Arc::new(MockNodeClient::new("node1", vec![
            make_result("doc1", 0.9, "hash1"),
            make_result("doc2", 0.8, "hash2"),
            make_result("doc3", 0.7, "hash3"),
        ]));
        engine.add_node(node);

        let results = engine.query(&[1.0, 0.0, 0.0]).unwrap();
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].id, "doc1");
    }

    #[test]
    fn test_strategy_sum() {
        let mut config = FederationConfig::default();
        config.strategy = AggregationStrategy::Sum;
        let engine = FederatedQueryEngine::new(config);

        let node1: Arc<dyn NodeClient> = Arc::new(MockNodeClient::new("node1", vec![
            make_result("doc1", 0.9, "hash1"),
        ]));
        let node2: Arc<dyn NodeClient> = Arc::new(MockNodeClient::new("node2", vec![
            make_result("doc1b", 0.8, "hash1"), // Same hash
        ]));
        engine.add_node(node1);
        engine.add_node(node2);

        let results = engine.query(&[1.0, 0.0, 0.0]).unwrap();
        assert_eq!(results.len(), 1);

        // SUM: 0.9 + 0.8 = 1.7
        assert!((results[0].combined_score - 1.7).abs() < 0.001);
        assert_eq!(results[0].source_count, 2);
    }

    #[test]
    fn test_strategy_max() {
        let mut config = FederationConfig::default();
        config.strategy = AggregationStrategy::Max;
        let engine = FederatedQueryEngine::new(config);

        let node1: Arc<dyn NodeClient> = Arc::new(MockNodeClient::new("node1", vec![
            make_result("doc1", 0.9, "hash1"),
        ]));
        let node2: Arc<dyn NodeClient> = Arc::new(MockNodeClient::new("node2", vec![
            make_result("doc1b", 0.8, "hash1"),
        ]));
        engine.add_node(node1);
        engine.add_node(node2);

        let results = engine.query(&[1.0, 0.0, 0.0]).unwrap();
        assert_eq!(results.len(), 1);

        // MAX: max(0.9, 0.8) = 0.9
        assert!((results[0].combined_score - 0.9).abs() < 0.001);
    }

    #[test]
    fn test_strategy_diversity() {
        let mut config = FederationConfig::default();
        config.strategy = AggregationStrategy::Diversity;
        let engine = FederatedQueryEngine::new(config);

        let node1: Arc<dyn NodeClient> = Arc::new(MockNodeClient::new("node1", vec![
            make_result("doc1", 0.9, "hash1"),
            make_result("doc1dup", 0.85, "hash1"), // Same hash, same source
        ]));
        let node2: Arc<dyn NodeClient> = Arc::new(MockNodeClient::new("node2", vec![
            make_result("doc1alt", 0.8, "hash1"), // Same hash, different source
        ]));
        engine.add_node(node1);
        engine.add_node(node2);

        let results = engine.query(&[1.0, 0.0, 0.0]).unwrap();
        assert_eq!(results.len(), 1);

        // DIVERSITY: 0.9 (first) + 0.8 (different source) = 1.7
        assert!((results[0].combined_score - 1.7).abs() < 0.001);
        assert_eq!(results[0].source_count, 2);
    }

    #[test]
    fn test_consensus_threshold() {
        let mut config = FederationConfig::default();
        config.consensus_threshold = 2;
        let engine = FederatedQueryEngine::new(config);

        let node1: Arc<dyn NodeClient> = Arc::new(MockNodeClient::new("node1", vec![
            make_result("doc1", 0.9, "hash1"),
            make_result("doc2", 0.7, "hash2"), // Only node1
        ]));
        let node2: Arc<dyn NodeClient> = Arc::new(MockNodeClient::new("node2", vec![
            make_result("doc1b", 0.8, "hash1"), // Same as doc1
            make_result("doc3", 0.6, "hash3"),  // Only node2
        ]));
        engine.add_node(node1);
        engine.add_node(node2);

        let results = engine.query(&[1.0, 0.0, 0.0]).unwrap();

        // Only hash1 appears in both
        assert_eq!(results.len(), 1);
        assert!(results[0].source_count >= 2);
    }

    #[test]
    fn test_federation_k() {
        let mut config = FederationConfig::default();
        config.federation_k = 2;
        let engine = FederatedQueryEngine::new(config);

        for i in 0..4 {
            let id = format!("node{}", i);
            let hash = format!("hash{}", i);
            let node: Arc<dyn NodeClient> = Arc::new(MockNodeClient::new(&id, vec![
                make_result(&format!("doc{}", i), 0.5, &hash),
            ]));
            engine.add_node(node);
        }

        let results = engine.query(&[1.0, 0.0, 0.0]).unwrap();

        // Only first 2 nodes queried
        assert!(results.len() <= 2);
    }

    #[test]
    fn test_set_strategy() {
        let engine = FederatedQueryEngine::new(FederationConfig::default());

        engine.set_strategy(AggregationStrategy::Max);
        assert_eq!(engine.config.read().unwrap().strategy, AggregationStrategy::Max);

        engine.set_strategy(AggregationStrategy::Diversity);
        assert_eq!(engine.config.read().unwrap().strategy, AggregationStrategy::Diversity);
    }

    #[test]
    fn test_set_federation_k() {
        let engine = FederatedQueryEngine::new(FederationConfig::default());

        engine.set_federation_k(5);
        assert_eq!(engine.config.read().unwrap().federation_k, 5);
    }

    #[test]
    fn test_result_sorting() {
        let engine = FederatedQueryEngine::new(FederationConfig::default());

        let node: Arc<dyn NodeClient> = Arc::new(MockNodeClient::new("node1", vec![
            make_result("doc3", 0.3, "hash3"),
            make_result("doc1", 0.9, "hash1"),
            make_result("doc2", 0.6, "hash2"),
        ]));
        engine.add_node(node);

        let results = engine.query(&[1.0, 0.0, 0.0]).unwrap();

        // Should be sorted descending by score
        for i in 1..results.len() {
            assert!(results[i].combined_score <= results[i - 1].combined_score);
        }
    }
}
