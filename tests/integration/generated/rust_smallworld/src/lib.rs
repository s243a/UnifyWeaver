// KG Topology Phase 7: Proper Small-World Network
// Generated from Prolog service definition
//
// Network structure enables true Kleinberg routing with O(log^2 n) path length.
// k_local = 10 nearest neighbors, k_long = 5 long-range shortcuts

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};
use rand::Rng;

// Configuration constants
pub const K_LOCAL: usize = 10;
pub const K_LONG: usize = 5;
pub const ALPHA: f64 = 2.0;

/// Neighbor with precomputed angle for binary search
#[derive(Clone, Debug)]
pub struct Neighbor {
    pub node_id: String,
    pub angle: f64, // Cosine-based angle
    pub is_long: bool,
}

/// Node in the small-world network
pub struct SmallWorldNode {
    pub id: String,
    pub centroid: Vec<f32>,
    pub neighbors: Vec<Neighbor>, // Sorted by angle
}

/// Proper small-world network with k_local + k_long structure
pub struct SmallWorldNetwork {
    pub nodes: RwLock<HashMap<String, Arc<RwLock<SmallWorldNode>>>>,
    pub k_local: usize,
    pub k_long: usize,
    pub alpha: f64,
}

impl SmallWorldNetwork {
    pub fn new() -> Self {
        Self {
            nodes: RwLock::new(HashMap::new()),
            k_local: K_LOCAL,
            k_long: K_LONG,
            alpha: ALPHA,
        }
    }

    /// Add a node and establish connections
    pub fn add_node(&self, node_id: &str, centroid: Vec<f32>) {
        let node = Arc::new(RwLock::new(SmallWorldNode {
            id: node_id.to_string(),
            centroid: centroid.clone(),
            neighbors: Vec::new(),
        }));

        {
            let mut nodes = self.nodes.write().unwrap();
            nodes.insert(node_id.to_string(), node.clone());
        }

        if self.nodes.read().unwrap().len() > 1 {
            self.connect_node(node_id, &centroid);
        }
    }

    /// Establish k_local + k_long connections
    fn connect_node(&self, node_id: &str, centroid: &[f32]) {
        let nodes = self.nodes.read().unwrap();

        // Collect similarities to all other nodes
        let mut others: Vec<(String, f64)> = nodes
            .iter()
            .filter(|(id, _)| *id != node_id)
            .map(|(id, n)| {
                let other = n.read().unwrap();
                (id.clone(), cosine_similarity(centroid, &other.centroid))
            })
            .collect();

        // Sort by similarity (descending)
        others.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut neighbors = Vec::new();

        // Add k_local nearest neighbors
        let local_count = self.k_local.min(others.len());
        for (id, _) in others.iter().take(local_count) {
            let other = nodes.get(id).unwrap().read().unwrap();
            let angle = compute_cosine_angle(centroid, &other.centroid);
            neighbors.push(Neighbor {
                node_id: id.clone(),
                angle,
                is_long: false,
            });
        }

        // Add k_long shortcuts using distance-weighted probability
        if others.len() > self.k_local {
            let remaining = &others[local_count..];
            let long_count = self.k_long.min(remaining.len());

            // Compute weights: P(v) ~ 1/distance^alpha
            let weights: Vec<f64> = remaining
                .iter()
                .map(|(_, sim)| {
                    let distance = (1.0 - sim).max(0.001);
                    1.0 / distance.powf(self.alpha)
                })
                .collect();
            let total_weight: f64 = weights.iter().sum();

            // Sample k_long shortcuts
            let mut selected = HashSet::new();
            let mut rng = rand::thread_rng();
            while selected.len() < long_count {
                let r: f64 = rng.gen::<f64>() * total_weight;
                let mut cumulative = 0.0;
                for (i, w) in weights.iter().enumerate() {
                    cumulative += w;
                    if r <= cumulative && !selected.contains(&i) {
                        selected.insert(i);
                        let (id, _) = &remaining[i];
                        let other = nodes.get(id).unwrap().read().unwrap();
                        let angle = compute_cosine_angle(centroid, &other.centroid);
                        neighbors.push(Neighbor {
                            node_id: id.clone(),
                            angle,
                            is_long: true,
                        });
                        break;
                    }
                }
            }
        }

        // Sort neighbors by angle for binary search
        neighbors.sort_by(|a, b| a.angle.partial_cmp(&b.angle).unwrap_or(std::cmp::Ordering::Equal));

        // Update node
        drop(nodes);
        let nodes = self.nodes.read().unwrap();
        if let Some(node) = nodes.get(node_id) {
            node.write().unwrap().neighbors = neighbors;
        }
    }

    /// Route to target using greedy routing on small-world structure
    pub fn route_to_target(&self, query: &[f32], max_hops: usize) -> Vec<String> {
        let nodes = self.nodes.read().unwrap();
        if nodes.is_empty() {
            return Vec::new();
        }

        // Start from first node
        let mut current_id = nodes.keys().next().unwrap().clone();
        let mut path = vec![current_id.clone()];
        let mut visited: HashSet<String> = HashSet::new();
        visited.insert(current_id.clone());

        for _ in 0..max_hops {
            let current = nodes.get(&current_id).unwrap().read().unwrap();
            let current_sim = cosine_similarity(&current.centroid, query);

            // Find best neighbor using binary search
            let query_angle = compute_cosine_angle(&current.centroid, query);
            let neighbors = &current.neighbors;

            if neighbors.is_empty() {
                break;
            }

            // Binary search for closest angle
            let idx = neighbors.partition_point(|n| n.angle < query_angle);

            // Check neighbors around index
            let mut best_id: Option<String> = None;
            let mut best_sim: f64 = -1.0;

            for check_idx in [
                idx.saturating_sub(1),
                idx,
                (idx + 1).min(neighbors.len().saturating_sub(1)),
            ] {
                if check_idx < neighbors.len() {
                    let nb = &neighbors[check_idx];
                    if !visited.contains(&nb.node_id) {
                        if let Some(neighbor) = nodes.get(&nb.node_id) {
                            let neighbor = neighbor.read().unwrap();
                            let sim = cosine_similarity(&neighbor.centroid, query);
                            if sim > best_sim {
                                best_sim = sim;
                                best_id = Some(nb.node_id.clone());
                            }
                        }
                    }
                }
            }

            match best_id {
                Some(id) if best_sim > current_sim => {
                    visited.insert(id.clone());
                    path.push(id.clone());
                    current_id = id;
                }
                _ => break,
            }
        }

        path
    }
}

impl Default for SmallWorldNetwork {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute cosine angle using full cosine similarity (not 2D projection)
pub fn compute_cosine_angle(a: &[f32], b: &[f32]) -> f64 {
    let sim = cosine_similarity(a, b);
    // Clamp for numerical stability
    let sim = sim.clamp(-1.0, 1.0);
    sim.acos()
}

/// Compute cosine similarity between two vectors
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    let mut dot: f64 = 0.0;
    let mut norm_a: f64 = 0.0;
    let mut norm_b: f64 = 0.0;

    for (ai, bi) in a.iter().zip(b.iter()) {
        dot += (*ai as f64) * (*bi as f64);
        norm_a += (*ai as f64) * (*ai as f64);
        norm_b += (*bi as f64) * (*bi as f64);
    }

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot / (norm_a.sqrt() * norm_b.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_network() {
        let n = SmallWorldNetwork::new();
        assert_eq!(n.k_local, K_LOCAL);
        assert_eq!(n.k_long, K_LONG);
        assert!((n.alpha - ALPHA).abs() < 0.001);
    }

    #[test]
    fn test_add_node() {
        let n = SmallWorldNetwork::new();

        n.add_node("node1", vec![1.0, 0.0, 0.0]);
        assert_eq!(n.nodes.read().unwrap().len(), 1);

        n.add_node("node2", vec![0.9, 0.1, 0.0]);
        assert_eq!(n.nodes.read().unwrap().len(), 2);

        // Check connections were created
        let nodes = n.nodes.read().unwrap();
        let node2 = nodes.get("node2").unwrap().read().unwrap();
        assert!(!node2.neighbors.is_empty(), "node2 should have neighbors");
    }

    #[test]
    fn test_cosine_similarity() {
        // Identical vectors
        let sim = cosine_similarity(&[1.0, 0.0, 0.0], &[1.0, 0.0, 0.0]);
        assert!((sim - 1.0).abs() < 0.001);

        // Orthogonal vectors
        let sim = cosine_similarity(&[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0]);
        assert!(sim.abs() < 0.001);

        // Opposite vectors
        let sim = cosine_similarity(&[1.0, 0.0, 0.0], &[-1.0, 0.0, 0.0]);
        assert!((sim - (-1.0)).abs() < 0.001);
    }

    #[test]
    fn test_compute_cosine_angle() {
        // Identical vectors (0 angle)
        let angle = compute_cosine_angle(&[1.0, 0.0, 0.0], &[1.0, 0.0, 0.0]);
        assert!(angle.abs() < 0.01);

        // Orthogonal vectors (pi/2)
        let angle = compute_cosine_angle(&[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0]);
        assert!((angle - std::f64::consts::FRAC_PI_2).abs() < 0.01);
    }

    #[test]
    fn test_route_to_target() {
        let n = SmallWorldNetwork::new();

        // Build a small network
        n.add_node("n1", vec![1.0, 0.0, 0.0]);
        n.add_node("n2", vec![0.9, 0.1, 0.0]);
        n.add_node("n3", vec![0.8, 0.2, 0.0]);
        n.add_node("n4", vec![0.0, 1.0, 0.0]);
        n.add_node("n5", vec![0.1, 0.9, 0.0]);

        // Route towards a target
        let target = vec![1.0, 0.0, 0.0];
        let path = n.route_to_target(&target, 10);

        assert!(!path.is_empty(), "Path should not be empty");

        // Final node should be close to target
        let nodes = n.nodes.read().unwrap();
        let final_node = nodes.get(path.last().unwrap()).unwrap().read().unwrap();
        let final_sim = cosine_similarity(&final_node.centroid, &target);
        assert!(final_sim >= 0.8, "Final node similarity should be >= 0.8");
    }
}
