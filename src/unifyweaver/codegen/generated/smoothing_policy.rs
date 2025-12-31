//! LDA Smoothing Policy - Rust Implementation
//!
//! Generated from lda_smoothing_policy.pl by UnifyWeaver.
//! Do not edit manually - regenerate from source.

use std::collections::HashMap;

/// Available smoothing techniques
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SmoothingTechnique {
    Fft,
    BasisK4,
    BasisK8,
    BasisK16,
    Baseline,
}

impl SmoothingTechnique {
    pub fn as_str(&self) -> &'static str {
        match self {
            SmoothingTechnique::Fft => "fft",
            SmoothingTechnique::BasisK4 => "basis_k4",
            SmoothingTechnique::BasisK8 => "basis_k8",
            SmoothingTechnique::BasisK16 => "basis_k16",
            SmoothingTechnique::Baseline => "baseline",
        }
    }
}

/// Policy constants
pub const FFT_THRESHOLD: usize = 30;
pub const BASIS_SWEET_SPOT: (usize, usize) = (10, 50);
pub const DISTINGUISH_THRESHOLD: f64 = 0.3;
pub const MAX_RECURSION_DEPTH: usize = 4;

/// Information about a node in the smoothing tree
#[derive(Debug, Clone)]
pub struct NodeInfo {
    pub node_id: String,
    pub cluster_count: usize,
    pub total_pairs: usize,
    pub depth: usize,
    pub avg_pairs: f64,
    pub similarity_score: f64,
}

impl NodeInfo {
    pub fn new(node_id: String, cluster_count: usize, total_pairs: usize, depth: usize) -> Self {
        let avg_pairs = if cluster_count > 0 {
            total_pairs as f64 / cluster_count as f64
        } else {
            0.0
        };
        Self {
            node_id,
            cluster_count,
            total_pairs,
            depth,
            avg_pairs,
            similarity_score: 0.5, // Default: moderately confusable
        }
    }

    pub fn with_similarity(mut self, score: f64) -> Self {
        self.similarity_score = score;
        self
    }
}

/// An action in the smoothing plan
#[derive(Debug, Clone)]
pub struct SmoothingAction {
    pub technique: SmoothingTechnique,
    pub node_id: String,
}

/// Check if clusters within this node are well-separated
pub fn clusters_distinguishable(node: &NodeInfo) -> bool {
    node.similarity_score < DISTINGUISH_THRESHOLD
}

/// Check if this node would benefit from further refinement
pub fn refinement_needed(node: &NodeInfo) -> bool {
    node.cluster_count > 10
        && node.depth < MAX_RECURSION_DEPTH
        && node.similarity_score > 0.7
}

/// Check if node has enough data for the technique
pub fn sufficient_data(node: &NodeInfo, technique: SmoothingTechnique) -> bool {
    let (min_c, max_c) = match technique {
        SmoothingTechnique::Fft => (10, 100000),
        SmoothingTechnique::BasisK4 => (5, 500),
        SmoothingTechnique::BasisK8 => (10, 200),
        SmoothingTechnique::BasisK16 => (20, 100),
        SmoothingTechnique::Baseline => (1, 100000),
    };
    node.cluster_count >= min_c && node.cluster_count <= max_c && node.avg_pairs >= 1.0
}

/// Recommend a smoothing technique based on node properties
pub fn recommended_technique(node: &NodeInfo) -> SmoothingTechnique {
    let c = node.cluster_count;
    let d = node.depth;
    let avg = node.avg_pairs;

    // Rule 1: Large clusters at shallow depths -> FFT
    if c >= FFT_THRESHOLD && d < 3 {
        return SmoothingTechnique::Fft;
    }

    // Rule 2: Medium clusters -> basis_k8
    if c >= BASIS_SWEET_SPOT.0 && c <= BASIS_SWEET_SPOT.1 && d >= 1 && avg >= 2.0 {
        return SmoothingTechnique::BasisK8;
    }

    // Rule 3: Smaller clusters at deeper levels -> basis_k4
    if c >= 5 && c < 20 && d >= 2 && avg >= 2.0 {
        return SmoothingTechnique::BasisK4;
    }

    // Rule 4: Very small clusters -> baseline
    if c < 5 {
        return SmoothingTechnique::Baseline;
    }

    // Rule 5: Large clusters at deep levels -> FFT
    if c >= 50 && d >= 3 {
        return SmoothingTechnique::Fft;
    }

    // Rule 6: Fallback
    if c >= 5 {
        return SmoothingTechnique::BasisK4;
    }

    SmoothingTechnique::Baseline
}

/// Generate a complete smoothing plan for the tree
pub fn generate_smoothing_plan(
    root: &NodeInfo,
    children: &HashMap<String, Vec<NodeInfo>>,
) -> Vec<SmoothingAction> {
    let mut plan = Vec::new();
    plan_recursive(root, children, &mut plan);
    plan
}

fn plan_recursive(
    node: &NodeInfo,
    children: &HashMap<String, Vec<NodeInfo>>,
    plan: &mut Vec<SmoothingAction>,
) {
    let technique = recommended_technique(node);
    plan.push(SmoothingAction {
        technique,
        node_id: node.node_id.clone(),
    });

    if refinement_needed(node) {
        if let Some(node_children) = children.get(&node.node_id) {
            for child in node_children {
                if !clusters_distinguishable(child) {
                    plan_recursive(child, children, plan);
                }
            }
        }
    }
}

/// Estimate total training cost in milliseconds
pub fn estimate_cost_ms(plan: &[SmoothingAction], nodes: &HashMap<String, NodeInfo>) -> f64 {
    let cost_per_cluster = |t: SmoothingTechnique| -> f64 {
        match t {
            SmoothingTechnique::Fft => 0.4,
            SmoothingTechnique::BasisK4 => 10.0,
            SmoothingTechnique::BasisK8 => 15.0,
            SmoothingTechnique::BasisK16 => 25.0,
            SmoothingTechnique::Baseline => 0.02,
        }
    };

    plan.iter()
        .filter_map(|action| nodes.get(&action.node_id))
        .map(|node| node.cluster_count as f64 * cost_per_cluster(recommended_technique(node)))
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_recommended_technique_large_shallow() {
        let node = NodeInfo::new("root".to_string(), 100, 300, 0);
        assert_eq!(recommended_technique(&node), SmoothingTechnique::Fft);
    }

    #[test]
    fn test_recommended_technique_medium() {
        let node = NodeInfo::new("seg1".to_string(), 25, 75, 1);
        assert_eq!(recommended_technique(&node), SmoothingTechnique::BasisK8);
    }

    #[test]
    fn test_recommended_technique_small() {
        let node = NodeInfo::new("leaf".to_string(), 3, 6, 3);
        assert_eq!(recommended_technique(&node), SmoothingTechnique::Baseline);
    }

    #[test]
    fn test_clusters_distinguishable() {
        let distinct = NodeInfo::new("a".to_string(), 10, 30, 1).with_similarity(0.2);
        let confusable = NodeInfo::new("b".to_string(), 10, 30, 1).with_similarity(0.8);

        assert!(clusters_distinguishable(&distinct));
        assert!(!clusters_distinguishable(&confusable));
    }

    #[test]
    fn test_generate_plan() {
        let root = NodeInfo::new("root".to_string(), 50, 150, 0).with_similarity(0.8);
        let children: HashMap<String, Vec<NodeInfo>> = HashMap::new();

        let plan = generate_smoothing_plan(&root, &children);
        assert_eq!(plan.len(), 1);
        assert_eq!(plan[0].technique, SmoothingTechnique::Fft);
    }
}
