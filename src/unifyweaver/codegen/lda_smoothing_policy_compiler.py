"""
LDA Smoothing Policy Compiler

Compiles the declarative Prolog smoothing policy (lda_smoothing_policy.pl)
to Python, Go, and Rust target code.

The policy defines rules for selecting smoothing techniques based on:
- Cluster count at each node
- Depth in the tree
- Data quality (avg pairs per cluster)
- Distinguishability (cluster separation)

Usage:
    python lda_smoothing_policy_compiler.py --target python --output smoothing_policy.py
    python lda_smoothing_policy_compiler.py --target go --output smoothing_policy.go
    python lda_smoothing_policy_compiler.py --target rust --output smoothing_policy.rs
"""

import argparse
from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path


# =============================================================================
# Policy Configuration (extracted from lda_smoothing_policy.pl)
# =============================================================================

@dataclass
class PolicyConfig:
    """Configuration constants from the Prolog policy."""
    fft_threshold: int = 30          # Minimum clusters for FFT
    basis_sweet_spot_min: int = 10   # Min clusters for basis methods
    basis_sweet_spot_max: int = 50   # Max clusters for basis methods
    distinguish_threshold: float = 0.3  # Similarity threshold for distinguishable
    max_recursion_depth: int = 4     # Maximum refinement depth


# =============================================================================
# Python Code Generator
# =============================================================================

def generate_python(config: PolicyConfig) -> str:
    """Generate Python implementation of smoothing policy."""
    return f'''"""
LDA Smoothing Policy - Python Implementation

Generated from lda_smoothing_policy.pl by UnifyWeaver.
Do not edit manually - regenerate from source.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum


class SmoothingTechnique(Enum):
    """Available smoothing techniques."""
    FFT = "fft"
    BASIS_K4 = "basis_k4"
    BASIS_K8 = "basis_k8"
    BASIS_K16 = "basis_k16"
    BASELINE = "baseline"


@dataclass
class NodeInfo:
    """Information about a node in the smoothing tree."""
    node_id: str
    cluster_count: int
    total_pairs: int
    depth: int
    avg_pairs: float
    similarity_score: float = 0.5  # Default: moderately confusable


@dataclass
class SmoothingAction:
    """An action in the smoothing plan."""
    technique: SmoothingTechnique
    node_id: str


# Policy constants
FFT_THRESHOLD = {config.fft_threshold}
BASIS_SWEET_SPOT = ({config.basis_sweet_spot_min}, {config.basis_sweet_spot_max})
DISTINGUISH_THRESHOLD = {config.distinguish_threshold}
MAX_RECURSION_DEPTH = {config.max_recursion_depth}


def clusters_distinguishable(node: NodeInfo) -> bool:
    """Check if clusters within this node are well-separated after projection.

    Low similarity score means clusters are distinct and don't need refinement.
    """
    return node.similarity_score < DISTINGUISH_THRESHOLD


def refinement_needed(node: NodeInfo) -> bool:
    """Check if this node would benefit from further refinement.

    Refinement is needed when:
    - Enough clusters to potentially confuse (>10)
    - Not too deep in the tree (<4)
    - Clusters still too similar (>0.7 similarity)
    """
    return (
        node.cluster_count > 10 and
        node.depth < MAX_RECURSION_DEPTH and
        node.similarity_score > 0.7
    )


def sufficient_data(node: NodeInfo, technique: SmoothingTechnique) -> bool:
    """Check if node has enough data for the technique to be meaningful."""
    min_clusters, max_clusters = {{
        SmoothingTechnique.FFT: (10, 100000),
        SmoothingTechnique.BASIS_K4: (5, 500),
        SmoothingTechnique.BASIS_K8: (10, 200),
        SmoothingTechnique.BASIS_K16: (20, 100),
        SmoothingTechnique.BASELINE: (1, 100000),
    }}.get(technique, (1, 100000))

    return (
        min_clusters <= node.cluster_count <= max_clusters and
        node.avg_pairs >= 1.0
    )


def recommended_technique(node: NodeInfo) -> SmoothingTechnique:
    """Recommend a smoothing technique based on node properties.

    Rules (in priority order):
    1. Large clusters (>=30) at shallow depths (<3) -> FFT
    2. Medium clusters (10-50) at depth >=1 with good data -> basis_k8
    3. Smaller clusters (5-20) at depth >=2 -> basis_k4
    4. Very small clusters (<5) -> baseline
    5. Large clusters at deep levels (>=50, depth >=3) -> FFT
    6. Fallback -> basis_k4
    """
    c = node.cluster_count
    d = node.depth
    avg = node.avg_pairs

    # Rule 1: Large clusters at shallow depths -> FFT
    if c >= FFT_THRESHOLD and d < 3:
        return SmoothingTechnique.FFT

    # Rule 2: Medium clusters -> basis_k8
    if BASIS_SWEET_SPOT[0] <= c <= BASIS_SWEET_SPOT[1] and d >= 1 and avg >= 2:
        return SmoothingTechnique.BASIS_K8

    # Rule 3: Smaller clusters at deeper levels -> basis_k4
    if 5 <= c < 20 and d >= 2 and avg >= 2:
        return SmoothingTechnique.BASIS_K4

    # Rule 4: Very small clusters -> baseline
    if c < 5:
        return SmoothingTechnique.BASELINE

    # Rule 5: Large clusters at deep levels -> FFT
    if c >= 50 and d >= 3:
        return SmoothingTechnique.FFT

    # Rule 6: Fallback
    if c >= 5:
        return SmoothingTechnique.BASIS_K4

    return SmoothingTechnique.BASELINE


def generate_smoothing_plan(
    root: NodeInfo,
    children: Dict[str, List[NodeInfo]]
) -> List[SmoothingAction]:
    """Generate a complete smoothing plan for the tree.

    Args:
        root: Root node info
        children: Mapping from parent node_id to list of child NodeInfo

    Returns:
        List of SmoothingAction in execution order
    """
    plan = []
    _plan_recursive(root, children, plan)
    return plan


def _plan_recursive(
    node: NodeInfo,
    children: Dict[str, List[NodeInfo]],
    plan: List[SmoothingAction]
) -> None:
    """Recursively build the plan."""
    technique = recommended_technique(node)
    plan.append(SmoothingAction(technique=technique, node_id=node.node_id))

    if refinement_needed(node) and node.node_id in children:
        for child in children[node.node_id]:
            if not clusters_distinguishable(child):
                _plan_recursive(child, children, plan)


def estimate_cost_ms(plan: List[SmoothingAction], nodes: Dict[str, NodeInfo]) -> float:
    """Estimate total training cost in milliseconds.

    Cost estimates per cluster:
    - FFT: ~0.4ms
    - basis_k4: ~10ms
    - basis_k8: ~15ms
    - baseline: ~0.02ms
    """
    cost_per_cluster = {{
        SmoothingTechnique.FFT: 0.4,
        SmoothingTechnique.BASIS_K4: 10.0,
        SmoothingTechnique.BASIS_K8: 15.0,
        SmoothingTechnique.BASIS_K16: 25.0,
        SmoothingTechnique.BASELINE: 0.02,
    }}

    total = 0.0
    for action in plan:
        if action.node_id in nodes:
            c = nodes[action.node_id].cluster_count
            total += c * cost_per_cluster.get(action.technique, 1.0)
    return total


def plan_summary(plan: List[SmoothingAction]) -> Dict:
    """Get a summary of the plan."""
    from collections import Counter

    tech_counts = Counter(a.technique.value for a in plan)

    return {{
        'num_actions': len(plan),
        'technique_counts': dict(tech_counts),
    }}
'''


# =============================================================================
# Go Code Generator
# =============================================================================

def generate_go(config: PolicyConfig) -> str:
    """Generate Go implementation of smoothing policy."""
    return f'''// LDA Smoothing Policy - Go Implementation
//
// Generated from lda_smoothing_policy.pl by UnifyWeaver.
// Do not edit manually - regenerate from source.

package smoothing

// SmoothingTechnique represents available smoothing techniques
type SmoothingTechnique string

const (
    TechniqueFFT      SmoothingTechnique = "fft"
    TechniqueBasisK4  SmoothingTechnique = "basis_k4"
    TechniqueBasisK8  SmoothingTechnique = "basis_k8"
    TechniqueBasisK16 SmoothingTechnique = "basis_k16"
    TechniqueBaseline SmoothingTechnique = "baseline"
)

// Policy constants
const (
    FFTThreshold        = {config.fft_threshold}
    BasisSweetSpotMin   = {config.basis_sweet_spot_min}
    BasisSweetSpotMax   = {config.basis_sweet_spot_max}
    DistinguishThreshold = {config.distinguish_threshold}
    MaxRecursionDepth   = {config.max_recursion_depth}
)

// NodeInfo contains information about a node in the smoothing tree
type NodeInfo struct {{
    NodeID          string
    ClusterCount    int
    TotalPairs      int
    Depth           int
    AvgPairs        float64
    SimilarityScore float64
}}

// SmoothingAction represents an action in the smoothing plan
type SmoothingAction struct {{
    Technique SmoothingTechnique
    NodeID    string
}}

// ClustersDistinguishable checks if clusters are well-separated
func ClustersDistinguishable(node NodeInfo) bool {{
    return node.SimilarityScore < DistinguishThreshold
}}

// RefinementNeeded checks if the node needs further refinement
func RefinementNeeded(node NodeInfo) bool {{
    return node.ClusterCount > 10 &&
        node.Depth < MaxRecursionDepth &&
        node.SimilarityScore > 0.7
}}

// SufficientData checks if node has enough data for the technique
func SufficientData(node NodeInfo, technique SmoothingTechnique) bool {{
    var minC, maxC int
    switch technique {{
    case TechniqueFFT:
        minC, maxC = 10, 100000
    case TechniqueBasisK4:
        minC, maxC = 5, 500
    case TechniqueBasisK8:
        minC, maxC = 10, 200
    case TechniqueBasisK16:
        minC, maxC = 20, 100
    default:
        minC, maxC = 1, 100000
    }}
    return node.ClusterCount >= minC && node.ClusterCount <= maxC && node.AvgPairs >= 1.0
}}

// RecommendedTechnique returns the recommended smoothing technique for a node
func RecommendedTechnique(node NodeInfo) SmoothingTechnique {{
    c := node.ClusterCount
    d := node.Depth
    avg := node.AvgPairs

    // Rule 1: Large clusters at shallow depths -> FFT
    if c >= FFTThreshold && d < 3 {{
        return TechniqueFFT
    }}

    // Rule 2: Medium clusters -> basis_k8
    if c >= BasisSweetSpotMin && c <= BasisSweetSpotMax && d >= 1 && avg >= 2 {{
        return TechniqueBasisK8
    }}

    // Rule 3: Smaller clusters at deeper levels -> basis_k4
    if c >= 5 && c < 20 && d >= 2 && avg >= 2 {{
        return TechniqueBasisK4
    }}

    // Rule 4: Very small clusters -> baseline
    if c < 5 {{
        return TechniqueBaseline
    }}

    // Rule 5: Large clusters at deep levels -> FFT
    if c >= 50 && d >= 3 {{
        return TechniqueFFT
    }}

    // Rule 6: Fallback
    if c >= 5 {{
        return TechniqueBasisK4
    }}

    return TechniqueBaseline
}}

// GenerateSmoothingPlan generates a complete smoothing plan for the tree
func GenerateSmoothingPlan(root NodeInfo, children map[string][]NodeInfo) []SmoothingAction {{
    plan := make([]SmoothingAction, 0)
    planRecursive(root, children, &plan)
    return plan
}}

func planRecursive(node NodeInfo, children map[string][]NodeInfo, plan *[]SmoothingAction) {{
    technique := RecommendedTechnique(node)
    *plan = append(*plan, SmoothingAction{{Technique: technique, NodeID: node.NodeID}})

    if RefinementNeeded(node) {{
        if nodeChildren, ok := children[node.NodeID]; ok {{
            for _, child := range nodeChildren {{
                if !ClustersDistinguishable(child) {{
                    planRecursive(child, children, plan)
                }}
            }}
        }}
    }}
}}

// EstimateCostMs estimates total training cost in milliseconds
func EstimateCostMs(plan []SmoothingAction, nodes map[string]NodeInfo) float64 {{
    costPerCluster := map[SmoothingTechnique]float64{{
        TechniqueFFT:      0.4,
        TechniqueBasisK4:  10.0,
        TechniqueBasisK8:  15.0,
        TechniqueBasisK16: 25.0,
        TechniqueBaseline: 0.02,
    }}

    total := 0.0
    for _, action := range plan {{
        if node, ok := nodes[action.NodeID]; ok {{
            cost, exists := costPerCluster[action.Technique]
            if !exists {{
                cost = 1.0
            }}
            total += float64(node.ClusterCount) * cost
        }}
    }}
    return total
}}
'''


# =============================================================================
# Rust Code Generator
# =============================================================================

def generate_rust(config: PolicyConfig) -> str:
    """Generate Rust implementation of smoothing policy."""
    return f'''//! LDA Smoothing Policy - Rust Implementation
//!
//! Generated from lda_smoothing_policy.pl by UnifyWeaver.
//! Do not edit manually - regenerate from source.

use std::collections::HashMap;

/// Available smoothing techniques
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SmoothingTechnique {{
    Fft,
    BasisK4,
    BasisK8,
    BasisK16,
    Baseline,
}}

impl SmoothingTechnique {{
    pub fn as_str(&self) -> &'static str {{
        match self {{
            SmoothingTechnique::Fft => "fft",
            SmoothingTechnique::BasisK4 => "basis_k4",
            SmoothingTechnique::BasisK8 => "basis_k8",
            SmoothingTechnique::BasisK16 => "basis_k16",
            SmoothingTechnique::Baseline => "baseline",
        }}
    }}
}}

/// Policy constants
pub const FFT_THRESHOLD: usize = {config.fft_threshold};
pub const BASIS_SWEET_SPOT: (usize, usize) = ({config.basis_sweet_spot_min}, {config.basis_sweet_spot_max});
pub const DISTINGUISH_THRESHOLD: f64 = {config.distinguish_threshold};
pub const MAX_RECURSION_DEPTH: usize = {config.max_recursion_depth};

/// Information about a node in the smoothing tree
#[derive(Debug, Clone)]
pub struct NodeInfo {{
    pub node_id: String,
    pub cluster_count: usize,
    pub total_pairs: usize,
    pub depth: usize,
    pub avg_pairs: f64,
    pub similarity_score: f64,
}}

impl NodeInfo {{
    pub fn new(node_id: String, cluster_count: usize, total_pairs: usize, depth: usize) -> Self {{
        let avg_pairs = if cluster_count > 0 {{
            total_pairs as f64 / cluster_count as f64
        }} else {{
            0.0
        }};
        Self {{
            node_id,
            cluster_count,
            total_pairs,
            depth,
            avg_pairs,
            similarity_score: 0.5, // Default: moderately confusable
        }}
    }}

    pub fn with_similarity(mut self, score: f64) -> Self {{
        self.similarity_score = score;
        self
    }}
}}

/// An action in the smoothing plan
#[derive(Debug, Clone)]
pub struct SmoothingAction {{
    pub technique: SmoothingTechnique,
    pub node_id: String,
}}

/// Check if clusters within this node are well-separated
pub fn clusters_distinguishable(node: &NodeInfo) -> bool {{
    node.similarity_score < DISTINGUISH_THRESHOLD
}}

/// Check if this node would benefit from further refinement
pub fn refinement_needed(node: &NodeInfo) -> bool {{
    node.cluster_count > 10
        && node.depth < MAX_RECURSION_DEPTH
        && node.similarity_score > 0.7
}}

/// Check if node has enough data for the technique
pub fn sufficient_data(node: &NodeInfo, technique: SmoothingTechnique) -> bool {{
    let (min_c, max_c) = match technique {{
        SmoothingTechnique::Fft => (10, 100000),
        SmoothingTechnique::BasisK4 => (5, 500),
        SmoothingTechnique::BasisK8 => (10, 200),
        SmoothingTechnique::BasisK16 => (20, 100),
        SmoothingTechnique::Baseline => (1, 100000),
    }};
    node.cluster_count >= min_c && node.cluster_count <= max_c && node.avg_pairs >= 1.0
}}

/// Recommend a smoothing technique based on node properties
pub fn recommended_technique(node: &NodeInfo) -> SmoothingTechnique {{
    let c = node.cluster_count;
    let d = node.depth;
    let avg = node.avg_pairs;

    // Rule 1: Large clusters at shallow depths -> FFT
    if c >= FFT_THRESHOLD && d < 3 {{
        return SmoothingTechnique::Fft;
    }}

    // Rule 2: Medium clusters -> basis_k8
    if c >= BASIS_SWEET_SPOT.0 && c <= BASIS_SWEET_SPOT.1 && d >= 1 && avg >= 2.0 {{
        return SmoothingTechnique::BasisK8;
    }}

    // Rule 3: Smaller clusters at deeper levels -> basis_k4
    if c >= 5 && c < 20 && d >= 2 && avg >= 2.0 {{
        return SmoothingTechnique::BasisK4;
    }}

    // Rule 4: Very small clusters -> baseline
    if c < 5 {{
        return SmoothingTechnique::Baseline;
    }}

    // Rule 5: Large clusters at deep levels -> FFT
    if c >= 50 && d >= 3 {{
        return SmoothingTechnique::Fft;
    }}

    // Rule 6: Fallback
    if c >= 5 {{
        return SmoothingTechnique::BasisK4;
    }}

    SmoothingTechnique::Baseline
}}

/// Generate a complete smoothing plan for the tree
pub fn generate_smoothing_plan(
    root: &NodeInfo,
    children: &HashMap<String, Vec<NodeInfo>>,
) -> Vec<SmoothingAction> {{
    let mut plan = Vec::new();
    plan_recursive(root, children, &mut plan);
    plan
}}

fn plan_recursive(
    node: &NodeInfo,
    children: &HashMap<String, Vec<NodeInfo>>,
    plan: &mut Vec<SmoothingAction>,
) {{
    let technique = recommended_technique(node);
    plan.push(SmoothingAction {{
        technique,
        node_id: node.node_id.clone(),
    }});

    if refinement_needed(node) {{
        if let Some(node_children) = children.get(&node.node_id) {{
            for child in node_children {{
                if !clusters_distinguishable(child) {{
                    plan_recursive(child, children, plan);
                }}
            }}
        }}
    }}
}}

/// Estimate total training cost in milliseconds
pub fn estimate_cost_ms(plan: &[SmoothingAction], nodes: &HashMap<String, NodeInfo>) -> f64 {{
    let cost_per_cluster = |t: SmoothingTechnique| -> f64 {{
        match t {{
            SmoothingTechnique::Fft => 0.4,
            SmoothingTechnique::BasisK4 => 10.0,
            SmoothingTechnique::BasisK8 => 15.0,
            SmoothingTechnique::BasisK16 => 25.0,
            SmoothingTechnique::Baseline => 0.02,
        }}
    }};

    plan.iter()
        .filter_map(|action| nodes.get(&action.node_id))
        .map(|node| node.cluster_count as f64 * cost_per_cluster(recommended_technique(node)))
        .sum()
}}

#[cfg(test)]
mod tests {{
    use super::*;

    #[test]
    fn test_recommended_technique_large_shallow() {{
        let node = NodeInfo::new("root".to_string(), 100, 300, 0);
        assert_eq!(recommended_technique(&node), SmoothingTechnique::Fft);
    }}

    #[test]
    fn test_recommended_technique_medium() {{
        let node = NodeInfo::new("seg1".to_string(), 25, 75, 1);
        assert_eq!(recommended_technique(&node), SmoothingTechnique::BasisK8);
    }}

    #[test]
    fn test_recommended_technique_small() {{
        let node = NodeInfo::new("leaf".to_string(), 3, 6, 3);
        assert_eq!(recommended_technique(&node), SmoothingTechnique::Baseline);
    }}

    #[test]
    fn test_clusters_distinguishable() {{
        let distinct = NodeInfo::new("a".to_string(), 10, 30, 1).with_similarity(0.2);
        let confusable = NodeInfo::new("b".to_string(), 10, 30, 1).with_similarity(0.8);

        assert!(clusters_distinguishable(&distinct));
        assert!(!clusters_distinguishable(&confusable));
    }}

    #[test]
    fn test_generate_plan() {{
        let root = NodeInfo::new("root".to_string(), 50, 150, 0).with_similarity(0.8);
        let children: HashMap<String, Vec<NodeInfo>> = HashMap::new();

        let plan = generate_smoothing_plan(&root, &children);
        assert_eq!(plan.len(), 1);
        assert_eq!(plan[0].technique, SmoothingTechnique::Fft);
    }}
}}
'''


# =============================================================================
# Main Compiler
# =============================================================================

def compile_policy(target: str, output_path: Optional[str] = None) -> str:
    """Compile the smoothing policy to target language.

    Args:
        target: Target language ('python', 'go', 'rust')
        output_path: Optional path to write output

    Returns:
        Generated code as string
    """
    config = PolicyConfig()

    generators = {
        'python': generate_python,
        'go': generate_go,
        'rust': generate_rust,
    }

    if target not in generators:
        raise ValueError(f"Unknown target: {target}. Available: {list(generators.keys())}")

    code = generators[target](config)

    if output_path:
        Path(output_path).write_text(code)
        print(f"Generated {target} code: {output_path}")

    return code


def main():
    parser = argparse.ArgumentParser(
        description='Compile LDA smoothing policy to target language'
    )
    parser.add_argument(
        '--target', '-t',
        choices=['python', 'go', 'rust', 'all'],
        default='all',
        help='Target language (default: all)'
    )
    parser.add_argument(
        '--output-dir', '-o',
        default='.',
        help='Output directory (default: current)'
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    targets = ['python', 'go', 'rust'] if args.target == 'all' else [args.target]

    extensions = {'python': '.py', 'go': '.go', 'rust': '.rs'}

    for target in targets:
        output_path = output_dir / f"smoothing_policy{extensions[target]}"
        compile_policy(target, str(output_path))


if __name__ == '__main__':
    main()
