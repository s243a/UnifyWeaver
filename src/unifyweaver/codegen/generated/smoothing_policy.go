// LDA Smoothing Policy - Go Implementation
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
    FFTThreshold        = 30
    BasisSweetSpotMin   = 10
    BasisSweetSpotMax   = 50
    DistinguishThreshold = 0.3
    MaxRecursionDepth   = 4
)

// NodeInfo contains information about a node in the smoothing tree
type NodeInfo struct {
    NodeID          string
    ClusterCount    int
    TotalPairs      int
    Depth           int
    AvgPairs        float64
    SimilarityScore float64
}

// SmoothingAction represents an action in the smoothing plan
type SmoothingAction struct {
    Technique SmoothingTechnique
    NodeID    string
}

// ClustersDistinguishable checks if clusters are well-separated
func ClustersDistinguishable(node NodeInfo) bool {
    return node.SimilarityScore < DistinguishThreshold
}

// RefinementNeeded checks if the node needs further refinement
func RefinementNeeded(node NodeInfo) bool {
    return node.ClusterCount > 10 &&
        node.Depth < MaxRecursionDepth &&
        node.SimilarityScore > 0.7
}

// SufficientData checks if node has enough data for the technique
func SufficientData(node NodeInfo, technique SmoothingTechnique) bool {
    var minC, maxC int
    switch technique {
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
    }
    return node.ClusterCount >= minC && node.ClusterCount <= maxC && node.AvgPairs >= 1.0
}

// RecommendedTechnique returns the recommended smoothing technique for a node
func RecommendedTechnique(node NodeInfo) SmoothingTechnique {
    c := node.ClusterCount
    d := node.Depth
    avg := node.AvgPairs

    // Rule 1: Large clusters at shallow depths -> FFT
    if c >= FFTThreshold && d < 3 {
        return TechniqueFFT
    }

    // Rule 2: Medium clusters -> basis_k8
    if c >= BasisSweetSpotMin && c <= BasisSweetSpotMax && d >= 1 && avg >= 2 {
        return TechniqueBasisK8
    }

    // Rule 3: Smaller clusters at deeper levels -> basis_k4
    if c >= 5 && c < 20 && d >= 2 && avg >= 2 {
        return TechniqueBasisK4
    }

    // Rule 4: Very small clusters -> baseline
    if c < 5 {
        return TechniqueBaseline
    }

    // Rule 5: Large clusters at deep levels -> FFT
    if c >= 50 && d >= 3 {
        return TechniqueFFT
    }

    // Rule 6: Fallback
    if c >= 5 {
        return TechniqueBasisK4
    }

    return TechniqueBaseline
}

// GenerateSmoothingPlan generates a complete smoothing plan for the tree
func GenerateSmoothingPlan(root NodeInfo, children map[string][]NodeInfo) []SmoothingAction {
    plan := make([]SmoothingAction, 0)
    planRecursive(root, children, &plan)
    return plan
}

func planRecursive(node NodeInfo, children map[string][]NodeInfo, plan *[]SmoothingAction) {
    technique := RecommendedTechnique(node)
    *plan = append(*plan, SmoothingAction{Technique: technique, NodeID: node.NodeID})

    if RefinementNeeded(node) {
        if nodeChildren, ok := children[node.NodeID]; ok {
            for _, child := range nodeChildren {
                if !ClustersDistinguishable(child) {
                    planRecursive(child, children, plan)
                }
            }
        }
    }
}

// EstimateCostMs estimates total training cost in milliseconds
func EstimateCostMs(plan []SmoothingAction, nodes map[string]NodeInfo) float64 {
    costPerCluster := map[SmoothingTechnique]float64{
        TechniqueFFT:      0.4,
        TechniqueBasisK4:  10.0,
        TechniqueBasisK8:  15.0,
        TechniqueBasisK16: 25.0,
        TechniqueBaseline: 0.02,
    }

    total := 0.0
    for _, action := range plan {
        if node, ok := nodes[action.NodeID]; ok {
            cost, exists := costPerCluster[action.Technique]
            if !exists {
                cost = 1.0
            }
            total += float64(node.ClusterCount) * cost
        }
    }
    return total
}
