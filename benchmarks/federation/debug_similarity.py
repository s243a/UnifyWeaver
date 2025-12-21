# Quick debug to understand similarity distributions
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import numpy as np
from benchmarks.federation.synthetic_network import create_synthetic_network
from benchmarks.federation.workload_generator import generate_workload

def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# Create small network
nodes = create_synthetic_network(num_nodes=50, seed=42)
workload = generate_workload(network=nodes, num_queries=10, seed=43)

print("Cluster distribution:")
for cluster_id in range(5):
    cluster_nodes = [n for i, n in enumerate(nodes) if i % 5 == cluster_id]
    print(f"  Cluster {cluster_id}: {len(cluster_nodes)} nodes")
    # Check intra-cluster similarity
    if len(cluster_nodes) >= 2:
        sims = []
        for i, n1 in enumerate(cluster_nodes):
            for n2 in cluster_nodes[i+1:]:
                sims.append(cosine_sim(n1.centroid, n2.centroid))
        print(f"    Intra-cluster similarity: min={min(sims):.3f}, max={max(sims):.3f}, avg={np.mean(sims):.3f}")

print("\nQuery analysis:")
for i, query in enumerate(workload[:5]):
    print(f"\nQuery {i} ({query.expected_type.value}):")
    print(f"  Ground truth size: {len(query.ground_truth_nodes)}")

    # Check similarities
    sims = [(n.node_id, cosine_sim(query.embedding, n.centroid)) for n in nodes]
    sims.sort(key=lambda x: x[1], reverse=True)

    print(f"  Top 5 similarities:")
    for nid, sim in sims[:5]:
        in_gt = "✓" if nid in query.ground_truth_nodes else " "
        print(f"    {nid}: {sim:.3f} {in_gt}")

    above_07 = sum(1 for _, sim in sims if sim > 0.7)
    above_05 = sum(1 for _, sim in sims if sim > 0.5)
    print(f"  Nodes > 0.7 sim: {above_07}")
    print(f"  Nodes > 0.5 sim: {above_05}")
