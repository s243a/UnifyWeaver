// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2025 John William Creighton (s243a)
//
// Integration test for Rust multi-head LDA projection with real NPY files.
//
// Run from examples/pearltrees: cargo run --bin test_projection_integration

mod importer;
mod crawler;
mod projection;
mod searcher;
mod embedding;

use std::collections::HashMap;
use std::path::Path;

use projection::{MultiHeadProjection, Config};

fn main() {
    println!("=== Rust Projection Integration Test ===\n");

    let embeddings_dir = "../../playbooks/lda-training-data/embeddings";

    if !Path::new(embeddings_dir).exists() {
        println!("Skipping: embeddings directory not found at {}", embeddings_dir);
        println!("Run from project root or ensure LDA training data exists.");
        return;
    }

    // Test 1: Load mh_2 projection with explicit file paths
    println!("Test 1: Loading mh_2 projection from NPY files...");

    let mut head_files: HashMap<i32, (String, String)> = HashMap::new();

    // Find all mh_2 cluster files
    for cluster_id in 1..=20 {
        let centroid_path = format!("{}/mh_2_cluster_{}_centroid.npy", embeddings_dir, cluster_id);
        let answer_path = format!("{}/mh_2_cluster_{}_answer.npy", embeddings_dir, cluster_id);

        if Path::new(&centroid_path).exists() && Path::new(&answer_path).exists() {
            head_files.insert(cluster_id, (centroid_path, answer_path));
        }
    }

    if head_files.is_empty() {
        println!("  No mh_2 cluster files found. Skipping.");
        return;
    }

    println!("  Found {} cluster head pairs", head_files.len());

    let config = Config {
        data_dir: None,
        temperature: 0.1,
        head_files: Some(head_files),
    };

    match MultiHeadProjection::load(config) {
        Ok(mh) => {
            println!("  ✓ Loaded {} heads, dimension={}", mh.num_heads(), mh.dimension);

            // Show first head info for debugging
            if let Some(head) = mh.heads.first() {
                let c_norm: f32 = head.centroid.iter().map(|x| x * x).sum::<f32>().sqrt();
                let a_norm: f32 = head.answer_emb.iter().map(|x| x * x).sum::<f32>().sqrt();
                println!("  First head: cluster={}, centroid_norm={:.4}, answer_norm={:.4}",
                    head.cluster_id, c_norm, a_norm);
            }

            // Test 2: Project a mock query embedding
            println!("\nTest 2: Projecting mock query embedding...");

            // Create a random-ish query vector of the right dimension
            let query: Vec<f32> = (0..mh.dimension)
                .map(|i| ((i as f32 * 0.1).sin() * 0.5))
                .collect();

            match mh.project_with_weights(&query) {
                Ok((projected, weights)) => {
                    println!("  ✓ Projection succeeded");
                    println!("  Projected vector norm: {:.4}",
                        projected.iter().map(|x| x * x).sum::<f32>().sqrt());

                    // Show top 3 routing weights
                    let mut weight_vec: Vec<_> = weights.iter().collect();
                    weight_vec.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));

                    println!("  Top routing weights:");
                    for (cluster_id, weight) in weight_vec.iter().take(3) {
                        println!("    Cluster {}: {:.4}", cluster_id, weight);
                    }

                    // Check for NaN
                    if projected.iter().any(|x| x.is_nan()) {
                        println!("  Warning: Projected vector contains NaN values");
                    }
                }
                Err(e) => {
                    println!("  ✗ Projection failed: {}", e);
                }
            }

            // Test 3: Verify softmax weights sum to 1
            println!("\nTest 3: Verifying softmax weights...");
            if let Ok((_, weights)) = mh.project_with_weights(&query) {
                let sum: f32 = weights.values().sum();
                if (sum - 1.0).abs() < 0.001 {
                    println!("  ✓ Weights sum to {:.6} (expected ~1.0)", sum);
                } else {
                    println!("  ✗ Weights sum to {:.6} (expected ~1.0)", sum);
                }
            }
        }
        Err(e) => {
            println!("  ✗ Failed to load projection: {}", e);
        }
    }

    println!("\n=== Integration Test Complete ===");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_real_npy_files() {
        let embeddings_dir = "../../playbooks/lda-training-data/embeddings";

        if !Path::new(embeddings_dir).exists() {
            println!("Skipping: embeddings directory not found");
            return;
        }

        let mut head_files: HashMap<i32, (String, String)> = HashMap::new();

        for cluster_id in 1..=20 {
            let centroid_path = format!("{}/mh_2_cluster_{}_centroid.npy", embeddings_dir, cluster_id);
            let answer_path = format!("{}/mh_2_cluster_{}_answer.npy", embeddings_dir, cluster_id);

            if Path::new(&centroid_path).exists() && Path::new(&answer_path).exists() {
                head_files.insert(cluster_id, (centroid_path, answer_path));
            }
        }

        if head_files.is_empty() {
            println!("No cluster files found, skipping");
            return;
        }

        let config = Config {
            data_dir: None,
            temperature: 0.1,
            head_files: Some(head_files.clone()),
        };

        let mh = MultiHeadProjection::load(config).expect("Failed to load projection");

        assert!(mh.num_heads() > 0, "Should have at least one head");
        assert!(mh.dimension > 0, "Dimension should be positive");

        // Test projection
        let query: Vec<f32> = (0..mh.dimension)
            .map(|i| ((i as f32 * 0.1).sin() * 0.5))
            .collect();

        let (projected, weights) = mh.project_with_weights(&query)
            .expect("Projection should succeed");

        assert_eq!(projected.len(), mh.dimension);

        let weight_sum: f32 = weights.values().sum();
        assert!((weight_sum - 1.0).abs() < 0.001, "Weights should sum to 1.0");
    }
}
