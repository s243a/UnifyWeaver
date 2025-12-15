// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2025 John William Creighton (s243a)
//
// Multi-head LDA projection for semantic search.
//
// Uses per-cluster centroids and answer embeddings to route queries,
// similar to transformer attention heads. Each "head" corresponds to a
// Q-A cluster, with its own centroid (mean of training questions) and
// answer embedding.
//
// See: docs/proposals/MULTI_HEAD_PROJECTION_THEORY.md

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;
use anyhow::{Result, anyhow};

/// A single projection head (one Q-A cluster).
#[derive(Debug, Clone)]
pub struct Head {
    pub cluster_id: i32,
    pub centroid: Vec<f32>,   // Mean of training question embeddings
    pub answer_emb: Vec<f32>, // Answer embedding for this cluster
}

/// Multi-head LDA projection model.
#[derive(Debug, Clone)]
pub struct MultiHeadProjection {
    pub heads: Vec<Head>,
    pub temperature: f32,
    pub dimension: usize,
}

/// Configuration for loading multi-head projection.
#[derive(Debug, Clone)]
pub struct Config {
    /// Directory containing centroid_*.npy and answer_emb_*.npy files
    pub data_dir: Option<String>,

    /// Temperature for softmax routing (lower = sharper routing)
    /// Recommended: 0.1
    pub temperature: f32,

    /// Explicit head file pairs (cluster_id -> (centroid_path, answer_emb_path))
    pub head_files: Option<HashMap<i32, (String, String)>>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            data_dir: None,
            temperature: 0.1,
            head_files: None,
        }
    }
}

impl MultiHeadProjection {
    /// Load a multi-head projection from numpy files.
    pub fn load(config: Config) -> Result<Self> {
        let temperature = if config.temperature <= 0.0 { 0.1 } else { config.temperature };

        let mut heads = Vec::new();
        let mut dimension = 0;

        // If explicit head files provided, use them
        if let Some(head_files) = config.head_files {
            for (cluster_id, (centroid_path, answer_path)) in head_files {
                let head = Self::load_head(cluster_id, &centroid_path, &answer_path)?;
                if dimension == 0 {
                    dimension = head.centroid.len();
                }
                heads.push(head);
            }
        } else if let Some(data_dir) = config.data_dir {
            // Auto-discover from data directory
            heads = Self::discover_heads(&data_dir)?;
            if let Some(first) = heads.first() {
                dimension = first.centroid.len();
            }
        } else {
            return Err(anyhow!("Either data_dir or head_files must be provided"));
        }

        if heads.is_empty() {
            return Err(anyhow!("No heads loaded"));
        }

        Ok(Self {
            heads,
            temperature,
            dimension,
        })
    }

    /// Auto-discover head files from a directory.
    fn discover_heads(data_dir: &str) -> Result<Vec<Head>> {
        let dir = Path::new(data_dir);
        if !dir.is_dir() {
            return Err(anyhow!("Data directory does not exist: {}", data_dir));
        }

        let mut heads = Vec::new();

        // Look for centroid_*.npy files
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();

            if let Some(filename) = path.file_name().and_then(|n| n.to_str()) {
                if filename.starts_with("centroid_") && filename.ends_with(".npy") {
                    // Extract cluster ID
                    let id_str = filename
                        .strip_prefix("centroid_")
                        .and_then(|s| s.strip_suffix(".npy"))
                        .unwrap_or("");

                    if let Ok(cluster_id) = id_str.parse::<i32>() {
                        let answer_path = dir.join(format!("answer_emb_{}.npy", cluster_id));

                        if answer_path.exists() {
                            let head = Self::load_head(
                                cluster_id,
                                path.to_str().unwrap_or(""),
                                answer_path.to_str().unwrap_or(""),
                            )?;
                            heads.push(head);
                        }
                    }
                }
            }
        }

        // Sort by cluster ID for deterministic order
        heads.sort_by_key(|h| h.cluster_id);

        Ok(heads)
    }

    /// Load a single head from numpy files.
    fn load_head(cluster_id: i32, centroid_path: &str, answer_path: &str) -> Result<Head> {
        let centroid = load_npy_f32(centroid_path)?;
        let answer_emb = load_npy_f32(answer_path)?;

        Ok(Head {
            cluster_id,
            centroid,
            answer_emb,
        })
    }

    /// Project a query embedding through multi-head routing.
    /// Returns the projected embedding as a weighted combination of answer embeddings.
    pub fn project(&self, query_emb: &[f32]) -> Result<Vec<f32>> {
        if query_emb.len() != self.dimension {
            return Err(anyhow!(
                "Query dimension {} != model dimension {}",
                query_emb.len(),
                self.dimension
            ));
        }

        if self.heads.is_empty() {
            return Err(anyhow!("No heads loaded"));
        }

        // Normalize query
        let query_normed = normalize(query_emb);

        // Compute similarity to each centroid
        let similarities: Vec<f32> = self.heads.iter()
            .map(|head| {
                let centroid_normed = normalize(&head.centroid);
                dot_product(&query_normed, &centroid_normed)
            })
            .collect();

        // Apply softmax with temperature
        let weights = softmax(&similarities, self.temperature);

        // Weighted combination of answer embeddings
        let mut projected = vec![0.0f32; self.dimension];
        for (i, head) in self.heads.iter().enumerate() {
            for (j, val) in head.answer_emb.iter().enumerate() {
                projected[j] += weights[i] * val;
            }
        }

        Ok(projected)
    }

    /// Project with routing weights returned for analysis/debugging.
    pub fn project_with_weights(&self, query_emb: &[f32]) -> Result<(Vec<f32>, HashMap<i32, f32>)> {
        if query_emb.len() != self.dimension {
            return Err(anyhow!(
                "Query dimension {} != model dimension {}",
                query_emb.len(),
                self.dimension
            ));
        }

        if self.heads.is_empty() {
            return Err(anyhow!("No heads loaded"));
        }

        // Normalize query
        let query_normed = normalize(query_emb);

        // Compute similarity to each centroid
        let similarities: Vec<f32> = self.heads.iter()
            .map(|head| {
                let centroid_normed = normalize(&head.centroid);
                dot_product(&query_normed, &centroid_normed)
            })
            .collect();

        // Apply softmax with temperature
        let weights = softmax(&similarities, self.temperature);

        // Build weight map
        let weight_map: HashMap<i32, f32> = self.heads.iter()
            .enumerate()
            .map(|(i, head)| (head.cluster_id, weights[i]))
            .collect();

        // Weighted combination of answer embeddings
        let mut projected = vec![0.0f32; self.dimension];
        for (i, head) in self.heads.iter().enumerate() {
            for (j, val) in head.answer_emb.iter().enumerate() {
                projected[j] += weights[i] * val;
            }
        }

        Ok((projected, weight_map))
    }

    /// Get the number of projection heads.
    pub fn num_heads(&self) -> usize {
        self.heads.len()
    }

    /// Get the temperature.
    pub fn get_temperature(&self) -> f32 {
        self.temperature
    }

    /// Set the temperature.
    pub fn set_temperature(&mut self, t: f32) {
        if t > 0.0 {
            self.temperature = t;
        }
    }
}

// ============================================================================
// Helper functions
// ============================================================================

/// Apply softmax with temperature to a slice of values.
fn softmax(x: &[f32], temperature: f32) -> Vec<f32> {
    if x.is_empty() {
        return Vec::new();
    }

    // Scale by temperature and find max for numerical stability
    let scaled: Vec<f64> = x.iter()
        .map(|&v| (v as f64) / (temperature as f64))
        .collect();

    let max_val = scaled.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    // Compute exp(x - max) for numerical stability
    let exp_vals: Vec<f64> = scaled.iter()
        .map(|&v| (v - max_val).exp())
        .collect();

    let sum: f64 = exp_vals.iter().sum();

    exp_vals.iter()
        .map(|&v| (v / sum) as f32)
        .collect()
}

/// Normalize a vector to unit length.
fn normalize(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm == 0.0 {
        return v.to_vec();
    }

    v.iter().map(|x| x / norm).collect()
}

/// Compute dot product of two vectors.
fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Load a 1D numpy array as float32 values from a .npy file.
/// Supports both float32 ('<f4') and float64 ('<f8') dtypes.
fn load_npy_f32(path: &str) -> Result<Vec<f32>> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    // Read magic number (6 bytes)
    let mut magic = [0u8; 6];
    reader.read_exact(&mut magic)?;

    if &magic != b"\x93NUMPY" {
        return Err(anyhow!("Invalid npy magic: {:?}", magic));
    }

    // Read version (2 bytes)
    let mut version = [0u8; 2];
    reader.read_exact(&mut version)?;

    // Read header length based on version
    let header_len: usize = if version[0] == 1 {
        let mut len_bytes = [0u8; 2];
        reader.read_exact(&mut len_bytes)?;
        u16::from_le_bytes(len_bytes) as usize
    } else if version[0] == 2 || version[0] == 3 {
        let mut len_bytes = [0u8; 4];
        reader.read_exact(&mut len_bytes)?;
        u32::from_le_bytes(len_bytes) as usize
    } else {
        return Err(anyhow!("Unsupported npy version: {}.{}", version[0], version[1]));
    };

    // Read header
    let mut header = vec![0u8; header_len];
    reader.read_exact(&mut header)?;
    let header_str = String::from_utf8_lossy(&header);

    // Parse shape from header
    let shape = parse_npy_shape(&header_str)?;

    // Calculate total elements
    let total_elements: usize = shape.iter().product();

    // Detect dtype from header
    let is_float64 = header_str.contains("<f8") || header_str.contains("float64");

    if is_float64 {
        // Read as float64 and convert to float32
        let mut data_f64 = vec![0f64; total_elements];
        let data_bytes = unsafe {
            std::slice::from_raw_parts_mut(
                data_f64.as_mut_ptr() as *mut u8,
                total_elements * 8,
            )
        };
        reader.read_exact(data_bytes)?;

        // Convert to f32
        let data: Vec<f32> = data_f64.iter().map(|&x| x as f32).collect();
        Ok(data)
    } else {
        // Read as float32 little-endian
        let mut data = vec![0f32; total_elements];
        let data_bytes = unsafe {
            std::slice::from_raw_parts_mut(
                data.as_mut_ptr() as *mut u8,
                total_elements * 4,
            )
        };
        reader.read_exact(data_bytes)?;
        Ok(data)
    }
}

/// Parse shape from numpy header string.
fn parse_npy_shape(header: &str) -> Result<Vec<usize>> {
    // Find 'shape': (...)
    let shape_start = header.find("'shape'")
        .ok_or_else(|| anyhow!("Shape not found in header"))?;

    let paren_start = header[shape_start..].find('(')
        .ok_or_else(|| anyhow!("Opening paren not found"))?
        + shape_start + 1;

    let paren_end = header[paren_start..].find(')')
        .ok_or_else(|| anyhow!("Closing paren not found"))?
        + paren_start;

    let shape_str = &header[paren_start..paren_end];

    if shape_str.trim().is_empty() {
        return Ok(vec![1]); // Scalar
    }

    let shape: Vec<usize> = shape_str
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();

    if shape.is_empty() {
        Ok(vec![1])
    } else {
        Ok(shape)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::TempDir;

    #[test]
    fn test_softmax() {
        let input = vec![0.85, 0.70, 0.60];
        let result = softmax(&input, 1.0);

        // Sum should be ~1.0
        let sum: f32 = result.iter().sum();
        assert!((sum - 1.0).abs() < 0.001, "softmax sum = {}, want 1.0", sum);

        // First value should be highest
        assert!(result[0] > result[1] && result[0] > result[2],
            "softmax[0] should be highest, got {:?}", result);
    }

    #[test]
    fn test_softmax_temperature() {
        let input = vec![0.85, 0.70, 0.60];

        let sharp_result = softmax(&input, 0.1);
        let diffuse_result = softmax(&input, 1.0);

        // Low temp should produce sharper distribution
        let sharp_diff = sharp_result[0] - sharp_result[1];
        let diffuse_diff = diffuse_result[0] - diffuse_result[1];

        assert!(sharp_diff > diffuse_diff,
            "low temp should produce sharper distribution: sharp={:?}, diffuse={:?}",
            sharp_result, diffuse_result);
    }

    #[test]
    fn test_normalize() {
        let v = vec![3.0, 4.0];
        let result = normalize(&v);

        // Norm should be 1.0
        let norm: f32 = result.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.001, "normalized norm = {}, want 1.0", norm);
    }

    #[test]
    fn test_dot_product() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];

        let result = dot_product(&a, &b);
        let expected = 1.0*4.0 + 2.0*5.0 + 3.0*6.0; // 32

        assert!((result - expected).abs() < 0.001,
            "dot_product = {}, want {}", result, expected);
    }

    #[test]
    fn test_multi_head_projection_mock() {
        // Create mock projection
        let mh = MultiHeadProjection {
            temperature: 0.1,
            dimension: 4,
            heads: vec![
                Head {
                    cluster_id: 1,
                    centroid: vec![1.0, 0.0, 0.0, 0.0],
                    answer_emb: vec![0.5, 0.5, 0.0, 0.0],
                },
                Head {
                    cluster_id: 2,
                    centroid: vec![0.0, 1.0, 0.0, 0.0],
                    answer_emb: vec![0.0, 0.0, 0.5, 0.5],
                },
            ],
        };

        // Query close to first centroid
        let query = vec![0.9, 0.1, 0.0, 0.0];
        let (projected, weights) = mh.project_with_weights(&query).unwrap();

        // Should route mostly to first head
        assert!(weights[&1] > weights[&2],
            "expected head 1 to dominate: {:?}", weights);

        // Projected should be closer to first answer embedding
        assert!(projected[0] > 0.4 && projected[1] > 0.4,
            "projected should be similar to first answer: {:?}", projected);
    }

    #[test]
    fn test_parse_npy_shape() {
        let cases = vec![
            ("{'descr': '<f4', 'fortran_order': False, 'shape': (384,), }", vec![384]),
            ("{'descr': '<f4', 'fortran_order': False, 'shape': (10, 384), }", vec![10, 384]),
            ("{'descr': '<f4', 'fortran_order': False, 'shape': (), }", vec![1]),
        ];

        for (header, expected) in cases {
            let result = parse_npy_shape(header).unwrap();
            assert_eq!(result, expected, "parse_npy_shape({:?})", header);
        }
    }

    #[test]
    fn test_load_npy_f32() {
        let tmp_dir = TempDir::new().unwrap();
        let npy_path = tmp_dir.path().join("test.npy");

        // Create a simple 1D npy file: [1.0, 2.0, 3.0]
        let mut data = Vec::new();

        // Magic number
        data.extend_from_slice(b"\x93NUMPY");

        // Version 1.0
        data.extend_from_slice(&[1, 0]);

        // Header
        let header = "{'descr': '<f4', 'fortran_order': False, 'shape': (3,), }";
        let padding = 64 - ((10 + header.len()) % 64);
        let padded_header = format!("{}{}\n", header, " ".repeat(padding - 1));

        // Header length
        let header_len = padded_header.len() as u16;
        data.extend_from_slice(&header_len.to_le_bytes());

        // Header content
        data.extend_from_slice(padded_header.as_bytes());

        // Data: 3 float32 values
        for val in [1.0f32, 2.0f32, 3.0f32] {
            data.extend_from_slice(&val.to_le_bytes());
        }

        // Write file
        let mut file = File::create(&npy_path).unwrap();
        file.write_all(&data).unwrap();

        // Load and verify
        let result = load_npy_f32(npy_path.to_str().unwrap()).unwrap();

        assert_eq!(result.len(), 3);
        assert!((result[0] - 1.0).abs() < 0.001);
        assert!((result[1] - 2.0).abs() < 0.001);
        assert!((result[2] - 3.0).abs() < 0.001);
    }
}
