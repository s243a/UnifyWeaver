use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config as BertConfig, HiddenAct};
use candle_transformers::models::modernbert::{ModernBert, Config as ModernBertConfig};
use tokenizers::Tokenizer;
use anyhow::Result;
use std::path::Path;

/// Model type selection for embedding generation
#[derive(Clone, Debug)]
pub enum ModelType {
    Bert,
    ModernBert,
}

/// Internal model wrapper to support multiple architectures
enum Model {
    Bert(BertModel),
    ModernBert(ModernBert),
}

#[derive(Clone)]
pub struct EmbeddingProvider {
    model: std::sync::Arc<Model>,
    model_type: ModelType,
    tokenizer: std::sync::Arc<Tokenizer>,
    device: Device,
}

impl EmbeddingProvider {
    /// Create a new EmbeddingProvider with automatic device selection
    ///
    /// Model paths can be configured via environment variables:
    /// - MODEL_DIR: Directory containing model files (default: models/all-MiniLM-L6-v2-safetensors)
    /// - MODEL_NAME: Descriptive name for logging (default: all-MiniLM-L6-v2)
    /// - MODEL_TYPE: Model architecture (bert or modernbert, default: bert)
    pub fn new<P: AsRef<Path>>(model_path: P, tokenizer_path: P) -> Result<Self> {
        let device = Self::auto_select_device();
        let model_type = Self::get_model_type();
        Self::with_device(model_path, tokenizer_path, model_type, device)
    }

    /// Get model directory from environment or use default
    fn get_model_dir() -> String {
        std::env::var("MODEL_DIR")
            .unwrap_or_else(|_| "models/all-MiniLM-L6-v2-safetensors".to_string())
    }

    /// Get model name from environment or use default
    fn get_model_name() -> String {
        std::env::var("MODEL_NAME")
            .unwrap_or_else(|_| "all-MiniLM-L6-v2".to_string())
    }

    /// Get model type from environment or use default
    fn get_model_type() -> ModelType {
        std::env::var("MODEL_TYPE")
            .ok()
            .and_then(|s| match s.to_lowercase().as_str() {
                "modernbert" => Some(ModelType::ModernBert),
                "bert" => Some(ModelType::Bert),
                _ => None,
            })
            .unwrap_or(ModelType::Bert)
    }

    /// Create a new EmbeddingProvider with explicit device and model type selection
    pub fn with_device<P: AsRef<Path>>(
        model_path: P,
        tokenizer_path: P,
        model_type: ModelType,
        device: Device,
    ) -> Result<Self> {
        // Load Tokenizer
        let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(|e| anyhow::anyhow!(e))?;

        eprintln!("Loading {:?} model on device: {:?}", model_type, device);

        // Load model weights
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[model_path.as_ref()],
                candle_core::DType::F32,
                &device,
            )?
        };

        // Create model based on type
        let model = match model_type {
            ModelType::Bert => {
                // BERT Model Configuration
                // Choose one of the configurations below by uncommenting it
                // and commenting out the others

                // OPTION 1: all-MiniLM-L6-v2 (small, fast, 384-dim, 512 tokens)
                // RAM: ~0.1-0.2 GB | Context: 512 tokens
                // let config = BertConfig {
                //     vocab_size: 30522,
                //     hidden_size: 384,
                //     num_hidden_layers: 6,
                //     num_attention_heads: 12,
                //     intermediate_size: 1536,
                //     hidden_act: HiddenAct::Gelu,
                //     hidden_dropout_prob: 0.1,
                //     max_position_embeddings: 512,
                //     type_vocab_size: 2,
                //     initializer_range: 0.02,
                //     layer_norm_eps: 1e-12,
                //     ..Default::default()
                // };

                // OPTION 2: intfloat/e5-small-v2 (medium quality, 384-dim, 512 tokens) *** ACTIVE ***
                // RAM: ~0.5 GB | Context: 512 tokens
                // Better quality than MiniLM with moderate RAM increase
                let config = BertConfig {
                    vocab_size: 30522,
                    hidden_size: 384,
                    num_hidden_layers: 12,
                    num_attention_heads: 12,
                    intermediate_size: 1536,
                    hidden_act: HiddenAct::Gelu,
                    hidden_dropout_prob: 0.1,
                    max_position_embeddings: 512,
                    type_vocab_size: 2,
                    initializer_range: 0.02,
                    layer_norm_eps: 1e-12,
                    ..Default::default()
                };

                let bert_model = BertModel::load(vb, &config)?;
                Model::Bert(bert_model)
            }
            ModelType::ModernBert => {
                // ModernBERT: Load config from JSON file
                // Assume config.json is in the same directory as model.safetensors
                let model_dir = model_path.as_ref().parent()
                    .ok_or_else(|| anyhow::anyhow!("Could not determine model directory"))?;
                let config_path = model_dir.join("config.json");

                let config_str = std::fs::read_to_string(&config_path)
                    .map_err(|e| anyhow::anyhow!("Failed to read config.json: {}", e))?;
                let config: ModernBertConfig = serde_json::from_str(&config_str)
                    .map_err(|e| anyhow::anyhow!("Failed to parse config.json: {}", e))?;

                eprintln!("ModernBERT config: {} layers, {} hidden size, {} max positions",
                    config.num_hidden_layers, config.hidden_size, config.max_position_embeddings);

                let modernbert_model = ModernBert::load(vb, &config)?;
                Model::ModernBert(modernbert_model)
            }
        };

        Ok(Self {
            model: std::sync::Arc::new(model),
            model_type,
            tokenizer: std::sync::Arc::new(tokenizer),
            device: device.clone(),
        })
    }

    /// Auto-select the best available device
    /// CONSERVATIVE: Defaults to CPU unless explicitly configured
    /// Respects CANDLE_DEVICE environment variable
    ///
    /// For GPU acceleration, explicitly set CANDLE_DEVICE=cuda or CANDLE_DEVICE=metal
    /// This ensures compatibility with constrained environments (proot, containers, etc.)
    fn auto_select_device() -> Device {
        // Check environment variable first (explicit user choice)
        if let Ok(device_str) = std::env::var("CANDLE_DEVICE") {
            match device_str.to_lowercase().as_str() {
                "cpu" => {
                    eprintln!("Using CPU device (via CANDLE_DEVICE=cpu)");
                    return Device::Cpu;
                }
                "cuda" | "gpu" => {
                    eprintln!("Attempting CUDA device (via CANDLE_DEVICE={})...", device_str);
                    match Device::new_cuda(0) {
                        Ok(device) => {
                            eprintln!("✓ CUDA device initialized successfully");
                            return device;
                        }
                        Err(e) => {
                            eprintln!("✗ CUDA initialization failed: {}", e);
                            eprintln!("  Falling back to CPU");
                            return Device::Cpu;
                        }
                    }
                }
                "metal" => {
                    eprintln!("Attempting Metal device (via CANDLE_DEVICE=metal)...");
                    match Device::new_metal(0) {
                        Ok(device) => {
                            eprintln!("✓ Metal device initialized successfully");
                            return device;
                        }
                        Err(e) => {
                            eprintln!("✗ Metal initialization failed: {}", e);
                            eprintln!("  Falling back to CPU");
                            return Device::Cpu;
                        }
                    }
                }
                "auto" => {
                    eprintln!("CANDLE_DEVICE=auto: Attempting GPU detection...");
                    // Try CUDA first
                    if let Ok(device) = Device::new_cuda(0) {
                        eprintln!("✓ Auto-detected CUDA device");
                        return device;
                    }
                    // Try Metal (macOS)
                    if let Ok(device) = Device::new_metal(0) {
                        eprintln!("✓ Auto-detected Metal device");
                        return device;
                    }
                    eprintln!("  No GPU detected, using CPU");
                    return Device::Cpu;
                }
                _ => {
                    eprintln!("⚠ Unknown CANDLE_DEVICE value: '{}', defaulting to CPU", device_str);
                    eprintln!("  Valid values: cpu, cuda, gpu, metal, auto");
                    return Device::Cpu;
                }
            }
        }

        // CONSERVATIVE DEFAULT: Use CPU if no explicit configuration
        // This ensures safety in unknown/constrained environments
        eprintln!("Using CPU device (no CANDLE_DEVICE set - conservative default)");
        eprintln!("  To use GPU: export CANDLE_DEVICE=cuda  (or =metal on macOS)");
        eprintln!("  To auto-detect: export CANDLE_DEVICE=auto");
        Device::Cpu
    }

    pub fn get_embedding(&self, text: &str) -> Result<Vec<f32>> {
        let device = &self.device;
        let tokenizer = self.tokenizer.clone();
        let model = self.model.clone();

        // Tokenize
        let tokens = tokenizer.encode(text, true).map_err(|e| anyhow::anyhow!(e))?;
        let token_ids = Tensor::new(tokens.get_ids(), device)?.unsqueeze(0)?;

        // Run Inference based on model type
        let embeddings = match &*model {
            Model::Bert(bert_model) => {
                let token_type_ids = Tensor::new(tokens.get_type_ids(), device)?.unsqueeze(0)?;
                bert_model.forward(&token_ids, &token_type_ids, None)?
            }
            Model::ModernBert(modernbert_model) => {
                // ModernBERT doesn't use token_type_ids, uses attention_mask instead
                let attention_mask = Tensor::ones(
                    (1, token_ids.dim(1)?),
                    candle_core::DType::U8,
                    device
                )?;
                modernbert_model.forward(&token_ids, &attention_mask)?
            }
        };

        // Mean Pooling (simplified: just taking [CLS] token or mean of last hidden state)
        // For sentence-transformers, usually mean pooling with attention mask is used.
        // Here, we'll just take the mean of the last hidden state for simplicity in this prototype.
        let (_n_sentence, n_tokens, _hidden_size) = embeddings.dims3()?;
        let embeddings = (embeddings.sum(1)? / (n_tokens as f64))?;
        let embeddings = embeddings.squeeze(0)?;

        // Normalize (L2 normalization)
        let norm = embeddings.sqr()?.sum_all()?.sqrt()?;
        let embeddings = embeddings.broadcast_div(&norm)?;

        Ok(embeddings.to_vec1()?)
    }
}
