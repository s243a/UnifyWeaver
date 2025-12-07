use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config};
use tokenizers::Tokenizer;
use anyhow::Result;
use std::path::Path;

#[derive(Clone)]
pub struct EmbeddingProvider {
    model: std::sync::Arc<BertModel>,
    tokenizer: std::sync::Arc<Tokenizer>,
    device: Device,
}

impl EmbeddingProvider {
    /// Create a new EmbeddingProvider with automatic device selection
    pub fn new<P: AsRef<Path>>(model_path: P, tokenizer_path: P) -> Result<Self> {
        let device = Self::auto_select_device();
        Self::with_device(model_path, tokenizer_path, device)
    }

    /// Create a new EmbeddingProvider with explicit device selection
    pub fn with_device<P: AsRef<Path>>(model_path: P, tokenizer_path: P, device: Device) -> Result<Self> {
        // Load Tokenizer
        let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(|e| anyhow::anyhow!(e))?;

        // Load Model
        eprintln!("Loading model on device: {:?}", device);
        let config = Config::default(); // Assuming bert-base or similar config match
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_path.as_ref()], candle_core::DType::F32, &device)? };
        let model = BertModel::load(vb, &config)?;

        Ok(Self {
            model: std::sync::Arc::new(model),
            tokenizer: std::sync::Arc::new(tokenizer),
            device: device.clone(),
        })
    }

    /// Auto-select the best available device
    /// Priority: CUDA > Metal > CPU
    /// Respects CANDLE_DEVICE environment variable
    fn auto_select_device() -> Device {
        // Check environment variable first
        if let Ok(device_str) = std::env::var("CANDLE_DEVICE") {
            match device_str.to_lowercase().as_str() {
                "cpu" => return Device::Cpu,
                "cuda" | "gpu" => {
                    if let Ok(device) = Device::new_cuda(0) {
                        eprintln!("Using CUDA device (via CANDLE_DEVICE)");
                        return device;
                    }
                    eprintln!("CUDA requested but not available, falling back to CPU");
                }
                "metal" => {
                    if let Ok(device) = Device::new_metal(0) {
                        eprintln!("Using Metal device (via CANDLE_DEVICE)");
                        return device;
                    }
                    eprintln!("Metal requested but not available, falling back to CPU");
                }
                _ => eprintln!("Unknown CANDLE_DEVICE value: {}, using auto-detect", device_str),
            }
        }

        // Auto-detect: Try CUDA first (most common for ML)
        if let Ok(device) = Device::new_cuda(0) {
            eprintln!("Auto-detected CUDA device");
            return device;
        }

        // Try Metal (macOS)
        if let Ok(device) = Device::new_metal(0) {
            eprintln!("Auto-detected Metal device");
            return device;
        }

        // Fallback to CPU
        eprintln!("Using CPU device (no GPU detected)");
        Device::Cpu
    }

    pub fn get_embedding(&self, text: &str) -> Result<Vec<f32>> {
        let device = &self.device;
        let tokenizer = self.tokenizer.clone();
        let model = self.model.clone();

        // Tokenize
        let tokens = tokenizer.encode(text, true).map_err(|e| anyhow::anyhow!(e))?;
        let token_ids = Tensor::new(tokens.get_ids(), device)?.unsqueeze(0)?;
        let token_type_ids = Tensor::new(tokens.get_type_ids(), device)?.unsqueeze(0)?;

        // Run Inference
        let embeddings = model.forward(&token_ids, &token_type_ids, None)?;
        
        // Mean Pooling (simplified: just taking [CLS] token or mean of last hidden state)
        // For sentence-transformers, usually mean pooling with attention mask is used.
        // Here, we'll just take the mean of the last hidden state for simplicity in this prototype.
        let (_n_sentence, n_tokens, _hidden_size) = embeddings.dims3()?;
        let embeddings = (embeddings.sum(1)? / (n_tokens as f64))?;
        let embeddings = embeddings.squeeze(0)?;

        // Normalize
        let embeddings = (embeddings.clone() / embeddings.sqr()?.sum_all()?.sqrt()?)?;
        
        Ok(embeddings.to_vec1()?)
    }
}
