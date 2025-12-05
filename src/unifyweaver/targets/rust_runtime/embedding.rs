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
}

impl EmbeddingProvider {
    pub fn new<P: AsRef<Path>>(model_path: P, tokenizer_path: P) -> Result<Self> {
        // Load Tokenizer
        let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(|e| anyhow::anyhow!(e))?;

        // Load Model
        let device = Device::Cpu;
        let config = Config::default(); // Assuming bert-base or similar config match
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_path.as_ref()], candle_core::DType::F32, &device)? };
        let model = BertModel::load(vb, &config)?;

        Ok(Self {
            model: std::sync::Arc::new(model),
            tokenizer: std::sync::Arc::new(tokenizer),
        })
    }

    pub fn get_embedding(&self, text: &str) -> Result<Vec<f32>> {
        let device = Device::Cpu;
        let tokenizer = self.tokenizer.clone();
        let model = self.model.clone();

        // Tokenize
        let tokens = tokenizer.encode(text, true).map_err(|e| anyhow::anyhow!(e))?;
        let token_ids = Tensor::new(tokens.get_ids(), &device)?.unsqueeze(0)?;
        let token_type_ids = Tensor::new(tokens.get_type_ids(), &device)?.unsqueeze(0)?;

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
