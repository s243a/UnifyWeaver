# Rust Vector Search Target

Semantic search capabilities for UnifyWeaver using Rust + Candle ML framework with BERT embeddings.

## Features

- **BERT-based embeddings** using Candle (Rust ML framework)
- **Text search** - Traditional keyword matching
- **Vector search** - Semantic similarity using embeddings
- **RDF parsing** - Import and index RDF/XML data
- **Persistent storage** - redb embedded database
- **GPU acceleration** - Automatic CUDA/Metal detection with CPU fallback

## Quick Start

### 1. Download Model

```bash
# Install Hugging Face CLI
pip install huggingface_hub

# Download all-MiniLM-L6-v2 (sentence-transformers model)
huggingface-cli download sentence-transformers/all-MiniLM-L6-v2 \
  --local-dir models/all-MiniLM-L6-v2
```

### 2. Generate Rust Code

```prolog
% example.pl
:- use_module('src/unifyweaver/targets/rust_target').

my_search(Query) :-
    crawler_run(["data/my_file.rdf"], 1),
    semantic_search(Query, 5, Results).

main :-
    compile_predicate_to_rust(my_search/1, [], Code),
    write_rust_project(Code, 'output/my_search').
```

```bash
swipl -g "consult('example.pl'), main, halt."
```

### 3. Build and Run

```bash
cd output/my_search
cargo build --release
./target/release/my_search
```

## Device Selection

### Auto-Detection (Default)

The system automatically selects the best available device:
1. **CUDA** (NVIDIA GPUs) - if available
2. **Metal** (Apple Silicon/AMD on macOS) - if available
3. **CPU** - fallback

```bash
# Auto-detect (tries GPU first)
./target/release/my_search
```

### Manual Override

Use the `CANDLE_DEVICE` environment variable:

```bash
# Force CPU (for constrained environments like proot/termux)
CANDLE_DEVICE=cpu ./target/release/my_search

# Force CUDA
CANDLE_DEVICE=cuda ./target/release/my_search

# Force Metal (macOS)
CANDLE_DEVICE=metal ./target/release/my_search
```

### When to Use CPU Mode

Force CPU mode in these scenarios:
- **proot environments** (Android Termux with proot Debian)
- **Containers without GPU passthrough**
- **Systems without GPU drivers**
- **Debugging/testing**

## Performance

### GPU vs CPU

| Device | Embeddings/sec | Use Case |
|--------|---------------|----------|
| CUDA (RTX 3090) | ~1000 | Production |
| Metal (M1 Max) | ~500 | macOS Development |
| CPU (8-core) | ~10-20 | Testing, Constrained Envs |

### Model Size

- **all-MiniLM-L6-v2**: 91 MB (384-dim embeddings, 6 layers)
- Fast inference, good quality for most use cases

## Architecture

### Components

```
┌─────────────────────────────────────┐
│  Prolog Source Code                 │
│  (semantic_search/3, crawler_run/2) │
└────────────────┬────────────────────┘
                 │ compiles to
                 ▼
┌─────────────────────────────────────┐
│  Generated Rust Project             │
│  ├── main.rs (entry point)          │
│  ├── embedding.rs (BERT model)      │
│  ├── crawler.rs (RDF parser)        │
│  ├── searcher.rs (search logic)     │
│  └── importer.rs (database)         │
└─────────────────────────────────────┘
                 │ uses
                 ▼
┌─────────────────────────────────────┐
│  Dependencies                        │
│  ├── candle (ML framework)          │
│  ├── redb (embedded database)       │
│  └── quick-xml (RDF parser)         │
└─────────────────────────────────────┘
```

### Workflow

1. **Compile**: Prolog → Rust code generation
2. **Build**: `cargo build` creates binary
3. **Index**: Parse RDF, generate embeddings, store in redb
4. **Search**: Query embeddings for semantic matches

## API Reference

### Generated Rust Code

```rust
use embedding::EmbeddingProvider;
use crawler::PtCrawler;
use searcher::PtSearcher;

// Auto-detect device (GPU if available, else CPU)
let embedding = EmbeddingProvider::new(
    "models/all-MiniLM-L6-v2/model.safetensors",
    "models/all-MiniLM-L6-v2/tokenizer.json"
)?;

// Or force specific device
let embedding = EmbeddingProvider::with_device(
    model_path,
    tokenizer_path,
    Device::Cpu  // or Device::new_cuda(0)
)?;

// Index data
let importer = PtImporter::new("data.redb")?;
let crawler = PtCrawler::new(importer, embedding.clone());
crawler.crawl(&["file.rdf".to_string()], 1)?;

// Search
let searcher = PtSearcher::new("data.redb", embedding)?;
let results = searcher.vector_search("my query", 5)?;
```

## Comparison with Python Target

| Feature | Rust (Candle) | Python (ONNX) |
|---------|--------------|---------------|
| Model Format | safetensors | ONNX |
| Embedding Dim | 384 | 384 |
| Model | all-MiniLM-L6-v2 | all-MiniLM-L6-v2 |
| Runtime | Native binary | Python interpreter |
| GPU Support | CUDA, Metal | CUDA, DirectML |
| Dependencies | ~170 crates | onnxruntime, numpy |
| Binary Size | ~20 MB | N/A (Python) |
| Startup Time | ~100ms | ~500ms |

Both use the same underlying model for **identical semantic search results**.

## Troubleshooting

### "shape mismatch" errors

The BERT config must match your model. For all-MiniLM-L6-v2:
- `hidden_size: 384` (not 768)
- `num_hidden_layers: 6` (not 12)

### "CUDA not available" on WSL2

Install NVIDIA drivers and CUDA toolkit:
```bash
# Check CUDA
nvidia-smi

# Install CUDA toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/3bf863cc.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/ /"
sudo apt-get update
sudo apt-get -y install cuda
```

### Slow on CPU

Expected! CPU inference is ~50-100x slower than GPU. Solutions:
1. Use GPU if available
2. Index offline, search online
3. Reduce batch size
4. Use smaller model

### Memory issues

The model + data must fit in RAM/VRAM:
- Model: ~400 MB
- Embeddings: 384 floats × num_documents × 4 bytes
- Example: 100K documents = ~150 MB embeddings

## Future Enhancements

- [ ] Multi-GPU support
- [ ] Batch embedding generation
- [ ] Model config auto-detection from config.json
- [ ] Alternative models (BGE, E5, etc.)
- [ ] Quantization support (int8, fp16)
- [ ] ONNX runtime option for consistency with Python

## References

- [Candle ML Framework](https://github.com/huggingface/candle)
- [all-MiniLM-L6-v2 Model](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- [redb Database](https://github.com/cberner/redb)
- [UnifyWeaver Documentation](README.md)
