# Model Download Guide

This guide shows how to download embedding models for the Rust vector search implementation.

## Prerequisites

For `hf` command method:
- Python virtual environment with `huggingface-hub[cli]` installed
- Located at: `~/hf_cli_env/`

```bash
# Create virtual environment (one-time setup)
python3 -m venv ~/hf_cli_env
source ~/hf_cli_env/bin/activate
pip install "huggingface-hub[cli]"
```

## Method 1: Using HuggingFace CLI (Recommended)

### Basic Usage

```bash
# Activate the virtual environment
source ~/hf_cli_env/bin/activate

# Download all files (may include unwanted files like ONNX)
hf download MODEL_NAME --local-dir models/MODEL_DIR

# Download specific files only (cleaner)
hf download MODEL_NAME \
  --include "model.safetensors" "tokenizer.json" "config.json" \
  --local-dir models/MODEL_DIR
```

### Known Issues

**Issue**: The `--include` flag may be ignored when specific filenames are provided
- **Symptom**: Warning "Ignoring `--include` since filenames have being explicitly set"
- **Result**: Only JSON files downloaded, model.safetensors missing
- **Workaround**: Use method without `--include` and manually clean up unwanted files

**Issue**: Network interruptions
- **Symptom**: Download fails partway through with connection errors
- **Result**: Partial download (some files may still be downloaded)
- **Workaround**: Check what was downloaded, use wget for missing files

## Method 2: Using wget (More Reliable)

### Direct Download from HuggingFace

```bash
# Create directory
mkdir -p models/MODEL_DIR

# Download each file individually
wget -O models/MODEL_DIR/config.json \
  https://huggingface.co/MODEL_NAME/resolve/main/config.json

wget -O models/MODEL_DIR/tokenizer.json \
  https://huggingface.co/MODEL_NAME/resolve/main/tokenizer.json

wget -O models/MODEL_DIR/model.safetensors \
  https://huggingface.co/MODEL_NAME/resolve/main/model.safetensors
```

### Advantages of wget
- More reliable for large files (auto-resume with `-c` flag)
- Works without Python/pip dependencies
- Explicit control over what gets downloaded
- Better progress reporting for large files

### Resume interrupted downloads

```bash
wget -c -O models/MODEL_DIR/model.safetensors \
  https://huggingface.co/MODEL_NAME/resolve/main/model.safetensors
```

## Downloading Specific Models

### 1. all-MiniLM-L6-v2 (Small, Fast)

**Specs**: 384-dim, 6 layers, ~0.2 GB RAM, 512 token context

```bash
# Method 1: HuggingFace CLI
source ~/hf_cli_env/bin/activate
hf download sentence-transformers/all-MiniLM-L6-v2 \
  --local-dir models/all-MiniLM-L6-v2-safetensors

# Method 2: wget
mkdir -p models/all-MiniLM-L6-v2-safetensors
wget -O models/all-MiniLM-L6-v2-safetensors/config.json \
  https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/config.json
wget -O models/all-MiniLM-L6-v2-safetensors/tokenizer.json \
  https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer.json
wget -O models/all-MiniLM-L6-v2-safetensors/model.safetensors \
  https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/model.safetensors
```

### 2. intfloat/e5-small-v2 (Medium Quality)

**Specs**: 384-dim, 12 layers, ~0.5 GB RAM, 512 token context

```bash
# Method 1: HuggingFace CLI
source ~/hf_cli_env/bin/activate
hf download intfloat/e5-small-v2 \
  --local-dir models/e5-small-v2-safetensors

# Method 2: wget
mkdir -p models/e5-small-v2-safetensors
wget -O models/e5-small-v2-safetensors/config.json \
  https://huggingface.co/intfloat/e5-small-v2/resolve/main/config.json
wget -O models/e5-small-v2-safetensors/tokenizer.json \
  https://huggingface.co/intfloat/e5-small-v2/resolve/main/tokenizer.json
wget -O models/e5-small-v2-safetensors/model.safetensors \
  https://huggingface.co/intfloat/e5-small-v2/resolve/main/model.safetensors
```

### 3. intfloat/e5-base-v2 (Better Quality)

**Specs**: 768-dim, 12 layers, ~1.5 GB RAM, 512 token context

```bash
# Method 1: HuggingFace CLI
source ~/hf_cli_env/bin/activate
hf download intfloat/e5-base-v2 \
  --local-dir models/e5-base-v2-safetensors

# Method 2: wget
mkdir -p models/e5-base-v2-safetensors
wget -O models/e5-base-v2-safetensors/config.json \
  https://huggingface.co/intfloat/e5-base-v2/resolve/main/config.json
wget -O models/e5-base-v2-safetensors/tokenizer.json \
  https://huggingface.co/intfloat/e5-base-v2/resolve/main/tokenizer.json
wget -O models/e5-base-v2-safetensors/model.safetensors \
  https://huggingface.co/intfloat/e5-base-v2/resolve/main/model.safetensors
```

### 4. ModernBERT-base (Long Context)

**Specs**: 768-dim, 22 layers, ~0.6 GB RAM, 8192 token context

```bash
# Method 1: HuggingFace CLI
source ~/hf_cli_env/bin/activate
hf download answerdotai/ModernBERT-base \
  --local-dir models/modernbert-base-safetensors

# Method 2: wget
mkdir -p models/modernbert-base-safetensors
wget -O models/modernbert-base-safetensors/config.json \
  https://huggingface.co/answerdotai/ModernBERT-base/resolve/main/config.json
wget -O models/modernbert-base-safetensors/tokenizer.json \
  https://huggingface.co/answerdotai/ModernBERT-base/resolve/main/tokenizer.json
wget -O models/modernbert-base-safetensors/model.safetensors \
  https://huggingface.co/answerdotai/ModernBERT-base/resolve/main/model.safetensors
```

## Verifying Downloads

### Check file sizes

```bash
# Should show config.json (small), tokenizer.json (~2MB), model.safetensors (large)
ls -lh models/MODEL_DIR/

# Check model file size matches expected
du -h models/MODEL_DIR/model.safetensors
```

### Expected model.safetensors sizes
- all-MiniLM-L6-v2: ~90 MB
- e5-small-v2: ~128 MB
- e5-base-v2: ~440 MB
- ModernBERT-base: ~571 MB

## Organizing Downloaded Models

### Recommended directory structure

```
models/
├── all-MiniLM-L6-v2-safetensors/
│   ├── config.json
│   ├── tokenizer.json
│   └── model.safetensors
├── e5-small-v2-safetensors/
│   ├── config.json
│   ├── tokenizer.json
│   └── model.safetensors
├── e5-base-v2-safetensors/
│   ├── config.json
│   ├── tokenizer.json
│   └── model.safetensors
└── modernbert-base-safetensors/
    ├── config.json
    ├── tokenizer.json
    └── model.safetensors
```

### Cleaning up extra files

If you downloaded all files and want to keep only safetensors:

```bash
cd models/MODEL_DIR/

# Remove ONNX directory if present
rm -rf onnx/

# Remove PyTorch weights (we use safetensors)
rm -f pytorch_model.bin

# Keep only: config.json, tokenizer.json, model.safetensors, README.md
```

## Installing HuggingFace CLI

### Python Version Requirement

The HuggingFace CLI (`hf`) requires **Python 3.9 or newer**. Check your version:

```bash
python3 --version
```

### If you have Python 3.8 or older

Install Python 3.9 alongside your existing Python (Ubuntu/Debian):

```bash
# Add deadsnakes PPA for newer Python versions
sudo apt update
sudo apt install software-properties-common -y
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update

# Install Python 3.9 with venv support
sudo apt install python3.9 python3.9-venv python3.9-distutils -y

# Verify installation
python3.9 --version
```

### Option 1: Install in Virtual Environment (Recommended)

Using a virtual environment isolates the HuggingFace CLI from system packages:

```bash
# Create virtual environment with Python 3.9
python3.9 -m venv ~/hf_cli_env

# Activate the environment
source ~/hf_cli_env/bin/activate

# Install huggingface-hub with CLI
pip install -U "huggingface-hub[cli]"

# Verify installation
hf --version

# Deactivate when done
deactivate
```

**Usage after setup:**
```bash
# Always activate before using hf commands
source ~/hf_cli_env/bin/activate

# Run hf commands
hf download MODEL_NAME --local-dir DIR

# Deactivate when done
deactivate
```

### Option 2: System-wide Installation

If you have Python 3.9+ as your default `python3`:

```bash
pip3 install -U "huggingface-hub[cli]"
hf --version
```

### Option 3: Using the install script

```bash
# Set Python version explicitly
export PYTHON=python3.9
curl -LsSf https://hf.co/cli/install.sh | bash
```

### Verifying HuggingFace CLI Installation

```bash
# Check version
hf --version

# Should show: hf 0.x.x

# Test download (small file)
hf whoami
```

## Troubleshooting

### Python version error: "requires Python 3.9+"

**Symptom:**
```
ERROR: Package 'huggingface-hub' requires Python >=3.9
```

**Solution:**
1. Check Python version: `python3 --version`
2. If < 3.9, install Python 3.9 using the deadsnakes PPA (see "Installing HuggingFace CLI" above)
3. Create virtual environment with Python 3.9: `python3.9 -m venv ~/hf_cli_env`

### "hf: command not found"

**Cause**: Virtual environment not activated or hf not installed
**Solution**:
```bash
# Activate virtual environment
source ~/hf_cli_env/bin/activate

# Verify hf is available
which hf
hf --version
```

### Virtual environment activation issues

**Symptom**: `source ~/hf_cli_env/bin/activate` fails

**Common causes:**
1. Virtual environment not created yet
   ```bash
   python3.9 -m venv ~/hf_cli_env
   ```

2. Wrong path - verify location:
   ```bash
   ls -la ~/hf_cli_env/bin/activate
   ```

### "model.safetensors not found" error

**Cause**: Only JSON files were downloaded
**Solution**:
1. Check if file exists: `ls models/MODEL_DIR/model.safetensors`
2. If missing, use wget to download it directly (see Method 2 above)

### --include flag ignored warning

**Symptom:**
```
UserWarning: Ignoring `--include` since filenames have being explicitly set.
```

**Cause**: Bug or quirk in hf CLI behavior
**Result**: Only config/tokenizer downloaded, model.safetensors missing
**Solution**: Download all files (without `--include`), then remove unwanted files:
```bash
# Download everything
source ~/hf_cli_env/bin/activate
hf download MODEL_NAME --local-dir models/MODEL_DIR

# Remove unwanted files
cd models/MODEL_DIR
rm -rf onnx/
rm -f pytorch_model.bin
```

### Network timeouts during download

**Cause**: Large file download interrupted
**Solution**:
1. Use wget with `-c` flag to resume
2. Or re-run the hf command (it should resume from where it left off)

### Connection errors with hf download

**Symptom:**
```
ConnectError: [Errno -3] Temporary failure in name resolution
```

**Causes:**
1. WSL networking issue
2. Internet connection problem
3. DNS resolution failure

**Solutions:**
1. Test basic connectivity: `ping huggingface.co`
2. If DNS fails, try Google DNS:
   ```bash
   # Edit /etc/resolv.conf (temporary fix)
   echo "nameserver 8.8.8.8" | sudo tee /etc/resolv.conf
   ```
3. Use wget as alternative (more robust for large files):
   ```bash
   wget -c -O models/MODEL_DIR/model.safetensors \
     https://huggingface.co/MODEL_NAME/resolve/main/model.safetensors
   ```

### Permission errors

**Cause**: Missing write permissions in models directory
**Solution**:
```bash
mkdir -p models
chmod 755 models
```

### WSL-specific issues

**Ubuntu version too old**: If your WSL is Ubuntu 20.04 or older, you may need to:
1. Install Python 3.9 via deadsnakes PPA (see above)
2. Or upgrade WSL distro to Ubuntu 22.04+ (ships with Python 3.10+):
   ```bash
   # Check current version
   lsb_release -a

   # Upgrade is complex - consider fresh Ubuntu 22.04 WSL install instead
   ```

## Notes

- The `-safetensors` suffix in directory names distinguishes safetensors format from ONNX or other formats
- Safetensors is the preferred format for Candle (faster loading, safer)
- Always verify downloaded file sizes match expectations
- HuggingFace CLI may download extra files (ONNX, PyTorch) - these can be removed
