# UnifyWeaver Scripts

This directory contains launcher scripts and utilities for running UnifyWeaver on different platforms.

## Quick Start

**Windows users:** Double-click one of these:
- `start_unifyweaver_windows.bat` - Native Windows (recommended for most users)
- `start_unifyweaver_cygwin.bat` - Cygwin/MSYS2/WSL environments

**Linux/macOS users:** Run from terminal:
```bash
bash unifyweaver_console.sh
```

---

## Windows Launchers

### `start_unifyweaver_windows.bat` ⭐ NEW
**Pure Windows launcher - no Cygwin/WSL required**

Uses native Windows SWI-Prolog directly, making it the simplest option for Windows users.

**Features:**
- No dependencies on Cygwin, MSYS2, or WSL
- Automatic SWI-Prolog detection and PATH configuration
- **ConEmu terminal support** for enhanced console experience
- Falls back to standard Windows console if ConEmu not installed
- Helper commands pre-loaded (load_stream, test_advanced, etc.)

**Requirements:**
- SWI-Prolog for Windows: https://www.swi-prolog.org/download/stable
- Optional: ConEmu for better terminal: https://conemu.github.io/

**Usage:**
```cmd
REM Just double-click the .bat file
REM Or run from command prompt:
start_unifyweaver_windows.bat
```

**ConEmu Integration:**
The launcher automatically detects ConEmu in these locations:
- `%ProgramFiles%\ConEmu\ConEmu64.exe`
- `%ProgramFiles(x86)%\ConEmu\ConEmu.exe`
- `%LocalAppData%\ConEmu\ConEmu64.exe`

If found, launches in ConEmu with proper working directory and syntax highlighting.

---

### `start_unifyweaver_cygwin.bat`
**Unix-like environment launcher (Cygwin/MSYS2/WSL)**

For users who prefer Unix tools on Windows or need bash-based workflows.

**Automatically detects and uses (in priority order):**
1. Cygwin (with ConEmu support ⭐ NEW)
2. MSYS2 (with ConEmu support ⭐ NEW)
3. WSL (Windows Subsystem for Linux)

**Features:**
- Uses `setlocal EnableExtensions EnableDelayedExpansion` for proper variable capture
- Converts Windows paths to POSIX using `cygpath`
- Adds SWI-Prolog to PATH automatically
- **ConEmu terminal support** for Cygwin and MSYS2
- Launches bash console with UnifyWeaver environment

**Usage:**
```cmd
REM Just double-click the .bat file
REM Or run from command prompt:
start_unifyweaver_cygwin.bat
```

**Note:** ConEmu integration works with Cygwin and MSYS2. For WSL, use Windows Terminal for best experience.

## Console Launcher

### `unifyweaver_console.sh`
**Bash script that launches SWI-Prolog with UnifyWeaver environment**

Called by the `.bat` launcher but can also be run directly in Unix environments.

**Features:**
- Auto-configures library paths
- Defines helper predicates
- Works in Cygwin, MSYS2, WSL, and native Linux

**Usage:**
```bash
# From project root
bash scripts/unifyweaver_console.sh

# Or if UNIFYWEAVER_ROOT is set
export UNIFYWEAVER_ROOT=/path/to/project
bash scripts/unifyweaver_console.sh
```

**Available commands in console:**
- `load_recursive.` - Load recursive_compiler
- `load_stream.` - Load stream_compiler
- `load_template.` - Load template_system
- `load_all_core.` - Load all core modules
- `test_advanced.` - Run advanced recursion tests
- `help.` - Show help

## Testing Scripts

### `testing/init_testing.sh`
**Initialize a test environment (Bash)**

Creates a standalone testing directory with all UnifyWeaver modules. Works on Linux, WSL, Cygwin, and MSYS2.

**Usage:**
```bash
cd scripts/testing
./init_testing.sh

# Or specify custom location
UNIFYWEAVER_ROOT=/tmp/my_test ./init_testing.sh

# Force Windows SWI-Prolog for testing
./init_testing.sh --force-windows
```

### `testing/Init-TestEnvironment.ps1` ⭐ NEW
**Initialize a test environment (PowerShell)**

Pure Windows version of init_testing.sh for users who prefer PowerShell or don't have bash available.

**Requirements:**
- PowerShell 5.1 or later
- SWI-Prolog for Windows

**Note:** If you get an execution policy error, run:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Usage:**
```powershell
cd scripts\testing
.\Init-TestEnvironment.ps1

# Specify custom location
.\Init-TestEnvironment.ps1 -TargetDir "C:\UnifyWeaver\test"

# Get help
.\Init-TestEnvironment.ps1 -Help
```

**Features:**
- Pure PowerShell - no bash required
- Same functionality as init_testing.sh
- Uses robocopy for fast file copying
- Creates Windows batch launcher
- Generates helpful README.txt

### `testing/find_swi-prolog.sh`
**Detect and configure SWI-Prolog**

Automatically sourced by `init_testing.sh`. Prioritizes native Linux SWI-Prolog over Windows version in WSL for better readline support (arrow keys).

---

## Terminal Customization

### ConEmu Terminal Emulator

ConEmu provides a vastly improved console experience on Windows with features like:
- Tabs and split panes
- Better copy/paste (with mouse support)
- 24-bit color support
- Customizable fonts and themes
- Unicode support
- Saved sessions

**Installation:**
Download from: https://conemu.github.io/

**Usage with UnifyWeaver:**
ConEmu support is automatically integrated into:
- `start_unifyweaver_windows.bat` (native Windows)
- `start_unifyweaver_cygwin.bat` (Cygwin/MSYS2)

Just install ConEmu and run the batch files - they'll detect and use it automatically!

**Manual ConEmu Setup:**
If you want to create a dedicated ConEmu task:

1. Open ConEmu Settings (Win+Alt+P)
2. Go to "Startup" → "Tasks"
3. Add new task with command:
```
cmd /k "%ProgramFiles%\swipl\bin\swipl.exe" -q -g "asserta(library_directory('C:/path/to/UnifyWeaver/src'))" -t prolog
```

**Legacy ConEmu Script:**
See `legacy/conemu_logiforge.bat` for the original LogiForge ConEmu integration pattern used as reference for the current implementation.

---

## Platform Recommendations

### Windows Users

**Recommended Setup:**
1. Install SWI-Prolog for Windows
2. Install ConEmu (optional but recommended)
3. Use `start_unifyweaver_windows.bat`

**Alternative (Unix-like):**
1. Install Cygwin or MSYS2
2. Install ConEmu (optional)
3. Use `start_unifyweaver_cygwin.bat`

### Linux/macOS Users

**Recommended Setup:**
1. Install SWI-Prolog from package manager
2. Run `bash unifyweaver_console.sh`

### WSL Users

**Recommended Setup:**
1. Install native Linux SWI-Prolog in WSL: `sudo apt install swi-prolog`
2. Use Windows Terminal for best experience
3. Run `bash unifyweaver_console.sh`

---

## LDA Training Scripts

Scripts for training and managing the LDA semantic projection system used by agents to find relevant playbook examples.

### `train_lda_projection.py`
**Train a W matrix from Q-A pairs**

Computes the projection matrix that maps query embeddings to answer space for improved semantic search.

```bash
python3 scripts/train_lda_projection.py \
    --input playbooks/lda-training-data/raw/qa_pairs_v1.json \
    --model all-MiniLM-L6-v2 \
    --output playbooks/lda-training-data/trained/all-MiniLM-L6-v2/W_matrix.npy
```

### `validate_lda_projection.py`
**Validate projection with novel queries**

Tests the trained W matrix with queries not in the training data, comparing projected vs direct cosine similarity.

```bash
python3 scripts/validate_lda_projection.py
```

### `migrate_to_lda_db.py`
**Database migration and batch training**

Manages Q-A training data in a SQLite database with batch tracking.

**Single file import:**
```bash
python3 scripts/migrate_to_lda_db.py \
    --input playbooks/lda-training-data/raw/qa_pairs_v1.json \
    --db playbooks/lda-training-data/lda.db
```

**Batch operations:**
```bash
# Scan for new/modified JSON files
python3 scripts/migrate_to_lda_db.py --scan --input playbooks/lda-training-data/raw/

# Process all pending batches
python3 scripts/migrate_to_lda_db.py --process-pending

# Retry failed batches
python3 scripts/migrate_to_lda_db.py --retry-failed

# List batch status
python3 scripts/migrate_to_lda_db.py --list-batches
```

**Features:**
- SHA256 file hash detection (skips unchanged files)
- Status tracking: pending → importing → embedding → training → completed
- Failed batch retry with error logging
- Full status history with timestamps

**Requirements:**
- `sentence-transformers` package: `pip install sentence-transformers`
- NumPy

See `docs/proposals/LDA_DATABASE_SCHEMA.md` for database schema details.

### `train_multi_head_projection.py`
**Train multi-head projection with per-cluster routing**

Creates a multi-head projection where each cluster acts as an "attention head". Queries are routed to heads based on similarity to cluster centroids using softmax.

```bash
python3 scripts/train_multi_head_projection.py \
    --db playbooks/lda-training-data/lda.db \
    --model all-MiniLM-L6-v2 \
    --temperature 0.1 \
    --validate
```

**Key parameters:**
- `--temperature`: Softmax temperature for routing (default: 1.0, recommended: 0.1)
  - Lower = sharper routing (best match dominates)
  - Higher = softer routing (more blending between heads)

### `validate_multi_head.py`
**Validate multi-head projection with novel queries**

Compares multi-head projection against direct cosine similarity on queries not in training data.

```bash
python3 scripts/validate_multi_head.py \
    --db playbooks/lda-training-data/lda.db \
    --mh-id 1
```

**Results (temp=0.1):**
- Multi-head: 76.7% Recall@1
- Direct: 70.0% Recall@1
- Improvement: +6.7%

---

## Sparse Routing for Memory-Efficient Projections

### `sparse_routing.py`

Memory-efficient embedding projections using hierarchical sparse routing. Achieves **95%+ memory reduction** compared to loading all embeddings.

**Key Components:**

| Component | Description |
|-----------|-------------|
| `RepresentativeConfig` | Configuration: max_reps, condition_threshold, subspace_multiple |
| `select_representatives()` | Priority selection: centroid → trees → SVD-generated |
| `LazyWMatrixLoader` | LRU-cached, memory-mapped W matrix loading |
| `compute_routing_weights()` | Softmax routing over representatives |
| `SparseRouter` | Full pipeline class |

**Memory Savings (290 clusters, 768-dim):**

| Component | Before | After | Reduction |
|-----------|--------|-------|-----------|
| W matrices | 652 MB | 11 MB | 98% |
| Representatives | 300 MB | 34 MB | 89% |
| Total | ~1 GB | ~45 MB | **95%** |

**Usage:**

```python
from scripts.sparse_routing import SparseRouter, RepresentativeConfig

# Configure representative selection
config = RepresentativeConfig(
    max_reps_per_cluster=30,      # Hard memory limit
    condition_threshold=10.0,      # Dominant subspace (κ ≤ 10)
    subspace_multiple=1.0,         # 1× dominant dimensions
)

# Initialize router
router = SparseRouter(
    centroids_path='models/centroids.npz',
    w_matrices_path='models/w_stack.npz',
    top_k_clusters=5,              # Route over top 5 clusters
    w_cache_size=10,               # Cache 10 W matrices
)

# Project query
query = np.random.randn(768)
projected = router.project(query)

# Check memory usage
print(router.memory_usage_mb())
```

**CLI Integration:**

The sparse routing is used by the harvester (`batch_repair.py`) for memory-efficient queue sorting. See `.local/tools/browser-automation/proposals/MEMORY_SAVINGS_HARVESTING.md` for design details.

---

## Legacy Scripts (LogiForge)

The following scripts are from the LogiForge project and kept in `legacy/` for reference:
- `logiforge_console.sh` - Original console launcher
- `logiforge_console_old.sh` - Older version
- `logiforge_console_simp.sh` - Simplified version
- `logiforge_cygwin.sh` - Cygwin-specific version
- `start_logiforge_cygwin.bat` - Original Windows launcher
- `start_logiforge_cygwin_back.bat` - Backup version
- `conemu_logiforge.bat` - ConEmu specific launcher

**Key techniques preserved from legacy:**
- `EnableDelayedExpansion` for proper variable capture in batch files
- `!variable!` delayed expansion syntax
- ConEmu detection and fallback patterns
- Path conversion with cygpath

See `legacy/README.md` for detailed documentation of these techniques.
