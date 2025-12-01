# UnifyWeaver Docker Environment

Docker-based development environment for UnifyWeaver with three container options.

## Quick Start

```bash
# From project root
cd docker

# Option 1: Standard development (most common)
./docker-dev.sh build
./docker-dev.sh run

# Option 2: PowerShell testing (lightweight, no Wine)
./docker-pwsh.sh build
./docker-pwsh.sh run

# Option 3: Windows testing with Wine
./docker-wine.sh build
./docker-wine.sh run
```

## What's Included

### Standard Development Container
- Ubuntu 22.04
- SWI-Prolog (latest stable)
- Node.js 20.x LTS
- Claude Code CLI
- Python 3
- Git, Vim, Nano
- Text processing tools (grep, sed, awk, ripgrep)

### PowerShell Container (Lightweight)
- Everything from standard container
- **Native Linux PowerShell 7.4**
- **No Wine** - faster and lighter
- Good for PowerShell script testing
- **~500MB smaller than Wine container**

### Wine Container (Windows Testing)
- Everything from standard container
- Wine (stable)
- Winetricks
- **PowerShell 7.4 for Windows** (via Wine)
- .NET Framework 4.8
- Virtual display support
- Tests actual Windows behavior

## Directory Structure

```
docker/
├── Dockerfile              # Main development image
├── Dockerfile.pwsh         # PowerShell (native Linux)
├── Dockerfile.wine         # Wine-enabled image
├── docker-compose.yml      # Container orchestration
├── docker-dev.sh          # Development launcher
├── docker-pwsh.sh         # PowerShell launcher
├── docker-wine.sh         # Wine launcher
├── .dockerignore          # Build optimization
├── DOCKER.md              # Full documentation
├── DOCKER-QUICKSTART.md   # Quick reference
└── README.md              # This file
```

## Usage

See [DOCKER-QUICKSTART.md](DOCKER-QUICKSTART.md) for common commands.

See [DOCKER.md](DOCKER.md) for complete documentation.

## Path Mounting

The container mounts your project at:
```
/mnt/c/Users/johnc/Dropbox/projects/UnifyWeaver
```

This matches your WSL path exactly, preserving Claude Code context.

## PowerShell Options

### Native PowerShell (Recommended for Testing)

```bash
# Start PowerShell container (lightweight, no Wine)
./docker-pwsh.sh run

# Inside container
pwsh                                              # Start PowerShell
pwsh -File scripts/testing/Init-TestEnvironment.ps1   # Run script
pwsh-init                                         # Alias for above
```

### PowerShell in Wine (For Windows Behavior Testing)

```bash
# Start Wine container
./docker-wine.sh run

# Inside Wine container
wine-powershell          # Alias for: wine pwsh
pwsh-wine               # Alternative alias

# Test PowerShell scripts as they would run on Windows
wine pwsh -File scripts/testing/Init-TestEnvironment.ps1
```

## Which Container Should I Use?

- **docker-dev.sh**: Daily development, running Prolog tests
- **docker-pwsh.sh**: Testing PowerShell scripts quickly
- **docker-wine.sh**: Testing Windows batch/PowerShell scripts for Windows compatibility

## Note About .gitignore

The `docker/` directory is ignored by git by default. To track Docker configs in git:

```bash
# Edit .gitignore and remove or comment out:
# docker/

# Then commit:
git add docker/
git commit -m "Add Docker configuration"
```

## Persistence Modes

Each launcher supports two modes:

1. **Standard Mode** (default): Fast, packages reset on rebuild
2. **Persistent Mode**: Installed packages survive container restarts

```bash
# Interactive menu on first run
./docker-dev.sh run

# Or use flags
./docker-dev.sh run --persistent
./docker-dev.sh run --standard

# Configure default
./docker-dev.sh config
```

See [PERSISTENCE.md](PERSISTENCE.md) for details.

## More Information

- [Full Documentation](DOCKER.md)
- [Quick Reference](DOCKER-QUICKSTART.md)
- [Persistence Modes](PERSISTENCE.md)
