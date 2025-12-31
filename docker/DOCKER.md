# UnifyWeaver Docker Development Environment

This directory contains Docker configurations for running UnifyWeaver in containerized environments. Three configurations are provided:

1. **Standard Development Environment** - Ubuntu-based with SWI-Prolog, Node.js, and development tools
2. **PowerShell Environment** - Same as standard plus native Linux PowerShell 7.4
3. **Wine Environment** - Same as standard plus Wine and Windows PowerShell for Windows testing

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) installed
- [Docker Compose](https://docs.docker.com/compose/install/) installed (or use `docker compose`)

## Quick Start

### Standard Development Environment

```bash
# Build the image
./docker-dev.sh build

# Start the container and open a shell
./docker-dev.sh run

# Run tests
./docker-dev.sh test
```

### PowerShell Environment (for PowerShell testing)

```bash
# Build the PowerShell image
./docker-pwsh.sh build

# Start the PowerShell container
./docker-pwsh.sh run

# Run tests
./docker-pwsh.sh test
```

### Wine Environment (for Windows testing)

```bash
# Build the Wine image
./docker-wine.sh build

# Start the Wine container
./docker-wine.sh run

# Install SWI-Prolog for Windows in Wine (optional)
./docker-wine.sh install-swipl-windows

# Test Windows batch scripts
./docker-wine.sh test-windows
```

## Docker Files

### Dockerfile
Main development environment with:
- **Ubuntu 22.04** base
- **SWI-Prolog** (latest stable from PPA)
- **Node.js 20.x** (LTS)
- **Claude Code CLI** (`@anthropic-ai/claude-code`)
- **Python 3** with pip
- **Development tools**: git, vim, nano, build-essential, cmake
- **Text processing**: grep, sed, awk, ripgrep
- **Bash** with custom aliases for testing

### Dockerfile.pwsh
PowerShell environment with all of the above plus:
- **PowerShell 7.4** (native Linux version)
- Lightweight alternative to Wine (~500MB smaller)
- Best for quick PowerShell script testing

### Dockerfile.wine
Wine-enabled environment with all of the above plus:
- **Wine** (stable branch)
- **Winetricks** for Wine configuration
- **.NET Framework 4.8** via Wine
- **PowerShell 7.4 for Windows** (running under Wine)
- **Xvfb** for virtual display
- Helper script for installing Windows SWI-Prolog

### docker-compose.yml
Orchestrates all three containers with:
- Proper volume mounting maintaining WSL path structure (`/mnt/c/...`)
- Persistent volumes for home directories
- Network configuration for inter-container communication

## Volume Mounting

The Docker setup preserves your WSL path structure:

```
Host:      /mnt/c/Users/johnc/Dropbox/projects/UnifyWeaver
Container: /mnt/c/Users/johnc/Dropbox/projects/UnifyWeaver
```

This ensures Claude Code context is maintained since the paths match exactly.

## Available Commands

### docker-dev.sh

```bash
./docker-dev.sh build      # Build the image
./docker-dev.sh run        # Start and enter container
./docker-dev.sh start      # Alias for 'run'
./docker-dev.sh stop       # Stop the container
./docker-dev.sh restart    # Restart the container
./docker-dev.sh clean      # Remove container and volumes
./docker-dev.sh logs       # View container logs
./docker-dev.sh test       # Run all UnifyWeaver tests
./docker-dev.sh shell      # Open shell in running container
```

### docker-pwsh.sh

```bash
./docker-pwsh.sh build     # Build the PowerShell image
./docker-pwsh.sh run       # Start and enter container
./docker-pwsh.sh start     # Alias for 'run'
./docker-pwsh.sh stop      # Stop the container
./docker-pwsh.sh restart   # Restart the container
./docker-pwsh.sh clean     # Remove container and volumes
./docker-pwsh.sh logs      # View container logs
./docker-pwsh.sh test      # Run all UnifyWeaver tests
./docker-pwsh.sh shell     # Open shell in running container
```

### docker-wine.sh

```bash
./docker-wine.sh build                 # Build the Wine image
./docker-wine.sh run                   # Start and enter container
./docker-wine.sh install-swipl-windows # Install Windows SWI-Prolog
./docker-wine.sh test-windows          # Test Windows batch scripts
./docker-wine.sh wine-cmd              # Open Wine command prompt
./docker-wine.sh shell                 # Open shell in running container
```

## Built-in Aliases

Once inside the container, you can use these aliases:

```bash
ll              # ls -lah
swipl           # swipl -q (quiet mode)
test-stream     # Run stream compiler tests
test-recursive  # Run recursive compiler tests
test-advanced   # Run advanced recursion tests
```

**PowerShell container:**
```bash
pwsh            # Start PowerShell 7.4 (native Linux)
```

**Wine container only:**
```bash
wine-cmd        # wine cmd
wine-powershell # wine powershell (PowerShell via Wine)
```

## Using Docker Compose Directly

You can also use docker-compose commands directly:

```bash
# Start all containers
docker-compose up -d

# Start only dev container
docker-compose up -d unifyweaver-dev

# Start only PowerShell container
docker-compose up -d unifyweaver-pwsh

# Start only Wine container
docker-compose up -d unifyweaver-wine

# Stop all containers
docker-compose down

# View logs
docker-compose logs -f

# Execute command in container
docker-compose exec unifyweaver-dev swipl
docker-compose exec unifyweaver-pwsh pwsh
```

## Development Workflow

### 1. Standard Development (Linux-only)

```bash
# Start container
./docker-dev.sh run

# Inside container:
cd /mnt/c/Users/johnc/Dropbox/projects/UnifyWeaver

# Run tests
test-stream
test-recursive
test-advanced

# Or manually:
swipl
?- use_module('src/unifyweaver/core/stream_compiler').
?- test_stream_compiler.
```

### 2. PowerShell Testing (Quick)

```bash
# Start PowerShell container
./docker-pwsh.sh run

# Inside container:
cd /mnt/c/Users/johnc/Dropbox/projects/UnifyWeaver

# Start PowerShell
pwsh

# In PowerShell:
PS> cd scripts/testing
PS> ./Init-TestEnvironment.ps1
```

### 3. Windows Testing with Wine

```bash
# Start Wine container
./docker-wine.sh run

# Inside container:
cd scripts

# Test batch files
wine cmd /c start_unifyweaver_windows.bat

# Or use PowerShell scripts (Windows version via Wine)
wine powershell -File testing/Init-TestEnvironment.ps1
```

## Claude Code Integration

The Docker containers preserve your WSL path structure, so Claude Code context should work seamlessly:

1. Your files are at: `/mnt/c/Users/johnc/Dropbox/projects/UnifyWeaver`
2. Claude Code can reference files with the same paths
3. All edits are immediately visible on the host

To use Claude Code inside the container:

```bash
# Inside container
claude-code
```

## Persistent Data

The following directories persist across container restarts:

- **unifyweaver-home**: `/root` in dev container (bash history, configs)
- **unifyweaver-pwsh-home**: `/root` in PowerShell container (bash/pwsh history, configs)
- **unifyweaver-wine-home**: `/root` in Wine container (Wine prefix, configs)

Project files are always mounted from the host, so changes persist automatically.

## Troubleshooting

### Container won't start
```bash
# Check Docker daemon
docker ps

# Check logs
./docker-dev.sh logs
```

### Wine issues
```bash
# Reinitialize Wine
docker exec -it unifyweaver-wine wine wineboot --init

# Check Wine version
docker exec -it unifyweaver-wine wine --version
```

### Permission issues
```bash
# Fix git safe directory
docker exec -it unifyweaver-dev git config --global --add safe.directory '*'
```

### Rebuild from scratch
```bash
# Remove everything
docker-compose down -v
docker rmi unifyweaver-dev unifyweaver-pwsh unifyweaver-wine

# Rebuild
./docker-dev.sh build
./docker-pwsh.sh build
./docker-wine.sh build
```

## Performance Notes

- **First build**: Takes 5-10 minutes to download and install all dependencies
- **Subsequent builds**: Much faster due to Docker layer caching
- **Wine initialization**: First Wine startup takes ~30 seconds

## Customization

### Adding packages

Edit `Dockerfile` or `Dockerfile.wine` and rebuild:

```dockerfile
RUN apt-get update && apt-get install -y \
    your-package-here
```

### Changing Node.js version

Modify the Node.js setup line in Dockerfile:

```dockerfile
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash -
```

### Adding more aliases

Edit the RUN command that adds to `.bashrc`:

```dockerfile
RUN echo 'alias myalias="my command"' >> /root/.bashrc
```

## Security Notes

- Containers run as `root` for simplicity in development
- Wine environment allows running Windows executables
- All mounted directories are writable from container
- Use only for development, not production

## Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [Wine Documentation](https://www.winehq.org/documentation)
- [SWI-Prolog Docker](https://hub.docker.com/_/swipl)
- [Claude Code Documentation](https://docs.claude.com/claude-code)
