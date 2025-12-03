# Docker Persistence Modes

Each Docker launcher script supports two persistence modes for system packages.

## Modes

### Standard Mode (Default)
- **What persists**: Home directory (`~/.bashrc`, history, configs)
- **What resets**: System packages installed via `apt-get`, `pip`, `npm`
- **Best for**: Daily development with pre-configured environment
- **Advantages**:
  - Fast, consistent environment
  - Always starts fresh from Dockerfile
  - Smaller disk usage

### Persistent System Mode
- **What persists**: Everything from standard mode PLUS `/usr/local` and `/opt`
- **Packages persist**: Anything installed via `apt-get`, `pip`, `npm`, manual builds
- **Best for**: Testing different versions of tools (e.g., different SWI-Prolog versions)
- **Advantages**:
  - Install packages once, use across container restarts
  - Test multiple tool versions without rebuilding images
  - Useful for experimental package installations

## Usage

### Interactive Menu
```bash
# First run shows menu
./docker-dev.sh run

Select mode [1-3] (default: 1):
1) Standard Mode
2) Persistent System Mode
3) Save choice to config file
```

### Command-Line Flags
```bash
# One-time persistent mode
./docker-dev.sh run --persistent

# One-time standard mode (override config)
./docker-dev.sh run --standard
```

### Config File
```bash
# Configure persistence mode
./docker-dev.sh config

# Config saved to docker/.docker-config
# Applies to all containers (dev, pwsh, wine)
```

## Example: Testing Different SWI-Prolog Versions

```bash
# Start in persistent mode
./docker-dev.sh run --persistent

# Inside container - install different SWI-Prolog version
apt-add-repository ppa:swi-prolog/devel
apt-get update
apt-get install -y swi-prolog

# Exit and restart - new version still there!
exit
./docker-dev.sh run
```

## Persisted Volumes

### Standard Mode
- `unifyweaver-home:/root` (dev container)
- `unifyweaver-pwsh-home:/root` (PowerShell container)
- `unifyweaver-wine-home:/root` (Wine container)

### Persistent Mode (Additional)
- `unifyweaver-dev-usr-local:/usr/local`
- `unifyweaver-dev-opt:/opt`
- `unifyweaver-pwsh-usr-local:/usr/local`
- `unifyweaver-pwsh-opt:/opt`
- `unifyweaver-wine-usr-local:/usr/local`
- `unifyweaver-wine-opt:/opt`

## Cleaning Up

### Remove all volumes (including persistent packages)
```bash
./docker-dev.sh clean
```

### Remove only persistent system volumes
```bash
docker volume rm unifyweaver-dev-usr-local unifyweaver-dev-opt
docker volume rm unifyweaver-pwsh-usr-local unifyweaver-pwsh-opt
docker volume rm unifyweaver-wine-usr-local unifyweaver-wine-opt
```

### Remove config file (show menu again)
```bash
rm docker/.docker-config
```

## Technical Details

Standard mode uses only `docker-compose.yml`:
```yaml
volumes:
  - unifyweaver-home:/root
```

Persistent mode adds `docker-compose.persistent.yml`:
```yaml
volumes:
  - unifyweaver-dev-usr-local:/usr/local
  - unifyweaver-dev-opt:/opt
```

The launcher scripts automatically use both compose files when persistent mode is selected:
```bash
docker-compose -f docker-compose.yml -f docker-compose.persistent.yml up
```
