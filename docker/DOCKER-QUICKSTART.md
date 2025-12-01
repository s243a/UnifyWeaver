# Docker Quick Start

## First Time Setup

```bash
# 1. Navigate to docker directory
cd docker

# 2. Build the development image
./docker-dev.sh build

# 3. Start and enter the container
./docker-dev.sh run
```

## Daily Usage

```bash
# Start container
./docker-dev.sh run

# Inside container - run tests
test-stream
test-recursive
test-advanced

# Exit container
exit

# Stop container when done
./docker-dev.sh stop
```

## Common Tasks

### Run All Tests
```bash
./docker-dev.sh test
```

### Access Running Container
```bash
./docker-dev.sh shell
```

### View Logs
```bash
./docker-dev.sh logs
```

### Clean Restart
```bash
./docker-dev.sh clean
./docker-dev.sh build
./docker-dev.sh run
```

## Windows Testing (Wine)

```bash
# Build Wine image (first time only)
./docker-wine.sh build

# Start Wine container
./docker-wine.sh run

# Inside container - test Windows scripts
cd scripts
wine cmd /c start_unifyweaver_windows.bat

# Or use PowerShell
wine-powershell -File scripts/testing/Init-TestEnvironment.ps1
```

## File Locations

- **Project**: `/mnt/c/Users/johnc/Dropbox/projects/UnifyWeaver`
- **Output**: `/mnt/c/Users/johnc/Dropbox/projects/UnifyWeaver/output`

Same paths as WSL for Claude Code compatibility!

## Troubleshooting

**Container won't start?**
```bash
docker ps -a
./docker-dev.sh logs
```

**Need to rebuild?**
```bash
./docker-dev.sh clean
./docker-dev.sh build
```

**Wine not working?**
```bash
docker exec -it unifyweaver-wine wine wineboot --init
```

## More Info

See [DOCKER.md](DOCKER.md) for complete documentation.
