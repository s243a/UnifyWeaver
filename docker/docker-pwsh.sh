#!/bin/bash
# UnifyWeaver Docker PowerShell Environment Launcher
# Starts the main development container with proper WSL path mounting

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_FILE="$SCRIPT_DIR/.docker-config"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}╔════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  UnifyWeaver PowerShell Environment (Docker)     ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════╝${NC}"
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${YELLOW}[ERROR] Docker is not installed${NC}"
    echo "Please install Docker: https://docs.docker.com/get-docker/"
    exit 1
fi

# Change to docker directory for docker-compose
cd "$SCRIPT_DIR"

# Check if docker-compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo -e "${YELLOW}[WARNING] docker-compose not found, trying 'docker compose'${NC}"
    DOCKER_COMPOSE="docker compose"
else
    DOCKER_COMPOSE="docker-compose"
fi

# Load shared functions
source "$SCRIPT_DIR/.docker-functions.sh"

# Default persistence mode
PERSISTENT_MODE="standard"

# Load config file if it exists
if [[ -f "$CONFIG_FILE" ]]; then
    source "$CONFIG_FILE"
fi

# Parse command line arguments
ACTION="${1:-run}"
shift || true

# Check for --persistent or --standard flags
while [[ $# -gt 0 ]]; do
    case $1 in
        --persistent)
            PERSISTENT_MODE="persistent"
            shift
            ;;
        --standard)
            PERSISTENT_MODE="standard"
            shift
            ;;
        *)
            shift
            ;;
    esac
done

case "$ACTION" in
    build)
        echo -e "${BLUE}[INFO] Building UnifyWeaver development image...${NC}"
        $DOCKER_COMPOSE build unifyweaver-pwsh
        echo -e "${GREEN}[SUCCESS] Build complete${NC}"
        ;;

    run|start)
        # Show menu if no flag was provided and no config exists
        if [[ ! -f "$CONFIG_FILE" ]] && [[ "$PERSISTENT_MODE" == "standard" ]] && [[ "$1" != "--persistent" ]]; then
            show_persistence_menu "$CONFIG_FILE" "$PERSISTENT_MODE"
        fi

        echo -e "${BLUE}[INFO] Starting UnifyWeaver development container...${NC}"
        echo -e "${BLUE}[INFO] Mode: ${PERSISTENT_MODE}${NC}"
        echo -e "${BLUE}[INFO] Project: ${PROJECT_ROOT}${NC}"
        echo -e "${BLUE}[INFO] Mounting as: /mnt/c/Users/johnc/Dropbox/projects/UnifyWeaver${NC}"
        echo ""

        # Check if container is already running
        if docker ps | grep -q unifyweaver-pwsh; then
            echo -e "${YELLOW}[INFO] Container already running, attaching...${NC}"
            docker exec -it unifyweaver-pwsh /bin/bash
        else
            # Start container with appropriate compose file
            if [[ "$PERSISTENT_MODE" == "persistent" ]]; then
                $DOCKER_COMPOSE -f docker-compose.yml -f docker-compose.persistent.yml up -d unifyweaver-pwsh
            else
                $DOCKER_COMPOSE up -d unifyweaver-pwsh
            fi
            echo -e "${GREEN}[SUCCESS] Container started${NC}"
            echo ""
            echo -e "${BLUE}[INFO] Entering container shell...${NC}"
            docker exec -it unifyweaver-pwsh /bin/bash
        fi
        ;;

    stop)
        echo -e "${BLUE}[INFO] Stopping UnifyWeaver development container...${NC}"
        $DOCKER_COMPOSE stop unifyweaver-pwsh
        echo -e "${GREEN}[SUCCESS] Container stopped${NC}"
        ;;

    restart)
        echo -e "${BLUE}[INFO] Restarting UnifyWeaver development container...${NC}"
        $DOCKER_COMPOSE restart unifyweaver-pwsh
        $0 run
        ;;

    clean)
        echo -e "${YELLOW}[WARNING] This will remove the container and volumes${NC}"
        read -p "Are you sure? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            $DOCKER_COMPOSE down -v
            echo -e "${GREEN}[SUCCESS] Containers and volumes removed${NC}"
        fi

# Change to docker directory for docker-compose
cd "$SCRIPT_DIR"
        ;;

    logs)
        $DOCKER_COMPOSE logs -f unifyweaver-pwsh
        ;;

    test)
        echo -e "${BLUE}[INFO] Running UnifyWeaver tests in container...${NC}"
        docker exec -it unifyweaver-pwsh bash -c "
            echo '=== Stream Compiler Tests ===' && \
            swipl -g 'use_module(src/unifyweaver/core/stream_compiler), test_stream_compiler, halt.' && \
            echo '' && \
            echo '=== Recursive Compiler Tests ===' && \
            swipl -g 'use_module(src/unifyweaver/core/recursive_compiler), test_recursive_compiler, halt.' && \
            echo '' && \
            echo '=== Advanced Recursion Tests ===' && \
            swipl -g 'use_module(src/unifyweaver/core/advanced/test_advanced), test_all_advanced, halt.'
        "
        ;;

    shell)
        echo -e "${BLUE}[INFO] Opening shell in container...${NC}"
        docker exec -it unifyweaver-pwsh /bin/bash
        ;;

    config)
        configure_persistence "$CONFIG_FILE"
        ;;

    *)
        echo "Usage: $0 {build|run|start|stop|restart|clean|logs|test|shell|config} [--persistent|--standard]"
        echo ""
        echo "Commands:"
        echo "  build    - Build the Docker image"
        echo "  run      - Start container and open shell (default)"
        echo "  start    - Alias for 'run'"
        echo "  stop     - Stop the container"
        echo "  restart  - Restart the container"
        echo "  clean    - Remove container and volumes"
        echo "  logs     - View container logs"
        echo "  test     - Run all UnifyWeaver tests"
        echo "  shell    - Open a shell in running container"
        echo "  config   - Configure persistence mode"
        echo ""
        echo "Flags:"
        echo "  --persistent  - Use persistent system mode (one-time)"
        echo "  --standard    - Use standard mode (one-time)"
        echo ""
        echo "Config file: $CONFIG_FILE"
        exit 1
        ;;
esac
