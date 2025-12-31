#!/bin/bash
# UnifyWeaver Docker - Mount Additional Projects
# Adds extra project mounts while preserving persistent mode data

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OVERRIDE_FILE="$SCRIPT_DIR/compose.mounts.override.yml"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}╔════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  UnifyWeaver - Mount Additional Projects          ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════╝${NC}"
echo ""

# Change to docker directory for docker-compose
cd "$SCRIPT_DIR"

# Check if docker-compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo -e "${YELLOW}[WARNING] docker-compose not found, trying 'docker compose'${NC}"
    DOCKER_COMPOSE="docker compose"
else
    DOCKER_COMPOSE="docker-compose"
fi

case "${1:-mount}" in
    mount)
        echo -e "${BLUE}[INFO] Creating mount override configuration...${NC}"
        
        # Create the override file with all mounts
        cat > "$OVERRIDE_FILE" << 'EOF'
# Mount override for UnifyWeaver development
# Adds additional project directories while preserving persistent mode
# IMPORTANT: Must re-declare ALL volumes because Docker Compose REPLACES lists, doesn't merge

version: '3.3'

services:
  unifyweaver-dev:
    volumes:
      # Original UnifyWeaver project
      - /mnt/c/Users/johnc/Dropbox/projects/UnifyWeaver:/mnt/c/Users/johnc/Dropbox/projects/UnifyWeaver
      
      # Additional projects
      - /mnt/c/Users/johnc/Dropbox/projects/agentRAG:/mnt/c/Users/johnc/Dropbox/projects/agentRAG
      - /mnt/c/Users/johnc/Dropbox/projects/JanusBridge:/mnt/c/Users/johnc/Dropbox/projects/JanusBridge
      - /mnt/c/Users/johnc/Dropbox/projects/InfoEcon:/mnt/c/Users/johnc/Dropbox/projects/InfoEcon
      
      # Persistent home directory (MUST re-declare - lists are replaced, not merged)
      - unifyweaver-home:/root
      
      # Persistent system volumes (MUST re-declare from docker-compose.persistent.yml)
      - unifyweaver-dev-usr-local:/usr/local
      - unifyweaver-dev-opt:/opt
      
    # Keep original working directory
    working_dir: /mnt/c/Users/johnc/Dropbox/projects/UnifyWeaver
EOF

        echo -e "${GREEN}✓ Override file created: ${OVERRIDE_FILE}${NC}"
        echo ""
        
        echo -e "${BLUE}[INFO] Mounted projects:${NC}"
        echo "  • /mnt/c/Users/johnc/Dropbox/projects/UnifyWeaver (original)"
        echo "  • /mnt/c/Users/johnc/Dropbox/projects/agentRAG"
        echo "  • /mnt/c/Users/johnc/Dropbox/projects/JanusBridge"
        echo "  • /mnt/c/Users/johnc/Dropbox/projects/InfoEcon"
        echo ""
        echo -e "${BLUE}[INFO] Persistent volumes included:${NC}"
        echo "  • ~/root (bash history, configs)"
        echo "  • /usr/local (installed packages)"
        echo "  • /opt (additional packages)"
        echo ""
        
        echo -e "${BLUE}[INFO] Stopping container to apply new mounts...${NC}"
        $DOCKER_COMPOSE stop unifyweaver-dev 2>/dev/null || true
        echo ""
        
        echo -e "${BLUE}[INFO] Starting with persistent mode + additional mounts...${NC}"
        echo -e "${BLUE}[INFO] Using:${NC}"
        echo "  - docker-compose.yml (base config)"
        echo "  - docker-compose.persistent.yml (persistent /usr/local and /opt)"
        echo "  - compose.mounts.override.yml (additional projects)"
        echo ""
        
        # Start with all three compose files
        $DOCKER_COMPOSE \
            -f docker-compose.yml \
            -f docker-compose.persistent.yml \
            -f compose.mounts.override.yml \
            up -d unifyweaver-dev
        
        echo ""
        echo -e "${GREEN}✓ Container restarted with additional mounts${NC}"
        echo ""
        echo -e "${BLUE}[INFO] Entering container shell...${NC}"
        echo ""
        
        # Attach to container
        docker exec -it unifyweaver-dev bash
        ;;
        
    verify)
        echo -e "${BLUE}[INFO] Verifying mounts in container...${NC}"
        echo ""
        
        if ! docker ps | grep -q unifyweaver-dev; then
            echo -e "${YELLOW}[ERROR] Container not running. Start with: $0 mount${NC}"
            exit 1
        fi
        
        docker exec unifyweaver-dev bash -c '
            echo "=== Current Working Directory ==="
            pwd
            echo ""
            
            echo "=== Mounted Projects ==="
            for proj in UnifyWeaver agentRAG JanusBridge InfoEcon; do
                path="/mnt/c/Users/johnc/Dropbox/projects/$proj"
                if [ -d "$path" ]; then
                    echo "✓ $proj: $path"
                    echo "  Files: $(ls -1 "$path" 2>/dev/null | wc -l)"
                else
                    echo "✗ $proj: NOT FOUND"
                fi
            done
            echo ""
            
            echo "=== Persistent Volumes ==="
            echo "✓ Home (~): $(ls -la ~ | wc -l) entries"
            echo "✓ /usr/local: $(ls -la /usr/local 2>/dev/null | wc -l) entries"
            echo "✓ /opt: $(ls -la /opt 2>/dev/null | wc -l) entries"
        '
        ;;
        
    revert)
        echo -e "${YELLOW}[INFO] Reverting to original mounts (keeping persistent mode)...${NC}"
        
        if [ -f "$OVERRIDE_FILE" ]; then
            echo -e "${BLUE}[INFO] Removing override file: ${OVERRIDE_FILE}${NC}"
            rm "$OVERRIDE_FILE"
        fi
        
        echo -e "${BLUE}[INFO] Stopping container...${NC}"
        $DOCKER_COMPOSE stop unifyweaver-dev 2>/dev/null || true
        
        echo -e "${BLUE}[INFO] Starting with persistent mode (original mounts only)...${NC}"
        $DOCKER_COMPOSE \
            -f docker-compose.yml \
            -f docker-compose.persistent.yml \
            up -d unifyweaver-dev
        
        echo ""
        echo -e "${GREEN}✓ Reverted to original configuration${NC}"
        echo -e "${BLUE}[INFO] Only UnifyWeaver project is now mounted${NC}"
        echo ""
        echo -e "${BLUE}[INFO] Entering container shell...${NC}"
        docker exec -it unifyweaver-dev bash
        ;;
        
    shell)
        echo -e "${BLUE}[INFO] Opening shell in container...${NC}"
        
        if ! docker ps | grep -q unifyweaver-dev; then
            echo -e "${YELLOW}[ERROR] Container not running. Start with: $0 mount${NC}"
            exit 1
        fi
        
        docker exec -it unifyweaver-dev bash
        ;;
        
    status)
        echo -e "${BLUE}[INFO] Container status:${NC}"
        echo ""
        
        if docker ps | grep -q unifyweaver-dev; then
            echo -e "${GREEN}✓ Container is running${NC}"
            echo ""
            
            # Check which compose files are likely in use
            if [ -f "$OVERRIDE_FILE" ]; then
                echo -e "${BLUE}Mount override file exists: ${OVERRIDE_FILE}${NC}"
                echo "Additional projects should be mounted"
            else
                echo -e "${BLUE}No mount override file${NC}"
                echo "Only original UnifyWeaver project is mounted"
            fi
        else
            echo -e "${YELLOW}✗ Container is not running${NC}"
        fi
        echo ""
        
        if [ -f "$OVERRIDE_FILE" ]; then
            echo -e "${BLUE}[INFO] Override file contents:${NC}"
            echo ""
            cat "$OVERRIDE_FILE"
        fi
        ;;
        
    *)
        echo "Usage: $0 {mount|verify|revert|shell|status}"
        echo ""
        echo "Commands:"
        echo "  mount   - Add extra project mounts and restart in persistent mode (default)"
        echo "  verify  - Check which projects are mounted and accessible"
        echo "  revert  - Remove extra mounts, keep only UnifyWeaver (persistent mode)"
        echo "  shell   - Open shell in running container"
        echo "  status  - Show container status and current mount configuration"
        echo ""
        echo "Mounted projects when using 'mount':"
        echo "  • UnifyWeaver (original)"
        echo "  • agentRAG"
        echo "  • JanusBridge"
        echo "  • InfoEcon"
        echo ""
        echo "All mounted at: /mnt/c/Users/johnc/Dropbox/projects/<project>"
        echo ""
        echo "Persistent volumes (preserved across restarts):"
        echo "  • ~/root - bash history, .bashrc, configs"
        echo "  • /usr/local - packages installed via apt-get, pip, npm, etc."
        echo "  • /opt - additional software installations"
        echo ""
        exit 1
        ;;
esac