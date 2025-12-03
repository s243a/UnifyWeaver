#!/bin/bash
# Shared functions for Docker launcher scripts

# Show persistence mode selection menu
# Args: $1 = config file path, $2 = current persistent mode
# Returns: Sets PERSISTENT_MODE variable
show_persistence_menu() {
    local config_file="$1"
    local current_mode="$2"

    echo -e "${BLUE}╔════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║  Select Persistence Mode                          ║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "1) Standard Mode (default)"
    echo "   - Fast, consistent environment"
    echo "   - Installed packages reset on image rebuild"
    echo "   - Home directory (~/.bashrc, history) persists"
    echo ""
    echo "2) Persistent System Mode"
    echo "   - Additional volumes for /usr/local, /opt"
    echo "   - Installed packages (apt-get, pip, npm) persist"
    echo "   - Slightly slower startup"
    echo "   - Useful for testing different SWI-Prolog versions"
    echo ""
    echo "3) Save choice to config file"
    echo ""
    read -p "Select mode [1-3] (default: 1): " choice

    case $choice in
        2)
            PERSISTENT_MODE="persistent"
            ;;
        3)
            read -p "Save as (1=standard, 2=persistent): " save_choice
            if [[ "$save_choice" == "2" ]]; then
                echo "PERSISTENT_MODE=persistent" > "$config_file"
                PERSISTENT_MODE="persistent"
                echo -e "${GREEN}[INFO] Saved persistent mode to config${NC}"
            else
                echo "PERSISTENT_MODE=standard" > "$config_file"
                PERSISTENT_MODE="standard"
                echo -e "${GREEN}[INFO] Saved standard mode to config${NC}"
            fi
            ;;
        *)
            PERSISTENT_MODE="standard"
            ;;
    esac
    echo ""
}

# Configure persistence mode
# Args: $1 = config file path
configure_persistence() {
    local config_file="$1"

    echo -e "${BLUE}[INFO] Persistence Mode Configuration${NC}"
    echo ""
    if [[ -f "$config_file" ]]; then
        source "$config_file"
        echo -e "Current mode: ${GREEN}${PERSISTENT_MODE:-standard}${NC}"
        echo ""
        read -p "Change to (1=standard, 2=persistent, 3=delete config): " choice
        case $choice in
            1)
                echo "PERSISTENT_MODE=standard" > "$config_file"
                echo -e "${GREEN}[INFO] Saved standard mode${NC}"
                ;;
            2)
                echo "PERSISTENT_MODE=persistent" > "$config_file"
                echo -e "${GREEN}[INFO] Saved persistent mode${NC}"
                ;;
            3)
                rm "$config_file"
                echo -e "${GREEN}[INFO] Config file deleted (will show menu on next run)${NC}"
                ;;
        esac
    else
        echo "No config file found"
        read -p "Create config? (1=standard, 2=persistent): " choice
        if [[ "$choice" == "2" ]]; then
            echo "PERSISTENT_MODE=persistent" > "$config_file"
            echo -e "${GREEN}[INFO] Saved persistent mode${NC}"
        else
            echo "PERSISTENT_MODE=standard" > "$config_file"
            echo -e "${GREEN}[INFO] Saved standard mode${NC}"
        fi
    fi
}
