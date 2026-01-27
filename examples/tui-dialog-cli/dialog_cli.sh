#!/bin/bash
# Generated Dialog TUI script: dialog_cli
# Interactive terminal UI using dialog

# Temporary file for dialog output
TEMP=$(mktemp)
trap "rm -f $TEMP" EXIT

# State variables
declare -A STATE
CURRENT_DIR="."
SELECTED_FILE=""
WORKING_DIR="."

# File browser function using menu (more reliable than fselect)
browse_files() {
    local dir="${1:-$CURRENT_DIR}"
    dir="${dir:-.}"

    # Convert to absolute path for reliable navigation
    dir=$(cd "$dir" 2>/dev/null && pwd) || dir="."

    # Build menu items and track paths separately
    # Use 0-9, a-z for quick select keys
    local keys="0123456789abcdefghijklmnopqrstuvwxyz"
    local items=()
    local paths=()
    local types=()
    local idx=0

    get_key() { echo "${keys:$1:1}"; }

    # Add parent directory option if not at root
    if [ "$dir" != "/" ]; then
        items+=("$(get_key $idx)" "[..] Parent Directory")
        paths+=("$(dirname "$dir")")
        types+=("parent")
        ((idx++))
    fi

    # Add directories first
    for entry in "$dir"/*/; do
        if [ -d "$entry" ]; then
            local name=$(basename "$entry")
            items+=("$(get_key $idx)" "[D] $name")
            paths+=("$entry")
            types+=("dir")
            ((idx++))
        fi
    done

    # Add files
    for entry in "$dir"/*; do
        if [ -f "$entry" ]; then
            local name=$(basename "$entry")
            local size=$(du -h "$entry" 2>/dev/null | cut -f1)
            items+=("$(get_key $idx)" "[F] $name ($size)")
            paths+=("$entry")
            types+=("file")
            ((idx++))
        fi
    done

    if [ ${#items[@]} -eq 0 ]; then
        dialog --msgbox "Empty directory: $dir" 6 40
        main_menu
        return
    fi

    dialog --clear --title "Browse: $dir" \
        --menu "Select file or directory:" 20 70 12 \
        "${items[@]}" \
        2>"$TEMP"

    local status=$?
    local choice=$(<"$TEMP")

    if [ $status -ne 0 ]; then
        main_menu
        return
    fi

    # Convert choice (0-9, a-z) back to index
    key_to_idx() {
        local key="$1"
        local pos="${keys%%$key*}"
        echo "${#pos}"
    }

    local idx=$(key_to_idx "$choice")
    local selected_path="${paths[$idx]}"
    local selected_type="${types[$idx]}"

    case "$selected_type" in
        parent|dir)
            CURRENT_DIR="$selected_path"
            browse_files
            ;;
        file)
            SELECTED_FILE="$selected_path"
            CURRENT_DIR="$dir"
            file_actions
            ;;
        *)
            main_menu
            ;;
    esac
}

# File action submenu
file_actions() {
    dialog --title "File: $(basename "$SELECTED_FILE")" \
        --menu "Select action:" 15 60 5 \
        1 "View Contents" \
        2 "Download/Copy" \
        3 "Search in Directory" \
        4 "Continue Browsing" \
        5 "Back to Main Menu" \
        2>"$TEMP"

    choice=$(<"$TEMP")
    case $choice in
        1) view_selected_file ;;
        2) download_selected_file ;;
        3) search_in_dir ;;
        4) browse_files ;;
        *) main_menu ;;
    esac
}

view_selected_file() {
    if [ -f "$SELECTED_FILE" ]; then
        dialog --title "$SELECTED_FILE" --textbox "$SELECTED_FILE" 22 76
    fi
    file_actions
}

download_selected_file() {
    if [ -f "$SELECTED_FILE" ]; then
        dialog --yesno "Copy to current directory?\n\n$SELECTED_FILE" 10 50
        if [ $? -eq 0 ]; then
            cp "$SELECTED_FILE" . 2>/dev/null && \
                dialog --msgbox "Copied: $(basename "$SELECTED_FILE")" 6 40 || \
                dialog --msgbox "Copy failed" 6 30
        fi
    fi
    file_actions
}

search_in_dir() {
    dialog --inputbox "Search pattern:" 8 50 2>"$TEMP"
    pattern=$(<"$TEMP")
    if [ -n "$pattern" ]; then
        results=$(grep -r "$pattern" "$CURRENT_DIR" 2>/dev/null | head -30)
        if [ -n "$results" ]; then
            echo "$results" > /tmp/search_results.txt
            dialog --title "Results for: $pattern" --textbox /tmp/search_results.txt 20 76
            rm -f /tmp/search_results.txt
        else
            dialog --msgbox "No matches found" 6 40
        fi
    fi
    file_actions
}

# Main menu function
main_menu() {
    dialog --clear --title "File Browser" \
        --menu "Select an action:" 18 60 7 \
        0 "📁 Browse Files" \
        1 "⬆️ Up" \
        2 "📌 Set as Working Dir" \
        3 "View Contents" \
        4 "📥 Download" \
        5 "Search Here" \
        6 "Exit" \
        2>"$TEMP"

    choice=$(<"$TEMP")
    case $choice in
        0) browse_files ;;
        1) action_1 ;;
        2) action_2 ;;
        3) action_3 ;;
        4) action_4 ;;
        5) action_5 ;;
        7) exit 0 ;;
        *) exit 0 ;;
    esac
}

action_1() {
    # Navigate up one directory (convert to absolute path first)
    local absdir=$(cd "${CURRENT_DIR:-.}" 2>/dev/null && pwd) || absdir="/"
    if [ "$absdir" != "/" ]; then
        CURRENT_DIR=$(dirname "$absdir")
    else
        CURRENT_DIR="/"
    fi
    browse_files
}

action_2() {
    # Set working directory
    if [ -n "$CURRENT_DIR" ]; then
        WORKING_DIR="$CURRENT_DIR"
        dialog --msgbox "Working directory set to:\n$WORKING_DIR" 8 50
    fi
    main_menu
}

action_3() {
    # View file contents
    if [ -n "$SELECTED_FILE" ] && [ -f "$SELECTED_FILE" ]; then
        dialog --title "$SELECTED_FILE" --textbox "$SELECTED_FILE" 22 76
    else
        dialog --msgbox "No file selected" 6 40
    fi
    main_menu
}

action_4() {
    # Download/copy file
    if [ -n "$SELECTED_FILE" ] && [ -f "$SELECTED_FILE" ]; then
        dialog --yesno "Copy $SELECTED_FILE to current directory?" 8 50
        if [ $? -eq 0 ]; then
            cp "$SELECTED_FILE" . 2>/dev/null && dialog --msgbox "File copied to $(pwd)/$(basename "$SELECTED_FILE")" 8 50 || dialog --msgbox "Copy failed" 6 40
        fi
    else
        dialog --msgbox "No file selected" 6 40
    fi
    main_menu
}

action_5() {
    # Search in current directory
    dialog --inputbox "Enter search pattern:" 8 50 2>"$TEMP"
    pattern=$(<"$TEMP")
    if [ -n "$pattern" ]; then
        results=$(grep -r "$pattern" "${CURRENT_DIR:-.}" 2>/dev/null | head -30)
        if [ -n "$results" ]; then
            echo "$results" > /tmp/search_results.txt
            dialog --title "Search Results" --textbox /tmp/search_results.txt 20 76
            rm -f /tmp/search_results.txt
        else
            dialog --msgbox "No matches found for: $pattern" 6 50
        fi
    fi
    main_menu
}




# Start main
main_menu
