#!/bin/bash
# Generated TUI script: http_cli
# This script renders a terminal UI from declarative specifications

# Enable Unicode support
export LANG=en_US.UTF-8

# ANSI color definitions
RESET="[0m"
BOLD="\033[1m"
DIM="\033[2m"

# Theme colors
PRIMARY="\033[38;5;204m"
SECONDARY="\033[38;5;75m"
BG="\033[48;5;234m"
SURFACE="\033[48;5;236m"
TEXT="\033[38;5;255m"
TEXT_DIM="\033[38;5;245m"
SUCCESS="\033[38;5;82m"
WARNING="\033[38;5;214m"
ERROR="\033[38;5;196m"
INFO="\033[38;5;39m"

# Clear screen and hide cursor
clear
tput civis

# Trap to restore cursor on exit
trap "tput cnorm; echo -e \"$RESET\"" EXIT

# Render UI
echo -e "[48;5;236m┌──────────────────────────────────────────────────────────┐[0m"
  echo -e "📁   [ 📌 Set as Working Dir ]  "
  echo ""
  echo ""
  echo ""
  echo ""
  echo ""
  echo ""
  echo ""
  echo ""
  echo ""
  echo ""
  echo ""
  echo ""
  echo ""
  echo ""
  echo ""
  echo "# Unknown node: when(var(browse.entries),[component(text,[content(var(browse.entries.length), items),class(count)])])"
  echo ""
  echo ""
  echo ""
  echo ""
  echo ""
  echo ""
  echo ""
  echo ""
  echo ""
  echo ""
  echo ""
  echo ""
  echo ""
  echo ""
  echo ""
  echo "# Unknown node: foreach(var(browse.entries),entry,[container(panel,[class(file_entry),on_click(handle_entry_click(var(entry))),border_left(var(entry.type),directory,primary,info)],[layout(flex,[justify(between),align(center)],[layout(flex,[gap(8)],[component(icon,[name(var(entry.type),directory,folder,file)]),component(text,[content(var(entry.name))])]),component(text,[content(format_size(var(entry.size))),style(muted),size(12)])])])])"
  echo ""
  echo ""
  echo ""
  echo ""
  echo ""
  echo ""
  echo ""
  echo ""
  echo ""
  echo ""
  echo ""
  echo ""
  echo ""
  echo ""
  echo ""
  echo "# Unknown node: when(and(empty(var(browse.entries)),not(var(loading))),[component(text,[content(Empty directory),style(muted),align(center)])])"
  echo ""
  echo ""
  echo ""
  echo ""
  echo ""
  echo ""
  echo ""
  echo ""
  echo ""
  echo ""
  echo ""
  echo ""
  echo ""
  echo ""
  echo ""
  echo "# Unknown node: when(var(browse.selected),[container(panel,[class(selected_file)],[layout(stack,[spacing(10)],[component(text,[content(Selected file:),style(muted),size(12)]),component(code,[content(var(browse.selected))]),layout(flex,[gap(10),wrap(true)],[component(button,[label(View Contents),on_click(view_file)]),component(button,[label(📥 Download),on_click(download_file),variant(primary)]),component(button,[label(Search Here),on_click(search_here),variant(secondary)])])])])])"
echo -e "[48;5;236m└──────────────────────────────────────────────────────────┘[0m"


# Show cursor
tput cnorm
