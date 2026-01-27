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
  if [ -n "$browse_entries" ]; then
    echo -e "[38;5;255m[0m"
  fi
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
  for entry in "$browse_entries"; do
    echo -e "[48;5;236m┌──────────────────────────────────────────────────────────┐[0m"
      echo -e "format_size(var(entry.size))  "
    echo -e "[48;5;236m└──────────────────────────────────────────────────────────┘[0m"
  done
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
  if [ -z "$browse_entries" ] && ! [ -n "$loading" ]; then
    echo -e "[38;5;255mEmpty directory[0m"
  fi
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
  if [ -n "$browse_selected" ]; then
    echo -e "[48;5;236m┌──────────────────────────────────────────────────────────┐[0m"
      echo -e "[38;5;255mSelected file:[0m"
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
      echo -e "[48;5;236m[38;5;255m var(browse.selected) [0m"
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
      echo -e "[ View Contents ]  [ 📥 Download ]  [ Search Here ]  "
    echo -e "[48;5;236m└──────────────────────────────────────────────────────────┘[0m"
  fi
echo -e "[48;5;236m└──────────────────────────────────────────────────────────┘[0m"


# Show cursor
tput cnorm
