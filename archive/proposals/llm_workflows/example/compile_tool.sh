#!/bin/bash

# This script compiles a "tool package" from a natural language program (program.md)
# and a directory of tool scripts.

# --- Configuration ---
# In a real scenario, these would be command-line arguments.
PROGRAM_MD_PATH="proposals/llm_workflows/example/program.md"
TOOL_DIR="proposals/llm_workflows/example/tools"
OUTPUT_SCRIPT="compiled_tool.sh"

# --- Read Inputs ---
echo "Reading natural language program from $PROGRAM_MD_PATH..."
PROGRAM_MD_CONTENT=$(cat "$PROGRAM_MD_PATH")

# --- Generate Tool Functions ---
echo "Compiling tools from $TOOL_DIR..."
TOOL_FUNCTIONS=""
for tool_script in $(find "$TOOL_DIR" -type f -name "*.sh"); do
    tool_name=$(basename "$tool_script" .sh)
    # Read the tool script, skipping the shebang if it exists.
    tool_content=$(sed '1s/^#\!.*//' "$tool_script")
    
    TOOL_FUNCTIONS+=$(cat <<EOF

# Tool: $tool_name
# Source: $tool_script
function $tool_name() {
$tool_content
}
EOF
)
done

# --- Assemble Final Script ---
echo "Assembling final tool package..."
cat << EOF > "$OUTPUT_SCRIPT"
#!/bin/bash

# =================================================================
#
#  This script was compiled by compile_tool.sh
#
#  It contains a natural language program and the tools it needs.
#  An LLM can read this script to understand how to perform a task.
#
# =================================================================

# -----------------------------------------------------------------
# Natural Language Program
# -----------------------------------------------------------------
: <<'END_OF_PROGRAM'

$PROGRAM_MD_CONTENT

END_OF_PROGRAM
# -----------------------------------------------------------------

# -----------------------------------------------------------------
# Compiled Tools
# -----------------------------------------------------------------
$TOOL_FUNCTIONS
# -----------------------------------------------------------------

# --- Main Execution ---
# The LLM can now read this script and call the tool functions
# as needed, based on the instructions in the natural language
# program.

# To demonstrate, we will just list the available functions.
echo "Compiled tool script ready. Functions available:"
compgen -A function

EOF

chmod +x "$OUTPUT_SCRIPT"

echo "✅ Compiled tool package to $OUTPUT_SCRIPT"
