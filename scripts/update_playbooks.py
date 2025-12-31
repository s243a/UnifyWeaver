#!/usr/bin/env python3
import os
import glob
import re

def update_playbook(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    if "## Finding Examples" in content:
        print(f"Skipping {filepath} (already updated)")
        return

    # Derive IDs
    filename = os.path.basename(filepath)
    base_name = filename.replace('_playbook.md', '')
    grep_id = base_name
    query = base_name.replace('_', ' ')

    # Construct section
    lines = [
        "",
        "## Finding Examples",
        "",
        "There are two ways to find the correct example record for this task:",
        "",
        "### Method 1: Manual Extraction",
        "Search the documentation using grep:",
        "```bash",
        f"grep -r \"{grep_id}\" playbooks/examples_library/",
        "```",
        "",
        "### Method 2: Semantic Search (Recommended)",
        "Use the LDA-based semantic search skill to find relevant examples by intent:",
        "```bash",
        f"./unifyweaver search \"how to use {query}\"",
        "",
        ""
    ]
    section = "\n".join(lines)

    # Insert before ## Workflow Overview
    if "## Workflow Overview" in content:
        new_content = content.replace("## Workflow Overview", f"{section}\n## Workflow Overview")
    elif "## Agent Inputs" in content:
        new_content = content.replace("## Agent Inputs", f"{section}\n## Agent Inputs")
    else:
        print(f"Warning: Could not find insertion point for {filepath}")
        return

    with open(filepath, 'w') as f:
        f.write(new_content)
    print(f"Updated {filepath}")

def main():
    playbooks = glob.glob('playbooks/*_playbook.md')
    for p in playbooks:
        update_playbook(p)

if __name__ == "__main__":
    main()
