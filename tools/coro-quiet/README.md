# coro-quiet

Clean output wrapper for coro-code CLI.

**Generated** from `quiet_module.pl` - do not edit directly.

## Features

- Collapses multiple consecutive blank lines into one
- Captures token usage and cost, shows summary at end
- Preserves ANSI colors and markdown formatting
- Passes all arguments through to `coro`

## Generate

```bash
cd tools/coro-quiet
swipl -g "consult('quiet_module.pl'), quiet_module:generate_all" -t halt
```

## Install

```bash
chmod +x coro-quiet.py

# Option 1: Symlink to PATH
ln -s $(pwd)/coro-quiet.py ~/.local/bin/coro-quiet

# Option 2: Alias
echo "alias coro-quiet='$(pwd)/coro-quiet.py'" >> ~/.bashrc
```

## Usage

```bash
# Single-task mode (recommended)
coro-quiet "explain this code"
coro-quiet --verbose "fix the bug"  # captures token stats

# Interactive mode (limited filtering - TUI uses cursor control)
coro-quiet

# Help
coro-quiet --help
```

## Example Output

Before (coro):
```
[blank line]
[blank line]
Input: 1,234 tokens
Output: 567 tokens
[blank line]
Here is the response...
[blank line]
[blank line]
[blank line]
Input: 890 tokens
Output: 123 tokens
```

After (coro-quiet):
```

Here is the response...

--- Session Summary ---
Input: 2,124 tokens
Output: 690 tokens
```

## Configuration

Edit `quiet_module.pl` to customize:

- `token_pattern/1` - Patterns to capture and summarize
- `config(max_consecutive_blanks, N)` - Max blank lines to keep
- `config(show_token_summary, Bool)` - Show/hide summary

Then regenerate.
