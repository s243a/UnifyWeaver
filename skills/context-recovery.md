# Context Recovery Skill

Recover information, app specs, and commands from previous Claude Code conversations.

## Usage

```
/context-recovery [search_term] [--date YYYY-MM-DD] [--output context/folder_name]
```

## Instructions

When the user invokes this skill, help them recover context from previous Claude Code sessions stored in `~/.claude/`.

### Step 1: Identify What to Search For

Ask the user what they're looking for:
- An app specification (Prolog `app(name, [...])`)
- Commands that were run
- A specific feature or file that was created
- Conversations around a certain date

### Step 2: Search Conversation History

Use these search patterns based on what's needed:

**Find session files by date:**
```bash
ls -lt ~/.claude/projects/-data-data-com-termux-files-home-UnifyWeaver/*.jsonl | head -20
```

**Search for keywords in history:**
```bash
rg "keyword" ~/.claude/history.jsonl | jq -r '"\(.timestamp | . / 1000 | strftime("%Y-%m-%d %H:%M")): \(.display[0:200])"'
```

**Find app specs:**
```bash
rg -o "app\([a-z_]+,\s*\[" ~/.claude/ 2>/dev/null | sort -u
```

**Search session JSONL files for content:**
```bash
rg "search_term" ~/.claude/projects/-data-data-com-termux-files-home-UnifyWeaver/*.jsonl 2>/dev/null | head -20
```

**Extract user messages around a date:**
```bash
rg '"type":"user"' ~/.claude/projects/-data-data-com-termux-files-home-UnifyWeaver/SESSION_ID.jsonl | \
  jq -r 'select(.timestamp >= "2026-01-17T00:00" and .timestamp <= "2026-01-17T23:59") | "\(.timestamp): \(.message.content | tostring | .[0:300])"'
```

**Find files in file-history:**
```bash
ls ~/.claude/file-history/SESSION_ID/
```

### Step 3: Extract Full Context

Once you find the relevant session, extract details:

**Get full app spec:**
```bash
rg -A 50 'app\(app_name, \[' ~/.claude/projects/.../SESSION.jsonl | head -80
```

**Find assistant responses with code:**
```bash
rg '"type":"assistant"' SESSION.jsonl | jq -r '.message.content[] | select(.type=="text") | .text' | head -100
```

**Check for generated files:**
```bash
ls ~/generated_apps/
find ~/generated_apps -name "*.vue" -o -name "*.ts" | head -20
```

### Step 4: Document the Recovery

If `--output` is specified, create documentation:

1. Create the output folder in `context/`
2. Save the app spec as `app_spec.pl`
3. Save any generation commands as `generate.sh`
4. Copy any manually-created files that aren't auto-generated
5. Create a `README.md` explaining:
   - What the app does
   - How to regenerate it
   - What files are generated vs manual
   - Original creation date

### Key Paths

| Path | Contents |
|------|----------|
| `~/.claude/history.jsonl` | Command history with timestamps |
| `~/.claude/projects/.../SESSION.jsonl` | Full conversation transcripts |
| `~/.claude/file-history/SESSION/` | File snapshots from session |
| `~/.claude/plans/` | Plan mode documents |
| `~/.claude/debug/SESSION.txt` | Debug logs |

### JSONL Structure

Each line in session JSONL files has:
- `type`: "user" or "assistant"
- `timestamp`: ISO 8601 format (e.g., "2026-01-17T08:52:00.000Z")
- `message.content`: Message content (string for user, array for assistant)

### Example Recovery

```bash
# Find when "guard_demo" was created
rg "guard_demo" ~/.claude/history.jsonl | jq -r '.timestamp | . / 1000 | strftime("%Y-%m-%d %H:%M")'

# Get the full app spec
rg -A 30 "app\(guard_demo" ~/.claude/projects/.../*.jsonl | head -50

# Check generated files
ls ~/generated_apps/vue_guard_test/
```

## Output

After recovery, the skill should:
1. Present findings to the user
2. Offer to document in `context/` folder
3. Create regeneration scripts if applicable
