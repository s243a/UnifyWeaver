<!--
SPDX-License-Identifier: MIT OR Apache-2.0
Copyright (c) 2025 John William Creighton (@s243a)
-->
# Line Endings Reference

## TL;DR

**UnifyWeaver uses Unix line endings (LF) everywhere.**

- `.gitattributes` enforces this automatically
- Windows users: Git handles conversion for you
- Bash scripts REQUIRE LF (won't work with CRLF)
- Multi-line strings in Prolog/Bash can break with CRLF

---

## Why Unix Line Endings?

### Technical Reasons

1. **Bash Scripts Break with CRLF**
   ```bash
   #!/usr/bin/env bash\r     # ← This \r breaks the shebang
   # Error: /usr/bin/env: 'bash\r': No such file or directory
   ```

2. **Multi-line Strings Have Issues**
   ```prolog
   % Prolog multi-line string with CRLF
   Template = "line1\r\n
   line2\r\n"
   % Results in: "line1\r\nline2\r\n" (literal \r\n in string)
   ```

3. **Cross-Platform Consistency**
   - Works on Windows, macOS, Linux
   - Git for Windows handles conversion transparently
   - Modern editors support LF on Windows

4. **Industry Standard**
   - Most open-source projects use LF
   - WSL requires LF
   - Docker containers use LF

### Exceptions (Future)

**Windows-specific scripts (not in project yet):**
- `.bat` files → CRLF (Windows batch scripts)
- `.ps1` files → CRLF (PowerShell scripts)
- `.cmd` files → CRLF (Windows command scripts)

**These would be marked in `.gitattributes` when added:**
```gitattributes
*.bat text eol=crlf
*.ps1 text eol=crlf
```

---

## How Git Handles This

### .gitattributes Configuration

**Our configuration (in `.gitattributes`):**
```gitattributes
# Default: LF for all text files
* text=auto eol=lf

# Explicitly mark important files
*.sh text eol=lf
*.pl text eol=lf
*.py text eol=lf
*.md text eol=lf
```

**What this means:**
- `text=auto` → Git detects text files automatically
- `eol=lf` → Convert to LF on checkout AND commit
- Files are stored as LF in the repository
- Windows users get LF locally (works fine!)

### Git Configuration (No Action Needed)

**Git for Windows handles this automatically.** You don't need to configure `core.autocrlf`.

**For reference, here's what Git does:**
```bash
# Check your setting (usually not needed)
git config --global core.autocrlf

# With .gitattributes, this is ignored anyway
# .gitattributes takes precedence
```

---

## Working on Windows

### Modern Git for Windows

**Git for Windows (2.10+) works perfectly with LF:**
- Bash scripts execute correctly with LF
- Editors like VS Code handle LF natively
- No manual conversion needed

### If You See Issues

**Symptom: Bash script fails with "command not found"**
```bash
./script.sh
# Error: /usr/bin/env: 'bash\r': No such file or directory
```

**Diagnosis:**
```bash
file script.sh
# Bad: ASCII text, with CRLF line terminators
# Good: ASCII text
```

**Fix:**
```bash
# Convert manually
dos2unix script.sh

# Or let Git fix it
git add script.sh
git commit -m "Fix line endings"
# Git will normalize to LF thanks to .gitattributes
```

### Configuring Your Editor

**VS Code (Recommended for Windows):**
```json
{
  "files.eol": "\n",
  "files.insertFinalNewline": true
}
```

**Check current file:**
- Bottom right corner shows "LF" or "CRLF"
- Click to change if needed

**Notepad++:**
- Edit → EOL Conversion → Unix (LF)
- Settings → Preferences → New Document → Unix (LF)

**Sublime Text:**
```json
{
  "default_line_ending": "unix"
}
```

**IntelliJ/PyCharm:**
- File → File Properties → Line Separators → LF

---

## Verifying Line Endings

### Check a Single File

**Using `file` command (Git Bash on Windows):**
```bash
file script.sh
# Good: ASCII text
# Good: UTF-8 Unicode text
# Bad:  ASCII text, with CRLF line terminators
```

**Using `git` command:**
```bash
git ls-files --eol script.sh
# Shows: i/lf w/lf   script.sh
# i/lf = stored as LF in repo
# w/lf = working directory has LF
```

### Check All Files

**Find files with CRLF:**
```bash
# In Git Bash
git ls-files --eol | grep crlf
# Should return nothing

# Or check working directory
find . -type f -name "*.sh" -exec file {} \; | grep CRLF
```

**Fix all files:**
```bash
# Convert all .sh files
find . -name "*.sh" -exec dos2unix {} \;

# Or let Git normalize
git add --renormalize .
git commit -m "Normalize line endings"
```

---

## Common Issues and Solutions

### Issue 1: Bash Script Won't Execute

**Symptoms:**
```bash
./script.sh
# /usr/bin/env: 'bash\r': No such file or directory
```

**Cause:** Script has CRLF line endings

**Solution:**
```bash
dos2unix script.sh
git add script.sh
git commit -m "Fix line endings in script.sh"
```

### Issue 2: Multi-line String Formatting Wrong

**Symptoms:**
```prolog
% String has literal \r characters
Template = "line1
line2"
% Renders as: "line1\r\nline2\r\n" with visible \r
```

**Cause:** File saved with CRLF

**Solution:**
```bash
# Convert file to LF
dos2unix file.pl
# Or configure editor to use LF (see above)
```

### Issue 3: Git Shows Every Line Changed

**Symptoms:**
```bash
git diff file.sh
# Every single line shows as changed
```

**Cause:** Line ending mismatch (file has CRLF, repo expects LF)

**Solution:**
```bash
# Let Git normalize
git add --renormalize file.sh
git commit -m "Normalize line endings"

# Or convert manually
dos2unix file.sh
```

### Issue 4: .gitattributes Not Working

**Symptoms:**
- Files still have CRLF after checkout
- Git not normalizing line endings

**Solution:**
```bash
# Refresh Git's understanding
git rm --cached -r .
git reset --hard

# This re-applies .gitattributes to all files
```

---

## FAQ

### Q: Why not use CRLF on Windows?

**A:** 
1. Bash scripts require LF
2. Git for Windows works fine with LF
3. Modern Windows editors support LF
4. Consistency across all platforms
5. Industry standard for open source

### Q: What if I'm editing on Windows?

**A:** 
- Git for Windows handles LF automatically
- VS Code and modern editors support LF natively
- `.gitattributes` ensures files are always LF in repo
- Just configure your editor once (see above)

### Q: Will this break anything on Windows?

**A:**
- No! Windows text files can use LF
- Notepad (old versions) shows it oddly, but Notepad++ and VS Code are fine
- Programs read LF files correctly
- Git Bash works perfectly with LF

### Q: What about .bat or .ps1 files?

**A:**
- Not in the project yet
- When added, they'll be marked as CRLF in `.gitattributes`
- Unix-style scripts (.sh) always use LF

### Q: Can I use CRLF in my local working directory?

**A:**
- `.gitattributes` forces LF on checkout
- You can configure your editor to show/use LF
- Repository always stores LF
- Don't fight it - just use LF everywhere

---

## Summary

**What You Need to Know:**

1. ✅ `.gitattributes` enforces LF automatically
2. ✅ Configure your editor to use LF (one-time setup)
3. ✅ Bash scripts require LF (will break with CRLF)
4. ✅ Multi-line strings work correctly with LF
5. ✅ Works perfectly on Windows with modern tools

**What You Don't Need to Do:**

- ❌ Configure `core.autocrlf` (`.gitattributes` overrides it)
- ❌ Manually convert files (Git does it automatically)
- ❌ Worry about platform differences (enforced consistently)

**If You Have Issues:**

1. Check file: `file script.sh`
2. Convert if needed: `dos2unix script.sh`
3. Commit: `git add script.sh && git commit`
4. Configure editor to use LF by default

---

## References

- [Git Book: gitattributes](https://git-scm.com/docs/gitattributes)
- [GitHub: Dealing with line endings](https://docs.github.com/en/get-started/getting-started-with-git/configuring-git-to-handle-line-endings)
- [Prolog multi-line strings](https://www.swi-prolog.org/pldoc/man?section=syntax-strings)
- [Bash scripting line endings](https://stackoverflow.com/questions/39527571/are-shell-scripts-sensitive-to-encoding-and-line-endings)