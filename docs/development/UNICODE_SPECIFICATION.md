# Unicode Handling Specification

**Version:** 1.0
**Date:** October 20, 2025
**Applies To:** UnifyWeaver v0.0.2+

---

## Overview

This document specifies how UnifyWeaver handles Unicode text across different platforms, with emphasis on emoji support, cross-platform compatibility, and graceful degradation.

## Design Principles

1. **Progressive Enhancement**: Full Unicode support where available, graceful fallback where not
2. **Explicit Over Implicit**: Use escape sequences in source code for clarity
3. **Platform Awareness**: Detect terminal capabilities and adapt output accordingly
4. **User Control**: Allow manual override of auto-detection

## Unicode Support Levels

### Level 1: ASCII (Baseline)

**Character Range:** U+0000 to U+007F
**Coverage:** English alphabet, digits, basic punctuation
**Emoji Rendering:** Text fallbacks (e.g., `[OK]`, `[STEP]`, `[DATA]`)

**Use Cases:**
- Legacy terminals
- Environments with poor Unicode support
- Plain text logs
- User preference for simplicity

**Example Output:**
```
[STEP] Starting process...
[DATA] Loading data...
[OK] Complete!
```

### Level 2: BMP (Basic Multilingual Plane)

**Character Range:** U+0000 to U+FFFF
**Coverage:** Most modern scripts, mathematical symbols, basic emoji
**Emoji Rendering:** BMP symbols only (✅ ❌ ⚠ ℹ ⚡)

**Use Cases:**
- ConEmu terminal
- Older Windows consoles
- Terminals with limited font support

**Supported Symbols:**
- ✅ U+2705 (White Heavy Check Mark)
- ❌ U+274C (Cross Mark)
- ⚠ U+26A0 (Warning Sign)
- ℹ U+2139 (Information Source)
- ⚡ U+26A1 (High Voltage)

**Example Output:**
```
⚡ Starting process...
📊 Loading data...  ← Falls back to [DATA]
✅ Complete!
```

### Level 3: Full Unicode (Preferred)

**Character Range:** U+0000 to U+10FFFF
**Coverage:** All Unicode planes including Supplementary Multilingual Plane
**Emoji Rendering:** Full color emoji (🚀📊📈💾🎉👥)

**Use Cases:**
- Modern terminals (Windows Terminal, iTerm2, GNOME Terminal)
- WSL environments
- macOS Terminal.app
- Most Linux terminal emulators

**Supported Emoji Categories:**

**Status Indicators:**
- ✅ U+2705 OK
- ❌ U+274C FAIL
- ⚠ U+26A0 WARN
- ℹ U+2139 INFO

**Progress Indicators:**
- 🚀 U+1F680 STEP
- 📡 U+1F4E1 LOAD
- 📊 U+1F4CA DATA
- 📈 U+1F4C8 PROC
- 💾 U+1F4BE SAVE
- 🔄 U+1F504 SYNC

**Task Indicators:**
- 👥 U+1F465 USER
- 🎯 U+1F3AF GOAL
- 🎉 U+1F389 DONE
- 🔧 U+1F527 TOOL
- 🔍 U+1F50D FIND
- 📝 U+1F4DD NOTE

**Category Indicators:**
- 🏗 U+1F3D7 BUILD
- 🧪 U+1F9EA TEST
- 📦 U+1F4E6 PKG
- 🌐 U+1F310 WEB
- 🔐 U+1F510 SEC
- ⚡ U+26A1 FAST

**Example Output:**
```
🚀 Starting process...
📊 Loading data...
✅ Complete!
```

## Source Code Conventions

### Unicode Escape Sequences

All Unicode characters in source code MUST be specified using escape sequences, not literal characters.

#### Syntax

**For BMP characters (U+0000 to U+FFFF):**
```prolog
'\uXXXX'
```
Where `XXXX` is the 4-digit hexadecimal code point.

**For non-BMP characters (U+10000 to U+10FFFF):**
```prolog
'\UXXXXXXXX'
```
Where `XXXXXXXX` is the 8-digit hexadecimal code point with leading zeros.

#### Examples

```prolog
% BMP examples
emoji_fallback('\u2705', '[OK]').      % ✅ U+2705
emoji_fallback('\u274C', '[FAIL]').    % ❌ U+274C
emoji_fallback('\u26A0', '[WARN]').    % ⚠ U+26A0

% Non-BMP examples
emoji_fallback('\U0001F680', '[STEP]').    % 🚀 U+1F680
emoji_fallback('\U0001F4CA', '[DATA]').    % 📊 U+1F4CA
emoji_fallback('\U0001F389', '[DONE]').    % 🎉 U+1F389
```

#### Comment Convention

Each Unicode escape MUST be followed by a comment showing:
1. The actual rendered character (for readability)
2. The Unicode code point in U+XXXX notation

```prolog
emoji_fallback('\U0001F680', '[STEP]').    % 🚀 U+1F680
```

### Rationale for Escape Sequences

1. **Cross-Platform Compatibility**: Avoids encoding issues when files are edited on different platforms
2. **Version Control**: Git diffs show actual code changes, not encoding variations
3. **Editor Independence**: Works regardless of editor's Unicode support
4. **Explicit Documentation**: Code point is visible in source
5. **Windows Compatibility**: Avoids file reading issues on Windows

## Terminal Detection

### Detection Strategy

UnifyWeaver detects terminal capabilities by examining environment variables:

```prolog
detect_terminal(Terminal) :-
    (   getenv('WT_SESSION', _)
    ->  Terminal = windows_terminal
    ;   getenv('ConEmuPID', _)
    ->  Terminal = conemu
    ;   getenv('WSL_DISTRO_NAME', _)
    ->  Terminal = wsl
    ;   Terminal = unknown
    ).
```

### Emoji Level Assignment

```prolog
auto_detect_and_set_emoji_level :-
    detect_terminal(Terminal),
    terminal_emoji_level(Terminal, Level),
    set_emoji_level(Level).

% Terminal capability mappings
terminal_emoji_level(windows_terminal, full).
terminal_emoji_level(wsl, full).
terminal_emoji_level(conemu, bmp).
terminal_emoji_level(unknown, ascii).
```

### Manual Override

Users can override auto-detection:

```prolog
% Set specific level
?- set_emoji_level(ascii).
?- set_emoji_level(bmp).
?- set_emoji_level(full).

% Query current level
?- get_emoji_level(Level).
```

## Output Formatting

### The `safe_format` Predicates

All emoji output MUST use `safe_format/2` or `safe_format/3` instead of direct `format/2` or `format/3` calls.

```prolog
%! safe_format(+Format:atom, +Args:list) is det.
%  Like format/2, but adapts emoji based on terminal capability.

safe_format(Format, Args) :-
    safe_format(current_output, Format, Args).

%! safe_format(+Stream, +Format:atom, +Args:list) is det.
%  Like format/3, but adapts emoji based on terminal capability.
```

### Usage Examples

```prolog
% Simple status message
safe_format('\u2705 Test passed~n', []).

% With format arguments
safe_format('\U0001F680 Loading ~w~n', [FileName]).

% Multiple emoji
safe_format('\U0001F4CA Processing ~w items... \u2705 Done!~n', [Count]).
```

### Adaptation Behavior

The `safe_format` predicates automatically adapt based on `get_emoji_level/1`:

**Full Mode:**
```prolog
safe_format('\U0001F680 test~n', []).
% Output: 🚀 test
```

**BMP Mode:**
```prolog
safe_format('\U0001F680 test~n', []).
% Output: [STEP] test
```

**ASCII Mode:**
```prolog
safe_format('\u2705 test~n', []).
% Output: [OK] test
```

## Platform-Specific Handling

### Windows SWI-Prolog

#### Issue: format/3 Unicode Mangling

On Windows, SWI-Prolog's `format/3` mangles non-BMP characters embedded in format strings, even when:
- Unicode escapes are used correctly
- Internal representation is correct (verified via `char_code/2`)
- The terminal supports emoji rendering

#### Workaround: Extract-and-Pass

The `safe_format` implementation on Windows extracts non-BMP characters from format strings and passes them as arguments:

```prolog
safe_format(Stream, Format, Args) :-
    get_emoji_level(Level),
    (   Level = full
    ->  extract_emoji_to_args(Format, AdaptedFormat, EmojiArgs),
        append(EmojiArgs, Args, AllArgs),
        format(Stream, AdaptedFormat, AllArgs)
    ;   adapt_format(Level, Format, AdaptedFormat),
        format(Stream, AdaptedFormat, Args)
    ).
```

**Before:**
```prolog
format('\U0001F680 test~n', [])
% Windows mangles this → mojibake
```

**After:**
```prolog
% Internally transformed to:
format('~w test~n', ['🚀'])
% Works correctly!
```

This workaround is **transparent** to users of `safe_format/2-3`.

### Linux/Unix

No special handling required. Modern Linux terminals with UTF-8 locales support full Unicode natively.

**Requirements:**
- `LANG=en_US.UTF-8` (or similar UTF-8 locale)
- Terminal with Unicode font (most modern terminals)

### macOS

Works identically to Linux. macOS Terminal.app supports full Unicode natively.

### WSL (Windows Subsystem for Linux)

Treated as Linux. Full Unicode support with proper locale configuration.

## File Encoding

### Source Files

All Prolog source files MUST:

1. **Declare encoding** at the top:
   ```prolog
   :- encoding(utf8).
   ```

2. **Use UTF-8 encoding** when saved
   - No BOM (Byte Order Mark)
   - Unix-style line endings (LF) preferred
   - Windows line endings (CRLF) acceptable

3. **Use escape sequences** for non-ASCII characters (see above)

### Generated Files

Bash scripts generated by UnifyWeaver:

1. **Use UTF-8 encoding** for all output
2. **Include shebang** with encoding hint if needed:
   ```bash
   #!/bin/bash
   # -*- coding: utf-8 -*-
   ```

3. **Document emoji** in comments:
   ```bash
   # Status: ✅ (U+2705)
   echo "✅ Complete"
   ```

## Testing Requirements

### Unicode Test Suite

Every release MUST pass the Unicode test suite:

```prolog
?- [examples/test_emoji_levels].
?- main.
```

### Test Coverage

1. **Terminal Detection**
   - Each supported terminal type
   - Unknown terminal fallback

2. **Emoji Levels**
   - ASCII mode output
   - BMP mode output
   - Full mode output
   - Mode switching

3. **Platform-Specific**
   - Windows SWI-Prolog format/3 workaround
   - Linux native Unicode
   - WSL environment

4. **Edge Cases**
   - Empty format strings
   - Format strings with only emoji
   - Mixed emoji and format specifiers
   - Very long format strings

### Example Test

```prolog
test_emoji_rendering :-
    % Test full mode
    set_emoji_level(full),
    safe_format('\U0001F680~n', []),

    % Test BMP mode
    set_emoji_level(bmp),
    safe_format('\U0001F680~n', []),

    % Test ASCII mode
    set_emoji_level(ascii),
    safe_format('\U0001F680~n', []).

% Expected output:
% 🚀
% [STEP]
% [STEP]
```

## Error Handling

### Invalid Unicode Sequences

If an invalid escape sequence is encountered:

```prolog
% Invalid: Not a valid code point
'\UFFFFFFFF'
% Result: Syntax error at parse time
```

SWI-Prolog will raise a syntax error during file loading.

### Missing Emoji Mappings

If a non-BMP character has no fallback mapping:

```prolog
safe_format('\U0001F999~n', []).  % 🦙 (llama, not in fallback table)
```

**Behavior:**
- **Full mode**: Outputs as-is (if terminal supports)
- **BMP/ASCII mode**: Character appears as-is (no fallback available)

**Recommendation:** Add fallback mappings for all emoji used in UnifyWeaver output.

### Terminal Detection Failures

If terminal detection fails (returns `unknown`):

- Defaults to **ASCII mode**
- Logs detection failure to stderr (optional)
- Allows manual override with `set_emoji_level/1`

## Best Practices

### For Core Developers

1. **Always use `safe_format`** for user-facing output
2. **Use escape sequences** for all Unicode in source
3. **Add fallback mappings** for new emoji
4. **Test on Windows** before release
5. **Document Unicode usage** in code comments

### For Plugin Developers

1. **Import platform_compat**:
   ```prolog
   :- use_module(library(unifyweaver/core/platform_compat)).
   ```

2. **Use safe_format** for output:
   ```prolog
   safe_format('\u2705 Plugin loaded~n', []).
   ```

3. **Provide ASCII alternatives** for emoji-heavy output

### For End Users

1. **Set preferred emoji level** in initialization:
   ```prolog
   :- initialization(set_emoji_level(ascii), now).
   ```

2. **Use Windows Terminal** on Windows for best emoji support

3. **Configure fonts** to support emoji (Cascadia Code, Noto Color Emoji)

## Future Enhancements

### Planned

1. **Emoji Configuration File**: Allow users to customize emoji mappings
2. **PowerShell REPL**: Native PowerShell REPL with proper Unicode I/O
3. **Rich Text Support**: ANSI color codes combined with emoji
4. **Performance Optimization**: Cache format string transformations

### Under Consideration

1. **Emoji Skin Tone Support**: Handle U+1F3FB - U+1F3FF modifiers
2. **ZWJ Sequences**: Support Zero-Width Joiner compound emoji
3. **Emoji Flags**: Country flag emoji (U+1F1E6 - U+1F1FF)
4. **Custom Emoji**: User-defined emoji mappings

## References

### Standards

- [Unicode Standard](https://www.unicode.org/standard/standard.html)
- [Unicode Character Database](https://www.unicode.org/ucd/)
- [Emoji Standard](https://unicode.org/reports/tr51/)

### SWI-Prolog

- [Wide Character Support](https://www.swi-prolog.org/man/widechars.html)
- [Character Escapes](https://www.swi-prolog.org/pldoc/man?section=charescapes)
- [I/O Encoding](https://www.swi-prolog.org/pldoc/man?section=encoding)

### Terminal Support

- [Windows Terminal](https://github.com/microsoft/terminal)
- [ConEmu Unicode](https://conemu.github.io/en/UnicodeSupport.html)
- [Terminal Emoji Support Matrix](https://github.com/microsoft/terminal/issues/4130)

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-10-20 | Initial specification |

---

**Maintainer:** UnifyWeaver Development Team
**Last Updated:** October 20, 2025
