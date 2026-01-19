<!--
SPDX-License-Identifier: MIT OR Apache-2.0
Copyright (c) 2025 John William Creighton (@s243a)
-->
# Shell Command Constraint System

A declarative, Prolog-inspired constraint system for validating shell commands before execution. Constraints are stored as facts in a database, and a generic analyzer validates all commands - no per-command code generation required.

## Overview

The shell constraint system provides:

- **Declarative constraint facts** that can be stored in JSON/database
- **Pattern-matching functor types** via TypeScript discriminated unions
- **Generic analyzer** that works for all commands
- **LLM-friendly format** - AI generates constraint facts, not code

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  LLM generates  │────▶│  Constraint DB   │────▶│ Generic Analyzer│
│  constraint     │     │  (JSON/SQLite)   │     │ evaluates all   │
│  facts          │     │                  │     │ commands        │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

## Constraint Vocabulary

### Arg Count Constraints

| Functor | Description |
|---------|-------------|
| `min_args(n)` | Minimum number of arguments |
| `max_args(n)` | Maximum number of arguments |
| `exact_args(n)` | Exact number required |

### Content Constraints

| Functor | Description |
|---------|-------------|
| `no_shell_operators` | Block `\|;&\`$()` in args |
| `no_glob_patterns` | Block `*?[]` in args |
| `alphanumeric_only` | Only `a-zA-Z0-9_-` allowed |

### Path Constraints

| Functor | Description |
|---------|-------------|
| `relative_paths_only` | Block absolute paths |
| `within_sandbox` | Paths must be under sandbox root |
| `no_parent_traversal` | Block `..` in paths |
| `allowed_extensions([exts])` | Whitelist file extensions |
| `blocked_extensions([exts])` | Blacklist file extensions |

### Positional Constraints

| Functor | Description |
|---------|-------------|
| `arg_matches(i, regex)` | Arg at index must match regex |
| `arg_in_set(i, [values])` | Arg at index must be one of values |
| `arg_is_path(i)` | Arg at index must be valid path |
| `arg_is_number(i)` | Arg at index must be numeric |
| `arg_is_positive(i)` | Arg at index must be positive |

### Relational Constraints

| Functor | Description |
|---------|-------------|
| `source_exists(i)` | Source file at index must exist |
| `dest_not_exists(i)` | Destination must not exist |
| `same_directory([i,j])` | Specified args must be in same dir |

### Flag Constraints

| Functor | Description |
|---------|-------------|
| `allowed_flags([flags])` | Whitelist flags |
| `blocked_flags([flags])` | Blacklist flags |
| `requires_flag(flag)` | Must include this flag |

## Usage

### TypeScript DSL

```typescript
import { C, registerConstraint, validateConstraints } from './shell';

// Register constraints using the DSL
registerConstraint('cp', C.minArgs(2));
registerConstraint('cp', C.noShellOps());
registerConstraint('cp', C.withinSandbox());
registerConstraint('cp', C.blockedFlags(['-r', '-R']));

// Validate a command
const result = validateConstraints('cp', ['file.txt', 'backup.txt']);
if (!result.ok) {
  console.error(result.error);
}
```

### JSON Constraint Files (LLM-Generated)

```json
{
  "version": "1.0",
  "description": "Constraints for backup.sh script",
  "constraints": [
    {
      "command": "./backup.sh",
      "functor": "min_args",
      "args": [1],
      "message": "backup.sh requires at least one directory"
    },
    {
      "command": "./backup.sh",
      "functor": "no_shell_operators",
      "priority": 10,
      "message": "Shell operators not allowed in backup paths"
    },
    {
      "command": "./backup.sh",
      "functor": "within_sandbox"
    },
    {
      "command": "./backup.sh",
      "functor": "blocked_extensions",
      "args": [[".exe", ".dll", ".so"]],
      "message": "Cannot backup binary files"
    }
  ]
}
```

### Loading Constraints

```typescript
import { loadConstraintsFromJson, loadConstraintsFromDirectory } from './shell';

// Load from JSON string
loadConstraintsFromJson(jsonString);

// Load all .json files from a directory
loadConstraintsFromDirectory('./constraints');
```

## LLM Constraint Generation

The system includes a prompt template for LLMs to generate constraint facts:

```typescript
import { generateConstraintPrompt } from './shell';

const prompt = generateConstraintPrompt('./backup.sh', scriptContent);
// Send to LLM, receive JSON constraint file
```

The LLM acts as a "security policy compiler" - it analyzes scripts and generates declarative constraint facts, not executable code.

## Full Analysis

For detailed constraint checking, use `analyzeCommand`:

```typescript
import { analyzeCommand } from './shell';

const analysis = analyzeCommand('rm', ['-rf', '../important']);

console.log(`Valid: ${analysis.valid}`);
for (const r of analysis.results) {
  console.log(`${r.constraint.type}: ${r.satisfied ? 'passed' : r.message}`);
}
```

## Pattern Matching

The constraint system uses TypeScript discriminated unions for type-safe pattern matching:

```typescript
type ConstraintFunctor =
  | { type: 'min_args'; count: number }
  | { type: 'max_args'; count: number }
  | { type: 'no_shell_operators' }
  | { type: 'arg_matches'; index: number; pattern: string }
  // ... etc

function evaluateConstraint(args: string[], constraint: ConstraintFunctor) {
  switch (constraint.type) {
    case 'min_args':
      return args.length >= constraint.count;
    case 'no_shell_operators':
      return !args.some(a => /[|;&`$()]/.test(a));
    // TypeScript ensures exhaustive handling
  }
}
```

## Storage Abstraction

The constraint system uses a database-agnostic storage layer. Swap backends without code changes:

```typescript
import { initStore, SQLiteConstraintStore, FileConstraintStore } from './shell';

// Use SQLite (persistent)
await initStore({ type: 'sqlite', path: './constraints.db' });

// Or use file-based JSON storage
await initStore({ type: 'file', path: './constraints.json' });

// Or use in-memory (default, for testing)
await initStore({ type: 'memory' });
```

Available backends:
- **MemoryConstraintStore** - Fast, non-persistent (testing/dev)
- **SQLiteConstraintStore** - Persistent with indexing
- **FileConstraintStore** - JSON file persistence

## LLM Constraint Generation

Automatically generate constraints for scripts using an LLM:

```typescript
import { generateConstraints, ClaudeCLIProvider, getBestProvider } from './shell';

// Auto-detect best available provider
const provider = await getBestProvider();

// Or use a specific provider
const result = await generateConstraints('./backup.sh', scriptContent, {
  provider: new ClaudeCLIProvider({ model: 'sonnet' })
});

if (result.success) {
  console.log(`Generated ${result.constraints.length} constraints`);
}
```

The LLM analyzes the script and outputs declarative constraint facts - no code generation.

### Available LLM Providers

**CLI-based (no API keys needed):**
```typescript
import { ClaudeCLIProvider, GeminiCLIProvider, OllamaCLIProvider } from './shell';

// Claude Code CLI
const claude = new ClaudeCLIProvider({ model: 'sonnet' });  // or opus, haiku

// Gemini CLI
const gemini = new GeminiCLIProvider({ model: 'gemini-2.0-flash' });

// Ollama (local models)
const ollama = new OllamaCLIProvider({ model: 'llama3' });
```

**API-based:**
```typescript
import { AnthropicProvider, OpenAIProvider } from './shell';

// Requires ANTHROPIC_API_KEY env var
const anthropic = new AnthropicProvider();

// Requires OPENAI_API_KEY env var (also works with local servers)
const openai = new OpenAIProvider({ baseUrl: 'http://localhost:1234/v1' });
```

**Auto-detection:**
```typescript
import { getBestProvider, getAvailableCLIProviders } from './shell';

// Returns best available provider (prefers CLI)
const provider = await getBestProvider();

// List installed CLI tools
const available = await getAvailableCLIProviders();
// Returns: ['claude-cli', 'gemini-cli', 'ollama']
```

## File Edit Review

Validate file edits against declarative constraints:

```typescript
import { EC, registerEditConstraint, reviewFileEdit } from './shell';

// Register edit constraints
registerEditConstraint('**/*.ts', EC.noRemoveSecurityChecks());
registerEditConstraint('**/*.ts', EC.maxLinesChanged(50));
registerEditConstraint('**/config.ts', EC.noChangeExports());

// Review a proposed edit
const result = await reviewFileEdit(
  'src/auth.ts',
  originalContent,
  proposedContent
);

if (!result.allowed) {
  console.log(result.summary);
  for (const r of result.constraintResults.filter(r => !r.satisfied)) {
    console.log(`  Violation: ${r.message}`);
  }
}
```

### Edit Constraint Vocabulary

| Functor | Description |
|---------|-------------|
| `no_delete_lines_containing([patterns])` | Block deletion of matching lines |
| `no_modify_lines_containing([patterns])` | Block modification of matching lines |
| `no_add_patterns([patterns])` | Block adding content matching patterns |
| `max_lines_changed(n)` | Limit number of changed lines |
| `no_change_imports` | Protect import statements |
| `no_change_exports` | Protect export statements |
| `no_remove_security_checks` | Block removal of validation code |
| `preserve_function(name)` | Protect specific function |
| `file_must_parse` | Require valid syntax after edit |

## Module Structure

```
src/unifyweaver/shell/
├── index.ts                    # Module exports
├── command-proxy.ts            # Original per-command validators
├── proxy-cli.ts                # CLI wrapper
├── constraints.ts              # Constraint vocabulary & analyzer
├── constraint-loader.ts        # JSON loading
├── constraint-store.ts         # Storage abstraction layer
├── llm-constraint-generator.ts # LLM API providers
├── llm-cli-providers.ts        # LLM CLI providers (claude, gemini, ollama)
├── edit-review.ts              # File edit validation
├── constraint-demo.ts          # Command constraint demo
├── store-and-edit-demo.ts      # Storage & edit demo
└── llm-provider-demo.ts        # LLM provider demo
```

## Running the Demos

```bash
# Command constraint demo
npx ts-node src/unifyweaver/shell/constraint-demo.ts

# Storage and edit review demo
npx ts-node src/unifyweaver/shell/store-and-edit-demo.ts

# LLM provider demo (check available providers)
npx ts-node src/unifyweaver/shell/llm-provider-demo.ts

# LLM provider demo with live tests
npx ts-node src/unifyweaver/shell/llm-provider-demo.ts --test
```

## Integration with Command Proxy

The constraint system can be used alongside or instead of the existing `command-proxy.ts` validators. For new commands, generate constraints via LLM rather than writing custom validation code.

## See Also

- [`command-proxy.ts`](../src/unifyweaver/shell/command-proxy.ts) - Original per-command validators
- [`CONSTRAINT_SYSTEM.md`](./CONSTRAINT_SYSTEM.md) - Prolog deduplication constraints (different system)
