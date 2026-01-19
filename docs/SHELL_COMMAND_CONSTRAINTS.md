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

## Module Structure

```
src/unifyweaver/shell/
├── index.ts              # Module exports
├── command-proxy.ts      # Original per-command validators
├── proxy-cli.ts          # CLI wrapper
├── constraints.ts        # Constraint vocabulary & analyzer
├── constraint-loader.ts  # JSON/DB loading
└── constraint-demo.ts    # Demo script
```

## Running the Demo

```bash
npx ts-node src/unifyweaver/shell/constraint-demo.ts
```

## Integration with Command Proxy

The constraint system can be used alongside or instead of the existing `command-proxy.ts` validators. For new commands, generate constraints via LLM rather than writing custom validation code.

## See Also

- [`command-proxy.ts`](../src/unifyweaver/shell/command-proxy.ts) - Original per-command validators
- [`CONSTRAINT_SYSTEM.md`](./CONSTRAINT_SYSTEM.md) - Prolog deduplication constraints (different system)
