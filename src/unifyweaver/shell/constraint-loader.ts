/**
 * Constraint Loader - Database-backed constraint storage
 *
 * Loads constraint facts from a database or JSON files.
 * The LLM generates these facts; this module just loads and registers them.
 *
 * @module unifyweaver/shell/constraint-loader
 */

import { readFileSync, existsSync, readdirSync } from 'fs';
import { join } from 'path';
import {
  ConstraintFunctor,
  registerConstraint,
  clearConstraints
} from './constraints';

// ============================================================================
// Serialized Constraint Format
// ============================================================================

/**
 * JSON-serializable constraint record.
 * This is what the LLM generates and what gets stored in the DB.
 */
export interface SerializedConstraint {
  command: string;
  functor: string;      // Functor type name
  args?: unknown[];     // Functor arguments
  priority?: number;
  message?: string;
}

/**
 * A constraint file containing multiple constraints for related commands.
 */
export interface ConstraintFile {
  version: string;
  description?: string;
  constraints: SerializedConstraint[];
}

// ============================================================================
// Functor Parser - Converts serialized functors to typed ConstraintFunctor
// ============================================================================

/**
 * Parse a serialized functor into a typed ConstraintFunctor.
 * This is where pattern matching happens - functor name maps to type.
 */
export function parseFunctor(
  functor: string,
  args: unknown[] = []
): ConstraintFunctor {
  switch (functor) {
    // Arg count
    case 'min_args':
      return { type: 'min_args', count: Number(args[0]) };
    case 'max_args':
      return { type: 'max_args', count: Number(args[0]) };
    case 'exact_args':
      return { type: 'exact_args', count: Number(args[0]) };

    // Content
    case 'no_shell_operators':
      return { type: 'no_shell_operators' };
    case 'no_glob_patterns':
      return { type: 'no_glob_patterns' };
    case 'alphanumeric_only':
      return { type: 'alphanumeric_only' };

    // Paths
    case 'relative_paths_only':
      return { type: 'relative_paths_only' };
    case 'within_sandbox':
      return { type: 'within_sandbox' };
    case 'no_parent_traversal':
      return { type: 'no_parent_traversal' };
    case 'allowed_extensions':
      return { type: 'allowed_extensions', exts: args[0] as string[] };
    case 'blocked_extensions':
      return { type: 'blocked_extensions', exts: args[0] as string[] };

    // Positional
    case 'arg_matches':
      return { type: 'arg_matches', index: Number(args[0]), pattern: String(args[1]) };
    case 'arg_in_set':
      return { type: 'arg_in_set', index: Number(args[0]), values: args[1] as string[] };
    case 'arg_is_path':
      return { type: 'arg_is_path', index: Number(args[0]) };
    case 'arg_is_number':
      return { type: 'arg_is_number', index: Number(args[0]) };
    case 'arg_is_positive':
      return { type: 'arg_is_positive', index: Number(args[0]) };

    // Relational
    case 'source_exists':
      return { type: 'source_exists', index: Number(args[0]) };
    case 'dest_not_exists':
      return { type: 'dest_not_exists', index: Number(args[0]) };
    case 'same_directory':
      return { type: 'same_directory', indices: args[0] as number[] };

    // Flags
    case 'allowed_flags':
      return { type: 'allowed_flags', flags: args[0] as string[] };
    case 'blocked_flags':
      return { type: 'blocked_flags', flags: args[0] as string[] };
    case 'requires_flag':
      return { type: 'requires_flag', flag: String(args[0]) };

    // Custom
    case 'custom':
      return {
        type: 'custom',
        name: String(args[0]),
        params: (args[1] as Record<string, unknown>) ?? {}
      };

    default:
      throw new Error(`Unknown functor: ${functor}`);
  }
}

// ============================================================================
// Constraint Loading Functions
// ============================================================================

/**
 * Load constraints from a JSON string.
 */
export function loadConstraintsFromJson(json: string): number {
  const data: ConstraintFile = JSON.parse(json);
  let loaded = 0;

  for (const c of data.constraints) {
    const functor = parseFunctor(c.functor, c.args);
    registerConstraint(c.command, functor, {
      priority: c.priority,
      message: c.message
    });
    loaded++;
  }

  return loaded;
}

/**
 * Load constraints from a JSON file.
 */
export function loadConstraintsFromFile(filepath: string): number {
  if (!existsSync(filepath)) {
    throw new Error(`Constraint file not found: ${filepath}`);
  }

  const json = readFileSync(filepath, 'utf-8');
  return loadConstraintsFromJson(json);
}

/**
 * Load all constraint files from a directory.
 */
export function loadConstraintsFromDirectory(dirpath: string): number {
  if (!existsSync(dirpath)) {
    return 0;
  }

  let totalLoaded = 0;
  const files = readdirSync(dirpath).filter(f => f.endsWith('.json'));

  for (const file of files) {
    const filepath = join(dirpath, file);
    totalLoaded += loadConstraintsFromFile(filepath);
  }

  return totalLoaded;
}

/**
 * Reload all constraints (clear and load fresh).
 */
export function reloadConstraints(dirpath: string): number {
  clearConstraints();
  return loadConstraintsFromDirectory(dirpath);
}

// ============================================================================
// Example Constraint File Format
// ============================================================================

/**
 * Example of what an LLM would generate for a new script.
 * This would be stored in the database and loaded at runtime.
 */
export const EXAMPLE_CONSTRAINT_FILE: ConstraintFile = {
  version: '1.0',
  description: 'Constraints for backup.sh script',
  constraints: [
    {
      command: './backup.sh',
      functor: 'min_args',
      args: [1],
      message: 'backup.sh requires at least one directory to backup'
    },
    {
      command: './backup.sh',
      functor: 'max_args',
      args: [5],
      message: 'backup.sh accepts at most 5 directories'
    },
    {
      command: './backup.sh',
      functor: 'no_shell_operators',
      priority: 10,
      message: 'Shell operators not allowed in backup paths'
    },
    {
      command: './backup.sh',
      functor: 'relative_paths_only',
      message: 'Only relative paths allowed for backup'
    },
    {
      command: './backup.sh',
      functor: 'no_parent_traversal',
      message: 'Cannot backup parent directories'
    },
    {
      command: './backup.sh',
      functor: 'within_sandbox',
      priority: 10,
      message: 'Backup paths must be within sandbox'
    },
    {
      command: './backup.sh',
      functor: 'blocked_extensions',
      args: [['.exe', '.dll', '.so']],
      message: 'Cannot backup binary files'
    }
  ]
};

// ============================================================================
// LLM Constraint Generation Interface
// ============================================================================

/**
 * Prompt template for LLM to generate constraints.
 * The LLM should respond with a JSON ConstraintFile.
 */
export const CONSTRAINT_GENERATION_PROMPT = `
You are a security policy compiler. Given a shell script, analyze it and generate
constraint facts that will validate safe usage of the script.

Available constraint functors:
- min_args(count): Minimum number of arguments
- max_args(count): Maximum number of arguments
- exact_args(count): Exact number of arguments required
- no_shell_operators: Block shell operators |;&\`$() in args
- no_glob_patterns: Block glob patterns *?[] in args
- alphanumeric_only: Only allow alphanumeric characters
- relative_paths_only: Block absolute paths
- within_sandbox: All paths must be under sandbox root
- no_parent_traversal: Block .. in paths
- allowed_extensions(exts[]): Whitelist file extensions
- blocked_extensions(exts[]): Blacklist file extensions
- arg_matches(index, pattern): Arg at index must match regex
- arg_in_set(index, values[]): Arg at index must be one of values
- arg_is_path(index): Arg at index must be valid path
- arg_is_number(index): Arg at index must be numeric
- arg_is_positive(index): Arg at index must be positive number
- source_exists(index): Source file at index must exist
- dest_not_exists(index): Destination at index must not exist
- same_directory(indices[]): All specified args in same directory
- allowed_flags(flags[]): Whitelist flags
- blocked_flags(flags[]): Blacklist flags
- requires_flag(flag): Must include this flag

Respond with a JSON object in this format:
{
  "version": "1.0",
  "description": "Description of constraints",
  "constraints": [
    {
      "command": "command_name",
      "functor": "functor_name",
      "args": [arg1, arg2, ...],  // optional
      "priority": 0,              // optional, higher = checked first
      "message": "Error message"  // optional
    }
  ]
}

Script to analyze:
`;

/**
 * Generate a constraint generation request for a script.
 */
export function generateConstraintPrompt(
  scriptPath: string,
  scriptContent: string
): string {
  return `${CONSTRAINT_GENERATION_PROMPT}

Path: ${scriptPath}

Content:
\`\`\`bash
${scriptContent}
\`\`\`

Generate appropriate constraints for safe execution of this script.
`;
}
