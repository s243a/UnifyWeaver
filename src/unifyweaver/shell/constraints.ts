/**
 * Constraint-Based Command Validation
 *
 * A Prolog-inspired constraint system where:
 * - Constraints are declarative facts stored in DB
 * - One generic analyzer validates all commands
 * - Functors pattern-match against args to get constraint operators
 *
 * @module unifyweaver/shell/constraints
 */

import { SANDBOX_ROOT } from './command-proxy';

// ============================================================================
// Constraint Vocabulary - Functor Definitions
// ============================================================================

/**
 * Constraint functors - pattern-matchable constraint types.
 * Each functor defines a constraint that can operate on one or more args.
 */
export type ConstraintFunctor =
  // Arg count constraints
  | { type: 'min_args'; count: number }
  | { type: 'max_args'; count: number }
  | { type: 'exact_args'; count: number }

  // Content constraints (operate on all args)
  | { type: 'no_shell_operators' }              // No |;&`$() in any arg
  | { type: 'no_glob_patterns' }                // No *?[] in any arg
  | { type: 'alphanumeric_only' }               // Only a-zA-Z0-9_- allowed

  // Path constraints
  | { type: 'relative_paths_only' }             // No absolute paths
  | { type: 'within_sandbox' }                  // All paths must be under SANDBOX_ROOT
  | { type: 'no_parent_traversal' }             // No .. in paths
  | { type: 'allowed_extensions'; exts: string[] }  // Whitelist file extensions
  | { type: 'blocked_extensions'; exts: string[] }  // Blacklist file extensions

  // Positional constraints (operate on specific arg indices)
  | { type: 'arg_matches'; index: number; pattern: string }  // Arg at index matches regex
  | { type: 'arg_in_set'; index: number; values: string[] }  // Arg at index in allowed set
  | { type: 'arg_is_path'; index: number }                   // Arg at index is valid path
  | { type: 'arg_is_number'; index: number }                 // Arg at index is numeric
  | { type: 'arg_is_positive'; index: number }               // Arg at index is positive number

  // Relational constraints (operate on multiple args)
  | { type: 'source_exists'; index: number }    // Source file/dir must exist
  | { type: 'dest_not_exists'; index: number }  // Dest must not exist (prevent overwrite)
  | { type: 'same_directory'; indices: number[] }  // All specified args in same dir

  // Flag constraints
  | { type: 'allowed_flags'; flags: string[] }  // Whitelist flags
  | { type: 'blocked_flags'; flags: string[] }  // Blacklist flags
  | { type: 'requires_flag'; flag: string }     // Must include this flag

  // Custom constraints (for extensibility)
  | { type: 'custom'; name: string; params: Record<string, unknown> };

/**
 * A command constraint record - stored in DB.
 */
export interface CommandConstraint {
  command: string;           // Command name or path
  constraint: ConstraintFunctor;
  priority?: number;         // Higher priority checked first (default: 0)
  message?: string;          // Custom error message
}

/**
 * Constraint check result.
 */
export interface ConstraintResult {
  satisfied: boolean;
  constraint: ConstraintFunctor;
  message?: string;
  violatingArgs?: string[];
}

// ============================================================================
// Constraint Evaluators - Pattern Matching Implementation
// ============================================================================

/**
 * Evaluate a single constraint against args.
 * Pattern-matches the functor type to select the appropriate evaluator.
 */
export function evaluateConstraint(
  args: string[],
  constraint: ConstraintFunctor
): ConstraintResult {
  const result: ConstraintResult = {
    satisfied: true,
    constraint
  };

  // Pattern match on functor type
  switch (constraint.type) {
    // === Arg count constraints ===
    case 'min_args':
      result.satisfied = args.length >= constraint.count;
      if (!result.satisfied) {
        result.message = `Expected at least ${constraint.count} args, got ${args.length}`;
      }
      break;

    case 'max_args':
      result.satisfied = args.length <= constraint.count;
      if (!result.satisfied) {
        result.message = `Expected at most ${constraint.count} args, got ${args.length}`;
      }
      break;

    case 'exact_args':
      result.satisfied = args.length === constraint.count;
      if (!result.satisfied) {
        result.message = `Expected exactly ${constraint.count} args, got ${args.length}`;
      }
      break;

    // === Content constraints ===
    case 'no_shell_operators': {
      const shellPattern = /[|;&`$()]/;
      const violating = args.filter(a => shellPattern.test(a));
      result.satisfied = violating.length === 0;
      if (!result.satisfied) {
        result.violatingArgs = violating;
        result.message = `Shell operators not allowed in args`;
      }
      break;
    }

    case 'no_glob_patterns': {
      const globPattern = /[*?\[\]]/;
      const violating = args.filter(a => globPattern.test(a));
      result.satisfied = violating.length === 0;
      if (!result.satisfied) {
        result.violatingArgs = violating;
        result.message = `Glob patterns not allowed in args`;
      }
      break;
    }

    case 'alphanumeric_only': {
      const alphaPattern = /^[a-zA-Z0-9_-]+$/;
      const violating = args.filter(a => !alphaPattern.test(a));
      result.satisfied = violating.length === 0;
      if (!result.satisfied) {
        result.violatingArgs = violating;
        result.message = `Only alphanumeric characters allowed`;
      }
      break;
    }

    // === Path constraints ===
    case 'relative_paths_only': {
      const violating = args.filter(a => a.startsWith('/'));
      result.satisfied = violating.length === 0;
      if (!result.satisfied) {
        result.violatingArgs = violating;
        result.message = `Absolute paths not allowed`;
      }
      break;
    }

    case 'within_sandbox': {
      const { resolve } = require('path');
      const sandboxResolved = resolve(SANDBOX_ROOT);
      const violating = args.filter(a => {
        // Only check path-like args
        if (!a.includes('/') && !a.includes('.')) return false;
        const resolved = resolve(SANDBOX_ROOT, a);
        return !resolved.startsWith(sandboxResolved);
      });
      result.satisfied = violating.length === 0;
      if (!result.satisfied) {
        result.violatingArgs = violating;
        result.message = `Paths must be within sandbox: ${SANDBOX_ROOT}`;
      }
      break;
    }

    case 'no_parent_traversal': {
      const violating = args.filter(a => a.includes('..'));
      result.satisfied = violating.length === 0;
      if (!result.satisfied) {
        result.violatingArgs = violating;
        result.message = `Parent directory traversal (..) not allowed`;
      }
      break;
    }

    case 'allowed_extensions': {
      const { extname } = require('path');
      const pathArgs = args.filter(a => a.includes('.'));
      const violating = pathArgs.filter(a => {
        const ext = extname(a).toLowerCase();
        return ext && !constraint.exts.includes(ext);
      });
      result.satisfied = violating.length === 0;
      if (!result.satisfied) {
        result.violatingArgs = violating;
        result.message = `Only these extensions allowed: ${constraint.exts.join(', ')}`;
      }
      break;
    }

    case 'blocked_extensions': {
      const { extname } = require('path');
      const violating = args.filter(a => {
        const ext = extname(a).toLowerCase();
        return constraint.exts.includes(ext);
      });
      result.satisfied = violating.length === 0;
      if (!result.satisfied) {
        result.violatingArgs = violating;
        result.message = `These extensions are blocked: ${constraint.exts.join(', ')}`;
      }
      break;
    }

    // === Positional constraints ===
    case 'arg_matches': {
      const arg = args[constraint.index];
      if (arg === undefined) {
        result.satisfied = false;
        result.message = `Arg ${constraint.index} not provided`;
      } else {
        const regex = new RegExp(constraint.pattern);
        result.satisfied = regex.test(arg);
        if (!result.satisfied) {
          result.violatingArgs = [arg];
          result.message = `Arg ${constraint.index} must match: ${constraint.pattern}`;
        }
      }
      break;
    }

    case 'arg_in_set': {
      const arg = args[constraint.index];
      if (arg === undefined) {
        result.satisfied = false;
        result.message = `Arg ${constraint.index} not provided`;
      } else {
        result.satisfied = constraint.values.includes(arg);
        if (!result.satisfied) {
          result.violatingArgs = [arg];
          result.message = `Arg ${constraint.index} must be one of: ${constraint.values.join(', ')}`;
        }
      }
      break;
    }

    case 'arg_is_path': {
      const arg = args[constraint.index];
      if (arg === undefined) {
        result.satisfied = false;
        result.message = `Arg ${constraint.index} not provided`;
      } else {
        // Basic path validation - no shell operators, reasonable length
        const pathPattern = /^[a-zA-Z0-9_./-]+$/;
        result.satisfied = pathPattern.test(arg) && arg.length < 4096;
        if (!result.satisfied) {
          result.violatingArgs = [arg];
          result.message = `Arg ${constraint.index} must be a valid path`;
        }
      }
      break;
    }

    case 'arg_is_number': {
      const arg = args[constraint.index];
      if (arg === undefined) {
        result.satisfied = false;
        result.message = `Arg ${constraint.index} not provided`;
      } else {
        result.satisfied = !isNaN(Number(arg));
        if (!result.satisfied) {
          result.violatingArgs = [arg];
          result.message = `Arg ${constraint.index} must be a number`;
        }
      }
      break;
    }

    case 'arg_is_positive': {
      const arg = args[constraint.index];
      if (arg === undefined) {
        result.satisfied = false;
        result.message = `Arg ${constraint.index} not provided`;
      } else {
        const num = Number(arg);
        result.satisfied = !isNaN(num) && num > 0;
        if (!result.satisfied) {
          result.violatingArgs = [arg];
          result.message = `Arg ${constraint.index} must be a positive number`;
        }
      }
      break;
    }

    // === Relational constraints ===
    case 'source_exists': {
      const { existsSync } = require('fs');
      const { resolve } = require('path');
      const arg = args[constraint.index];
      if (arg === undefined) {
        result.satisfied = false;
        result.message = `Source arg ${constraint.index} not provided`;
      } else {
        const resolved = resolve(SANDBOX_ROOT, arg);
        result.satisfied = existsSync(resolved);
        if (!result.satisfied) {
          result.violatingArgs = [arg];
          result.message = `Source does not exist: ${arg}`;
        }
      }
      break;
    }

    case 'dest_not_exists': {
      const { existsSync } = require('fs');
      const { resolve } = require('path');
      const arg = args[constraint.index];
      if (arg === undefined) {
        // If dest not provided, that's OK (constraint passes)
        result.satisfied = true;
      } else {
        const resolved = resolve(SANDBOX_ROOT, arg);
        result.satisfied = !existsSync(resolved);
        if (!result.satisfied) {
          result.violatingArgs = [arg];
          result.message = `Destination already exists: ${arg}`;
        }
      }
      break;
    }

    case 'same_directory': {
      const { dirname, resolve } = require('path');
      const paths = constraint.indices.map(i => args[i]).filter(Boolean);
      if (paths.length < 2) {
        result.satisfied = true; // Not enough paths to compare
      } else {
        const dirs = paths.map(p => dirname(resolve(SANDBOX_ROOT, p)));
        const firstDir = dirs[0];
        result.satisfied = dirs.every(d => d === firstDir);
        if (!result.satisfied) {
          result.violatingArgs = paths;
          result.message = `All paths must be in the same directory`;
        }
      }
      break;
    }

    // === Flag constraints ===
    case 'allowed_flags': {
      const flags = args.filter(a => a.startsWith('-'));
      const violating = flags.filter(f => !constraint.flags.includes(f));
      result.satisfied = violating.length === 0;
      if (!result.satisfied) {
        result.violatingArgs = violating;
        result.message = `Only these flags allowed: ${constraint.flags.join(', ')}`;
      }
      break;
    }

    case 'blocked_flags': {
      const violating = args.filter(a =>
        a.startsWith('-') && constraint.flags.includes(a)
      );
      result.satisfied = violating.length === 0;
      if (!result.satisfied) {
        result.violatingArgs = violating;
        result.message = `These flags are blocked: ${constraint.flags.join(', ')}`;
      }
      break;
    }

    case 'requires_flag': {
      result.satisfied = args.some(a => a === constraint.flag);
      if (!result.satisfied) {
        result.message = `Required flag missing: ${constraint.flag}`;
      }
      break;
    }

    // === Custom constraints ===
    case 'custom':
      // Custom constraints require external handler registration
      result.satisfied = true; // Pass by default, or look up handler
      result.message = `Custom constraint: ${constraint.name}`;
      break;

    default:
      // Exhaustiveness check - TypeScript will error if we miss a case
      const _exhaustive: never = constraint;
      result.satisfied = false;
      result.message = `Unknown constraint type`;
  }

  return result;
}

// ============================================================================
// Constraint Analyzer - Generic Validation Engine
// ============================================================================

/**
 * In-memory constraint store (would be replaced with DB in production).
 */
const constraintStore: CommandConstraint[] = [];

/**
 * Register a constraint for a command.
 */
export function registerConstraint(
  command: string,
  constraint: ConstraintFunctor,
  options?: { priority?: number; message?: string }
): void {
  constraintStore.push({
    command,
    constraint,
    priority: options?.priority ?? 0,
    message: options?.message
  });
}

/**
 * Get all constraints for a command, sorted by priority (desc).
 */
export function getConstraints(command: string): CommandConstraint[] {
  return constraintStore
    .filter(c => c.command === command || c.command === '*')
    .sort((a, b) => (b.priority ?? 0) - (a.priority ?? 0));
}

/**
 * Clear all constraints (for testing).
 */
export function clearConstraints(): void {
  constraintStore.length = 0;
}

/**
 * Analyze a command against all registered constraints.
 * Returns all constraint results, including violations.
 */
export function analyzeCommand(
  command: string,
  args: string[]
): { valid: boolean; results: ConstraintResult[] } {
  const constraints = getConstraints(command);
  const results: ConstraintResult[] = [];

  for (const { constraint, message } of constraints) {
    const result = evaluateConstraint(args, constraint);

    // Override message if custom message provided
    if (!result.satisfied && message) {
      result.message = message;
    }

    results.push(result);
  }

  return {
    valid: results.every(r => r.satisfied),
    results
  };
}

/**
 * Validate command and return first violation (if any).
 * More efficient than analyzeCommand when you just need pass/fail.
 */
export function validateConstraints(
  command: string,
  args: string[]
): { ok: boolean; error?: string; violatingArgs?: string[] } {
  const constraints = getConstraints(command);

  for (const { constraint, message } of constraints) {
    const result = evaluateConstraint(args, constraint);

    if (!result.satisfied) {
      return {
        ok: false,
        error: message ?? result.message,
        violatingArgs: result.violatingArgs
      };
    }
  }

  return { ok: true };
}

// ============================================================================
// Constraint DSL - Helper functions for defining constraints
// ============================================================================

/**
 * Constraint builder helpers for cleaner syntax.
 */
export const C = {
  // Arg count
  minArgs: (count: number): ConstraintFunctor => ({ type: 'min_args', count }),
  maxArgs: (count: number): ConstraintFunctor => ({ type: 'max_args', count }),
  exactArgs: (count: number): ConstraintFunctor => ({ type: 'exact_args', count }),

  // Content
  noShellOps: (): ConstraintFunctor => ({ type: 'no_shell_operators' }),
  noGlobs: (): ConstraintFunctor => ({ type: 'no_glob_patterns' }),
  alphanumeric: (): ConstraintFunctor => ({ type: 'alphanumeric_only' }),

  // Paths
  relativePaths: (): ConstraintFunctor => ({ type: 'relative_paths_only' }),
  withinSandbox: (): ConstraintFunctor => ({ type: 'within_sandbox' }),
  noTraversal: (): ConstraintFunctor => ({ type: 'no_parent_traversal' }),
  allowedExts: (exts: string[]): ConstraintFunctor => ({ type: 'allowed_extensions', exts }),
  blockedExts: (exts: string[]): ConstraintFunctor => ({ type: 'blocked_extensions', exts }),

  // Positional
  argMatches: (index: number, pattern: string): ConstraintFunctor =>
    ({ type: 'arg_matches', index, pattern }),
  argIn: (index: number, values: string[]): ConstraintFunctor =>
    ({ type: 'arg_in_set', index, values }),
  argIsPath: (index: number): ConstraintFunctor => ({ type: 'arg_is_path', index }),
  argIsNumber: (index: number): ConstraintFunctor => ({ type: 'arg_is_number', index }),
  argIsPositive: (index: number): ConstraintFunctor => ({ type: 'arg_is_positive', index }),

  // Relational
  sourceExists: (index: number): ConstraintFunctor => ({ type: 'source_exists', index }),
  destNotExists: (index: number): ConstraintFunctor => ({ type: 'dest_not_exists', index }),
  sameDir: (indices: number[]): ConstraintFunctor => ({ type: 'same_directory', indices }),

  // Flags
  allowedFlags: (flags: string[]): ConstraintFunctor => ({ type: 'allowed_flags', flags }),
  blockedFlags: (flags: string[]): ConstraintFunctor => ({ type: 'blocked_flags', flags }),
  requiresFlag: (flag: string): ConstraintFunctor => ({ type: 'requires_flag', flag }),

  // Custom
  custom: (name: string, params: Record<string, unknown> = {}): ConstraintFunctor =>
    ({ type: 'custom', name, params })
};

// ============================================================================
// Example Constraint Definitions
// ============================================================================

/**
 * Example: Register constraints for common commands.
 * In production, these would be loaded from a database.
 */
export function registerDefaultConstraints(): void {
  // cp command constraints
  registerConstraint('cp', C.minArgs(2), { message: 'cp requires source and destination' });
  registerConstraint('cp', C.noShellOps());
  registerConstraint('cp', C.noTraversal());
  registerConstraint('cp', C.withinSandbox());
  registerConstraint('cp', C.blockedFlags(['-r', '-R', '--recursive']), {
    message: 'Recursive copy not allowed'
  });

  // mv command constraints
  registerConstraint('mv', C.minArgs(2));
  registerConstraint('mv', C.noShellOps());
  registerConstraint('mv', C.noTraversal());
  registerConstraint('mv', C.withinSandbox());

  // rm command constraints
  registerConstraint('rm', C.minArgs(1));
  registerConstraint('rm', C.noShellOps());
  registerConstraint('rm', C.noTraversal());
  registerConstraint('rm', C.withinSandbox());
  registerConstraint('rm', C.blockedFlags(['-r', '-R', '--recursive', '-f', '--force']), {
    message: 'Recursive/force delete not allowed'
  });

  // cat command constraints
  registerConstraint('cat', C.minArgs(1));
  registerConstraint('cat', C.noShellOps());
  registerConstraint('cat', C.withinSandbox());
  registerConstraint('cat', C.maxArgs(5), { message: 'Too many files' });

  // Global constraints (apply to all commands)
  registerConstraint('*', C.maxArgs(100), {
    priority: -1,
    message: 'Too many arguments'
  });
}
