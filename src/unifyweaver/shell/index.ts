/**
 * Shell Command Proxy Module
 *
 * Provides secure command execution with per-command validation.
 * Each command in the registry has its own safety checks.
 *
 * @module unifyweaver/shell
 */

export {
  // Types
  Risk,
  ExecutionContext,
  ValidationResult,
  ExecutionResult,
  CommandDefinition,
  ParsedCommand,
  CommandInfo,

  // Functions
  execute,
  validateCommand,
  executeCommand,
  parseCommand,
  isKnownCommand,
  getCommand,
  listCommands,
  registerCommand,

  // Constants
  SANDBOX_ROOT
} from './command-proxy';

// Re-export default
export { default } from './command-proxy';

// Constraint-based validation system
export {
  // Types
  ConstraintFunctor,
  CommandConstraint,
  ConstraintResult,

  // Functions
  evaluateConstraint,
  registerConstraint,
  getConstraints,
  clearConstraints,
  analyzeCommand,
  validateConstraints,
  registerDefaultConstraints,

  // DSL builder
  C
} from './constraints';

// Constraint loader (DB-backed)
export {
  // Types
  SerializedConstraint,
  ConstraintFile,

  // Functions
  parseFunctor,
  loadConstraintsFromJson,
  loadConstraintsFromFile,
  loadConstraintsFromDirectory,
  reloadConstraints,
  generateConstraintPrompt,

  // Examples
  EXAMPLE_CONSTRAINT_FILE,
  CONSTRAINT_GENERATION_PROMPT
} from './constraint-loader';
