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

// Constraint store (storage abstraction)
export {
  // Types
  ConstraintStore,
  StoredConstraint,
  StoreType,
  StoreOptions,

  // Implementations
  MemoryConstraintStore,
  SQLiteConstraintStore,
  FileConstraintStore,

  // Functions
  createConstraintStore,
  getStore,
  setStore,
  initStore,
  toStoredConstraint
} from './constraint-store';

// LLM constraint generator
export {
  // Types
  LLMProvider,
  LLMOptions,
  GeneratorOptions,
  GenerationResult,
  EditReviewResult as LLMEditReviewResult,
  EditConstraintFunctor,
  EditConstraint,

  // Providers
  AnthropicProvider,
  OpenAIProvider,

  // Functions
  generateConstraints,
  generateConstraintsBatch,
  reviewEdit as llmReviewEdit,
  generateEditConstraints,
  getDefaultProvider,
  setDefaultProvider
} from './llm-constraint-generator';

// Edit review (constraint-based file validation)
export {
  // Types
  LineDiff,
  EditConstraintResult,
  EditReviewOptions,
  EditReviewResult,

  // Functions
  registerEditConstraint,
  getEditConstraints,
  clearEditConstraints,
  computeDiff,
  evaluateEditConstraint,
  reviewFileEdit,
  registerDefaultEditConstraints,

  // DSL builder
  EC
} from './edit-review';
