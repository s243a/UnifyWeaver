/**
 * LLM Constraint Generator
 *
 * Uses an LLM to analyze scripts and generate constraint facts.
 * The LLM acts as a "security policy compiler" - it outputs declarative
 * constraints, not executable code.
 *
 * @module unifyweaver/shell/llm-constraint-generator
 */

import { ConstraintFunctor } from './constraints';
import { StoredConstraint, ConstraintStore, getStore } from './constraint-store';
import { parseFunctor, SerializedConstraint, ConstraintFile } from './constraint-loader';

// ============================================================================
// LLM Provider Interface
// ============================================================================

/**
 * Abstract interface for LLM providers.
 * Implementations can use Anthropic, OpenAI, local models, etc.
 */
export interface LLMProvider {
  /**
   * Generate a completion for the given prompt.
   */
  complete(prompt: string, options?: LLMOptions): Promise<string>;

  /**
   * Name of the provider for logging.
   */
  readonly name: string;
}

export interface LLMOptions {
  maxTokens?: number;
  temperature?: number;
  systemPrompt?: string;
}

// ============================================================================
// Anthropic Provider
// ============================================================================

/**
 * Anthropic Claude LLM provider.
 */
export class AnthropicProvider implements LLMProvider {
  readonly name = 'anthropic';
  private apiKey: string;
  private model: string;
  private baseUrl: string;

  constructor(options: {
    apiKey?: string;
    model?: string;
    baseUrl?: string;
  } = {}) {
    this.apiKey = options.apiKey || process.env.ANTHROPIC_API_KEY || '';
    this.model = options.model || 'claude-sonnet-4-20250514';
    this.baseUrl = options.baseUrl || 'https://api.anthropic.com';

    if (!this.apiKey) {
      throw new Error('Anthropic API key required. Set ANTHROPIC_API_KEY env var.');
    }
  }

  async complete(prompt: string, options: LLMOptions = {}): Promise<string> {
    const response = await fetch(`${this.baseUrl}/v1/messages`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': this.apiKey,
        'anthropic-version': '2023-06-01'
      },
      body: JSON.stringify({
        model: this.model,
        max_tokens: options.maxTokens || 4096,
        system: options.systemPrompt || CONSTRAINT_SYSTEM_PROMPT,
        messages: [{ role: 'user', content: prompt }]
      })
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`Anthropic API error: ${response.status} - ${error}`);
    }

    const data = await response.json() as any;
    return data.content[0]?.text || '';
  }
}

// ============================================================================
// OpenAI-Compatible Provider
// ============================================================================

/**
 * OpenAI-compatible LLM provider.
 * Works with OpenAI, Azure OpenAI, local servers (llama.cpp, ollama), etc.
 */
export class OpenAIProvider implements LLMProvider {
  readonly name = 'openai';
  private apiKey: string;
  private model: string;
  private baseUrl: string;

  constructor(options: {
    apiKey?: string;
    model?: string;
    baseUrl?: string;
  } = {}) {
    this.apiKey = options.apiKey || process.env.OPENAI_API_KEY || '';
    this.model = options.model || 'gpt-4';
    this.baseUrl = options.baseUrl || 'https://api.openai.com/v1';
  }

  async complete(prompt: string, options: LLMOptions = {}): Promise<string> {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json'
    };

    if (this.apiKey) {
      headers['Authorization'] = `Bearer ${this.apiKey}`;
    }

    const messages: any[] = [];

    if (options.systemPrompt) {
      messages.push({ role: 'system', content: options.systemPrompt });
    } else {
      messages.push({ role: 'system', content: CONSTRAINT_SYSTEM_PROMPT });
    }

    messages.push({ role: 'user', content: prompt });

    const response = await fetch(`${this.baseUrl}/chat/completions`, {
      method: 'POST',
      headers,
      body: JSON.stringify({
        model: this.model,
        max_tokens: options.maxTokens || 4096,
        temperature: options.temperature ?? 0.1,
        messages
      })
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`OpenAI API error: ${response.status} - ${error}`);
    }

    const data = await response.json() as any;
    return data.choices[0]?.message?.content || '';
  }
}

// ============================================================================
// System Prompts
// ============================================================================

const CONSTRAINT_SYSTEM_PROMPT = `You are a security policy compiler. Your task is to analyze shell scripts and generate constraint facts that will validate safe usage of the script.

You MUST respond with ONLY a valid JSON object. No explanations, no markdown code blocks, just the JSON.

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

Response format:
{
  "version": "1.0",
  "description": "Brief description",
  "constraints": [
    {
      "command": "command_name",
      "functor": "functor_name",
      "args": [arg1, arg2],
      "priority": 0,
      "message": "Error message"
    }
  ]
}`;

const SCRIPT_ANALYSIS_PROMPT = `Analyze this shell script and generate security constraints for safe execution.

Script path: {path}

Script content:
\`\`\`bash
{content}
\`\`\`

Consider:
1. What arguments does the script expect?
2. What could go wrong if malicious input is provided?
3. What paths should be restricted?
4. What flags should be allowed/blocked?

Generate appropriate constraints as a JSON object.`;

const EDIT_REVIEW_PROMPT = `Review this proposed file edit and determine if it should be allowed.

File: {path}

Original content:
\`\`\`
{original}
\`\`\`

Proposed new content:
\`\`\`
{proposed}
\`\`\`

Analyze:
1. Does this edit introduce security vulnerabilities?
2. Does it add shell injection risks?
3. Does it modify security-critical code?
4. Does it bypass existing constraints?

Respond with a JSON object:
{
  "allowed": true/false,
  "reason": "explanation",
  "risks": ["list", "of", "concerns"],
  "suggestions": ["optional", "improvements"]
}`;

// ============================================================================
// Constraint Generator
// ============================================================================

export interface GeneratorOptions {
  provider?: LLMProvider;
  store?: ConstraintStore;
  source?: string;  // Source tag for generated constraints
}

/**
 * Result of constraint generation.
 */
export interface GenerationResult {
  success: boolean;
  command: string;
  constraints: StoredConstraint[];
  ids: number[];
  error?: string;
  raw?: string;  // Raw LLM response for debugging
}

/**
 * Generate constraints for a script using an LLM.
 */
export async function generateConstraints(
  scriptPath: string,
  scriptContent: string,
  options: GeneratorOptions = {}
): Promise<GenerationResult> {
  const provider = options.provider || getDefaultProvider();
  const store = options.store || getStore();
  const source = options.source || 'llm';

  // Build prompt
  const prompt = SCRIPT_ANALYSIS_PROMPT
    .replace('{path}', scriptPath)
    .replace('{content}', scriptContent);

  try {
    // Get LLM response
    const response = await provider.complete(prompt);

    // Parse JSON from response (handle markdown code blocks)
    const json = extractJson(response);
    const constraintFile: ConstraintFile = JSON.parse(json);

    // Convert to stored constraints
    const storedConstraints: StoredConstraint[] = [];
    for (const c of constraintFile.constraints) {
      const functor = parseFunctor(c.functor, c.args);
      storedConstraints.push({
        command: c.command,
        constraint: functor,
        priority: c.priority,
        message: c.message,
        source
      });
    }

    // Store in database
    const ids = await store.addBatch(storedConstraints);

    return {
      success: true,
      command: scriptPath,
      constraints: storedConstraints,
      ids,
      raw: response
    };
  } catch (error) {
    return {
      success: false,
      command: scriptPath,
      constraints: [],
      ids: [],
      error: error instanceof Error ? error.message : String(error)
    };
  }
}

/**
 * Generate constraints for multiple scripts.
 */
export async function generateConstraintsBatch(
  scripts: Array<{ path: string; content: string }>,
  options: GeneratorOptions = {}
): Promise<GenerationResult[]> {
  const results: GenerationResult[] = [];

  for (const script of scripts) {
    const result = await generateConstraints(script.path, script.content, options);
    results.push(result);
  }

  return results;
}

// ============================================================================
// Edit Review
// ============================================================================

export interface EditReviewResult {
  allowed: boolean;
  reason: string;
  risks: string[];
  suggestions: string[];
  raw?: string;
}

/**
 * Review a proposed file edit using an LLM.
 */
export async function reviewEdit(
  filePath: string,
  originalContent: string,
  proposedContent: string,
  options: GeneratorOptions = {}
): Promise<EditReviewResult> {
  const provider = options.provider || getDefaultProvider();

  const prompt = EDIT_REVIEW_PROMPT
    .replace('{path}', filePath)
    .replace('{original}', originalContent)
    .replace('{proposed}', proposedContent);

  try {
    const response = await provider.complete(prompt);
    const json = extractJson(response);
    const result = JSON.parse(json);

    return {
      allowed: result.allowed ?? false,
      reason: result.reason ?? 'Unknown',
      risks: result.risks ?? [],
      suggestions: result.suggestions ?? [],
      raw: response
    };
  } catch (error) {
    return {
      allowed: false,
      reason: `Failed to review: ${error instanceof Error ? error.message : String(error)}`,
      risks: ['Review failed'],
      suggestions: []
    };
  }
}

// ============================================================================
// Constraint Generation for File Edits
// ============================================================================

/**
 * Edit constraint types for file modifications.
 */
export type EditConstraintFunctor =
  | { type: 'no_delete_lines_containing'; patterns: string[] }
  | { type: 'no_modify_lines_containing'; patterns: string[] }
  | { type: 'no_add_patterns'; patterns: string[] }
  | { type: 'max_lines_changed'; count: number }
  | { type: 'no_change_imports' }
  | { type: 'no_change_exports' }
  | { type: 'no_remove_security_checks' }
  | { type: 'preserve_function'; name: string }
  | { type: 'file_must_parse' }  // Must be valid syntax after edit
  | { type: 'custom_edit'; name: string; params: Record<string, unknown> };

export interface EditConstraint {
  id?: number;
  filePattern: string;  // Glob pattern for files this applies to
  constraint: EditConstraintFunctor;
  priority?: number;
  message?: string;
  source?: string;
}

const EDIT_CONSTRAINT_SYSTEM_PROMPT = `You are a security policy compiler for file edits. Your task is to generate constraints that restrict what edits can be made to files.

Respond with ONLY valid JSON. No explanations.

Available edit constraint functors:
- no_delete_lines_containing(patterns[]): Cannot delete lines matching patterns
- no_modify_lines_containing(patterns[]): Cannot modify lines matching patterns
- no_add_patterns(patterns[]): Cannot add content matching patterns
- max_lines_changed(count): Limit on number of lines that can change
- no_change_imports: Cannot modify import statements
- no_change_exports: Cannot modify export statements
- no_remove_security_checks: Cannot remove validation/auth code
- preserve_function(name): Cannot delete or rename this function
- file_must_parse: File must be valid syntax after edit

Response format:
{
  "version": "1.0",
  "constraints": [
    {
      "filePattern": "*.ts",
      "functor": "functor_name",
      "args": [arg1],
      "priority": 0,
      "message": "Error message"
    }
  ]
}`;

/**
 * Generate edit constraints for a file or file pattern.
 */
export async function generateEditConstraints(
  filePattern: string,
  sampleContent: string,
  options: GeneratorOptions = {}
): Promise<{
  success: boolean;
  constraints: EditConstraint[];
  error?: string;
}> {
  const provider = options.provider || getDefaultProvider();
  const source = options.source || 'llm';

  const prompt = `Analyze this file and generate edit constraints to protect important code.

File pattern: ${filePattern}

Sample content:
\`\`\`
${sampleContent}
\`\`\`

Generate constraints that:
1. Protect security-critical code
2. Prevent removal of validation
3. Preserve important functions
4. Limit scope of changes

Respond with JSON only.`;

  try {
    const response = await provider.complete(prompt, {
      systemPrompt: EDIT_CONSTRAINT_SYSTEM_PROMPT
    });

    const json = extractJson(response);
    const data = JSON.parse(json);

    const constraints: EditConstraint[] = data.constraints.map((c: any) => ({
      filePattern: c.filePattern || filePattern,
      constraint: parseEditFunctor(c.functor, c.args),
      priority: c.priority,
      message: c.message,
      source
    }));

    return { success: true, constraints };
  } catch (error) {
    return {
      success: false,
      constraints: [],
      error: error instanceof Error ? error.message : String(error)
    };
  }
}

/**
 * Parse an edit constraint functor.
 */
function parseEditFunctor(functor: string, args: unknown[] = []): EditConstraintFunctor {
  switch (functor) {
    case 'no_delete_lines_containing':
      return { type: 'no_delete_lines_containing', patterns: args[0] as string[] };
    case 'no_modify_lines_containing':
      return { type: 'no_modify_lines_containing', patterns: args[0] as string[] };
    case 'no_add_patterns':
      return { type: 'no_add_patterns', patterns: args[0] as string[] };
    case 'max_lines_changed':
      return { type: 'max_lines_changed', count: Number(args[0]) };
    case 'no_change_imports':
      return { type: 'no_change_imports' };
    case 'no_change_exports':
      return { type: 'no_change_exports' };
    case 'no_remove_security_checks':
      return { type: 'no_remove_security_checks' };
    case 'preserve_function':
      return { type: 'preserve_function', name: String(args[0]) };
    case 'file_must_parse':
      return { type: 'file_must_parse' };
    default:
      return { type: 'custom_edit', name: functor, params: { args } };
  }
}

// ============================================================================
// Helpers
// ============================================================================

/**
 * Extract JSON from a potentially markdown-wrapped response.
 */
function extractJson(text: string): string {
  // Try to find JSON in markdown code block
  const codeBlockMatch = text.match(/```(?:json)?\s*([\s\S]*?)```/);
  if (codeBlockMatch) {
    return codeBlockMatch[1].trim();
  }

  // Try to find raw JSON object
  const jsonMatch = text.match(/\{[\s\S]*\}/);
  if (jsonMatch) {
    return jsonMatch[0];
  }

  // Return as-is and let JSON.parse fail with a good error
  return text;
}

let defaultProvider: LLMProvider | null = null;

/**
 * Get the default LLM provider.
 */
export function getDefaultProvider(): LLMProvider {
  if (!defaultProvider) {
    // Try Anthropic first, fall back to OpenAI
    if (process.env.ANTHROPIC_API_KEY) {
      defaultProvider = new AnthropicProvider();
    } else if (process.env.OPENAI_API_KEY) {
      defaultProvider = new OpenAIProvider();
    } else {
      throw new Error(
        'No LLM provider configured. Set ANTHROPIC_API_KEY or OPENAI_API_KEY.'
      );
    }
  }
  return defaultProvider;
}

/**
 * Set the default LLM provider.
 */
export function setDefaultProvider(provider: LLMProvider): void {
  defaultProvider = provider;
}
