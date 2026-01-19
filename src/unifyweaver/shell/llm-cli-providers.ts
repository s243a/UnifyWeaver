/**
 * CLI-based LLM Providers
 *
 * Provides LLM integration via command-line tools:
 * - Claude Code CLI (`claude`)
 * - Gemini CLI (`gemini`)
 * - Ollama (`ollama`)
 *
 * These complement the API-based providers for scenarios where:
 * - CLI authentication is already configured
 * - Local models are preferred
 * - API keys should not be in code
 *
 * @module unifyweaver/shell/llm-cli-providers
 */

import { spawn } from 'child_process';
import { LLMProvider, LLMOptions } from './llm-constraint-generator';

// ============================================================================
// CLI Execution Helper
// ============================================================================

interface CLIResult {
  stdout: string;
  stderr: string;
  code: number | null;
}

/**
 * Execute a CLI command and capture output.
 */
async function execCLI(
  command: string,
  args: string[],
  options: {
    timeout?: number;
    cwd?: string;
    input?: string;
  } = {}
): Promise<CLIResult> {
  const { timeout = 120000, cwd = process.cwd(), input } = options;

  return new Promise((resolve, reject) => {
    const proc = spawn(command, args, {
      cwd,
      stdio: ['pipe', 'pipe', 'pipe']
    });

    let stdout = '';
    let stderr = '';

    proc.stdout.on('data', (data) => {
      stdout += data.toString();
    });

    proc.stderr.on('data', (data) => {
      stderr += data.toString();
    });

    // Handle timeout
    const timer = setTimeout(() => {
      proc.kill('SIGTERM');
      reject(new Error(`CLI timeout after ${timeout}ms`));
    }, timeout);

    proc.on('close', (code) => {
      clearTimeout(timer);
      resolve({ stdout, stderr, code });
    });

    proc.on('error', (err) => {
      clearTimeout(timer);
      reject(err);
    });

    // Send input if provided
    if (input) {
      proc.stdin.write(input);
      proc.stdin.end();
    } else {
      proc.stdin.end();
    }
  });
}

// ============================================================================
// Claude Code CLI Provider
// ============================================================================

/**
 * Claude Code CLI provider.
 *
 * Uses the `claude` command-line tool with print mode.
 *
 * CLI format: claude -p --model <model> <prompt>
 */
export class ClaudeCLIProvider implements LLMProvider {
  readonly name = 'claude-cli';
  private model: string;
  private timeout: number;

  constructor(options: {
    model?: string;  // sonnet, opus, haiku
    timeout?: number;
  } = {}) {
    this.model = options.model || 'sonnet';
    this.timeout = options.timeout || 120000;
  }

  async complete(prompt: string, options: LLMOptions = {}): Promise<string> {
    const model = options.maxTokens ? this.model : this.model;  // Could map maxTokens to model

    // Build system prompt into the main prompt if provided
    const fullPrompt = options.systemPrompt
      ? `${options.systemPrompt}\n\n${prompt}`
      : prompt;

    try {
      const result = await execCLI('claude', ['-p', '--model', this.model, fullPrompt], {
        timeout: this.timeout
      });

      if (result.code !== 0) {
        throw new Error(`Claude CLI failed: ${result.stderr.slice(0, 500)}`);
      }

      return result.stdout.trim();
    } catch (err) {
      if (err instanceof Error && err.message.includes('ENOENT')) {
        throw new Error('Claude CLI not found. Install with: npm install -g @anthropic-ai/claude-code');
      }
      throw err;
    }
  }
}

// ============================================================================
// Gemini CLI Provider
// ============================================================================

/**
 * Gemini CLI provider.
 *
 * Uses the `gemini` command-line tool.
 *
 * CLI format: gemini -p <prompt> -m <model> --output-format text
 */
export class GeminiCLIProvider implements LLMProvider {
  readonly name = 'gemini-cli';
  private model: string;
  private timeout: number;

  constructor(options: {
    model?: string;  // gemini-2.0-flash, gemini-1.5-pro, etc.
    timeout?: number;
  } = {}) {
    this.model = options.model || 'gemini-2.0-flash';
    this.timeout = options.timeout || 120000;
  }

  async complete(prompt: string, options: LLMOptions = {}): Promise<string> {
    // Build system prompt into the main prompt if provided
    const fullPrompt = options.systemPrompt
      ? `${options.systemPrompt}\n\n${prompt}`
      : prompt;

    try {
      const result = await execCLI('gemini', [
        '-p', fullPrompt,
        '-m', this.model,
        '--output-format', 'text'
      ], {
        timeout: this.timeout
      });

      if (result.code !== 0) {
        throw new Error(`Gemini CLI failed: ${result.stderr.slice(0, 500)}`);
      }

      return result.stdout.trim();
    } catch (err) {
      if (err instanceof Error && err.message.includes('ENOENT')) {
        throw new Error('Gemini CLI not found. Install from: https://github.com/google-gemini/gemini-cli');
      }
      throw err;
    }
  }
}

// ============================================================================
// Ollama CLI Provider
// ============================================================================

/**
 * Ollama CLI provider.
 *
 * Uses the `ollama` command-line tool for local models.
 *
 * CLI format: ollama run <model> (with prompt via stdin)
 *
 * Popular models: llama3, codellama, mistral, mixtral, phi3
 */
export class OllamaCLIProvider implements LLMProvider {
  readonly name = 'ollama';
  private model: string;
  private timeout: number;
  private host?: string;

  constructor(options: {
    model?: string;  // llama3, codellama, mistral, etc.
    timeout?: number;
    host?: string;  // OLLAMA_HOST override
  } = {}) {
    this.model = options.model || 'llama3';
    this.timeout = options.timeout || 300000;  // Local models can be slower
    this.host = options.host;
  }

  async complete(prompt: string, options: LLMOptions = {}): Promise<string> {
    // Build system prompt into the main prompt if provided
    const fullPrompt = options.systemPrompt
      ? `${options.systemPrompt}\n\n${prompt}`
      : prompt;

    try {
      // Set OLLAMA_HOST if specified
      const env = this.host
        ? { ...process.env, OLLAMA_HOST: this.host }
        : process.env;

      const result = await execCLI('ollama', ['run', this.model], {
        timeout: this.timeout,
        input: fullPrompt
      });

      if (result.code !== 0) {
        throw new Error(`Ollama failed: ${result.stderr.slice(0, 500)}`);
      }

      return result.stdout.trim();
    } catch (err) {
      if (err instanceof Error && err.message.includes('ENOENT')) {
        throw new Error('Ollama not found. Install from: https://ollama.ai');
      }
      throw err;
    }
  }
}

// ============================================================================
// Generic CLI Provider
// ============================================================================

/**
 * Generic CLI provider for custom LLM tools.
 *
 * Configure the command template with {prompt} placeholder.
 */
export class GenericCLIProvider implements LLMProvider {
  readonly name: string;
  private command: string;
  private args: string[];
  private timeout: number;
  private useStdin: boolean;

  constructor(options: {
    name: string;
    command: string;
    args: string[];  // Use {prompt} as placeholder
    timeout?: number;
    useStdin?: boolean;  // Send prompt via stdin instead of args
  }) {
    this.name = options.name;
    this.command = options.command;
    this.args = options.args;
    this.timeout = options.timeout || 120000;
    this.useStdin = options.useStdin || false;
  }

  async complete(prompt: string, options: LLMOptions = {}): Promise<string> {
    const fullPrompt = options.systemPrompt
      ? `${options.systemPrompt}\n\n${prompt}`
      : prompt;

    // Replace {prompt} placeholder in args
    const args = this.args.map(a => a.replace('{prompt}', fullPrompt));

    const result = await execCLI(this.command, args, {
      timeout: this.timeout,
      input: this.useStdin ? fullPrompt : undefined
    });

    if (result.code !== 0) {
      throw new Error(`${this.name} CLI failed: ${result.stderr.slice(0, 500)}`);
    }

    return result.stdout.trim();
  }
}

// ============================================================================
// Provider Factory
// ============================================================================

export type CLIProviderType = 'claude-cli' | 'gemini-cli' | 'ollama' | 'generic';

export interface CLIProviderOptions {
  type: CLIProviderType;
  model?: string;
  timeout?: number;
  // For generic provider
  command?: string;
  args?: string[];
  useStdin?: boolean;
}

/**
 * Create a CLI-based LLM provider.
 */
export function createCLIProvider(options: CLIProviderOptions): LLMProvider {
  switch (options.type) {
    case 'claude-cli':
      return new ClaudeCLIProvider({
        model: options.model,
        timeout: options.timeout
      });

    case 'gemini-cli':
      return new GeminiCLIProvider({
        model: options.model,
        timeout: options.timeout
      });

    case 'ollama':
      return new OllamaCLIProvider({
        model: options.model,
        timeout: options.timeout
      });

    case 'generic':
      if (!options.command || !options.args) {
        throw new Error('Generic provider requires command and args');
      }
      return new GenericCLIProvider({
        name: 'generic',
        command: options.command,
        args: options.args,
        timeout: options.timeout,
        useStdin: options.useStdin
      });

    default:
      throw new Error(`Unknown CLI provider type: ${options.type}`);
  }
}

// ============================================================================
// Availability Checks
// ============================================================================

/**
 * Check if a CLI tool is available.
 */
export async function isCLIAvailable(command: string): Promise<boolean> {
  try {
    const result = await execCLI('which', [command], { timeout: 5000 });
    return result.code === 0;
  } catch {
    return false;
  }
}

/**
 * Check which CLI providers are available.
 * Returns only auto-detectable providers (not 'generic').
 */
export async function getAvailableCLIProviders(): Promise<Array<'claude-cli' | 'gemini-cli' | 'ollama'>> {
  const checks = await Promise.all([
    isCLIAvailable('claude').then(ok => ok ? 'claude-cli' as const : null),
    isCLIAvailable('gemini').then(ok => ok ? 'gemini-cli' as const : null),
    isCLIAvailable('ollama').then(ok => ok ? 'ollama' as const : null)
  ]);

  return checks.filter((c): c is 'claude-cli' | 'gemini-cli' | 'ollama' => c !== null);
}

/**
 * Get the best available provider (prefers CLI, falls back to API).
 */
export async function getBestProvider(): Promise<LLMProvider> {
  const available = await getAvailableCLIProviders();

  if (available.includes('claude-cli')) {
    return new ClaudeCLIProvider();
  }

  if (available.includes('gemini-cli')) {
    return new GeminiCLIProvider();
  }

  if (available.includes('ollama')) {
    return new OllamaCLIProvider();
  }

  // Fall back to API providers
  const { getDefaultProvider } = require('./llm-constraint-generator');
  return getDefaultProvider();
}
