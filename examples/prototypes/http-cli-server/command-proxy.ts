/**
 * command-proxy.ts - Command Interception and Validation Proxy
 *
 * Each allowed command has its own safety checks and validation logic.
 * Commands not in the registry are blocked or escalated for LLM review.
 *
 * Usage:
 *   import { execute, validateCommand } from './command-proxy';
 *   const result = await execute('ls -la /home', { role: 'user' });
 */

import { spawn, SpawnOptions } from 'child_process';
import * as path from 'path';

// ============================================================================
// Types
// ============================================================================

export enum Risk {
  SAFE = 'safe',           // No review needed
  MODERATE = 'moderate',   // Logged, basic validation
  HIGH = 'high',           // Requires confirmation or LLM review
  BLOCKED = 'blocked'      // Never allowed
}

export interface ExecutionContext {
  role?: string;
  ip?: string;
  confirmed?: boolean;
  cwd?: string;
  env?: Record<string, string>;
  timeout?: number;
}

export interface ValidationResult {
  ok: boolean;
  reason?: string;
  warning?: string;
  suggestion?: string;
  risk?: Risk;
  blocked?: boolean;
  requiresConfirmation?: boolean;
  requiresPty?: boolean;
  description?: string;
}

export interface ExecutionResult {
  success: boolean;
  code?: number;
  stdout?: string;
  stderr?: string;
  error?: string;
  warning?: string;
  blocked?: boolean;
  suggestion?: string;
  requiresConfirmation?: boolean;
  description?: string;
  risk?: Risk;
}

export interface CommandDefinition {
  risk: Risk;
  description: string;
  validate: (args: string[], ctx: ExecutionContext) => ValidationResult;
  transform: (args: string[]) => string[];
  requiresConfirmation?: boolean;
  requiresPty?: boolean;
  isBuiltin?: boolean;
}

export interface ParsedCommand {
  cmd: string;
  args: string[];
  raw: string;
}

export interface CommandInfo {
  name: string;
  risk: Risk;
  description: string;
  requiresConfirmation: boolean;
  requiresPty: boolean;
}

// ============================================================================
// Configuration
// ============================================================================

export const SANDBOX_ROOT = process.env.SANDBOX_ROOT || path.join(process.env.HOME || '/tmp', 'sandbox');

// ============================================================================
// Helper Functions
// ============================================================================

const isSensitivePath = (p: string): boolean => {
  const sensitive = ['/etc/shadow', '/etc/passwd', '/root', '.ssh/id_', '.env', 'credentials', 'secrets'];
  return sensitive.some(s => p.includes(s));
};

const isOutsideSandbox = (p: string): boolean => {
  return p.startsWith('/') && !p.startsWith(SANDBOX_ROOT);
};

const containsDestructivePattern = (args: string[]): boolean => {
  const hasRecursive = args.includes('-rf') || args.includes('-r') || args.includes('--recursive');
  const hasDangerousTarget = args.some(a =>
    a === '/' || a === '~' || a === '$HOME' || a === process.env.HOME
  );
  return hasRecursive && hasDangerousTarget;
};

// ============================================================================
// Command Registry
// ============================================================================

const commandRegistry: Record<string, CommandDefinition> = {

  // --- File listing (safe) ---
  'ls': {
    risk: Risk.SAFE,
    description: 'List directory contents',
    validate: (args: string[], ctx: ExecutionContext): ValidationResult => {
      if (args.some(a => isSensitivePath(a))) {
        return { ok: false, reason: 'Access to sensitive paths blocked' };
      }
      return { ok: true };
    },
    transform: (args: string[]) => ['ls', '--color=auto', ...args]
  },

  'cat': {
    risk: Risk.MODERATE,
    description: 'Display file contents',
    validate: (args: string[], ctx: ExecutionContext): ValidationResult => {
      if (args.some(a => isSensitivePath(a))) {
        return { ok: false, reason: 'Reading sensitive files blocked' };
      }
      return { ok: true };
    },
    transform: (args: string[]) => ['cat', ...args]
  },

  'head': {
    risk: Risk.SAFE,
    description: 'Display first lines of file',
    validate: (): ValidationResult => ({ ok: true }),
    transform: (args: string[]) => ['head', ...args]
  },

  'tail': {
    risk: Risk.SAFE,
    description: 'Display last lines of file',
    validate: (): ValidationResult => ({ ok: true }),
    transform: (args: string[]) => ['tail', ...args]
  },

  // --- Navigation (safe) ---
  'pwd': {
    risk: Risk.SAFE,
    description: 'Print working directory',
    validate: (): ValidationResult => ({ ok: true }),
    transform: () => ['pwd']
  },

  'cd': {
    risk: Risk.SAFE,
    description: 'Change directory',
    isBuiltin: true,
    validate: (): ValidationResult => ({ ok: true }),
    transform: (args: string[]) => ['cd', ...args]
  },

  // --- File operations (moderate to high) ---
  'cp': {
    risk: Risk.MODERATE,
    description: 'Copy files',
    validate: (args: string[], ctx: ExecutionContext): ValidationResult => {
      if (args.some(a => isOutsideSandbox(a))) {
        return { ok: false, reason: 'Copy must stay within sandbox directory' };
      }
      return { ok: true };
    },
    transform: (args: string[]) => ['cp', ...args]
  },

  'mv': {
    risk: Risk.MODERATE,
    description: 'Move/rename files',
    validate: (args: string[], ctx: ExecutionContext): ValidationResult => {
      if (args.some(a => isOutsideSandbox(a))) {
        return { ok: false, reason: 'Move must stay within sandbox directory' };
      }
      return { ok: true };
    },
    transform: (args: string[]) => ['mv', ...args]
  },

  'mkdir': {
    risk: Risk.SAFE,
    description: 'Create directory',
    validate: (args: string[], ctx: ExecutionContext): ValidationResult => {
      if (args.some(a => isOutsideSandbox(a))) {
        return { ok: false, reason: 'mkdir must stay within sandbox' };
      }
      return { ok: true };
    },
    transform: (args: string[]) => ['mkdir', '-p', ...args]
  },

  'rm': {
    risk: Risk.HIGH,
    description: 'Remove files',
    requiresConfirmation: true,
    validate: (args: string[], ctx: ExecutionContext): ValidationResult => {
      if (containsDestructivePattern(args)) {
        return { ok: false, reason: 'Recursive delete of system paths blocked' };
      }
      if (args.some(a => isOutsideSandbox(a))) {
        return { ok: false, reason: 'rm must stay within sandbox directory' };
      }
      return { ok: true, warning: 'This will permanently delete files' };
    },
    transform: (args: string[]) => ['rm', ...args]
  },

  'touch': {
    risk: Risk.SAFE,
    description: 'Create empty file or update timestamp',
    validate: (): ValidationResult => ({ ok: true }),
    transform: (args: string[]) => ['touch', ...args]
  },

  // --- Search (safe) ---
  'grep': {
    risk: Risk.SAFE,
    description: 'Search file contents',
    validate: (): ValidationResult => ({ ok: true }),
    transform: (args: string[]) => ['grep', '--color=auto', ...args]
  },

  'find': {
    risk: Risk.SAFE,
    description: 'Find files',
    validate: (args: string[], ctx: ExecutionContext): ValidationResult => {
      const execIdx = args.indexOf('-exec');
      if (execIdx !== -1) {
        const execCmd = args.slice(execIdx + 1).join(' ');
        if (/rm|chmod|chown/.test(execCmd)) {
          return { ok: false, reason: 'find -exec with destructive commands blocked' };
        }
      }
      return { ok: true };
    },
    transform: (args: string[]) => ['find', ...args]
  },

  // --- Git (moderate) ---
  'git': {
    risk: Risk.MODERATE,
    description: 'Git version control',
    validate: (args: string[], ctx: ExecutionContext): ValidationResult => {
      // Block force push
      if (args.includes('push') && (args.includes('--force') || args.includes('-f'))) {
        return { ok: false, reason: 'Force push blocked - use with caution manually' };
      }
      // Block credential commands
      if (args.includes('config') && args.some(a => /credential|password/.test(a))) {
        return { ok: false, reason: 'Git credential manipulation blocked' };
      }
      // Warn on push to main/master
      if (args.includes('push') && (args.includes('main') || args.includes('master'))) {
        return { ok: true, warning: 'Pushing to main/master branch' };
      }
      return { ok: true };
    },
    transform: (args: string[]) => ['git', ...args]
  },

  // --- Node/npm (moderate) ---
  'node': {
    risk: Risk.MODERATE,
    description: 'Run Node.js',
    validate: (args: string[], ctx: ExecutionContext): ValidationResult => {
      if (args.includes('-e') || args.includes('--eval')) {
        return { ok: false, reason: 'Node eval blocked - use script files instead' };
      }
      return { ok: true };
    },
    transform: (args: string[]) => ['node', ...args]
  },

  'npm': {
    risk: Risk.MODERATE,
    description: 'Node package manager',
    validate: (args: string[], ctx: ExecutionContext): ValidationResult => {
      const blocked = ['publish', 'unpublish', 'adduser', 'login', 'token'];
      const cmd = args[0];

      if (blocked.includes(cmd)) {
        return { ok: false, reason: `npm ${cmd} blocked for security` };
      }
      if (args.includes('-g') || args.includes('--global')) {
        return { ok: false, reason: 'Global npm operations blocked' };
      }
      return { ok: true };
    },
    transform: (args: string[]) => ['npm', ...args]
  },

  // --- Python (moderate) ---
  'python': {
    risk: Risk.MODERATE,
    description: 'Run Python',
    validate: (args: string[], ctx: ExecutionContext): ValidationResult => {
      if (args.includes('-c')) {
        return { ok: false, reason: 'Python -c blocked - use script files instead' };
      }
      return { ok: true };
    },
    transform: (args: string[]) => ['python3', ...args]
  },

  'python3': {
    risk: Risk.MODERATE,
    description: 'Run Python 3',
    validate: (args: string[], ctx: ExecutionContext): ValidationResult => {
      if (args.includes('-c')) {
        return { ok: false, reason: 'Python -c blocked - use script files instead' };
      }
      return { ok: true };
    },
    transform: (args: string[]) => ['python3', ...args]
  },

  'pip': {
    risk: Risk.MODERATE,
    description: 'Python package manager',
    validate: (args: string[], ctx: ExecutionContext): ValidationResult => {
      if (args[0] === 'uninstall') {
        return { ok: false, reason: 'pip uninstall requires confirmation' };
      }
      return { ok: true };
    },
    transform: (args: string[]) => ['pip', ...args]
  },

  // --- Text processing (safe) ---
  'echo': {
    risk: Risk.SAFE,
    description: 'Print text',
    validate: (): ValidationResult => ({ ok: true }),
    transform: (args: string[]) => ['echo', ...args]
  },

  'wc': {
    risk: Risk.SAFE,
    description: 'Word count',
    validate: (): ValidationResult => ({ ok: true }),
    transform: (args: string[]) => ['wc', ...args]
  },

  'sort': {
    risk: Risk.SAFE,
    description: 'Sort lines',
    validate: (): ValidationResult => ({ ok: true }),
    transform: (args: string[]) => ['sort', ...args]
  },

  'uniq': {
    risk: Risk.SAFE,
    description: 'Filter unique lines',
    validate: (): ValidationResult => ({ ok: true }),
    transform: (args: string[]) => ['uniq', ...args]
  },

  // --- Editors (moderate) ---
  'nano': {
    risk: Risk.MODERATE,
    description: 'Text editor',
    requiresPty: true,
    validate: (): ValidationResult => ({ ok: true }),
    transform: (args: string[]) => ['nano', ...args]
  },

  'vim': {
    risk: Risk.MODERATE,
    description: 'Text editor',
    requiresPty: true,
    validate: (): ValidationResult => ({ ok: true }),
    transform: (args: string[]) => ['vim', ...args]
  },

  // --- System info (safe) ---
  'date': {
    risk: Risk.SAFE,
    description: 'Show date/time',
    validate: (): ValidationResult => ({ ok: true }),
    transform: (args: string[]) => ['date', ...args]
  },

  'whoami': {
    risk: Risk.SAFE,
    description: 'Show current user',
    validate: (): ValidationResult => ({ ok: true }),
    transform: () => ['whoami']
  },

  'uname': {
    risk: Risk.SAFE,
    description: 'System information',
    validate: (): ValidationResult => ({ ok: true }),
    transform: (args: string[]) => ['uname', ...args]
  },

  'df': {
    risk: Risk.SAFE,
    description: 'Disk usage',
    validate: (): ValidationResult => ({ ok: true }),
    transform: (args: string[]) => ['df', '-h', ...args]
  },

  'du': {
    risk: Risk.SAFE,
    description: 'Directory size',
    validate: (): ValidationResult => ({ ok: true }),
    transform: (args: string[]) => ['du', '-h', ...args]
  },

  // --- Network (moderate) ---
  'curl': {
    risk: Risk.MODERATE,
    description: 'HTTP client',
    validate: (args: string[], ctx: ExecutionContext): ValidationResult => {
      const outputIdx = args.findIndex(a => a === '-o' || a === '--output');
      if (outputIdx !== -1 && args[outputIdx + 1]) {
        const outputPath = args[outputIdx + 1];
        if (isOutsideSandbox(outputPath)) {
          return { ok: false, reason: 'curl output must be within sandbox' };
        }
      }
      return { ok: true };
    },
    transform: (args: string[]) => ['curl', ...args]
  },

  'wget': {
    risk: Risk.MODERATE,
    description: 'Download files',
    validate: (): ValidationResult => ({ ok: true }),
    transform: (args: string[]) => ['wget', ...args]
  },

  // --- BLOCKED commands ---
  'sudo': {
    risk: Risk.BLOCKED,
    description: 'Superuser do',
    validate: (): ValidationResult => ({ ok: false, reason: 'sudo is not available in sandbox' }),
    transform: (args: string[]) => ['sudo', ...args]
  },

  'su': {
    risk: Risk.BLOCKED,
    description: 'Switch user',
    validate: (): ValidationResult => ({ ok: false, reason: 'su is not available in sandbox' }),
    transform: (args: string[]) => ['su', ...args]
  },

  'chmod': {
    risk: Risk.HIGH,
    description: 'Change permissions',
    requiresConfirmation: true,
    validate: (args: string[], ctx: ExecutionContext): ValidationResult => {
      // Block setuid/setgid
      if (args.some(a => /[sg]|[42]\d{2,}/.test(a))) {
        return { ok: false, reason: 'Setting setuid/setgid blocked' };
      }
      return { ok: true };
    },
    transform: (args: string[]) => ['chmod', ...args]
  },

  'chown': {
    risk: Risk.BLOCKED,
    description: 'Change ownership',
    validate: (): ValidationResult => ({ ok: false, reason: 'chown not available in sandbox' }),
    transform: (args: string[]) => ['chown', ...args]
  }
};

// ============================================================================
// Proxy Functions
// ============================================================================

/**
 * Parse a command string into command and arguments
 */
export function parseCommand(cmdString: string): ParsedCommand {
  const parts = cmdString.trim().split(/\s+/);
  const cmd = parts[0];
  const args = parts.slice(1);
  return { cmd, args, raw: cmdString };
}

/**
 * Check if a command is in the registry
 */
export function isKnownCommand(cmd: string): boolean {
  return cmd in commandRegistry;
}

/**
 * Get command definition from registry
 */
export function getCommand(cmd: string): CommandDefinition | null {
  return commandRegistry[cmd] || null;
}

/**
 * Validate a command with its specific validator
 */
export function validateCommand(cmd: string, args: string[], ctx: ExecutionContext = {}): ValidationResult {
  const def = getCommand(cmd);

  if (!def) {
    return {
      ok: false,
      reason: `Unknown command: ${cmd}`,
      suggestion: 'Command not in allow-list. Request may need review.'
    };
  }

  if (def.risk === Risk.BLOCKED) {
    const result = def.validate(args, ctx);
    return {
      ok: false,
      reason: result.reason || `${cmd} is blocked`,
      blocked: true
    };
  }

  const result = def.validate(args, ctx);
  return {
    ...result,
    risk: def.risk,
    requiresConfirmation: def.requiresConfirmation || false,
    requiresPty: def.requiresPty || false,
    description: def.description
  };
}

/**
 * Execute a validated command
 */
export function executeCommand(cmd: string, args: string[], options: ExecutionContext = {}): Promise<ExecutionResult> {
  return new Promise((resolve, reject) => {
    const def = getCommand(cmd);
    if (!def) {
      reject(new Error(`Unknown command: ${cmd}`));
      return;
    }

    const transformedArgs = def.transform(args);
    const executable = transformedArgs[0];
    const execArgs = transformedArgs.slice(1);

    const spawnOptions: SpawnOptions = {
      cwd: options.cwd || SANDBOX_ROOT,
      env: { ...process.env, ...options.env },
      timeout: options.timeout || 30000
    };

    const proc = spawn(executable, execArgs, spawnOptions);

    let stdout = '';
    let stderr = '';

    proc.stdout?.on('data', (data: Buffer) => {
      stdout += data.toString();
    });

    proc.stderr?.on('data', (data: Buffer) => {
      stderr += data.toString();
    });

    proc.on('close', (code: number | null) => {
      resolve({
        success: code === 0,
        code: code ?? undefined,
        stdout,
        stderr
      });
    });

    proc.on('error', (err: Error) => {
      reject(err);
    });
  });
}

/**
 * Main proxy entry point
 */
export async function execute(cmdString: string, ctx: ExecutionContext = {}): Promise<ExecutionResult> {
  const { cmd, args } = parseCommand(cmdString);

  // Log the attempt
  const logEntry = {
    timestamp: new Date().toISOString(),
    command: cmd,
    args,
    role: ctx.role || 'unknown',
    ip: ctx.ip || 'unknown'
  };
  console.log('[command-proxy]', JSON.stringify(logEntry));

  // Validate
  const validation = validateCommand(cmd, args, ctx);

  if (!validation.ok) {
    return {
      success: false,
      error: validation.reason,
      blocked: validation.blocked || false,
      suggestion: validation.suggestion
    };
  }

  // Check if confirmation required
  if (validation.requiresConfirmation && !ctx.confirmed) {
    return {
      success: false,
      requiresConfirmation: true,
      warning: validation.warning,
      description: validation.description,
      risk: validation.risk
    };
  }

  // Execute
  try {
    const result = await executeCommand(cmd, args, ctx);
    return {
      ...result,
      warning: validation.warning
    };
  } catch (err) {
    return {
      success: false,
      error: (err as Error).message
    };
  }
}

/**
 * Get list of available commands with their risk levels
 */
export function listCommands(): CommandInfo[] {
  return Object.entries(commandRegistry).map(([name, def]) => ({
    name,
    risk: def.risk,
    description: def.description,
    requiresConfirmation: def.requiresConfirmation || false,
    requiresPty: def.requiresPty || false
  }));
}

/**
 * Add a custom command to the registry
 */
export function registerCommand(name: string, definition: Partial<CommandDefinition>): void {
  commandRegistry[name] = {
    risk: definition.risk || Risk.HIGH,
    description: definition.description || `Custom command: ${name}`,
    validate: definition.validate || (() => ({ ok: true })),
    transform: definition.transform || ((args: string[]) => [name, ...args]),
    requiresConfirmation: definition.requiresConfirmation || false,
    requiresPty: definition.requiresPty || false
  };
}

// ============================================================================
// Default Export
// ============================================================================

export default {
  execute,
  validateCommand,
  parseCommand,
  isKnownCommand,
  getCommand,
  listCommands,
  registerCommand,
  Risk,
  SANDBOX_ROOT
};
