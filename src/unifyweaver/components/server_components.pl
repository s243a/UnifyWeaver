% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% server_components.pl - Server-side components for HTTP servers
%
% Reusable building blocks for generating HTTP server code:
%   - command_validator: Command validation with risk assessment
%   - file_browser: Directory browsing within sandbox
%   - feedback_store: JSON feedback logging
%   - websocket_shell: PTY-like shell over WebSocket
%
% Usage:
%   use_module('src/unifyweaver/components/server_components').
%   command_validator_spec(Spec),
%   generate_command_validator(Spec, typescript, Code).

:- module(server_components, [
    % Command Validator
    command_validator_spec/2,
    generate_command_validator/3,
    default_command_registry/1,

    % File Browser
    file_browser_spec/2,
    generate_file_browser/3,

    % Feedback Store
    feedback_store_spec/2,
    generate_feedback_store/3,

    % WebSocket Shell
    websocket_shell_spec/2,
    generate_websocket_shell/3,

    % Endpoint Handlers
    endpoint_handler_spec/3,
    generate_endpoint_handler/4,

    % Utility
    generate_server_component/4
]).

:- use_module(library(lists)).

% ============================================================================
% Command Validator Component
% ============================================================================

%! command_validator_spec(+Options, -Spec) is det
%  Create a command validator specification.
%  Options:
%    sandbox_root(Path) - Root directory for sandbox
%    commands(List) - List of command definitions
%    default_timeout(Ms) - Default execution timeout
command_validator_spec(Options, Spec) :-
    get_option(sandbox_root, Options, SandboxRoot, '$HOME/sandbox'),
    get_option(commands, Options, Commands, default),
    get_option(default_timeout, Options, Timeout, 30000),
    (Commands == default -> default_command_registry(CmdList) ; CmdList = Commands),
    Spec = command_validator_spec([
        sandbox_root(SandboxRoot),
        commands(CmdList),
        default_timeout(Timeout)
    ]).

%! default_command_registry(-Commands) is det
%  Returns default list of commands with risk levels.
default_command_registry([
    % File listing (safe)
    cmd(ls, safe, 'List directory contents', []),
    cmd(cat, moderate, 'Display file contents', [sensitive_paths]),
    cmd(head, safe, 'Display first lines', []),
    cmd(tail, safe, 'Display last lines', []),

    % Navigation (safe)
    cmd(pwd, safe, 'Print working directory', []),
    cmd(cd, safe, 'Change directory', [builtin]),

    % Search (safe)
    cmd(grep, safe, 'Search file contents', []),
    cmd(find, safe, 'Find files', [no_destructive_exec]),

    % File operations (moderate to high)
    cmd(cp, moderate, 'Copy files', [sandbox_only]),
    cmd(mv, moderate, 'Move/rename files', [sandbox_only]),
    cmd(mkdir, safe, 'Create directory', [sandbox_only]),
    cmd(rm, high, 'Remove files', [sandbox_only, requires_confirmation]),
    cmd(touch, safe, 'Create empty file', []),

    % Text processing (safe)
    cmd(echo, safe, 'Print text', []),
    cmd(wc, safe, 'Word count', []),
    cmd(sort, safe, 'Sort lines', []),
    cmd(uniq, safe, 'Filter unique lines', []),

    % System info (safe)
    cmd(date, safe, 'Show date/time', []),
    cmd(whoami, safe, 'Show current user', []),
    cmd(uname, safe, 'System information', []),
    cmd(df, safe, 'Disk usage', []),
    cmd(du, safe, 'Directory size', []),

    % Git (moderate)
    cmd(git, moderate, 'Git version control', [no_force_push, no_credentials]),

    % Node/npm (moderate)
    cmd(node, moderate, 'Run Node.js', [no_eval]),
    cmd(npm, moderate, 'Node package manager', [no_global, no_publish]),

    % Python (moderate)
    cmd(python, moderate, 'Run Python', [no_eval]),
    cmd(python3, moderate, 'Run Python 3', [no_eval]),
    cmd(pip, moderate, 'Python package manager', []),

    % Network (moderate)
    cmd(curl, moderate, 'HTTP client', [sandbox_output]),
    cmd(wget, moderate, 'Download files', []),

    % Blocked
    cmd(sudo, blocked, 'Superuser do', []),
    cmd(su, blocked, 'Switch user', []),
    cmd(chown, blocked, 'Change ownership', []),
    cmd(chmod, high, 'Change permissions', [requires_confirmation])
]).

%! generate_command_validator(+Spec, +Target, -Code) is det
%  Generate command validator code for the target language.
generate_command_validator(command_validator_spec(Options), typescript, Code) :-
    get_option(sandbox_root, Options, SandboxRoot, '$HOME/sandbox'),
    get_option(commands, Options, Commands, []),
    get_option(default_timeout, Options, Timeout, 30000),
    generate_ts_command_validator(SandboxRoot, Commands, Timeout, Code).

% Generate TypeScript command validator
generate_ts_command_validator(SandboxRoot, Commands, Timeout, Code) :-
    generate_risk_enum(RiskEnum),
    generate_types(Types),
    generate_helpers(SandboxRoot, Helpers),
    generate_command_registry(Commands, Registry),
    generate_validator_functions(Timeout, ValidatorFns),
    format(atom(Code), '~w~n~n~w~n~n~w~n~n~w~n~n~w', [
        RiskEnum, Types, Helpers, Registry, ValidatorFns
    ]).

generate_risk_enum(Code) :-
    Code = '// Risk levels for commands
export enum Risk {
  SAFE = \'safe\',
  MODERATE = \'moderate\',
  HIGH = \'high\',
  BLOCKED = \'blocked\'
}'.

generate_types(Code) :-
    Code = '// Types
export interface ExecutionContext {
  role?: string;
  cwd?: string;
  confirmed?: boolean;
  timeout?: number;
}

export interface ValidationResult {
  ok: boolean;
  reason?: string;
  warning?: string;
  risk?: Risk;
  requiresConfirmation?: boolean;
}

export interface ExecutionResult {
  success: boolean;
  code?: number;
  stdout?: string;
  stderr?: string;
  error?: string;
  warning?: string;
}

export interface CommandDefinition {
  risk: Risk;
  description: string;
  validate: (args: string[], ctx: ExecutionContext) => ValidationResult;
  flags: string[];
}'.

generate_helpers(SandboxRoot, Code) :-
    format(atom(Code), '// Configuration
export const SANDBOX_ROOT = process.env.SANDBOX_ROOT || \'~w\';

// Helper functions
const isSensitivePath = (p: string): boolean => {
  const sensitive = [\'/etc/shadow\', \'/etc/passwd\', \'/root\', \'.ssh/id_\', \'.env\', \'credentials\'];
  return sensitive.some(s => p.includes(s));
};

const isOutsideSandbox = (p: string): boolean => {
  return p.startsWith(\'/\') && !p.startsWith(SANDBOX_ROOT);
};

const containsDestructivePattern = (args: string[]): boolean => {
  const hasRecursive = args.includes(\'-rf\') || args.includes(\'-r\');
  const hasDangerousTarget = args.some(a => a === \'/\' || a === \'~\');
  return hasRecursive && hasDangerousTarget;
};', [SandboxRoot]).

generate_command_registry(Commands, Code) :-
    maplist(generate_command_entry, Commands, Entries),
    atomic_list_concat(Entries, ',\n\n', EntriesStr),
    format(atom(Code), '// Command Registry
const commandRegistry: Record<string, CommandDefinition> = {
~w
};', [EntriesStr]).

generate_command_entry(cmd(Name, Risk, Desc, Flags), Entry) :-
    risk_to_ts(Risk, TsRisk),
    generate_validator_for_flags(Name, Flags, ValidatorCode),
    flags_to_ts_array(Flags, TsFlags),
    format(atom(Entry), '  \'~w\': {
    risk: Risk.~w,
    description: \'~w\',
    validate: ~w,
    flags: ~w
  }', [Name, TsRisk, Desc, ValidatorCode, TsFlags]).

risk_to_ts(safe, 'SAFE').
risk_to_ts(moderate, 'MODERATE').
risk_to_ts(high, 'HIGH').
risk_to_ts(blocked, 'BLOCKED').

flags_to_ts_array(Flags, TsArray) :-
    maplist(atom_string, Flags, FlagStrs),
    maplist(quote_string, FlagStrs, QuotedStrs),
    atomic_list_concat(QuotedStrs, ', ', Inner),
    format(atom(TsArray), '[~w]', [Inner]).

quote_string(S, Q) :- format(atom(Q), '\'~w\'', [S]).

generate_validator_for_flags(_, Flags, Code) :-
    member(sensitive_paths, Flags), !,
    Code = '(args) => {
      if (args.some(a => isSensitivePath(a))) {
        return { ok: false, reason: \'Access to sensitive paths blocked\' };
      }
      return { ok: true };
    }'.
generate_validator_for_flags(_, Flags, Code) :-
    member(sandbox_only, Flags), !,
    Code = '(args) => {
      if (args.some(a => isOutsideSandbox(a))) {
        return { ok: false, reason: \'Operation must stay within sandbox\' };
      }
      return { ok: true };
    }'.
generate_validator_for_flags(_, Flags, Code) :-
    member(no_destructive_exec, Flags), !,
    Code = '(args) => {
      const execIdx = args.indexOf(\'-exec\');
      if (execIdx !== -1) {
        const execCmd = args.slice(execIdx + 1).join(\' \');
        if (/rm|chmod|chown/.test(execCmd)) {
          return { ok: false, reason: \'Destructive -exec commands blocked\' };
        }
      }
      return { ok: true };
    }'.
generate_validator_for_flags(_, _, Code) :-
    Code = '() => ({ ok: true })'.

generate_validator_functions(Timeout, Code) :-
    format(atom(Code), '// Validation and execution
export function validateCommand(cmd: string, args: string[], ctx: ExecutionContext = {}): ValidationResult {
  const def = commandRegistry[cmd];
  if (!def) {
    return { ok: false, reason: `Unknown command: ${cmd}` };
  }
  if (def.risk === Risk.BLOCKED) {
    return { ok: false, reason: `${cmd} is blocked` };
  }
  const result = def.validate(args, ctx);
  return {
    ...result,
    risk: def.risk,
    requiresConfirmation: def.flags.includes(\'requires_confirmation\')
  };
}

export async function execute(cmdString: string, ctx: ExecutionContext = {}): Promise<ExecutionResult> {
  const parts = cmdString.trim().split(/\\s+/);
  const cmd = parts[0];
  const args = parts.slice(1);

  const validation = validateCommand(cmd, args, ctx);
  if (!validation.ok) {
    return { success: false, error: validation.reason };
  }

  if (validation.requiresConfirmation && !ctx.confirmed) {
    return { success: false, error: \'Requires confirmation\' };
  }

  return new Promise((resolve) => {
    const { spawn } = require(\'child_process\');
    const proc = spawn(cmd, args, {
      cwd: ctx.cwd || SANDBOX_ROOT,
      timeout: ctx.timeout || ~w
    });

    let stdout = \'\';
    let stderr = \'\';

    proc.stdout?.on(\'data\', (data: Buffer) => { stdout += data.toString(); });
    proc.stderr?.on(\'data\', (data: Buffer) => { stderr += data.toString(); });

    proc.on(\'close\', (code: number) => {
      resolve({ success: code === 0, code, stdout, stderr, warning: validation.warning });
    });

    proc.on(\'error\', (err: Error) => {
      resolve({ success: false, error: err.message });
    });
  });
}

export function listCommands() {
  return Object.entries(commandRegistry).map(([name, def]) => ({
    name,
    risk: def.risk,
    description: def.description
  }));
}', [Timeout]).

% ============================================================================
% File Browser Component
% ============================================================================

%! file_browser_spec(+Options, -Spec) is det
%  Create a file browser specification.
file_browser_spec(Options, Spec) :-
    get_option(sandbox_root, Options, SandboxRoot, '$HOME/sandbox'),
    get_option(show_hidden, Options, ShowHidden, false),
    get_option(max_depth, Options, MaxDepth, 10),
    Spec = file_browser_spec([
        sandbox_root(SandboxRoot),
        show_hidden(ShowHidden),
        max_depth(MaxDepth)
    ]).

%! generate_file_browser(+Spec, +Target, -Code) is det
generate_file_browser(file_browser_spec(Options), typescript, Code) :-
    get_option(sandbox_root, Options, SandboxRoot, '$HOME/sandbox'),
    get_option(show_hidden, Options, ShowHidden, false),
    format(atom(Code), '// File Browser Component
import * as fs from \'fs\';
import * as path from \'path\';

export const SANDBOX_ROOT = process.env.SANDBOX_ROOT || \'~w\';

export interface FileEntry {
  name: string;
  type: \'file\' | \'directory\';
  size?: number;
  modified?: string;
}

export interface BrowseResult {
  path: string;
  absolutePath: string;
  parent: string | null;
  entries: FileEntry[];
  count: number;
}

export function browse(browsePath: string = \'.\'): BrowseResult | { error: string } {
  const targetPath = path.resolve(SANDBOX_ROOT, browsePath);

  // Security: ensure within sandbox
  if (!targetPath.startsWith(SANDBOX_ROOT)) {
    return { error: \'Path outside sandbox\' };
  }

  if (!fs.existsSync(targetPath)) {
    return { error: \'Path not found\' };
  }

  const stats = fs.statSync(targetPath);
  if (!stats.isDirectory()) {
    return { error: \'Not a directory\' };
  }

  const items = fs.readdirSync(targetPath);
  const entries: FileEntry[] = [];

  for (const name of items) {
    // Skip hidden files unless configured
    if (name.startsWith(\'.\') && !~w) continue;

    try {
      const itemPath = path.join(targetPath, name);
      const itemStats = fs.statSync(itemPath);
      entries.push({
        name,
        type: itemStats.isDirectory() ? \'directory\' : \'file\',
        size: itemStats.isFile() ? itemStats.size : undefined,
        modified: itemStats.mtime.toISOString()
      });
    } catch {
      continue;
    }
  }

  // Sort: directories first, then alphabetically
  entries.sort((a, b) => {
    if (a.type !== b.type) return a.type === \'directory\' ? -1 : 1;
    return a.name.localeCompare(b.name);
  });

  const relativePath = path.relative(SANDBOX_ROOT, targetPath) || \'.\';

  return {
    path: relativePath,
    absolutePath: targetPath,
    parent: relativePath !== \'.\' ? path.dirname(relativePath) || \'.\' : null,
    entries,
    count: entries.length
  };
}

export function resolveWorkingDir(cwd?: string): string {
  if (!cwd || cwd === \'.\') return SANDBOX_ROOT;
  const resolved = path.resolve(SANDBOX_ROOT, cwd);
  if (!resolved.startsWith(SANDBOX_ROOT)) return SANDBOX_ROOT;
  if (!fs.existsSync(resolved) || !fs.statSync(resolved).isDirectory()) return SANDBOX_ROOT;
  return resolved;
}', [SandboxRoot, ShowHidden]).

% ============================================================================
% Feedback Store Component
% ============================================================================

%! feedback_store_spec(+Options, -Spec) is det
feedback_store_spec(Options, Spec) :-
    get_option(file_path, Options, FilePath, 'feedback.jsonl'),
    get_option(types, Options, Types, [info, success, warning, error, suggestion]),
    Spec = feedback_store_spec([
        file_path(FilePath),
        types(Types)
    ]).

%! generate_feedback_store(+Spec, +Target, -Code) is det
generate_feedback_store(feedback_store_spec(Options), typescript, Code) :-
    get_option(file_path, Options, FilePath, 'feedback.jsonl'),
    get_option(types, Options, Types, [info, success, warning, error, suggestion]),
    types_to_union(Types, TypeUnion),
    format(atom(Code), '// Feedback Store Component
import * as fs from \'fs\';
import * as path from \'path\';

export type FeedbackType = ~w;

export interface FeedbackEntry {
  timestamp: string;
  type: FeedbackType;
  message: string;
  context?: string;
}

const FEEDBACK_FILE = process.env.FEEDBACK_FILE || \'~w\';

export function submitFeedback(message: string, type: FeedbackType = \'info\', context?: string): FeedbackEntry {
  const entry: FeedbackEntry = {
    timestamp: new Date().toISOString(),
    type,
    message,
    context
  };

  const logLine = JSON.stringify(entry) + \'\\n\';
  fs.appendFileSync(FEEDBACK_FILE, logLine, \'utf-8\');

  return entry;
}

export function getFeedback(): FeedbackEntry[] {
  if (!fs.existsSync(FEEDBACK_FILE)) {
    return [];
  }

  const content = fs.readFileSync(FEEDBACK_FILE, \'utf-8\');
  const lines = content.trim().split(\'\\n\').filter(Boolean);

  return lines.map(line => {
    try {
      return JSON.parse(line) as FeedbackEntry;
    } catch {
      return { timestamp: \'\', type: \'info\' as FeedbackType, message: line };
    }
  });
}', [TypeUnion, FilePath]).

types_to_union(Types, Union) :-
    maplist(quote_string, Types, Quoted),
    atomic_list_concat(Quoted, ' | ', Union).

% ============================================================================
% WebSocket Shell Component
% ============================================================================

%! websocket_shell_spec(+Options, -Spec) is det
websocket_shell_spec(Options, Spec) :-
    get_option(sandbox_root, Options, SandboxRoot, '$HOME/sandbox'),
    get_option(builtins, Options, Builtins, [help, cd, pwd, exit]),
    get_option(welcome_message, Options, Welcome, 'Connected to Shell'),
    Spec = websocket_shell_spec([
        sandbox_root(SandboxRoot),
        builtins(Builtins),
        welcome_message(Welcome)
    ]).

%! generate_websocket_shell(+Spec, +Target, -Code) is det
generate_websocket_shell(websocket_shell_spec(Options), typescript, Code) :-
    get_option(sandbox_root, Options, SandboxRoot, '$HOME/sandbox'),
    get_option(welcome_message, Options, Welcome, 'Connected to Shell'),
    format(atom(Code), '// WebSocket Shell Component
import WebSocket from \'ws\';
import { spawn, ChildProcessWithoutNullStreams } from \'child_process\';
import * as path from \'path\';
import * as fs from \'fs\';

export const SANDBOX_ROOT = process.env.SANDBOX_ROOT || \'~w\';

export interface ShellSession {
  ws: WebSocket;
  user: { email: string; roles: string[] };
  process: ChildProcessWithoutNullStreams | null;
  currentDir: string;
  inputBuffer: string;
}

export function createShellSession(ws: WebSocket, user: { email: string; roles: string[] }): ShellSession {
  return {
    ws,
    user,
    process: null,
    currentDir: SANDBOX_ROOT,
    inputBuffer: \'\'
  };
}

export function handleShellConnection(ws: WebSocket, user: { email: string; roles: string[] }): void {
  const session = createShellSession(ws, user);

  // Welcome message
  const welcome = `\\r\\n~w\\r\\nUser: ${user.email} [${user.roles.join(\', \')}]\\r\\nWorking directory: ${SANDBOX_ROOT}\\r\\n\\r\\n`;
  ws.send(JSON.stringify({ type: \'output\', data: welcome }));
  ws.send(JSON.stringify({ type: \'prompt\', cwd: session.currentDir }));

  ws.on(\'message\', (data: Buffer | string) => {
    try {
      const msg = JSON.parse(data.toString());
      if (msg.type === \'input\') {
        handleShellInput(session, msg.data);
      }
    } catch (err) {
      console.error(\'Shell message error:\', err);
    }
  });

  ws.on(\'close\', () => {
    if (session.process) {
      session.process.kill();
      session.process = null;
    }
  });
}

function handleShellInput(session: ShellSession, char: string): void {
  const { ws } = session;

  // Enter key
  if (char === \'\\r\' || char === \'\\n\') {
    ws.send(JSON.stringify({ type: \'output\', data: \'\\r\\n\' }));
    const command = session.inputBuffer.trim();
    session.inputBuffer = \'\';

    if (command) {
      executeShellCommand(session, command);
    } else {
      ws.send(JSON.stringify({ type: \'prompt\', cwd: session.currentDir }));
    }
  }
  // Backspace
  else if (char === \'\\x7f\' || char === \'\\b\') {
    if (session.inputBuffer.length > 0) {
      session.inputBuffer = session.inputBuffer.slice(0, -1);
      ws.send(JSON.stringify({ type: \'output\', data: \'\\b \\b\' }));
    }
  }
  // Ctrl+C
  else if (char === \'\\x03\') {
    if (session.process) {
      session.process.kill(\'SIGINT\');
      session.process = null;
    }
    ws.send(JSON.stringify({ type: \'output\', data: \'^C\\r\\n\' }));
    session.inputBuffer = \'\';
    ws.send(JSON.stringify({ type: \'prompt\', cwd: session.currentDir }));
  }
  // Printable characters
  else if (char >= \' \' && char <= \'~\') {
    session.inputBuffer += char;
    ws.send(JSON.stringify({ type: \'output\', data: char }));
  }
}

function executeShellCommand(session: ShellSession, command: string): void {
  const { ws } = session;

  // Built-in: help
  if (command === \'help\') {
    const helpText = `\\r\\nBuilt-in Commands:\\r\\n  help - Show this help\\r\\n  cd DIR - Change directory\\r\\n  pwd - Print working directory\\r\\n  exit - Disconnect\\r\\n\\r\\n`;
    ws.send(JSON.stringify({ type: \'output\', data: helpText }));
    ws.send(JSON.stringify({ type: \'prompt\', cwd: session.currentDir }));
    return;
  }

  // Built-in: exit
  if (command === \'exit\' || command === \'quit\') {
    ws.send(JSON.stringify({ type: \'output\', data: \'Goodbye!\\r\\n\' }));
    ws.close();
    return;
  }

  // Built-in: cd
  if (command.startsWith(\'cd \') || command === \'cd\') {
    const targetDir = command.slice(3).trim() || SANDBOX_ROOT;
    let newDir: string;

    if (path.isAbsolute(targetDir)) {
      newDir = targetDir;
    } else if (targetDir === \'~\') {
      newDir = SANDBOX_ROOT;
    } else if (targetDir.startsWith(\'~/\')) {
      newDir = path.join(SANDBOX_ROOT, targetDir.slice(2));
    } else {
      newDir = path.resolve(session.currentDir, targetDir);
    }

    // Security: ensure within sandbox
    if (!newDir.startsWith(SANDBOX_ROOT)) {
      ws.send(JSON.stringify({ type: \'output\', data: `\\x1b[31mcd: Access denied - outside sandbox\\x1b[0m\\r\\n` }));
      ws.send(JSON.stringify({ type: \'prompt\', cwd: session.currentDir }));
      return;
    }

    if (fs.existsSync(newDir) && fs.statSync(newDir).isDirectory()) {
      session.currentDir = newDir;
    } else {
      ws.send(JSON.stringify({ type: \'output\', data: `\\x1b[31mcd: ${targetDir}: No such directory\\x1b[0m\\r\\n` }));
    }
    ws.send(JSON.stringify({ type: \'prompt\', cwd: session.currentDir }));
    return;
  }

  // Built-in: pwd
  if (command === \'pwd\') {
    ws.send(JSON.stringify({ type: \'output\', data: session.currentDir + \'\\r\\n\' }));
    ws.send(JSON.stringify({ type: \'prompt\', cwd: session.currentDir }));
    return;
  }

  // Execute command
  const proc = spawn(\'sh\', [\'-c\', command], {
    cwd: session.currentDir,
    env: { ...process.env, HOME: SANDBOX_ROOT, TERM: \'xterm-256color\' }
  });

  session.process = proc;

  proc.stdout.on(\'data\', (data: Buffer) => {
    ws.send(JSON.stringify({ type: \'output\', data: data.toString().replace(/\\n/g, \'\\r\\n\') }));
  });

  proc.stderr.on(\'data\', (data: Buffer) => {
    ws.send(JSON.stringify({ type: \'output\', data: `\\x1b[31m${data.toString().replace(/\\n/g, \'\\r\\n\')}\\x1b[0m` }));
  });

  proc.on(\'close\', (code: number) => {
    session.process = null;
    if (code !== 0) {
      ws.send(JSON.stringify({ type: \'output\', data: `\\x1b[33m[exit code: ${code}]\\x1b[0m\\r\\n` }));
    }
    ws.send(JSON.stringify({ type: \'prompt\', cwd: session.currentDir }));
  });

  proc.on(\'error\', (err: Error) => {
    session.process = null;
    ws.send(JSON.stringify({ type: \'output\', data: `\\x1b[31mError: ${err.message}\\x1b[0m\\r\\n` }));
    ws.send(JSON.stringify({ type: \'prompt\', cwd: session.currentDir }));
  });
}', [SandboxRoot, Welcome]).

% ============================================================================
% Endpoint Handler Component
% ============================================================================

%! endpoint_handler_spec(+Name, +Options, -Spec) is det
%  Create an endpoint handler specification.
%  Options:
%    type(Type) - Handler type: grep, find, cat, browse, exec, feedback, custom
%    roles(Roles) - Required roles
%    use_command_validator(Bool) - Whether to use command validation
endpoint_handler_spec(Name, Options, Spec) :-
    get_option(type, Options, Type, custom),
    get_option(roles, Options, Roles, []),
    get_option(use_command_validator, Options, UseValidator, false),
    get_option(sandbox_root, Options, SandboxRoot, 'SANDBOX_ROOT'),
    Spec = endpoint_handler_spec(Name, [
        type(Type),
        roles(Roles),
        use_command_validator(UseValidator),
        sandbox_root(SandboxRoot)
    ]).

%! generate_endpoint_handler(+Spec, +Target, +Imports, -Code) is det
generate_endpoint_handler(endpoint_handler_spec(Name, Options), typescript, Imports, Code) :-
    get_option(type, Options, Type, custom),
    generate_handler_by_type(Name, Type, Options, Imports, Code).

generate_handler_by_type(Name, grep, Options, Imports, Code) :-
    get_option(sandbox_root, Options, _SandboxRoot, 'SANDBOX_ROOT'),
    Imports = ['execute', 'resolveWorkingDir'],
    format(atom(Code), 'async function handle_~w(
  req: http.IncomingMessage,
  res: http.ServerResponse,
  body: RequestBody,
  user: TokenPayload | null
): Promise<void> {
  const { pattern, path: searchPath = \'.\', options = [], cwd } = body as any;

  if (!pattern) {
    return sendJSON(res, 400, { success: false, error: \'Missing pattern\' });
  }

  const workingDir = resolveWorkingDir(cwd as string);
  const args = [\'-r\', \'-n\', \'--color=never\', ...options, pattern, searchPath];
  const cmdString = [\'grep\', ...args].join(\' \');
  const result = await execute(cmdString, { role: \'user\', cwd: workingDir });

  sendJSON(res, 200, {
    success: true,
    data: {
      matches: result.stdout?.split(\'\\n\').filter(Boolean) || [],
      count: result.stdout?.split(\'\\n\').filter(Boolean).length || 0,
      stderr: result.stderr
    },
    warning: result.warning
  });
}', [Name]).

generate_handler_by_type(Name, find, Options, Imports, Code) :-
    get_option(sandbox_root, Options, _SandboxRoot, 'SANDBOX_ROOT'),
    Imports = ['execute', 'resolveWorkingDir'],
    format(atom(Code), 'async function handle_~w(
  req: http.IncomingMessage,
  res: http.ServerResponse,
  body: RequestBody,
  user: TokenPayload | null
): Promise<void> {
  const { pattern, path: searchPath = \'.\', options = [], cwd } = body as any;

  const workingDir = resolveWorkingDir(cwd as string);
  const args = [searchPath];
  if (pattern) args.push(\'-name\', pattern);
  args.push(...options);

  const cmdString = [\'find\', ...args].join(\' \');
  const result = await execute(cmdString, { role: \'user\', cwd: workingDir });

  sendJSON(res, 200, {
    success: true,
    data: {
      files: result.stdout?.split(\'\\n\').filter(Boolean) || [],
      count: result.stdout?.split(\'\\n\').filter(Boolean).length || 0
    },
    warning: result.warning
  });
}', [Name]).

generate_handler_by_type(Name, cat, Options, Imports, Code) :-
    get_option(sandbox_root, Options, _SandboxRoot, 'SANDBOX_ROOT'),
    Imports = ['execute', 'resolveWorkingDir'],
    format(atom(Code), 'async function handle_~w(
  req: http.IncomingMessage,
  res: http.ServerResponse,
  body: RequestBody,
  user: TokenPayload | null
): Promise<void> {
  const { path: filePath, options = [], cwd } = body as any;

  if (!filePath) {
    return sendJSON(res, 400, { success: false, error: \'Missing path\' });
  }

  const workingDir = resolveWorkingDir(cwd as string);
  const cmdString = [\'cat\', ...options, filePath].join(\' \');
  const result = await execute(cmdString, { role: \'user\', cwd: workingDir });

  if (!result.success) {
    return sendJSON(res, 400, { success: false, error: result.error });
  }

  sendJSON(res, 200, {
    success: true,
    data: {
      content: result.stdout,
      lines: result.stdout?.split(\'\\n\').length || 0
    }
  });
}', [Name]).

generate_handler_by_type(Name, browse, _Options, Imports, Code) :-
    Imports = ['browse'],
    format(atom(Code), 'async function handle_~w(
  req: http.IncomingMessage,
  res: http.ServerResponse,
  body: RequestBody,
  user: TokenPayload | null
): Promise<void> {
  const { path: browsePath = \'.\' } = body as any;
  const result = browse(browsePath);

  if (\'error\' in result) {
    return sendJSON(res, 400, { success: false, error: result.error });
  }

  sendJSON(res, 200, { success: true, data: result });
}', [Name]).

generate_handler_by_type(Name, exec, _Options, Imports, Code) :-
    Imports = ['execute', 'validateCommand', 'resolveWorkingDir'],
    format(atom(Code), 'async function handle_~w(
  req: http.IncomingMessage,
  res: http.ServerResponse,
  body: RequestBody,
  user: TokenPayload | null
): Promise<void> {
  const { command, args = [], cwd } = body as any;

  if (!command) {
    return sendJSON(res, 400, { success: false, error: \'Missing command\' });
  }

  const workingDir = resolveWorkingDir(cwd as string);
  const validation = validateCommand(command, args);

  if (!validation.ok) {
    return sendJSON(res, 403, { success: false, error: validation.reason });
  }

  const cmdString = [command, ...args].join(\' \');
  const result = await execute(cmdString, { role: \'user\', cwd: workingDir });

  sendJSON(res, result.success ? 200 : 400, {
    success: result.success,
    data: result.success ? { stdout: result.stdout, stderr: result.stderr, code: result.code } : undefined,
    error: result.error,
    warning: result.warning
  });
}', [Name]).

generate_handler_by_type(Name, feedback, _Options, Imports, Code) :-
    Imports = ['submitFeedback', 'getFeedback'],
    format(atom(Code), 'async function handle_~w(
  req: http.IncomingMessage,
  res: http.ServerResponse,
  body: RequestBody,
  user: TokenPayload | null,
  method: string
): Promise<void> {
  if (method === \'GET\') {
    const entries = getFeedback();
    return sendJSON(res, 200, {
      success: true,
      data: { entries, count: entries.length }
    });
  }

  // POST
  const { message, type = \'info\', context } = body as any;

  if (!message) {
    return sendJSON(res, 400, { success: false, error: \'Missing message\' });
  }

  const entry = submitFeedback(message, type, context);
  sendJSON(res, 200, {
    success: true,
    data: { recorded: true, timestamp: entry.timestamp }
  });
}', [Name]).

generate_handler_by_type(Name, login, _Options, Imports, Code) :-
    Imports = ['login'],
    format(atom(Code), 'async function handle_~w(
  req: http.IncomingMessage,
  res: http.ServerResponse,
  body: RequestBody,
  user: TokenPayload | null
): Promise<void> {
  const { email, password } = body as any;

  if (!email || !password) {
    return sendJSON(res, 400, { success: false, error: \'Email and password required\' });
  }

  const result = login(email, password);

  if (!result.success) {
    return sendJSON(res, 401, { success: false, error: result.error || \'Login failed\' });
  }

  sendJSON(res, 200, {
    success: true,
    data: { token: result.token, user: result.user }
  });
}', [Name]).

generate_handler_by_type(Name, auth_me, _Options, Imports, Code) :-
    Imports = [],
    format(atom(Code), 'async function handle_~w(
  req: http.IncomingMessage,
  res: http.ServerResponse,
  body: RequestBody,
  user: TokenPayload | null
): Promise<void> {
  if (!user) {
    return sendJSON(res, 401, { success: false, error: \'Not authenticated\' });
  }

  sendJSON(res, 200, {
    success: true,
    data: {
      id: user.sub,
      email: user.email,
      roles: user.roles,
      permissions: user.permissions
    }
  });
}', [Name]).

generate_handler_by_type(Name, auth_status, _Options, Imports, Code) :-
    Imports = [],
    format(atom(Code), 'async function handle_~w(
  req: http.IncomingMessage,
  res: http.ServerResponse,
  body: RequestBody,
  user: TokenPayload | null
): Promise<void> {
  sendJSON(res, 200, {
    success: true,
    data: {
      authRequired: AUTH_REQUIRED,
      authenticated: user !== null,
      user: user ? { id: user.sub, email: user.email, roles: user.roles } : null
    }
  });
}', [Name]).

generate_handler_by_type(Name, health, _Options, Imports, Code) :-
    Imports = [],
    format(atom(Code), 'async function handle_~w(
  req: http.IncomingMessage,
  res: http.ServerResponse,
  body: RequestBody,
  user: TokenPayload | null
): Promise<void> {
  sendJSON(res, 200, {
    success: true,
    data: {
      status: \'ok\',
      timestamp: new Date().toISOString()
    }
  });
}', [Name]).

generate_handler_by_type(Name, commands, _Options, Imports, Code) :-
    Imports = ['listCommands'],
    format(atom(Code), 'async function handle_~w(
  req: http.IncomingMessage,
  res: http.ServerResponse,
  body: RequestBody,
  user: TokenPayload | null
): Promise<void> {
  const commands = listCommands();
  sendJSON(res, 200, {
    success: true,
    data: { commands }
  });
}', [Name]).

generate_handler_by_type(Name, custom, _Options, Imports, Code) :-
    Imports = [],
    format(atom(Code), 'async function handle_~w(
  req: http.IncomingMessage,
  res: http.ServerResponse,
  body: RequestBody,
  user: TokenPayload | null
): Promise<void> {
  // TODO: Implement ~w logic
  sendJSON(res, 200, { success: true, data: { endpoint: \'~w\' } });
}', [Name, Name, Name]).

% ============================================================================
% Utility: Generate full server component
% ============================================================================

%! generate_server_component(+ComponentType, +Options, +Target, -Code) is det
generate_server_component(command_validator, Options, Target, Code) :-
    command_validator_spec(Options, Spec),
    generate_command_validator(Spec, Target, Code).
generate_server_component(file_browser, Options, Target, Code) :-
    file_browser_spec(Options, Spec),
    generate_file_browser(Spec, Target, Code).
generate_server_component(feedback_store, Options, Target, Code) :-
    feedback_store_spec(Options, Spec),
    generate_feedback_store(Spec, Target, Code).
generate_server_component(websocket_shell, Options, Target, Code) :-
    websocket_shell_spec(Options, Spec),
    generate_websocket_shell(Spec, Target, Code).

% ============================================================================
% Option helpers
% ============================================================================

get_option(Key, Options, Value, _Default) :-
    member(Key(Value), Options), !.
get_option(_Key, _Options, Default, Default).
