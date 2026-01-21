#!/usr/bin/env ts-node
/**
 * HTTP/HTTPS CLI Server
 *
 * Exposes shell search commands (grep, find, cat, rg) via HTTP/HTTPS endpoints
 * for use with AI browsers like Comet (Perplexity).
 *
 * Uses command-proxy.ts for validation and execution.
 *
 * Usage:
 *   ts-node http-server.ts
 *   ts-node http-server.ts --port 3001
 *   ts-node http-server.ts --cert cert.pem --key key.pem  # HTTPS mode
 *
 * Endpoints:
 *   GET  /health              - Health check
 *   GET  /commands            - List available commands
 *   POST /exec                - Execute a command
 *   POST /grep                - Search file contents
 *   POST /find                - Find files by pattern
 *   POST /cat                 - Read file contents
 *   GET  /                    - Serve HTML interface
 *   WS   /shell               - WebSocket shell (requires shell role)
 *
 * @module unifyweaver/shell/http-server
 */

import * as http from 'http';
import * as https from 'https';
import * as url from 'url';
import * as path from 'path';
import * as fs from 'fs';
import { execSync, spawn, ChildProcessWithoutNullStreams } from 'child_process';
import WebSocket, { WebSocketServer } from 'ws';
import {
  execute,
  validateCommand,
  listCommands,
  Risk,
  SANDBOX_ROOT
} from './command-proxy';
import {
  login,
  getUserFromToken,
  verifyToken,
  hasRole,
  canAccessShell,
  canExecuteCommands,
  canBrowse,
  TokenPayload
} from './auth';

// ============================================================================
// Configuration
// ============================================================================

const DEFAULT_PORT = 3001;
const MAX_BODY_SIZE = 1024 * 1024; // 1MB
const ALLOWED_ORIGINS = ['*']; // Configure for production

// Authentication: set AUTH_REQUIRED=true to require login
const AUTH_REQUIRED = process.env.AUTH_REQUIRED === 'true';

// Commands specifically allowed for search operations
const SEARCH_COMMANDS = ['grep', 'find', 'cat', 'head', 'tail', 'ls', 'wc', 'pwd'];

// Feedback log file
const FEEDBACK_FILE = process.env.FEEDBACK_FILE || path.join(SANDBOX_ROOT, 'comet-feedback.log');

// ============================================================================
// Types
// ============================================================================

interface SearchRequest {
  pattern?: string;
  path?: string;
  options?: string[];
  command?: string;
  args?: string[];
  cwd?: string;  // Working directory relative to sandbox root
}

interface FeedbackRequest {
  message: string;
  type?: 'info' | 'success' | 'warning' | 'error' | 'suggestion';
  context?: string;
}

interface BrowseRequest {
  path?: string;
}

interface FileEntry {
  name: string;
  type: 'file' | 'directory';
  size?: number;
  modified?: string;
}

interface APIResponse {
  success: boolean;
  data?: unknown;
  error?: string;
  warning?: string;
}

// ============================================================================
// Helpers
// ============================================================================

function parseBody(req: http.IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    let body = '';
    let size = 0;

    req.on('data', (chunk: Buffer) => {
      size += chunk.length;
      if (size > MAX_BODY_SIZE) {
        reject(new Error('Request body too large'));
        req.destroy();
        return;
      }
      body += chunk.toString();
    });

    req.on('end', () => resolve(body));
    req.on('error', reject);
  });
}

function sendJSON(res: http.ServerResponse, data: APIResponse, status = 200): void {
  res.writeHead(status, {
    'Content-Type': 'application/json',
    'Access-Control-Allow-Origin': ALLOWED_ORIGINS[0],
    'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type, Authorization'
  });
  res.end(JSON.stringify(data, null, 2));
}

function sendHTML(res: http.ServerResponse, html: string): void {
  res.writeHead(200, {
    'Content-Type': 'text/html; charset=utf-8',
    'Access-Control-Allow-Origin': ALLOWED_ORIGINS[0]
  });
  res.end(html);
}

function sendError(res: http.ServerResponse, message: string, status = 400): void {
  sendJSON(res, { success: false, error: message }, status);
}

/**
 * Extract user from Authorization header.
 * Returns null if no valid token, or the token payload if valid.
 */
function getAuthUser(req: http.IncomingMessage): TokenPayload | null {
  const authHeader = req.headers.authorization;
  if (!authHeader || !authHeader.startsWith('Bearer ')) {
    return null;
  }
  const token = authHeader.substring(7);
  return verifyToken(token);
}

/**
 * Check if request is authenticated (when AUTH_REQUIRED is true).
 * Returns the user if authenticated, or sends 401 and returns null.
 */
function requireAuth(req: http.IncomingMessage, res: http.ServerResponse): TokenPayload | null {
  if (!AUTH_REQUIRED) {
    // Return a default user when auth is not required
    return {
      sub: 'anonymous',
      email: 'anonymous@local',
      roles: ['shell', 'admin', 'user'],  // Full access when auth disabled
      permissions: ['read', 'write', 'delete', 'shell'],
      iat: 0,
      exp: 0
    };
  }

  const user = getAuthUser(req);
  if (!user) {
    sendError(res, 'Authentication required', 401);
    return null;
  }
  return user;
}

/**
 * Check if user has the required role.
 * Returns true if access granted, false if denied (and sends 403).
 */
function requireRole(user: TokenPayload, res: http.ServerResponse, ...requiredRoles: string[]): boolean {
  if (requiredRoles.some(role => user.roles.includes(role))) {
    return true;
  }
  sendError(res, `Forbidden: requires one of roles: ${requiredRoles.join(', ')}`, 403);
  return false;
}

/**
 * Resolve a working directory path safely within the sandbox.
 * Returns the absolute path or SANDBOX_ROOT if invalid.
 */
function resolveWorkingDir(cwd?: string): string {
  if (!cwd || cwd === '.') {
    return SANDBOX_ROOT;
  }
  const resolved = path.resolve(SANDBOX_ROOT, cwd);
  // Ensure it's within sandbox and exists
  if (!resolved.startsWith(SANDBOX_ROOT)) {
    return SANDBOX_ROOT;
  }
  if (!fs.existsSync(resolved) || !fs.statSync(resolved).isDirectory()) {
    return SANDBOX_ROOT;
  }
  return resolved;
}

/**
 * Expand glob patterns in arguments using the shell.
 * Only expands patterns containing * or ?, and only within sandbox.
 */
function expandGlobs(args: string[], cwd: string): string[] {
  const expanded: string[] = [];
  for (const arg of args) {
    // Only expand if it looks like a glob pattern
    if (arg.includes('*') || arg.includes('?')) {
      try {
        // Use shell to expand the glob, constrained to cwd
        const result = execSync(`printf '%s\\n' ${arg}`, {
          cwd,
          encoding: 'utf-8',
          timeout: 5000,
          shell: '/bin/sh'
        }).trim();

        // If expansion found matches, add them; otherwise keep original
        if (result && result !== arg) {
          expanded.push(...result.split('\n').filter(Boolean));
        } else {
          expanded.push(arg);
        }
      } catch {
        // If expansion fails, keep the original argument
        expanded.push(arg);
      }
    } else {
      expanded.push(arg);
    }
  }
  return expanded;
}

// ============================================================================
// Route Handlers
// ============================================================================

async function handleHealth(req: http.IncomingMessage, res: http.ServerResponse): Promise<void> {
  sendJSON(res, {
    success: true,
    data: {
      status: 'ok',
      sandboxRoot: SANDBOX_ROOT,
      commands: SEARCH_COMMANDS,
      timestamp: new Date().toISOString()
    }
  });
}

async function handleCommands(req: http.IncomingMessage, res: http.ServerResponse): Promise<void> {
  const commands = listCommands().filter(c =>
    SEARCH_COMMANDS.includes(c.name) || c.risk === Risk.SAFE
  );

  sendJSON(res, {
    success: true,
    data: {
      commands,
      searchCommands: SEARCH_COMMANDS
    }
  });
}

async function handleExec(req: http.IncomingMessage, res: http.ServerResponse): Promise<void> {
  try {
    const body = await parseBody(req);
    const { command, args = [], cwd } = JSON.parse(body) as SearchRequest;

    if (!command) {
      return sendError(res, 'Missing command');
    }

    // Only allow search commands
    if (!SEARCH_COMMANDS.includes(command)) {
      return sendError(res, `Command not allowed. Allowed: ${SEARCH_COMMANDS.join(', ')}`, 403);
    }

    // Resolve working directory
    const workingDir = resolveWorkingDir(cwd);

    // Expand glob patterns in arguments
    const expandedArgs = expandGlobs(args, workingDir);

    // Validate first
    const validation = validateCommand(command, expandedArgs, { role: 'user' });
    if (!validation.ok) {
      return sendError(res, validation.reason || 'Validation failed', 403);
    }

    // Execute
    const cmdString = [command, ...expandedArgs].join(' ');
    const result = await execute(cmdString, { role: 'user', cwd: workingDir });

    if (!result.success) {
      return sendJSON(res, {
        success: false,
        error: result.error,
        warning: result.warning
      }, 400);
    }

    sendJSON(res, {
      success: true,
      data: {
        stdout: result.stdout,
        stderr: result.stderr,
        code: result.code
      },
      warning: result.warning
    });
  } catch (err) {
    sendError(res, (err as Error).message, 500);
  }
}

async function handleGrep(req: http.IncomingMessage, res: http.ServerResponse): Promise<void> {
  try {
    const body = await parseBody(req);
    const { pattern, path: searchPath = '.', options = [], cwd } = JSON.parse(body) as SearchRequest;

    if (!pattern) {
      return sendError(res, 'Missing pattern');
    }

    // Resolve working directory
    const workingDir = resolveWorkingDir(cwd);

    // Build grep command
    const args = [
      '-r',           // Recursive
      '-n',           // Line numbers
      '--color=never', // No color codes in output
      ...options,
      pattern,
      searchPath
    ];

    const cmdString = ['grep', ...args].join(' ');
    const result = await execute(cmdString, { role: 'user', cwd: workingDir });

    sendJSON(res, {
      success: true,
      data: {
        matches: result.stdout?.split('\n').filter(Boolean) || [],
        count: result.stdout?.split('\n').filter(Boolean).length || 0,
        stderr: result.stderr
      },
      warning: result.warning
    });
  } catch (err) {
    sendError(res, (err as Error).message, 500);
  }
}

async function handleFind(req: http.IncomingMessage, res: http.ServerResponse): Promise<void> {
  try {
    const body = await parseBody(req);
    const { pattern, path: searchPath = '.', options = [], cwd } = JSON.parse(body) as SearchRequest;

    // Resolve working directory
    const workingDir = resolveWorkingDir(cwd);

    // Build find command
    const args = [searchPath];

    if (pattern) {
      args.push('-name', pattern);
    }

    args.push(...options);

    const cmdString = ['find', ...args].join(' ');
    const result = await execute(cmdString, { role: 'user', cwd: workingDir });

    sendJSON(res, {
      success: true,
      data: {
        files: result.stdout?.split('\n').filter(Boolean) || [],
        count: result.stdout?.split('\n').filter(Boolean).length || 0,
        stderr: result.stderr
      },
      warning: result.warning
    });
  } catch (err) {
    sendError(res, (err as Error).message, 500);
  }
}

async function handleCat(req: http.IncomingMessage, res: http.ServerResponse): Promise<void> {
  try {
    const body = await parseBody(req);
    const { path: filePath, options = [], cwd } = JSON.parse(body) as SearchRequest;

    if (!filePath) {
      return sendError(res, 'Missing path');
    }

    // Resolve working directory
    const workingDir = resolveWorkingDir(cwd);

    const args = [...options, filePath];
    const cmdString = ['cat', ...args].join(' ');
    const result = await execute(cmdString, { role: 'user', cwd: workingDir });

    if (!result.success) {
      return sendJSON(res, {
        success: false,
        error: result.error
      }, 400);
    }

    sendJSON(res, {
      success: true,
      data: {
        content: result.stdout,
        lines: result.stdout?.split('\n').length || 0
      },
      warning: result.warning
    });
  } catch (err) {
    sendError(res, (err as Error).message, 500);
  }
}

async function handleFeedback(req: http.IncomingMessage, res: http.ServerResponse): Promise<void> {
  try {
    const body = await parseBody(req);
    const { message, type = 'info', context } = JSON.parse(body) as FeedbackRequest;

    if (!message) {
      return sendError(res, 'Missing message');
    }

    // Create timestamped record
    const timestamp = new Date().toISOString();
    const record = {
      timestamp,
      type,
      message,
      context: context || null
    };

    const logLine = JSON.stringify(record) + '\n';

    // Append to feedback file
    fs.appendFileSync(FEEDBACK_FILE, logLine, 'utf-8');

    sendJSON(res, {
      success: true,
      data: {
        recorded: true,
        timestamp,
        file: FEEDBACK_FILE
      }
    });
  } catch (err) {
    sendError(res, (err as Error).message, 500);
  }
}

async function handleGetFeedback(req: http.IncomingMessage, res: http.ServerResponse): Promise<void> {
  try {
    if (!fs.existsSync(FEEDBACK_FILE)) {
      return sendJSON(res, {
        success: true,
        data: {
          entries: [],
          count: 0
        }
      });
    }

    const content = fs.readFileSync(FEEDBACK_FILE, 'utf-8');
    const lines = content.trim().split('\n').filter(Boolean);
    const entries = lines.map(line => {
      try {
        return JSON.parse(line);
      } catch {
        return { raw: line };
      }
    });

    sendJSON(res, {
      success: true,
      data: {
        entries,
        count: entries.length,
        file: FEEDBACK_FILE
      }
    });
  } catch (err) {
    sendError(res, (err as Error).message, 500);
  }
}

async function handleBrowse(req: http.IncomingMessage, res: http.ServerResponse): Promise<void> {
  try {
    const body = await parseBody(req);
    const { path: browsePath = '.' } = JSON.parse(body || '{}') as BrowseRequest;

    // Resolve and validate path is within sandbox
    const targetPath = path.resolve(SANDBOX_ROOT, browsePath);
    if (!targetPath.startsWith(SANDBOX_ROOT)) {
      return sendError(res, 'Path outside sandbox', 403);
    }

    if (!fs.existsSync(targetPath)) {
      return sendError(res, 'Path not found', 404);
    }

    const stats = fs.statSync(targetPath);
    if (!stats.isDirectory()) {
      return sendError(res, 'Not a directory', 400);
    }

    // Read directory contents
    const items = fs.readdirSync(targetPath);
    const entries: FileEntry[] = [];

    for (const name of items) {
      // Skip hidden files starting with . unless explicitly in root
      if (name.startsWith('.') && browsePath !== '.') continue;

      try {
        const itemPath = path.join(targetPath, name);
        const itemStats = fs.statSync(itemPath);
        entries.push({
          name,
          type: itemStats.isDirectory() ? 'directory' : 'file',
          size: itemStats.isFile() ? itemStats.size : undefined,
          modified: itemStats.mtime.toISOString()
        });
      } catch {
        // Skip items we can't stat (permission errors, etc.)
        continue;
      }
    }

    // Sort: directories first, then alphabetically
    entries.sort((a, b) => {
      if (a.type !== b.type) {
        return a.type === 'directory' ? -1 : 1;
      }
      return a.name.localeCompare(b.name);
    });

    // Calculate relative path from sandbox root
    const relativePath = path.relative(SANDBOX_ROOT, targetPath) || '.';

    sendJSON(res, {
      success: true,
      data: {
        path: relativePath,
        absolutePath: targetPath,
        parent: relativePath !== '.' ? path.dirname(relativePath) || '.' : null,
        entries,
        count: entries.length
      }
    });
  } catch (err) {
    sendError(res, (err as Error).message, 500);
  }
}

// ============================================================================
// Auth Handlers
// ============================================================================

async function handleLogin(req: http.IncomingMessage, res: http.ServerResponse): Promise<void> {
  try {
    const body = await parseBody(req);
    const { email, password } = JSON.parse(body);

    if (!email || !password) {
      return sendError(res, 'Email and password required', 400);
    }

    const result = login(email, password);

    if (!result.success) {
      return sendError(res, result.error || 'Login failed', 401);
    }

    sendJSON(res, {
      success: true,
      data: {
        token: result.token,
        user: result.user
      }
    });
  } catch (err) {
    sendError(res, (err as Error).message, 500);
  }
}

async function handleMe(req: http.IncomingMessage, res: http.ServerResponse): Promise<void> {
  const user = getAuthUser(req);

  if (!user) {
    return sendError(res, 'Not authenticated', 401);
  }

  sendJSON(res, {
    success: true,
    data: {
      id: user.sub,
      email: user.email,
      roles: user.roles,
      permissions: user.permissions
    }
  });
}

async function handleAuthStatus(req: http.IncomingMessage, res: http.ServerResponse): Promise<void> {
  const user = getAuthUser(req);

  sendJSON(res, {
    success: true,
    data: {
      authRequired: AUTH_REQUIRED,
      authenticated: user !== null,
      user: user ? {
        id: user.sub,
        email: user.email,
        roles: user.roles
      } : null
    }
  });
}

// ============================================================================
// HTML Interface
// ============================================================================

function getHTMLInterface(): string {
  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>UnifyWeaver CLI Search</title>
  <script src="https://unpkg.com/vue@3/dist/vue.global.prod.js"></script>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, monospace;
      background: #1a1a2e;
      color: #eee;
      min-height: 100vh;
      padding: 20px;
    }
    .container { max-width: 1200px; margin: 0 auto; }
    h1 { color: #e94560; margin-bottom: 20px; }
    .tabs {
      display: flex;
      flex-wrap: wrap;
      gap: 5px;
      margin-bottom: 20px;
    }
    .tab {
      padding: 10px 20px;
      background: #16213e;
      border: none;
      color: #94a3b8;
      cursor: pointer;
      border-radius: 5px 5px 0 0;
    }
    .tab.active { background: #0f3460; color: #e94560; }
    .panel {
      background: #16213e;
      padding: 20px;
      border-radius: 0 5px 5px 5px;
    }
    .form-group { margin-bottom: 15px; }
    label { display: block; margin-bottom: 5px; color: #94a3b8; }
    input, textarea {
      width: 100%;
      padding: 10px;
      background: #0f3460;
      border: 1px solid #1a1a2e;
      color: #eee;
      border-radius: 5px;
      font-family: monospace;
    }
    input:focus, textarea:focus {
      outline: none;
      border-color: #e94560;
    }
    button {
      padding: 10px 20px;
      background: #e94560;
      border: none;
      color: #fff;
      cursor: pointer;
      border-radius: 5px;
      font-weight: bold;
    }
    button:hover { background: #ff6b6b; }
    button:disabled { background: #444; cursor: not-allowed; }
    .results {
      margin-top: 20px;
      background: #0f3460;
      padding: 15px;
      border-radius: 5px;
      max-height: 500px;
      overflow: auto;
    }
    .results pre {
      white-space: pre-wrap;
      word-wrap: break-word;
      font-size: 13px;
      line-height: 1.5;
    }
    .error { color: #ff6b6b; }
    .success { color: #4ade80; }
    .warning { color: #fbbf24; }
    .count { color: #94a3b8; font-size: 14px; margin-bottom: 10px; }
    .help {
      background: #0f3460;
      padding: 15px;
      border-radius: 5px;
      margin-top: 20px;
      font-size: 14px;
    }
    .help h3 { color: #e94560; margin-bottom: 10px; }
    .help code {
      background: #1a1a2e;
      padding: 2px 6px;
      border-radius: 3px;
    }
    .login-container {
      max-width: 400px;
      margin: 50px auto;
      padding: 30px;
      background: #16213e;
      border-radius: 10px;
    }
    .login-container h2 {
      color: #e94560;
      margin-bottom: 20px;
      text-align: center;
    }
    .user-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      background: #0f3460;
      padding: 10px 15px;
      border-radius: 5px;
      margin-bottom: 15px;
      flex-wrap: wrap;
      gap: 10px;
    }
    .user-info {
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .user-email { color: #4ade80; font-weight: bold; }
    .user-roles {
      display: flex;
      gap: 5px;
    }
    .role-badge {
      padding: 2px 8px;
      background: #1a1a2e;
      border-radius: 3px;
      font-size: 11px;
      color: #94a3b8;
    }
    .role-badge.shell { background: #e94560; color: #fff; }
    .role-badge.admin { background: #3b82f6; color: #fff; }
  </style>
</head>
<body>
  <div id="app" class="container">
    <h1>🔍 UnifyWeaver CLI Search</h1>

    <!-- Login Form (shown when auth required and not logged in) -->
    <div v-if="authRequired && !user" class="login-container">
      <h2>Login Required</h2>
      <div class="form-group">
        <label>Email</label>
        <input v-model="loginEmail" type="email" placeholder="e.g., shell@local" @keyup.enter="doLogin">
      </div>
      <div class="form-group">
        <label>Password</label>
        <input v-model="loginPassword" type="password" placeholder="Password" @keyup.enter="doLogin">
      </div>
      <button @click="doLogin" :disabled="loading" style="width: 100%;">{{ loading ? 'Logging in...' : 'Login' }}</button>
      <p v-if="loginError" class="error" style="margin-top: 10px; text-align: center;">{{ loginError }}</p>
      <p style="margin-top: 15px; color: #94a3b8; font-size: 12px; text-align: center;">
        Default users: shell@local/shell, admin@local/admin, user@local/user
      </p>
    </div>

    <!-- Main App (shown when auth not required OR logged in) -->
    <template v-if="!authRequired || user">
      <!-- User Header -->
      <div v-if="user" class="user-header">
        <div class="user-info">
          <span class="user-email">{{ user.email }}</span>
          <div class="user-roles">
            <span v-for="role in user.roles" :key="role" class="role-badge" :class="role">{{ role }}</span>
          </div>
        </div>
        <button @click="doLogout" style="background: #16213e; padding: 5px 15px;">Logout</button>
      </div>

      <!-- Working Directory Bar -->
      <div style="background: #0f3460; padding: 10px 15px; border-radius: 5px; margin-bottom: 15px; display: flex; align-items: center; gap: 10px; flex-wrap: wrap;">
        <span style="color: #94a3b8;">Working directory:</span>
        <code style="background: #1a1a2e; padding: 4px 8px; border-radius: 3px; font-family: monospace;">{{ workingDir }}</code>
        <button v-if="workingDir !== '.'" @click="workingDir = '.'" style="background: #16213e; padding: 5px 10px; font-size: 12px;">Reset to root</button>
      </div>

      <div class="tabs">
      <button class="tab" :class="{ active: tab === 'browse' }" @click="tab = 'browse'; loadBrowse()">Browse</button>
      <button class="tab" :class="{ active: tab === 'grep' }" @click="tab = 'grep'">Grep</button>
      <button class="tab" :class="{ active: tab === 'find' }" @click="tab = 'find'">Find</button>
      <button class="tab" :class="{ active: tab === 'cat' }" @click="tab = 'cat'">Cat</button>
      <button class="tab" :class="{ active: tab === 'exec' }" @click="tab = 'exec'">Custom</button>
      <button class="tab" :class="{ active: tab === 'feedback' }" @click="tab = 'feedback'">Feedback</button>
      <button v-if="user && user.roles && user.roles.includes('shell')" class="tab" :class="{ active: tab === 'shell' }" @click="tab = 'shell'; focusShellInput()" style="background: #a855f7;">🔐 Shell</button>
    </div>

    <!-- Browse Panel -->
    <div class="panel" v-if="tab === 'browse'">
      <div style="margin-bottom: 15px;">
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 10px; flex-wrap: wrap;">
          <button v-if="browse.parent !== null" @click="navigateTo(browse.parent)" style="background: #0f3460; padding: 8px 12px;">⬆️ Up</button>
          <span style="color: #94a3b8; font-family: monospace;">📁 {{ browse.path || '.' }}</span>
          <button @click="workingDir = browse.path" :disabled="workingDir === browse.path" style="background: #4ade80; color: #000; padding: 6px 12px; font-size: 12px;">📌 Set as Working Dir</button>
        </div>
        <p class="count" v-if="browse.entries.length">{{ browse.entries.length }} items</p>
      </div>
      <div style="max-height: 400px; overflow-y: auto;">
        <div v-for="entry in browse.entries" :key="entry.name"
             @click="entry.type === 'directory' ? navigateTo(browse.path === '.' ? entry.name : browse.path + '/' + entry.name) : selectFile(browse.path === '.' ? entry.name : browse.path + '/' + entry.name)"
             style="padding: 10px; background: #0f3460; margin: 3px 0; border-radius: 5px; cursor: pointer; display: flex; justify-content: space-between; align-items: center;"
             :style="{ borderLeft: entry.type === 'directory' ? '3px solid #e94560' : '3px solid #3b82f6' }">
          <span>
            <span style="margin-right: 8px;">{{ entry.type === 'directory' ? '📁' : '📄' }}</span>
            {{ entry.name }}
          </span>
          <span style="color: #94a3b8; font-size: 12px;">{{ entry.size ? formatSize(entry.size) : '' }}</span>
        </div>
        <div v-if="browse.entries.length === 0 && !loading" style="color: #94a3b8; text-align: center; padding: 20px;">
          Empty directory
        </div>
      </div>
      <div style="margin-top: 15px; padding: 10px; background: #0f3460; border-radius: 5px;" v-if="browse.selected">
        <p style="color: #94a3b8; font-size: 12px; margin-bottom: 5px;">Selected file:</p>
        <p style="font-family: monospace;">{{ browse.selected }}</p>
        <div style="margin-top: 10px; display: flex; gap: 10px;">
          <button @click="cat.path = browse.selected; tab = 'cat'; doCat()">View Contents</button>
          <button @click="grep.path = browse.path; tab = 'grep'" style="background: #0f3460;">Search Here</button>
        </div>
      </div>
    </div>

    <!-- Grep Panel -->
    <div class="panel" v-if="tab === 'grep'">
      <div class="form-group">
        <label>Search Pattern (regex)</label>
        <input v-model="grep.pattern" placeholder="e.g., function.*export" @keyup.enter="doGrep">
      </div>
      <div class="form-group">
        <label>Path (relative to sandbox)</label>
        <input v-model="grep.path" placeholder="e.g., src/ or .">
      </div>
      <div class="form-group">
        <label>Options (space-separated)</label>
        <input v-model="grep.options" placeholder="e.g., -i --include=*.ts">
      </div>
      <button @click="doGrep" :disabled="loading">{{ loading ? 'Searching...' : 'Search' }}</button>
    </div>

    <!-- Find Panel -->
    <div class="panel" v-if="tab === 'find'">
      <div class="form-group">
        <label>File Pattern</label>
        <input v-model="find.pattern" placeholder="e.g., *.ts or index.*" @keyup.enter="doFind">
      </div>
      <div class="form-group">
        <label>Search Path</label>
        <input v-model="find.path" placeholder="e.g., src/ or .">
      </div>
      <div class="form-group">
        <label>Options (space-separated)</label>
        <input v-model="find.options" placeholder="e.g., -type f -maxdepth 3">
      </div>
      <button @click="doFind" :disabled="loading">{{ loading ? 'Finding...' : 'Find Files' }}</button>
    </div>

    <!-- Cat Panel -->
    <div class="panel" v-if="tab === 'cat'">
      <div class="form-group">
        <label>File Path</label>
        <input v-model="cat.path" placeholder="e.g., src/index.ts" @keyup.enter="doCat">
      </div>
      <button @click="doCat" :disabled="loading">{{ loading ? 'Reading...' : 'Read File' }}</button>
    </div>

    <!-- Custom Exec Panel -->
    <div class="panel" v-if="tab === 'exec'">
      <div class="form-group">
        <label>Command (as you'd type in shell)</label>
        <input v-model="exec.commandLine" placeholder="e.g., ls -la src/ or wc -l *.ts" @keyup.enter="doExec">
      </div>
      <button @click="doExec" :disabled="loading">{{ loading ? 'Running...' : 'Execute' }}</button>
      <p class="count" style="margin-top: 10px;">Allowed: cd, pwd, grep, find, cat, head, tail, ls, wc</p>
    </div>

    <!-- Feedback Panel -->
    <div class="panel" v-if="tab === 'feedback'">
      <div class="form-group">
        <label>Feedback Type</label>
        <select v-model="feedback.type" style="width: 100%; padding: 10px; background: #0f3460; border: 1px solid #1a1a2e; color: #eee; border-radius: 5px;">
          <option value="info">Info</option>
          <option value="success">Success</option>
          <option value="suggestion">Suggestion</option>
          <option value="warning">Warning</option>
          <option value="error">Error</option>
        </select>
      </div>
      <div class="form-group">
        <label>Message</label>
        <textarea v-model="feedback.message" rows="4" placeholder="Enter your feedback, notes, or observations..." @keyup.ctrl.enter="doFeedback"></textarea>
      </div>
      <div class="form-group">
        <label>Context (optional)</label>
        <input v-model="feedback.context" placeholder="e.g., search query, file path, or related info">
      </div>
      <div style="display: flex; gap: 10px;">
        <button @click="doFeedback" :disabled="loading || !feedback.message">{{ loading ? 'Submitting...' : 'Submit Feedback' }}</button>
        <button @click="loadFeedback" :disabled="loading" style="background: #0f3460;">{{ loading ? 'Loading...' : 'View History' }}</button>
      </div>
      <div v-if="feedbackHistory.length > 0" style="margin-top: 15px;">
        <p class="count">{{ feedbackHistory.length }} feedback entries</p>
        <div style="max-height: 300px; overflow-y: auto;">
          <div v-for="(entry, i) in feedbackHistory" :key="i" style="background: #0f3460; padding: 10px; margin: 5px 0; border-radius: 5px; border-left: 3px solid;" :style="{ borderColor: typeColor(entry.type) }">
            <div style="display: flex; justify-content: space-between; font-size: 12px; color: #94a3b8;">
              <span>{{ entry.type?.toUpperCase() || 'INFO' }}</span>
              <span>{{ formatTime(entry.timestamp) }}</span>
            </div>
            <div style="margin-top: 5px;">{{ entry.message }}</div>
            <div v-if="entry.context" style="font-size: 12px; color: #94a3b8; margin-top: 5px;">Context: {{ entry.context }}</div>
          </div>
        </div>
      </div>
    </div>

    <!-- Shell Panel (superadmin only) -->
    <div class="panel" v-if="tab === 'shell'" style="padding: 0;">
      <div style="background: #0a0a0a; border-radius: 5px; overflow: hidden;">
        <div style="background: #16213e; padding: 8px 12px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
          <span style="color: #a855f7; font-weight: bold;">🔐 Shell</span>
          <div style="display: flex; align-items: center; gap: 8px; flex-wrap: wrap;">
            <span v-if="shellConnected" style="color: #4ade80; font-size: 12px;">● Connected</span>
            <span v-else style="color: #ff6b6b; font-size: 12px;">● Disconnected</span>
            <button @click="shellTextMode = !shellTextMode" :style="{ background: shellTextMode ? '#a855f7' : '#0f3460' }" style="padding: 4px 10px; font-size: 12px;">
              {{ shellTextMode ? 'Capture' : 'Text Mode' }}
            </button>
            <button @click="clearShellOutput" style="padding: 4px 10px; font-size: 12px; background: #0f3460;">Clear</button>
            <button @click="connectShell" :disabled="shellConnected" style="padding: 4px 10px; font-size: 12px;">Connect</button>
            <button @click="disconnectShell" :disabled="!shellConnected" style="padding: 4px 10px; font-size: 12px;">Disconnect</button>
          </div>
        </div>
        <div style="position: relative;">
          <div id="shell-output" @click="focusCaptureInput" style="background: #0a0a0a; padding: 10px; height: 350px; overflow-y: auto; font-family: 'JetBrains Mono', 'Fira Code', monospace; font-size: 13px; white-space: pre-wrap; word-break: break-all; user-select: text; -webkit-user-select: text;">
            <span style="color: #94a3b8;">Click "Connect" to start a shell session.<br>Only users with the "shell" role can access this feature.</span>
          </div>
          <!-- Hidden input for capture mode (triggers mobile keyboard) -->
          <input
            v-if="!shellTextMode"
            id="shell-capture-input"
            @input="handleCaptureInput"
            @keydown="handleCaptureKeydown"
            type="text"
            autocomplete="off"
            autocapitalize="off"
            autocorrect="off"
            spellcheck="false"
            style="position: absolute; bottom: 10px; left: 10px; right: 10px; opacity: 0.01; height: 40px; font-size: 16px; background: transparent; border: none; color: transparent; caret-color: #4ade80;"
          />
        </div>
        <!-- Text Mode Input -->
        <div v-if="shellTextMode" style="display: flex; align-items: center; padding: 8px 12px; background: #16213e; border-top: 1px solid #0f3460; gap: 8px;">
          <span style="color: #4ade80; font-family: monospace; font-size: 13px;">$</span>
          <input
            id="shell-input"
            v-model="shellInput"
            @keydown.enter="sendShellCommand"
            type="text"
            placeholder="Type command and press Enter..."
            style="flex: 1; background: #0a0a0a; border: 1px solid #0f3460; border-radius: 4px; padding: 8px 12px; color: #eee; font-family: monospace; font-size: 16px;"
            autocomplete="off"
            autocapitalize="off"
            spellcheck="false"
          />
          <button @click="sendShellCommand" style="padding: 8px 16px; font-size: 13px;">Send</button>
        </div>
        <div v-else style="padding: 8px 12px; background: #16213e; border-top: 1px solid #0f3460; font-size: 12px; color: #94a3b8;">
          Capture mode: Tap the terminal area to open keyboard. Characters sent immediately.
        </div>
      </div>
    </div>

    <!-- Results -->
    <div class="results" v-if="result">
      <p v-if="result.warning" class="warning">⚠️ {{ result.warning }}</p>
      <p v-if="result.error" class="error">❌ {{ result.error }}</p>
      <p v-if="result.data?.count !== undefined" class="count">Found {{ result.data.count }} results</p>
      <pre v-if="result.data?.stdout">{{ result.data.stdout }}</pre>
      <pre v-if="result.data?.content">{{ result.data.content }}</pre>
      <pre v-if="result.data?.matches">{{ result.data.matches.join('\\n') }}</pre>
      <pre v-if="result.data?.files">{{ result.data.files.join('\\n') }}</pre>
    </div>

    <!-- Help -->
    <div class="help">
      <h3>Quick Help</h3>
      <p><strong>Working Dir:</strong> All commands run relative to the working directory. Use Browse to navigate and "Set as Working Dir", or use <code>cd</code> in Custom.</p>
      <p><strong>Browse:</strong> Navigate the file tree. Click folders to enter, files to select.</p>
      <p><strong>Grep:</strong> Search file contents with regex. Use <code>-i</code> for case-insensitive, <code>--include=*.ts</code> for file filters.</p>
      <p><strong>Find:</strong> Find files by name pattern. Use <code>-type f</code> for files only, <code>-maxdepth N</code> to limit depth.</p>
      <p><strong>Custom:</strong> Supports globs (<code>*.ts</code>, <code>*/</code>) and <code>cd</code> to change working directory.</p>
    </div>
    </template>
  </div>

  <script>
    const { createApp, ref, reactive } = Vue;

    createApp({
      setup() {
        const tab = ref('grep');
        const loading = ref(false);
        const result = ref(null);
        const workingDir = ref('.');

        // Auth state
        const authRequired = ref(false);
        const user = ref(null);
        const token = ref(localStorage.getItem('token') || '');
        const loginEmail = ref('');
        const loginPassword = ref('');
        const loginError = ref('');

        const grep = reactive({ pattern: '', path: '.', options: '' });
        const find = reactive({ pattern: '', path: '.', options: '' });
        const cat = reactive({ path: '' });
        const exec = reactive({ commandLine: '' });
        const feedback = reactive({ message: '', type: 'info', context: '' });
        const feedbackHistory = ref([]);
        const browse = reactive({ path: '.', entries: [], parent: null, selected: null });

        // Shell state
        const shellConnected = ref(false);
        const shellOutput = ref(null);
        const shellTextMode = ref(true);  // Default to text mode for mobile
        const shellInput = ref('');
        let shellWs = null;

        // Shell functions
        function connectShell() {
          if (shellWs) return;
          if (!token.value) {
            appendShellOutput('\\x1b[31mError: Not authenticated\\x1b[0m\\r\\n');
            return;
          }

          const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
          const wsUrl = wsProtocol + '//' + window.location.host + '/?token=' + encodeURIComponent(token.value);

          try {
            shellWs = new WebSocket(wsUrl);

            shellWs.onopen = () => {
              shellConnected.value = true;
              clearShellOutput();
              // Auto-focus input for mobile keyboard
              setTimeout(() => {
                const input = document.getElementById('shell-input');
                if (input) input.focus();
              }, 100);
            };

            shellWs.onmessage = (event) => {
              try {
                const msg = JSON.parse(event.data);
                if (msg.type === 'output') {
                  appendShellOutput(msg.data);
                } else if (msg.type === 'prompt') {
                  const cwd = msg.cwd || '~';
                  const shortCwd = cwd.length > 30 ? '...' + cwd.slice(-27) : cwd;
                  appendShellOutput('\\x1b[32m' + shortCwd + '\\x1b[0m $ ');
                } else if (msg.type === 'error') {
                  appendShellOutput('\\x1b[31mError: ' + msg.data + '\\x1b[0m\\r\\n');
                }
              } catch (e) {
                console.error('Shell message parse error:', e);
              }
            };

            shellWs.onclose = () => {
              shellConnected.value = false;
              shellWs = null;
              appendShellOutput('\\r\\n\\x1b[33m[Connection closed]\\x1b[0m\\r\\n');
            };

            shellWs.onerror = (err) => {
              console.error('Shell WebSocket error:', err);
              appendShellOutput('\\x1b[31mWebSocket error\\x1b[0m\\r\\n');
            };

            // Handle keyboard input
            document.addEventListener('keydown', handleShellKeyDown);
          } catch (err) {
            appendShellOutput('\\x1b[31mFailed to connect: ' + err.message + '\\x1b[0m\\r\\n');
          }
        }

        function disconnectShell() {
          if (shellWs) {
            shellWs.close();
            shellWs = null;
          }
          shellConnected.value = false;
          document.removeEventListener('keydown', handleShellKeyDown);
        }

        function handleShellKeyDown(e) {
          if (tab.value !== 'shell' || !shellConnected.value || !shellWs) return;

          // Ignore if typing in input field
          if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

          e.preventDefault();

          let char = '';
          if (e.key === 'Enter') char = '\\r';
          else if (e.key === 'Backspace') char = '\\x7f';
          else if (e.key === 'c' && e.ctrlKey) char = '\\x03';
          else if (e.key.length === 1) char = e.key;
          else return;

          shellWs.send(JSON.stringify({ type: 'input', data: char }));
        }

        function appendShellOutput(text) {
          const el = document.getElementById('shell-output');
          if (!el) return;
          // Simple ANSI to HTML conversion
          const html = text
            .replace(/\\x1b\\[31m/g, '<span style="color: #ff6b6b;">')
            .replace(/\\x1b\\[32m/g, '<span style="color: #4ade80;">')
            .replace(/\\x1b\\[33m/g, '<span style="color: #fbbf24;">')
            .replace(/\\x1b\\[0m/g, '</span>')
            .replace(/\\r\\n/g, '<br>')
            .replace(/\\r/g, '')
            .replace(/\\n/g, '<br>');
          el.innerHTML += html;
          el.scrollTop = el.scrollHeight;
        }

        function clearShellOutput() {
          const el = document.getElementById('shell-output');
          if (el) el.innerHTML = '';
        }

        function sendShellCommand() {
          const cmd = shellInput.value.trim();
          if (!cmd || !shellWs || !shellConnected.value) return;

          // Send command character by character then enter
          for (const char of cmd) {
            shellWs.send(JSON.stringify({ type: 'input', data: char }));
          }
          shellWs.send(JSON.stringify({ type: 'input', data: '\\r' }));
          shellInput.value = '';
        }

        function focusShellInput() {
          setTimeout(() => {
            const input = document.getElementById('shell-input');
            if (input) input.focus();
          }, 100);
        }

        function focusCaptureInput() {
          if (shellTextMode.value) return;
          const input = document.getElementById('shell-capture-input');
          if (input) input.focus();
        }

        function handleCaptureInput(e) {
          if (!shellWs || !shellConnected.value) return;
          const value = e.target.value;
          if (value) {
            // Send each character
            for (const char of value) {
              shellWs.send(JSON.stringify({ type: 'input', data: char }));
            }
            e.target.value = '';
          }
        }

        function handleCaptureKeydown(e) {
          if (!shellWs || !shellConnected.value) return;

          if (e.key === 'Enter') {
            e.preventDefault();
            shellWs.send(JSON.stringify({ type: 'input', data: '\\r' }));
          } else if (e.key === 'Backspace') {
            e.preventDefault();
            shellWs.send(JSON.stringify({ type: 'input', data: '\\x7f' }));
          }
        }

        // Auth helper
        function getAuthHeaders() {
          const headers = { 'Content-Type': 'application/json' };
          if (token.value) {
            headers['Authorization'] = 'Bearer ' + token.value;
          }
          return headers;
        }

        // Check auth status on mount
        async function checkAuthStatus() {
          try {
            const res = await fetch('/auth/status');
            const data = await res.json();
            authRequired.value = data.data?.authRequired || false;

            // If we have a token, verify it
            if (token.value) {
              const meRes = await fetch('/auth/me', { headers: getAuthHeaders() });
              const meData = await meRes.json();
              if (meData.success && meData.user) {
                user.value = meData.user;
              } else {
                // Token invalid, clear it
                token.value = '';
                localStorage.removeItem('token');
              }
            }
          } catch (err) {
            console.error('Auth check error:', err);
          }
        }

        async function doLogin() {
          if (!loginEmail.value || !loginPassword.value) return;
          loading.value = true;
          loginError.value = '';
          try {
            const res = await fetch('/auth/login', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ email: loginEmail.value, password: loginPassword.value })
            });
            const data = await res.json();
            if (data.success && data.data?.token) {
              token.value = data.data.token;
              localStorage.setItem('token', data.data.token);
              user.value = data.data.user;
              loginEmail.value = '';
              loginPassword.value = '';
            } else {
              loginError.value = data.error || 'Login failed';
            }
          } catch (err) {
            loginError.value = err.message;
          }
          loading.value = false;
        }

        function doLogout() {
          token.value = '';
          user.value = null;
          localStorage.removeItem('token');
        }

        // Check auth on mount
        checkAuthStatus();

        async function loadBrowse(path = '.') {
          loading.value = true;
          try {
            const res = await fetch('/browse', {
              method: 'POST',
              headers: getAuthHeaders(),
              body: JSON.stringify({ path })
            });
            const data = await res.json();
            if (data.success) {
              browse.path = data.data.path;
              browse.entries = data.data.entries;
              browse.parent = data.data.parent;
              browse.selected = null;
            }
          } catch (err) {
            console.error('Browse error:', err);
          }
          loading.value = false;
        }

        function navigateTo(path) {
          loadBrowse(path);
        }

        function selectFile(filePath) {
          browse.selected = filePath;
        }

        function formatSize(bytes) {
          if (bytes < 1024) return bytes + ' B';
          if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
          return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
        }

        async function doGrep() {
          loading.value = true;
          result.value = null;
          try {
            const res = await fetch('/grep', {
              method: 'POST',
              headers: getAuthHeaders(),
              body: JSON.stringify({
                pattern: grep.pattern,
                path: grep.path,
                options: grep.options.split(/\\s+/).filter(Boolean),
                cwd: workingDir.value
              })
            });
            result.value = await res.json();
          } catch (err) {
            result.value = { error: err.message };
          }
          loading.value = false;
        }

        async function doFind() {
          loading.value = true;
          result.value = null;
          try {
            const res = await fetch('/find', {
              method: 'POST',
              headers: getAuthHeaders(),
              body: JSON.stringify({
                pattern: find.pattern,
                path: find.path,
                options: find.options.split(/\\s+/).filter(Boolean),
                cwd: workingDir.value
              })
            });
            result.value = await res.json();
          } catch (err) {
            result.value = { error: err.message };
          }
          loading.value = false;
        }

        async function doCat() {
          loading.value = true;
          result.value = null;
          try {
            const res = await fetch('/cat', {
              method: 'POST',
              headers: getAuthHeaders(),
              body: JSON.stringify({ path: cat.path, cwd: workingDir.value })
            });
            result.value = await res.json();
          } catch (err) {
            result.value = { error: err.message };
          }
          loading.value = false;
        }

        async function doExec() {
          loading.value = true;
          result.value = null;
          try {
            // Parse command line: split on whitespace, first part is command, rest are args
            const parts = exec.commandLine.trim().split(/\\s+/);
            const command = parts[0] || '';
            const args = parts.slice(1);

            // Handle cd command client-side
            if (command === 'cd') {
              const targetDir = args[0] || '.';
              // Validate via browse endpoint
              const browseRes = await fetch('/browse', {
                method: 'POST',
                headers: getAuthHeaders(),
                body: JSON.stringify({ path: workingDir.value === '.' ? targetDir : workingDir.value + '/' + targetDir })
              });
              const browseData = await browseRes.json();
              if (browseData.success) {
                workingDir.value = browseData.data.path;
                result.value = { success: true, data: { stdout: 'Changed to: ' + browseData.data.path } };
              } else {
                result.value = { success: false, error: browseData.error || 'Directory not found' };
              }
              loading.value = false;
              return;
            }

            const res = await fetch('/exec', {
              method: 'POST',
              headers: getAuthHeaders(),
              body: JSON.stringify({ command, args, cwd: workingDir.value })
            });
            result.value = await res.json();
          } catch (err) {
            result.value = { error: err.message };
          }
          loading.value = false;
        }

        async function doFeedback() {
          if (!feedback.message) return;
          loading.value = true;
          try {
            const res = await fetch('/feedback', {
              method: 'POST',
              headers: getAuthHeaders(),
              body: JSON.stringify({
                message: feedback.message,
                type: feedback.type,
                context: feedback.context || undefined
              })
            });
            const data = await res.json();
            if (data.success) {
              feedback.message = '';
              feedback.context = '';
              await loadFeedback();
            }
          } catch (err) {
            console.error('Feedback error:', err);
          }
          loading.value = false;
        }

        async function loadFeedback() {
          loading.value = true;
          try {
            const res = await fetch('/feedback', { headers: getAuthHeaders() });
            const data = await res.json();
            if (data.success) {
              feedbackHistory.value = (data.data.entries || []).reverse();
            }
          } catch (err) {
            console.error('Load feedback error:', err);
          }
          loading.value = false;
        }

        function typeColor(type) {
          const colors = {
            info: '#3b82f6',
            success: '#4ade80',
            suggestion: '#a855f7',
            warning: '#fbbf24',
            error: '#ff6b6b'
          };
          return colors[type] || colors.info;
        }

        function formatTime(timestamp) {
          if (!timestamp) return '';
          const d = new Date(timestamp);
          return d.toLocaleString();
        }

        return { tab, loading, result, workingDir, authRequired, user, loginEmail, loginPassword, loginError, doLogin, doLogout, shellConnected, shellOutput, shellTextMode, shellInput, connectShell, disconnectShell, sendShellCommand, clearShellOutput, focusShellInput, focusCaptureInput, handleCaptureInput, handleCaptureKeydown, grep, find, cat, exec, feedback, feedbackHistory, browse, doGrep, doFind, doCat, doExec, doFeedback, loadFeedback, loadBrowse, navigateTo, selectFile, formatSize, typeColor, formatTime };
      }
    }).mount('#app');
  </script>
</body>
</html>`;
}

// ============================================================================
// Router
// ============================================================================

async function handleRequest(req: http.IncomingMessage, res: http.ServerResponse): Promise<void> {
  const parsedUrl = url.parse(req.url || '/', true);
  const pathname = parsedUrl.pathname || '/';
  const method = req.method || 'GET';

  // Handle CORS preflight
  if (method === 'OPTIONS') {
    res.writeHead(204, {
      'Access-Control-Allow-Origin': ALLOWED_ORIGINS[0],
      'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type, Authorization'
    });
    res.end();
    return;
  }

  // Log request
  console.log(`[${new Date().toISOString()}] ${method} ${pathname}`);

  try {
    // Route requests

    // Public routes (no auth required)
    if (pathname === '/' && method === 'GET') {
      sendHTML(res, getHTMLInterface());
    } else if (pathname === '/health' && method === 'GET') {
      await handleHealth(req, res);
    } else if (pathname === '/auth/status' && method === 'GET') {
      await handleAuthStatus(req, res);
    } else if (pathname === '/auth/login' && method === 'POST') {
      await handleLogin(req, res);
    } else if (pathname === '/auth/me' && method === 'GET') {
      await handleMe(req, res);

    // Protected routes (auth required when AUTH_REQUIRED=true)
    } else if (pathname === '/commands' && method === 'GET') {
      if (!requireAuth(req, res)) return;
      await handleCommands(req, res);
    } else if (pathname === '/exec' && method === 'POST') {
      const user = requireAuth(req, res);
      if (!user) return;
      // /exec requires admin or shell role
      if (!requireRole(user, res, 'admin', 'shell')) return;
      await handleExec(req, res);
    } else if (pathname === '/grep' && method === 'POST') {
      if (!requireAuth(req, res)) return;
      await handleGrep(req, res);
    } else if (pathname === '/find' && method === 'POST') {
      if (!requireAuth(req, res)) return;
      await handleFind(req, res);
    } else if (pathname === '/cat' && method === 'POST') {
      if (!requireAuth(req, res)) return;
      await handleCat(req, res);
    } else if (pathname === '/feedback' && method === 'POST') {
      if (!requireAuth(req, res)) return;
      await handleFeedback(req, res);
    } else if (pathname === '/feedback' && method === 'GET') {
      if (!requireAuth(req, res)) return;
      await handleGetFeedback(req, res);
    } else if (pathname === '/browse' && method === 'POST') {
      if (!requireAuth(req, res)) return;
      await handleBrowse(req, res);
    } else {
      sendError(res, 'Not found', 404);
    }
  } catch (err) {
    console.error('Request error:', err);
    sendError(res, 'Internal server error', 500);
  }
}

// ============================================================================
// Server
// ============================================================================

// ============================================================================
// WebSocket Shell Session
// ============================================================================

interface ShellSession {
  ws: WebSocket;
  user: TokenPayload;
  process: ChildProcessWithoutNullStreams | null;
  currentDir: string;
  inputBuffer: string;
}

function handleShellConnection(ws: WebSocket, user: TokenPayload): void {
  const session: ShellSession = {
    ws,
    user,
    process: null,
    currentDir: SANDBOX_ROOT,
    inputBuffer: ''
  };

  // Send welcome message
  const welcome = `\r\nConnected to UnifyWeaver Shell\r\n` +
    `User: ${user.email} [${user.roles.join(', ')}]\r\n` +
    `Working directory: ${SANDBOX_ROOT}\r\n` +
    `Type "help" for commands, "exit" to disconnect\r\n\r\n`;
  ws.send(JSON.stringify({ type: 'output', data: welcome }));
  ws.send(JSON.stringify({ type: 'prompt', cwd: session.currentDir }));

  ws.on('message', (data: Buffer | string) => {
    try {
      const msg = JSON.parse(data.toString());

      if (msg.type === 'input') {
        const char = msg.data;

        // Handle Enter key
        if (char === '\r' || char === '\n') {
          ws.send(JSON.stringify({ type: 'output', data: '\r\n' }));
          const command = session.inputBuffer.trim();
          session.inputBuffer = '';

          if (command) {
            executeShellCommand(session, command);
          } else {
            ws.send(JSON.stringify({ type: 'prompt', cwd: session.currentDir }));
          }
        }
        // Handle Backspace
        else if (char === '\x7f' || char === '\b') {
          if (session.inputBuffer.length > 0) {
            session.inputBuffer = session.inputBuffer.slice(0, -1);
            ws.send(JSON.stringify({ type: 'output', data: '\b \b' }));
          }
        }
        // Handle Ctrl+C
        else if (char === '\x03') {
          if (session.process) {
            session.process.kill('SIGINT');
            session.process = null;
          }
          ws.send(JSON.stringify({ type: 'output', data: '^C\r\n' }));
          session.inputBuffer = '';
          ws.send(JSON.stringify({ type: 'prompt', cwd: session.currentDir }));
        }
        // Handle printable characters
        else if (char >= ' ' && char <= '~') {
          session.inputBuffer += char;
          ws.send(JSON.stringify({ type: 'output', data: char }));
        }
      }
    } catch (err) {
      console.error('Shell message error:', err);
    }
  });

  ws.on('close', () => {
    if (session.process) {
      session.process.kill();
      session.process = null;
    }
    console.log(`Shell session closed for ${user.email}`);
  });
}

function executeShellCommand(session: ShellSession, command: string): void {
  const { ws } = session;

  // Built-in: help
  if (command === 'help') {
    const helpText = `\r\nUnifyWeaver Shell - Built-in Commands:\r\n` +
      `  help    - Show this help\r\n` +
      `  cd DIR  - Change directory\r\n` +
      `  pwd     - Print working directory\r\n` +
      `  exit    - Disconnect\r\n` +
      `\r\nYou can run any shell command. Output is streamed in real-time.\r\n\r\n`;
    ws.send(JSON.stringify({ type: 'output', data: helpText }));
    ws.send(JSON.stringify({ type: 'prompt', cwd: session.currentDir }));
    return;
  }

  // Built-in: exit
  if (command === 'exit' || command === 'quit') {
    ws.send(JSON.stringify({ type: 'output', data: 'Goodbye!\r\n' }));
    ws.close();
    return;
  }

  // Built-in: cd
  if (command.startsWith('cd ') || command === 'cd') {
    const targetDir = command.slice(3).trim() || SANDBOX_ROOT;
    let newDir: string;

    if (path.isAbsolute(targetDir)) {
      newDir = targetDir;
    } else if (targetDir === '~') {
      newDir = SANDBOX_ROOT;
    } else if (targetDir.startsWith('~/')) {
      newDir = path.join(SANDBOX_ROOT, targetDir.slice(2));
    } else {
      newDir = path.resolve(session.currentDir, targetDir);
    }

    // Security: Ensure within sandbox
    if (!newDir.startsWith(SANDBOX_ROOT)) {
      ws.send(JSON.stringify({ type: 'output', data: `\x1b[31mcd: Access denied - outside sandbox\x1b[0m\r\n` }));
      ws.send(JSON.stringify({ type: 'prompt', cwd: session.currentDir }));
      return;
    }

    if (fs.existsSync(newDir) && fs.statSync(newDir).isDirectory()) {
      session.currentDir = newDir;
    } else {
      ws.send(JSON.stringify({ type: 'output', data: `\x1b[31mcd: ${targetDir}: No such directory\x1b[0m\r\n` }));
    }
    ws.send(JSON.stringify({ type: 'prompt', cwd: session.currentDir }));
    return;
  }

  // Built-in: pwd
  if (command === 'pwd') {
    ws.send(JSON.stringify({ type: 'output', data: session.currentDir + '\r\n' }));
    ws.send(JSON.stringify({ type: 'prompt', cwd: session.currentDir }));
    return;
  }

  // Execute shell command
  const proc = spawn('sh', ['-c', command], {
    cwd: session.currentDir,
    env: { ...process.env, HOME: SANDBOX_ROOT, TERM: 'xterm-256color' }
  });

  session.process = proc;

  proc.stdout.on('data', (data: Buffer) => {
    ws.send(JSON.stringify({ type: 'output', data: data.toString().replace(/\n/g, '\r\n') }));
  });

  proc.stderr.on('data', (data: Buffer) => {
    ws.send(JSON.stringify({ type: 'output', data: `\x1b[31m${data.toString().replace(/\n/g, '\r\n')}\x1b[0m` }));
  });

  proc.on('close', (code: number) => {
    session.process = null;
    if (code !== 0) {
      ws.send(JSON.stringify({ type: 'output', data: `\x1b[33m[exit code: ${code}]\x1b[0m\r\n` }));
    }
    ws.send(JSON.stringify({ type: 'prompt', cwd: session.currentDir }));
  });

  proc.on('error', (err: Error) => {
    session.process = null;
    ws.send(JSON.stringify({ type: 'output', data: `\x1b[31mError: ${err.message}\x1b[0m\r\n` }));
    ws.send(JSON.stringify({ type: 'prompt', cwd: session.currentDir }));
  });
}

interface SSLOptions {
  cert?: string;
  key?: string;
}

function startServer(port: number, ssl?: SSLOptions): void {
  let server: http.Server | https.Server;
  let protocol: string;

  if (ssl?.cert && ssl?.key) {
    // HTTPS mode
    try {
      const sslOptions = {
        cert: fs.readFileSync(ssl.cert),
        key: fs.readFileSync(ssl.key)
      };
      server = https.createServer(sslOptions, handleRequest);
      protocol = 'https';
    } catch (err) {
      console.error(`Failed to load SSL certificates: ${err}`);
      console.error(`  cert: ${ssl.cert}`);
      console.error(`  key: ${ssl.key}`);
      process.exit(1);
    }
  } else {
    // HTTP mode
    server = http.createServer(handleRequest);
    protocol = 'http';
  }

  // Create WebSocket server
  const wss = new WebSocketServer({ server });

  wss.on('connection', (ws: WebSocket, req: http.IncomingMessage) => {
    // Extract token from URL query string
    const urlParsed = url.parse(req.url || '', true);
    const token = urlParsed.query.token as string;

    if (!token) {
      ws.send(JSON.stringify({ type: 'error', data: 'Authentication required' }));
      ws.close(1008, 'Authentication required');
      return;
    }

    const payload = verifyToken(token);
    if (!payload) {
      ws.send(JSON.stringify({ type: 'error', data: 'Invalid token' }));
      ws.close(1008, 'Invalid token');
      return;
    }

    // Check for shell role
    if (!payload.roles.includes('shell')) {
      ws.send(JSON.stringify({ type: 'error', data: 'Shell access requires "shell" role' }));
      ws.close(1008, 'Access denied');
      return;
    }

    console.log(`Shell session started for ${payload.email}`);
    handleShellConnection(ws, payload);
  });

  server.listen(port, () => {
    const authStatus = AUTH_REQUIRED ? 'ENABLED' : 'disabled';
    const wsProtocol = protocol === 'https' ? 'wss' : 'ws';
    const serverUrl = `${protocol}://localhost:${port}`;
    console.log(`
╔════════════════════════════════════════════════════════════╗
║           UnifyWeaver HTTP CLI Server                      ║
╠════════════════════════════════════════════════════════════╣
║  Server running at: ${serverUrl.padEnd(37)}║
║  Sandbox root:      ${SANDBOX_ROOT.slice(0, 35).padEnd(36)}║
║  Authentication:    ${authStatus.padEnd(36)}║
║  SSL/TLS:           ${(protocol === 'https' ? 'ENABLED' : 'disabled').padEnd(36)}║
║                                                            ║
║  Auth Endpoints:                                           ║
║    GET  /auth/status - Auth status                         ║
║    POST /auth/login  - Login (email, password)             ║
║    GET  /auth/me     - Current user info                   ║
║                                                            ║
║  Protected Endpoints:                                      ║
║    GET  /           - HTML interface                       ║
║    POST /browse     - Browse directories                   ║
║    POST /grep       - Search contents                      ║
║    POST /find       - Find files                           ║
║    POST /cat        - Read file                            ║
║    POST /exec       - Execute command                      ║
║    POST /feedback   - Submit feedback                      ║
║                                                            ║
║  WebSocket Shell:                                          ║
║    ${wsProtocol}://localhost:${port}/shell?token=JWT (shell role)       ║
╚════════════════════════════════════════════════════════════╝
`);
    if (!AUTH_REQUIRED) {
      console.log('Auth disabled. Set AUTH_REQUIRED=true to enable.');
      console.log('Default users: shell@local/shell, admin@local/admin, user@local/user\n');
    }
    if (protocol === 'https') {
      console.log(`SSL certificates loaded from --cert and --key options.\n`);
    }
  });

  server.on('error', (err: NodeJS.ErrnoException) => {
    if (err.code === 'EADDRINUSE') {
      console.error(`Port ${port} is already in use`);
    } else {
      console.error('Server error:', err);
    }
    process.exit(1);
  });
}

// ============================================================================
// CLI Entry
// ============================================================================

function main(): void {
  const args = process.argv.slice(2);
  let port = DEFAULT_PORT;
  let certPath: string | undefined;
  let keyPath: string | undefined;

  // Parse --port argument
  const portIdx = args.indexOf('--port');
  if (portIdx !== -1 && args[portIdx + 1]) {
    port = parseInt(args[portIdx + 1], 10);
    if (isNaN(port)) {
      console.error('Invalid port number');
      process.exit(1);
    }
  }

  // Parse --cert argument
  const certIdx = args.indexOf('--cert');
  if (certIdx !== -1 && args[certIdx + 1]) {
    certPath = args[certIdx + 1];
  }

  // Parse --key argument
  const keyIdx = args.indexOf('--key');
  if (keyIdx !== -1 && args[keyIdx + 1]) {
    keyPath = args[keyIdx + 1];
  }

  // Validate SSL options - both or neither required
  if ((certPath && !keyPath) || (!certPath && keyPath)) {
    console.error('Both --cert and --key are required for HTTPS');
    process.exit(1);
  }

  // Help
  if (args.includes('--help') || args.includes('-h')) {
    console.log(`
UnifyWeaver HTTP CLI Server

Usage:
  ts-node http-server.ts [options]

Options:
  --port <number>   Port to listen on (default: ${DEFAULT_PORT})
  --cert <path>     Path to SSL certificate file (enables HTTPS)
  --key <path>      Path to SSL private key file (enables HTTPS)
  --help, -h        Show this help

Environment:
  SANDBOX_ROOT      Root directory for sandbox operations
  AUTH_REQUIRED     Set to 'true' to require authentication

Examples:
  # HTTP mode (development)
  ts-node http-server.ts
  ts-node http-server.ts --port 8080

  # HTTPS mode (production)
  ts-node http-server.ts --cert server.crt --key server.key
  ts-node http-server.ts --port 443 --cert /etc/ssl/cert.pem --key /etc/ssl/key.pem

  # Generate self-signed certs for development:
  openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes
`);
    process.exit(0);
  }

  const ssl = certPath && keyPath ? { cert: certPath, key: keyPath } : undefined;
  startServer(port, ssl);
}

main();
