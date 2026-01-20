#!/usr/bin/env ts-node
/**
 * HTTP CLI Server
 *
 * Exposes shell search commands (grep, find, cat, rg) via HTTP endpoints
 * for use with AI browsers like Comet (Perplexity).
 *
 * Uses command-proxy.ts for validation and execution.
 *
 * Usage:
 *   ts-node http-server.ts
 *   ts-node http-server.ts --port 3001
 *
 * Endpoints:
 *   GET  /health              - Health check
 *   GET  /commands            - List available commands
 *   POST /exec                - Execute a command
 *   POST /grep                - Search file contents
 *   POST /find                - Find files by pattern
 *   POST /cat                 - Read file contents
 *   GET  /                    - Serve HTML interface
 *
 * @module unifyweaver/shell/http-server
 */

import * as http from 'http';
import * as url from 'url';
import * as path from 'path';
import * as fs from 'fs';
import {
  execute,
  validateCommand,
  listCommands,
  Risk,
  SANDBOX_ROOT
} from './command-proxy';

// ============================================================================
// Configuration
// ============================================================================

const DEFAULT_PORT = 3001;
const MAX_BODY_SIZE = 1024 * 1024; // 1MB
const ALLOWED_ORIGINS = ['*']; // Configure for production

// Commands specifically allowed for search operations
const SEARCH_COMMANDS = ['grep', 'find', 'cat', 'head', 'tail', 'ls', 'wc'];

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
}

interface FeedbackRequest {
  message: string;
  type?: 'info' | 'success' | 'warning' | 'error' | 'suggestion';
  context?: string;
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
    'Access-Control-Allow-Headers': 'Content-Type'
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
    const { command, args = [] } = JSON.parse(body) as SearchRequest;

    if (!command) {
      return sendError(res, 'Missing command');
    }

    // Only allow search commands
    if (!SEARCH_COMMANDS.includes(command)) {
      return sendError(res, `Command not allowed. Allowed: ${SEARCH_COMMANDS.join(', ')}`, 403);
    }

    // Validate first
    const validation = validateCommand(command, args, { role: 'user' });
    if (!validation.ok) {
      return sendError(res, validation.reason || 'Validation failed', 403);
    }

    // Execute
    const cmdString = [command, ...args].join(' ');
    const result = await execute(cmdString, { role: 'user', cwd: SANDBOX_ROOT });

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
    const { pattern, path: searchPath = '.', options = [] } = JSON.parse(body) as SearchRequest;

    if (!pattern) {
      return sendError(res, 'Missing pattern');
    }

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
    const result = await execute(cmdString, { role: 'user', cwd: SANDBOX_ROOT });

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
    const { pattern, path: searchPath = '.', options = [] } = JSON.parse(body) as SearchRequest;

    // Build find command
    const args = [searchPath];

    if (pattern) {
      args.push('-name', pattern);
    }

    args.push(...options);

    const cmdString = ['find', ...args].join(' ');
    const result = await execute(cmdString, { role: 'user', cwd: SANDBOX_ROOT });

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
    const { path: filePath, options = [] } = JSON.parse(body) as SearchRequest;

    if (!filePath) {
      return sendError(res, 'Missing path');
    }

    const args = [...options, filePath];
    const cmdString = ['cat', ...args].join(' ');
    const result = await execute(cmdString, { role: 'user', cwd: SANDBOX_ROOT });

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
  </style>
</head>
<body>
  <div id="app" class="container">
    <h1>🔍 UnifyWeaver CLI Search</h1>

    <div class="tabs">
      <button class="tab" :class="{ active: tab === 'grep' }" @click="tab = 'grep'">Grep</button>
      <button class="tab" :class="{ active: tab === 'find' }" @click="tab = 'find'">Find</button>
      <button class="tab" :class="{ active: tab === 'cat' }" @click="tab = 'cat'">Cat</button>
      <button class="tab" :class="{ active: tab === 'exec' }" @click="tab = 'exec'">Custom</button>
      <button class="tab" :class="{ active: tab === 'feedback' }" @click="tab = 'feedback'">Feedback</button>
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
        <label>Command</label>
        <input v-model="exec.command" placeholder="e.g., ls, grep, find, cat, head, tail, wc" @keyup.enter="doExec">
      </div>
      <div class="form-group">
        <label>Arguments (JSON array)</label>
        <input v-model="exec.args" placeholder='e.g., ["-la", "src/"]'>
      </div>
      <button @click="doExec" :disabled="loading">{{ loading ? 'Running...' : 'Execute' }}</button>
      <p class="count" style="margin-top: 10px;">Allowed: grep, find, cat, head, tail, ls, wc</p>
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
      <p><strong>Grep:</strong> Search file contents with regex. Use <code>-i</code> for case-insensitive, <code>--include=*.ts</code> for file filters.</p>
      <p><strong>Find:</strong> Find files by name pattern. Use <code>-type f</code> for files only, <code>-maxdepth N</code> to limit depth.</p>
      <p><strong>Cat:</strong> Read entire file contents.</p>
      <p><strong>Sandbox:</strong> All operations are restricted to the sandbox directory.</p>
    </div>
  </div>

  <script>
    const { createApp, ref, reactive } = Vue;

    createApp({
      setup() {
        const tab = ref('grep');
        const loading = ref(false);
        const result = ref(null);

        const grep = reactive({ pattern: '', path: '.', options: '' });
        const find = reactive({ pattern: '', path: '.', options: '' });
        const cat = reactive({ path: '' });
        const exec = reactive({ command: '', args: '[]' });
        const feedback = reactive({ message: '', type: 'info', context: '' });
        const feedbackHistory = ref([]);

        async function doGrep() {
          loading.value = true;
          result.value = null;
          try {
            const res = await fetch('/grep', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                pattern: grep.pattern,
                path: grep.path,
                options: grep.options.split(/\\s+/).filter(Boolean)
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
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                pattern: find.pattern,
                path: find.path,
                options: find.options.split(/\\s+/).filter(Boolean)
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
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ path: cat.path })
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
            const res = await fetch('/exec', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                command: exec.command,
                args: JSON.parse(exec.args || '[]')
              })
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
              headers: { 'Content-Type': 'application/json' },
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
            const res = await fetch('/feedback');
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

        return { tab, loading, result, grep, find, cat, exec, feedback, feedbackHistory, doGrep, doFind, doCat, doExec, doFeedback, loadFeedback, typeColor, formatTime };
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
      'Access-Control-Allow-Headers': 'Content-Type'
    });
    res.end();
    return;
  }

  // Log request
  console.log(`[${new Date().toISOString()}] ${method} ${pathname}`);

  try {
    // Route requests
    if (pathname === '/' && method === 'GET') {
      sendHTML(res, getHTMLInterface());
    } else if (pathname === '/health' && method === 'GET') {
      await handleHealth(req, res);
    } else if (pathname === '/commands' && method === 'GET') {
      await handleCommands(req, res);
    } else if (pathname === '/exec' && method === 'POST') {
      await handleExec(req, res);
    } else if (pathname === '/grep' && method === 'POST') {
      await handleGrep(req, res);
    } else if (pathname === '/find' && method === 'POST') {
      await handleFind(req, res);
    } else if (pathname === '/cat' && method === 'POST') {
      await handleCat(req, res);
    } else if (pathname === '/feedback' && method === 'POST') {
      await handleFeedback(req, res);
    } else if (pathname === '/feedback' && method === 'GET') {
      await handleGetFeedback(req, res);
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

function startServer(port: number): void {
  const server = http.createServer(handleRequest);

  server.listen(port, () => {
    console.log(`
╔════════════════════════════════════════════════════════════╗
║           UnifyWeaver HTTP CLI Server                      ║
╠════════════════════════════════════════════════════════════╣
║  Server running at: http://localhost:${port.toString().padEnd(24)}║
║  Sandbox root:      ${SANDBOX_ROOT.slice(0, 35).padEnd(36)}║
║                                                            ║
║  Endpoints:                                                ║
║    GET  /           - HTML interface                       ║
║    GET  /health     - Health check                         ║
║    GET  /commands   - List commands                        ║
║    POST /grep       - Search contents                      ║
║    POST /find       - Find files                           ║
║    POST /cat        - Read file                            ║
║    POST /exec       - Execute command                      ║
║    POST /feedback   - Submit feedback                      ║
║    GET  /feedback   - Read feedback log                    ║
╚════════════════════════════════════════════════════════════╝
`);
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

  // Parse --port argument
  const portIdx = args.indexOf('--port');
  if (portIdx !== -1 && args[portIdx + 1]) {
    port = parseInt(args[portIdx + 1], 10);
    if (isNaN(port)) {
      console.error('Invalid port number');
      process.exit(1);
    }
  }

  // Help
  if (args.includes('--help') || args.includes('-h')) {
    console.log(`
UnifyWeaver HTTP CLI Server

Usage:
  ts-node http-server.ts [options]

Options:
  --port <number>   Port to listen on (default: ${DEFAULT_PORT})
  --help, -h        Show this help

Environment:
  SANDBOX_ROOT      Root directory for sandbox operations

Examples:
  ts-node http-server.ts
  ts-node http-server.ts --port 8080
  SANDBOX_ROOT=/tmp/sandbox ts-node http-server.ts
`);
    process.exit(0);
  }

  startServer(port);
}

main();
