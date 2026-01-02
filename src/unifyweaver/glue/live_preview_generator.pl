% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% Live Preview Generator - Development Server and Hot-Reload
%
% This module provides declarative specifications for generating
% a development server with hot-reload capabilities for visualization
% prototyping.
%
% Usage:
%   % Configure dev server
%   dev_server_config(my_project, [
%       port(3000),
%       hot_reload(true),
%       watch_paths(['src/**/*.pl', 'src/**/*.ts'])
%   ]).
%
%   % Generate preview component
%   ?- generate_preview_app(my_chart, App).

:- module(live_preview_generator, [
    % Server configuration
    dev_server_config/2,            % dev_server_config(+Project, +Options)
    preview_config/2,               % preview_config(+Component, +Options)

    % Generation predicates
    generate_dev_server/2,          % generate_dev_server(+Project, -ServerCode)
    generate_vite_config/2,         % generate_vite_config(+Project, -Config)
    generate_preview_app/2,         % generate_preview_app(+Component, -App)
    generate_preview_wrapper/2,     % generate_preview_wrapper(+Component, -Wrapper)
    generate_hot_reload_hook/1,     % generate_hot_reload_hook(-Hook)
    generate_state_sync_hook/1,     % generate_state_sync_hook(-Hook)
    generate_preview_panel/2,       % generate_preview_panel(+Component, -Panel)
    generate_code_editor/2,         % generate_code_editor(+Component, -Editor)
    generate_preview_css/1,         % generate_preview_css(-CSS)

    % Package configuration
    generate_package_json/2,        % generate_package_json(+Project, -JSON)
    generate_tsconfig/1,            % generate_tsconfig(-Config)

    % Utility predicates
    get_watch_paths/2,              % get_watch_paths(+Project, -Paths)
    get_preview_port/2,             % get_preview_port(+Project, -Port)

    % Management
    declare_dev_server_config/2,    % declare_dev_server_config(+Project, +Options)
    clear_dev_configs/0,            % clear_dev_configs

    % Testing
    test_live_preview_generator/0
]).

:- use_module(library(lists)).

% ============================================================================
% DYNAMIC PREDICATES
% ============================================================================

:- dynamic dev_server_config/2.
:- dynamic preview_config/2.

:- discontiguous dev_server_config/2.
:- discontiguous preview_config/2.

% ============================================================================
% DEFAULT SERVER CONFIGURATIONS
% ============================================================================

dev_server_config(default, [
    port(3000),
    host('localhost'),
    hot_reload(true),
    open_browser(true),
    watch_paths(['src/**/*.pl', 'src/**/*.ts', 'src/**/*.tsx']),
    ignore_paths(['node_modules', 'dist', '.git']),
    debounce_ms(100),
    https(false)
]).

dev_server_config(visualization_preview, [
    port(3001),
    host('localhost'),
    hot_reload(true),
    open_browser(true),
    watch_paths([
        'src/unifyweaver/glue/**/*.pl',
        'src/**/*.ts',
        'src/**/*.tsx',
        'src/**/*.css'
    ]),
    ignore_paths(['node_modules', 'dist', '.git', '*.test.*']),
    debounce_ms(150),
    https(false),
    proxy([
        rule('/api', 'http://localhost:5000')
    ])
]).

% ============================================================================
% PREVIEW CONFIGURATIONS
% ============================================================================

preview_config(default, [
    layout(split),
    editor_position(left),
    editor_width('40%'),
    preview_width('60%'),
    show_console(true),
    show_props_panel(true),
    theme(dark),
    auto_refresh(true)
]).

preview_config(chart_preview, [
    layout(split),
    editor_position(left),
    editor_width('35%'),
    preview_width('65%'),
    show_console(true),
    show_props_panel(true),
    show_data_panel(true),
    theme(dark),
    auto_refresh(true),
    controls([
        data_editor,
        props_inspector,
        export_options
    ])
]).

preview_config(graph_preview, [
    layout(split),
    editor_position(bottom),
    editor_height('30%'),
    preview_height('70%'),
    show_console(true),
    theme(dark),
    controls([
        node_inspector,
        edge_editor,
        layout_selector
    ])
]).

% ============================================================================
% DEV SERVER GENERATION
% ============================================================================

%% generate_dev_server(+Project, -ServerCode)
%  Generate Node.js development server code.
generate_dev_server(Project, ServerCode) :-
    (dev_server_config(Project, Config) -> true ; dev_server_config(default, Config)),
    (member(port(Port), Config) -> true ; Port = 3000),
    (member(host(Host), Config) -> true ; Host = 'localhost'),
    format(atom(ServerCode), 'import express from "express";
import { createServer as createViteServer } from "vite";
import { WebSocketServer } from "ws";
import chokidar from "chokidar";
import path from "path";

const PORT = ~w;
const HOST = "~w";

async function createDevServer() {
  const app = express();

  // Create Vite server in middleware mode
  const vite = await createViteServer({
    server: { middlewareMode: true },
    appType: "spa"
  });

  // Use Vite middleware
  app.use(vite.middlewares);

  // Create HTTP server
  const server = app.listen(PORT, HOST, () => {
    console.log(`🚀 Dev server running at http://${HOST}:${PORT}`);
  });

  // WebSocket for hot reload notifications
  const wss = new WebSocketServer({ server, path: "/__hmr" });

  const clients = new Set<WebSocket>();
  wss.on("connection", (ws) => {
    clients.add(ws);
    ws.on("close", () => clients.delete(ws));
  });

  const broadcast = (message: object) => {
    const data = JSON.stringify(message);
    clients.forEach((client) => {
      if (client.readyState === 1) client.send(data);
    });
  };

  // File watcher for Prolog files
  const watcher = chokidar.watch([
    "src/unifyweaver/glue/**/*.pl",
    "src/**/*.ts",
    "src/**/*.tsx"
  ], {
    ignored: /(node_modules|dist|\\.git)/,
    persistent: true,
    ignoreInitial: true
  });

  let debounceTimer: NodeJS.Timeout;
  const debounceMs = 150;

  watcher.on("change", (filePath) => {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {
      console.log(`📝 File changed: ${filePath}`);
      broadcast({ type: "update", file: filePath, timestamp: Date.now() });
    }, debounceMs);
  });

  // Graceful shutdown
  process.on("SIGINT", () => {
    console.log("\\n👋 Shutting down...");
    watcher.close();
    server.close();
    process.exit(0);
  });

  return { app, server, wss, watcher };
}

createDevServer().catch(console.error);', [Port, Host]).

%% generate_vite_config(+Project, -Config)
%  Generate Vite configuration file.
generate_vite_config(Project, Config) :-
    (dev_server_config(Project, ServerConfig) -> true ; dev_server_config(default, ServerConfig)),
    (member(port(Port), ServerConfig) -> true ; Port = 3000),
    (member(host(Host), ServerConfig) -> true ; Host = 'localhost'),
    (member(https(UseHTTPS), ServerConfig) -> true ; UseHTTPS = false),
    (UseHTTPS = true -> HTTPSStr = 'true' ; HTTPSStr = 'false'),
    format(atom(Config), 'import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";

export default defineConfig({
  plugins: [react()],
  server: {
    port: ~w,
    host: "~w",
    https: ~w,
    open: true,
    hmr: {
      overlay: true,
    },
  },
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
      "@components": path.resolve(__dirname, "./src/components"),
      "@hooks": path.resolve(__dirname, "./src/hooks"),
      "@utils": path.resolve(__dirname, "./src/utils"),
    },
  },
  css: {
    modules: {
      localsConvention: "camelCase",
    },
  },
  build: {
    sourcemap: true,
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ["react", "react-dom"],
          charts: ["chart.js", "react-chartjs-2"],
        },
      },
    },
  },
});', [Port, Host, HTTPSStr]).

% ============================================================================
% PREVIEW COMPONENT GENERATION
% ============================================================================

%% generate_preview_app(+Component, -App)
%  Generate the main preview application component.
generate_preview_app(Component, App) :-
    (preview_config(Component, Config) -> true ; preview_config(default, Config)),
    (member(layout(Layout), Config) -> true ; Layout = split),
    (member(theme(Theme), Config) -> true ; Theme = dark),
    atom_string(Component, CompStr),
    atom_string(Layout, LayoutStr),
    atom_string(Theme, ThemeStr),
    format(atom(App), 'import React, { useState, useCallback, useEffect } from "react";
import { useHotReload } from "./hooks/useHotReload";
import { useStateSync } from "./hooks/useStateSync";
import { PreviewPanel } from "./components/PreviewPanel";
import { CodeEditor } from "./components/CodeEditor";
import { PropsPanel } from "./components/PropsPanel";
import { ConsolePanel } from "./components/ConsolePanel";
import styles from "./PreviewApp.module.css";

interface PreviewAppProps {
  initialCode?: string;
  component: React.ComponentType<unknown>;
}

export const PreviewApp: React.FC<PreviewAppProps> = ({ initialCode, component: Component }) => {
  const [code, setCode] = useState(initialCode || "");
  const [props, setProps] = useState<Record<string, unknown>>({});
  const [logs, setLogs] = useState<string[]>([]);
  const [error, setError] = useState<Error | null>(null);

  // Hot reload hook
  const { isConnected, lastUpdate } = useHotReload({
    onUpdate: useCallback((file: string) => {
      console.log(`Reloading due to change in: ${file}`);
      setLogs(prev => [...prev, `[HMR] Updated: ${file}`]);
    }, [])
  });

  // State synchronization
  const { syncState, loadState } = useStateSync("~w-preview");

  useEffect(() => {
    const saved = loadState();
    if (saved) {
      setCode(saved.code || "");
      setProps(saved.props || {});
    }
  }, [loadState]);

  useEffect(() => {
    syncState({ code, props });
  }, [code, props, syncState]);

  // Console capture
  useEffect(() => {
    const originalLog = console.log;
    console.log = (...args) => {
      setLogs(prev => [...prev, args.map(String).join(" ")]);
      originalLog.apply(console, args);
    };
    return () => { console.log = originalLog; };
  }, []);

  return (
    <div className={styles.previewApp} data-layout="~w" data-theme="~w">
      <header className={styles.header}>
        <h1>Preview: ~w</h1>
        <div className={styles.status}>
          <span className={isConnected ? styles.connected : styles.disconnected}>
            {isConnected ? "🟢 Connected" : "🔴 Disconnected"}
          </span>
          {lastUpdate && (
            <span className={styles.lastUpdate}>
              Last update: {new Date(lastUpdate).toLocaleTimeString()}
            </span>
          )}
        </div>
      </header>

      <main className={styles.main}>
        <aside className={styles.editor}>
          <CodeEditor
            value={code}
            onChange={setCode}
            language="prolog"
          />
        </aside>

        <section className={styles.preview}>
          <PreviewPanel error={error}>
            <Component {...props} />
          </PreviewPanel>
        </section>

        <aside className={styles.sidebar}>
          <PropsPanel props={props} onChange={setProps} />
          <ConsolePanel logs={logs} onClear={() => setLogs([])} />
        </aside>
      </main>
    </div>
  );
};

export default PreviewApp;', [CompStr, LayoutStr, ThemeStr, CompStr]).

%% generate_preview_wrapper(+Component, -Wrapper)
%  Generate an error boundary wrapper for previews.
generate_preview_wrapper(_Component, Wrapper) :-
    Wrapper = 'import React, { Component, ErrorInfo, ReactNode } from "react";
import styles from "./PreviewWrapper.module.css";

interface Props {
  children: ReactNode;
  onError?: (error: Error, errorInfo: ErrorInfo) => void;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

export class PreviewWrapper extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error("Preview error:", error, errorInfo);
    this.props.onError?.(error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className={styles.errorBoundary}>
          <h2>Something went wrong</h2>
          <pre className={styles.errorMessage}>
            {this.state.error?.message}
          </pre>
          <pre className={styles.errorStack}>
            {this.state.error?.stack}
          </pre>
          <button
            onClick={() => this.setState({ hasError: false, error: null })}
            className={styles.retryButton}
          >
            Try Again
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}'.

%% generate_hot_reload_hook(-Hook)
%  Generate React hook for hot reload connection.
generate_hot_reload_hook(Hook) :-
    Hook = 'import { useState, useEffect, useCallback, useRef } from "react";

interface HotReloadOptions {
  onUpdate?: (file: string) => void;
  onConnect?: () => void;
  onDisconnect?: () => void;
  reconnectInterval?: number;
}

interface HotReloadState {
  isConnected: boolean;
  lastUpdate: number | null;
}

export const useHotReload = (options: HotReloadOptions = {}): HotReloadState => {
  const {
    onUpdate,
    onConnect,
    onDisconnect,
    reconnectInterval = 3000
  } = options;

  const [state, setState] = useState<HotReloadState>({
    isConnected: false,
    lastUpdate: null
  });

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimerRef = useRef<NodeJS.Timeout>();

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const ws = new WebSocket(`${protocol}//${window.location.host}/__hmr`);

    ws.onopen = () => {
      setState(prev => ({ ...prev, isConnected: true }));
      onConnect?.();
      console.log("[HMR] Connected");
    };

    ws.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data);
        if (message.type === "update") {
          setState(prev => ({ ...prev, lastUpdate: message.timestamp }));
          onUpdate?.(message.file);
        }
      } catch (e) {
        console.error("[HMR] Failed to parse message:", e);
      }
    };

    ws.onclose = () => {
      setState(prev => ({ ...prev, isConnected: false }));
      onDisconnect?.();
      console.log("[HMR] Disconnected, reconnecting...");
      reconnectTimerRef.current = setTimeout(connect, reconnectInterval);
    };

    ws.onerror = (error) => {
      console.error("[HMR] WebSocket error:", error);
    };

    wsRef.current = ws;
  }, [onUpdate, onConnect, onDisconnect, reconnectInterval]);

  useEffect(() => {
    connect();
    return () => {
      clearTimeout(reconnectTimerRef.current);
      wsRef.current?.close();
    };
  }, [connect]);

  return state;
};'.

%% generate_state_sync_hook(-Hook)
%  Generate React hook for state persistence.
generate_state_sync_hook(Hook) :-
    Hook = 'import { useCallback } from "react";

interface SyncState {
  syncState: (state: Record<string, unknown>) => void;
  loadState: () => Record<string, unknown> | null;
  clearState: () => void;
}

export const useStateSync = (key: string): SyncState => {
  const syncState = useCallback((state: Record<string, unknown>) => {
    try {
      sessionStorage.setItem(key, JSON.stringify(state));
    } catch (e) {
      console.warn("Failed to sync state:", e);
    }
  }, [key]);

  const loadState = useCallback(() => {
    try {
      const saved = sessionStorage.getItem(key);
      return saved ? JSON.parse(saved) : null;
    } catch (e) {
      console.warn("Failed to load state:", e);
      return null;
    }
  }, [key]);

  const clearState = useCallback(() => {
    sessionStorage.removeItem(key);
  }, [key]);

  return { syncState, loadState, clearState };
};'.

%% generate_preview_panel(+Component, -Panel)
%  Generate the preview panel component.
generate_preview_panel(_Component, Panel) :-
    Panel = 'import React, { ReactNode } from "react";
import { PreviewWrapper } from "./PreviewWrapper";
import styles from "./PreviewPanel.module.css";

interface PreviewPanelProps {
  children: ReactNode;
  error?: Error | null;
  onRetry?: () => void;
}

export const PreviewPanel: React.FC<PreviewPanelProps> = ({
  children,
  error,
  onRetry
}) => {
  if (error) {
    return (
      <div className={styles.errorPanel}>
        <div className={styles.errorIcon}>⚠️</div>
        <h3>Preview Error</h3>
        <pre className={styles.errorMessage}>{error.message}</pre>
        {onRetry && (
          <button onClick={onRetry} className={styles.retryButton}>
            Retry
          </button>
        )}
      </div>
    );
  }

  return (
    <div className={styles.previewPanel}>
      <PreviewWrapper>
        {children}
      </PreviewWrapper>
    </div>
  );
};'.

%% generate_code_editor(+Component, -Editor)
%  Generate a simple code editor component.
generate_code_editor(_Component, Editor) :-
    Editor = 'import React, { useRef, useEffect } from "react";
import styles from "./CodeEditor.module.css";

interface CodeEditorProps {
  value: string;
  onChange: (value: string) => void;
  language?: string;
  readOnly?: boolean;
}

export const CodeEditor: React.FC<CodeEditorProps> = ({
  value,
  onChange,
  language = "prolog",
  readOnly = false
}) => {
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-resize textarea
  useEffect(() => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = "auto";
      textarea.style.height = `${textarea.scrollHeight}px`;
    }
  }, [value]);

  // Handle tab key
  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Tab") {
      e.preventDefault();
      const textarea = e.currentTarget;
      const start = textarea.selectionStart;
      const end = textarea.selectionEnd;
      const newValue = value.substring(0, start) + "  " + value.substring(end);
      onChange(newValue);
      // Restore cursor position
      setTimeout(() => {
        textarea.selectionStart = textarea.selectionEnd = start + 2;
      }, 0);
    }
  };

  return (
    <div className={styles.codeEditor} data-language={language}>
      <div className={styles.header}>
        <span className={styles.language}>{language}</span>
        {!readOnly && <span className={styles.hint}>Tab for indent</span>}
      </div>
      <textarea
        ref={textareaRef}
        className={styles.textarea}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={handleKeyDown}
        readOnly={readOnly}
        spellCheck={false}
        wrap="off"
      />
      <div className={styles.lineNumbers}>
        {value.split("\\n").map((_, i) => (
          <span key={i}>{i + 1}</span>
        ))}
      </div>
    </div>
  );
};'.

%% generate_preview_css(-CSS)
%  Generate CSS for the preview system.
generate_preview_css(CSS) :-
    CSS = '/* Preview App Styles */
.previewApp {
  display: grid;
  grid-template-rows: auto 1fr;
  height: 100vh;
  background: var(--background, #1a1a2e);
  color: var(--text, #e0e0e0);
}

.previewApp[data-theme="dark"] {
  --background: #1a1a2e;
  --surface: #16213e;
  --text: #e0e0e0;
  --text-secondary: #888;
  --accent: #00d4ff;
  --border: rgba(255, 255, 255, 0.1);
  --error: #ff6b6b;
  --success: #4ecdc4;
}

.previewApp[data-theme="light"] {
  --background: #f8fafc;
  --surface: #ffffff;
  --text: #1a1a2e;
  --text-secondary: #64748b;
  --accent: #7c3aed;
  --border: #e2e8f0;
  --error: #dc2626;
  --success: #16a34a;
}

.header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.75rem 1.5rem;
  background: var(--surface);
  border-bottom: 1px solid var(--border);
}

.header h1 {
  margin: 0;
  font-size: 1.125rem;
  font-weight: 600;
}

.status {
  display: flex;
  align-items: center;
  gap: 1rem;
  font-size: 0.875rem;
}

.connected {
  color: var(--success);
}

.disconnected {
  color: var(--error);
}

.lastUpdate {
  color: var(--text-secondary);
}

.main {
  display: grid;
  grid-template-columns: 1fr 1.5fr auto;
  gap: 1px;
  background: var(--border);
  overflow: hidden;
}

.previewApp[data-layout="split"] .main {
  grid-template-columns: 40% 60%;
}

.previewApp[data-layout="vertical"] .main {
  grid-template-columns: 1fr;
  grid-template-rows: 1fr 1fr;
}

.editor,
.preview,
.sidebar {
  background: var(--background);
  overflow: auto;
}

.editor {
  border-right: 1px solid var(--border);
}

.sidebar {
  width: 280px;
  border-left: 1px solid var(--border);
  display: flex;
  flex-direction: column;
}

/* Code Editor Styles */
.codeEditor {
  display: flex;
  flex-direction: column;
  height: 100%;
  font-family: "JetBrains Mono", "Fira Code", monospace;
  font-size: 13px;
  line-height: 1.5;
}

.codeEditor .header {
  display: flex;
  justify-content: space-between;
  padding: 0.5rem 1rem;
  background: var(--surface);
  border-bottom: 1px solid var(--border);
}

.codeEditor .language {
  color: var(--accent);
  text-transform: uppercase;
  font-size: 0.75rem;
  font-weight: 600;
}

.codeEditor .hint {
  color: var(--text-secondary);
  font-size: 0.75rem;
}

.codeEditor .textarea {
  flex: 1;
  padding: 1rem;
  padding-left: 3.5rem;
  background: var(--background);
  color: var(--text);
  border: none;
  outline: none;
  resize: none;
  font-family: inherit;
  font-size: inherit;
  line-height: inherit;
  tab-size: 2;
}

.codeEditor .lineNumbers {
  position: absolute;
  left: 0;
  top: 0;
  padding: 1rem 0.5rem;
  text-align: right;
  color: var(--text-secondary);
  user-select: none;
  display: flex;
  flex-direction: column;
}

/* Preview Panel Styles */
.previewPanel {
  padding: 1.5rem;
  height: 100%;
  overflow: auto;
}

.errorPanel {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 2rem;
  text-align: center;
}

.errorIcon {
  font-size: 3rem;
  margin-bottom: 1rem;
}

.errorMessage {
  background: rgba(255, 107, 107, 0.1);
  color: var(--error);
  padding: 1rem;
  border-radius: 4px;
  max-width: 100%;
  overflow-x: auto;
  font-size: 0.875rem;
}

.retryButton {
  margin-top: 1rem;
  padding: 0.5rem 1.5rem;
  background: var(--accent);
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-weight: 500;
}

.retryButton:hover {
  opacity: 0.9;
}

/* Error Boundary Styles */
.errorBoundary {
  padding: 2rem;
  text-align: center;
}

.errorBoundary h2 {
  color: var(--error);
  margin-bottom: 1rem;
}

.errorStack {
  text-align: left;
  background: var(--surface);
  padding: 1rem;
  border-radius: 4px;
  font-size: 0.75rem;
  overflow-x: auto;
  max-height: 200px;
  margin-top: 1rem;
}

/* Console Panel */
.consolePanel {
  flex: 1;
  display: flex;
  flex-direction: column;
  border-top: 1px solid var(--border);
}

.consoleHeader {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.5rem 1rem;
  background: var(--surface);
}

.consoleLogs {
  flex: 1;
  padding: 0.5rem 1rem;
  font-family: monospace;
  font-size: 0.75rem;
  overflow-y: auto;
}

.consoleLogs pre {
  margin: 0.25rem 0;
  white-space: pre-wrap;
  word-break: break-all;
}

/* Responsive */
@media (max-width: 1024px) {
  .main {
    grid-template-columns: 1fr;
    grid-template-rows: 1fr 1fr;
  }

  .sidebar {
    display: none;
  }
}'.

% ============================================================================
% PACKAGE CONFIGURATION
% ============================================================================

%% generate_package_json(+Project, -JSON)
%  Generate package.json for the preview project.
generate_package_json(Project, JSON) :-
    atom_string(Project, ProjectStr),
    format(atom(JSON), '{
  "name": "~w-preview",
  "version": "1.0.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc && vite build",
    "preview": "vite preview",
    "server": "tsx src/server.ts"
  },
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "chart.js": "^4.4.0",
    "react-chartjs-2": "^5.2.0"
  },
  "devDependencies": {
    "@types/react": "^18.2.0",
    "@types/react-dom": "^18.2.0",
    "@vitejs/plugin-react": "^4.2.0",
    "chokidar": "^3.5.3",
    "express": "^4.18.2",
    "typescript": "^5.3.0",
    "tsx": "^4.6.0",
    "vite": "^5.0.0",
    "ws": "^8.14.0"
  }
}', [ProjectStr]).

%% generate_tsconfig(-Config)
%  Generate TypeScript configuration.
generate_tsconfig(Config) :-
    Config = '{
  "compilerOptions": {
    "target": "ES2020",
    "useDefineForClassFields": true,
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "skipLibCheck": true,
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx",
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true,
    "paths": {
      "@/*": ["./src/*"],
      "@components/*": ["./src/components/*"],
      "@hooks/*": ["./src/hooks/*"]
    }
  },
  "include": ["src"],
  "references": [{ "path": "./tsconfig.node.json" }]
}'.

% ============================================================================
% UTILITY PREDICATES
% ============================================================================

%% get_watch_paths(+Project, -Paths)
get_watch_paths(Project, Paths) :-
    (dev_server_config(Project, Config) -> true ; dev_server_config(default, Config)),
    (member(watch_paths(Paths), Config) -> true ; Paths = ['src/**/*']).

%% get_preview_port(+Project, -Port)
get_preview_port(Project, Port) :-
    (dev_server_config(Project, Config) -> true ; dev_server_config(default, Config)),
    (member(port(Port), Config) -> true ; Port = 3000).

% ============================================================================
% MANAGEMENT
% ============================================================================

%% declare_dev_server_config(+Project, +Options)
declare_dev_server_config(Project, Options) :-
    retractall(dev_server_config(Project, _)),
    assertz(dev_server_config(Project, Options)).

%% clear_dev_configs/0
clear_dev_configs :-
    retractall(dev_server_config(_, _)),
    retractall(preview_config(_, _)).

% ============================================================================
% TESTING
% ============================================================================

test_live_preview_generator :-
    format('========================================~n'),
    format('Live Preview Generator Tests~n'),
    format('========================================~n~n'),

    % Test 1: Dev server generation
    format('Test 1: Dev server generation~n'),
    generate_dev_server(default, ServerCode),
    (sub_atom(ServerCode, _, _, _, 'express')
    -> format('  PASS: Uses express~n')
    ; format('  FAIL: Missing express~n')),
    (sub_atom(ServerCode, _, _, _, 'WebSocketServer')
    -> format('  PASS: Has WebSocket server~n')
    ; format('  FAIL: Missing WebSocket~n')),
    (sub_atom(ServerCode, _, _, _, 'chokidar')
    -> format('  PASS: Uses chokidar watcher~n')
    ; format('  FAIL: Missing chokidar~n')),

    % Test 2: Vite config generation
    format('~nTest 2: Vite config generation~n'),
    generate_vite_config(default, ViteConfig),
    (sub_atom(ViteConfig, _, _, _, 'defineConfig')
    -> format('  PASS: Has defineConfig~n')
    ; format('  FAIL: Missing defineConfig~n')),
    (sub_atom(ViteConfig, _, _, _, 'hmr')
    -> format('  PASS: Has HMR config~n')
    ; format('  FAIL: Missing HMR~n')),

    % Test 3: Preview app generation
    format('~nTest 3: Preview app generation~n'),
    generate_preview_app(chart_preview, PreviewApp),
    (sub_atom(PreviewApp, _, _, _, 'useHotReload')
    -> format('  PASS: Uses hot reload hook~n')
    ; format('  FAIL: Missing hot reload hook~n')),
    (sub_atom(PreviewApp, _, _, _, 'PreviewPanel')
    -> format('  PASS: Has preview panel~n')
    ; format('  FAIL: Missing preview panel~n')),

    % Test 4: Hot reload hook
    format('~nTest 4: Hot reload hook~n'),
    generate_hot_reload_hook(HRHook),
    (sub_atom(HRHook, _, _, _, 'WebSocket')
    -> format('  PASS: Uses WebSocket~n')
    ; format('  FAIL: Missing WebSocket~n')),
    (sub_atom(HRHook, _, _, _, 'reconnect')
    -> format('  PASS: Has reconnect logic~n')
    ; format('  FAIL: Missing reconnect~n')),

    % Test 5: State sync hook
    format('~nTest 5: State sync hook~n'),
    generate_state_sync_hook(SSHook),
    (sub_atom(SSHook, _, _, _, 'sessionStorage')
    -> format('  PASS: Uses sessionStorage~n')
    ; format('  FAIL: Missing sessionStorage~n')),

    % Test 6: Preview CSS
    format('~nTest 6: Preview CSS~n'),
    generate_preview_css(CSS),
    (sub_atom(CSS, _, _, _, '.previewApp')
    -> format('  PASS: Has previewApp class~n')
    ; format('  FAIL: Missing previewApp~n')),
    (sub_atom(CSS, _, _, _, 'data-theme')
    -> format('  PASS: Has theme support~n')
    ; format('  FAIL: Missing theme support~n')),

    % Test 7: Package.json generation
    format('~nTest 7: Package.json generation~n'),
    generate_package_json(test_project, PackageJSON),
    (sub_atom(PackageJSON, _, _, _, '"vite"')
    -> format('  PASS: Has vite dependency~n')
    ; format('  FAIL: Missing vite~n')),
    (sub_atom(PackageJSON, _, _, _, '"react"')
    -> format('  PASS: Has react dependency~n')
    ; format('  FAIL: Missing react~n')),

    % Test 8: Utility predicates
    format('~nTest 8: Utility predicates~n'),
    get_preview_port(default, Port),
    (Port =:= 3000
    -> format('  PASS: Default port is 3000~n')
    ; format('  FAIL: Wrong default port~n')),
    get_watch_paths(visualization_preview, Paths),
    (member('src/unifyweaver/glue/**/*.pl', Paths)
    -> format('  PASS: Watches Prolog files~n')
    ; format('  FAIL: Not watching Prolog~n')),

    format('~nAll tests completed.~n').
