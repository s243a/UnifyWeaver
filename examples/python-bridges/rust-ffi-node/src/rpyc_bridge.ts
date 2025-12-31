/**
 * TypeScript wrapper for the Rust FFI RPyC bridge.
 *
 * This module provides a type-safe interface to call Python functions
 * via the Rust FFI bridge (librpyc_bridge.so).
 *
 * Uses koffi for FFI (compatible with Node.js 18+).
 */

import koffi from 'koffi';
import path from 'path';
import fs from 'fs';

// Library path - look for the shared library
function getLibraryPath(): string {
  const possiblePaths = [
    path.join(__dirname, '..', 'librpyc_bridge.so'),
    path.join(__dirname, '..', '..', 'rust-ffi-go', 'target', 'release', 'librpyc_bridge.so'),
    path.join(__dirname, '..', '..', 'rust-ffi-go', 'librpyc_bridge.so'),
  ];

  for (const p of possiblePaths) {
    try {
      fs.accessSync(p);
      return p;
    } catch {
      continue;
    }
  }

  throw new Error(
    `librpyc_bridge.so not found. Build it with:\n` +
    `  cd examples/python-bridges/rust-ffi-go && cargo build --release`
  );
}

// Load the native library
const lib = koffi.load(getLibraryPath());

// Define the FFI functions
const rpyc_init = lib.func('void rpyc_init()');
const rpyc_connect = lib.func('int rpyc_connect(const char* host, int port)');
const rpyc_disconnect = lib.func('void rpyc_disconnect()');
const rpyc_is_connected = lib.func('int rpyc_is_connected()');
const rpyc_call = lib.func('const char* rpyc_call(const char* module, const char* func, const char* args_json)');
const rpyc_getattr = lib.func('const char* rpyc_getattr(const char* module, const char* attr)');
const rpyc_free_string = lib.func('void rpyc_free_string(const char* s)');

/**
 * RPyC Bridge class for calling Python functions from Node.js.
 */
export class RpycBridge {
  private connected: boolean = false;
  private initialized: boolean = false;

  constructor() {
    // Initialize on first use
  }

  /**
   * Initialize the Python runtime.
   */
  init(): void {
    if (!this.initialized) {
      rpyc_init();
      this.initialized = true;
    }
  }

  /**
   * Connect to an RPyC server.
   */
  connect(host: string = 'localhost', port: number = 18812): void {
    this.init();
    const result = rpyc_connect(host, port);
    if (result !== 0) {
      throw new Error(`Failed to connect to RPyC server at ${host}:${port}`);
    }
    this.connected = true;
  }

  /**
   * Disconnect from the RPyC server.
   */
  disconnect(): void {
    if (this.connected) {
      rpyc_disconnect();
      this.connected = false;
    }
  }

  /**
   * Check if connected to RPyC server.
   */
  isConnected(): boolean {
    return rpyc_is_connected() !== 0;
  }

  /**
   * Call a Python function via RPyC.
   *
   * @param module - Python module name (e.g., 'numpy', 'math')
   * @param func - Function name (e.g., 'mean', 'sqrt')
   * @param args - Arguments to pass (will be JSON-serialized)
   * @returns The result (JSON-parsed)
   */
  call<T = unknown>(module: string, func: string, args: unknown[] = []): T {
    if (!this.connected) {
      throw new Error('Not connected to RPyC server. Call connect() first.');
    }

    const argsJson = JSON.stringify(args);
    const resultPtr = rpyc_call(module, func, argsJson);

    if (resultPtr === null) {
      throw new Error(`Failed to call ${module}.${func}`);
    }

    const resultJson = resultPtr as string;

    try {
      return JSON.parse(resultJson) as T;
    } catch {
      // Return as string if not valid JSON
      return resultJson as unknown as T;
    }
  }

  /**
   * Get an attribute from a Python module.
   *
   * @param module - Python module name
   * @param attr - Attribute name
   * @returns The attribute value
   */
  getattr<T = unknown>(module: string, attr: string): T {
    if (!this.connected) {
      throw new Error('Not connected to RPyC server. Call connect() first.');
    }

    const resultPtr = rpyc_getattr(module, attr);

    if (resultPtr === null) {
      throw new Error(`Failed to get ${module}.${attr}`);
    }

    const resultJson = resultPtr as string;

    try {
      return JSON.parse(resultJson) as T;
    } catch {
      return resultJson as unknown as T;
    }
  }
}

// Singleton instance for convenience
let defaultBridge: RpycBridge | null = null;

/**
 * Get the default bridge instance.
 */
export function getBridge(): RpycBridge {
  if (!defaultBridge) {
    defaultBridge = new RpycBridge();
  }
  return defaultBridge;
}

/**
 * Convenience function to call a Python function.
 */
export function pythonCall<T = unknown>(
  module: string,
  func: string,
  args: unknown[] = []
): T {
  return getBridge().call<T>(module, func, args);
}

export default RpycBridge;
