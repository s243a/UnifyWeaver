/**
 * Express API server for the Rust FFI RPyC bridge.
 *
 * Provides HTTP endpoints to call Python functions via the bridge.
 * Includes security measures against injection attacks.
 */

import express, { Request, Response, NextFunction } from 'express';
import cors from 'cors';
import { RpycBridge } from './rpyc_bridge';

const app = express();
const PORT = process.env.PORT || 3001;
const RPYC_HOST = process.env.RPYC_HOST || 'localhost';
const RPYC_PORT = parseInt(process.env.RPYC_PORT || '18812', 10);

// ==========================================
// Security: Whitelisted modules and functions
// ==========================================

const ALLOWED_MODULES: Record<string, Set<string>> = {
  'math': new Set(['sqrt', 'sin', 'cos', 'tan', 'log', 'log10', 'exp', 'pow', 'floor', 'ceil', 'abs']),
  'numpy': new Set(['mean', 'std', 'sum', 'min', 'max', 'median', 'var', 'array', 'zeros', 'ones', 'linspace']),
  'statistics': new Set(['mean', 'median', 'mode', 'stdev', 'variance']),
};

const ALLOWED_ATTRS: Record<string, Set<string>> = {
  'math': new Set(['pi', 'e', 'tau', 'inf', 'nan']),
  'numpy': new Set(['pi', 'e']),
};

/**
 * Validate module and function against whitelist.
 * Prevents arbitrary code execution.
 */
function validateCall(module: string, func: string): { valid: boolean; error?: string } {
  // Check module name format (alphanumeric + underscore only)
  if (!/^[a-zA-Z_][a-zA-Z0-9_]*$/.test(module)) {
    return { valid: false, error: `Invalid module name format: ${module}` };
  }

  // Check function name format
  if (!/^[a-zA-Z_][a-zA-Z0-9_]*$/.test(func)) {
    return { valid: false, error: `Invalid function name format: ${func}` };
  }

  // Check whitelist
  const allowedFuncs = ALLOWED_MODULES[module];
  if (!allowedFuncs) {
    return { valid: false, error: `Module not allowed: ${module}. Allowed: ${Object.keys(ALLOWED_MODULES).join(', ')}` };
  }

  if (!allowedFuncs.has(func)) {
    return { valid: false, error: `Function not allowed: ${module}.${func}. Allowed: ${Array.from(allowedFuncs).join(', ')}` };
  }

  return { valid: true };
}

/**
 * Validate attribute access against whitelist.
 */
function validateAttr(module: string, attr: string): { valid: boolean; error?: string } {
  // Check format
  if (!/^[a-zA-Z_][a-zA-Z0-9_]*$/.test(module) || !/^[a-zA-Z_][a-zA-Z0-9_]*$/.test(attr)) {
    return { valid: false, error: 'Invalid name format' };
  }

  const allowedAttrs = ALLOWED_ATTRS[module];
  if (!allowedAttrs || !allowedAttrs.has(attr)) {
    return { valid: false, error: `Attribute not allowed: ${module}.${attr}` };
  }

  return { valid: true };
}

/**
 * Validate arguments - only allow safe JSON types.
 * Prevents code injection via arguments.
 */
function validateArgs(args: unknown): { valid: boolean; error?: string } {
  if (!Array.isArray(args)) {
    return { valid: false, error: 'Arguments must be an array' };
  }

  // Recursively check for safe types
  function isSafe(val: unknown, depth: number = 0): boolean {
    if (depth > 10) return false; // Prevent deep nesting

    if (val === null || val === undefined) return true;
    if (typeof val === 'number' && isFinite(val)) return true;
    if (typeof val === 'string' && val.length < 10000) return true;
    if (typeof val === 'boolean') return true;

    if (Array.isArray(val)) {
      if (val.length > 10000) return false; // Limit array size
      return val.every(v => isSafe(v, depth + 1));
    }

    if (typeof val === 'object') {
      const keys = Object.keys(val as object);
      if (keys.length > 100) return false; // Limit object size
      return keys.every(k => /^[a-zA-Z0-9_]+$/.test(k) && isSafe((val as Record<string, unknown>)[k], depth + 1));
    }

    return false;
  }

  if (!args.every(arg => isSafe(arg))) {
    return { valid: false, error: 'Arguments contain unsafe values' };
  }

  return { valid: true };
}

// ==========================================
// Rate limiting (simple in-memory)
// ==========================================

const requestCounts = new Map<string, { count: number; resetTime: number }>();
const RATE_LIMIT = 100; // requests per window
const RATE_WINDOW = 60000; // 1 minute

function checkRateLimit(ip: string): boolean {
  const now = Date.now();
  const record = requestCounts.get(ip);

  if (!record || now > record.resetTime) {
    requestCounts.set(ip, { count: 1, resetTime: now + RATE_WINDOW });
    return true;
  }

  if (record.count >= RATE_LIMIT) {
    return false;
  }

  record.count++;
  return true;
}

// Middleware
app.use(cors({
  origin: process.env.CORS_ORIGIN || '*', // Restrict in production
  methods: ['GET', 'POST'],
}));
app.use(express.json({ limit: '100kb' })); // Limit body size

// Rate limiting middleware
app.use((req: Request, res: Response, next: NextFunction) => {
  const ip = req.ip || req.socket.remoteAddress || 'unknown';
  if (!checkRateLimit(ip)) {
    res.status(429).json({ success: false, error: 'Rate limit exceeded' });
    return;
  }
  next();
});

// Bridge instance
let bridge: RpycBridge | null = null;

/**
 * Initialize the bridge connection.
 */
async function initBridge(): Promise<void> {
  if (!bridge) {
    bridge = new RpycBridge();
    try {
      bridge.connect(RPYC_HOST, RPYC_PORT);
      console.log(`Connected to RPyC server at ${RPYC_HOST}:${RPYC_PORT}`);
    } catch (error) {
      console.error('Failed to connect to RPyC server:', error);
      bridge = null;
      throw error;
    }
  }
}

// Health check
app.get('/health', (_req: Request, res: Response) => {
  res.json({
    status: 'ok',
    connected: bridge?.isConnected() ?? false,
    rpycServer: `${RPYC_HOST}:${RPYC_PORT}`,
  });
});

// Connect to RPyC server
app.post('/connect', async (_req: Request, res: Response) => {
  try {
    await initBridge();
    res.json({ success: true, message: 'Connected to RPyC server' });
  } catch (error) {
    res.status(500).json({
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error',
    });
  }
});

// Disconnect from RPyC server
app.post('/disconnect', (_req: Request, res: Response) => {
  if (bridge) {
    bridge.disconnect();
    bridge = null;
  }
  res.json({ success: true, message: 'Disconnected from RPyC server' });
});

/**
 * Generic Python call endpoint.
 *
 * POST /python/call
 * Body: { module: string, func: string, args: any[] }
 *
 * Security: Module/function must be whitelisted, args are validated.
 */
app.post('/python/call', async (req: Request, res: Response) => {
  try {
    if (!bridge) {
      await initBridge();
    }

    const { module, func, args = [] } = req.body;

    if (!module || !func) {
      res.status(400).json({
        success: false,
        error: 'Missing required fields: module, func',
      });
      return;
    }

    // Security: Validate module and function
    const callValidation = validateCall(module, func);
    if (!callValidation.valid) {
      res.status(403).json({
        success: false,
        error: callValidation.error,
      });
      return;
    }

    // Security: Validate arguments
    const argsValidation = validateArgs(args);
    if (!argsValidation.valid) {
      res.status(400).json({
        success: false,
        error: argsValidation.error,
      });
      return;
    }

    const result = bridge!.call(module, func, args);
    res.json({ success: true, result });
  } catch (error) {
    res.status(500).json({
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error',
    });
  }
});

/**
 * Get Python module attribute.
 *
 * GET /python/attr/:module/:attr
 *
 * Security: Module/attr must be whitelisted.
 */
app.get('/python/attr/:module/:attr', async (req: Request, res: Response) => {
  try {
    if (!bridge) {
      await initBridge();
    }

    const { module, attr } = req.params;

    // Security: Validate attribute access
    const attrValidation = validateAttr(module, attr);
    if (!attrValidation.valid) {
      res.status(403).json({
        success: false,
        error: attrValidation.error,
      });
      return;
    }

    const result = bridge!.getattr(module, attr);
    res.json({ success: true, result });
  } catch (error) {
    res.status(500).json({
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error',
    });
  }
});

// ==========================================
// Convenience endpoints for common operations
// ==========================================

/**
 * NumPy mean calculation.
 *
 * POST /numpy/mean
 * Body: { data: number[] }
 */
app.post('/numpy/mean', async (req: Request, res: Response) => {
  try {
    if (!bridge) {
      await initBridge();
    }

    const { data } = req.body;

    if (!Array.isArray(data)) {
      res.status(400).json({
        success: false,
        error: 'Missing or invalid "data" array',
      });
      return;
    }

    const result = bridge!.call<number>('numpy', 'mean', [data]);
    res.json({ success: true, result });
  } catch (error) {
    res.status(500).json({
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error',
    });
  }
});

/**
 * NumPy standard deviation.
 *
 * POST /numpy/std
 * Body: { data: number[] }
 */
app.post('/numpy/std', async (req: Request, res: Response) => {
  try {
    if (!bridge) {
      await initBridge();
    }

    const { data } = req.body;

    if (!Array.isArray(data)) {
      res.status(400).json({
        success: false,
        error: 'Missing or invalid "data" array',
      });
      return;
    }

    const result = bridge!.call<number>('numpy', 'std', [data]);
    res.json({ success: true, result });
  } catch (error) {
    res.status(500).json({
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error',
    });
  }
});

/**
 * Math sqrt.
 *
 * POST /math/sqrt
 * Body: { value: number }
 */
app.post('/math/sqrt', async (req: Request, res: Response) => {
  try {
    if (!bridge) {
      await initBridge();
    }

    const { value } = req.body;

    if (typeof value !== 'number') {
      res.status(400).json({
        success: false,
        error: 'Missing or invalid "value" number',
      });
      return;
    }

    const result = bridge!.call<number>('math', 'sqrt', [value]);
    res.json({ success: true, result });
  } catch (error) {
    res.status(500).json({
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error',
    });
  }
});

/**
 * Get math.pi constant.
 *
 * GET /math/pi
 */
app.get('/math/pi', async (_req: Request, res: Response) => {
  try {
    if (!bridge) {
      await initBridge();
    }

    const result = bridge!.getattr<number>('math', 'pi');
    res.json({ success: true, result });
  } catch (error) {
    res.status(500).json({
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error',
    });
  }
});

// Error handling middleware
app.use((err: Error, _req: Request, res: Response, _next: NextFunction) => {
  console.error('Unhandled error:', err);
  res.status(500).json({
    success: false,
    error: err.message || 'Internal server error',
  });
});

// Start server
app.listen(PORT, () => {
  console.log(`
╔══════════════════════════════════════════════════════════╗
║  Node.js + Rust FFI + RPyC API Server                    ║
╠══════════════════════════════════════════════════════════╣
║  Server:     http://localhost:${PORT}                       ║
║  RPyC:       ${RPYC_HOST}:${RPYC_PORT}                               ║
╠══════════════════════════════════════════════════════════╣
║  Endpoints:                                              ║
║    GET  /health              - Health check              ║
║    POST /connect             - Connect to RPyC           ║
║    POST /disconnect          - Disconnect                ║
║    POST /python/call         - Generic Python call       ║
║    GET  /python/attr/:m/:a   - Get module attribute      ║
║    POST /numpy/mean          - Calculate mean            ║
║    POST /numpy/std           - Calculate std dev         ║
║    POST /math/sqrt           - Square root               ║
║    GET  /math/pi             - Get pi constant           ║
╚══════════════════════════════════════════════════════════╝
`);
});

// Graceful shutdown
process.on('SIGINT', () => {
  console.log('\nShutting down...');
  if (bridge) {
    bridge.disconnect();
  }
  process.exit(0);
});

export default app;
