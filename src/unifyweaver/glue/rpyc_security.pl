% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% RPyC Security Rules - Declarative Python Module Whitelisting
%
% This module provides declarative security rules for RPyC access,
% allowing fine-grained control over which Python modules and functions
% can be called from generated TypeScript/Node.js code.
%
% Usage:
%   % Define allowed modules
%   rpyc_allowed_module(math, [sqrt, sin, cos, tan, log, exp, pow, floor, ceil, abs]).
%   rpyc_allowed_module(numpy, [mean, std, sum, min, max, array, zeros, ones]).
%
%   % Generate TypeScript whitelist
%   ?- generate_typescript_whitelist(Code).
%
%   % Generate TypeScript validator
%   ?- generate_typescript_validator(Code).

:- module(rpyc_security, [
    % Security rule predicates
    rpyc_allowed_module/2,          % rpyc_allowed_module(+Module, +Functions)
    rpyc_allowed_attr/2,            % rpyc_allowed_attr(+Module, +Attributes)
    rpyc_rate_limit/1,              % rpyc_rate_limit(+Config)
    rpyc_security_policy/2,         % rpyc_security_policy(+Name, +Rules)

    % Rule management
    add_allowed_module/2,           % add_allowed_module(+Module, +Functions)
    add_allowed_attr/2,             % add_allowed_attr(+Module, +Attributes)
    remove_allowed_module/1,        % remove_allowed_module(+Module)
    clear_security_rules/0,         % clear_security_rules

    % Validation
    is_call_allowed/3,              % is_call_allowed(+Module, +Function, -Allowed)
    is_attr_allowed/3,              % is_attr_allowed(+Module, +Attr, -Allowed)
    validate_call/3,                % validate_call(+Module, +Function, -Result)

    % Code generation
    generate_typescript_whitelist/1,    % generate_typescript_whitelist(-Code)
    generate_typescript_validator/1,    % generate_typescript_validator(-Code)
    generate_typescript_security/1,     % generate_typescript_security(-Code) - combined
    generate_express_security_middleware/1,  % generate_express_security_middleware(-Code)

    % Preferences/Firewall integration
    validate_with_firewall/3,           % validate_with_firewall(+Module, +Function, -Result)
    get_config_rate_limit/1,            % get_config_rate_limit(-RateLimit)
    get_config_timeout/1,               % get_config_timeout(-Timeout)
    generate_typescript_whitelist_with_config/1,  % generate_typescript_whitelist_with_config(-Code)

    % Testing
    test_rpyc_security/0
]).

:- use_module(library(lists)).

% ============================================================================
% DYNAMIC PREDICATES FOR SECURITY RULES
% ============================================================================

:- dynamic rpyc_allowed_module/2.
:- dynamic rpyc_allowed_attr/2.
:- dynamic rpyc_rate_limit/1.
:- dynamic rpyc_security_policy/2.

% Allow discontiguous definitions for security rules
:- discontiguous rpyc_allowed_module/2.
:- discontiguous rpyc_allowed_attr/2.

% ============================================================================
% DEFAULT SECURITY RULES
% ============================================================================

% Math module - safe mathematical functions
rpyc_allowed_module(math, [
    sqrt, sin, cos, tan, asin, acos, atan, atan2,
    log, log10, log2, exp, pow,
    floor, ceil, trunc, round,
    abs, fabs, fmod,
    degrees, radians,
    factorial, gcd, lcm,
    isfinite, isinf, isnan
]).

rpyc_allowed_attr(math, [pi, e, tau, inf, nan]).

% Statistics module - statistical functions
rpyc_allowed_module(statistics, [
    mean, median, mode,
    stdev, pstdev, variance, pvariance,
    harmonic_mean, geometric_mean,
    quantiles, median_low, median_high
]).

% NumPy - common array operations (safe subset)
rpyc_allowed_module(numpy, [
    % Array creation
    array, zeros, ones, empty, full,
    arange, linspace, logspace,
    eye, identity,
    % Array operations
    sum, mean, std, var,
    min, max, argmin, argmax,
    sort, argsort,
    reshape, transpose, flatten, ravel,
    concatenate, stack, vstack, hstack,
    % Math operations
    add, subtract, multiply, divide,
    power, sqrt, exp, log, log10,
    sin, cos, tan, arcsin, arccos, arctan,
    floor, ceil, round, abs,
    % Linear algebra basics
    dot, matmul, inner, outer,
    % Statistics
    median, percentile, histogram
]).

rpyc_allowed_attr(numpy, ['__version__', pi, e, inf, nan, newaxis]).

% JSON module - serialization
rpyc_allowed_module(json, [loads, dumps]).

% Datetime - time operations
rpyc_allowed_module(datetime, [now, utcnow, today, fromisoformat]).

% Default rate limit
rpyc_rate_limit(config(
    requests_per_second(100),
    burst_size(20),
    timeout_ms(5000)
)).

% ============================================================================
% RULE MANAGEMENT
% ============================================================================

%% add_allowed_module(+Module, +Functions)
%  Add or update allowed functions for a module.
add_allowed_module(Module, Functions) :-
    (   rpyc_allowed_module(Module, Existing)
    ->  retract(rpyc_allowed_module(Module, Existing)),
        append(Existing, Functions, Combined),
        sort(Combined, Unique),
        assertz(rpyc_allowed_module(Module, Unique))
    ;   assertz(rpyc_allowed_module(Module, Functions))
    ).

%% add_allowed_attr(+Module, +Attributes)
%  Add or update allowed attributes for a module.
add_allowed_attr(Module, Attributes) :-
    (   rpyc_allowed_attr(Module, Existing)
    ->  retract(rpyc_allowed_attr(Module, Existing)),
        append(Existing, Attributes, Combined),
        sort(Combined, Unique),
        assertz(rpyc_allowed_attr(Module, Unique))
    ;   assertz(rpyc_allowed_attr(Module, Attributes))
    ).

%% remove_allowed_module(+Module)
%  Remove all rules for a module.
remove_allowed_module(Module) :-
    retractall(rpyc_allowed_module(Module, _)),
    retractall(rpyc_allowed_attr(Module, _)).

%% clear_security_rules
%  Clear all dynamic security rules (keeps defaults).
clear_security_rules :-
    retractall(rpyc_allowed_module(_, _)),
    retractall(rpyc_allowed_attr(_, _)),
    retractall(rpyc_rate_limit(_)),
    retractall(rpyc_security_policy(_, _)).

% ============================================================================
% VALIDATION
% ============================================================================

%% is_call_allowed(+Module, +Function, -Allowed)
%  Check if a function call is allowed.
is_call_allowed(Module, Function, Allowed) :-
    (   rpyc_allowed_module(Module, Functions),
        member(Function, Functions)
    ->  Allowed = true
    ;   Allowed = false
    ).

%% is_attr_allowed(+Module, +Attr, -Allowed)
%  Check if an attribute access is allowed.
is_attr_allowed(Module, Attr, Allowed) :-
    (   rpyc_allowed_attr(Module, Attrs),
        member(Attr, Attrs)
    ->  Allowed = true
    ;   Allowed = false
    ).

%% validate_call(+Module, +Function, -Result)
%  Validate a call and return detailed result.
validate_call(Module, Function, Result) :-
    (   \+ rpyc_allowed_module(Module, _)
    ->  Result = error(module_not_allowed(Module))
    ;   rpyc_allowed_module(Module, Functions),
        \+ member(Function, Functions)
    ->  Result = error(function_not_allowed(Module, Function))
    ;   Result = ok
    ).

% ============================================================================
% TYPESCRIPT CODE GENERATION
% ============================================================================

%% generate_typescript_whitelist(-Code)
%  Generate TypeScript code defining the security whitelist.
generate_typescript_whitelist(Code) :-
    findall(Module-Functions, rpyc_allowed_module(Module, Functions), ModulePairs),
    findall(Module-Attrs, rpyc_allowed_attr(Module, Attrs), AttrPairs),

    % Generate module whitelist entries
    generate_module_entries(ModulePairs, ModuleEntries),
    atomic_list_concat(ModuleEntries, ',\n  ', ModuleCode),

    % Generate attribute whitelist entries
    generate_attr_entries(AttrPairs, AttrEntries),
    atomic_list_concat(AttrEntries, ',\n  ', AttrCode),

    format(atom(Code), '/**
 * RPyC Security Whitelist
 * Generated by UnifyWeaver - DO NOT EDIT
 *
 * This file defines which Python modules and functions can be called
 * through the RPyC bridge. Any call not in this whitelist will be rejected.
 */

// Allowed modules and their functions
export const ALLOWED_MODULES: Record<string, Set<string>> = {
  ~w
};

// Allowed module attributes (constants, etc.)
export const ALLOWED_ATTRS: Record<string, Set<string>> = {
  ~w
};

// Check if a function call is allowed
export function isCallAllowed(module: string, func: string): boolean {
  const allowed = ALLOWED_MODULES[module];
  return allowed !== undefined && allowed.has(func);
}

// Check if an attribute access is allowed
export function isAttrAllowed(module: string, attr: string): boolean {
  const allowed = ALLOWED_ATTRS[module];
  return allowed !== undefined && allowed.has(attr);
}

// Get all allowed modules
export function getAllowedModules(): string[] {
  return Object.keys(ALLOWED_MODULES);
}

// Get allowed functions for a module
export function getAllowedFunctions(module: string): string[] {
  const allowed = ALLOWED_MODULES[module];
  return allowed ? Array.from(allowed) : [];
}
', [ModuleCode, AttrCode]).

%% generate_module_entries(+ModulePairs, -Entries)
%  Generate TypeScript entries for module whitelist.
generate_module_entries([], []).
generate_module_entries([Module-Functions|Rest], [Entry|Entries]) :-
    atom_string(Module, ModuleStr),
    maplist(atom_string, Functions, FunctionStrs),
    maplist(quote_string, FunctionStrs, QuotedFuncs),
    atomic_list_concat(QuotedFuncs, ', ', FuncList),
    format(atom(Entry), '~w: new Set([~w])', [ModuleStr, FuncList]),
    generate_module_entries(Rest, Entries).

%% generate_attr_entries(+AttrPairs, -Entries)
%  Generate TypeScript entries for attribute whitelist.
generate_attr_entries([], []).
generate_attr_entries([Module-Attrs|Rest], [Entry|Entries]) :-
    atom_string(Module, ModuleStr),
    maplist(atom_string, Attrs, AttrStrs),
    maplist(quote_string, AttrStrs, QuotedAttrs),
    atomic_list_concat(QuotedAttrs, ', ', AttrList),
    format(atom(Entry), '~w: new Set([~w])', [ModuleStr, AttrList]),
    generate_attr_entries(Rest, Entries).

%% quote_string(+Str, -Quoted)
%  Add quotes around a string for JavaScript.
quote_string(Str, Quoted) :-
    format(atom(Quoted), '"~w"', [Str]).

%% generate_typescript_validator(-Code)
%  Generate TypeScript validation middleware.
generate_typescript_validator(Code) :-
    rpyc_rate_limit(config(
        requests_per_second(RPS),
        burst_size(Burst),
        timeout_ms(Timeout)
    )),
    format(atom(Code), '/**
 * RPyC Call Validator
 * Generated by UnifyWeaver - DO NOT EDIT
 *
 * Validates and sanitizes RPyC calls before execution.
 */

import { isCallAllowed, isAttrAllowed } from ''./whitelist'';

// Rate limiting configuration
export const RATE_LIMIT = {
  requestsPerSecond: ~w,
  burstSize: ~w,
  timeoutMs: ~w,
};

// Validation result type
export interface ValidationResult {
  valid: boolean;
  error?: string;
  sanitized?: {
    module: string;
    func?: string;
    attr?: string;
    args?: unknown[];
  };
}

// Validate a function call request
export function validateCall(
  module: unknown,
  func: unknown,
  args: unknown
): ValidationResult {
  // Type validation
  if (typeof module !== ''string'') {
    return { valid: false, error: ''Module must be a string'' };
  }
  if (typeof func !== ''string'') {
    return { valid: false, error: ''Function must be a string'' };
  }
  if (!Array.isArray(args)) {
    return { valid: false, error: ''Args must be an array'' };
  }

  // Sanitize inputs (prevent injection)
  const sanitizedModule = module.replace(/[^a-zA-Z0-9_]/g, '''');
  const sanitizedFunc = func.replace(/[^a-zA-Z0-9_]/g, '''');

  if (sanitizedModule !== module) {
    return { valid: false, error: `Invalid module name: ${module}` };
  }
  if (sanitizedFunc !== func) {
    return { valid: false, error: `Invalid function name: ${func}` };
  }

  // Whitelist check
  if (!isCallAllowed(sanitizedModule, sanitizedFunc)) {
    return {
      valid: false,
      error: `Call not allowed: ${sanitizedModule}.${sanitizedFunc}`,
    };
  }

  // Validate args (basic serialization check)
  try {
    JSON.stringify(args);
  } catch {
    return { valid: false, error: ''Args contain non-serializable values'' };
  }

  return {
    valid: true,
    sanitized: {
      module: sanitizedModule,
      func: sanitizedFunc,
      args: args as unknown[],
    },
  };
}

// Validate an attribute access request
export function validateAttr(
  module: unknown,
  attr: unknown
): ValidationResult {
  if (typeof module !== ''string'') {
    return { valid: false, error: ''Module must be a string'' };
  }
  if (typeof attr !== ''string'') {
    return { valid: false, error: ''Attribute must be a string'' };
  }

  const sanitizedModule = module.replace(/[^a-zA-Z0-9_]/g, '''');
  const sanitizedAttr = attr.replace(/[^a-zA-Z0-9_]/g, '''');

  if (sanitizedModule !== module) {
    return { valid: false, error: `Invalid module name: ${module}` };
  }
  if (sanitizedAttr !== attr) {
    return { valid: false, error: `Invalid attribute name: ${attr}` };
  }

  if (!isAttrAllowed(sanitizedModule, sanitizedAttr)) {
    return {
      valid: false,
      error: `Attribute access not allowed: ${sanitizedModule}.${sanitizedAttr}`,
    };
  }

  return {
    valid: true,
    sanitized: {
      module: sanitizedModule,
      attr: sanitizedAttr,
    },
  };
}
', [RPS, Burst, Timeout]).

%% generate_typescript_security(-Code)
%  Generate combined security module (whitelist + validator).
generate_typescript_security(Code) :-
    generate_typescript_whitelist(WhitelistCode),
    generate_typescript_validator(ValidatorCode),
    format(atom(Code), '~w

// ============================================================================

~w
', [WhitelistCode, ValidatorCode]).

%% generate_express_security_middleware(-Code)
%  Generate Express.js security middleware.
generate_express_security_middleware(Code) :-
    rpyc_rate_limit(config(
        requests_per_second(RPS),
        burst_size(Burst),
        timeout_ms(Timeout)
    )),
    format(atom(Code), '/**
 * Express Security Middleware for RPyC
 * Generated by UnifyWeaver - DO NOT EDIT
 */

import { Request, Response, NextFunction } from ''express'';
import { validateCall, validateAttr, RATE_LIMIT } from ''./validator'';

// Simple in-memory rate limiter
const requestCounts = new Map<string, { count: number; resetTime: number }>();

function getRateLimitKey(req: Request): string {
  return req.ip || req.socket.remoteAddress || ''unknown'';
}

// Rate limiting middleware
export function rateLimiter(req: Request, res: Response, next: NextFunction): void {
  const key = getRateLimitKey(req);
  const now = Date.now();
  const windowMs = 1000; // 1 second window

  let record = requestCounts.get(key);
  if (!record || now > record.resetTime) {
    record = { count: 0, resetTime: now + windowMs };
    requestCounts.set(key, record);
  }

  record.count++;

  if (record.count > ~w) {
    res.status(429).json({
      error: ''Rate limit exceeded'',
      retryAfter: Math.ceil((record.resetTime - now) / 1000),
    });
    return;
  }

  next();
}

// Request timeout middleware
export function timeoutMiddleware(req: Request, res: Response, next: NextFunction): void {
  const timeout = setTimeout(() => {
    if (!res.headersSent) {
      res.status(408).json({ error: ''Request timeout'' });
    }
  }, ~w);

  res.on(''finish'', () => clearTimeout(timeout));
  res.on(''close'', () => clearTimeout(timeout));

  next();
}

// Call validation middleware
export function validateCallMiddleware(req: Request, res: Response, next: NextFunction): void {
  const { module, func, args } = req.body;

  const result = validateCall(module, func, args);
  if (!result.valid) {
    res.status(400).json({ error: result.error });
    return;
  }

  // Attach sanitized data to request
  (req as any).sanitized = result.sanitized;
  next();
}

// Attribute validation middleware
export function validateAttrMiddleware(req: Request, res: Response, next: NextFunction): void {
  const { module, attr } = req.body;

  const result = validateAttr(module, attr);
  if (!result.valid) {
    res.status(400).json({ error: result.error });
    return;
  }

  (req as any).sanitized = result.sanitized;
  next();
}

// Combined security middleware stack
export const securityMiddleware = [
  rateLimiter,
  timeoutMiddleware,
];

// Burst limit for rate limiter
export const BURST_LIMIT = ~w;
', [RPS, Timeout, Burst]).

% ============================================================================
% TESTING
% ============================================================================
% PREFERENCES/FIREWALL INTEGRATION
% ============================================================================

%% validate_with_firewall(+Module, +Function, -Result) is det.
%
% Validate a call using both local rules and firewall system.
% This integrates with typescript_glue_config for context-aware validation.
validate_with_firewall(Module, Function, Result) :-
    % First check local security rules
    validate_call(Module, Function, LocalResult),
    (   LocalResult \= ok
    ->  Result = LocalResult
    ;   % Then check firewall if available
        (   current_module(typescript_glue_config)
        ->  catch(typescript_glue_config:validate_rpyc_call(Module, Function, FirewallResult), _, FirewallResult = ok),
            Result = FirewallResult
        ;   Result = ok
        )
    ).

%% get_config_rate_limit(-RateLimit) is det.
%
% Get rate limit from config or use default.
get_config_rate_limit(RateLimit) :-
    (   current_module(typescript_glue_config)
    ->  catch(typescript_glue_config:get_rpyc_config(Config), _, Config = []),
        (   member(rate_limit(RateLimit), Config)
        ->  true
        ;   RateLimit = 100
        )
    ;   (   rpyc_rate_limit(RateLimit)
        ->  true
        ;   RateLimit = 100
        )
    ).

%% get_config_timeout(-Timeout) is det.
%
% Get timeout from config or use default.
get_config_timeout(Timeout) :-
    (   current_module(typescript_glue_config)
    ->  catch(typescript_glue_config:get_rpyc_config(Config), _, Config = []),
        (   member(timeout(Timeout), Config)
        ->  true
        ;   Timeout = 30000
        )
    ;   Timeout = 30000
    ).

%% generate_typescript_whitelist_with_config(-Code) is det.
%
% Generate whitelist code using configuration from preferences.
generate_typescript_whitelist_with_config(Code) :-
    get_config_rate_limit(RateLimit),
    get_config_timeout(Timeout),
    generate_typescript_whitelist(BaseCode),
    format(atom(Code),
'// Configuration from UnifyWeaver preferences
export const SECURITY_CONFIG = {
  rateLimit: ~w,
  rateLimitWindow: 60000,
  timeout: ~w,
  maxDepth: 10,
  maxArrayLength: 10000
};

~w', [RateLimit, Timeout, BaseCode]).

% ============================================================================
% TESTING
% ============================================================================

test_rpyc_security :-
    format('~n=== RPyC Security Tests ===~n~n'),

    % Test validation
    format('Validation Tests:~n'),
    (   is_call_allowed(math, sqrt, true)
    ->  format('  math.sqrt: ALLOWED (OK)~n')
    ;   format('  math.sqrt: DENIED (FAIL)~n')
    ),
    (   is_call_allowed(os, system, false)
    ->  format('  os.system: DENIED (OK)~n')
    ;   format('  os.system: ALLOWED (FAIL - security issue!)~n')
    ),
    (   is_attr_allowed(math, pi, true)
    ->  format('  math.pi: ALLOWED (OK)~n')
    ;   format('  math.pi: DENIED (FAIL)~n')
    ),

    % Test code generation
    format('~nCode Generation Tests:~n'),
    (   generate_typescript_whitelist(WL),
        atom_length(WL, WLLen),
        format('  Whitelist: ~d chars~n', [WLLen])
    ;   format('  Whitelist: FAILED~n')
    ),
    (   generate_typescript_validator(V),
        atom_length(V, VLen),
        format('  Validator: ~d chars~n', [VLen])
    ;   format('  Validator: FAILED~n')
    ),
    (   generate_express_security_middleware(M),
        atom_length(M, MLen),
        format('  Middleware: ~d chars~n', [MLen])
    ;   format('  Middleware: FAILED~n')
    ),

    format('~n=== Tests Complete ===~n').
