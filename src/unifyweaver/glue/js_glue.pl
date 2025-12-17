% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% js_glue.pl - JavaScript Runtime Selection and Compatibility
%
% Provides variant selection for JavaScript runtimes (Node, Deno, Bun, Browser)
% similar to how dotnet_glue.pl handles Python variants (CPython, IronPython).
%
% Pattern borrowed from: dotnet_glue.pl python_runtime_choice/2

:- encoding(utf8).

:- module(js_glue, [
    % Runtime selection
    js_runtime_choice/2,            % js_runtime_choice(+Features, -Runtime)
    
    % Compatibility checking
    node_supports/1,                % node_supports(+Feature)
    deno_supports/1,                % deno_supports(+Feature)
    bun_supports/1,                 % bun_supports(+Feature)
    browser_supports/1,             % browser_supports(+Feature)
    
    % Runtime detection
    detect_node/1,                  % detect_node(-Version)
    detect_deno/1,                  % detect_deno(-Version)
    detect_bun/1,                   % detect_bun(-Version)
    
    % Module compatibility
    can_use_runtime/2,              % can_use_runtime(+Imports, +Runtime)
    js_module_compatible/2          % js_module_compatible(+Module, +Runtime)
]).

:- use_module(library(lists)).

%% ============================================
%% RUNTIME SELECTION
%% ============================================

%% js_runtime_choice(+Features, -Runtime)
%  Choose the appropriate JavaScript runtime based on required features.
%  Runtime = node | deno | bun | browser
%
js_runtime_choice(Features, deno) :-
    member(typescript, Features),
    member(secure, Features), !.
js_runtime_choice(Features, deno) :-
    member(permissions, Features), !.
js_runtime_choice(Features, bun) :-
    member(fast, Features),
    member(bundled, Features), !.
js_runtime_choice(Features, browser) :-
    member(dom, Features), !.
js_runtime_choice(Features, browser) :-
    member(localStorage, Features), !.
js_runtime_choice(Features, browser) :-
    member(document, Features), !.
js_runtime_choice(Features, node) :-
    member(npm, Features), !.
js_runtime_choice(Features, node) :-
    member(filesystem, Features), !.
js_runtime_choice(_, node).  % Default to Node.js

%% ============================================
%% NODE.JS COMPATIBILITY
%% ============================================

%% node_supports(+Feature)
%  True if Node.js supports the given feature/module.
%
% Core modules
node_supports(fs).
node_supports(path).
node_supports(http).
node_supports(https).
node_supports(crypto).
node_supports(buffer).
node_supports(stream).
node_supports(events).
node_supports(util).
node_supports(os).
node_supports(child_process).
node_supports(cluster).
node_supports(readline).
node_supports(net).
node_supports(dgram).
node_supports(dns).
node_supports(url).
node_supports(querystring).
node_supports(zlib).
node_supports(assert).
node_supports(process).

% Built-in globals
node_supports('__dirname').
node_supports('__filename').
node_supports(require).
node_supports(exports).
node_supports(module).

% npm ecosystem
node_supports(npm).
node_supports(package_json).

%% ============================================
%% DENO COMPATIBILITY
%% ============================================

%% deno_supports(+Feature)
%  True if Deno supports the given feature.
%
% Native TypeScript
deno_supports(typescript).

% Deno namespace APIs
deno_supports('Deno.readTextFile').
deno_supports('Deno.writeTextFile').
deno_supports('Deno.serve').
deno_supports('Deno.Command').
deno_supports('Deno.cwd').
deno_supports('Deno.env').
deno_supports('Deno.exit').
deno_supports('Deno.args').

% Web standard APIs
deno_supports(fetch).
deno_supports('Request').
deno_supports('Response').
deno_supports('URL').
deno_supports('URLSearchParams').
deno_supports('TextEncoder').
deno_supports('TextDecoder').
deno_supports('crypto').  % Web Crypto API

% Security
deno_supports(permissions).
deno_supports(secure).

%% ============================================
%% BUN COMPATIBILITY
%% ============================================

%% bun_supports(+Feature)
%  True if Bun supports the given feature.
%
% Node.js compatibility
bun_supports(fs).
bun_supports(path).
bun_supports(http).
bun_supports(crypto).

% Bun-specific
bun_supports('Bun.serve').
bun_supports('Bun.file').
bun_supports('Bun.write').
bun_supports('Bun.spawn').

% Performance
bun_supports(fast).
bun_supports(bundled).
bun_supports(npm_compat).

%% ============================================
%% BROWSER COMPATIBILITY
%% ============================================

%% browser_supports(+Feature)
%  True if browsers support the given feature.
%
% DOM APIs
browser_supports(document).
browser_supports(window).
browser_supports(navigator).
browser_supports(localStorage).
browser_supports(sessionStorage).
browser_supports(indexedDB).

% Fetch API
browser_supports(fetch).
browser_supports('Request').
browser_supports('Response').
browser_supports('Headers').

% Web APIs
browser_supports(dom).
browser_supports('addEventListener').
browser_supports('querySelector').
browser_supports('getElementById').
browser_supports('createElement').

% Modern APIs
browser_supports('Promise').
browser_supports('async').
browser_supports('await').
browser_supports('WebSocket').
browser_supports('Worker').

%% ============================================
%% RUNTIME DETECTION
%% ============================================

%% detect_node(-Version)
%  Detect if Node.js is available and get version.
%
detect_node(Version) :-
    catch(
        (process_create(path(node), ['--version'], [stdout(pipe(S))]),
         read_line_to_string(S, Version),
         close(S)),
        _, fail
    ).

%% detect_deno(-Version)
%  Detect if Deno is available and get version.
%
detect_deno(Version) :-
    catch(
        (process_create(path(deno), ['--version'], [stdout(pipe(S))]),
         read_line_to_string(S, Version),
         close(S)),
        _, fail
    ).

%% detect_bun(-Version)
%  Detect if Bun is available and get version.
%
detect_bun(Version) :-
    catch(
        (process_create(path(bun), ['--version'], [stdout(pipe(S))]),
         read_line_to_string(S, Version),
         close(S)),
        _, fail
    ).

%% ============================================
%% MODULE COMPATIBILITY
%% ============================================

%% can_use_runtime(+Imports, +Runtime)
%  True if all imports in the list are compatible with the runtime.
%
can_use_runtime([], _Runtime).
can_use_runtime([Import|Rest], Runtime) :-
    js_module_compatible(Import, Runtime),
    can_use_runtime(Rest, Runtime).

%% js_module_compatible(+Module, +Runtime)
%  Check if a module is compatible with a runtime.
%
js_module_compatible(Module, node) :-
    node_supports(Module), !.
js_module_compatible(Module, deno) :-
    deno_supports(Module), !.
js_module_compatible(Module, bun) :-
    bun_supports(Module), !.
js_module_compatible(Module, browser) :-
    browser_supports(Module), !.
% Universal modules (work everywhere)
js_module_compatible(console, _).
js_module_compatible(JSON, _) :- atom(JSON), JSON == 'JSON'.
js_module_compatible('Math', _).
js_module_compatible('Date', _).
js_module_compatible('Array', _).
js_module_compatible('Object', _).
js_module_compatible('String', _).
js_module_compatible('Number', _).
js_module_compatible('Promise', _).
