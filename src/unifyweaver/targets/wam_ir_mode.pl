:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% WAM representation-mode policy shared by WAM targets.

:- module(wam_ir_mode, [
    wam_ir_mode/4,             % +Target, +EmitMode, +Options, -Mode
    wam_ir_mode_supported/1,   % ?Mode
    wam_ir_mode_uses_wam/1,    % +Mode
    wam_ir_mode_skips_text/1   % +Mode
]).

:- use_module(library(option)).

%% wam_ir_mode_supported(?Mode) is semidet.
%
%  Supported representation modes:
%
%    wam_text          Prolog clauses -> WAM text -> target path
%    wam_items_bridge  Prolog clauses -> WAM text -> WAM items -> target path
%    wam_items_native  Prolog clauses -> WAM items -> target path
%    direct_target     Existing non-WAM target-native codegen path
%
%  `direct_target` is an alias for a target's non-WAM emitter. It is not a WAM
%  path and WAM-specific targets may reject it even though the shared resolver
%  recognizes the name.
wam_ir_mode_supported(wam_text).
wam_ir_mode_supported(wam_items_bridge).
wam_ir_mode_supported(wam_items_native).
wam_ir_mode_supported(direct_target).

%% wam_ir_mode(+Target, +EmitMode, +Options, -Mode) is det.
%
%  Resolve the WAM representation mode for Target and EmitMode. Explicit
%  `wam_ir(Mode)` options win; otherwise the per-target default is used.
wam_ir_mode(Target, EmitMode, Options, Mode) :-
    (   option(wam_ir(Explicit), Options)
    ->  validate_wam_ir_mode(Explicit, Mode)
    ;   wam_ir_mode_default(Target, EmitMode, Mode)
    ).

validate_wam_ir_mode(Mode, Mode) :-
    wam_ir_mode_supported(Mode),
    !.
validate_wam_ir_mode(Other, _) :-
    throw(error(domain_error(wam_ir_mode, Other), wam_ir_mode/4)).

%% wam_ir_mode_uses_wam(+Mode) is semidet.
wam_ir_mode_uses_wam(wam_text).
wam_ir_mode_uses_wam(wam_items_bridge).
wam_ir_mode_uses_wam(wam_items_native).

%% wam_ir_mode_skips_text(+Mode) is semidet.
wam_ir_mode_skips_text(wam_items_native).
wam_ir_mode_skips_text(direct_target).

% Conservative current defaults. Interpreter-mode Python already consumes the
% common items bridge; lowered Python still needs WAM text for its analyzer and
% lowered emitter. Lua/R are marked as likely bridge consumers for their
% interpreter paths, but their target code has not been migrated yet.
wam_ir_mode_default(wam_python, interpreter, wam_items_bridge) :- !.
wam_ir_mode_default(wam_python, lowered, wam_text) :- !.
wam_ir_mode_default(wam_python, functions, wam_text) :- !.
wam_ir_mode_default(wam_python, mixed(_), wam_text) :- !.
wam_ir_mode_default(wam_lua, interpreter, wam_items_bridge) :- !.
wam_ir_mode_default(wam_r, interpreter, wam_items_bridge) :- !.
wam_ir_mode_default(_, direct_target, direct_target) :- !.
wam_ir_mode_default(_, _, wam_text).
