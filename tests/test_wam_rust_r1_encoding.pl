:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (s243a)
%
% test_wam_rust_r1_encoding.pl
%
% Phase R1 of docs/proposals/wam_rust_heap_cell_rewrite_design.md (§4): the
% Rust WAM generator's instruction encoding. Generator-level checks (no cargo):
%
%   * register operands: rust_reg_literal/2 emits Reg::a/x/y(n) for the
%     canonical names, Reg::x(99 - N mod 99) for the aggregate template
%     registers `_XT<N>` (bug-fix batch 1, B6), the raw encoding of
%     `Reg::from_name` for any other name the runtime decodes, and refuses
%     the Y spellings the runtime keys by text;
%   * builtin routes: the core builtins get their own route, every other name
%     enters the cascade at the first family that names it; that is exact only
%     while each family claims names through literal match arms alone, which
%     is pinned here;
%   * PC-resolved control: every PC a generated shared table branches to is
%     the PC of one of its labels, and the by-name forms remain only where the
%     runtime needs them (lowered_call targets, unlabelled targets).
%
% The generated crate's `mod r1_reg_operand_tests` checks the runtime side of
% the register encoding.
%
%   swipl -q -g run_tests -t halt tests/test_wam_rust_r1_encoding.pl

:- use_module(library(plunit)).
:- use_module(library(lists)).
:- use_module(library(readutil)).
:- use_module('../src/unifyweaver/targets/wam_rust_target').
:- use_module('../src/unifyweaver/targets/wam_rust_reg').

:- begin_tests(wam_rust_r1_encoding).

% --- registers ---------------------------------------------------------------

test(reg_literals) :-
    forall(member(Name-Expect,
                  [ 'A1'-"Reg::a(1)", 'A42'-"Reg::a(42)", 'X3'-"Reg::x(3)",
                    'X150'-"Reg::x(150)", 'Y1'-"Reg::y(1)", 'Y100'-"Reg::y(100)",
                    "A2"-"Reg::a(2)",
                    % non-canonical names: the raw Reg::from_name encoding
                    % the template scratch registers (bug-fix batch 1, B6)
                    '_XT0'-"Reg::x(99)", '_XT7'-"Reg::x(92)",
                    'A01'-"Reg(16384)",
                    'X0'-"Reg(99)", 'A0'-"Reg(32767)", 'Y'-"Reg(199)",
                    'B7'-"Reg(0)", 'A+3'-"Reg(16386)" ]),
           (   rust_reg_literal(Name, Got),
               (   Got == Expect
               ->  true
               ;   format(user_error, '~w: got ~w, expected ~w~n', [Name, Got, Expect]),
                   fail
               )
           )).

test(text_keyed_y_names_are_refused) :-
    forall(member(Name, ['Y0', 'Y01', 'Y1x', 'Y123456']),
           catch(( rust_reg_literal(Name, _), fail ),
                 error(domain_error(wam_rust_register, _), _),
                 true)).

% --- builtin routes ----------------------------------------------------------

family(4, arith).
family(5, io).
family(6, type).
family(7, term).
family(8, ext).
family(9, meta).

family_code(Family, Code) :-
    atomic_list_concat([compile_execute_, Family, '_builtin_to_rust'], Pred),
    Goal =.. [Pred, Code],
    call(wam_rust_target:Goal).

test(core_routes) :-
    forall(member(N-R, ["true/0"-0, "fail/0"-1, "!/0"-2, "=/2"-3]),
           wam_rust_target:rust_builtin_route(N, R)).

test(family_routes) :-
    forall(member(N-R, ["is/2"-4, "format/2"-5, "atom/1"-6, "functor/3"-7,
                        "msort/2"-8, "call/1"-9, "no_such_builtin/7"-10]),
           (   wam_rust_target:rust_builtin_route(N, Got),
               assertion(Got == R)
           )).

% The route skips the families before the first one that names a builtin.
% That is exact only if a family claims names solely through literal match
% arms: each family is either `match op { "..." | ... => ..., _ => false }`
% (every top-level arm a string literal, an or-continuation, or the final
% `_ => false`), or opens with an `if !matches!(op, <literals>) { return
% false; }` filter.
test(families_claim_only_literal_names, [forall(family(_, F))]) :-
    family_code(F, Code),
    split_string(Code, "\n", "", AllLines),
    format(string(Head), "fn execute_~w_builtin(", [F]),
    append(_, [H|Rest], AllLines),
    sub_string(H, _, _, _, Head),
    !,
    append(Lines, ["    }"|_], Rest),
    !,
    (   member(L, Lines), sub_string(L, _, _, _, "if !matches!(op,")
    ->  true
    ;   forall(( member(L, Lines),
                 sub_string(L, 0, 12, _, "            "),
                 sub_string(L, 12, 1, _, C), C \== " ",
                 sub_string(L, _, _, _, "=>") ),
               (   ( C == "\"" ; C == "|" ; sub_string(L, 12, _, _, "_ => false") )
               ->  true
               ;   format(user_error, '~w: non-literal arm: ~w~n', [F, L]),
                   fail
               ))
    ).

% --- PC-resolved control -------------------------------------------------------

r1_prog_clauses([
    (r1c(a)), (r1c(b)), (r1c(c)),
    (r1d(X, Y) :- r1c(X), ( X == b -> Y = yes ; Y = no )),
    (r1e(L, N) :- findall(X, r1c(X), L), length(L, N)),
    (r1f(X) :- r1d(X, _), !)
]).

:- dynamic user:r1c/1, user:r1d/2, user:r1e/2, user:r1f/1.

install_r1_prog :-
    forall(member(H, [r1c(_), r1d(_, _), r1e(_, _), r1f(_)]),
           retractall(user:H)),
    r1_prog_clauses(Cs),
    forall(member(C, Cs), assertz(user:C)).

r1_lib(S) :-
    install_r1_prog,
    Dir = 'output/rust_wam_r1_encoding',
    write_wam_rust_project([user:r1c/1, user:r1d/2, user:r1e/2, user:r1f/1],
                           [module_name(r1_encoding), emit_mode(interpreter),
                            fact_table_inline(false)], Dir),
    atomic_list_concat([Dir, '/src/lib.rs'], LibRs),
    read_file_to_string(LibRs, S, []).

label_pcs(S, PCs) :-
    split_string(S, "\n", " ", Lines),
    findall(PC,
            ( member(L, Lines),
              string_concat("labels.insert(", _, L),
              sub_string(L, B, _, _, ".to_string(), "),
              E is B + 14,
              sub_string(L, E, _, 0, T),
              split_string(T, ")", "", [PCS|_]),
              number_string(PC, PCS) ),
            PCs0),
    sort(PCs0, PCs).

pc_operands(S, Ops) :-
    findall(PC,
            ( member(Op, ["CallPc(", "ExecutePc(", "TryMeElsePc(", "TryMeElsePcArgs(",
                          "TryMeElseItePc(", "RetryMeElsePc(", "TryPc(", "RetryPc(",
                          "TrustPc(", "JumpPc("]),
              format(string(Needle), "Instruction::~w", [Op]),
              sub_string(S, B, L, _, Needle),
              E is B + L,
              sub_string(S, E, 8, _, T),
              split_string(T, ",)", "", [PCS|_]),
              number_string(PC, PCS) ),
            Ops).

test(pc_operands_are_label_pcs) :-
    r1_lib(S),
    label_pcs(S, Labels),
    pc_operands(S, Ops),
    Ops \== [],
    forall(member(PC, Ops), memberchk(PC, Labels)),
    % the clause chain, the if-then-else guard and the calls all resolved
    once(sub_string(S, _, _, _, "Instruction::RetryMeElsePc(")),
    once(sub_string(S, _, _, _, "Instruction::TryMeElseItePc(")),
    once(sub_string(S, _, _, _, "Instruction::CallPc(")),
    \+ sub_string(S, _, _, _, "Instruction::TryMeElse(\""),
    \+ sub_string(S, _, _, _, "Instruction::RetryMeElse(\""),
    \+ sub_string(S, _, _, _, "Instruction::Jump(\""),
    % the call to r1c/1 targets its label
    sub_string(S, _, _, _, "labels.insert(\"r1c/1\".to_string(), "), !.

test(builtins_are_routed) :-
    r1_lib(S),
    once(sub_string(S, _, _, _, "Instruction::BuiltinCallR(")),
    \+ sub_string(S, _, _, _, "Instruction::BuiltinCall(\""),
    !.

test(named_calls_option_keeps_names) :-
    install_r1_prog,
    Dir = 'output/rust_wam_r1_encoding_named',
    write_wam_rust_project([user:r1c/1, user:r1d/2, user:r1e/2, user:r1f/1],
                           [module_name(r1_encoding_named), emit_mode(interpreter),
                            fact_table_inline(false), rust_named_calls(true)], Dir),
    atomic_list_concat([Dir, '/src/lib.rs'], LibRs),
    read_file_to_string(LibRs, S, []),
    once(sub_string(S, _, _, _, "Instruction::Call(\"r1c/1\"")),
    once(sub_string(S, _, _, _, "Instruction::BuiltinCall(\"")),
    \+ sub_string(S, _, _, _, "Instruction::CallPc("),
    \+ sub_string(S, _, _, _, "Instruction::BuiltinCallR("),
    !.

:- end_tests(wam_rust_r1_encoding).
