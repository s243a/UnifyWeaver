:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)

:- use_module(library(plunit)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module(library(process)).
:- use_module('helpers/smoke_paths', [tmp_root/1, clean_dir/1]).
:- use_module('../src/unifyweaver/targets/wam_llvm_target').
:- use_module('../examples/plawk/parser/plawk_parser').
:- use_module('../examples/plawk/codegen/plawk_native_codegen').

:- dynamic user:plawk_is_error/1.
:- dynamic user:plawk_severity_rank/2.
:- dynamic user:plawk_double_plus/2.

% Fact-based guard: succeeds for the severities considered errors.
user:plawk_is_error('ERROR').
user:plawk_is_error('FATAL').

% Deterministic classifier: atom in, integer rank out.
user:plawk_severity_rank(Kind, Rank) :-
    (   Kind == 'ERROR'
    ->  Rank = 3
    ;   Kind == 'WARN'
    ->  Rank = 2
    ;   Rank = 1
    ).

% Integer arithmetic through is/2.
user:plawk_double_plus(N, M) :-
    M is N * 2 + 1.

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(plawk_surface_prolog_calls, [condition(clang_available)]).

test(parses_prolog_guard_pattern) :-
    plawk_parse_string("plawk_is_error($1) { print $0 }\n", Program),
    assertion(Program == program([], [rule(
        prolog_guard(plawk_is_error, [field(1)]),
        [print([field(0)])])], [])).

test(parses_prolog_guard_in_combined_pattern) :-
    plawk_parse_string("plawk_is_error($1) && $3 > 100 { hits++ } END { print hits }\n", Program),
    assertion(Program == program([], [rule(
        and_pat(prolog_guard(plawk_is_error, [field(1)]), field_cmp(3, gt, 100)),
        [inc(var(hits))])],
        [end([print([var(hits)])])])).

test(parses_prolog_call_expression) :-
    plawk_parse_string("{ rank = plawk_severity_rank($1); total += rank } END { print total }\n", Program),
    assertion(Program == program([], [rule(always,
        [set(var(rank), prolog_call(plawk_severity_rank, [field(1)])),
         add(var(total), var(rank))])],
        [end([print([var(total)])])])).

test(parses_prolog_call_with_mixed_args) :-
    plawk_parse_string("check($0, \"limit\", -5) { hits++ } END { print hits }\n", Program),
    assertion(Program == program([], [rule(
        prolog_guard(check, [field(0), string("limit"), int(-5)]),
        [inc(var(hits))])],
        [end([print([var(hits)])])])).

test(foreign_specs_collect_guards_and_calls) :-
    plawk_parse_string("plawk_is_error($1) { total += plawk_severity_rank($1) } END { print total }\n", Program),
    plawk_program_foreign_specs(Program, GuardSpecs, CallSpecs),
    assertion(GuardSpecs == [plawk_is_error-1]),
    assertion(CallSpecs == [plawk_severity_rank-1]).

test(foreign_driver_requires_vm_counts) :-
    plawk_parse_string("plawk_is_error($1) { print $0 }\n", Program),
    assertion(\+ plawk_program_native_driver_ir(Program, 'input.txt', [], _)),
    !.

test(plain_program_needs_no_vm_counts) :-
    plawk_parse_string("$1 == \"ERROR\" { print $0 }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.txt', [], DriverIR),
    assertion(\+ sub_atom(DriverIR, _, _, _, '@plawk_foreign_vm')),
    !.

test(foreign_driver_ir_has_wrappers_and_lazy_vm) :-
    plawk_parse_string("plawk_is_error($1) { total += plawk_severity_rank($1) } END { print total }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.txt', [wam_vm(100, 20)], DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _, 'define i1 @plawk_foreign_guard_plawk_is_error_1(%Value %a0)'))),
    assertion(once(sub_atom(DriverIR, _, _, _, 'define { i64, i1 } @plawk_foreign_call_plawk_severity_rank_1(%Value %a0)'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '@plawk_foreign_vm = internal global %WamState* null'))),
    assertion(once(sub_atom(DriverIR, _, _, _, 'load i32, i32* @plawk_is_error_start_pc'))),
    % Wrappers rewind the heap top and arena after every call.
    assertion(once(sub_atom(DriverIR, _, _, _, '%hs_saved = load i32, i32* %hs_ptr\n  %pc = load i32, i32* @plawk_is_error_start_pc'))),
    !.

test(surface_prolog_fact_guard_filters_records) :-
    run_prolog_call_smoke("plawk_is_error($1) { print $0 }\n",
        "ERROR disk 300\nWARN cpu 50\nINFO net 15\nFATAL mem 8\n",
        "ERROR disk 300\nFATAL mem 8\n").

test(surface_prolog_guard_composes_with_native_guard) :-
    run_prolog_call_smoke("plawk_is_error($1) && $3 > 100 { hits++ } END { print hits }\n",
        "ERROR disk 300\nWARN cpu 50\nINFO net 15\nFATAL mem 8\n",
        "1\n").

test(surface_prolog_call_accumulates_ranks) :-
    run_prolog_call_smoke("{ total += plawk_severity_rank($1) } END { print total }\n",
        "ERROR disk 300\nWARN cpu 50\nINFO net 15\nFATAL mem 8\n",
        "7\n").

test(surface_prolog_call_prints_per_record) :-
    run_prolog_call_smoke("{ print $1, plawk_severity_rank($1) }\n",
        "ERROR disk 300\nWARN cpu 50\nINFO net 15\nFATAL mem 8\n",
        "ERROR 3\nWARN 2\nINFO 1\nFATAL 1\n").

test(surface_prolog_call_with_integer_literal_arg) :-
    run_prolog_call_smoke("{ print plawk_double_plus(21) }\n",
        "one row\n",
        "43\n").

test(surface_prolog_guard_in_if_condition) :-
    run_prolog_call_smoke("{ if (plawk_is_error($1)) { e++ } } END { print e }\n",
        "ERROR disk 300\nWARN cpu 50\nINFO net 15\nFATAL mem 8\n",
        "2\n").

test(surface_prolog_call_composes_with_arithmetic) :-
    run_prolog_call_smoke("{ r = plawk_severity_rank($1) * 10 + 1; total += r } END { print total }\n",
        "ERROR disk 300\nWARN cpu 50\nINFO net 15\nFATAL mem 8\n",
        "74\n").

test(surface_negated_prolog_guard) :-
    run_prolog_call_smoke("!plawk_is_error($1) { ok++ } END { print ok }\n",
        "ERROR disk 300\nWARN cpu 50\nINFO net 15\nFATAL mem 8\n",
        "2\n").

run_prolog_call_smoke(Source, Input, ExpectedOutput) :-
    tmp_root(Root),
    directory_file_path(Root, 'uw_plawk_surface_prolog_calls', Dir),
    clean_dir(Dir),
    make_directory_path(Dir),
    directory_file_path(Dir, 'input.txt', InputPath),
    setup_call_cleanup(
        open(InputPath, write, In, [type(binary)]),
        format(In, '~s', [Input]),
        close(In)),
    plawk_parse_string(Source, Program),
    directory_file_path(Dir, 'plawk_surface_prolog_calls.ll', LLPath),
    write_wam_llvm_project(
        [ user:plawk_is_error/1,
          user:plawk_severity_rank/2,
          user:plawk_double_plus/2
        ],
        [module_name('plawk_surface_prolog_calls')], LLPath),
    wam_llvm_last_compile_counts(InstrCount, LabelCount),
    plawk_program_native_driver_ir(Program, InputPath,
        [wam_vm(InstrCount, LabelCount)], DriverIR),
    setup_call_cleanup(
        open(LLPath, append, Out, [encoding(utf8)]),
        ( nl(Out), write(Out, DriverIR) ),
        close(Out)),
    directory_file_path(Dir, 'plawk_surface_prolog_calls_bin', BinPath),
    format(atom(Cmd), 'clang -w ~w -o ~w -lm 2>&1 && ~w',
        [LLPath, BinPath, BinPath]),
    process_create(path(sh), ['-c', Cmd],
                   [stdout(pipe(Stdout)), stderr(std), process(Pid)]),
    read_string(Stdout, _, OutStr),
    close(Stdout),
    process_wait(Pid, Status),
    ( Status == exit(0)
    -> assertion(OutStr == ExpectedOutput)
    ;  format(user_error, "~n[plawk surface prolog calls output]~n~w~n",
              [OutStr]),
       throw(plawk_surface_prolog_calls_failed(Status))
    ),
    !.

:- end_tests(plawk_surface_prolog_calls).
