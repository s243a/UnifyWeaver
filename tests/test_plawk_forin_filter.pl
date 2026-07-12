:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% Assoc for-in with per-entry logic, stage 1: a FILTER in the rule-body
% for-in. `for (k in arr) { if (GUARD) print ... }` where GUARD compares
% the loop key `k` or the iterated value `arr[k]` to an integer, gating
% the per-key print. `k` / `arr[k]` are for-in-scoped operands (they exist
% only inside the loop), parsed by a for-in-body-local guard grammar and
% lowered by the assoc rule-chain loop emitter. See PLAWK_ASSOC_FORIN.md.
% (Per-record running-snapshot semantics, like every rule-body for-in;
% the END-side filter is stage 1b.)

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').
:- use_module('../examples/plawk/codegen/plawk_native_codegen').

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(plawk_forin_filter).

% Value guard: `arr[k] CMP int` parses to a for-in body if wrapping a
% print, with a forin_val_cmp guard node.
test(value_guard_parses) :-
    plawk_parse_string(
        "{ c[$1]++ ; for (k in c) { if (c[k] >= 2) print k, c[k] } }\n\c
         END { print c[\"a\"] }\n",
        program(_, [rule(_, Actions)], _)),
    memberchk(for_in(var(k), var(c),
        [if(forin_val_cmp(c, k, ge, 2), [print([var(k), assoc(var(c), var(k))])], [])]),
        Actions),
    !.

% Key guard: `k CMP int` parses to a forin_key_cmp guard node.
test(key_guard_parses) :-
    plawk_parse_string(
        "{ c[$1]++ ; for (k in c) { if (k > 0) print k } }\n\c
         END { print c[\"a\"] }\n",
        program(_, [rule(_, Actions)], _)),
    memberchk(for_in(var(k), var(c),
        [if(forin_key_cmp(k, gt, 0), [print([var(k)])], [])]), Actions),
    !.

% Value filter, end to end. Input a a b (per-record running snapshot):
%   rec "a" -> c={a:1}: 1>=2 no.
%   rec "a" -> c={a:2}: a passes -> "a 2".
%   rec "b" -> c={a:2,b:1}: a passes -> "a 2"; b (1) no.
% END prints c["a"] = 2. Lines (order-independent): 2, a 2, a 2.
test(value_filter_runs, [condition(clang_available)]) :-
    ff_dir(Dir),
    Src = "{ c[$1]++ ; for (k in c) { if (c[k] >= 2) print k, c[k] } }\n\c
           END { print c[\"a\"] }\n",
    build_run_text(Dir, 'ffv', Src, "a\na\nb\n", Out),
    split_string(Out, "\n", "", L0), exclude(==(""), L0, L), msort(L, S),
    assertion(S == ["2", "a 2", "a 2"]),
    !.

% A stricter threshold filters everything out except the END read.
% Input a a b, guard arr[k] >= 3: nothing ever reaches 3, so no loop
% output; END prints c["a"] = 2.
test(value_filter_excludes_all, [condition(clang_available)]) :-
    ff_dir(Dir),
    Src = "{ c[$1]++ ; for (k in c) { if (c[k] >= 3) print k, c[k] } }\n\c
           END { print c[\"a\"] }\n",
    build_run_text(Dir, 'ffx', Src, "a\na\nb\n", Out),
    assertion(Out == "2\n"),
    !.

:- end_tests(plawk_forin_filter).

% --- helpers ---------------------------------------------------------------

ff_dir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_forin_filter', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

build_run_text(Dir, Name, Src, InputText, Out) :-
    directory_file_path(Dir, Name, Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    atom_concat(Prog0, '_bin', Bin),
    cli([build, Prog, '-o', Bin], _, 0),
    atom_concat(Prog0, '_in.txt', Input),
    setup_call_cleanup(open(Input, write, SI, [encoding(utf8)]),
        write(SI, InputText), close(SI)),
    run_bin(Bin, [Input], Out, 0).

cli(Args, Out, ExpectedStatus) :-
    process_create(path(swipl), ['examples/plawk/bin/plawk' | Args],
        [stdout(pipe(S)), stderr(std), process(Pid)]),
    read_string(S, _, Out),
    close(S),
    process_wait(Pid, exit(Status)),
    assertion(Status == ExpectedStatus).

run_bin(Bin, Args, Out, ExpectedStatus) :-
    process_create(Bin, Args,
        [stdout(pipe(S)), stderr(std), process(Pid)]),
    read_string(S, _, Out),
    close(S),
    process_wait(Pid, exit(Status)),
    assertion(Status == ExpectedStatus).
