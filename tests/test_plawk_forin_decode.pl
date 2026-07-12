:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% Assoc for-in with per-entry logic, stage 3: DECODE A VALUE INTO A STRUCT.
% An END `for (k in arr) { (n, m) = dyncall@decode(arr[k]) as (i64 i64) ;
% print k, n, m }` destructures each iterated value through a grammar into
% typed fields and prints them. The per-entry value `arr[k]` is boxed and
% passed as the grammar argument (a for-in-scoped `forin_val` operand); the
% record shim and object handle come from the program-wide dyncall support
% IR (the collectors recurse into the for-in body and find the destructure).
% This is the hash-shaped counterpart to record destructure inside a
% `foreach` body -- iterate, read the entry, decode it into a struct.
% i64 and f64 fields decode into one scalar slot each; string fields (a
% (ptr,len) pair) are a separate round. See PLAWK_ASSOC_FORIN.md.

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').
:- use_module('../examples/plawk/codegen/plawk_native_codegen').
:- use_module('../src/unifyweaver/targets/wam_llvm_target').

% grammar object: decode(N) -> r(N, N+100)
decode(N, r(N, M)) :- M is N + 100.
% f64 fields: decf(N) -> r(N*1.5, N+0.25); mixed: decm(N) -> r(N, N*1.5)
decf(N, r(F1, F2)) :- F1 is N * 1.5, F2 is N + 0.25.
decm(N, r(N, F)) :- F is N * 1.5.

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(plawk_forin_decode).

% The decode body parses to a for-in whose body is a destructure over the
% for-in value `forin_val(c)` plus a print of the loop key and slots.
test(decode_parses) :-
    plawk_parse_string(
        "BEGIN { DYNLOAD = \"d.wamo\" }\n\c
         { c[$1]++ }\n\c
         END { for (k in c) { (n, m) = dyncall@decode(c[k]) as (i64 i64) ; print k, n, m } }\n",
        program(_, _, [end([for_in(var(k), var(c),
            [dynrec_bind([n, m], dyncall_named(decode, [forin_val(c)]),
                 [i64, i64]),
             print([var(k), var(n), var(m)])])])])),
    !.

% End to end: counts a=3, b=2, c=1; decode(N) = r(N, N+100). Each entry
% prints "key count count+100" (order-independent).
test(decode_runs, [condition(clang_available)]) :-
    fd_dir(Dir),
    decode_wamo(Dir, Wamo),
    format(string(Src),
        "BEGIN { DYNLOAD = \"~w\" }\n\c
         { c[$1]++ }\n\c
         END { for (k in c) { (n, m) = dyncall@decode(c[k]) as (i64 i64) ; print k, n, m } }\n",
        [Wamo]),
    build_run_text(Dir, 'dec', Src, "a\na\na\nb\nb\nc\n", Out),
    split_string(Out, "\n", "", L0), exclude(==(""), L0, L), msort(L, S),
    assertion(S == ["a 3 103", "b 2 102", "c 1 101"]),
    !.

% Print only the decoded fields (no key): the destructured struct stands
% on its own. Same counts -> "count count+100" per entry.
test(decode_fields_only_runs, [condition(clang_available)]) :-
    fd_dir(Dir),
    decode_wamo(Dir, Wamo),
    format(string(Src),
        "BEGIN { DYNLOAD = \"~w\" }\n\c
         { c[$1]++ }\n\c
         END { for (k in c) { (n, m) = dyncall@decode(c[k]) as (i64 i64) ; print n, m } }\n",
        [Wamo]),
    build_run_text(Dir, 'decf', Src, "a\na\na\nb\nb\nc\n", Out),
    split_string(Out, "\n", "", L0), exclude(==(""), L0, L), msort(L, S),
    assertion(S == ["1 101", "2 102", "3 103"]),
    !.

% A string literal alongside the decoded fields.
test(decode_labeled_runs, [condition(clang_available)]) :-
    fd_dir(Dir),
    decode_wamo(Dir, Wamo),
    format(string(Src),
        "BEGIN { DYNLOAD = \"~w\" }\n\c
         { c[$1]++ }\n\c
         END { for (k in c) { (n, m) = dyncall@decode(c[k]) as (i64 i64) ; print \"e\", n, m } }\n",
        [Wamo]),
    build_run_text(Dir, 'decl', Src, "a\na\nb\n", Out),
    split_string(Out, "\n", "", L0), exclude(==(""), L0, L), msort(L, S),
    assertion(S == ["e 1 101", "e 2 102"]),
    !.

% f64 decoded fields: decf(N) = r(N*1.5, N+0.25). Counts a=3,b=2,c=1 ->
% a:(4.5,3.25) b:(3,2.25) c:(1.5,1.25) (%g drops the trailing .0).
test(decode_f64_runs, [condition(clang_available)]) :-
    fd_dir(Dir),
    named_wamo(Dir, decf, Wamo),
    format(string(Src),
        "BEGIN { DYNLOAD = \"~w\" }\n\c
         { c[$1]++ }\n\c
         END { for (k in c) { (a, b) = dyncall@decf(c[k]) as (f64 f64) ; print k, a, b } }\n",
        [Wamo]),
    build_run_text(Dir, 'decff', Src, "a\na\na\nb\nb\nc\n", Out),
    split_string(Out, "\n", "", L0), exclude(==(""), L0, L), msort(L, S),
    assertion(S == ["a 4.5 3.25", "b 3 2.25", "c 1.5 1.25"]),
    !.

% Mixed i64 + f64 fields in one destructure: decm(N) = r(N, N*1.5).
test(decode_mixed_runs, [condition(clang_available)]) :-
    fd_dir(Dir),
    named_wamo(Dir, decm, Wamo),
    format(string(Src),
        "BEGIN { DYNLOAD = \"~w\" }\n\c
         { c[$1]++ }\n\c
         END { for (k in c) { (n, f) = dyncall@decm(c[k]) as (i64 f64) ; print k, n, f } }\n",
        [Wamo]),
    build_run_text(Dir, 'decmm', Src, "a\na\na\nb\nb\nc\n", Out),
    split_string(Out, "\n", "", L0), exclude(==(""), L0, L), msort(L, S),
    assertion(S == ["a 3 4.5", "b 2 3", "c 1 1.5"]),
    !.

:- end_tests(plawk_forin_decode).

% --- helpers ---------------------------------------------------------------

fd_dir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_forin_decode', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

decode_wamo(Dir, Wamo) :-
    directory_file_path(Dir, 'decode.wamo', Wamo),
    ( exists_file(Wamo) -> true
    ; write_wam_object([user:decode/2], [wamo_entries([decode/2])], Wamo) ).

% A .wamo exporting Name/2 (e.g. decf, decm), built on first use.
named_wamo(Dir, Name, Wamo) :-
    atom_concat(Name, '.wamo', File),
    directory_file_path(Dir, File, Wamo),
    ( exists_file(Wamo) -> true
    ; write_wam_object([user:Name/2], [wamo_entries([Name/2])], Wamo) ).

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
