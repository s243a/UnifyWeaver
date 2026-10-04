#!/usr/bin/env swipl
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% ir_corpus -- the IR corpus tool for refactors that must not change generated
% code (docs/design/PLAN_TEMPLATE_REFACTOR.md §5).
%
% Every program string in tests/test_plawk_*.pl is a corpus entry. A refactor is
% checked by dumping the corpus before and after and diffing: an unchanged program
% must produce byte-identical IR, and every change is listed with its program.
%
%   swipl examples/plawk/scripts/ir_corpus.pl dump OUT.txt
%       driver IR (plawk_program_native_driver_ir/3) of every corpus program, or
%       DECLINE, each with the sha1 of its IR. Fast: no clang.
%   swipl examples/plawk/scripts/ir_corpus.pl diff A.txt B.txt
%       the programs whose result differs, with the status change; exit 1 if any.
%   swipl examples/plawk/scripts/ir_corpus.pl golden LIST.txt OUTDIR
%       FULL module (runtime included) for each program in LIST (one corpus line
%       per line, as written by `list`), via `plawk build --keep-ll`, as
%       OUTDIR/NNNN.ll. Compare two OUTDIRs with `diff -r` / `cmp`.
%   swipl examples/plawk/scripts/ir_corpus.pl list OUT.txt
%       the corpus itself, one program per line (escaped), for `golden`.
%
% Run from the repository root. Files are always read and written as UTF-8: a dump
% written through a locale-encoded stream under LC_ALL=C was silently truncated at
% the first non-ASCII program (a crash mid-dump that looked like a short corpus).

:- initialization(main, main).

:- use_module(library(lists)).
:- use_module(library(apply)).
:- use_module(library(process)).
:- use_module(library(sha)).
:- use_module(library(filesex), [make_directory_path/1, directory_file_path/3]).
:- use_module('../parser/plawk_parser').
:- use_module('../codegen/llvm/plawk_native_codegen').

main([dump, Out]) :- !, corpus(Entries), dump(Entries, Out).
main([diff, A, B]) :- !, diff(A, B, Changed), ( Changed =:= 0 -> halt(0) ; halt(1) ).
main([golden, List, OutDir]) :- !, golden(List, OutDir).
main([list, Out]) :- !, corpus(Entries), write_list(Entries, Out).
main(_) :-
    format(user_error, "usage: ir_corpus.pl dump OUT | diff A B | golden LIST OUTDIR | list OUT~n", []),
    halt(2).

%% corpus(-Entries) is det.
%  Every distinct double-quoted string literal containing `{` in
%  tests/test_plawk_*.pl, in file order, as Escaped-Source pairs (Escaped is the
%  literal as written; Source its value).
corpus(Entries) :-
    expand_file_name('tests/test_plawk_*.pl', Files0),
    msort(Files0, Files),
    foldl(file_literals, Files, [], Rev),
    reverse(Rev, Entries).

file_literals(File, Acc0, Acc) :-
    read_file_to_codes(File, Codes, [encoding(utf8)]),
    string_literals(Codes, Lits),
    foldl(add_literal, Lits, Acc0, Acc).

add_literal(Escaped, Acc0, Acc) :-
    (   sub_string(Escaped, _, _, _, "{"),
        \+ memberchk(Escaped-_, Acc0),
        atomic_list_concat(['"', Escaped, '"'], Quoted),
        catch(term_string(Source, Quoted), _, fail),
        string(Source)
    ->  Acc = [Escaped-Source | Acc0]
    ;   Acc = Acc0
    ).

% the raw text of every "..." literal (escapes kept; a literal may not span lines)
string_literals([], []).
string_literals([0'" | Cs], [Lit | Lits]) :-
    literal_body(Cs, Body, Rest),
    !,
    string_codes(Lit, Body),
    string_literals(Rest, Lits).
string_literals([0'0, 0'' , _ | Cs], Lits) :-     % 0'c character code
    !,
    string_literals(Cs, Lits).
string_literals([_ | Cs], Lits) :-
    string_literals(Cs, Lits).

literal_body([0'" | Rest], [], Rest) :- !.
literal_body([0'\\, C | Cs], [0'\\, C | Body], Rest) :- !, C \== 0'\n, literal_body(Cs, Body, Rest).
literal_body([C | Cs], [C | Body], Rest) :- C \== 0'\n, literal_body(Cs, Body, Rest).

%% dump(+Entries, +Out)
dump(Entries, Out) :-
    setup_call_cleanup(open(Out, write, S, [encoding(utf8)]),
        forall(member(Escaped-Source, Entries), dump_one(S, Escaped, Source)),
        close(S)),
    length(Entries, N),
    format(user_error, "ir_corpus: ~w programs -> ~w~n", [N, Out]).

dump_one(S, Escaped, Source) :-
    (   catch(( plawk_parse_string(Source, Program),
                plawk_program_native_driver_ir(Program, 'input.txt', IR) ), _, fail)
    ->  sha_hash(IR, Hash, [algorithm(sha1), encoding(utf8)]),
        hash_atom(Hash, Hex),
        format(S, "=== ~w~nIR ~w~n~w~n", [Escaped, Hex, IR])
    ;   format(S, "=== ~w~nDECLINE~n", [Escaped])
    ).

%% diff(+A, +B, -Changed)
diff(A, B, Changed) :-
    records(A, RA),
    records(B, RB),
    findall(K, ( member(K-_, RA) ; member(K-_, RB) ), Ks0),
    list_to_set(Ks0, Ks),
    findall(K-SA-SB,
        ( member(K, Ks),
          ( memberchk(K-VA, RA) -> true ; VA = missing ),
          ( memberchk(K-VB, RB) -> true ; VB = missing ),
          VA \== VB,
          status(VA, SA), status(VB, SB) ),
        Diffs),
    forall(member(K-SA-SB, Diffs), format("CHANGED ~w -> ~w: ~w~n", [SA, SB, K])),
    length(RA, NA), length(RB, NB), length(Diffs, Changed),
    format("ir_corpus diff: ~w vs ~w programs, ~w changed~n", [NA, NB, Changed]).

status(missing, missing).
status(["DECLINE"], decline) :- !.
status(_, compiled).

% Key-Lines records of a dump: the program line, and the lines that follow it
records(File, Records) :-
    read_file_to_string(File, Text, [encoding(utf8)]),
    split_string(Text, "\n", "", Lines),
    records_(Lines, Records).

records_([], []).
records_([L | Ls], [Key-Body | Rs]) :-
    string_concat("=== ", Key0, L),
    !,
    atom_string(Key, Key0),
    take_body(Ls, Body0, Rest),
    exclude(==(""), Body0, Body),
    records_(Rest, Rs).
records_([_ | Ls], Rs) :-
    records_(Ls, Rs).

take_body([], [], []).
take_body([L | Ls], [], [L | Ls]) :- string_concat("=== ", _, L), !.
take_body([L | Ls], [L | Body], Rest) :- take_body(Ls, Body, Rest).

%% write_list(+Entries, +Out)
write_list(Entries, Out) :-
    setup_call_cleanup(open(Out, write, S, [encoding(utf8)]),
        forall(member(Escaped-_, Entries), format(S, "~w~n", [Escaped])),
        close(S)).

%% golden(+List, +OutDir)
%  Full module per listed program: `plawk build --keep-ll` keeps OUT.ll next to
%  the binary; the binary is discarded.
golden(List, OutDir) :-
    make_directory_path(OutDir),
    read_file_to_string(List, Text, [encoding(utf8)]),
    split_string(Text, "\n", "", Lines0),
    exclude(==(""), Lines0, Lines),
    foldl(golden_one(OutDir), Lines, 1, _),
    length(Lines, N),
    format(user_error, "ir_corpus golden: ~w programs -> ~w~n", [N, OutDir]).

golden_one(OutDir, Escaped, I, I1) :-
    I1 is I + 1,
    format(atom(Stem), '~|~`0t~d~4+', [I]),
    directory_file_path(OutDir, Stem, Base),
    atom_concat(Base, '.plawk', Prog),
    atomic_list_concat(['"', Escaped, '"'], Quoted),
    (   catch(term_string(Source, Quoted), _, fail), string(Source)
    ->  true
    ;   Source = Escaped
    ),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]), write(S, Source), close(S)),
    atom_concat(Base, '_bin', Bin),
    process_create(path(swipl), ['examples/plawk/bin/plawk', build, Prog, '-o', Bin, '--keep-ll'],
        [stdout(null), stderr(null), process(Pid)]),
    process_wait(Pid, exit(Status)),
    atom_concat(Bin, '.ll', KeptLL),
    atom_concat(Base, '.ll', LL),
    (   exists_file(KeptLL)
    ->  rename_file(KeptLL, LL)
    ;   setup_call_cleanup(open(LL, write, S2), format(S2, "; build exit ~w~n", [Status]), close(S2))
    ),
    ( exists_file(Bin) -> delete_file(Bin) ; true ).
