:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% plawk regex substitution and search:
%   sub(/re/, "repl")  / gsub(/re/, "repl")  -- substitute the first / every ERE
%     match in $0, an unescaped & in the replacement expanding to the matched
%     text; backed by @wam_regex_gsub. v1 target is $0 (the stream-editor idiom
%     `Pattern { sub/gsub(...); print $0 }`).
%   match(SRC, /re/)   -- return the 1-based match position (0 if none) and set
%     RSTART / RLENGTH (RLENGTH = -1 on no match); an i64 expression usable in
%     print. Backed by @wam_regex_match.
% sub/gsub into a scalar or field, and capturing the substitution count, are
% documented follow-ons.

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').

:- begin_tests(plawk_subgsub_match).

% --- parsing ----------------------------------------------------------------

test(gsub_parses) :-
    plawk_parse_string("{ gsub(/[0-9]+/, \"#\"); print $0 }\n",
        program([], [rule(always, [regex_sub(1, "[0-9]+", "#"), print([field(0)])])], [])),
    !.

test(sub_parses) :-
    plawk_parse_string("{ sub(/x/, \"y\"); print $0 }\n",
        program([], [rule(always, [regex_sub(0, "x", "y"), print([field(0)])])], [])),
    !.

test(match_parses) :-
    plawk_parse_string("{ print match($0, /[0-9]+/), RSTART, RLENGTH }\n",
        program([], [rule(always,
            [print([match_expr(field(0), "[0-9]+"), special('RSTART'), special('RLENGTH')])])], [])),
    !.

% --- gsub / sub runtime -----------------------------------------------------

% gsub replaces every match; non-matching records pass through unchanged.
test(gsub_all, [condition(clang_available)]) :-
    ldir(Dir),
    build_run(Dir, 'ga', "{ gsub(/[0-9]+/, \"#\"); print $0 }\n",
        "a1b22c333\nxyz\n", Out, St),
    assertion(St == 0), assertion(Out == "a#b#c#\nxyz\n"), !.

% sub replaces only the first match.
test(sub_first, [condition(clang_available)]) :-
    ldir(Dir),
    build_run(Dir, 'sf', "{ sub(/o/, \"0\"); print $0 }\n",
        "foo boo\n", Out, St),
    assertion(St == 0), assertion(Out == "f0o boo\n"), !.

% an unescaped & in the replacement expands to the matched text.
test(gsub_amp, [condition(clang_available)]) :-
    ldir(Dir),
    build_run(Dir, 'ga2', "{ gsub(/[0-9]+/, \"[&]\"); print $0 }\n",
        "a1b22\n", Out, St),
    assertion(St == 0), assertion(Out == "a[1]b[22]\n"), !.

% a pattern guard gates the substitution+print.
test(gsub_guarded, [condition(clang_available)]) :-
    ldir(Dir),
    build_run(Dir, 'gg', "$1 ~ /ERR/ { gsub(/[0-9]/, \"#\"); print $0 }\n",
        "ERR 42\nok 7\n", Out, St),
    assertion(St == 0), assertion(Out == "ERR ##\n"), !.

% chained substitutions compose on the running record.
test(gsub_chained, [condition(clang_available)]) :-
    ldir(Dir),
    build_run(Dir, 'gc', "{ gsub(/a/, \"X\"); gsub(/b/, \"Y\"); print $0 }\n",
        "abab\n", Out, St),
    assertion(St == 0), assertion(Out == "XYXY\n"), !.

% a string-literal pattern works like a regex literal.
test(gsub_string_pat, [condition(clang_available)]) :-
    ldir(Dir),
    build_run(Dir, 'gs', "{ gsub(\"[0-9]\", \"#\"); print $0 }\n",
        "a1b2\n", Out, St),
    assertion(St == 0), assertion(Out == "a#b#\n"), !.

% --- match / RSTART / RLENGTH -----------------------------------------------

% match($0, /re/) returns the position and sets RSTART/RLENGTH; a miss is 0/-1.
test(match_record, [condition(clang_available)]) :-
    ldir(Dir),
    build_run(Dir, 'mr', "{ print match($0, /[0-9]+/), RSTART, RLENGTH }\n",
        "abc123def\nxyz\n", Out, St),
    assertion(St == 0), assertion(Out == "4 4 3\n0 0 -1\n"), !.

% match on a positive field.
test(match_field, [condition(clang_available)]) :-
    ldir(Dir),
    build_run(Dir, 'mf', "BEGIN { FS = \",\" }\n{ print match($2, /o+/), RSTART, RLENGTH }\n",
        "a,foo,b\n", Out, St),
    assertion(St == 0), assertion(Out == "2 2 2\n"), !.

% match in a scalar assignment: n gets the position, RSTART/RLENGTH are set.
test(match_scalar_assign, [condition(clang_available)]) :-
    ldir(Dir),
    build_run(Dir, 'ms', "{ n = match($0, /[0-9]+/); print n, RSTART, RLENGTH }\n",
        "ab123\nxy\n", Out, St),
    assertion(St == 0), assertion(Out == "3 3 3\n0 0 -1\n"), !.

% RSTART / RLENGTH readable as scalar RHS.
test(rstart_scalar_rhs, [condition(clang_available)]) :-
    ldir(Dir),
    build_run(Dir, 'rr', "{ n = match($0, /[0-9]+/); x = RLENGTH; print x }\n",
        "ab777\n", Out, St),
    assertion(St == 0), assertion(Out == "3\n"), !.

% capture into a slot, then guard on the slot (the guard idiom for match).
test(match_capture_then_guard, [condition(clang_available)]) :-
    ldir(Dir),
    build_run(Dir, 'mg', "{ n = match($0, /[0-9]+/); if (n > 0) print RSTART, RLENGTH }\n",
        "ab123\nxy\n", Out, St),
    assertion(St == 0), assertion(Out == "3 3\n"), !.

% RSTART as a guard comparison LHS (reads the special, not a phantom slot).
test(rstart_guard, [condition(clang_available)]) :-
    ldir(Dir),
    build_run(Dir, 'rg', "{ n = match($0, /[0-9]+/); if (RSTART > 0) print RSTART, RLENGTH }\n",
        "ab12\nxy\n", Out, St),
    assertion(St == 0), assertion(Out == "3 2\n"), !.

% RLENGTH in a guard comparison.
test(rlength_guard, [condition(clang_available)]) :-
    ldir(Dir),
    build_run(Dir, 'lg', "{ n = match($0, /[0-9]+/); if (RLENGTH >= 3) print \"long\" }\n",
        "ab7\ncd999\n", Out, St),
    assertion(St == 0), assertion(Out == "long\n"), !.

:- end_tests(plawk_subgsub_match).

% --- helpers ---------------------------------------------------------------

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

ldir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_subgsub_match', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

build_run(Dir, Name, Src, Input, Out, RunStatus) :-
    directory_file_path(Dir, Name, Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    atom_concat(Prog0, '_bin', Bin),
    process_create(path(swipl), ['examples/plawk/bin/plawk', build, Prog, '-o', Bin],
        [stdout(null), stderr(null), process(BPid)]),
    process_wait(BPid, exit(0)),
    process_create(Bin, [],
        [stdin(pipe(In)), stdout(pipe(RS)), stderr(std), process(RPid)]),
    format(In, "~w", [Input]),
    close(In),
    read_string(RS, _, Out),
    close(RS),
    process_wait(RPid, exit(RunStatus)).
