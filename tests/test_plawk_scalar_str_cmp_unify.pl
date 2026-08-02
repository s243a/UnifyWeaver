:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% plawk: ONE emitter for the string-scalar comparison.
%
% "Compare a string-holding scalar slot against a string literal" was written
% TWICE -- same semantics, same six operators:
%
%   the scalar-`if` guard              if (s == "text") / if (s < "text")
%   the bare string-scalar PATTERN     name == "text" { … }
%
% They had already drifted, in the way duplicated emitters do: the pattern copy
% required a text-holding slot and the `if` copy did not. That was a WRONG-OUTPUT
% bug -- `{ n++; if (n == "3") print "eq" }` answered false where awk (number vs
% string => string comparison) says true -- fixed in #4078 by adding the missing
% guard to the copy that lacked it. This removes the duplication that allowed it.
%
% ---------------------------------------------------------------------------
% WHY THERE IS A `Flavour` PARAMETER
%
% The two sites named their temporaries differently -- `lit`/`slit`,
% `empty`/`sempty`, `scmp`/`sscmp` -- and one COMPUTED its condition variable
% (`%Base_cond`) while the other was HANDED one (MatchValue).
%
% Unifying on either naming would rewrite the emitted IR of every program using
% the other surface. That spends the byte-identity this campaign relies on to
% prove unrelated output has not drifted -- a bad trade for a cosmetic gain, and
% an especially bad one on the emitter behind the #4078 bug.
%
% So the shared emitter takes a flavour ('' or s) plus the caller's condition
% variable, and reproduces BOTH historical spellings exactly. The duplication is
% gone; the IR does not move. Verified by golden dump over 12 programs covering
% both surfaces -- all 12 .ll files identical before and after.
%
% The tests below pin those historical names. Without them the flavour parameter
% looks like dead complexity and the next refactor "simplifies" it away, silently
% churning the IR of one surface. The names are the contract.

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').
:- use_module('../examples/plawk/codegen/plawk_native_codegen').

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

% Three records: "a 1" / "b 2" / "c 3".
input("a 1\nb 2\nc 3\n").

:- begin_tests(plawk_scalar_str_cmp_unify).

% --- behaviour: both surfaces, all six operators --------------------------

test(if_guard_equality, [condition(clang_available)]) :-
    run("{ s = $1; if (s == \"b\") print \"hit\" }\n", "hit\n"),
    !.

test(if_guard_inequality, [condition(clang_available)]) :-
    run("{ s = $1; if (s != \"b\") print $1 }\n", "a\nc\n"),
    !.

test(if_guard_ordering, [condition(clang_available)]) :-
    run("{ s = $1; if (s < \"c\") print $1 }\n", "a\nb\n"),
    !.

test(if_guard_ordering_ge, [condition(clang_available)]) :-
    run("{ s = $1; if (s >= \"b\") print $1 }\n", "b\nc\n"),
    !.

test(pattern_equality, [condition(clang_available)]) :-
    run("{ s = $1 } s == \"b\" { print \"p\" }\n", "p\n"),
    !.

test(pattern_ordering, [condition(clang_available)]) :-
    run("{ s = $1 } s < \"c\" { print \"p\" }\n", "p\np\n"),
    !.

test(pattern_reversed, [condition(clang_available)]) :-
    run("{ s = $1 } \"b\" == s { print \"p\" }\n", "p\n"),
    !.

% The END guard, which reaches the `if` emitter too (#4078).
test(end_guard_equality, [condition(clang_available)]) :-
    run("{ s = $1 } END { if (s == \"c\") print s }\n", "c\n"),
    !.

% A literal-assigned (scalar_string) slot as well as a field-assigned
% (scalar_strnum) one -- both hold an interned id, and one emitter serves both.
test(literal_assigned_slot, [condition(clang_available)]) :-
    run("{ s = \"x\"; if (s == \"x\") print \"yes\" }\n", "yes\nyes\nyes\n"),
    !.

% --- the guard that used to be missing from one copy ----------------------

% A numeric counter is not a text slot. Both surfaces decline rather than
% comparing a count against an interned id. Asserted as a PAIR: with one emitter
% they cannot disagree, and this test is what would catch a re-split.
test(counter_slot_declines_on_both_surfaces) :-
    build_status("{ n++; if (n == \"3\") print \"eq\" }\n", 3),
    build_status("{ n++ } n == \"3\" { print \"p\" }\n", 3),
    !.

% --- structure: one emitter, two flavours ---------------------------------

% The `if` flavour ('') keeps `lit` / `empty` / `scmp` and writes the caller's
% condition variable.
test(plain_flavour_keeps_its_historical_names) :-
    plawk_native_codegen:plawk_scalar_str_cmp_ir('', '%slot_0', eq, "b",
        base, '%base_cond', _G1, EqIR),
    assertion(sub_atom(EqIR, _, _, _, '%base_litptr')),
    assertion(sub_atom(EqIR, _, _, _, '%base_litid')),
    assertion(sub_atom(EqIR, _, _, _, '%base_cond = icmp eq i64 %slot_0')),
    plawk_native_codegen:plawk_scalar_str_cmp_ir('', '%slot_0', lt, "b",
        base, '%base_cond', _G2, LtIR),
    assertion(sub_atom(LtIR, _, _, _, '%base_empty')),
    assertion(sub_atom(LtIR, _, _, _, '%base_scmp')),
    !.

% The pattern flavour (s) keeps `slit` / `sempty` / `sscmp`.
test(s_flavour_keeps_its_historical_names) :-
    plawk_native_codegen:plawk_scalar_str_cmp_ir(s, '%rule_0_in_slot_0', eq,
        "b", base, '%base_match', _G1, EqIR),
    assertion(sub_atom(EqIR, _, _, _, '%base_slitptr')),
    assertion(sub_atom(EqIR, _, _, _, '%base_slitid')),
    assertion(sub_atom(EqIR, _, _, _, '%base_match = icmp eq i64')),
    plawk_native_codegen:plawk_scalar_str_cmp_ir(s, '%rule_0_in_slot_0', lt,
        "b", base, '%base_match', _G2, LtIR),
    assertion(sub_atom(LtIR, _, _, _, '%base_sempty')),
    assertion(sub_atom(LtIR, _, _, _, '%base_sscmp')),
    !.

% The two flavours differ ONLY in those names -- same instruction sequence, same
% operators. If this ever fails, the flavours have grown a semantic difference
% and the unification has quietly come undone.
test(flavours_differ_only_in_naming) :-
    forall(member(Op, [eq, ne, lt, le, gt, ge]),
        ( plawk_native_codegen:plawk_scalar_str_cmp_ir('', '%v', Op, "b",
              base, '%c', _GA, PlainIR),
          plawk_native_codegen:plawk_scalar_str_cmp_ir(s, '%v', Op, "b",
              base, '%c', _GB, SIR),
          % strip the flavour letter from the s-form's distinguishing names
          atomic_list_concat(P1, '_slit', SIR),  atomic_list_concat(P1, '_lit', S1),
          atomic_list_concat(P2, '_sempty', S1), atomic_list_concat(P2, '_empty', S2),
          atomic_list_concat(P3, '_sscmp', S2),  atomic_list_concat(P3, '_scmp', Normalised),
          assertion(Normalised == PlainIR)
        )),
    !.

% Both operator families are served by the one predicate.
test(one_predicate_covers_all_six_operators) :-
    forall(member(Op, [eq, ne, lt, le, gt, ge]),
        assertion(plawk_native_codegen:plawk_scalar_str_cmp_ir('', '%v', Op,
            "b", base, '%c', _G, _IR))),
    !.

:- end_tests(plawk_scalar_str_cmp_unify).

% --- helpers ---------------------------------------------------------------

odir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_strcmp_unify', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

run(Src, Expected) :-
    odir(Dir),
    input(Input),
    directory_file_path(Dir, 'su_bin', Bin),
    % A build that unexpectedly declines must not silently run a stale binary.
    ( exists_file(Bin) -> delete_file(Bin) ; true ),
    directory_file_path(Dir, 'su', Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    atom_concat(Prog0, '_in.txt', In),
    setup_call_cleanup(open(In, write, SI, [encoding(utf8)]),
        write(SI, Input), close(SI)),
    cli([build, Prog, '-o', Bin], 0),
    process_create(Bin, [In], [stdout(pipe(PS)), stderr(std), process(Pid)]),
    read_string(PS, _, Out),
    close(PS),
    process_wait(Pid, exit(0)),
    ( Out == Expected
    -> true
    ;  format(user_error, "~n~w~n  got      ~q~n  expected ~q~n",
           [Src, Out, Expected]), fail
    ).

build_status(Src, ExpectedStatus) :-
    odir(Dir),
    directory_file_path(Dir, 'su_reject', Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    atom_concat(Prog0, '_bin', Bin),
    cli([build, Prog, '-o', Bin], ExpectedStatus).

cli(Args, ExpectedStatus) :-
    process_create(path(swipl), ['examples/plawk/bin/plawk' | Args],
        [stdout(pipe(S)), stderr(std), process(Pid)]),
    read_string(S, _, _), close(S),
    process_wait(Pid, exit(Status)),
    assertion(Status == ExpectedStatus).
