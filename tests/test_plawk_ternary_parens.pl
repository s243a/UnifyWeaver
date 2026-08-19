:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% plawk: a parenthesised WHOLE ternary -- `(c ? a : b)`.
%
% #4035 added a parenthesised CONDITION, `(c) ? a : b`. Wrapping the WHOLE
% ternary is a different production and had no rule at all:
%
%   x = ($2 > 1 ? "hi" : "lo")            PARSE ERROR
%   print $1, ($2 > 1 ? "hi" : "lo")      PARSE ERROR
%
% The gap bites hardest in a print list, because of an awk subtlety worth
% stating: a bare `>` there is output REDIRECTION, not comparison. So
%
%   print $1, $2 > 1 ? "hi" : "lo"        gawk: SYNTAX ERROR
%   print $1, ($2 > 1 ? "hi" : "lo")      gawk: fine
%
% the parenthesised form is the ONLY legal spelling of a `>`-conditioned ternary
% in a print list -- and it was exactly the one plawk could not read, while plawk
% happily accepted the form gawk rejects. (That permissiveness is a separate,
% pre-existing divergence; tightening it would turn working programs into syntax
% errors, so it is left alone and recorded, not changed here.)
%
% Implemented as ONE recursive clause delegating to ternary_expr//1 itself, not a
% third copy of the two existing clauses -- so a parenthesised ternary admits
% exactly what an unparenthesised one does, including nesting, and cannot drift
% from it. Placed LAST, so both existing clauses are tried first and no parse that
% worked before can change.
%
% gawk 5.2 is the oracle for every expectation here.

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').
:- use_module('../examples/plawk/codegen/llvm/plawk_native_codegen').

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

% Three records: "a 1" / "b 2" / "c 3".
input("a 1\nb 2\nc 3\n").

:- begin_tests(plawk_ternary_parens).

% --- the parenthesised whole ternary, every context -----------------------

test(paren_whole_ternary_in_assignment, [condition(clang_available)]) :-
    run("{ x = ($2 > 1 ? 10 : 20); print x }\n", "20\n10\n10\n"),
    !.

test(paren_whole_ternary_string_branches, [condition(clang_available)]) :-
    run("{ x = ($2 > 1 ? \"hi\" : \"lo\"); print x }\n", "lo\nhi\nhi\n"),
    !.

test(paren_whole_ternary_in_print, [condition(clang_available)]) :-
    run("{ print ($2 > 1 ? 10 : 20) }\n", "20\n10\n10\n"),
    !.

% THE case this unblocks: a `>`-conditioned ternary in a print LIST, which awk
% accepts only in this spelling.
test(paren_whole_ternary_in_print_list, [condition(clang_available)]) :-
    run("{ print $1, ($2 > 1 ? \"hi\" : \"lo\") }\n", "a lo\nb hi\nc hi\n"),
    !.

test(paren_whole_ternary_after_a_literal, [condition(clang_available)]) :-
    run("{ print \"t:\", ($1 == \"b\" ? \"y\" : \"n\") }\n",
        "t: n\nt: y\nt: n\n"),
    !.

test(paren_whole_ternary_in_printf, [condition(clang_available)]) :-
    run("{ printf \"%s\\n\", ($2 > 1 ? \"hi\" : \"lo\") }\n", "lo\nhi\nhi\n"),
    !.

% A parenthesised CONDITION inside a parenthesised ternary: the recursive clause
% delegates, so the two forms compose.
test(paren_condition_inside_paren_ternary, [condition(clang_available)]) :-
    run("{ x = (($2 > 1) ? 10 : 20); print x }\n", "20\n10\n10\n"),
    !.

% --- regressions: the two forms that already parsed -----------------------

test(paren_condition_form_unchanged, [condition(clang_available)]) :-
    run("{ x = ($2 > 1) ? 10 : 20; print x }\n", "20\n10\n10\n"),
    !.

test(bare_form_unchanged, [condition(clang_available)]) :-
    run("{ x = $2 > 1 ? 10 : 20; print x }\n", "20\n10\n10\n"),
    !.

test(paren_condition_in_print_unchanged, [condition(clang_available)]) :-
    run("{ print ($1 == \"b\") ? \"yes\" : \"no\" }\n", "no\nyes\nno\n"),
    !.

% --- structure: all three spellings are ONE term --------------------------

% The parenthesised whole form parses to exactly the term the bare form does, so
% nothing downstream can distinguish them -- which is why no codegen path
% changed for this.
test(all_three_spellings_agree) :-
    plawk_parse_string("{ x = ($2 > 1 ? 10 : 20) }\n",
        program([], [rule(always, [set(var(x), ParenWhole)])], [])),
    plawk_parse_string("{ x = ($2 > 1) ? 10 : 20 }\n",
        program([], [rule(always, [set(var(x), ParenCond)])], [])),
    plawk_parse_string("{ x = $2 > 1 ? 10 : 20 }\n",
        program([], [rule(always, [set(var(x), Bare)])], [])),
    assertion(ParenWhole == Bare),
    assertion(ParenCond == Bare),
    assertion(Bare = ternary(cmp(field(2), gt, int(1)), int(10), int(20))),
    !.

% Nesting: the clause is recursive, so redundant parens collapse to the same
% term rather than being a special case.
test(nested_parens_collapse) :-
    plawk_parse_string("{ x = (($2 > 1 ? 10 : 20)) }\n",
        program([], [rule(always, [set(var(x), Nested)])], [])),
    assertion(Nested = ternary(cmp(field(2), gt, int(1)), int(10), int(20))),
    !.

% The recursive clause admits what the unparenthesised form admits -- including
% string branches, which are a separate feature it must not have to know about.
test(paren_form_inherits_string_branches) :-
    plawk_parse_string("{ x = ($1 == \"b\" ? \"y\" : \"n\") }\n",
        program([], [rule(always, [set(var(x), Ternary)])], [])),
    assertion(Ternary = ternary(cmp(field(1), eq, string("b")),
        string("y"), string("n"))),
    !.

:- end_tests(plawk_ternary_parens).

% --- helpers ---------------------------------------------------------------

odir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_ternary_parens', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

run(Src, Expected) :-
    odir(Dir),
    input(Input),
    directory_file_path(Dir, 'tp_bin', Bin),
    % A build that unexpectedly declines must not silently run a stale binary.
    ( exists_file(Bin) -> delete_file(Bin) ; true ),
    directory_file_path(Dir, 'tp', Prog0),
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

cli(Args, ExpectedStatus) :-
    process_create(path(swipl), ['examples/plawk/bin/plawk' | Args],
        [stdout(pipe(S)), stderr(std), process(Pid)]),
    read_string(S, _, _), close(S),
    process_wait(Pid, exit(Status)),
    assertion(Status == ExpectedStatus).
