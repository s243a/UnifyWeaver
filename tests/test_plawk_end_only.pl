:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% END-only programs: a program whose only action block is `END { ... }`, with no
% main rule. `END { print "done" }` and the like used to decline (exit 3) for every
% form; new users hit this on the simplest possible awk one-liners.
%
% ---------------------------------------------------------------------------
% AN END-ONLY PROGRAM STILL READS ALL INPUT
%
% The subtle part, and the reason this is not "skip the loop": an END block sees
% NR (total record count), $0/$N (the LAST record), NF and length -- so the record
% loop must still run, consuming all of stdin, counting NR and retaining the last
% record, with only the per-record RULE CHAIN empty. On empty input NR is 0, $0 is
% empty, NF is 0. gawk 5.x is the oracle for all of this.
%
% This is why an END-only program is the existing rules+END driver with an empty
% rule chain, not the BEGIN-only driver (which reads NO input). The fix was three
% coordinated pieces, each gated on the rules being empty so no program with rules
% is touched (20/20 golden-corpus programs byte-identical):
%
%   1. plawk_scalar_state_plan/3 admits an empty plan when Rules == [] (its guard
%      otherwise requires some action / print / getline, none of which a rule-less
%      program has).
%   2. a dedicated plawk_scalar_rule_chain_ir([], ...) clause emits
%      `br label %continue_loop` -- the terminator the empty `lowered_match:` block
%      needs. Without it the general clause emits an unterminated block: invalid
%      LLVM, an exit-4 clang failure. The general clause keeps its `RuleCount > 0`.
%   3. the END-loop driver clause is guarded `Rules0 \== []`, so `END { while ... }`
%      with no rule DECLINES cleanly (exit 3) instead of committing past its cut and
%      miscompiling -- END-loop-with-no-rule is a follow-on, not part of this change.
%
% The distinction that matters most below: a supported END-only form must COMPILE
% and match gawk; an unsupported one must DECLINE cleanly (exit 3), never exit 4.

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').
:- use_module('../examples/plawk/codegen/llvm/plawk_native_codegen').

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

% Three records; the LAST is "7 disk" (NR 3, $1 7, $2 disk, $0 "7 disk", NF 2,
% length 6). The last record differs from the first so a test cannot pass by
% reading the wrong record.
input("5 boot\n5 trace\n7 disk\n").

:- begin_tests(plawk_end_only).

% --- the simplest END-only programs, which used to decline --------------

test(constant_print, [condition(clang_available)]) :-
    run("END { print \"done\" }\n", "done\n"),
    !.

test(two_constants, [condition(clang_available)]) :-
    run("END { print \"a\", \"b\" }\n", "a b\n"),
    !.

test(arithmetic, [condition(clang_available)]) :-
    run("END { print 1 + 2 }\n", "3\n"),
    !,
    run("END { print 3 % 2 }\n", "1\n"),
    !,
    run("END { print -1 }\n", "-1\n"),
    !.

% --- NR: the record count, which needs the loop to have run --------------

test(nr_is_the_record_count, [condition(clang_available)]) :-
    run("END { print NR }\n", "3\n"),
    !.

test(nr_on_empty_input_is_zero, [condition(clang_available)]) :-
    run_with("", "END { print NR }\n", "0\n"),
    !.

test(nr_in_a_concat, [condition(clang_available)]) :-
    run("END { print \"n=\" NR }\n", "n=3\n"),
    !.

% --- the retained last record: $0 / $N / NF / length --------------------
%
% These are the reason an END-only program is not "skip the loop": the record loop
% must retain the last record for END to project from it. That capability already
% existed (END field reads); this confirms it composes with an empty rule chain.

test(whole_last_record, [condition(clang_available)]) :-
    run("END { print $0 }\n", "7 disk\n"),
    !.

test(field_and_nf_of_last_record, [condition(clang_available)]) :-
    run("END { print $1, NF }\n", "7 2\n"),
    !.

test(length_of_last_record, [condition(clang_available)]) :-
    run("END { print length }\n", "6\n"),
    !,
    run("END { print length($0) }\n", "6\n"),
    !.

test(length_on_empty_input_is_zero, [condition(clang_available)]) :-
    run_with("", "END { print length }\n", "0\n"),
    !.

test(nr_field_and_nf_together, [condition(clang_available)]) :-
    run("END { print NR, $1, NF }\n", "3 7 2\n"),
    !.

% bare `print` in END is `print $0` of the last record.
test(bare_print_is_whole_last_record, [condition(clang_available)]) :-
    run("END { print }\n", "7 disk\n"),
    !.

% --- printf, and the multi-statement END list (clause 851) --------------

test(printf_in_end_only, [condition(clang_available)]) :-
    run("END { printf \"%d\\n\", NR }\n", "3\n"),
    !,
    run("END { printf \"%s\\n\", $1 }\n", "7\n"),
    !.

test(multi_statement_end, [condition(clang_available)]) :-
    run("END { print \"a\"; print \"b\" }\n", "a\nb\n"),
    !,
    run("END { print NR; print $1 }\n", "3\n7\n"),
    !.

% --- END-if with no rule (came along on the same state-plan path) -------

test(end_if_no_rule, [condition(clang_available)]) :-
    run("END { if (NR == 3) print \"x\" }\n", "x\n"),
    !.

test(end_if_else_reads_the_record, [condition(clang_available)]) :-
    run("END { if (NR > 1) print $1; else print \"few\" }\n", "7\n"),
    !.

test(end_if_on_empty_input, [condition(clang_available)]) :-
    run_with("", "END { if (NR == 0) print \"empty\" }\n", "empty\n"),
    !.

% --- the empty program: reads all input, prints nothing -----------------
%
% A scope expansion that fell out of admitting an empty rule chain, and it is
% correct gawk behaviour (`awk ''` drains input, prints nothing, exits 0). Pinned
% as intended rather than left unremarked.

test(empty_program_reads_input_prints_nothing, [condition(clang_available)]) :-
    run("\n", ""),
    !,
    run_with("", "\n", ""),
    !.

% --- REGRESSIONS: programs WITH rules are untouched ---------------------

test(rules_with_end_unchanged, [condition(clang_available)]) :-
    run("{ n++ } END { print n }\n", "3\n"),
    !,
    run("{ n++; c[$1]++ } END { print NR, c[\"5\"] }\n", "3 2\n"),
    !.

test(begin_only_unchanged, [condition(clang_available)]) :-
    run("BEGIN { print \"hi\" }\n", "hi\n"),
    !.

test(rule_only_unchanged, [condition(clang_available)]) :-
    run("{ print $1 }\n", "5\n5\n7\n"),
    !.

% END-loop WITH a rule still compiles -- the guard is on empty rules only.
test(end_loop_with_a_rule_still_compiles, [condition(clang_available)]) :-
    build_status("{ n++ } END { while (i < n) { print i; i++ } }\n", 0),
    !.

% --- BOUNDARIES: unsupported END-only forms DECLINE, never miscompile ---

% The miscompile that this change had to avoid: END-loop with no rule. It must
% decline cleanly (exit 3), NOT emit invalid LLVM (exit 4). This is the sharpest
% pin in the suite -- the whole END-loop clause guard exists for it.
test(end_only_loop_declines_cleanly_not_exit_4) :-
    build_status("END { while (i < 3) { print i; i++ } }\n", 3),
    !.

% END-only with a scalar ASSIGNMENT is a follow-on (the assignment machinery is
% not wired for an END-only context yet). Declines, exit 3.
test(end_only_assignment_declines) :-
    build_status("END { x = 5; print x }\n", 3),
    !.

% for-in over an (empty) array in END-only: declines, exit 3.
test(end_only_forin_declines) :-
    build_status("END { for (k in a) print k }\n", 3),
    !.

% builtins over the retained record in END are a separate follow-on (the LITERAL
% forms fold; these read the record). Decline, exit 3.
test(end_only_record_builtins_decline) :-
    build_status("END { print substr($0, 1, 3) }\n", 3),
    !,
    build_status("END { print toupper($1) }\n", 3),
    !.

% getline in an END block is unsupported: declines, exit 3.
test(end_only_getline_declines) :-
    build_status("END { getline x; print x }\n", 3),
    !.

% A scalar VARIABLE read/write in an END-only program declines cleanly (exit 3), NOT
% exit 4. This is the case an ultra review caught that the first cut missed: admitting
% an empty rule chain let the state plan collect the scalar as a slot, and the loop's
% next-slot phi then referenced %rule_-1_* (LastRuleIndex = RuleCount-1 = -1) --
% undefined SSA, a clang miscompile. The dedicated empty rule-chain clause is now
% guarded to fire only when the state plan has NO slots, so a scalar-var END-only
% program declines. It is a genuine follow-on: an unset scalar's value in END is
% context-dependent (empty in string context, 0 in numeric -- the uninitialised-scalar
% representation problem), so even a correct pass-through phi needs that settled first.
% Pinned so the miscompile cannot silently return and the follow-on stays visible.
test(end_only_scalar_var_read_declines_not_miscompiles) :-
    build_status("END { print x }\n", 3),
    build_status("END { print x; print \"done\" }\n", 3),
    build_status("END { printf \"%d\\n\", x }\n", 3),
    build_status("END { if (x == 0) print \"zero\" }\n", 3),
    build_status("END { if (NR > 0) print x }\n", 3),
    !.

% --- clang never rejects a supported END-only program -------------------
%
% Belt-and-braces over the whole supported surface: build each and require a
% binary. An exit-4 here would mean invalid LLVM slipped through; that is the
% failure the dedicated rule-chain clause exists to prevent.
test(no_supported_end_only_form_miscompiles, [condition(clang_available)]) :-
    forall(member(Src, [
        "END { print \"done\" }\n",
        "END { print NR }\n",
        "END { print $0 }\n",
        "END { print $1, NF }\n",
        "END { print length }\n",
        "END { printf \"%d\\n\", NR }\n",
        "END { print \"a\"; print \"b\" }\n",
        "END { if (NR == 3) print $1; else print NF }\n"
    ]), build_status(Src, 0)),
    !.

% --- clause enumeration: NO rule-less shape miscompiles (exit 4) ---------
%
% The sharpest risk in this change: relaxing the shared state-plan guard could let a
% rule-less program COMMIT past a cut in some driver clause other than the ones tested
% above and emit invalid LLVM. This pins the enumeration -- one rule-less program routed
% at each END-driver clause type (scalar print/list/if, loop, MIXED and ASSOC ends whose
% RuleCount>0 gates were deliberately NOT relaxed, BEGIN+end, binfmt, getline/redirect/
% assignment). Every one must build to a real exit code (0 compile / 2 parse / 3 decline)
% and NEVER 4. The mixed/assoc cases are the load-bearing ones: they confirm the two
% unrelaxed gates keep those clauses declining instead of committing on empty rules.
test(no_rule_less_shape_miscompiles) :-
    forall(member(Src, [
        "END { print \"x\" }\n",
        "END { print \"a\"; print \"b\" }\n",
        "END { if (NR == 3) print \"x\" }\n",
        "END { while (i < 3) { print i; i++ } }\n",
        "END { print c[\"x\"] }\n",
        "END { print c[\"x\"], NR }\n",
        "END { for (k in c) print k }\n",
        "END { c[\"x\"]++; print c[\"x\"] }\n",
        "BEGIN { FS=\":\" } END { print NR }\n",
        "BEGIN { print \"start\" } END { print NR }\n",
        "END { getline x }\n",
        "END { x = 5 }\n",
        "END { print x }\n",
        "END { print x; print \"done\" }\n",
        "END { printf \"%d\\n\", x }\n",
        "END { if (x == 0) print \"zero\" }\n",
        "END { if (NR > 0) print x }\n",
        "BEGIN { FS=\":\" } END { print x }\n",
        "BEGIN { BINFMT = \"i64\" } END { print x }\n"
    ]), build_status_not_4(Src)),
    !.

% --- the empty rule chain lowers to a single loop-continue branch -------
%
% The IR property that keeps the fix additive: with no rules, the record loop's
% match block is just a branch to continue_loop. Pinned on the emitted IR so a
% future change that reintroduces per-rule scaffolding here shows up.
test(empty_rule_chain_is_a_continue_branch) :-
    build_ll("END { print \"done\" }\n", LL),
    assertion(sub_string(LL, _, _, _, "br label %continue_loop")),
    % ...and no rule-match scaffolding, since there are no rules.
    assertion(\+ sub_string(LL, _, _, _, "rule_0_match")),
    !.

:- end_tests(plawk_end_only).

% --- helpers ---------------------------------------------------------------

odir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_end_only', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

run(Src, Expected) :-
    input(Input),
    run_with(Input, Src, Expected).

run_with(Input, Src, Expected) :-
    odir(Dir),
    directory_file_path(Dir, 'eo_bin', Bin),
    ( exists_file(Bin) -> delete_file(Bin) ; true ),
    directory_file_path(Dir, 'eo', Prog0),
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
    directory_file_path(Dir, 'eo_reject', Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    atom_concat(Prog0, '_bin', Bin),
    ( exists_file(Bin) -> delete_file(Bin) ; true ),
    cli([build, Prog, '-o', Bin], ExpectedStatus).

% Build and assert the exit code is anything but 4 (a clang miscompile). Used by the
% clause-enumeration test: a rule-less shape may compile, parse-fail, or decline, but
% must never produce invalid LLVM.
build_status_not_4(Src) :-
    odir(Dir),
    directory_file_path(Dir, 'eo_no4', Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    atom_concat(Prog0, '_bin', Bin),
    ( exists_file(Bin) -> delete_file(Bin) ; true ),
    process_create(path(swipl), ['examples/plawk/bin/plawk', build, Prog, '-o', Bin],
        [stdout(pipe(Out)), stderr(std), process(Pid)]),
    read_string(Out, _, _), close(Out),
    process_wait(Pid, exit(Status)),
    ( Status =\= 4
    -> true
    ;  format(user_error, "~nMISCOMPILE (exit 4): ~w~n", [Src]), fail
    ).

build_ll(Src, LL) :-
    plawk_parse_string(Src, Program),
    plawk_program_native_driver_ir(Program, 'input.txt', IR),
    atom_string(IR, LL).

cli(Args, ExpectedStatus) :-
    process_create(path(swipl), ['examples/plawk/bin/plawk' | Args],
        [stdout(pipe(S)), stderr(std), process(Pid)]),
    read_string(S, _, _), close(S),
    process_wait(Pid, exit(Status)),
    assertion(Status == ExpectedStatus).
