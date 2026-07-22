% SPDX-License-Identifier: MIT OR Apache-2.0
%
% test_wam_r_iso_smoke.pl - ISO-R-0/2A generated-R smoke for is/2 and the
% six arithmetic-comparison three-form families.
% Hand-written PASS/FAIL; exit nonzero on failure (F# smoke convention).
%
% Usage: swipl -q -f init.pl -s tests/test_wam_r_iso_smoke.pl -g main -t halt

:- encoding(utf8).
:- use_module('../src/unifyweaver/targets/wam_r_target',
              [write_wam_r_project/3]).
:- use_module(library(filesex), [delete_directory_and_contents/1,
                                  make_directory_path/1,
                                  directory_file_path/3]).
:- use_module(library(process)).

pass(Name) :- format('[PASS] ~w~n', [Name]).
fail_test(Name, Reason) :-
    format('[FAIL] ~w: ~w~n', [Name, Reason]),
    nb_setval(r_iso_smoke_failed, true).

rscript_available :-
    catch(
        ( process_create(path('Rscript'), ['--version'],
                         [stdout(null), stderr(null), process(PID)]),
          process_wait(PID, exit(0))
        ), _, fail).

run_rscript(Drive, Dir, ExitCode, OutText) :-
    setup_call_cleanup(
        process_create(path('Rscript'), [Drive],
                       [cwd(Dir), stdout(pipe(Out)), stderr(pipe(Err)),
                        process(PID)]),
        ( read_string(Out, _, OutStr),
          read_string(Err, _, ErrStr),
          process_wait(PID, exit(ExitCode)),
          string_concat(OutStr, ErrStr, OutText)
        ),
        ( catch(close(Out), _, true), catch(close(Err), _, true) )).

write_driver(Path, Checks) :-
    open(Path, write, S),
    format(S,
'source("R/wam_runtime.R")
source("R/generated_program.R")
passes <- 0L; fails <- 0L
check <- function(name, key) {
  start_pc <- shared_labels[[key]]
  ok <- tryCatch(
    !is.null(start_pc) && isTRUE(WamRuntime$run_predicate(shared_program, start_pc, list())),
    error = function(e) { message(conditionMessage(e)); FALSE })
  if (isTRUE(ok)) { passes <<- passes + 1L; cat(sprintf("[PASS] %s\\n", name)) }
  else { fails <<- fails + 1L; cat(sprintf("[FAIL] %s\\n", name)) }
}
', []),
    forall(member(Name-Key, Checks),
           (   r_iso_smoke_r_string(Name, NameR),
               r_iso_smoke_r_string(Key, KeyR),
               format(S, 'check(~w, ~w)~n', [NameR, KeyR])
           )),
    write(S,
'cat(sprintf("RESULT %d/%d\\n", passes, passes + fails))
quit(status = if (fails == 0L) 0L else 1L)
'),
    close(S).

%% Escape a Prolog string/atom for an R double-quoted literal.
r_iso_smoke_r_string(In, Out) :-
    format(string(S0), '~w', [In]),
    atom_chars(S0, Cs),
    maplist(r_iso_smoke_escape_char, Cs, EscParts),
    atomic_list_concat(EscParts, Esc),
    format(atom(Out), '"~w"', [Esc]).

r_iso_smoke_escape_char('\\', '\\\\') :- !.
r_iso_smoke_escape_char('"', '\\"') :- !.
r_iso_smoke_escape_char(C, C).

unique_tmp(Base, Dir) :-
    get_time(T), format(atom(Dir), '/tmp/~w_~w', [Base, T]),
    catch(delete_directory_and_contents(Dir), _, true),
    make_directory_path(Dir).

run_project(Label, Preds, Opts, Checks) :-
    format('--- ~w ---~n', [Label]),
    unique_tmp('uw_r_iso_smoke', Tmp),
    write_wam_r_project(Preds, Opts, Tmp),
    directory_file_path(Tmp, 'drive.R', Drive),
    write_driver(Drive, Checks),
    (   rscript_available
    ->  run_rscript(Drive, Tmp, Exit, Out),
        write(Out),
        (   Exit == 0 -> pass(Label)
        ;   fail_test(Label, ['Rscript exit=', Exit])
        )
    ;   directory_file_path(Tmp, 'R/generated_program.R', Prog),
        read_file_to_string(Prog, Code, []),
        (   ( sub_string(Code,_,_,_,"is_iso/2")
            ; sub_string(Code,_,_,_,"is_lax/2")
            ; sub_string(Code,_,_,_,"<_iso/2")
            ; sub_string(Code,_,_,_,"<_lax/2")
            ; sub_string(Code,_,_,_,">_iso/2")
            ; sub_string(Code,_,_,_,"=:=_iso/2")
            ; sub_string(Code,_,_,_,"succ_iso")
            ; sub_string(Code,_,_,_,"succ_lax")
            )
        ->  pass(Label-codegen-skip-rscript)
        ;   fail_test(Label, 'missing rewritten iso/lax key in codegen')
        )
    ),
    catch(delete_directory_and_contents(Tmp), _, true).

assert_iso_helpers :-
    % is/2 family (ISO-R-0)
    assertz((user:r_ok :- X is 1 + 2, X == 3)),
    assertz((user:r_iso_inst :-
        catch(is_iso(_, _Y), error(instantiation_error, _), true))),
    assertz((user:r_iso_type :-
        catch(is_iso(_, foo), error(type_error(evaluable, _), _), true))),
    assertz((user:r_iso_zdiv :-
        catch(is_iso(_, 1 / 0), error(evaluation_error(zero_divisor), _), true))),
    assertz((user:r_lax_bad :- \+ is_lax(_, foo))),
    assertz((user:r_plain_zdiv :-
        catch((_X is 1 / 0), error(evaluation_error(zero_divisor), _), true))),
    assertz((user:r_plain_bad :- \+ (_ is foo))),
    assertz((user:r_plain_ok_iso :- X is 40 + 2, X == 42)),

    % Six comparison families: success + false (ISO-R-2A)
    assertz((user:r_cmp_ok :- 1 < 2, 3 > 2, 2 =< 2, 3 >= 3, 4 =:= 2+2, 1 =\= 2)),
    assertz((user:r_cmp_false :- \+ (2 < 1), \+ (1 > 2), \+ (3 =< 2),
                                 \+ (2 >= 3), \+ (1 =:= 2), \+ (2 =\= 2))),

    % ISO errors: unbound / type / zero_divisor on either operand
    assertz((user:r_lt_iso_inst_l :-
        catch('<_iso'(_, 3), error(instantiation_error, _), true))),
    assertz((user:r_lt_iso_inst_r :-
        catch('<_iso'(3, _), error(instantiation_error, _), true))),
    assertz((user:r_lt_iso_type_l :-
        catch('<_iso'(foo, 3), error(type_error(evaluable, _), _), true))),
    assertz((user:r_lt_iso_type_r :-
        catch('<_iso'(3, foo), error(type_error(evaluable, _), _), true))),
    assertz((user:r_lt_iso_zdiv_l :-
        catch('<_iso'(1/0, 3), error(evaluation_error(zero_divisor), _), true))),
    assertz((user:r_lt_iso_zdiv_r :-
        catch('<_iso'(3, 1/0), error(evaluation_error(zero_divisor), _), true))),
    % Fleet-wide precedence scans both operands before evaluation.
    assertz((user:r_lt_iso_precedence_inst :-
        catch('<_iso'(foo, _), error(instantiation_error, _), true))),
    assertz((user:r_lt_iso_precedence_zdiv :-
        catch('<_iso'(foo, 1/0), error(evaluation_error(zero_divisor), _), true))),
    assertz((user:r_gt_iso_inst :-
        catch('>_iso'(_, 3), error(instantiation_error, _), true))),
    assertz((user:r_ge_iso_type :-
        catch('>=_iso'(foo, 3), error(type_error(evaluable, _), _), true))),
    assertz((user:r_le_iso_zdiv :-
        catch('=<_iso'(1/0, 1), error(evaluation_error(zero_divisor), _), true))),
    assertz((user:r_eq_iso_type :-
        catch('=:=_iso'(foo, 3), error(type_error(evaluable, _), _), true))),
    assertz((user:r_ne_iso_inst :-
        catch('=\\=_iso'(_, 3), error(instantiation_error, _), true))),

    % Lax silent failure for same malformed inputs
    assertz((user:r_lt_lax_bad :-
        \+ '<_lax'(_, 3), \+ '<_lax'(foo, 3), \+ '<_lax'(1/0, 3))),
    assertz((user:r_eq_lax_bad :-
        \+ '=:=_lax'(foo, 3), \+ '=:=_lax'(_, 1))),

    % Explicit bypass of mode
    assertz((user:r_cmp_iso_under_lax :-
        catch('<_iso'(foo, 3), error(type_error(evaluable, _), _), true))),
    assertz((user:r_cmp_lax_under_iso :- \+ '<_lax'(foo, 3))),

    % Default rewrite paths
    assertz((user:r_cmp_plain_zdiv :-
        catch((1/0 < 3), error(evaluation_error(zero_divisor), _), true))),
    assertz((user:r_cmp_plain_bad :- \+ (foo < 3))),
    assertz((user:r_cmp_plain_ok :- 1 < 2)),

    % succ/2 family (ISO-R-2B)
    assertz((user:r_succ_fwd :- succ(0, 1), succ(4, 5))),
    assertz((user:r_succ_back :- succ(X, 5), X =:= 4)),
    assertz((user:r_succ_pair_ok :- succ(3, 4))),
    assertz((user:r_succ_pair_bad :- \+ succ(3, 5))),
    assertz((user:r_succ_lax_bad :-
        \+ succ_lax(_, _), \+ succ_lax(foo, _), \+ succ_lax(_, foo),
        \+ succ_lax(-1, _), \+ succ_lax(_, 0), \+ succ_lax(_, -1))),
    assertz((user:r_succ_iso_inst :-
        catch(succ_iso(_, _), error(instantiation_error, _), true))),
    assertz((user:r_succ_iso_type_x :-
        catch(succ_iso(foo, _), error(type_error(integer, _), _), true))),
    assertz((user:r_succ_iso_type_y :-
        catch(succ_iso(_, bar), error(type_error(integer, _), _), true))),
    % Classification precedence: X type before Y type; Y type before X range.
    assertz((user:r_succ_iso_precedence_x :-
        catch(succ_iso(foo, bar), error(type_error(integer, foo), _), true))),
    assertz((user:r_succ_iso_precedence_y :-
        catch(succ_iso(-1, bar), error(type_error(integer, bar), _), true))),
    assertz((user:r_succ_iso_neg :-
        catch(succ_iso(-1, _),
              error(type_error(not_less_than_zero, _), _), true))),
    assertz((user:r_succ_iso_domain :-
        catch(succ_iso(_, 0),
              error(domain_error(not_less_than_zero, _), _), true))),
    assertz((user:r_succ_iso_mismatch :- \+ succ_iso(3, 5))),
    assertz((user:r_succ_iso_under_lax :-
        catch(succ_iso(_, _), error(instantiation_error, _), true))),
    assertz((user:r_succ_lax_under_iso :- \+ succ_lax(_, _))),
    assertz((user:r_succ_plain_iso :-
        catch((succ(_, _)), error(instantiation_error, _), true))),
    assertz((user:r_succ_plain_lax :- \+ (succ(_, _)))),
    assertz((user:r_succ_mod_b :- succ(1, 2))).

retract_iso_helpers :-
    forall(member(P, [
        r_ok/0, r_iso_inst/0, r_iso_type/0, r_iso_zdiv/0,
        r_lax_bad/0, r_plain_zdiv/0, r_plain_bad/0, r_plain_ok_iso/0,
        r_cmp_ok/0, r_cmp_false/0,
        r_lt_iso_inst_l/0, r_lt_iso_inst_r/0,
        r_lt_iso_type_l/0, r_lt_iso_type_r/0,
        r_lt_iso_zdiv_l/0, r_lt_iso_zdiv_r/0,
        r_lt_iso_precedence_inst/0, r_lt_iso_precedence_zdiv/0,
        r_gt_iso_inst/0, r_ge_iso_type/0, r_le_iso_zdiv/0,
        r_eq_iso_type/0, r_ne_iso_inst/0,
        r_lt_lax_bad/0, r_eq_lax_bad/0,
        r_cmp_iso_under_lax/0, r_cmp_lax_under_iso/0,
        r_cmp_plain_zdiv/0, r_cmp_plain_bad/0, r_cmp_plain_ok/0,
        r_succ_fwd/0, r_succ_back/0, r_succ_pair_ok/0, r_succ_pair_bad/0,
        r_succ_lax_bad/0, r_succ_iso_inst/0, r_succ_iso_type_x/0,
        r_succ_iso_type_y/0, r_succ_iso_precedence_x/0,
        r_succ_iso_precedence_y/0, r_succ_iso_neg/0, r_succ_iso_domain/0,
        r_succ_iso_mismatch/0, r_succ_iso_under_lax/0, r_succ_lax_under_iso/0,
        r_succ_plain_iso/0, r_succ_plain_lax/0, r_succ_mod_b/0
    ]), retractall(user:P)).

main :-
    nb_setval(r_iso_smoke_failed, false),
    assert_iso_helpers,

    % ----- is/2 (ISO-R-0 regression) -----
    run_project('successful is/2 (iso default, interpreter)',
        [user:r_ok/0],
        [iso_errors(true)],
        ['successful is/2'-'r_ok/0']),

    run_project('ISO-default structured errors (interpreter)',
        [user:r_plain_zdiv/0, user:r_plain_ok_iso/0],
        [iso_errors(true)],
        ['ISO-default zero_divisor'-'r_plain_zdiv/0',
         'ISO-default success'-'r_plain_ok_iso/0']),

    run_project('lax-default silent fail (interpreter)',
        [user:r_plain_bad/0],
        [iso_errors(false)],
        ['lax-default silent'-'r_plain_bad/0']),

    run_project('per-predicate override to lax (interpreter)',
        [user:r_plain_bad/0, user:r_plain_zdiv/0],
        [iso_errors(true), iso_errors(r_plain_bad/0, false)],
        ['override lax silent'-'r_plain_bad/0',
         'sibling stays ISO'-'r_plain_zdiv/0']),

    run_project('explicit is_iso under lax mode (interpreter)',
        [user:r_iso_type/0, user:r_iso_inst/0, user:r_iso_zdiv/0],
        [iso_errors(false)],
        ['explicit is_iso type_error'-'r_iso_type/0',
         'explicit is_iso instantiation_error'-'r_iso_inst/0',
         'explicit is_iso zero_divisor'-'r_iso_zdiv/0']),

    run_project('explicit is_lax under ISO mode (interpreter)',
        [user:r_lax_bad/0],
        [iso_errors(true)],
        ['explicit is_lax silent'-'r_lax_bad/0']),

    run_project('functions mode ISO-default + explicit',
        [user:r_ok/0, user:r_iso_type/0, user:r_lax_bad/0, user:r_plain_zdiv/0],
        [emit_mode(functions), iso_errors(true)],
        ['functions successful is'-'r_ok/0',
         'functions explicit is_iso'-'r_iso_type/0',
         'functions explicit is_lax'-'r_lax_bad/0',
         'functions ISO-default zdiv'-'r_plain_zdiv/0']),

    % ----- comparisons (ISO-R-2A) -----
    run_project('cmp success + false (iso default, interpreter)',
        [user:r_cmp_ok/0, user:r_cmp_false/0],
        [iso_errors(true)],
        ['cmp success'-'r_cmp_ok/0',
         'cmp false'-'r_cmp_false/0']),

    run_project('cmp ISO errors either operand (interpreter)',
        [user:r_lt_iso_inst_l/0, user:r_lt_iso_inst_r/0,
         user:r_lt_iso_type_l/0, user:r_lt_iso_type_r/0,
         user:r_lt_iso_zdiv_l/0, user:r_lt_iso_zdiv_r/0,
         user:r_lt_iso_precedence_inst/0, user:r_lt_iso_precedence_zdiv/0,
         user:r_gt_iso_inst/0, user:r_ge_iso_type/0,
         user:r_le_iso_zdiv/0, user:r_eq_iso_type/0,
         user:r_ne_iso_inst/0],
        [iso_errors(false)],
        ['<_iso inst L'-'r_lt_iso_inst_l/0',
         '<_iso inst R'-'r_lt_iso_inst_r/0',
         '<_iso type L'-'r_lt_iso_type_l/0',
         '<_iso type R'-'r_lt_iso_type_r/0',
         '<_iso zdiv L'-'r_lt_iso_zdiv_l/0',
         '<_iso zdiv R'-'r_lt_iso_zdiv_r/0',
         'cmp precedence instantiation'-'r_lt_iso_precedence_inst/0',
         'cmp precedence zero_divisor'-'r_lt_iso_precedence_zdiv/0',
         '>_iso inst'-'r_gt_iso_inst/0',
         '>=_iso type'-'r_ge_iso_type/0',
         '=<_iso zdiv'-'r_le_iso_zdiv/0',
         '=:=_iso type'-'r_eq_iso_type/0',
         'ne_iso inst'-'r_ne_iso_inst/0']),

    run_project('cmp lax silent (interpreter)',
        [user:r_lt_lax_bad/0, user:r_eq_lax_bad/0],
        [iso_errors(true)],
        ['<_lax silent'-'r_lt_lax_bad/0',
         '=:=_lax silent'-'r_eq_lax_bad/0']),

    run_project('cmp explicit iso under lax / lax under iso',
        [user:r_cmp_iso_under_lax/0, user:r_cmp_lax_under_iso/0],
        [iso_errors(false)],
        ['explicit <_iso under lax'-'r_cmp_iso_under_lax/0',
         'explicit <_lax under iso-mode sibling'-'r_cmp_lax_under_iso/0']),

    run_project('cmp explicit lax under ISO mode',
        [user:r_cmp_lax_under_iso/0],
        [iso_errors(true)],
        ['explicit <_lax under ISO'-'r_cmp_lax_under_iso/0']),

    run_project('cmp default rewrite ISO + lax',
        [user:r_cmp_plain_ok/0, user:r_cmp_plain_zdiv/0, user:r_cmp_plain_bad/0],
        [iso_errors(true), iso_errors(r_cmp_plain_bad/0, false)],
        ['default cmp ok'-'r_cmp_plain_ok/0',
         'default cmp zdiv ISO'-'r_cmp_plain_zdiv/0',
         'override cmp silent'-'r_cmp_plain_bad/0']),

    run_project('cmp functions mode ISO + explicit',
        [user:r_cmp_ok/0, user:r_cmp_false/0,
         user:r_lt_iso_type_l/0, user:r_lt_lax_bad/0,
         user:r_lt_iso_precedence_inst/0, user:r_lt_iso_precedence_zdiv/0,
         user:r_cmp_plain_zdiv/0],
        [emit_mode(functions), iso_errors(true)],
        ['functions cmp ok'-'r_cmp_ok/0',
         'functions cmp false'-'r_cmp_false/0',
         'functions <_iso type'-'r_lt_iso_type_l/0',
         'functions <_lax silent'-'r_lt_lax_bad/0',
         'functions precedence instantiation'-'r_lt_iso_precedence_inst/0',
         'functions precedence zero_divisor'-'r_lt_iso_precedence_zdiv/0',
         'functions default zdiv'-'r_cmp_plain_zdiv/0']),

    % ----- succ/2 (ISO-R-2B) -----
    run_project('succ happy paths (iso default, interpreter)',
        [user:r_succ_fwd/0, user:r_succ_back/0,
         user:r_succ_pair_ok/0, user:r_succ_pair_bad/0],
        [iso_errors(true)],
        ['succ forward'-'r_succ_fwd/0',
         'succ reverse'-'r_succ_back/0',
         'succ pair ok'-'r_succ_pair_ok/0',
         'succ pair mismatch'-'r_succ_pair_bad/0']),

    run_project('succ ISO errors (interpreter)',
        [user:r_succ_iso_inst/0, user:r_succ_iso_type_x/0,
         user:r_succ_iso_type_y/0, user:r_succ_iso_precedence_x/0,
         user:r_succ_iso_precedence_y/0, user:r_succ_iso_neg/0,
         user:r_succ_iso_domain/0, user:r_succ_iso_mismatch/0],
        [iso_errors(false)],
        ['succ_iso inst'-'r_succ_iso_inst/0',
         'succ_iso type X'-'r_succ_iso_type_x/0',
         'succ_iso type Y'-'r_succ_iso_type_y/0',
         'succ_iso precedence X type'-'r_succ_iso_precedence_x/0',
         'succ_iso precedence Y type'-'r_succ_iso_precedence_y/0',
         'succ_iso neg'-'r_succ_iso_neg/0',
         'succ_iso domain'-'r_succ_iso_domain/0',
         'succ_iso mismatch'-'r_succ_iso_mismatch/0']),

    run_project('succ lax silent + explicit lax under ISO',
        [user:r_succ_lax_bad/0, user:r_succ_lax_under_iso/0],
        [iso_errors(true)],
        ['succ_lax silent'-'r_succ_lax_bad/0',
         'succ_lax under ISO'-'r_succ_lax_under_iso/0']),

    run_project('succ explicit iso under lax mode',
        [user:r_succ_iso_under_lax/0, user:r_succ_iso_domain/0],
        [iso_errors(false)],
        ['succ_iso under lax'-'r_succ_iso_under_lax/0',
         'succ_iso domain under lax'-'r_succ_iso_domain/0']),

    run_project('succ default rewrite ISO + lax override',
        [user:r_succ_plain_iso/0, user:r_succ_plain_lax/0],
        [iso_errors(true), iso_errors(r_succ_plain_lax/0, false)],
        ['default succ ISO inst'-'r_succ_plain_iso/0',
         'override succ lax silent'-'r_succ_plain_lax/0']),

    run_project('succ module-scoped override isolation',
        [user:r_succ_mod_b/0, user:r_succ_plain_lax/0],
        [iso_errors(false),
         iso_errors(mod_a:r_succ_mod_b/0, true)],
        ['mod_b stays lax success'-'r_succ_mod_b/0',
         'bare override sibling lax'-'r_succ_plain_lax/0']),

    run_project('succ functions mode ISO + explicit',
        [user:r_succ_fwd/0, user:r_succ_back/0,
         user:r_succ_iso_domain/0, user:r_succ_iso_precedence_x/0,
         user:r_succ_iso_precedence_y/0, user:r_succ_lax_bad/0,
         user:r_succ_plain_iso/0],
        [emit_mode(functions), iso_errors(true)],
        ['functions succ forward'-'r_succ_fwd/0',
         'functions succ reverse'-'r_succ_back/0',
         'functions succ_iso domain'-'r_succ_iso_domain/0',
         'functions succ_iso precedence X'-'r_succ_iso_precedence_x/0',
         'functions succ_iso precedence Y'-'r_succ_iso_precedence_y/0',
         'functions succ_lax silent'-'r_succ_lax_bad/0',
         'functions default succ ISO'-'r_succ_plain_iso/0']),

    retract_iso_helpers,
    (   nb_getval(r_iso_smoke_failed, true) -> halt(1) ; halt(0) ).
