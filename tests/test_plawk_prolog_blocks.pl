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

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(plawk_prolog_blocks, [condition(clang_available)]).

test(blocks_lift_out_of_the_awk_surface) :-
    Src = "@prolog\nplawk_pbt_w(I, F, R) :- R is I * F.\n@end\nBEGIN { BINFMT = \"i64 f64\" }\n{ wsum += float(plawk_pbt_w($1, $2)) }\nEND { print wsum }\n",
    plawk_parse_source(Src, Program, Clauses),
    assertion(Clauses = [(plawk_pbt_w(_, _, _) :- _)]),
    Program = program(_, [rule(always, _)], _),
    % the same source parses through the 2-arity entry too
    plawk_parse_string(Src, _).

test(tagged_markers_and_rejections) :-
    % heredoc-style tags close only on the exact same tag
    plawk_parse_source("@prolog-x9\nplawk_pbt_t(1).\n@end-x9\nBEGIN { BINFMT = \"i64\" } { n++ } END { print n }\n",
        _, [plawk_pbt_t(1)]),
    % a mismatched tag leaves the block unterminated
    assertion(\+ plawk_parse_source("@prolog-a\nfoo(1).\n@end-b\nBEGIN { BINFMT = \"i64\" } { n++ } END { print n }\n", _, _)),
    % ... as does a missing @end
    assertion(\+ plawk_parse_source("@prolog\nfoo(1).\nBEGIN { BINFMT = \"i64\" } { n++ } END { print n }\n", _, _)),
    % directives are not clauses
    plawk_parse_source("@prolog\n:- dynamic foo/1.\nfoo(1).\n@end\nBEGIN { BINFMT = \"i64\" } { n++ } END { print n }\n",
        _, DirClauses),
    assertion(\+ plawk_prolog_block_preds(DirClauses, _)).

test(surface_embedded_predicates_end_to_end) :-
    % Guard and expression functions defined in the program text.
    Src = "# weights, with the logic in Prolog\n@prolog\nplawk_pbt_weight(I, F, R) :- R is I * F.\nplawk_pbt_hot(X) :- X > 100.\n@end\n\nBEGIN { BINFMT = \"i64 f64\" }\nplawk_pbt_hot($1) {\n  wsum += float(plawk_pbt_weight($1, $2))\n}\nEND { print wsum }\n",
    run_pb_smoke(Src,
        [rec(200, 1.5), rec(5, 2.5), rec(300, 0.25)],
        "375\n").

test(surface_embedded_dcg_parses_blob_payloads) :-
    % The Tier-2 story in ONE file: the record loop frames natively,
    % and a DCG defined in the @prolog block parses each payload.
    % backtracking-based grammar (greedy digit clause before the stop
    % clause, a constant-in-list head), with the leaf rules in -->
    % form to exercise DCG expansion of embedded clauses
    Src = "@prolog-t2\nplawk_pbt_sum(Payload, Sum) :- atom_codes(Payload, Codes), plawk_pbt_nums(Sum, Codes, []).\nplawk_pbt_nums(Sum, S0, S) :- plawk_pbt_num(N, S0, S1), plawk_pbt_rest(N, Sum, S1, S).\nplawk_pbt_rest(Acc, Sum, [0', | S0], S) :- plawk_pbt_num(N, S0, S1), Acc2 is Acc + N, plawk_pbt_rest(Acc2, Sum, S1, S).\nplawk_pbt_rest(Sum, Sum, S, S).\nplawk_pbt_num(N) --> plawk_pbt_digit(D), plawk_pbt_digits(D, N).\nplawk_pbt_digits(Acc, N) --> plawk_pbt_digit(D), { Acc2 is Acc * 10 + D }, plawk_pbt_digits(Acc2, N).\nplawk_pbt_digits(N, N) --> [].\nplawk_pbt_digit(D) --> [C], { C >= 48, C =< 57, D is C - 48 }.\n@end-t2\nBEGIN { BINFMT = \"i64 blob32\" }\n$1 > 0 { total += plawk_pbt_sum($2) }\nEND { print total }\n",
    run_pb_blob_smoke(Src,
        [brec(1, "12,7"), brec(2, "100"), brec(-5, "9")],
        "119\n").

test(surface_embedded_dcg_with_ite_cut_and_code_type) :-
    % Regression for the "returned 0" incident: this exact grammar --
    % if-then-else with a binding condition, a cut in the digits rule,
    % and code_type/2 -- silently returned 0 because code_type was not
    % a WAM builtin and its call lowered to label index 0. code_type
    % is now a builtin (sharing char_type's classifier), and unknown
    % callees fail the COMPILE loudly instead.
    Src = "@prolog-t3\nplawk_pbt3_sum(Payload, Sum) :- atom_codes(Payload, Codes), plawk_pbt3_nums(0, Sum, Codes, []).\nplawk_pbt3_nums(Acc, Sum) --> plawk_pbt3_num(N), ( \",\" -> { Acc1 is Acc + N }, plawk_pbt3_nums(Acc1, Sum) ; { Sum is Acc + N } ).\nplawk_pbt3_num(N) --> plawk_pbt3_digits(Ds), { Ds \\== [], number_codes(N, Ds) }.\nplawk_pbt3_digits([D | Ds]) --> [D], { code_type(D, digit) }, !, plawk_pbt3_digits(Ds).\nplawk_pbt3_digits([]) --> [].\n@end-t3\nBEGIN { BINFMT = \"i64 blob32\" }\n$1 > 0 { total += plawk_pbt3_sum($2) }\nEND { print total }\n",
    run_pb_blob_smoke(Src,
        [brec(1, "12,7"), brec(2, "100"), brec(-5, "9")],
        "119\n").

test(calling_an_uncompiled_predicate_fails_the_compile,
        [throws(error(existence_error(procedure, _), _))]) :-
    % The trap behind that incident: a call to a predicate that is not
    % compiled into the module used to lower to label index 0 with
    % only a stderr warning -- a silent runtime failure. It is now a
    % compile-time existence error.
    Src = "@prolog\nplawk_pbt_broken(X, R) :- plawk_pbt_no_such_helper(X, R).\n@end\nBEGIN { BINFMT = \"i64\" }\n{ total += plawk_pbt_broken($1) }\nEND { print total }\n",
    build_pb_probe(Src, _, _).

:- end_tests(plawk_prolog_blocks).

% --- helpers ---------------------------------------------------------------

write_i64_le(Out, V) :-
    V64 is V /\ 0xFFFFFFFFFFFFFFFF,
    forall(between(0, 7, I),
        ( Byte is (V64 >> (8 * I)) /\ 0xFF, put_byte(Out, Byte) )).

% dyadic doubles used by these tests
double_bits(1.5,  0x3FF8000000000000).
double_bits(2.5,  0x4004000000000000).
double_bits(0.25, 0x3FD0000000000000).

write_pb_records(Path, Recs) :-
    setup_call_cleanup(
        open(Path, write, Out, [type(binary)]),
        forall(member(Rec, Recs), write_pb_record(Out, Rec)),
        close(Out)).

write_pb_record(Out, rec(I, F)) :-
    write_i64_le(Out, I),
    double_bits(F, Bits),
    write_i64_le(Out, Bits).
write_pb_record(Out, brec(I, Payload)) :-
    write_i64_le(Out, I),
    string_codes(Payload, Codes),
    length(Codes, Len),
    write_i64_le(Out, Len),
    forall(member(C, Codes), put_byte(Out, C)).

% The whole pipeline from ONE source string: lift @prolog clauses,
% install them, compile them with the WAM target, then append the
% driver built with the vm counts.
build_pb_probe(Src, Dir, BinPath) :-
    tmp_root(Root),
    directory_file_path(Root, 'uw_plawk_prolog_blocks', Dir),
    clean_dir(Dir),
    make_directory_path(Dir),
    directory_file_path(Dir, 'input.bin', InputPath),
    plawk_parse_source(Src, Program, Clauses),
    plawk_prolog_block_preds(Clauses, Preds),
    directory_file_path(Dir, 'probe.ll', LLPath),
    write_wam_llvm_project(Preds, [module_name('plawk_prolog_blocks')], LLPath),
    wam_llvm_last_compile_counts(InstrCount, LabelCount),
    plawk_program_native_driver_ir(Program, InputPath,
        [wam_vm(InstrCount, LabelCount)], DriverIR),
    setup_call_cleanup(
        open(LLPath, append, Out, [encoding(utf8)]),
        ( nl(Out), write(Out, DriverIR) ),
        close(Out)),
    directory_file_path(Dir, 'probe_bin', BinPath),
    format(atom(Cmd), 'clang -w ~w -o ~w -lm 2>&1', [LLPath, BinPath]),
    process_create(path(sh), ['-c', Cmd],
                   [stdout(pipe(Stdout)), stderr(std), process(Pid)]),
    read_string(Stdout, _, BuildOut),
    close(Stdout),
    process_wait(Pid, Status),
    ( Status == exit(0)
    -> true
    ;  format(user_error, "~n[plawk prolog blocks build output]~n~w~n", [BuildOut]),
       throw(plawk_prolog_blocks_build_failed(Status))
    ).

run_pb(BinPath, OutStr) :-
    process_create(BinPath, [],
                   [stdout(pipe(Stdout)), stderr(std), process(Pid)]),
    read_string(Stdout, _, OutStr),
    close(Stdout),
    process_wait(Pid, Status),
    assertion(Status == exit(0)).

run_pb_smoke(Src, Recs, ExpectedOutput) :-
    build_pb_probe(Src, Dir, BinPath),
    directory_file_path(Dir, 'input.bin', InputPath),
    write_pb_records(InputPath, Recs),
    run_pb(BinPath, OutStr),
    assertion(OutStr == ExpectedOutput),
    !.

run_pb_blob_smoke(Src, Recs, ExpectedOutput) :-
    build_pb_probe(Src, Dir, BinPath),
    directory_file_path(Dir, 'input.bin', InputPath),
    write_pb_records(InputPath, Recs),
    run_pb(BinPath, OutStr),
    assertion(OutStr == ExpectedOutput),
    !.
