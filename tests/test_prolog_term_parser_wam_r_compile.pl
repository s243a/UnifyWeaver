% =============================================================================
% test_prolog_term_parser_wam_r_compile.pl
%
% Structural compile-to-target test: feeds every predicate of
% src/unifyweaver/core/prolog_term_parser.pl through
% write_wam_r_project/3 and verifies (a) the generated R is syntactically
% valid, (b) every parser predicate has a generated label and wrapper
% function, and (c) a cut-free subset runs end-to-end via Rscript.
%
% This is the cross-target proof point: the same canonical-Prolog parser
% source can be transpiled to a WAM-R project. End-to-end execution of
% the FULL parser is not yet covered; the WAM-R cut scaffold drops only
% the most recent choice point (runtime.R.mustache: cut comment near
% `pred == "!/0"`), which loses cut-barrier semantics when the cut sits
% in a clause body that calls another multi-clause predicate. The parser
% relies on those exact semantics via take_ident, ident_cont, and
% friends. Filed as a separate follow-up; this PR ships the source +
% structural compilation, not the runtime swap-in.
% =============================================================================

:- use_module(library(plunit)).
:- use_module(library(filesex)).
:- use_module(library(process)).
:- use_module('../src/unifyweaver/core/prolog_term_parser').
:- use_module('../src/unifyweaver/targets/wam_r_target').

:- begin_tests(prolog_term_parser_wam_r_compile).

% Helper: enumerate every clause-defined predicate in the parser module
% and return them as a sorted list of Module:Name/Arity terms.
parser_predicates(Preds) :-
    findall(prolog_term_parser:N/A,
            (   current_predicate(prolog_term_parser:N/A),
                functor(H, N, A),
                once(clause(prolog_term_parser:H, _)),
                % Exclude imported predicates (member/2, append/3,
                % reverse/2 from the lists library) -- clause/2 finds
                % them through the module qualifier, but they aren't
                % defined here and shouldn't be compiled as if they
                % were. Without this filter, the SWI tests' calls
                % autoload list helpers into prolog_term_parser, the
                % filter sees them as "predicates with clauses", and
                % the compiled R project ends up with conflicting
                % shadow definitions of the runtime's library calls.
                \+ predicate_property(prolog_term_parser:H,
                                       imported_from(_))
            ),
            Raw),
    sort(Raw, Preds).

% Helper: tmp project dir under /tmp/prolog_term_parser_compile_e2e_<stamp>.
fresh_tmp_dir(Dir) :-
    get_time(T),
    StampF is T,
    format(atom(Dir), '/tmp/prolog_term_parser_compile_e2e_~w', [StampF]),
    (   exists_directory(Dir)
    ->  delete_directory_and_contents(Dir)
    ;   true
    ).

rscript_available :-
    catch((
        process_create(path('Rscript'), ['--version'],
                       [ stdout(null), stderr(null), process(PID) ]),
        process_wait(PID, exit(0))
    ), _, fail).

% ---------------------------------------------------------------------------
% Test: every parser predicate compiles without error and the project
% writer produces the expected files.
% ---------------------------------------------------------------------------

test(parser_compiles_to_wam_r_project) :- once((
    parser_predicates(ParserPreds),
    fresh_tmp_dir(TmpDir),
    retractall(user:drv_compile_only/0),
    assertz((user:drv_compile_only :-
        canonical_op_table(_Ops),
        write(ok), nl)),
    write_wam_r_project([user:drv_compile_only/0 | ParserPreds],
                        [], TmpDir),
    directory_file_path(TmpDir, 'R/generated_program.R', GenPath),
    directory_file_path(TmpDir, 'R/wam_runtime.R',       RtPath),
    assertion(exists_file(GenPath)),
    assertion(exists_file(RtPath)),
    read_file_to_string(GenPath, Code, []),
    % Sanity: every parser predicate has a generated dispatcher
    % wrapper (pred_<name>) so the runtime can resolve calls into
    % the compiled body.
    forall(member(prolog_term_parser:N/A, ParserPreds), (
        format(atom(WrapperPattern), 'pred_~w <- function(', [N]),
        assertion(sub_atom(Code, _, _, _, WrapperPattern)),
        format(atom(LabelPattern), '"~w/~w" =', [N, A]),
        assertion(sub_atom(Code, _, _, _, LabelPattern))
    )),
    delete_directory_and_contents(TmpDir))).

% ---------------------------------------------------------------------------
% Test: a cut-free subset runs end-to-end via Rscript. Auto-skips when
% Rscript isn't on PATH (matches the convention in
% test_wam_r_generator.pl).
% ---------------------------------------------------------------------------

test(cut_free_subset_runs_via_rscript) :-
    once((
        rscript_available
    ->  cut_free_subset_runs_body
    ;   true
    )).

cut_free_subset_runs_body :-
    parser_predicates(ParserPreds),
    fresh_tmp_dir(TmpDir),
    % Cut-free drivers: exercise predicates whose semantics don't
    % depend on cut-barrier scoping.
    %   * canonical_op_table/1: single fact, no body.
    %   * is_infix_type/1: 3 facts; head-pattern dispatch.
    %   * rhs_max_prec/3: 3 clauses, head-pattern dispatch + arithmetic.
    %   * starts_term/1: 6 facts.
    retractall(user:drv_canon_len/0),
    retractall(user:drv_infix/0),
    retractall(user:drv_rhs_xfy/0),
    retractall(user:drv_rhs_yfx/0),
    retractall(user:drv_starts/0),
    assertz((user:drv_canon_len :-
        canonical_op_table(Ops),
        length(Ops, N),
        write('len='), write(N), nl)),
    assertz((user:drv_infix :-
        is_infix_type(xfy),
        write(ok), nl)),
    assertz((user:drv_rhs_xfy :-
        rhs_max_prec(xfy, 700, R),
        write(R), nl)),
    assertz((user:drv_rhs_yfx :-
        rhs_max_prec(yfx, 500, R),
        write(R), nl)),
    assertz((user:drv_starts :-
        starts_term(tk_lparen),
        write(ok), nl)),
    Drivers = [user:drv_canon_len/0, user:drv_infix/0,
               user:drv_rhs_xfy/0, user:drv_rhs_yfx/0,
               user:drv_starts/0],
    append(Drivers, ParserPreds, Compile),
    write_wam_r_project(Compile, [], TmpDir),
    directory_file_path(TmpDir, 'R', RDir),
    run_rscript_query(RDir, 'drv_canon_len/0', LenOut),
    assertion(sub_string(LenOut, _, _, _, "len=")),
    run_rscript_query(RDir, 'drv_infix/0', InfixOut),
    assertion(sub_string(InfixOut, _, _, _, "ok")),
    run_rscript_query(RDir, 'drv_rhs_xfy/0', RhsXfyOut),
    assertion(sub_string(RhsXfyOut, _, _, _, "700")),
    run_rscript_query(RDir, 'drv_rhs_yfx/0', RhsYfxOut),
    assertion(sub_string(RhsYfxOut, _, _, _, "499")),
    run_rscript_query(RDir, 'drv_starts/0', StartsOut),
    assertion(sub_string(StartsOut, _, _, _, "ok")),
    delete_directory_and_contents(TmpDir).

% ---------------------------------------------------------------------------
% Test: the full compiled parser produces the same terms under Rscript as
% it does under SWI. End-to-end proof that the cross-target loop works.
% Auto-skips when Rscript isn't on PATH.
% ---------------------------------------------------------------------------

test(full_parser_e2e_via_rscript) :-
    once((
        rscript_available
    ->  full_parser_e2e_body
    ;   true
    )).

full_parser_e2e_body :-
    parser_predicates(ParserPreds),
    fresh_tmp_dir(TmpDir),
    % Drivers cover the parser surface: bare atom, integer, simple
    % compound, multi-arg compound, list, infix operator with
    % precedence, postfix operator, repeated variable (var sharing
    % within one parse), nested compound + list. Each prints the
    % parsed term so we can sub-string match the expected output.
    retractall(user:dpe_atom/0),
    retractall(user:dpe_num/0),
    retractall(user:dpe_compound/0),
    retractall(user:dpe_arith/0),
    retractall(user:dpe_list/0),
    retractall(user:dpe_postfix/0),
    retractall(user:dpe_var/0),
    retractall(user:dpe_nested/0),
    assertz((user:dpe_atom :-
        canonical_op_table(O), parse_term_from_atom('foo', O, T),
        write(T), nl)),
    assertz((user:dpe_num :-
        canonical_op_table(O), parse_term_from_atom('42', O, T),
        write(T), nl)),
    assertz((user:dpe_compound :-
        canonical_op_table(O), parse_term_from_atom('foo(a, b)', O, T),
        write(T), nl)),
    assertz((user:dpe_arith :-
        canonical_op_table(O), parse_term_from_atom('1 + 2 * 3', O, T),
        write(T), nl)),
    assertz((user:dpe_list :-
        canonical_op_table(O), parse_term_from_atom('[a, b]', O, T),
        write(T), nl)),
    assertz((user:dpe_postfix :-
        canonical_op_table(B), Ops = [op('!', 100, yf) | B],
        parse_term_from_atom('5!', Ops, T),
        write(T), nl)),
    assertz((user:dpe_var :-
        canonical_op_table(O), parse_term_from_atom('f(X, X)', O, T),
        ( T = f(A, B), var(A), A == B
        -> write(shared_var) ; write(not_shared) ),
        nl)),
    assertz((user:dpe_nested :-
        canonical_op_table(O),
        parse_term_from_atom('foo(bar(1), [a, b])', O, T),
        write(T), nl)),
    Drivers = [user:dpe_atom/0, user:dpe_num/0, user:dpe_compound/0,
               user:dpe_arith/0, user:dpe_list/0, user:dpe_postfix/0,
               user:dpe_var/0, user:dpe_nested/0],
    append(Drivers, ParserPreds, Compile),
    write_wam_r_project(Compile, [intern_atoms(['!'])], TmpDir),
    directory_file_path(TmpDir, 'R', RDir),
    run_rscript_query(RDir, 'dpe_atom/0', AtomOut),
    assertion(sub_string(AtomOut, _, _, _, "foo")),
    run_rscript_query(RDir, 'dpe_num/0', NumOut),
    assertion(sub_string(NumOut, _, _, _, "42")),
    run_rscript_query(RDir, 'dpe_compound/0', CompOut),
    assertion(sub_string(CompOut, _, _, _, "foo(a,b)")),
    run_rscript_query(RDir, 'dpe_arith/0', ArithOut),
    % WAM-R's write/1 prints structs in canonical-functor form; the
    % Pratt parser produced `+(1, *(2, 3))` so precedence is right
    % (mult binds tighter than add).
    assertion(sub_string(ArithOut, _, _, _, "+(1,*(2,3))")),
    run_rscript_query(RDir, 'dpe_list/0', ListOut),
    assertion(sub_string(ListOut, _, _, _, "[a,b]")),
    run_rscript_query(RDir, 'dpe_postfix/0', PostfixOut),
    assertion(sub_string(PostfixOut, _, _, _, "!(5)")),
    run_rscript_query(RDir, 'dpe_var/0', VarOut),
    assertion(sub_string(VarOut, _, _, _, "shared_var")),
    run_rscript_query(RDir, 'dpe_nested/0', NestedOut),
    assertion(sub_string(NestedOut, _, _, _, "foo(bar(1),[a,b])")),
    delete_directory_and_contents(TmpDir).

run_rscript_query(RDir, Query, Out) :-
    process_create(path('Rscript'),
                   ['generated_program.R', Query],
                   [ cwd(RDir),
                     stdout(pipe(OutStream)),
                     stderr(pipe(ErrStream)),
                     process(PID)
                   ]),
    read_string(OutStream, _, Out), close(OutStream),
    read_string(ErrStream, _, _),   close(ErrStream),
    process_wait(PID, _).

:- end_tests(prolog_term_parser_wam_r_compile).
