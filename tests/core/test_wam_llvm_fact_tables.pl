:- encoding(utf8).
% test_wam_llvm_fact_tables.pl
% Verifies M4: indexed fact table emission as LLVM global constants.
%
% M4 provides Prolog predicates that take fact lists and emit LLVM IR
% global constant definitions. Kernels (M5) will scan these tables via
% direct GEP loops. This test verifies:
%   - Atom fact pairs are interned to stable IDs
%   - Weighted edges encode weights as LLVM double literals
%   - Empty lists produce a valid placeholder (not a syntax error)
%   - Full generated modules embedding these tables parse with llvm-as

:- use_module('../../src/unifyweaver/targets/wam_llvm_target',
    [llvm_emit_atom_fact2_table/3, llvm_emit_weighted_edge_table/3,
     write_wam_llvm_project/3]).
:- use_module(library(process)).

% Test fixture for llvm-as validation (must be at top level).
:- dynamic color/1.
color(red).

test_atom_fact2_emission :-
    format('--- atom_fact2 table emission ---~n'),
    llvm_emit_atom_fact2_table(test_cat_parent,
        [ fact('Physics', 'Science'),
          fact('Chemistry', 'Science'),
          fact('Science', 'Knowledge')
        ], Code),
    ( sub_atom_or_string(Code, _, _, _, 'AtomFactPair')
    -> format('  PASS: output contains %AtomFactPair references~n')
    ;  format('  FAIL: missing %AtomFactPair~n'),
       throw(missing_atom_fact_pair)
    ),
    ( sub_atom_or_string(Code, _, _, _, 'private constant [3 x %AtomFactPair]')
    -> format('  PASS: array type matches fact count~n')
    ;  format('  FAIL: array type mismatch~n'),
       throw(wrong_array_type)
    ),
    ( sub_atom_or_string(Code, _, _, _, '@test_cat_parent')
    -> format('  PASS: global name @test_cat_parent present~n')
    ;  format('  FAIL: global name missing~n'),
       throw(missing_global_name)
    ).

test_weighted_edge_emission :-
    format('--- weighted_edge table emission ---~n'),
    llvm_emit_weighted_edge_table(test_weights,
        [ edge(ml, ai, 0.12),
          edge(ai, cs, 0.18),
          edge(cs, science, 0.30)
        ], Code),
    ( sub_atom_or_string(Code, _, _, _, 'WeightedFact')
    -> format('  PASS: output contains %WeightedFact references~n')
    ;  format('  FAIL: missing %WeightedFact~n'),
       throw(missing_weighted_fact)
    ),
    ( sub_atom_or_string(Code, _, _, _, 'double 0.12')
    -> format('  PASS: weight 0.12 encoded as double literal~n')
    ;  format('  FAIL: weight 0.12 not found~n'),
       throw(missing_weight)
    ),
    ( sub_atom_or_string(Code, _, _, _, '@test_weights')
    -> format('  PASS: global name @test_weights present~n')
    ;  format('  FAIL: global name missing~n')
    ).

test_integer_weight_formatting :-
    format('--- integer weights formatted as doubles ---~n'),
    % LLVM requires "1.0" not "1" for double literals
    llvm_emit_weighted_edge_table(test_int_w,
        [edge(a, b, 1), edge(c, d, 0)], Code),
    ( sub_atom_or_string(Code, _, _, _, 'double 1.0')
    -> format('  PASS: integer 1 emitted as 1.0~n')
    ;  format('  FAIL: integer not formatted as double~n'),
       throw(int_weight_bad_format)
    ),
    ( sub_atom_or_string(Code, _, _, _, 'double 0.0')
    -> format('  PASS: integer 0 emitted as 0.0~n')
    ;  format('  FAIL: zero not formatted as double~n')
    ).

test_empty_table :-
    format('--- empty fact list produces valid placeholder ---~n'),
    llvm_emit_atom_fact2_table(empty_test, [], Code),
    ( sub_atom_or_string(Code, _, _, _, '[1 x %AtomFactPair]')
    -> format('  PASS: empty list uses 1-entry placeholder~n')
    ;  format('  FAIL: empty list produced unexpected output~n'),
       throw(bad_empty_table)
    ).

test_llvm_as_accepts_tables :-
    format('--- llvm-as accepts emitted tables ---~n'),
    ( process_which('llvm-as')
    -> test_llvm_as
    ;  format('  SKIP: llvm-as not found on PATH~n')
    ).

test_llvm_as :-
    llvm_emit_atom_fact2_table(demo_atoms,
        [fact(a, b), fact(b, c), fact(c, d)], AtomCode),
    llvm_emit_weighted_edge_table(demo_weights,
        [edge(a, b, 0.5), edge(b, c, 1.5)], WeightCode),
    tmp_file_stream(text, LLPath, Stream), close(Stream),
    write_wam_llvm_project([user:color/1], [module_name('m4_test')], LLPath),
    setup_call_cleanup(
        open(LLPath, append, Out),
        ( write(Out, '\n'), write(Out, AtomCode),
          write(Out, '\n'), write(Out, WeightCode),
          write(Out, '\n') ),
        close(Out)),
    format('  Wrote module: ~w~n', [LLPath]),
    atom_concat(LLPath, '.bc', BCPath),
    format(atom(Cmd), 'llvm-as ~w -o ~w 2>&1', [LLPath, BCPath]),
    shell(Cmd, Exit),
    ( Exit == 0
    -> format('  PASS: llvm-as accepted module with fact tables~n')
    ;  format('  FAIL: llvm-as exit=~w~n', [Exit])
    ),
    catch(delete_file(LLPath), _, true),
    catch(delete_file(BCPath), _, true).

process_which(Tool) :-
    catch(
        ( process_create(path(which), [Tool],
              [stdout(pipe(Out)), stderr(null), process(PID)]),
          read_string(Out, _, _),
          close(Out),
          process_wait(PID, exit(0))
        ),
        _,
        fail).

sub_atom_or_string(Haystack, Before, Length, After, Needle) :-
    ( atom(Haystack) -> sub_atom(Haystack, Before, Length, After, Needle)
    ; string(Haystack) -> sub_string(Haystack, Before, Length, After, Needle)
    ; atom_string(Atom, Haystack), sub_atom(Atom, Before, Length, After, Needle)
    ).

test_all :-
    test_atom_fact2_emission,
    test_weighted_edge_emission,
    test_integer_weight_formatting,
    test_empty_table,
    catch(test_llvm_as_accepts_tables, E,
        format('  ERROR in llvm-as test: ~w~n', [E])).

:- initialization(test_all, main).
