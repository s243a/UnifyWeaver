% test_wam_rust_fact_table_emit.pl
%
% T9 fact-table inline (Rust) — emission + classification-seam checks (no cargo).
% Verifies that with fact_table_inline(true) an eligible facts predicate lowers to
% the fact_table strategy (static row table + fact_table_attempt enumerator), and
% that WITHOUT the option it is unaffected (default path, no fact table).

:- use_module('../src/unifyweaver/targets/wam_rust_target',
              [write_wam_rust_project/3, emit_fact_table_rust/4]).

:- dynamic edge/2.
edge(a, 1). edge(a, 2). edge(b, 3). edge(a, 4). edge(c, 5). edge(b, 6).

safe_rmdir(Dir) :- ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ).

gen(Opts, Dir, Src) :-
    safe_rmdir(Dir),
    once(write_wam_rust_project([user:edge/2], [module_name(eg)|Opts], Dir)),
    atom_concat(Dir, '/src/lib.rs', LibRs),
    read_file_to_string(LibRs, Src, []).

:- begin_tests(wam_rust_fact_table_emit).

% the value-literal emitter maps each ground term kind to its Value variant, and
% the fn builds a OnceLock-cached table + first-arg hash index.
test(value_literals) :-
    emit_fact_table_rust(p/2, fact_info(2, [[foo, 7], [bar, -3]]), [], Code),
    assertion(sub_string(Code, _, _, _, "Value::Atom(\"foo\".to_string())")),
    assertion(sub_string(Code, _, _, _, "Value::Integer(7)")),
    assertion(sub_string(Code, _, _, _, "Value::Integer(-3)")),
    assertion(sub_string(Code, _, _, _, "pub fn p_2(vm: &mut WamState, a1: Value, a2: Value) -> bool")),
    assertion(sub_string(Code, _, _, _, "std::sync::OnceLock")),
    assertion(sub_string(Code, _, _, _, "fact_index_key()")),
    assertion(sub_string(Code, _, _, _, "vm.fact_table_attempt(__args, __cands, cont_pc)")).

% nested terms / lists / floats lower correctly.
test(value_literals_compound) :-
    emit_fact_table_rust(q/1, fact_info(1, [[f(a, [1, 2])], [3.5]]), [], Code),
    assertion(sub_string(Code, _, _, _, "Value::Str(\"f\".to_string(), vec![Value::Atom(\"a\".to_string()), Value::List(vec![Value::Integer(1), Value::Integer(2)])])")),
    assertion(sub_string(Code, _, _, _, "Value::Float(3.5)")).

% with the option: eligible predicate is classified fact_table.
test(opt_in_classifies_fact_table, [cleanup(safe_rmdir('output/test_t9_emit_on'))]) :-
    gen([fact_table_inline(true), t9_min_rows(4)], 'output/test_t9_emit_on', Src),
    assertion(sub_string(Src, _, _, _, "Strategy: fact_table")),
    assertion(sub_string(Src, _, _, _, "fn edge_2_table()")),
    assertion(sub_string(Src, _, _, _, "vm.fact_table_attempt")).

% default (no option): unchanged, no fact table emitted.
test(default_off_no_fact_table, [cleanup(safe_rmdir('output/test_t9_emit_off'))]) :-
    gen([], 'output/test_t9_emit_off', Src),
    assertion(\+ sub_string(Src, _, _, _, "Strategy: fact_table")),
    assertion(\+ sub_string(Src, _, _, _, "edge_2_table")).

% below threshold even with the option: not a fact table.
test(below_threshold_off, [cleanup(safe_rmdir('output/test_t9_emit_thresh'))]) :-
    gen([fact_table_inline(true), t9_min_rows(100)], 'output/test_t9_emit_thresh', Src),
    assertion(\+ sub_string(Src, _, _, _, "Strategy: fact_table")).

:- end_tests(wam_rust_fact_table_emit).
