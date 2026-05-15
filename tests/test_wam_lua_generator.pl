:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0

:- use_module(library(plunit)).
:- use_module(library(filesex)).
:- use_module(library(process)).
:- use_module(library(readutil)).
:- use_module('../src/unifyweaver/targets/wam_lua_target').
:- use_module('../src/unifyweaver/targets/wam_lua_lowered_emitter').
:- use_module('../src/unifyweaver/core/target_registry').

:- begin_tests(wam_lua_generator).

:- dynamic user:wam_lua_fact/1.
:- dynamic user:wam_lua_choice/1.
:- dynamic user:wam_lua_caller/1.
:- dynamic user:wam_lua_fa_fact/1.
:- dynamic user:wam_lua_fa_basic/0.
:- dynamic user:wam_lua_fa_greet/0.
:- dynamic user:wam_lua_greet/2.
:- dynamic user:wam_lua_num/1.
:- dynamic user:wam_lua_agg_count/0.
:- dynamic user:wam_lua_agg_sum/0.
:- dynamic user:wam_lua_agg_min/0.
:- dynamic user:wam_lua_agg_max/0.
:- dynamic user:wam_lua_agg_set/0.
:- dynamic user:wam_lua_stream_fact/2.
:- dynamic user:wam_lua_stream_caller/2.
:- dynamic user:wam_lua_ext_fact/2.
:- dynamic user:wam_lua_ext_caller/2.
:- dynamic user:wam_lua_member_basic/0.
:- dynamic user:wam_lua_member_backtrack/0.
:- dynamic user:wam_lua_member_no/0.
:- dynamic user:wam_lua_length_basic/0.
:- dynamic user:wam_lua_length_bind/0.
:- dynamic user:wam_lua_length_no/0.
:- dynamic user:wam_lua_type_builtins/0.
:- dynamic user:wam_lua_type_var/0.
:- dynamic user:wam_lua_type_no/0.
:- dynamic user:wam_lua_compare_builtins/0.
:- dynamic user:wam_lua_functor_read/0.
:- dynamic user:wam_lua_functor_atom/0.
:- dynamic user:wam_lua_functor_construct/0.
:- dynamic user:wam_lua_arg_struct/0.
:- dynamic user:wam_lua_arg_list/0.
:- dynamic user:wam_lua_arg_no/0.
:- dynamic user:wam_lua_univ_decompose_struct/0.
:- dynamic user:wam_lua_univ_decompose_atom/0.
:- dynamic user:wam_lua_univ_decompose_list/0.
:- dynamic user:wam_lua_univ_compose_struct/0.
:- dynamic user:wam_lua_univ_compose_atom/0.
:- dynamic user:wam_lua_univ_no/0.
:- dynamic user:wam_lua_copy_term_ground/0.
:- dynamic user:wam_lua_copy_term_fresh/0.
:- dynamic user:wam_lua_copy_term_sharing/0.
:- dynamic user:wam_lua_copy_term_independent_vars/0.
:- dynamic user:wam_lua_cut_choice/1.
:- dynamic user:wam_lua_if_then_else_then/0.
:- dynamic user:wam_lua_if_then_else_else/0.
:- dynamic user:wam_lua_naf_true/0.
:- dynamic user:wam_lua_naf_false/0.
:- dynamic user:wam_lua_naf_member/0.
:- dynamic user:wam_lua_naf_member_no/0.
:- dynamic user:wam_lua_write_line/0.

user:wam_lua_fact(a).
user:wam_lua_choice(a).
user:wam_lua_choice(b).
user:wam_lua_caller(X) :- user:wam_lua_fact(X).
user:wam_lua_fa_fact(a).
user:wam_lua_fa_fact(b).
user:wam_lua_fa_basic :-
    findall(X, user:wam_lua_fa_fact(X), L),
    L = [a, b].
user:wam_lua_greet(X, hello) :- X = world.
user:wam_lua_greet(X, goodbye) :- X = moon.
user:wam_lua_fa_greet :-
    findall(X, user:wam_lua_greet(X, hello), L),
    L = [world].
user:wam_lua_num(1).
user:wam_lua_num(2).
user:wam_lua_num(2).
user:wam_lua_num(3).
user:wam_lua_agg_count :-
    aggregate_all(count, user:wam_lua_num(_), N),
    N = 4.
user:wam_lua_agg_sum :-
    aggregate_all(sum(X), user:wam_lua_num(X), S),
    S = 8.
user:wam_lua_agg_min :-
    aggregate_all(min(X), user:wam_lua_num(X), M),
    M = 1.
user:wam_lua_agg_max :-
    aggregate_all(max(X), user:wam_lua_num(X), M),
    M = 3.
user:wam_lua_agg_set :-
    aggregate_all(set(X), user:wam_lua_num(X), S),
    S = [1, 2, 3].
user:wam_lua_stream_fact(a, b).
user:wam_lua_stream_fact(a, c).
user:wam_lua_stream_caller(X, Y) :- user:wam_lua_stream_fact(X, Y).
user:wam_lua_ext_caller(X, Y) :- user:wam_lua_ext_fact(X, Y).
user:wam_lua_member_basic :- member(b, [a, b, c]).
user:wam_lua_member_backtrack :- member(X, [a, b, c]), X = c.
user:wam_lua_member_no :- member(d, [a, b, c]).
user:wam_lua_length_basic :- length([a, b, c], 3).
user:wam_lua_length_bind :- length([a, b, c], N), N = 3.
user:wam_lua_length_no :- length([a, b, c], 2).
user:wam_lua_type_builtins :-
    atom(a),
    integer(3),
    number(3),
    compound(f(a)),
    nonvar(a),
    is_list([a, b]).
user:wam_lua_type_var :- var(_).
user:wam_lua_type_no :- atom(3).
user:wam_lua_compare_builtins :-
    3 >= 2,
    2 =< 3,
    3 =:= 3,
    3 =\= 2,
    a == a.
user:wam_lua_functor_read :- functor(f(a, b), f, 2).
user:wam_lua_functor_atom :- functor(a, a, 0).
user:wam_lua_functor_construct :-
    functor(T, f, 2),
    compound(T),
    arg(1, T, A),
    var(A).
user:wam_lua_arg_struct :- arg(2, f(a, b), b).
user:wam_lua_arg_list :- arg(2, [a, b], [b]).
user:wam_lua_arg_no :- arg(3, f(a, b), _).
user:wam_lua_univ_decompose_struct :- f(a, b) =.. [f, a, b].
user:wam_lua_univ_decompose_atom :- a =.. [a].
user:wam_lua_univ_decompose_list :-
    [a, b] =.. L,
    arg(1, L, '[|]'),
    arg(2, L, Tail),
    arg(1, Tail, a).
user:wam_lua_univ_compose_struct :-
    T =.. [f, a, b],
    T = f(a, b).
user:wam_lua_univ_compose_atom :-
    T =.. [a],
    T = a.
user:wam_lua_univ_no :- f(a) =.. [g, a].
user:wam_lua_copy_term_ground :-
    copy_term(f(a, b), C),
    C = f(a, b).
user:wam_lua_copy_term_fresh :-
    copy_term(f(X), C),
    C = f(a),
    var(X).
user:wam_lua_copy_term_sharing :-
    copy_term(f(X, X), C),
    C = f(A, B),
    A = b,
    B = b,
    var(X).
user:wam_lua_copy_term_independent_vars :-
    copy_term(f(_, _), C),
    C = f(A, B),
    A = a,
    B = b.
user:wam_lua_cut_choice(X) :-
    (X = a ; X = b),
    !,
    X = a.
user:wam_lua_if_then_else_then :-
    (true -> true ; fail).
user:wam_lua_if_then_else_else :-
    (fail -> fail ; true).
user:wam_lua_naf_true :- \+ fail.
user:wam_lua_naf_false :- \+ true.
user:wam_lua_naf_member :- \+ member(d, [a, b, c]).
user:wam_lua_naf_member_no :- \+ member(a, [a, b, c]).
user:wam_lua_write_line :-
    write(hello),
    nl.

test(exports) :-
    assertion(current_predicate(wam_lua_target:write_wam_lua_project/3)),
    assertion(current_predicate(wam_lua_target:compile_wam_predicate_to_lua/4)),
    assertion(current_predicate(wam_lua_target:lua_foreign_predicate/3)),
    assertion(current_predicate(wam_lua_target:init_lua_atom_intern_table/0)),
    assertion(current_predicate(wam_lua_lowered_emitter:wam_lua_lowerable/3)).

test(registry) :-
    assertion(target_exists(wam_lua)),
    assertion(target_family(wam_lua, lua)),
    assertion(target_module(wam_lua, wam_lua_target)).

test(shared_wam_tokenizer_bridge) :-
    wam_lua_target:tokenize_wam_line("    put_constant 'Has,comma', A1", T1),
    assertion(T1 == ["put_constant", "Has,comma", "A1"]),
    wam_lua_target:tokenize_wam_line("    put_structure ,/2, A1", T2),
    assertion(T2 == ["put_structure", ",/2", "A1"]).

test(shared_wam_parser_keeps_lua_extensions) :-
    WamText = "p/0:\n    arg 1, Y1, X3\n    proceed\n",
    once(wam_lua_target:wam_code_to_lua_data(WamText, [], Instrs, Labels)),
    assertion(Labels == ["  [\"p/0\"] = 1"]),
    assertion(Instrs == ["I.ArgInstr(1, 201, 103)", 'I.Proceed()']).

test(lua_parity_guard) :-
    read_file_to_string('templates/targets/lua_wam/runtime.lua.mustache', Runtime, []),
    read_file_to_string('tests/test_wam_lua_generator.pl', Tests, []),
    read_file_to_string('docs/design/WAM_LUA_PARITY_AUDIT.md', Audit, []),
    assert_contains_all(Runtime,
        [ 'builtin_member',
          'builtin_length',
          'type_builtin',
          'builtin_functor',
          'builtin_arg',
          'builtin_univ',
          'builtin_copy_term',
          'builtin_naf',
          'builtin_write',
          'CutIte'
        ]),
    assert_contains_all(Tests,
        [ 'lua_structural_builtins_e2e',
          'lua_type_and_compare_builtins_e2e',
          'lua_term_inspection_builtins_e2e',
          'lua_univ_builtin_e2e',
          'lua_copy_term_builtin_e2e',
          'lua_control_builtins_e2e',
          'lua_io_builtins_e2e'
        ]),
    assert_contains_all(Audit,
        [ 'member/2',
          'length/2',
          'functor/3',
          'arg/3',
          '=../2',
          'copy_term/2',
          'CutIte',
          'write/1'
        ]).

test(project_layout) :-
    unique_lua_tmp_dir('tmp_lua_layout', TmpDir),
    setup_call_cleanup(
        write_wam_lua_project([user:wam_lua_fact/1], [], TmpDir),
        ( directory_file_path(TmpDir, 'lua/wam_runtime.lua', Runtime),
          directory_file_path(TmpDir, 'lua/generated_program.lua', Program),
          assertion(exists_file(Runtime)),
          assertion(exists_file(Program))
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(instructions_and_labels) :-
    unique_lua_tmp_dir('tmp_lua_instrs', TmpDir),
    setup_call_cleanup(
        write_wam_lua_project([user:wam_lua_fact/1], [], TmpDir),
        ( directory_file_path(TmpDir, 'lua/generated_program.lua', Program),
          read_file_to_string(Program, Code, []),
          assertion(sub_string(Code, _, _, _, 'local intern_seed = {')),
          assertion(sub_string(Code, _, _, _, 'local shared_instructions = {')),
          assertion(sub_string(Code, _, _, _, 'Runtime.resolve_program(shared_program)')),
          assertion(sub_string(Code, _, _, _, 'indexed_atom_fact2 = {}')),
          assertion(sub_string(Code, _, _, _, 'I.GetConstant(V.Atom(')),
          assertion(sub_string(Code, _, _, _, '["wam_lua_fact/1"] = 1'))
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(choice_points_emitted) :-
    unique_lua_tmp_dir('tmp_lua_choice', TmpDir),
    setup_call_cleanup(
        write_wam_lua_project([user:wam_lua_choice/1], [], TmpDir),
        ( directory_file_path(TmpDir, 'lua/generated_program.lua', Program),
          read_file_to_string(Program, Code, []),
          assertion(sub_string(Code, _, _, _, 'I.TryMeElse(')),
          assertion(sub_string(Code, _, _, _, 'I.TrustMe()'))
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(aggregate_and_second_arg_switch_emitted) :-
    unique_lua_tmp_dir('tmp_lua_agg_emit', TmpDir),
    setup_call_cleanup(
        write_wam_lua_project(
            [user:wam_lua_fa_basic/0, user:wam_lua_fa_fact/1,
             user:wam_lua_greet/2],
            [],
            TmpDir),
        ( directory_file_path(TmpDir, 'lua/generated_program.lua', Program),
          read_file_to_string(Program, Code, []),
          assertion(sub_string(Code, _, _, _, 'I.BeginAggregate("collect"')),
          assertion(sub_string(Code, _, _, _, 'I.EndAggregate(')),
          assertion(sub_string(Code, _, _, _, 'I.SwitchOnConstantA2('))
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(call_indexed_atom_fact2_literal) :-
    once(wam_parts_to_lua(["call_indexed_atom_fact2", "edge/2"], [], Lit)),
    assertion(Lit == "I.CallIndexedAtomFact2(\"edge/2\")").

test(fact_stream_emitted) :-
    unique_lua_tmp_dir('tmp_lua_fact_stream_emit', TmpDir),
    setup_call_cleanup(
        write_wam_lua_project([user:wam_lua_stream_fact/2], [], TmpDir),
        ( directory_file_path(TmpDir, 'lua/generated_program.lua', Program),
          read_file_to_string(Program, Code, []),
          assertion(sub_string(Code, _, _, _, 'I.CallFactStream("wam_lua_stream_fact/2", 2)')),
          assertion(sub_string(Code, _, _, _, 'local inline_facts = {')),
          assertion(sub_string(Code, _, _, _, '["wam_lua_stream_fact/2"] = {{V.Atom('))
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(lua_indexed_atom_fact2_e2e, [condition(lua_available)]) :-
    unique_lua_tmp_dir('tmp_lua_indexed_fact2_e2e', TmpDir),
    setup_call_cleanup(
        write_wam_lua_project([user:wam_lua_fact/1], [], TmpDir),
        ( directory_file_path(TmpDir, 'lua', LuaDir),
          atomic_list_concat([
              'local rt=require("wam_runtime"); ',
              'local R=rt.Runtime; local V=rt.V; local I=rt.I; ',
              'local p={instructions={I.CallIndexedAtomFact2("edge/2"),I.Proceed()}, labels={["idx/2"]=1}, indexed_atom_fact2={}, intern_table=R.new_intern_table({"true","fail","[]",".","","[|]"})}; ',
              'R.register_indexed_atom_fact2_pairs(p, "edge/2", {{"a","b"},{"a","c"}}); ',
              'local function atom(s) return V.Atom(R.intern(p.intern_table,s)) end; ',
              'print(R.run_predicate(p,1,{atom("a"),atom("b")})); ',
              'print(R.run_predicate(p,1,{atom("a"),atom("c")})); ',
              'print(R.run_predicate(p,1,{atom("a"),atom("d")}))'
          ], Script),
          run_lua_script(LuaDir, Script, Output),
          normalize_space(string(Trimmed), Output),
          assertion(Trimmed == "true true false")
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(lua_fact_stream_e2e, [condition(lua_available)]) :-
    unique_lua_tmp_dir('tmp_lua_fact_stream_e2e', TmpDir),
    setup_call_cleanup(
        write_wam_lua_project([user:wam_lua_stream_fact/2, user:wam_lua_stream_caller/2], [], TmpDir),
        ( directory_file_path(TmpDir, 'lua', LuaDir),
          run_lua_query(LuaDir, 'wam_lua_stream_fact/2', [a, b], true),
          run_lua_query(LuaDir, 'wam_lua_stream_fact/2', [a, c], true),
          run_lua_query(LuaDir, 'wam_lua_stream_fact/2', [a, d], false),
          run_lua_query(LuaDir, 'wam_lua_stream_fact/2', [x, c], false),
          run_lua_query(LuaDir, 'wam_lua_stream_caller/2', [a, c], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(lua_external_fact_source_e2e, [condition(lua_available)]) :-
    unique_lua_tmp_dir('tmp_lua_ext_fact_e2e', TmpDir),
    setup_call_cleanup(
        ( make_directory_path(TmpDir),
          directory_file_path(TmpDir, 'ext_facts.csv', CsvPath),
          setup_call_cleanup(
              open(CsvPath, write, Stream),
              ( writeln(Stream, 'a,b'),
                writeln(Stream, 'a,c'),
                writeln(Stream, 'x,z')
              ),
              close(Stream)),
          write_wam_lua_project(
              [user:wam_lua_ext_fact/2, user:wam_lua_ext_caller/2],
              [lua_fact_sources([source(wam_lua_ext_fact/2, file(CsvPath))])],
              TmpDir)
        ),
        ( directory_file_path(TmpDir, 'lua/generated_program.lua', Program),
          read_file_to_string(Program, Code, []),
          assertion(sub_string(Code, _, _, _, 'local fact_sources = {')),
          assertion(sub_string(Code, _, _, _, '["wam_lua_ext_fact/2"] = { path = ')),
          directory_file_path(TmpDir, 'lua', LuaDir),
          run_lua_query(LuaDir, 'wam_lua_ext_fact/2', [a, b], true),
          run_lua_query(LuaDir, 'wam_lua_ext_fact/2', [a, c], true),
          run_lua_query(LuaDir, 'wam_lua_ext_fact/2', [a, d], false),
          run_lua_query(LuaDir, 'wam_lua_ext_fact/2', [x, z], true),
          run_lua_query(LuaDir, 'wam_lua_ext_fact/2', [a, z], false),
          run_lua_query(LuaDir, 'wam_lua_ext_fact/2', [x, c], false),
          run_lua_query(LuaDir, 'wam_lua_ext_caller/2', [a, c], true),
          run_lua_script(
              LuaDir,
              'local m=require("generated_program"); local R=m.Runtime; local V=m.V; local function atom(s) return V.Atom(R.intern(m.program.intern_table,s)) end; print(m.wam_lua_ext_fact(atom("a"), atom("b"))); local src=m.program.fact_sources["wam_lua_ext_fact/2"]; local aid=R.intern(m.program.intern_table,"a"); print(#src.cache.rows, #src.cache.arg1_index["a:"..aid])',
              Output),
          normalize_space(string(CacheInfo), Output),
          assertion(CacheInfo == "true 3 2")
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(resolved_dispatch_loaded, [condition(lua_available)]) :-
    unique_lua_tmp_dir('tmp_lua_resolved_dispatch', TmpDir),
    setup_call_cleanup(
        write_wam_lua_project(
            [ user:wam_lua_caller/1,
              user:wam_lua_fact/1,
              user:wam_lua_choice/1,
              user:wam_lua_greet/2,
              user:wam_lua_fa_basic/0,
              user:wam_lua_fa_fact/1
            ],
            [],
            TmpDir),
        ( directory_file_path(TmpDir, 'lua', LuaDir),
          run_lua_script(
              LuaDir,
              'local m=require("generated_program"); for _,i in ipairs(m.program.instructions) do print(i.op) end',
              Output),
          assertion(sub_string(Output, _, _, _, 'CallPc')),
          assertion(sub_string(Output, _, _, _, 'TryMeElsePc')),
          assertion(sub_string(Output, _, _, _, 'SwitchOnConstantA2Pc'))
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(lowered_functions_mode) :-
    unique_lua_tmp_dir('tmp_lua_lowered', TmpDir),
    setup_call_cleanup(
        write_wam_lua_project(
            [user:wam_lua_caller/1, user:wam_lua_fact/1],
            [emit_mode(functions)],
            TmpDir),
        ( directory_file_path(TmpDir, 'lua/generated_program.lua', Program),
          read_file_to_string(Program, Code, []),
          assertion(sub_string(Code, _, _, _, 'local function lowered_wam_lua_fact_1')),
          assertion(sub_string(Code, _, _, _, 'return lowered_wam_lua_fact_1(shared_program, state) == true')),
          assertion(sub_string(Code, _, _, _, 'function M.wam_lua_caller(a1)'))
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(lua_cli_e2e, [condition(lua_available)]) :-
    unique_lua_tmp_dir('tmp_lua_e2e', TmpDir),
    setup_call_cleanup(
        write_wam_lua_project([user:wam_lua_caller/1, user:wam_lua_fact/1], [], TmpDir),
        ( directory_file_path(TmpDir, 'lua', LuaDir),
          run_lua_query(LuaDir, 'wam_lua_caller/1', [a], true),
          run_lua_query(LuaDir, 'wam_lua_caller/1', [b], false)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(lua_choice_e2e, [condition(lua_available)]) :-
    unique_lua_tmp_dir('tmp_lua_choice_e2e', TmpDir),
    setup_call_cleanup(
        write_wam_lua_project([user:wam_lua_choice/1], [], TmpDir),
        ( directory_file_path(TmpDir, 'lua', LuaDir),
          run_lua_query(LuaDir, 'wam_lua_choice/1', [a], true),
          run_lua_query(LuaDir, 'wam_lua_choice/1', [b], true),
          run_lua_query(LuaDir, 'wam_lua_choice/1', [c], false)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(lua_findall_e2e, [condition(lua_available)]) :-
    unique_lua_tmp_dir('tmp_lua_findall_e2e', TmpDir),
    setup_call_cleanup(
        write_wam_lua_project(
            [ user:wam_lua_fa_basic/0,
              user:wam_lua_fa_fact/1,
              user:wam_lua_fa_greet/0,
              user:wam_lua_greet/2
            ],
            [],
            TmpDir),
        ( directory_file_path(TmpDir, 'lua', LuaDir),
          run_lua_query(LuaDir, 'wam_lua_fa_basic/0', [], true),
          run_lua_query(LuaDir, 'wam_lua_fa_greet/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(lua_aggregate_all_e2e, [condition(lua_available)]) :-
    unique_lua_tmp_dir('tmp_lua_agg_all_e2e', TmpDir),
    setup_call_cleanup(
        write_wam_lua_project(
            [ user:wam_lua_num/1,
              user:wam_lua_agg_count/0,
              user:wam_lua_agg_sum/0,
              user:wam_lua_agg_min/0,
              user:wam_lua_agg_max/0,
              user:wam_lua_agg_set/0
            ],
            [],
            TmpDir),
        ( directory_file_path(TmpDir, 'lua', LuaDir),
          run_lua_query(LuaDir, 'wam_lua_agg_count/0', [], true),
          run_lua_query(LuaDir, 'wam_lua_agg_sum/0', [], true),
          run_lua_query(LuaDir, 'wam_lua_agg_min/0', [], true),
          run_lua_query(LuaDir, 'wam_lua_agg_max/0', [], true),
          run_lua_query(LuaDir, 'wam_lua_agg_set/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(lua_structural_builtins_e2e, [condition(lua_available)]) :-
    unique_lua_tmp_dir('tmp_lua_structural_builtins_e2e', TmpDir),
    setup_call_cleanup(
        write_wam_lua_project(
            [ user:wam_lua_member_basic/0,
              user:wam_lua_member_backtrack/0,
              user:wam_lua_member_no/0,
              user:wam_lua_length_basic/0,
              user:wam_lua_length_bind/0,
              user:wam_lua_length_no/0
            ],
            [],
            TmpDir),
        ( directory_file_path(TmpDir, 'lua', LuaDir),
          run_lua_query(LuaDir, 'wam_lua_member_basic/0', [], true),
          run_lua_query(LuaDir, 'wam_lua_member_backtrack/0', [], true),
          run_lua_query(LuaDir, 'wam_lua_member_no/0', [], false),
          run_lua_query(LuaDir, 'wam_lua_length_basic/0', [], true),
          run_lua_query(LuaDir, 'wam_lua_length_bind/0', [], true),
          run_lua_query(LuaDir, 'wam_lua_length_no/0', [], false)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(lua_type_and_compare_builtins_e2e, [condition(lua_available)]) :-
    unique_lua_tmp_dir('tmp_lua_type_compare_builtins_e2e', TmpDir),
    setup_call_cleanup(
        write_wam_lua_project(
            [ user:wam_lua_type_builtins/0,
              user:wam_lua_type_var/0,
              user:wam_lua_type_no/0,
              user:wam_lua_compare_builtins/0
            ],
            [],
            TmpDir),
        ( directory_file_path(TmpDir, 'lua', LuaDir),
          run_lua_query(LuaDir, 'wam_lua_type_builtins/0', [], true),
          run_lua_query(LuaDir, 'wam_lua_type_var/0', [], true),
          run_lua_query(LuaDir, 'wam_lua_type_no/0', [], false),
          run_lua_query(LuaDir, 'wam_lua_compare_builtins/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(lua_term_inspection_builtins_e2e, [condition(lua_available)]) :-
    unique_lua_tmp_dir('tmp_lua_term_inspection_builtins_e2e', TmpDir),
    setup_call_cleanup(
        write_wam_lua_project(
            [ user:wam_lua_functor_read/0,
              user:wam_lua_functor_atom/0,
              user:wam_lua_functor_construct/0,
              user:wam_lua_arg_struct/0,
              user:wam_lua_arg_list/0,
              user:wam_lua_arg_no/0
            ],
            [],
            TmpDir),
        ( directory_file_path(TmpDir, 'lua', LuaDir),
          run_lua_query(LuaDir, 'wam_lua_functor_read/0', [], true),
          run_lua_query(LuaDir, 'wam_lua_functor_atom/0', [], true),
          run_lua_query(LuaDir, 'wam_lua_functor_construct/0', [], true),
          run_lua_query(LuaDir, 'wam_lua_arg_struct/0', [], true),
          run_lua_query(LuaDir, 'wam_lua_arg_list/0', [], true),
          run_lua_query(LuaDir, 'wam_lua_arg_no/0', [], false)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(lua_univ_builtin_e2e, [condition(lua_available)]) :-
    unique_lua_tmp_dir('tmp_lua_univ_builtin_e2e', TmpDir),
    setup_call_cleanup(
        write_wam_lua_project(
            [ user:wam_lua_univ_decompose_struct/0,
              user:wam_lua_univ_decompose_atom/0,
              user:wam_lua_univ_decompose_list/0,
              user:wam_lua_univ_compose_struct/0,
              user:wam_lua_univ_compose_atom/0,
              user:wam_lua_univ_no/0
            ],
            [],
            TmpDir),
        ( directory_file_path(TmpDir, 'lua', LuaDir),
          run_lua_query(LuaDir, 'wam_lua_univ_decompose_struct/0', [], true),
          run_lua_query(LuaDir, 'wam_lua_univ_decompose_atom/0', [], true),
          run_lua_query(LuaDir, 'wam_lua_univ_decompose_list/0', [], true),
          run_lua_query(LuaDir, 'wam_lua_univ_compose_struct/0', [], true),
          run_lua_query(LuaDir, 'wam_lua_univ_compose_atom/0', [], true),
          run_lua_query(LuaDir, 'wam_lua_univ_no/0', [], false)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(lua_copy_term_builtin_e2e, [condition(lua_available)]) :-
    unique_lua_tmp_dir('tmp_lua_copy_term_builtin_e2e', TmpDir),
    setup_call_cleanup(
        write_wam_lua_project(
            [ user:wam_lua_copy_term_ground/0,
              user:wam_lua_copy_term_fresh/0,
              user:wam_lua_copy_term_sharing/0,
              user:wam_lua_copy_term_independent_vars/0
            ],
            [],
            TmpDir),
        ( directory_file_path(TmpDir, 'lua', LuaDir),
          run_lua_query(LuaDir, 'wam_lua_copy_term_ground/0', [], true),
          run_lua_query(LuaDir, 'wam_lua_copy_term_fresh/0', [], true),
          run_lua_query(LuaDir, 'wam_lua_copy_term_sharing/0', [], true),
          run_lua_query(LuaDir, 'wam_lua_copy_term_independent_vars/0', [], true)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(lua_control_builtins_e2e, [condition(lua_available)]) :-
    unique_lua_tmp_dir('tmp_lua_control_builtins_e2e', TmpDir),
    setup_call_cleanup(
        write_wam_lua_project(
            [ user:wam_lua_cut_choice/1,
              user:wam_lua_if_then_else_then/0,
              user:wam_lua_if_then_else_else/0,
              user:wam_lua_naf_true/0,
              user:wam_lua_naf_false/0,
              user:wam_lua_naf_member/0,
              user:wam_lua_naf_member_no/0
            ],
            [],
            TmpDir),
        ( directory_file_path(TmpDir, 'lua', LuaDir),
          run_lua_query(LuaDir, 'wam_lua_cut_choice/1', [a], true),
          run_lua_query(LuaDir, 'wam_lua_cut_choice/1', [b], false),
          run_lua_query(LuaDir, 'wam_lua_if_then_else_then/0', [], true),
          run_lua_query(LuaDir, 'wam_lua_if_then_else_else/0', [], true),
          run_lua_query(LuaDir, 'wam_lua_naf_true/0', [], true),
          run_lua_query(LuaDir, 'wam_lua_naf_false/0', [], false),
          run_lua_query(LuaDir, 'wam_lua_naf_member/0', [], true),
          run_lua_query(LuaDir, 'wam_lua_naf_member_no/0', [], false)
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(lua_io_builtins_e2e, [condition(lua_available)]) :-
    unique_lua_tmp_dir('tmp_lua_io_builtins_e2e', TmpDir),
    setup_call_cleanup(
        write_wam_lua_project([user:wam_lua_write_line/0], [], TmpDir),
        ( directory_file_path(TmpDir, 'lua', LuaDir),
          run_lua_script(
              LuaDir,
              'local m=require("generated_program"); assert(m.wam_lua_write_line())',
              WriteOutput),
          assertion(WriteOutput == "hello\n"),
          atomic_list_concat([
              'local rt=require("wam_runtime"); ',
              'local R=rt.Runtime; local V=rt.V; local I=rt.I; ',
              'local p={instructions={}, labels={}, ',
              'intern_table=R.new_intern_table({"true","fail","[]",".","","[|]","f","a"})}; ',
              'local fid=R.intern(p.intern_table,"f"); ',
              'local aid=R.intern(p.intern_table,"a"); ',
              'p.instructions={I.PutStructure(fid,1,1),I.SetConstant(V.Atom(aid)),',
              'I.BuiltinCall("display/1",1),I.BuiltinCall("nl/0",0),I.Proceed()}; ',
              'R.run_predicate(p,1,{})'
          ], DisplayScript),
          run_lua_script(
              LuaDir,
              DisplayScript,
              DisplayOutput),
          assertion(DisplayOutput == "f(a)\n")
        ),
        delete_directory_and_contents(TmpDir)
    ).

test(lua_second_arg_switch_e2e, [condition(lua_available)]) :-
    unique_lua_tmp_dir('tmp_lua_a2_e2e', TmpDir),
    setup_call_cleanup(
        write_wam_lua_project([user:wam_lua_greet/2], [], TmpDir),
        ( directory_file_path(TmpDir, 'lua', LuaDir),
          run_lua_query(LuaDir, 'wam_lua_greet/2', [world, hello], true),
          run_lua_query(LuaDir, 'wam_lua_greet/2', [moon, goodbye], true),
          run_lua_query(LuaDir, 'wam_lua_greet/2', [moon, hello], false)
        ),
        delete_directory_and_contents(TmpDir)
    ).

:- end_tests(wam_lua_generator).

unique_lua_tmp_dir(Prefix, TmpDir) :-
    get_time(T),
    format(atom(TmpDir), 'output/~w_~0f', [Prefix, T]).

lua_available :-
    catch(process_create(path(lua), ['-v'], [stdout(null), stderr(null)]), _, fail).

run_lua_query(LuaDir, PredArity, Args, Expected) :-
    maplist(atom_string, Args, ArgStrings),
    append(['generated_program.lua', PredArity], ArgStrings, LuaArgs),
    process_create(path(lua), LuaArgs,
                   [cwd(LuaDir), stdout(pipe(Out)), stderr(null), process(PID)]),
    read_string(Out, _, Output),
    close(Out),
    process_wait(PID, _Status),
    normalize_space(string(Trimmed), Output),
    (Expected == true -> assertion(Trimmed == "true") ; assertion(Trimmed == "false")).

run_lua_script(LuaDir, Script, Output) :-
    process_create(path(lua), ['-e', Script],
                   [cwd(LuaDir), stdout(pipe(Out)), stderr(null), process(PID)]),
    read_string(Out, _, Output),
    close(Out),
    process_wait(PID, exit(0)).

assert_contains_all(Haystack, Needles) :-
    forall(member(Needle, Needles),
           ( atom_string(Needle, Text),
             assertion(sub_string(Haystack, _, _, _, Text))
           )).
