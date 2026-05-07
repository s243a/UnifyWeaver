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

user:wam_lua_fact(a).
user:wam_lua_choice(a).
user:wam_lua_choice(b).
user:wam_lua_caller(X) :- user:wam_lua_fact(X).

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
