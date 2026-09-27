:- use_module('../src/unifyweaver/targets/wam_cpp_lowered_emitter').
:- use_module('../src/unifyweaver/targets/wam_cpp_target').
:- use_module(library(process)).
:- use_module(library(filesex)).
:- begin_tests(wam_cpp_lowered_head_unify).

test(lowered_emit_head_unification) :-
    with_output_to(string(S1), wam_cpp_lowered_emitter:emit_one(get_integer("42","A1"), "")),
    assertion(sub_string(S1, _, _, _, "Value::Integer(42)")),
    with_output_to(string(S2), wam_cpp_lowered_emitter:emit_one(get_nil("A1"), "")),
    assertion(sub_string(S2, _, _, _, "match_reg_atom(\"A1\", \"[]\")")),
    with_output_to(string(S3), wam_cpp_lowered_emitter:emit_one(get_constant("foo","A1"), "")),
    assertion(sub_string(S3, _, _, _, "Value::Atom(\"foo\")")).

gpp_available :-
    catch((process_create(path('g++'), ['--version'],
                          [stdout(null), stderr(null), process(Pid)]),
           process_wait(Pid, exit(0))), _, fail).

test(head_match_family_native, [condition(gpp_available)]) :-
    tmp_file(cpp_head_match, Dir),
    setup_call_cleanup(
        make_directory(Dir),
        run_head_match_native(Dir),
        delete_directory_and_contents(Dir)).

run_head_match_native(Dir) :-
    wam_cpp_target:compile_wam_runtime_header_to_cpp([], Header),
    wam_cpp_target:compile_wam_runtime_to_cpp([], Runtime),
    directory_file_path(Dir, 'wam_runtime.h', HeaderPath),
    directory_file_path(Dir, 'wam_runtime.cpp', RuntimePath),
    directory_file_path(Dir, 'head_match.cpp', TestPath),
    directory_file_path(Dir, 'head_match', ExePath),
    write_text(HeaderPath, Header),
    write_text(RuntimePath, Runtime),
    findall(Code,
        (member(Name-Instrs,
            [head_integer-[get_integer("42", "A1"), proceed],
             head_nil-[get_nil("A1"), proceed],
             head_atom-[get_constant("foo", "A1"), proceed],
             head_float-[get_constant("3.5", "A1"), proceed],
             head_pair-[get_integer("42", "A1"), get_nil("A2"), proceed]]),
         lower_predicate_to_cpp(Name/1, Instrs, [], Lines),
         atomic_list_concat(Lines, '\n', Code)),
        Codes),
    atomic_list_concat(Codes, '\n', Functions),
    head_match_harness(Harness),
    format(string(Source), '#include "wam_runtime.h"~n~s~n~s',
           [Functions, Harness]),
    write_text(TestPath, Source),
    process_create(path('g++'), ['-std=c++17', '-O0', RuntimePath, TestPath,
                                  '-o', ExePath],
                   [stdout(pipe(BuildOut)), stderr(pipe(BuildErr)), process(BuildPid)]),
    read_string(BuildOut, _, BuildStdout), close(BuildOut),
    read_string(BuildErr, _, BuildStderr), close(BuildErr),
    process_wait(BuildPid, BuildStatus),
    ( BuildStatus == exit(0) -> true
    ; format(user_error, '~s~s', [BuildStdout, BuildStderr]),
      throw(error(head_match_build_failed(BuildStatus), _)) ),
    process_create(ExePath, [],
                   [stdout(pipe(RunOut)), stderr(pipe(RunErr)), process(RunPid)]),
    read_string(RunOut, _, RunStdout), close(RunOut),
    read_string(RunErr, _, RunStderr), close(RunErr),
    process_wait(RunPid, RunStatus),
    ( RunStatus == exit(0), RunStdout == "ALL HEAD MATCH PASS\n" -> true
    ; format(user_error, '~s~s', [RunStdout, RunStderr]),
      throw(error(head_match_exec_failed(RunStatus), _)) ).

write_text(Path, Text) :-
    setup_call_cleanup(open(Path, write, Stream),
        format(Stream, '~s', [Text]), close(Stream)).

head_match_harness(
"#include <iostream>
static int failures = 0;
static void chk(const char* name, bool got) {
    if (!got) { std::cerr << \"FAIL \" << name << '\\n'; ++failures; }
}
int main() {
    { WamState v; v.set_cell(\"A2\", v.get_cell(\"A1\"));
      auto mark = v.trail.size();
      chk(\"integer bind\", lowered_head_integer_1(&v));
      chk(\"integer alias\", v.get_reg(\"A2\") == Value::Integer(42));
      chk(\"integer trailed\", v.trail.size() > mark);
      v.unwind_trail_to(mark);
      chk(\"integer rollback\", v.get_reg(\"A1\").is_unbound() && v.get_reg(\"A2\").is_unbound()); }
    { WamState v; v.put_reg(\"A1\", Value::Integer(42)); chk(\"integer equal\", lowered_head_integer_1(&v)); }
    { WamState v; v.put_reg(\"A1\", Value::Integer(7)); chk(\"integer mismatch\", !lowered_head_integer_1(&v)); }
    { WamState v; v.set_cell(\"A2\", v.get_cell(\"A1\"));
      auto mark = v.trail.size();
      chk(\"nil bind\", lowered_head_nil_1(&v));
      chk(\"nil alias\", v.get_reg(\"A2\") == Value::Atom(\"[]\"));
      chk(\"nil trailed\", v.trail.size() > mark);
      v.unwind_trail_to(mark);
      chk(\"nil rollback\", v.get_reg(\"A1\").is_unbound() && v.get_reg(\"A2\").is_unbound()); }
    { WamState v; v.put_reg(\"A1\", Value::Atom(\"[]\")); chk(\"nil equal\", lowered_head_nil_1(&v)); }
    { WamState v; v.put_reg(\"A1\", Value::Atom(\"foo\")); chk(\"nil mismatch\", !lowered_head_nil_1(&v)); }
    { WamState v; v.put_reg(\"A1\", Value::Atom(\"foo\")); chk(\"atom equal\", lowered_head_atom_1(&v)); }
    { WamState v; v.put_reg(\"A1\", Value::Atom(\"bar\")); chk(\"atom mismatch\", !lowered_head_atom_1(&v)); }
    { WamState v; v.put_reg(\"A1\", Value::Float(3.5)); chk(\"float equal\", lowered_head_float_1(&v)); }
    { WamState v; v.put_reg(\"A1\", Value::Float(4.5)); chk(\"float mismatch\", !lowered_head_float_1(&v)); }
    // trail_binding only records an existing register cell; model a caller's A1.
    { WamState v; v.get_cell(\"A1\"); v.put_reg(\"A2\", Value::Atom(\"other\"));
      auto mark = v.trail.size();
      chk(\"pair fails\", !lowered_head_pair_1(&v));
      chk(\"pair first bound\", v.get_reg(\"A1\") == Value::Integer(42));
      v.unwind_trail_to(mark);
      chk(\"pair rollback\", v.get_reg(\"A1\").is_unbound()); }
    if (failures) return 1;
    std::cout << \"ALL HEAD MATCH PASS\\n\";
    return 0;
}
").

:- end_tests(wam_cpp_lowered_head_unify).
