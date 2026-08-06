:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
%
% test_wam_go_parser_smoke.pl — WAM-Go compiled runtime parser, end to end
%
% Closes the PARSE-GO card in docs/WAM_FLEET_GAP_TASKS.md. Structured
% after tests/core/test_wam_fsharp_parser_smoke.pl: compile the portable
% `prolog_term_parser` to Go through the WAM pipeline, build it, then
% drive `read_term_from_atom/2` from a Go driver over a battery of
% inputs.
%
% The capability module only advertises compiled-parser support for
% targets with compile *and runtime* proof, so this file is that proof
% for `wam_go`. It is also the regression net for the WAM-Go runtime
% fixes the bring-up required — nested write contexts, get_structure
% dereferencing, get_list cons destructuring, A-registers above A8, and
% atom-intern initialisation order — each of which affects ordinary
% programs, not just the parser.
%
% Build/run is skipped (with a message) when `go` isn't on PATH; the
% codegen assertions always run.
%
% Usage: swipl -q -g run_tests -t halt tests/test_wam_go_parser_smoke.pl

:- use_module(library(plunit)).
:- use_module(library(filesex)).
:- use_module(library(process)).
:- use_module('../src/unifyweaver/targets/wam_go_target').
:- use_module('../src/unifyweaver/targets/wam_runtime_parser_capability').

:- begin_tests(wam_go_parser_smoke).

% =====================================================================
% Capability registration
% =====================================================================

test(go_defaults_to_no_parser) :-
    wam_target_runtime_parser(wam_go, [], none),
    wam_target_runtime_parser(go, [], none).

test(go_can_opt_into_compiled_parser) :-
    wam_target_runtime_parser(wam_go, [runtime_parser(compiled)],
                              compiled(prolog_term_parser)),
    wam_target_runtime_parser(go, [runtime_parser(compiled)],
                              compiled(prolog_term_parser)).

test(go_off_disables_parser) :-
    wam_target_runtime_parser(wam_go, [runtime_parser(off)], none).

% Go has no hand-written runtime parser — only the compiled portable one.
test(go_native_request_errors,
     [error(domain_error(runtime_parser_mode(wam_go), native))]) :-
    wam_target_runtime_parser(wam_go, [runtime_parser(native)], _).

% =====================================================================
% Project generation + execution
% =====================================================================

:- dynamic user:gops_parse/2.

test(go_compiled_parser_end_to_end) :-
    get_time(T),
    format(atom(TmpDir), 'tmp_wam_go_parser_~w', [T]),
    setup_call_cleanup(
        assertz((user:gops_parse(Text, Term) :- read_term_from_atom(Text, Term))),
        gops_run(TmpDir),
        ( retractall(user:gops_parse(_, _)),
          ( exists_directory(TmpDir) -> delete_directory_and_contents(TmpDir) ; true ) )
    ).

gops_run(TmpDir) :-
    % runtime_parser(compiled) pulls the portable parser and the
    % target-agnostic wrappers into the project automatically.
    write_wam_go_project([user:gops_parse/2],
                         [module_name(gops_test), prefer_wam(true),
                          runtime_parser(compiled)],
                         TmpDir),

    directory_file_path(TmpDir, 'lib.go', LibPath),
    read_file_to_string(LibPath, LibCode, []),
    % The parser predicates are compiled in alongside the user's.
    assertion(sub_string(LibCode, _, _, _, 'func Tokenize(')),
    assertion(sub_string(LibCode, _, _, _, 'func Parse_expr(')),
    assertion(sub_string(LibCode, _, _, _, 'func Canonical_op_table(')),
    % read_term_from_atom/2 and /3 share a name at two arities; Go has no
    % overloading, so they must carry arity suffixes or the project will
    % not compile ("redeclared in this block").
    assertion(sub_string(LibCode, _, _, _, 'func Read_term_from_atom2(')),
    assertion(sub_string(LibCode, _, _, _, 'func Read_term_from_atom3(')),

    gops_drive(TmpDir).

gops_drive(TmpDir) :-
    (   catch(process_create(path(go), ['version'],
                             [stdout(null), stderr(null)]), _, fail)
    ->  directory_file_path(TmpDir, 'cmd', CmdDir),
        directory_file_path(CmdDir, 'gops', DriverDir),
        make_directory_path(DriverDir),
        directory_file_path(DriverDir, 'main.go', MainPath),
        gops_driver_source(Source),
        gops_write_file(MainPath, Source),
        format(string(RunCmd), "cd ~w && go run ./cmd/gops 2>&1", [TmpDir]),
        process_create(path(sh), ['-c', RunCmd],
                       [stdout(pipe(Out)), process(Pid)]),
        read_string(Out, _, Output),
        process_wait(Pid, Exit),
        format('~nWAM-Go parser driver output:~n~s~n', [Output]),
        assertion(Exit == exit(0)),
        gops_check(Output)
    ;   format('~n[skip] go not on PATH — skipping WAM-Go parser execution check~n')
    ).

%% gops_check(+Output)
%  Each line is "<input> => <rendered term>". The renderer writes
%  compounds in canonical functor form, so `1+2` reads back as
%  `+(1, 2)`.
gops_check(Output) :-
    forall(member(Expected,
                  [ % atoms, numbers, quoted atoms
                    "foo => foo",
                    "42 => 42",
                    "'a b' => a b",
                    % functor application and nesting
                    "foo(a,b) => foo(a, b)",
                    "f(g(h(1))) => f(g(h(1)))",
                    % operator precedence: * binds tighter than -
                    "a-b*c => -(a, *(b, c))",
                    % ... and than +, on the left this time
                    "2*3+4 => +(*(2, 3), 4)",
                    "1+2 => +(1, 2)",
                    % yfx left-associativity
                    "1-2-3 => -(-(1, 2), 3)",
                    % xfy right-associativity
                    "a^b^c => ^(a, ^(b, c))",
                    % lists, empty list, parenthesised term
                    "[1,2,3] => [1, 2, 3]",
                    "[] => []",
                    "(a) => a",
                    % xfx clause operator and prefix operator
                    "a:-b => :-(a, b)",
                    "a=b => =(a, b)",
                    "X is 1+2 => is(",
                    % prefix op applied to an atom
                    "\\+ x => \\+(x)"
                  ]),
           assertion(sub_string(Output, _, _, _, Expected))),
    % Every input parsed; no FAIL lines.
    assertion(\+ sub_string(Output, _, _, _, "FAIL")).

gops_driver_source(
'package main

import (
	"fmt"
	wam "gops_test"
)

// parse drives the user predicate gops_parse/2, which is just
// read_term_from_atom/2 — so this exercises the wrapper, the portable
// parser, and the WAM runtime together.
func parse(src string) {
	vm := wam.NewWamState(wam.Gops_parseCode, wam.Gops_parseLabels)
	vm.PC = wam.Gops_parseStartPC
	out := &wam.Unbound{Name: "T", Idx: 0}
	vm.Regs[0] = wam.InternAtom(src)
	vm.Regs[1] = out
	if vm.Run() {
		fmt.Printf("%s => %s\\n", src, vm.WriteTerm(vm.Deref(out)))
	} else {
		fmt.Printf("%s => FAIL\\n", src)
	}
}

func main() {
	for _, s := range []string{
		"foo", "42", "-7", "''a b''", "X", "_",
		"foo(a,b)", "p(X,Y)", "f(g(h(1)))",
		"1+2", "a-b*c", "2*3+4", "1-2-3", "a^b^c",
		"[1,2,3]", "[]", "[H|T]", "(a)",
		"a:-b", "a=b", "X is 1+2", "\\\\+ x",
	} {
		parse(s)
	}
}
').

gops_write_file(Path, Content) :-
    setup_call_cleanup(
        open(Path, write, Stream),
        format(Stream, "~w", [Content]),
        close(Stream)
    ).

:- end_tests(wam_go_parser_smoke).
