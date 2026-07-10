:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% dyncall_at@name(Source, args...): a NAMED entry on a RUNTIME source --
% the composition of the two selection axes. The source is runtime data
% (a path expression or a compile() handle); the entry name is a
% compile-time token, resolved per call against the loaded VM's
% materialized entry table (@wam_object_vm_entry_pc) -- no file re-scan,
% so it works uniformly for path loads and eval-compiled handles.

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').
:- use_module('../examples/plawk/codegen/plawk_native_codegen').
:- use_module('../src/unifyweaver/targets/wam_llvm_target').

% a two-entry grammar family (text fields arrive as atoms)
square(X, R) :- atom_number(X, N), R is N * N.
cube(X, R)   :- atom_number(X, N), R is N * N * N.

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(plawk_dyncall_at_named).

% dyncall_at@name(Source, args) parses to its own node, source split
% from args like the bare form.
test(dyncall_at_named_parses) :-
    plawk_parse_string(
        "{ t += dyncall_at@square($2, $1) }\nEND { print t }\n",
        program(_, [rule(_, Actions)], _)),
    memberchk(add(var(t), dyncall_at_named(square, field(2), [field(1)])),
        Actions),
    !.

% The Name-NArgs entry is collected and the emitted IR routes through the
% named-at shim and the VM entry resolver (per call, no startup PC cache).
test(dyncall_at_named_entries_and_ir) :-
    plawk_parse_string(
        "{ t += dyncall_at@square(\"lib.wamo\", $1) }\nEND { print t }\n",
        Program),
    plawk_program_dyncall_at_named_entries(Program, Entries),
    assertion(Entries == [square-1]),
    plawk_program_dyncall_at_arities(Program, PlainArities),
    assertion(PlainArities == []),
    plawk_program_native_driver_ir(Program, stdin_or_argv,
        [wam_vm(10, 10)], IR),
    sub_atom(IR, _, _, _, '@plawk_dyncall_at_named_square_1'),
    sub_atom(IR, _, _, _, '@wam_object_vm_entry_pc'),
    sub_atom(IR, _, _, _, '@plawk_dyncall_at_get'),
    !.

% Full round trip: one multi-entry object on disk, two entries summed by
% name at the call sites. Over inputs 2, 3: squares 4+9=13, cubes
% 8+27=35, total 48.
test(named_entries_on_runtime_source, [condition(clang_available)]) :-
    an_dir(Dir),
    directory_file_path(Dir, 'pair.wamo', Wamo),
    write_wam_object([user:square/2, user:cube/2],
        [wamo_entries([square/2, cube/2])], Wamo),
    format(string(Src),
        "{ total += dyncall_at@square(\"~w\", $1) + dyncall_at@cube(\"~w\", $1) }\n\c
         END { print total }\n", [Wamo, Wamo]),
    build_run(Dir, 'an', Src, "2\n3\n", Out),
    assertion(Out == "48\n"),
    !.

% A name the object does not expose contributes 0 (the shim's
% resolve-miss path), like any failed dyncall.
test(missing_name_yields_zero, [condition(clang_available)]) :-
    an_dir(Dir),
    directory_file_path(Dir, 'pair.wamo', Wamo),
    ( exists_file(Wamo)
    -> true
    ;  write_wam_object([user:square/2, user:cube/2],
           [wamo_entries([square/2, cube/2])], Wamo)
    ),
    format(string(Src),
        "{ total += dyncall_at@nosuch(\"~w\", $1) }\n\c
         END { print total }\n", [Wamo]),
    build_run(Dir, 'anmiss', Src, "2\n3\n", Out),
    assertion(Out == "0\n"),
    !.

% Mode "off" loads fresh per call and frees -- including on the
% resolve-miss path; the named result is the same as cached mode.
test(named_entries_off_mode, [condition(clang_available)]) :-
    an_dir(Dir),
    directory_file_path(Dir, 'pair.wamo', Wamo),
    ( exists_file(Wamo)
    -> true
    ;  write_wam_object([user:square/2, user:cube/2],
           [wamo_entries([square/2, cube/2])], Wamo)
    ),
    format(string(Src),
        "BEGIN { DYNCACHE = \"off\" }\n\c
         { total += dyncall_at@square(\"~w\", $1) + dyncall_at@cube(\"~w\", $1) }\n\c
         END { print total }\n", [Wamo, Wamo]),
    build_run(Dir, 'anoff', Src, "2\n3\n", Out),
    assertion(Out == "48\n"),
    !.

% THE COMPOSITION: a named entry on a compile() HANDLE. The runtime-
% compiled object's entry table comes from the eval pipeline's loaded
% bytes (no file exists to re-scan), and the bootstrap compiler names
% its first predicate ("sq/2" here), so the name resolves against the
% freshly compiled VM. Squares over 3,4,5,10 -> 150.
test(named_entry_on_compile_handle, [condition(clang_available)]) :-
    an_dir(Dir),
    format(string(Src),
        "{ total += dyncall_at@sq(compile(\"[(sq(X2, R2) :- atom_number(X2, N2), R2 is N2 * N2)]\"), $1) }\n\c
         END { print total }\n", []),
    build_run(Dir, 'anh', Src, "3\n4\n5\n10\n", Out),
    assertion(Out == "150\n"),
    !.

:- end_tests(plawk_dyncall_at_named).

% --- helpers ---------------------------------------------------------------

an_dir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_dyncall_at_named', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

build_run(Dir, Name, Src, InputText, Out) :-
    directory_file_path(Dir, Name, Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    atom_concat(Prog0, '_bin', Bin),
    cli([build, Prog, '-o', Bin], _, 0),
    atom_concat(Prog0, '_in.txt', Input),
    setup_call_cleanup(open(Input, write, SI, [encoding(utf8)]),
        write(SI, InputText), close(SI)),
    run_bin(Bin, [Input], Out, 0).

cli(Args, Out, ExpectedStatus) :-
    process_create(path(swipl), ['examples/plawk/bin/plawk' | Args],
        [stdout(pipe(S)), stderr(std), process(Pid)]),
    read_string(S, _, Out),
    close(S),
    process_wait(Pid, exit(Status)),
    assertion(Status == ExpectedStatus).

run_bin(Bin, Args, Out, ExpectedStatus) :-
    process_create(Bin, Args,
        [stdout(pipe(S)), stderr(std), process(Pid)]),
    read_string(S, _, Out),
    close(S),
    process_wait(Pid, exit(Status)),
    assertion(Status == ExpectedStatus).
