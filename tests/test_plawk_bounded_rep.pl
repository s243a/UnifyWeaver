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

:- dynamic user:plawk_brep_marker/0.

user:plawk_brep_marker.

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(plawk_bounded_rep, [condition(clang_available)]).

test(parses_foreach_action) :-
    plawk_parse_string("BEGIN { BINFMT = \"i64 rep4(i64 f64)\" } { foreach { n++ } } END { print n }\n", Program),
    Program = program(_, [rule(always, [foreach(Body)])], _),
    assertion(Body == [inc(var(n))]).

test(rep_ir_reads_count_then_bulk_elements) :-
    plawk_parse_string("BEGIN { BINFMT = \"i64 rep4(i64 f64)\" } { foreach { n++ } } END { print n }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.bin', DriverIR),
    % buffer: 8 (id) + 8 (count) + 4*16 (elems) + 16 (staging) = 96
    assertion(once(sub_atom(DriverIR, _, _, _, 'malloc(i64 96)'))),
    % count read into its record slot, bounded by the cap
    assertion(once(sub_atom(DriverIR, _, _, _, '%vr_f1_fits = icmp ule i64 %vr_f1_n, 4'))),
    % one memset of the element region + one bulk read of count*16 bytes
    assertion(once(sub_atom(DriverIR, _, _, _, 'i8 0, i64 64, i1 false'))),
    assertion(once(sub_atom(DriverIR, _, _, _, 'mul i64 %vr_f1_n, 16'))),
    % foreach is a RUNTIME loop: index phi, per-iteration staging
    % memcpy, back edge -- one body copy regardless of cap
    assertion(once(sub_atom(DriverIR, _, _, _, '_j = phi i64 [1, %'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '_cont = icmp sle i64 '))),
    assertion(once(sub_atom(DriverIR, _, _, _, '_dst, i8* %rule_0_body_fe_0_src, i64 16'))),
    assertion(\+ sub_atom(DriverIR, _, _, _, 'icmp sge i64 %')),
    assertion(\+ sub_atom(DriverIR, _, _, _, '@run_loop')),
    !.

test(rep_rejections) :-
    Rejects = [
        % foreach demands a rep layout
        "BEGIN { BINFMT = \"i64 i64\" } { foreach { n++ } } END { print n }\n",
        % ... and binary mode
        "{ foreach { n++ } } END { print n }\n",
        % no nesting
        "BEGIN { BINFMT = \"i64 rep4(i64)\" } { foreach { foreach { n++ } } } END { print n }\n",
        % $2 does not exist in a 1-field element
        "BEGIN { BINFMT = \"i64 rep4(i64)\" } { foreach { s += $2 } } END { print s }\n",
        % one rep per layout in this slice
        "BEGIN { BINFMT = \"rep2(i64) rep2(i64)\" } { foreach { n++ } } END { print n }\n",
        % elements must be fixed-width
        "BEGIN { BINFMT = \"rep2(lps8)\" } { foreach { n++ } } END { print n }\n"
    ],
    forall(member(Source, Rejects),
        ( plawk_parse_string(Source, Program)
        -> assertion(\+ plawk_program_native_driver_ir(Program, 'input.bin', _))
        ;  true
        )).

test(foreach_code_size_is_cap_independent) :-
    % The scaling property: rep64 emits the same single loop body as
    % rep4 -- one increment site, not 64.
    plawk_parse_string("BEGIN { BINFMT = \"i64 rep64(i64 f64)\" } { foreach { n++ } } END { print n }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.bin', DriverIR),
    findall(B, sub_atom(DriverIR, B, _, _, '_slot_0_op_0 = add i64 '), IncSites),
    assertion(IncSites = [_]),
    !.

test(surface_foreach_aggregation) :-
    % 4 records with 2, 0, 4, and 1 elements; the guarded record ($1>0)
    % set has 6 elements, weight sum 6.5, three with value > 10.
    run_brep_smoke("BEGIN { BINFMT = \"i64 rep4(i64 f64)\" } $1 > 0 { foreach { n++ ; wsum += float($2) ; if ($1 > 10) { big++ } } } END { print n, wsum, big }\n",
        [rec(1, [e(5, 1.5), e(20, 2.5)]),
         rec(2, []),
         rec(3, [e(30, 0.25), e(4, 0.5), e(11, 1.0), e(2, 0.75)]),
         rec(-1, [e(99, 9.0)])],
        "6 6.5 3\n").

test(surface_count_field_and_direct_access) :-
    % The count is an ordinary i64 field ($2); element slots are
    % directly addressable as flat fields ($3 = elem 1's first field),
    % zero-filled beyond the count.
    run_brep_smoke("BEGIN { BINFMT = \"i64 rep2(i64 f64)\" } { csum += $2 ; s += $3 } END { print csum, s }\n",
        [rec(1, [e(10, 1.0), e(20, 2.0)]),
         rec(2, []),
         rec(3, [e(7, 0.5)])],
        "3 17\n").

test(surface_foreach_empty_records_only) :-
    run_brep_smoke("BEGIN { BINFMT = \"i64 rep4(i64 f64)\" } { foreach { n++ } } END { print n }\n",
        [rec(1, []), rec(2, [])],
        "0\n").

test(surface_rep_error_paths) :-
    build_brep_probe("BEGIN { BINFMT = \"i64 rep4(i64 f64)\" } { foreach { n++ } } END { print n }\n",
        Dir, BinPath),
    directory_file_path(Dir, 'input.bin', InputPath),
    % count exceeds the cap
    write_raw(InputPath, [i64(1), i64(5), pad(80)]),
    run_status(BinPath, exit(11)),
    % truncated element region (count 2, one element present)
    write_raw(InputPath, [i64(1), i64(2), i64(7), f64bits(0x3FF0000000000000)]),
    run_status(BinPath, exit(11)),
    % clean EOF runs END
    write_raw(InputPath, []),
    run_expect(BinPath, "0\n"),
    !.

% --- helpers ---------------------------------------------------------------

write_i64_le(Out, V) :-
    V64 is V /\ 0xFFFFFFFFFFFFFFFF,
    forall(between(0, 7, I),
        ( Byte is (V64 >> (8 * I)) /\ 0xFF, put_byte(Out, Byte) )).

double_bits(1.5,  0x3FF8000000000000).
double_bits(2.5,  0x4004000000000000).
double_bits(0.25, 0x3FD0000000000000).
double_bits(0.5,  0x3FE0000000000000).
double_bits(1.0,  0x3FF0000000000000).
double_bits(0.75, 0x3FE8000000000000).
double_bits(9.0,  0x4022000000000000).
double_bits(2.0,  0x4000000000000000).

% rec(Id, Elems) with e(V, W): i64 id, i64 count, count x (i64, f64)
write_rep_records(Path, Recs) :-
    setup_call_cleanup(
        open(Path, write, Out, [type(binary)]),
        forall(member(rec(Id, Elems), Recs),
            ( write_i64_le(Out, Id),
              length(Elems, Count),
              write_i64_le(Out, Count),
              forall(member(e(V, W), Elems),
                  ( write_i64_le(Out, V),
                    double_bits(W, Bits),
                    write_i64_le(Out, Bits) )) )),
        close(Out)).

write_raw(Path, Items) :-
    setup_call_cleanup(
        open(Path, write, Out, [type(binary)]),
        forall(member(Item, Items),
            ( Item = i64(V) -> write_i64_le(Out, V)
            ; Item = f64bits(B) -> write_i64_le(Out, B)
            ; Item = pad(N) -> forall(between(1, N, _), put_byte(Out, 0))
            )),
        close(Out)).

emit_probe(Dir, DriverIR, BinPath) :-
    directory_file_path(Dir, 'probe.ll', LLPath),
    write_wam_llvm_project(
        [ user:plawk_brep_marker/0 ],
        [module_name('plawk_bounded_rep')], LLPath),
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
    ;  format(user_error, "~n[plawk bounded rep build output]~n~w~n", [BuildOut]),
       throw(plawk_bounded_rep_build_failed(Status))
    ).

build_brep_probe(Source, Dir, BinPath) :-
    tmp_root(Root),
    directory_file_path(Root, 'uw_plawk_bounded_rep', Dir),
    clean_dir(Dir),
    make_directory_path(Dir),
    directory_file_path(Dir, 'input.bin', InputPath),
    plawk_parse_string(Source, Program),
    plawk_program_native_driver_ir(Program, InputPath, DriverIR),
    emit_probe(Dir, DriverIR, BinPath).

run_status(BinPath, ExpectedStatus) :-
    process_create(BinPath, [],
                   [stdout(null), stderr(std), process(Pid)]),
    process_wait(Pid, Status),
    assertion(Status == ExpectedStatus).

run_expect(BinPath, ExpectedOutput) :-
    process_create(BinPath, [],
                   [stdout(pipe(Stdout)), stderr(std), process(Pid)]),
    read_string(Stdout, _, OutStr),
    close(Stdout),
    process_wait(Pid, Status),
    assertion(Status == exit(0)),
    assertion(OutStr == ExpectedOutput).

run_brep_smoke(Source, Recs, ExpectedOutput) :-
    build_brep_probe(Source, Dir, BinPath),
    directory_file_path(Dir, 'input.bin', InputPath),
    write_rep_records(InputPath, Recs),
    run_expect(BinPath, ExpectedOutput),
    !.

:- end_tests(plawk_bounded_rep).
