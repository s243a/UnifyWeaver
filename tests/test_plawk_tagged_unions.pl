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

:- dynamic user:plawk_tun_marker/0.

user:plawk_tun_marker.

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(plawk_tagged_unions, [condition(clang_available)]).

test(parses_case_blocks) :-
    plawk_parse_string("BEGIN { BINFMT = \"case(i64 f64 | lps16 i64)\" } case 0 { $1 > 100 { c++ } } case 1 { { d++ } } END { print c, d }\n", Program),
    Program = program(_, case_blocks(Blocks), _),
    assertion(Blocks == [case_arm(0, [rule(field_cmp(1, gt, 100), [inc(var(c))])]),
                         case_arm(1, [rule(always, [inc(var(d))])])]).

test(union_ir_has_tag_switch_and_arm_reads) :-
    plawk_parse_string("BEGIN { BINFMT = \"case(i64 f64 | lps16 i64)\" } case 0 { $1 > 100 { c++ } } case 1 { { d++ } } END { print c, d }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.bin', DriverIR),
    % tag read + native switch dispatching per-arm read sequences
    assertion(once(sub_atom(DriverIR, _, _, _, 'switch i64 %vr_tag, label %fail_read [ i64 0, label %vr_a0 i64 1, label %vr_a1 ]'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '%vr_a0_f0_dst'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '%vr_a1_f0_fits'))),
    % buffer = widest arm (lps16 + i64 = 24), tag lives only in %vr_tag
    assertion(once(sub_atom(DriverIR, _, _, _, 'malloc(i64 24)'))),
    % rule guards check the tag before their own pattern
    assertion(once(sub_atom(DriverIR, _, _, _, '_tag_ok = icmp eq i64 %vr_tag, 0'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '_tag_ok = icmp eq i64 %vr_tag, 1'))),
    assertion(\+ sub_atom(DriverIR, _, _, _, '@run_loop')),
    !.

test(union_rejections) :-
    Rejects = [
        % case index beyond the declared arms
        "BEGIN { BINFMT = \"case(i64 | lps8)\" } case 2 { { c++ } } END { print c }\n",
        % arm-typed field misuse: string equality on an i64 field
        "BEGIN { BINFMT = \"case(i64 | lps8)\" } case 0 { $1 == \"x\" { c++ } } END { print c }\n",
        % numeric compare on an lps field
        "BEGIN { BINFMT = \"case(i64 | lps8)\" } case 1 { $1 > 5 { c++ } } END { print c }\n",
        % assoc arrays are not in the union slice
        "BEGIN { BINFMT = \"case(i64 | lps8)\" } case 0 { { counts[$1]++ } } END { print counts[5] }\n",
        % case blocks demand a union BINFMT
        "case 0 { { c++ } } END { print c }\n",
        "BEGIN { BINFMT = \"i64 i64\" } case 0 { { c++ } } END { print c }\n"
    ],
    forall(member(Source, Rejects),
        ( plawk_parse_string(Source, Program)
        -> assertion(\+ plawk_program_native_driver_ir(Program, 'input.bin', _))
        ;  true
        )).

test(surface_union_dispatch_and_state) :-
    % Two record kinds interleaved: metrics (i64 f64) aggregate, events
    % (lps16 i64) print and count -- one native loop, shared END.
    run_tun_smoke("BEGIN { BINFMT = \"case(i64 f64 | lps16 i64)\" } case 0 { $1 > 100 { msum += float($2) ; mhits++ } } case 1 { $2 == 7 { print $1 } $1 == \"boom\" { events++ } } END { print mhits, msum, events }\n",
        [m(50, 1.5), e("hello", 7), m(200, 2.5), e("boom", 3), m(300, 0.25), e("boom", 7)],
        "hello\nboom\n2 2.75 2\n").

test(surface_union_unhandled_arm_still_reads) :-
    % No case 0 block: metric records are read (and skipped) correctly,
    % keeping stream framing intact for the events that follow.
    run_tun_smoke("BEGIN { BINFMT = \"case(i64 f64 | lps16 i64)\" } case 1 { { events++ } } END { print events }\n",
        [m(50, 1.5), e("x", 1), m(200, 2.5), e("y", 2)],
        "2\n").

test(surface_union_next_and_nr) :-
    % NR counts records of every arm; next works inside a case block.
    run_tun_smoke("BEGIN { BINFMT = \"case(i64 f64 | lps16 i64)\" } case 1 { $1 == \"skip\" { next } { events++ } } END { print NR, events }\n",
        [m(1, 1.0), e("skip", 0), e("go", 0), m(2, 2.0)],
        "4 1\n").

test(surface_union_endless_program) :-
    run_tun_smoke("BEGIN { BINFMT = \"case(i64 f64 | lps16 i64)\" } case 1 { { print $1, $2 } }\n",
        [m(1, 1.0), e("aa", 5), e("bb", 6)],
        "aa 5\nbb 6\n").

test(tag_guard_sugar_matches_case_blocks_exactly) :-
    % TAG == K && P is pure sugar: it must compile to byte-identical IR
    % as the case-block spelling.
    plawk_parse_string("BEGIN { BINFMT = \"case(i64 f64 | lps16 i64)\" } TAG == 1 && $2 > 5 { b++ } END { print b }\n", Sugar),
    plawk_parse_string("BEGIN { BINFMT = \"case(i64 f64 | lps16 i64)\" } case 1 { $2 > 5 { b++ } } END { print b }\n", Blocks),
    plawk_program_native_driver_ir(Sugar, 'input.bin', SugarIR),
    plawk_program_native_driver_ir(Blocks, 'input.bin', BlocksIR),
    assertion(SugarIR == BlocksIR),
    !.

test(tag_guard_rejections) :-
    Rejects = [
        % every rule must lead with a tag guard (an unguarded rule has
        % no arm to type its fields against)
        "BEGIN { BINFMT = \"case(i64 | lps8)\" } TAG == 0 { a++ } { b++ } END { print a, b }\n",
        % a tag test under || has no single-arm meaning
        "BEGIN { BINFMT = \"case(i64 | lps8)\" } TAG == 0 || TAG == 1 { a++ } END { print a }\n",
        % ... nor one that is not the leftmost conjunct
        "BEGIN { BINFMT = \"case(i64 | lps8)\" } $1 > 3 && TAG == 0 { a++ } END { print a }\n",
        % tag beyond the declared arms
        "BEGIN { BINFMT = \"case(i64 | lps8)\" } TAG == 5 { a++ } END { print a }\n",
        % TAG guards demand a union BINFMT
        "BEGIN { BINFMT = \"i64 i64\" } TAG == 0 { a++ } END { print a }\n"
    ],
    forall(member(Source, Rejects),
        ( plawk_parse_string(Source, Program)
        -> assertion(\+ plawk_program_native_driver_ir(Program, 'input.bin', _))
        ;  true
        )).

test(surface_tag_guard_dispatch) :-
    % The case-block dispatch test, respelled with tag guards --
    % including rules for different arms interleaved in source order.
    run_tun_smoke("BEGIN { BINFMT = \"case(i64 f64 | lps16 i64)\" } TAG == 0 && $1 > 100 { msum += float($2) ; mhits++ } TAG == 1 && $2 == 7 { print $1 } TAG == 1 && $1 == \"boom\" { events++ } END { print mhits, msum, events }\n",
        [m(50, 1.5), e("hello", 7), m(200, 2.5), e("boom", 3), m(300, 0.25), e("boom", 7)],
        "hello\nboom\n2 2.75 2\n").

test(surface_union_error_paths) :-
    build_tun_probe("BEGIN { BINFMT = \"case(i64 f64 | lps16 i64)\" } case 0 { { c++ } } END { print c }\n",
        Dir, BinPath),
    directory_file_path(Dir, 'input.bin', InputPath),
    % unknown tag
    write_bytes(InputPath, [i64(9), pad(16)]),
    run_status(BinPath, exit(11)),
    % truncated arm 0 (tag + i64, missing the f64)
    write_bytes(InputPath, [i64(0), i64(5)]),
    run_status(BinPath, exit(11)),
    % clean EOF at a record boundary runs END
    write_bytes(InputPath, []),
    run_expect(BinPath, "0\n"),
    !.

% --- helpers ---------------------------------------------------------------

write_i64_le(Out, V) :-
    V64 is V /\ 0xFFFFFFFFFFFFFFFF,
    forall(between(0, 7, I),
        ( Byte is (V64 >> (8 * I)) /\ 0xFF, put_byte(Out, Byte) )).

% dyadic doubles used by these tests
double_bits(1.5,  0x3FF8000000000000).
double_bits(2.5,  0x4004000000000000).
double_bits(0.25, 0x3FD0000000000000).
double_bits(1.0,  0x3FF0000000000000).
double_bits(2.0,  0x4000000000000000).

% m(V, F): arm 0 = tag 0, i64 V, f64 F.  e(S, C): arm 1 = tag 1, lps16 S, i64 C.
write_union_records(Path, Recs) :-
    setup_call_cleanup(
        open(Path, write, Out, [type(binary)]),
        forall(member(Rec, Recs),
            ( Rec = m(V, F)
            -> ( write_i64_le(Out, 0), write_i64_le(Out, V),
                 double_bits(F, Bits), write_i64_le(Out, Bits) )
            ;  Rec = e(S, C),
               string_codes(S, Codes),
               length(Codes, Len),
               write_i64_le(Out, 1), write_i64_le(Out, Len),
               forall(member(Ch, Codes), put_byte(Out, Ch)),
               write_i64_le(Out, C)
            )),
        close(Out)).

write_bytes(Path, Items) :-
    setup_call_cleanup(
        open(Path, write, Out, [type(binary)]),
        forall(member(Item, Items),
            ( Item = i64(V) -> write_i64_le(Out, V)
            ; Item = pad(N) -> forall(between(1, N, _), put_byte(Out, 0))
            )),
        close(Out)).

emit_probe(Dir, DriverIR, BinPath) :-
    directory_file_path(Dir, 'probe.ll', LLPath),
    write_wam_llvm_project(
        [ user:plawk_tun_marker/0 ],
        [module_name('plawk_tagged_unions')], LLPath),
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
    ;  format(user_error, "~n[plawk tagged unions build output]~n~w~n", [BuildOut]),
       throw(plawk_tagged_unions_build_failed(Status))
    ).

build_tun_probe(Source, Dir, BinPath) :-
    tmp_root(Root),
    directory_file_path(Root, 'uw_plawk_tagged_unions', Dir),
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

run_tun_smoke(Source, Recs, ExpectedOutput) :-
    build_tun_probe(Source, Dir, BinPath),
    directory_file_path(Dir, 'input.bin', InputPath),
    write_union_records(InputPath, Recs),
    run_expect(BinPath, ExpectedOutput),
    !.

:- end_tests(plawk_tagged_unions).
