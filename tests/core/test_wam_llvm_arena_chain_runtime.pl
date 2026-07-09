:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% Chained-arena runtime tests (M6 self-host, runtime finding no. 8).
% The bump arena links a new block on exhaustion instead of returning
% null; blocks never move, and mark/rewind work across block boundaries
% via virtual offsets. These drivers exercise the arena functions
% directly in a native binary:
%   1. Chain semantics: tiny first block, forced links, monotonic marks,
%      cross-block rewind (pops + frees newer blocks), block stability
%      (sentinels survive growth), first-block reuse after rewind(0),
%      destroy + ensure re-init.
%   2. Growth stress: ~100 MB through a 1 KiB initial block (many
%      doublings), every allocation written and read back.

:- use_module(library(filesex), [make_directory_path/1]).
:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module('../helpers/smoke_paths', [tmp_root/1, clean_dir/1]).
:- use_module('../../src/unifyweaver/targets/wam_llvm_target',
    [write_wam_llvm_project/3]).

:- dynamic user:arena_chain_marker/0.

user:arena_chain_marker.

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(wam_llvm_arena_chain_runtime, [condition(clang_available)]).

test(arena_chain_marks_rewind_and_block_stability) :-
    arena_chain_semantics_driver_ir(DriverIR),
    run_arena_chain_smoke('uw_wam_arena_chain_semantics', DriverIR).

test(arena_chain_growth_stress) :-
    arena_chain_growth_driver_ir(DriverIR),
    run_arena_chain_smoke('uw_wam_arena_chain_growth', DriverIR).

run_arena_chain_smoke(Name, DriverIR) :-
    tmp_root(Root),
    directory_file_path(Root, Name, Dir),
    clean_dir(Dir),
    make_directory_path(Dir),
    directory_file_path(Dir, 'arena_chain_runtime.ll', LLPath),
    write_wam_llvm_project(
        [ user:arena_chain_marker/0 ],
        [ module_name('arena_chain_runtime') ],
        LLPath),
    setup_call_cleanup(
        open(LLPath, append, Out, [encoding(utf8)]),
        ( nl(Out), write(Out, DriverIR) ),
        close(Out)),
    directory_file_path(Dir, 'arena_chain_runtime_bin', BinPath),
    format(atom(Cmd), 'clang -w ~w -o ~w -lm 2>&1 && ~w',
        [LLPath, BinPath, BinPath]),
    process_create(path(sh), ['-c', Cmd],
                   [stdout(pipe(Stdout)), stderr(std), process(Pid)]),
    read_string(Stdout, _, OutStr),
    close(Stdout),
    process_wait(Pid, Status),
    ( Status == exit(0)
    -> assertion(OutStr == "")
    ;  format(user_error, "~n[arena chain runtime output]~n~w~n", [OutStr]),
       throw(arena_chain_runtime_failed(Status))
    ),
    !.

% Exit codes: 90 alloc returned null, 91-93 wrong mark after allocs,
% 94 first-block sentinel lost (block moved or was clobbered by growth),
% 95 second-block sentinel lost, 96/97 wrong mark after cross-block
% rewind, 98 reset-to-zero mark wrong, 99 first block not reused in
% place after rewind(0), 100 wrong mark after destroy + ensure re-init.
arena_chain_semantics_driver_ir('
define i32 @main() {
entry:
  ; Tiny first block: cap 64, so the second allocation must chain.
  call void @wam_arena_init(i64 64)
  %p1 = call i8* @wam_arena_alloc(i64 48)
  %p1_null = icmp eq i8* %p1, null
  br i1 %p1_null, label %alloc_fail, label %fill1

fill1:
  %p1_i64 = bitcast i8* %p1 to i64*
  store i64 111, i64* %p1_i64
  %p1_tail_raw = getelementptr i8, i8* %p1, i64 40
  %p1_tail = bitcast i8* %p1_tail_raw to i64*
  store i64 222, i64* %p1_tail
  %m1 = call i64 @wam_arena_mark()
  %m1_ok = icmp eq i64 %m1, 48
  br i1 %m1_ok, label %chain1, label %bad_m1

chain1:
  ; 48 + 64 > 64: links a second block (cap max(128, 64) = 128, base 48).
  %p2 = call i8* @wam_arena_alloc(i64 64)
  %p2_null = icmp eq i8* %p2, null
  br i1 %p2_null, label %alloc_fail, label %fill2

fill2:
  %p2_tail_raw = getelementptr i8, i8* %p2, i64 56
  %p2_tail = bitcast i8* %p2_tail_raw to i64*
  store i64 333, i64* %p2_tail
  %m2 = call i64 @wam_arena_mark()
  %m2_ok = icmp eq i64 %m2, 112
  br i1 %m2_ok, label %chain2, label %bad_m2

chain2:
  ; 64 + 200 > 128: links a third block (cap max(256, 200) = 256, base 112).
  %p3 = call i8* @wam_arena_alloc(i64 200)
  %p3_null = icmp eq i8* %p3, null
  br i1 %p3_null, label %alloc_fail, label %fill3

fill3:
  %p3_tail_raw = getelementptr i8, i8* %p3, i64 192
  %p3_tail = bitcast i8* %p3_tail_raw to i64*
  store i64 444, i64* %p3_tail
  %m3 = call i64 @wam_arena_mark()
  %m3_ok = icmp eq i64 %m3, 312
  br i1 %m3_ok, label %stability, label %bad_m3

stability:
  ; Earlier blocks must be untouched by growth (blocks never move).
  %r1 = load i64, i64* %p1_i64
  %r1_ok = icmp eq i64 %r1, 111
  %r1t = load i64, i64* %p1_tail
  %r1t_ok = icmp eq i64 %r1t, 222
  %b1_ok = and i1 %r1_ok, %r1t_ok
  br i1 %b1_ok, label %stability2, label %bad_block1

stability2:
  %r2 = load i64, i64* %p2_tail
  %r2_ok = icmp eq i64 %r2, 333
  br i1 %r2_ok, label %rewind_mid, label %bad_block2

rewind_mid:
  ; Rewind to m2 (virtual 112): discards the 200-byte allocation.
  call void @wam_arena_rewind(i64 112)
  %mm2 = call i64 @wam_arena_mark()
  %mm2_ok = icmp eq i64 %mm2, 112
  br i1 %mm2_ok, label %rewind_far, label %bad_rw_mid

rewind_far:
  ; Rewind to m1 (virtual 48): pops across a block boundary.
  call void @wam_arena_rewind(i64 48)
  %mm1 = call i64 @wam_arena_mark()
  %mm1_ok = icmp eq i64 %mm1, 48
  br i1 %mm1_ok, label %check_survivor, label %bad_rw_far

check_survivor:
  ; The first block (below the mark) must still hold its data.
  %r1b = load i64, i64* %p1_i64
  %r1b_ok = icmp eq i64 %r1b, 111
  br i1 %r1b_ok, label %reset_zero, label %bad_block1

reset_zero:
  call void @wam_arena_reset()
  %mz = call i64 @wam_arena_mark()
  %mz_ok = icmp eq i64 %mz, 0
  br i1 %mz_ok, label %reuse, label %bad_reset

reuse:
  ; After reset the first block is reused in place: same data pointer.
  %p5 = call i8* @wam_arena_alloc(i64 48)
  %p5_null = icmp eq i8* %p5, null
  br i1 %p5_null, label %alloc_fail, label %reuse_check

reuse_check:
  %same = icmp eq i8* %p5, %p1
  br i1 %same, label %reinit, label %bad_reuse

reinit:
  ; Destroy releases the whole chain; ensure re-initializes lazily.
  call void @wam_arena_destroy()
  call void @wam_arena_ensure()
  %p6 = call i8* @wam_arena_alloc(i64 16)
  %p6_null = icmp eq i8* %p6, null
  br i1 %p6_null, label %alloc_fail, label %reinit_check

reinit_check:
  %m6 = call i64 @wam_arena_mark()
  %m6_ok = icmp eq i64 %m6, 16
  call void @wam_arena_destroy()
  br i1 %m6_ok, label %ok, label %bad_reinit

ok:
  ret i32 0
alloc_fail:
  ret i32 90
bad_m1:
  ret i32 91
bad_m2:
  ret i32 92
bad_m3:
  ret i32 93
bad_block1:
  ret i32 94
bad_block2:
  ret i32 95
bad_rw_mid:
  ret i32 96
bad_rw_far:
  ret i32 97
bad_reset:
  ret i32 98
bad_reuse:
  ret i32 99
bad_reinit:
  ret i32 100
}
').

% Exit codes: 90 alloc returned null, 91 read-back mismatch (a growth
% block clobbered an earlier one), 92 reset did not return to zero.
arena_chain_growth_driver_ir('
define i32 @main() {
entry:
  ; 1 KiB initial block; 100000 x 1000 bytes (~100 MB) forces many
  ; doublings. Every allocation is stamped with its index and re-read
  ; immediately (a fresh block clobbering live data would surface as a
  ; mismatch on a later iteration via the mark checksum below).
  call void @wam_arena_init(i64 1024)
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i_next, %step ]
  %done = icmp uge i64 %i, 100000
  br i1 %done, label %check_total, label %body

body:
  %p = call i8* @wam_arena_alloc(i64 1000)
  %p_null = icmp eq i8* %p, null
  br i1 %p_null, label %alloc_fail, label %stamp

stamp:
  %slot = bitcast i8* %p to i64*
  store i64 %i, i64* %slot
  %back = load i64, i64* %slot
  %back_ok = icmp eq i64 %back, %i
  br i1 %back_ok, label %step, label %bad_readback

step:
  %i_next = add i64 %i, 1
  br label %loop

check_total:
  ; 1000 aligns to 1000 (already a multiple of 8): total = 100000000.
  %m = call i64 @wam_arena_mark()
  %m_ok = icmp eq i64 %m, 100000000
  br i1 %m_ok, label %reset, label %bad_readback

reset:
  call void @wam_arena_reset()
  %mz = call i64 @wam_arena_mark()
  %mz_ok = icmp eq i64 %mz, 0
  call void @wam_arena_destroy()
  br i1 %mz_ok, label %ok, label %bad_reset

ok:
  ret i32 0
alloc_fail:
  ret i32 90
bad_readback:
  ret i32 91
bad_reset:
  ret i32 92
}
').

:- end_tests(wam_llvm_arena_chain_runtime).

:- initialization(run_tests, main).
