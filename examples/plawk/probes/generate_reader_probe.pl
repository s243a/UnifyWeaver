% SPDX-License-Identifier: MIT
% Copyright (c) 2026 John William Creighton (s243a)

:- use_module('../../../src/unifyweaver/targets/wam_llvm_target').
:- use_module('../../../src/unifyweaver/targets/wam_target').

:- initialization(main, main).

stream_reader_probe(Path, First, Second, EOF) :-
    stream_open(Path, Handle),
    read_line(Handle, First),
    read_line(Handle, Second),
    read_line(Handle, EOF),
    stream_close(Handle).

main :-
    write_wam_llvm_project(
        [stream_reader_probe/4],
        [module_name(plawk_reader_probe)],
        'examples/plawk/generated/plawk_reader_probe.ll'
    ),
    halt.
