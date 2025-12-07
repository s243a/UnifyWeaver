% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% This file is part of UnifyWeaver.
% Licensed under either MIT or Apache-2.0 at your option.

:- ['/mnt/c/Users/johnc/Dropbox/projects/UnifyWeaver/education/init.pl'].
:- ['/mnt/c/Users/johnc/Dropbox/projects/UnifyWeaver/education/family_tree.pl'].
:- use_module(unifyweaver(core/stream_compiler)).

generate_parent_script :-
    stream_compiler:compile_facts(parent, 2, [], BashCode),
    open('/mnt/c/Users/johnc/Dropbox/projects/UnifyWeaver/education/parent.sh', write, Stream),
    write(Stream, BashCode),
    close(Stream).

:- generate_parent_script.
:- halt.
