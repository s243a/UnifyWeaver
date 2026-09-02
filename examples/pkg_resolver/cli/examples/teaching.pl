:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% teaching.pl -- the small catalog. A `catalog/6` term (the P0 shape: no
% layers, no exclusions, no aliases), just big enough to show the one thing
% uw-resolve exists for: `resolve` and `install-plan` disagree, and the
% disagreement has a name.
%
%   libc 1.0.0 sits in the frozen base as a blanket hold.
%   editor 1.1.0 wants syntax >= 2.0.0, and syntax 2.0.0 wants libc >= 2.0.0.
%
% So classic resolve upgrades libc out from under the base; the layered plan
% steps back to editor 1.0.0 + syntax 1.0.0 and leaves the base alone.

:- module(catalog_teaching, [example_catalog/2]).

example_catalog(teaching, catalog(Packages, Depends, Conflicts,
                                  Base, Installed, Requested)) :-
    Packages = [
        package(editor, v(1,0,0)),
        package(editor, v(1,1,0)),
        package(syntax, v(1,0,0)),
        package(syntax, v(2,0,0)),
        package(libc,   v(1,0,0)),
        package(libc,   v(2,0,0)),
        package(theme,  v(1,0,0))
    ],
    Depends = [
        depends(editor, v(1,0,0), syntax, gte(v(1,0,0))),
        depends(editor, v(1,0,0), libc,   gte(v(1,0,0))),
        depends(editor, v(1,1,0), syntax, gte(v(2,0,0))),
        depends(editor, v(1,1,0), libc,   gte(v(1,0,0))),
        depends(syntax, v(1,0,0), libc,   gte(v(1,0,0))),
        depends(syntax, v(2,0,0), libc,   gte(v(2,0,0))),
        depends(theme,  v(1,0,0), editor, gte(v(1,0,0)))
    ],
    Conflicts = [],
    Base = [
        libc-v(1,0,0)                      % P0 bare pair == blanket hold
    ],
    Installed = [
        editor-v(1,1,0),
        syntax-v(2,0,0),
        libc-v(2,0,0),
        theme-v(1,0,0)
    ],
    Requested = [theme].
