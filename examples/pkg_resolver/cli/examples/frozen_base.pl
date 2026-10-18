:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% frozen_base.pl -- the P0.5 catalog (`catalog/9`). A Puppy-shaped frozen
% base, with every P0.5 feature exercised at least once:
%
%   * freeze reasons     -- abi_anchor / modified / blanket / footprint
%   * a named layer      -- `devx` holding gcc, loaded but NOT freeze-audited
%   * an exclusion       -- systemd is blacklisted at candidate generation
%   * aliases            -- urxvt -> rxvt, firefox-esr -> firefox
%   * an over-frozen hold (pango) and a suggest(abi_anchor) hold (gtk)
%   * a coordinated upgrade: glibc is the ABI anchor; moving it drags gtk
%     and pango with it, and nothing else.

:- module(catalog_frozen_base, [example_catalog/2]).

example_catalog(frozen_base, catalog(Packages, Depends, Conflicts,
                                     Base, Installed, Requested,
                                     Layers, Excluded, Aliases)) :-
    Packages = [
        package(glibc,   v(2,31,0)),
        package(glibc,   v(2,35,0)),
        package(gtk,     v(2,24,0)),
        package(gtk,     v(3,24,0)),
        package(pango,   v(1,48,0)),
        package(pango,   v(1,50,0)),
        package(nss,     v(3,68,0)),
        package(nss,     v(3,90,0)),
        package(firefox, v(91,0,0)),
        package(firefox, v(115,0,0)),
        package(mplayer, v(1,4,0)),
        package(rxvt,    v(9,22,0)),
        package(busybox, v(1,31,0)),
        package(busybox, v(1,35,0)),
        package(gcc,     v(10,2,0)),
        package(gcc,     v(12,2,0)),
        package(systemd, v(249,0,0))
    ],
    Depends = [
        % old gtk pins glibc BELOW 2.35 -- this is what makes glibc an anchor
        depends(gtk,     v(2,24,0),  glibc, range(v(2,31,0), v(2,35,0))),
        depends(gtk,     v(3,24,0),  glibc, gte(v(2,35,0))),
        depends(pango,   v(1,48,0),  gtk,   range(v(2,24,0), v(3,0,0))),
        depends(pango,   v(1,48,0),  glibc, gte(v(2,31,0))),
        depends(pango,   v(1,50,0),  gtk,   gte(v(3,24,0))),
        depends(pango,   v(1,50,0),  glibc, gte(v(2,35,0))),
        depends(firefox, v(91,0,0),  gtk,   gte(v(2,24,0))),
        depends(firefox, v(91,0,0),  glibc, gte(v(2,31,0))),
        depends(firefox, v(91,0,0),  nss,   gte(v(3,68,0))),
        depends(firefox, v(115,0,0), gtk,   gte(v(3,24,0))),
        depends(firefox, v(115,0,0), glibc, gte(v(2,35,0))),
        depends(firefox, v(115,0,0), nss,   gte(v(3,90,0))),
        depends(mplayer, v(1,4,0),   gtk,   gte(v(2,24,0))),
        depends(mplayer, v(1,4,0),   glibc, gte(v(2,31,0))),
        depends(rxvt,    v(9,22,0),  glibc, gte(v(2,31,0))),
        depends(gcc,     v(10,2,0),  glibc, gte(v(2,31,0))),
        depends(gcc,     v(12,2,0),  glibc, gte(v(2,35,0))),
        depends(systemd, v(249,0,0), glibc, gte(v(2,31,0)))
    ],
    Conflicts = [
        conflicts(systemd, v(249,0,0), busybox)
    ],
    Base = [
        base(glibc-v(2,31,0),   abi_anchor),   % held: everything links it
        base(gtk-v(2,24,0),     blanket),      % blanket, but pango pins it
        base(pango-v(1,48,0),   blanket),      % blanket, nothing pins it
        base(rxvt-v(9,22,0),    footprint),    % held to keep the ISO small
        base(busybox-v(1,31,0), modified)      % locally patched in Woof-CE
    ],
    Installed = [
        firefox-v(91,0,0),
        gtk-v(2,24,0),
        glibc-v(2,31,0),
        pango-v(1,48,0),
        nss-v(3,68,0),
        mplayer-v(1,4,0),
        rxvt-v(9,22,0),
        busybox-v(1,31,0)
    ],
    Requested = [firefox, mplayer],
    Layers = [
        layer(devx, [gcc-v(10,2,0)])           % loaded, but not freeze-audited
    ],
    Excluded = [systemd],
    Aliases = [
        alias(urxvt, rxvt),
        alias('firefox-esr', firefox)
    ].
