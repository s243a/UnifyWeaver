:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% demo.pl -- three demonstration queries on the committed Debian slice.
%
%   swipl -q -g demo -t halt examples/pkg_resolver/debian/demo.pl

:- module(debian_demo, [demo/0]).

:- use_module('../resolver').
:- use_module(sample_catalog).
:- use_module(deb_parse, [format_deb_version/2]).

demo :-
    sample_stats(S),
    format("=== Debian slice stats ===~n~w~n~n", [S]),
    sample_catalog(Cat),
    format("=== 1. resolve(hostname) — non-trivial closure ===~n", []),
    (   resolve(Cat, [hostname], Sel)
    ->  print_sel(Sel)
    ;   format("FAIL~n", []),
        explain_blocked_list(Cat, hostname, BL),
        print_blocked(BL)
    ),
    nl,
    sample_layered_catalog(LCat),
    format("=== 2. resolve_layered vs frozen Essential+libc6-too-old ===~n", []),
    format("    hold: libc6 at 2.31-13+deb11u11 (blanket); essentials at slice vers~n", []),
    (   resolve_layered(LCat, [hostname], LSel)
    ->  format("resolve_layered succeeded (unexpected with too-old libc6):~n", []),
        print_sel(LSel)
    ;   format("resolve_layered failed (expected). explain_blocked:~n", []),
        explain_blocked_list(LCat, hostname, List),
        print_blocked(List)
    ),
    nl,
    format("=== 3. freeze_audit on that frozen base ===~n", []),
    freeze_audit(LCat, Audit),
    print_audit(Audit).

print_sel(Sel) :-
    length(Sel, N),
    format("  ~w packages~n", [N]),
    forall(member(Name-Ver, Sel),
           (   fmt_ver(Ver, VS),
               format("    ~w ~w~n", [Name, VS])
           )).

print_blocked([]).
print_blocked([B|Bs]) :-
    print_one_blocked(B),
    print_blocked(Bs).

print_one_blocked(blocked(Name, needs(C), base_has(V))) :-
    atom(Name),
    !,
    fmt_ver(V, VS),
    fmt_c(C, CS),
    format("  blocked ~w needs ~w base_has ~w~n", [Name, CS, VS]).
print_one_blocked(blocked(Name, needs(C), providers(Ps))) :-
    !,
    fmt_c(C, CS),
    format("  blocked virtual ~w needs ~w providers:~n", [Name, CS]),
    print_blocked(Ps).
print_one_blocked(blocked(alternatives(Rs))) :-
    !,
    format("  blocked alternatives:~n", []),
    forall(member(alt(N, Reason), Rs),
           format("    ~w -> ~w~n", [N, Reason])).

print_audit([]).
print_audit([A|As]) :-
    format("  ~w~n", [A]),
    print_audit(As).

fmt_ver(v(A, B, C), S) :-
    !,
    format(atom(S), '~w.~w.~w', [A, B, C]).
fmt_ver(Deb, S) :-
    format_deb_version(Deb, S).
fmt_ver(V, V).

fmt_c(any, any).
fmt_c(eq(V), S) :- fmt_ver(V, VS), format(atom(S), '=~w', [VS]).
fmt_c(gte(V), S) :- fmt_ver(V, VS), format(atom(S), '>=~w', [VS]).
fmt_c(lte(V), S) :- fmt_ver(V, VS), format(atom(S), '<=~w', [VS]).
fmt_c(gt(V), S) :- fmt_ver(V, VS), format(atom(S), '>>~w', [VS]).
fmt_c(lt(V), S) :- fmt_ver(V, VS), format(atom(S), '<<~w', [VS]).
fmt_c(range(Lo, Hi), S) :-
    fmt_ver(Lo, A), fmt_ver(Hi, B), format(atom(S), '[~w,~w)', [A, B]).
