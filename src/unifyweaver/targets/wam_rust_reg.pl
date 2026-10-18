:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% wam_rust_reg.pl - register operands of the Rust WAM instruction encoding.
%
% R1 (docs/proposals/wam_rust_heap_cell_rewrite_design.md §4): an
% `Instruction` carries a decoded register operand (`Reg`, a u16, see
% templates/targets/rust_wam/instructions.rs.mustache) instead of the
% register's name. The generators emit it as a constructor call:
%
%     A<n>  ->  Reg::a(n)      (flat register-file index n-1)
%     X<n>  ->  Reg::x(n)      (flat index n+99)
%     Y<n>  ->  Reg::y(n)      (slot n-1 of the topmost environment frame)
%
% These are exactly the slots the runtime's name decoder
% (`WamState::reg_index`, `is_y_reg_name`, `YRegs::slot`) assigns to the same
% names, so an instruction behaves as it did when it carried the name.
%
% The shared compiler also emits a few names outside that pattern: the
% findall/bagof template scratch registers `_XT<N>` (wam_target.pl
% construct_template/7). The runtime decodes those with the generic
% `reg_index_parse` rule (any prefix other than A/X/Y -> flat slot 0, the
% same slot as A1). rust_reg_literal/2 reproduces that rule and emits the
% raw encoding, `Reg(<u16>)`, so they keep exactly today's behaviour.

:- module(wam_rust_reg, [
    rust_reg_literal/2,     % +RegName, -RustExpr
    rust_reg_code/2         % +RegName, -U16 (the Reg encoding)
]).

%% rust_reg_literal(+Name, -Expr) is det.
%  Name is a register token (atom or string, e.g. "A1", 'Y3', "_XT0").
%  Throws domain_error(wam_rust_register, Name) for a name the encoding
%  cannot represent exactly: a `Y<digit>...` name the runtime keys by text
%  (Y0, Y01, Y1x, more than five digits) or a name whose first character is
%  not ASCII (the runtime's decoder panics on it).
rust_reg_literal(Name, Expr) :-
    atom_codes(Name, Cs),
    (   rust_reg_canonical(Cs, Bank, N)
    ->  format(string(Expr), 'Reg::~w(~w)', [Bank, N])
    ;   rust_reg_code(Name, Code)
    ->  format(string(Expr), 'Reg(~w)', [Code])
    ;   throw(error(domain_error(wam_rust_register, Name),
                    context(rust_reg_literal/2, _)))
    ).

% A/X/Y + 1..5 decimal digits, no leading zero, n >= 1 (n <= 32767 for Y).
rust_reg_canonical([P, D0|Ds], Bank, N) :-
    memberchk(P-Bank, [0'A-a, 0'X-x, 0'Y-y]),
    D0 >= 0'1, D0 =< 0'9,
    length(Ds, L), L =< 4,
    forall(member(D, Ds), (D >= 0'0, D =< 0'9)),
    number_codes(N, [D0|Ds]),
    (   Bank == y -> N =< 32767 ; true ).

%% rust_reg_code(+Name, -Code) is semidet.
%  The u16 encoding of `Reg::from_name(Name)` (instructions.rs.mustache).
rust_reg_code(Name, Code) :-
    atom_codes(Name, Cs),
    (   rust_is_y_name(Cs)
    ->  rust_reg_canonical(Cs, y, N),
        Code is 0x8000 \/ N
    ;   Cs = [C0|_], C0 > 127
    ->  fail
    ;   rust_reg_index(Cs, Idx),
        (   Idx >= 600 -> Flat = 0x3FFF ; Flat = Idx ),
        (   Cs = [0'A|_] -> AF = 0x4000 ; AF = 0 ),
        Code is AF \/ Flat
    ).

% WamState::is_y_reg_name: len > 1, byte 0 'Y', byte 1 an ASCII digit.
rust_is_y_name([0'Y, D|_]) :- D >= 0'0, D =< 0'9.

% WamState::reg_index (D123 fast path + reg_index_parse), as an unbounded
% integer; anything >= 600 is out of range, as `usize::MAX` from a wrapping
% subtraction is.
rust_reg_index([P, D], Idx) :-
    memberchk(P, [0'A, 0'X, 0'Y]), D >= 0'0, D =< 0'9, !,
    N is D - 0'0,
    rust_reg_bank_index(P, N, Idx).
rust_reg_index([P, D0, D1], Idx) :-
    memberchk(P, [0'A, 0'X, 0'Y]),
    D0 >= 0'0, D0 =< 0'9, D1 >= 0'0, D1 =< 0'9, !,
    N is (D0 - 0'0) * 10 + (D1 - 0'0),
    rust_reg_bank_index(P, N, Idx).
rust_reg_index([], 0) :- !.
rust_reg_index([P|Rest], Idx) :-
    rust_usize_parse(Rest, N),
    (   memberchk(P, [0'A, 0'X, 0'Y])
    ->  rust_reg_bank_index(P, N, Idx)
    ;   Idx = 0
    ).

rust_reg_bank_index(0'A, N, Idx) :- ( N =:= 0 -> Idx = 1000000 ; Idx is N - 1 ).
rust_reg_bank_index(0'X, N, Idx) :- Idx is N + 99.
rust_reg_bank_index(0'Y, N, Idx) :- Idx is N + 199.

% `str::parse::<usize>().unwrap_or(0)`: an optional leading '+', then one or
% more ASCII digits, value below 2^64; anything else is 0.
rust_usize_parse(Cs0, N) :-
    (   Cs0 = [0'+|Cs] -> true ; Cs = Cs0 ),
    (   Cs \== [],
        forall(member(D, Cs), (D >= 0'0, D =< 0'9)),
        number_codes(N0, Cs),
        N0 < 18446744073709551616
    ->  N = N0
    ;   N = 0
    ).
