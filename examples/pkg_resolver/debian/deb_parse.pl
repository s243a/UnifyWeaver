:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% deb_parse.pl -- ingestion-edge Debian version parser.
% Produces pre-segmented deb/3 terms for resolver:version_lt/2.
% No hot-path parsing: the resolver walks s(OrderCodes, Digit) lists.
%
%   deb(Epoch, UpstreamSegs, RevisionSegs)
%   Seg = s(OrderCodes, DigitInt)   % OrderCodes = list of character codes
%
%   swipl -q -g test_deb_parse -t halt examples/pkg_resolver/debian/test_deb_version.pl

:- module(deb_parse, [
    parse_deb_version/2,
    parse_deb_version_atom/2,
    format_deb_version/2,
    deb_json/2,
    json_deb/2
]).

%!  parse_deb_version(+CodesOrAtomOrString, -deb(Epoch, Up, Rev)) is semidet.
parse_deb_version(In, Deb) :-
    to_codes(In, Codes),
    parse_deb_codes(Codes, Deb).

parse_deb_version_atom(In, Deb) :-
    parse_deb_version(In, Deb).

to_codes(In, Codes) :-
    atom(In), !, atom_codes(In, Codes).
to_codes(In, Codes) :-
    string(In), !, string_codes(In, Codes).
to_codes(In, In) :-
    is_list(In).

parse_deb_codes(Codes, deb(Epoch, Up, Rev)) :-
    (   append(EpC, [0':|Rest], Codes),
        EpC \== [],
        all_digits(EpC)
    ->  number_codes(Epoch, EpC),
        split_revision(Rest, UpC, RevC)
    ;   Epoch = 0,
        split_revision(Codes, UpC, RevC)
    ),
    segment_codes(UpC, Up),
    segment_codes(RevC, Rev).

all_digits([]).
all_digits([C|Cs]) :-
    C >= 0'0, C =< 0'9,
    all_digits(Cs).

% Last hyphen separates upstream from debian_revision (Policy §5.6.12).
split_revision(Codes, Up, Rev) :-
    (   append(Prefix, [0'-|Rev0], Codes),
        \+ member(0'-, Rev0)
    ->  Up = Prefix,
        Rev = Rev0
    ;   Up = Codes,
        Rev = []
    ).

segment_codes([], []).
segment_codes(Codes, [s(Order, Num)|Rest]) :-
    Codes \== [],
    take_nondigits(Codes, Order, AfterOrder),
    take_digits(AfterOrder, Digits, AfterDigits),
    (   Digits == []
    ->  Num = 0
    ;   number_codes(Num, Digits)
    ),
    segment_codes(AfterDigits, Rest).

take_nondigits([C|Cs], [C|Os], Rest) :-
    \+ (C >= 0'0, C =< 0'9),
    !,
    take_nondigits(Cs, Os, Rest).
take_nondigits(Cs, [], Cs).

take_digits([C|Cs], [C|Ds], Rest) :-
    C >= 0'0, C =< 0'9,
    !,
    take_digits(Cs, Ds, Rest).
take_digits(Cs, [], Cs).

format_deb_version(deb(Epoch, Up, Rev), Atom) :-
    segs_to_codes(Up, UpC),
    segs_to_codes(Rev, RevC),
    number_codes(Epoch, EpC),
    (   Epoch =:= 0
    ->  Prefix = []
    ;   append(EpC, [0':], Prefix)
    ),
    (   RevC == []
    ->  append(Prefix, UpC, All)
    ;   append(Prefix, UpC, T1),
        append(T1, [0'-|RevC], All)
    ),
    atom_codes(Atom, All).

segs_to_codes([], []).
segs_to_codes([s(Order, Num)|Rest], Codes) :-
    number_codes(Num, NC),
    segs_to_codes(Rest, RC),
    append(Order, NC, T),
    append(T, RC, Codes).

% JSON encoding used by dump / wamjs / store:
%   {"deb":[Epoch, [[OrderString, Num], ...], [[OrderString, Num], ...]]}
deb_json(deb(E, Up, Rev), _{deb: [E, UJ, RJ]}) :-
    maplist(seg_json, Up, UJ),
    maplist(seg_json, Rev, RJ).

seg_json(s(Codes, N), [S, N]) :-
    string_codes(S, Codes).

json_deb(_{deb: [E, UJ, RJ]}, deb(E, Up, Rev)) :-
    maplist(json_seg, UJ, Up),
    maplist(json_seg, RJ, Rev).
json_deb(Dict, Deb) :-
    is_dict(Dict),
    get_dict(deb, Dict, [E, UJ, RJ]),
    maplist(json_seg, UJ, Up),
    maplist(json_seg, RJ, Rev),
    Deb = deb(E, Up, Rev).

json_seg([S, N], s(Codes, N)) :-
    (   string(S) -> string_codes(S, Codes)
    ;   atom(S) -> atom_codes(S, Codes)
    ;   S = Codes
    ).
