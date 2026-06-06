% wam_ite_structurer.pl
%
% Shared if-then-else / negation / once structurer for the WAM "lowered"
% emitters (Scala, Go, Rust, C++, …).
%
% The WAM compiler lowers ( Cond -> Then ; Else ), \+ Goal and once/1 into
% a choice-point block:
%
%     try_me_else(LElse)  <cond>  <commit>  <then>  jump(LCont)
%     LElse: trust_me     <else>
%     LCont: <continuation>
%
% where <commit> is cut_ite (for ->) or the !/0 builtin (for \+). The base
% per-target parsers drop label lines, which erases the LElse/LCont
% boundaries; each lowered emitter therefore parses a *label-preserving*
% instruction stream (keeping label(Name) markers) and calls structure_ite/2
% to fold every such block into an ite(Cond,Then,Else) term, emitting the
% continuation as a sibling of the block (not nested in the else).
%
% This logic is identical across targets, so it lives here once. Each target
% keeps its own label-preserving parser (coupled to its instr_from_parts/2)
% and its own emit_one(ite(...)) rendering; only the structural fold is
% shared. Previously this was duplicated verbatim as structure_ite_scala/
% _go/_rust (see git history).

:- module(wam_ite_structurer, [
    structure_ite/2,    % +FlatLabeledInstrs, -StructuredInstrs
    split_commit/3,     % +ThenPath, -Cond, -Then
    is_commit/1         % +Instr
]).

:- use_module(library(lists)).

%% structure_ite(+Flat, -Structured) is semidet.
%  Folds each well-formed ITE block into ite(Cond,Then,Else); drops the
%  structural markers (try_me_else/trust_me/cut_ite/jump/label). Cond, Then
%  and Else are themselves structured (nested blocks recurse), and the
%  block's continuation is emitted as a following sibling. Fails if a
%  try_me_else cannot be matched to a clean block, so callers can decline
%  and fall back.
structure_ite([], []).
structure_ite([try_me_else(LE)|Rest0], [ite(CondS,ThenS,ElseS)|Out]) :-
    !,
    append(ThenWithJump, [label(LE), trust_me | ElseAndRest], Rest0),
    \+ member(label(LE), ThenWithJump),          % first (matching) else label
    append(ThenPath, [jump(LC)], ThenWithJump),  % then-path ends in its jump
    append(ElsePath, [label(LC) | AfterCont], ElseAndRest),
    \+ member(label(LC), ElsePath),
    split_commit(ThenPath, Cond, Then),
    structure_ite(Cond, CondS),
    structure_ite(Then, ThenS),
    structure_ite(ElsePath, ElseS),
    structure_ite(AfterCont, Out).
structure_ite([label(_)|Rest], Out) :- !,
    structure_ite(Rest, Out).
structure_ite([I|Rest], [I|Out]) :-
    structure_ite(Rest, Out).

%% split_commit(+ThenPath, -Cond, -Then) is semidet.
%  Splits a then-path at its commit instruction (cut_ite for ->, the !/0
%  builtin for \+). The commit itself is dropped.
split_commit(Path, Cond, Then) :-
    append(Cond, [Commit|Then], Path),
    is_commit(Commit),
    \+ ( member(C0, Cond), is_commit(C0) ),   % split at the first commit
    !.

%% is_commit(+Instr)
%  The commit that separates a condition from its then-branch: cut_ite for
%  if-then-else, the !/0 builtin for negation. Builtin arity is carried as a
%  string by some targets and a number by others, hence both literals.
is_commit(cut_ite).
is_commit(builtin_call("!/0", _)).
is_commit(builtin_call('!/0', _)).
