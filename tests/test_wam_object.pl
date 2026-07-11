:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% Phase 5 (JIT) slice 1: runtime-loadable WAM objects (.wamo).
% Covers the writer (subset validation + byte-stream shape) and, when
% clang is available, the full round trip: write a .wamo grammar, build a
% host binary carrying the loader, load the object at runtime, and run it.

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../src/unifyweaver/targets/wam_llvm_target').
:- use_module('../src/unifyweaver/targets/wam_target').
:- use_module('../src/unifyweaver/targets/wam_bootstrap_compiler').

% --- grammar predicates the tests compile into objects ---------------------
% Accumulator sum: exercises try_me_else / trust_me / get_list /
% unify_variable / builtin_call is/2 / execute (the tier2 loadable subset).
sum3([], A, A).
sum3([H|T], A, R) :- A1 is A + H, sum3(T, A1, R).
answer(R) :- sum3([100, 10, 9], 0, R).          % -> 119
answer_swapped(R) :- sum3([7, 8, 5, 1000], 0, R). % -> 1020

uses_float(X) :- X is 1.5 + 2.5.                % float constant -> rejected

% returns a compound record rather than a scalar: rec2f(Integer, Float).
% Exercises the structured-return primitive (@wam_object_call_record).
makerec(R) :- X = 10, Y is X + 0.5, R = rec2f(X, Y).   % -> rec2f(10, 10.5)

% a record with an integer and an atom (string) field: rs(7, hello).
% Exercises string fields (typecode 2 -> (ptr, len)).
makerecs(R) :- R = rs(7, hello).

% returns a list of integer key-value pairs, for the assoc-table variant
% (@wam_object_call_assoc): each K-V is inserted into an i64 assoc table.
tally(R) :- R = [1-100, 2-200, 3-30].

% atom-first-argument clause indexing: coltab/2 compiles to a
% switch_on_constant. pick/1 looks up `green`. The loader nops the switch
% and runs the try_me_else chain (unindexed but correct) -> 2.
coltab(red, 1).
coltab(green, 2).
coltab(blue, 3).
pick(R) :- coltab(green, R).          % -> 2

% get_structure functor-check regression: dispatch over a tagged compound to
% one of several clauses keyed by the first argument's functor. Before
% get_structure compared the functor (it accepted ANY compound of the right
% shape), calling dp_sel(three(7),R) wrongly matched the FIRST clause
% (one/1) -- reading three(7)'s arg as if it were one(7) -- and returned 8
% instead of 42. The dp_many clause has a list-typed permanent head-arg (the
% shape whose body does not cleanly fail, which made the bug visible). With
% the functor check, one/1 and many/1 fail cleanly and three/1 matches -> 42.
dp_go(R)          :- dp_sel(three(7), R).
dp_sel(one(X), R)   :- R is X + 1.
dp_sel(many(L), R)  :- length(L, N), R is N * 100.
dp_sel(three(X), R) :- R is X * 6.          % 7 * 6 = 42

% call/N meta-call inside a loaded object (eval bootstrap milestone 2). The
% goal is built at runtime and dispatched through the object's OWN meta-call
% table, so a loaded .wamo can call its own predicates.
mfoo(100).
metaatom(R) :- G = mfoo, call(G, R).  % atom goal -> mfoo(R) -> 100
maddk(X, R) :- R is X + 32.
metacomp(R) :- G = maddk(10), call(G, R). % compound goal -> maddk(10,R) -> 42

% Aggregate control (findall / setof / bagof) inside a loaded object (eval
% bootstrap milestone 3). The tier-2 compiler brackets the collected goal
% with begin_aggregate/end_aggregate; those opcodes operate purely on VM
% state, so a loaded object runs them like the host. The goal is a user
% predicate (agnum/1) -- the case a compiler actually hits (iterating over
% clauses), as opposed to a backtracking list builtin.
agnum(30).
agnum(10).
agnum(20).
agnum(10).
collectsum(S) :- findall(X, agnum(X), L), sum_list(L, S).  % 30+10+20+10 -> 70
setcard(N)    :- setof(X, agnum(X), L), length(L, N).      % {10,20,30} -> 3
bagcard(N)    :- bagof(X, agnum(X), L), length(L, N).      % 4 solutions -> 4

% Regression: an aggregate whose goal yields ZERO solutions. end_aggregate
% never runs, so the aggregate frame's return PC must have been set at
% begin_aggregate time (via the forward scan) rather than left at the
% placeholder 0 -- otherwise finalize jumps to PC 0 and the predicate
% re-executes forever (empty findall used to OOM / segfault). Exercises the
% same @run_loop the host uses, so it covers the AOT path too.
noagg(_) :- fail.
emptycount(N) :- findall(X, noagg(X), L), length(L, N).    % [] -> 0

% term_to_atom/2 (eval bootstrap milestone 3b): render a term to its text and
% intern it. Exercises nested compound + list rendering; list detection is by
% functor bytes, not pointer identity, so it works with the loaded object's
% own functor copies (a pointer compare would mis-render [x,y,z]).
ttalen(N) :- term_to_atom(pt(3, [x,y,z]), A), atom_length(A, N).  % "pt(3,[x,y,z])" -> 13

% read_term_from_atom/2 (eval bootstrap milestone 3b, reader). Atomic terms:
% parse an integer from text at runtime and use it arithmetically.
readint_obj(R) :- read_term_from_atom('40', T), R is T + 2.  % -> 42

% Compound reader: a parsed compound unifies against a source-level literal
% pattern -- exercising @wam_functor_eq (the reader's atom-table functor
% pointer differs from the AOT @.fn_point global; the strcmp fallback makes
% them equal). Then arithmetic on the decomposed args.
readcompound_obj(R) :- read_term_from_atom('point(3,4)', T), T = point(X,Y), R is X*10 + Y.  % 34

% Nested compound decomposition.
readnested_obj(R) :- read_term_from_atom('f(g(7),h(5))', T), T = f(g(A),h(B)), R is A + B.  % 12

% List reader: parse a list and reduce it.
readlist_obj(S) :- read_term_from_atom('[10,20,30]', L), sum_list(L, S).  % 60

% Operator reader: parse an infix arithmetic expression with precedence /
% associativity / parens / negatives, then evaluate it with is/2 (the
% arithmetic evaluator dispatches on the functor bytes, so it evaluates the
% reader-built + / * / - compounds).
readop_prec(R)  :- read_term_from_atom('1+2*3', T), R is T.        % 7
readop_paren(R) :- read_term_from_atom('(1+2)*3', T), R is T.      % 9
readop_assoc(R) :- read_term_from_atom('100 - 2 * -3', T), R is T. % 106

% Variable reader: a repeated variable name shares one cell within the term, so
% binding it once via unification propagates to every occurrence. Anonymous _
% are distinct. (Reader vars are bound by unifying the parsed TERM, not by the
% surrounding clause's variables.)
readvar_shared(R) :- read_term_from_atom('p(X,X)', T), T = p(9, Y), R is Y.   % 9
readvar_arith(R)  :- read_term_from_atom('v(A,A)', T), T = v(6, X), R is X*7. % 42
readvar_anon(R)   :- read_term_from_atom('q(_,_)', T), T = q(3,4), R is 1.    % 1 (distinct)

% Control operators: :- (1200), , (1000), ; (1100), -> (1050). With variables
% and these, a whole clause parses. readclause parses "foo(X) :- bar(X), baz(X)"
% into :-(foo(X), ,(bar(X),baz(X))) with X shared across head and body: binding
% X once (via V) is visible in the body goal (W).
readclause(R) :- read_term_from_atom('foo(X) :- bar(X), baz(X)', T),
                 T = (H :- B), H = foo(V), B = (G1, _), G1 = bar(W),
                 V = 7, R is W.                                  % 7
% Right-associative conjunction: 1,2,3 = ,(1,,(2,3)).
readconj(R)  :- read_term_from_atom('1,2,3', T), T = (A,Bc,C), R is A*100+Bc*10+C. % 123
readsemi(R)  :- read_term_from_atom('11;22', T), T = (A;Bd), R is A+Bd.            % 33

% Floats (parsed via strtod, tag Float) and quoted atoms (spaces/specials).
readfloat(R)  :- read_term_from_atom('3.5 + 1.5', T), F is T, R is truncate(F).       % 5
readfloatc(R) :- read_term_from_atom('pt(2.5,4.5)', T), T = pt(A,B), R is truncate(A+B). % 7
readquoted(R) :- read_term_from_atom('foo(\'hello world\')', T), T = foo(A), atom_length(A, R). % 11

% Byte-buffer output (eval bootstrap milestone 4): a loaded grammar BUILDS a
% byte string at runtime -- via arithmetic + the string/codes builtins -- and
% returns it as an atom. The host reads the bytes back through
% @wam_object_call_bytes ({ptr, len, ok}). This is the shape the eventual eval
% path uses to hand assembled .wamo text back across the blob bridge; it
% composes primitives that already exist (items 2-3), so it needs no new
% target IR. Three grammars: a computed decimal, a synthesized ".wamo"-style
% header line, and a literal code list.
emitnum(S)   :- N is 6*7, number_codes(N, Cs), atom_codes(S, Cs).            % "42"
emithdr(S)   :- number_codes(2, VC), atom_codes(V, VC), atom_concat('WAMO ', V, S). % "WAMO 2"
emitcodes(S) :- atom_codes(S, [104,105]).                                   % "hi"

% Dynamic clause store (eval bootstrap milestone 3b-db): assert facts at
% runtime into the process-global store and call them back through the call/1
% meta-call, which consults the store when the meta table misses (see
% PLAWK_DYNAMIC_DB.md). Ground facts, PR 1. These run in a loaded object -- the
% store is process-global, shared by the host and any loaded .wamo.
% Deterministic call of an asserted fact.
dsingle(R) :- assertz(fact(42)), G = fact(X), call(G), R = X.                  % 42
% Argument-directed selection: the iterator scans past pair(a,_) (functor
% matches, args do not) to unify pair(b,X).
dselect(R) :- assertz(pair(a,1)), assertz(pair(b,2)), assertz(pair(c,3)),
              G = pair(b,X), call(G), R = X.                                   % 2
% Real choice-point backtracking (no findall): num(1) is yielded first, the
% continuation X>=2 fails, backtrack re-enters the clause iterator, num(2)
% succeeds. Proves the -3 choice point advances the store scan.
dfirst(R) :- assertz(num(1)), assertz(num(2)),
             G = num(X), call(G), X >= 2, R = X.                              % 2
% asserta prepends: the prepended clause is the first solution.
dorder(R) :- assertz(item(1)), assertz(item(2)), asserta(item(9)),
             G = item(X), call(G), R = X.                                     % 9
% retractall tombstones every match; the subsequent call then fails.
dret(R) :- assertz(k(1)), assertz(k(2)), retractall(k(_)),
           G = k(X), ( call(G) -> R = X ; R = 0 ).                            % 0
% Partial retractall: only k2(2,_) is removed; k2(1,10) survives.
dretp(R) :- assertz(k2(1,10)), assertz(k2(2,20)), retractall(k2(2,_)),
            G = k2(A,B), call(G), R is A*100+B.                               % 110

% findall over a call/1 meta-call goal. This used to collect nothing: the
% tier-2 compiler re-initialised the aggregate template variable with a fresh
% cell (put_variable), disconnecting it from the copy embedded in the
% pre-built goal G, so the goal bound one cell while the aggregate collected
% another. Fixed by not re-initialising a template var that already has a
% register (it is shared with G; backtracking refreshes it). Two flavours:
% a predicate compiled into the object, and dynamically asserted facts.
fanum(1). fanum(2). fanum(3).
faobjc(S)  :- G = fanum(X), findall(X, call(G), L), sum_list(L, S).           % 6
fadynsum(S) :- assertz(dv(10)), assertz(dv(20)), assertz(dv(30)),
               G = dv(X), findall(X, call(G), L), sum_list(L, S).             % 60

% Milestone 3b-db PR 2: DIRECT calls to :- dynamic predicates (no explicit
% call/1) and nondet retract/1. A body goal calling a dynamic predicate with
% no compiled clauses is rewritten to a call/1 meta-call, which consults the
% store; retract/1 lowers to a dedicated remove+unify+backtrack iterator.
% The :- dynamic declarations let the compiler detect the direct-call case.
:- dynamic w/1, nn/1, ee/2, cc/1.
ddirect(R)  :- assertz(w(5)), w(X), R = X.                                    % 5 (direct call)
ddirbt(R)   :- assertz(nn(1)), assertz(nn(2)), nn(X), X >= 2, R = X.          % 2 (direct + backtrack)
dtwoarg(R)  :- assertz(ee(a,10)), assertz(ee(b,20)), ee(b,V), R = V.          % 20 (2-arg direct)
dretract(R) :- assertz(cc(1)), assertz(cc(2)), assertz(cc(3)), retract(cc(X)), R = X.  % 1
dretbt(R)   :- assertz(cc(1)), assertz(cc(2)), assertz(cc(3)),
               retract(cc(X)), X >= 2, R = X.                                 % 2 (retract + backtrack)
dretgone(R) :- assertz(cc(1)), assertz(cc(2)), retract(cc(_)), retract(cc(_)),
               ( retract(cc(_)) -> R = 1 ; R = 0 ).                           % 0 (both removed)
dretcount(N) :- assertz(cc(1)), assertz(cc(2)), assertz(cc(3)),
                findall(X, retract(cc(X)), L), length(L, N).                  % 3 (nondet retract)

% Milestone 3c: catch/3 + throw/1. catch pushes a side-stack frame and
% meta-calls Goal; throw deep-copies the ball and unwinds to the nearest frame
% whose catcher unifies with it, running Recovery. All helper predicates must
% be compiled into the object so the meta-call resolves them.
ctrisky(_) :- throw(myerr(42)).
cthandle(V, V).
ct_catch(R)   :- catch(ctrisky(R), myerr(V), cthandle(V, R)).                 % 42 (catch + recover)
ctok(5).
ctnorec(_) :- throw(unused).
ct_nothrow(R) :- catch(ctok(R), _, ctnorec(R)).                              % 5 (Goal succeeds)
ctthrower(_) :- throw(typeb(9)).
cthia(_) :- throw(nope).
ctmid(R) :- catch(ctthrower(R), typea(_), cthia(R)).
cthb(V, V).
ct_nested(R)  :- catch(ctmid(R), typeb(V), cthb(V, R)).                       % 9 (propagate to outer)
ctcompute(R) :- X is 6*7, throw(result(X)), R = X.
ctgrab(V, V).
ct_ball(R)    :- catch(ctcompute(R), result(V), ctgrab(V, R)).               % 42 (recovery uses ball)
ct_uncaught(R) :- throw(oops), R = 1.                                        % uncaught -> fails

% Milestone 5 (eval/compile pipeline): a loaded compiler object runs on source
% text and emits .wamo bytes, which @wam_object_eval loads into a fresh VM in
% the same process; running the result closes the eval loop. ea/1 is the
% "program" that gets compiled+loaded+run; echocompile/2 is a stand-in compiler
% (echoes its source as the emitted object -- a real compiler is milestone 6).
ea(R) :- R = 42.
echocompile(Src, Wamo) :- Wamo = Src.

% The bootstrap compiler itself (Stage A serializer through the unified
% cgfull/cgfull_term chain) lives in src/unifyweaver/targets/
% wam_bootstrap_compiler.pl since the self-host fixpoint closed --
% it is a shippable artifact now, not test scaffolding. The tests
% below compile it with write_wam_object/3 (roots qualified with the
% wam_bootstrap_compiler module) and exercise every stage.

% The fixpoint source: the Stage A serializer (wz_* chain) restated in the
% accepted subset (cut-free, builtins + calls + list/structure patterns).
% The entry checksums its own output (byte sum + length). Like the Stage A
% chain above, the emitters are DIFFERENCE LISTS (A0 = open list, A1 = its
% tail after the item), so the doubly-compiled serializer is linear too --
% and compiling it exercises open-tail call operands ([10|A1], [32|Mid])
% and append/3 in construct mode with a partial second argument.
fixpoint_serializer_source('[(main0(R) :- serz(Cs), sum_list(Cs, S), length(Cs, L), R is S + L), (serz(Out) :- atom_codes(''ea/1'', NC), wzs(0, NC, 0, 0, 0, [0], [enc(0,42,65536,0), enc(20,0,0,0)], Out)), (wzi(N, A0, A1) :- number_codes(N, Cs), append(Cs, [10|A1], A0)), (wza(X, A0, A1) :- atom_codes(X, Cs), append(Cs, [10|A1], A0)), (wzn(Cs, A0, A1) :- length(Cs, L), number_codes(L, LC), append(LC, [32|Mid], A0), append(Cs, [10|A1], Mid)), (wzsi(N, A0, A1) :- number_codes(N, Cs), append([32|Cs], A1, A0)), (wzs(EI, NC, LI, NA, NF, PCs, Is, Out) :- wzh(EI, NC, LI, NA, NF, Out, H), wzb(PCs, Is, H, [])), (wzh(EI, NC, LI, NA, NF, A0, Out) :- wza(''WAMO'', A0, A1), wzi(2, A1, A2), wzi(EI, A2, A3), wzi(1, A3, A4), wzh2(NC, LI, NA, NF, A4, Out)), (wzh2(NC, LI, NA, NF, A0, Out) :- wzn(NC, A0, A1), wzi(LI, A1, A2), wzi(NA, A2, A3), wzi(NF, A3, Out)), (wzb(PCs, Is, A0, Out) :- wzp(PCs, A0, A1), wzc(Is, A1, A2), wzi(0, A2, Out)), (wzp(PCs, A0, A2) :- length(PCs, NL), wzi(NL, A0, A1), wzpr(PCs, A1, A2)), wzpr([], A, A), (wzpr([P|Ps], A0, A2) :- wzi(P, A0, A1), wzpr(Ps, A1, A2)), (wzc(Is, A0, A2) :- length(Is, NC2), wzi(NC2, A0, A1), wzcr(Is, A1, A2)), wzcr([], A, A), (wzcr([enc(T,O1,O2,Rl)|Is], A0, A2) :- wzr(T, O1, O2, Rl, A0, A1), wzcr(Is, A1, A2)), (wzr(T, O1, O2, Rl, A0, A5) :- number_codes(T, Tc), append(Tc, A1, A0), wzsi(O1, A1, A2), wzsi(O2, A2, A3), wzsi(Rl, A3, A4), A4 = [10|A5])]').

% The GEN-3 source: a mini-COMPILER (not just a serializer) in the accepted
% subset. Where the serializer source above starts from a hard-coded
% instruction list, cmp2/2 starts from SOURCE TEXT: it runs the runtime
% reader as a compiled goal (read_term_from_atom/2, builtin id 174),
% decomposes the clause with =../2 (avoiding a (H :- B) pattern literal --
% control functors are excluded from the data tables by design), makes a
% DISPATCHING codegen decision -- ( integer(V) -> int get_constant ; atom
% get_constant with an ATOM TABLE row emitted from the compiled program,
% reloc class 1 ) -- assembles the entry name codes ("<pred>/1"), and
% serializes with the same difference-list wz chain (wzs here takes the
% atom LIST and emits NA + its rows via wzat). Three generations: the
% AOT-compiled cgfull (gen 1) compiles THIS source into gen 2; gen 2
% compiles TWO golden programs (quoted atoms below, which must survive
% collection into the atom table, relocation, and re-parsing by the
% loaded reader) into exactly the bytes the Stage A serializer yields.
fixpoint_compiler_source('[(main0(R) :- cmp2(''ea(R2) :- R2 = 42'', Cs1), cmp2(''eb(R2) :- R2 = foo'', Cs2), sum_list(Cs1, S1), length(Cs1, L1), sum_list(Cs2, S2), length(Cs2, L2), R is S1 + L1 + S2 + L2), (cmp2(Src, Out) :- read_term_from_atom(Src, C), C =.. [_, H, B], functor(H, P, 1), B =.. [_, _, V], atom_codes(P, PC), append(PC, [47, 49], NC), (integer(V) -> As = [], Is1 = [enc(0, V, 65536, 0)] ; As = [V], Is1 = [enc(0, 0, 0, 1)]), append(Is1, [enc(20, 0, 0, 0)], Is), wzs(0, NC, 0, As, [0], Is, Out)), (wzi(N, A0, A1) :- number_codes(N, Cs), append(Cs, [10|A1], A0)), (wza(X, A0, A1) :- atom_codes(X, Cs), append(Cs, [10|A1], A0)), (wzn(Cs, A0, A1) :- length(Cs, L), number_codes(L, LC), append(LC, [32|Mid], A0), append(Cs, [10|A1], Mid)), (wzsi(N, A0, A1) :- number_codes(N, Cs), append([32|Cs], A1, A0)), (wzs(EI, NC, LI, As, PCs, Is, Out) :- wzh(EI, NC, LI, Out, A6), length(As, NA), wzi(NA, A6, A7), wzat(As, A7, A8), wzi(0, A8, A9), wzp(PCs, A9, A10), wzc(Is, A10, A11), wzi(0, A11, [])), (wzh(EI, NC, LI, A0, Out) :- wza(''WAMO'', A0, A1), wzi(2, A1, A2), wzi(EI, A2, A3), wzi(1, A3, A4), wzn(NC, A4, A5), wzi(LI, A5, Out)), wzat([], A, A), (wzat([X|Xs], A0, A2) :- atom_codes(X, Cs), wzn(Cs, A0, A1), wzat(Xs, A1, A2)), (wzp(PCs, A0, A2) :- length(PCs, NL), wzi(NL, A0, A1), wzpr(PCs, A1, A2)), wzpr([], A, A), (wzpr([P|Ps], A0, A2) :- wzi(P, A0, A1), wzpr(Ps, A1, A2)), (wzc(Is, A0, A2) :- length(Is, NC2), wzi(NC2, A0, A1), wzcr(Is, A1, A2)), wzcr([], A, A), (wzcr([enc(T,O1,O2,Rl)|Is], A0, A2) :- wzr(T, O1, O2, Rl, A0, A1), wzcr(Is, A1, A2)), (wzr(T, O1, O2, Rl, A0, A5) :- number_codes(T, Tc), append(Tc, A1, A0), wzsi(O1, A1, A2), wzsi(O2, A2, A3), wzsi(Rl, A3, A4), A4 = [10|A5])]').

% The MIDDLE source (fixpoint, next slice): cgfull's own single-clause
% CODEGEN restated cut-free in the accepted subset. Where gen 2 above
% hard-coded its instruction templates, mid2 performs real register
% allocation and instruction generation, mirroring the cgfull middle
% exactly so its output is byte-identical:
%   - copy_term + numbervars (both loadable builtins) turn the read
%     clause's variables into matchable ''$VAR''(N) structures;
%   - mgl splits the ,/2 conjunction via =.. + == (control functors
%     cannot be pattern literals in the subset);
%   - mcol collects is-expression operators into the functor table
%     (first occurrence order, like collect_tables);
%   - mhead walks the head with arg/3 (variable index -- the loadable
%     form) emitting get_variable/get_value by first-occurrence, exactly
%     like head_arg_instrs;
%   - mgoal compiles `L is E` the way f_goal does post-nested-lift:
%     operand to A1 (put_variable/put_value), expression to A2
%     (put_structure op/2 + set_value/set_constant), builtin is/2;
%   - wzs/wzat serialize with BOTH tables (atoms + functors).
% The init-set threads as a plain N-list with memberchk, and every
% dispatch is an ITE -- no cuts anywhere.
fixpoint_middle_source('[(main0(R) :- cm2(''sum3(A, B, R2) :- T is A + B, R2 is T + 1'', Cs), sum_list(Cs, S), length(Cs, L), R is S + L), (cm2(Src, Out) :- read_term_from_atom(Src, C0), copy_term(C0, C1), numbervars(C1, 0, _), C1 =.. [_, H, B], mgl(B, Gs), functor(H, P, Ar), madd(P, [], FT0), mcol(Gs, FT0, FT), mname(P, Ar, NC), mhead(1, Ar, H, [], In1, HIs), mgoals(Gs, FT, In1, _, GIs), append(HIs, GIs, Body), append([enc(16, 0, 0, 0)|Body], [enc(17, 0, 0, 0), enc(20, 0, 0, 0)], Is), wzs(0, NC, 0, [], FT, [0], Is, Out)), (mgl(G, L) :- (G =.. [F, X, Y], F == '','' -> mgl(X, L1), mgl(Y, L2), append(L1, L2, L) ; L = [G])), mcol([], A, A), (mcol([G|Gs], A0, A2) :- (G =.. [F, _, E], F == is, functor(E, Op, 2) -> madd(Op, A0, A1) ; A1 = A0), mcol(Gs, A1, A2)), (madd(Op, A0, A1) :- (memberchk(Op, A0) -> A1 = A0 ; append(A0, [Op], A1))), (mname(P, Ar, NC) :- atom_codes(P, PC), number_codes(Ar, AC), append(PC, [47|AC], NC)), (mhead(I, Ar, H, In0, In, Is) :- (I > Ar -> In = In0, Is = [] ; arg(I, H, A), mharg(A, I, In0, In1, AI), I1 is I + 1, mhead(I1, Ar, H, In1, In, RIs), append(AI, RIs, Is))), (mharg(''$VAR''(N), I, In0, In1, [enc(T, Y, Ai, 0)]) :- Y is 48 + N, Ai is I - 1, (memberchk(N, In0) -> T = 2, In1 = In0 ; T = 1, In1 = [N|In0])), mgoals([], _, In, In, []), (mgoals([G|Gs], FT, In0, In, Is) :- mgoal(G, FT, In0, In1, GI), mgoals(Gs, FT, In1, In, RI), append(GI, RI, Is)), (mgoal(G, FT, In0, In2, Is) :- G =.. [_, L, E], mop(L, 0, In0, In1, IL), mexpr(E, FT, In1, In2, IE), append(IL, IE, LE), append(LE, [enc(21, 0, 2, 0)], Is)), (mop(''$VAR''(N), Ai, In0, In1, [enc(T, Y, Ai, 0)]) :- Y is 48 + N, (memberchk(N, In0) -> T = 10, In1 = In0 ; T = 9, In1 = [N|In0])), (mexpr(E, FT, In0, In1, Is) :- functor(E, Op, 2), fidx(Op, FT, FI), arg(1, E, X), arg(2, E, Y), mset(X, In0, Ina, SX), mset(Y, Ina, In1, SY), append([enc(11, FI, 131073, 2)|SX], SY, Is)), (mset(''$VAR''(N), In0, In1, [enc(T, Y, 0, 0)]) :- Y is 48 + N, (memberchk(N, In0) -> T = 14, In1 = In0 ; T = 13, In1 = [N|In0])), (mset(V, In, In, [enc(15, V, 1, 0)]) :- integer(V)), (fidx(Op, [F|Fs], I) :- (Op == F -> I = 0 ; fidx(Op, Fs, I1), I is I1 + 1)), (wzi(N, A0, A1) :- number_codes(N, Cs), append(Cs, [10|A1], A0)), (wza(X, A0, A1) :- atom_codes(X, Cs), append(Cs, [10|A1], A0)), (wzn(Cs, A0, A1) :- length(Cs, L), number_codes(L, LC), append(LC, [32|Mid], A0), append(Cs, [10|A1], Mid)), (wzsi(N, A0, A1) :- number_codes(N, Cs), append([32|Cs], A1, A0)), (wzs(EI, NC, LI, As, Fs, PCs, Is, Out) :- wzh(EI, NC, LI, Out, A6), length(As, NA), wzi(NA, A6, A7), wzat(As, A7, A8), length(Fs, NF), wzi(NF, A8, A9), wzat(Fs, A9, A10), wzp(PCs, A10, A11), wzc(Is, A11, A12), wzi(0, A12, [])), (wzh(EI, NC, LI, A0, Out) :- wza(''WAMO'', A0, A1), wzi(2, A1, A2), wzi(EI, A2, A3), wzi(1, A3, A4), wzn(NC, A4, A5), wzi(LI, A5, Out)), wzat([], A, A), (wzat([X|Xs], A0, A2) :- atom_codes(X, Cs), wzn(Cs, A0, A1), wzat(Xs, A1, A2)), (wzp(PCs, A0, A2) :- length(PCs, NL), wzi(NL, A0, A1), wzpr(PCs, A1, A2)), wzpr([], A, A), (wzpr([P|Ps], A0, A2) :- wzi(P, A0, A1), wzpr(Ps, A1, A2)), (wzc(Is, A0, A2) :- length(Is, NC2), wzi(NC2, A0, A1), wzcr(Is, A1, A2)), wzcr([], A, A), (wzcr([enc(T,O1,O2,Rl)|Is], A0, A2) :- wzr(T, O1, O2, Rl, A0, A1), wzcr(Is, A1, A2)), (wzr(T, O1, O2, Rl, A0, A5) :- number_codes(T, Tc), append(Tc, A1, A0), wzsi(O1, A1, A2), wzsi(O2, A2, A3), wzsi(Rl, A3, A4), A4 = [10|A5])]').

% The FRONT source (fixpoint, next slice): cgfull's clause-grouping,
% label, and chain machinery restated cut-free, on top of the middle.
% New over fixpoint_middle_source: facts vs rules (mhb tags facts with
% body atom true), the GENERIC table walk mwt/mwl mirroring
% collect_tables/walk_term (var/nil/integer/atom/compound dispatch with
% the control-functor skip list), group_clauses/take_same (mgrp/mtks),
% group_labels + label lookup (mlbl/mlook), the try_me_else /
% retry_me_else / trust_me chain builders with PC threading and
% Label-PC pair collection (mggs/mprd/malt -- keysort + pairs_values
% close the PC table exactly like cgfull), integer/atom head constants,
% predicate-call goals (mlook + call), and =/2 goals. The serializer
% emits BOTH tables. Compiling THIS source is also the regression that
% found the reader var-dict cap (campaign finding no. 11): it has more
% than 128 distinct variable names, so before the growable dict the
% later clauses (the wz chain) silently lost variable sharing and the
% doubly-compiled serializer dropped every instruction opcode.
fixpoint_front_source('[(main0(R) :- cm3(''[(dbl(X, Y) :- Y is X + X), (tst(R2) :- dbl(4, R2)), pick(1, a), pick(2, b)]'', Cs), sum_list(Cs, S), length(Cs, L), R is S + L), (cm3(Src, Out) :- read_term_from_atom(Src, Cls), mwl(Cls, [], [], At, FT), mgrp(Cls, Gs), mlbl(Gs, 0, PL), length(Gs, NP0), mggs(Gs, PL, At, FT, 0, NP0, AllIs, EPCs, Prs), keysort(Prs, SPrs), pairs_values(SPrs, XPCs), append(EPCs, XPCs, PCs), Cls = [C1|_], mhb(C1, H1, _), functor(H1, P1, A1), mname(P1, A1, NC), wzs(0, NC, 0, At, FT, PCs, AllIs, Out)), (mhb(C, H, B) :- (C =.. [F, X, Y], F == '':-'' -> H = X, B = Y ; H = C, B = true)), mwl([], A, F, A, F), (mwl([T|Ts], A0, F0, A, F) :- mwt(T, A0, F0, A1, F1), mwl(Ts, A1, F1, A, F)), (mwt(T, A0, F0, A, F) :- (var(T) -> A = A0, F = F0 ; T == [] -> madd(''[]'', A0, A1), A = A1, F = F0 ; integer(T) -> A = A0, F = F0 ; atom(T) -> madd(T, A0, A1), A = A1, F = F0 ; T =.. [Fn|Args], mwf(Fn, F0, F1), mwl(Args, A0, F1, A, F))), (mwf(Fn, F0, F1) :- (mskip(Fn) -> F1 = F0 ; madd(Fn, F0, F1))), (mskip(F) :- memberchk(F, ['':-'', '','', '';'', ''->'', ''is'', ''='', ''>'', ''<'', ''>='', ''=<'', ''=:='', ''=\\='', ''=='', ''\\==''])), (madd(X, L0, L1) :- (memberchk(X, L0) -> L1 = L0 ; append(L0, [X], L1))), mgrp([], []), (mgrp([C|Cs], [g(P, Ar, [C|Same])|Gs]) :- mhb(C, H, _), functor(H, P, Ar), mtks(Cs, P, Ar, Same, Rest), mgrp(Rest, Gs)), mtks([], _, _, [], []), (mtks([C|Cs], P, Ar, Same, Rest) :- mhb(C, H, _), functor(H, P2, A2), (P2 == P, A2 =:= Ar -> Same = [C|S1], mtks(Cs, P, Ar, S1, Rest) ; Same = [], Rest = [C|Cs])), mlbl([], _, []), (mlbl([g(P, Ar, _)|Gs], I, [pl(P, Ar, I)|R]) :- I1 is I + 1, mlbl(Gs, I1, R)), (mlook(P, Ar, [pl(P2, A2, L2)|Ps], L) :- (P == P2, Ar =:= A2 -> L = L2 ; mlook(P, Ar, Ps, L))), mggs([], _, _, _, _, _, [], [], []), (mggs([g(_, _, Cls)|Gs], PL, At, FT, PC, NPin, AllIs, [PC|EPCs], Prs) :- mprd(Cls, PL, At, FT, PC, NPin, Is, PC1, L1, Prs1), mggs(Gs, PL, At, FT, PC1, L1, RestIs, EPCs, Prs2), append(Is, RestIs, AllIs), append(Prs1, Prs2, Prs)), (mprd([C], PL, At, FT, PC, L0, Is, PCout, L, Prs) :- mone(C, PL, At, FT, PC, L0, L, Prs, Is), length(Is, N), PCout is PC + N), (mprd([C1, C2|Rest], PL, At, FT, PC, L0, Is, PCout, L, Prs) :- ChainL = L0, L1 is L0 + 1, PCc is PC + 1, mone(C1, PL, At, FT, PCc, L1, L2, Prs1, C1Is), length(C1Is, N1), PC1 is PCc + N1, malt([C2|Rest], PL, At, FT, PC1, ChainL, L2, AltIs, PCout, L, Prs2), append(Prs1, Prs2, Prs), append([enc(22, ChainL, 0, 0)|C1Is], AltIs, Is)), (malt([C], PL, At, FT, PC, MyL, L0, Is, PCout, L, [MyL-PC|Prs]) :- PCc is PC + 1, mone(C, PL, At, FT, PCc, L0, L, Prs, CIs), Is = [enc(24, 0, 0, 0)|CIs], length(Is, N), PCout is PC + N), (malt([C1, C2|Rest], PL, At, FT, PC, MyL, L0, Is, PCout, L, Prs) :- NextL = L0, L1 is L0 + 1, PCc is PC + 1, mone(C1, PL, At, FT, PCc, L1, L2, Prs1, C1Is), length(C1Is, N1), PC1 is PCc + N1, malt([C2|Rest], PL, At, FT, PC1, NextL, L2, RestIs, PCout, L, Prs2), append([MyL-PC|Prs1], Prs2, Prs), append([enc(23, NextL, 0, 0)|C1Is], RestIs, Is)), (mone(C, PL, At, FT, _, L0, L, Prs, Is) :- copy_term(C, C2), numbervars(C2, 0, _), mhb(C2, H, B), functor(H, _, Ar), mhead(1, Ar, H, At, [], In1, HIs), L = L0, Prs = [], (B == true -> append([enc(16, 0, 0, 0)|HIs], [enc(17, 0, 0, 0), enc(20, 0, 0, 0)], Is) ; mgl(B, Gs), mgoals(Gs, PL, At, FT, In1, _, GIs), append([enc(16, 0, 0, 0)|HIs], GIs, B0), append(B0, [enc(17, 0, 0, 0), enc(20, 0, 0, 0)], Is))), (mgl(G, L) :- (G =.. [F, X, Y], F == '','' -> mgl(X, L1), mgl(Y, L2), append(L1, L2, L) ; L = [G])), (mhead(I, Ar, H, At, In0, In, Is) :- (I > Ar -> In = In0, Is = [] ; arg(I, H, A), mharg(A, I, At, In0, In1, AI), I1 is I + 1, mhead(I1, Ar, H, At, In1, In, RIs), append(AI, RIs, Is))), (mharg(''$VAR''(N), I, _, In0, In1, [enc(T, Y, Ai, 0)]) :- Y is 48 + N, Ai is I - 1, (memberchk(N, In0) -> T = 2, In1 = In0 ; T = 1, In1 = [N|In0])), (mharg(V, I, _, In, In, [enc(0, V, O2, 0)]) :- integer(V), O2 is 65536 + I - 1), (mharg(A2, I, At, In, In, [enc(0, Idx, Ai, 1)]) :- atom(A2), Ai is I - 1, fidx(A2, At, Idx)), mgoals([], _, _, _, In, In, []), (mgoals([G|Gs], PL, At, FT, In0, In, Is) :- mgoal(G, PL, At, FT, In0, In1, GI), mgoals(Gs, PL, At, FT, In1, In, RI), append(GI, RI, Is)), (mgoal(G, PL, At, FT, In0, In2, Is) :- (G =.. [F, L1a, E1], F == is -> mopd(L1a, 0, At, In0, In1, IL), mexpr(E1, FT, In1, In2, IE), append(IL, IE, LE), append(LE, [enc(21, 0, 2, 0)], Is) ; G =.. [F2, L2a, R2a], F2 == ''='' -> mopd(L2a, 0, At, In0, Inb, IL2), mopd(R2a, 1, At, Inb, In2, IR2), append(IL2, IR2, LR), append(LR, [enc(21, 24, 2, 0)], Is) ; functor(G, P, Ar), mlook(P, Ar, PL, Lbl), mcargs(1, Ar, G, At, In0, In2, SIs), append(SIs, [enc(18, Lbl, Ar, 0)], Is))), (mcargs(I, Ar, G, At, In0, In, Is) :- (I > Ar -> In = In0, Is = [] ; arg(I, G, A), Ai is I - 1, mopd(A, Ai, At, In0, In1, AI), I1 is I + 1, mcargs(I1, Ar, G, At, In1, In, R), append(AI, R, Is))), (mopd(''$VAR''(N), Ai, _, In0, In1, [enc(T, Y, Ai, 0)]) :- Y is 48 + N, (memberchk(N, In0) -> T = 10, In1 = In0 ; T = 9, In1 = [N|In0])), (mopd(V, Ai, _, In, In, [enc(8, V, O2, 0)]) :- integer(V), O2 is 65536 + Ai), (mopd(A2, Ai, At, In, In, [enc(8, Idx, Ai, 1)]) :- atom(A2), fidx(A2, At, Idx)), (mexpr(E, FT, In0, In1, Is) :- functor(E, Op, 2), fidx(Op, FT, FI), arg(1, E, X), arg(2, E, Y), mset(X, In0, Ina, SX), mset(Y, Ina, In1, SY), append([enc(11, FI, 131073, 2)|SX], SY, Is)), (mset(''$VAR''(N), In0, In1, [enc(T, Y, 0, 0)]) :- Y is 48 + N, (memberchk(N, In0) -> T = 14, In1 = In0 ; T = 13, In1 = [N|In0])), (mset(V, In, In, [enc(15, V, 1, 0)]) :- integer(V)), (fidx(Op, [F|Fs], I) :- (Op == F -> I = 0 ; fidx(Op, Fs, I1), I is I1 + 1)), (mname(P, Ar, NC) :- atom_codes(P, PC), number_codes(Ar, AC), append(PC, [47|AC], NC)), (wzi(N, A0, A1) :- number_codes(N, Cs), append(Cs, [10|A1], A0)), (wza(X, A0, A1) :- atom_codes(X, Cs), append(Cs, [10|A1], A0)), (wzn(Cs, A0, A1) :- length(Cs, L), number_codes(L, LC), append(LC, [32|Mid], A0), append(Cs, [10|A1], Mid)), (wzsi(N, A0, A1) :- number_codes(N, Cs), append([32|Cs], A1, A0)), (wzs(EI, NC, LI, As, Fs, PCs, Is, Out) :- wzh(EI, NC, LI, Out, A6), length(As, NA), wzi(NA, A6, A7), wzat(As, A7, A8), length(Fs, NF), wzi(NF, A8, A9), wzat(Fs, A9, A10), wzp(PCs, A10, A11), wzc(Is, A11, A12), wzi(0, A12, [])), (wzh(EI, NC, LI, A0, Out) :- wza(''WAMO'', A0, A1), wzi(2, A1, A2), wzi(EI, A2, A3), wzi(1, A3, A4), wzn(NC, A4, A5), wzi(LI, A5, Out)), wzat([], A, A), (wzat([X|Xs], A0, A2) :- atom_codes(X, Cs), wzn(Cs, A0, A1), wzat(Xs, A1, A2)), (wzp(PCs, A0, A2) :- length(PCs, NL), wzi(NL, A0, A1), wzpr(PCs, A1, A2)), wzpr([], A, A), (wzpr([P|Ps], A0, A2) :- wzi(P, A0, A1), wzpr(Ps, A1, A2)), (wzc(Is, A0, A2) :- length(Is, NC2), wzi(NC2, A0, A1), wzcr(Is, A1, A2)), wzcr([], A, A), (wzcr([enc(T,O1,O2,Rl)|Is], A0, A2) :- wzr(T, O1, O2, Rl, A0, A1), wzcr(Is, A1, A2)), (wzr(T, O1, O2, Rl, A0, A5) :- number_codes(T, Tc), append(Tc, A1, A0), wzsi(O1, A1, A2), wzsi(O2, A2, A3), wzsi(Rl, A3, A4), A4 = [10|A5])]').

% The WALKERS source (fixpoint, next slice): the LAST codegen walkers on
% top of the front -- PC/label-threaded goal compilation with ITE codegen
% (mite mirrors f_goal's ITE clause: else/join labels, cut_ite, jump,
% Label-PC pairs, init-set intersection via minter), comparison guards
% (mcmp table), the builtin whitelist dispatch (mbi table), structure and
% list patterns with X-temp deferral on BOTH sides (musq/muarg/mdfr read
% side; mbl/mbs/mssq/msarg/mdfb build side), and general operands. The
% embedded golden exercises ITE + comparisons + atom operands, builtin
% calls with list-literal arguments, and structure head patterns with
% repeated variables -- compiled byte-identically to the production
% cgfull. The DEFERRAL paths (nested patterns / nested expression
% builds) are proven byte-exact in SWI (see the swi_walkers test) but
% blocked LOADED on runtime finding no. 12 (see PLAWK_SELFHOST.md).
fixpoint_walkers_source('[(main0(R) :- cm3(''[(cls(N, R2) :- (N >= 10 -> R2 = big ; R2 = small)), (sum2(Xs, R3) :- append(Xs, [7], Ys), sum_list(Ys, R3)), swap(p(X, Y), p(Y, X)), w(f(g(Z)), Z), (tot([A, B], R4) :- R4 is (A + B) * 2)]'', Cs), sum_list(Cs, S), length(Cs, L), R is S + L), (cm3(Src, Out) :- read_term_from_atom(Src, Cls), mwl(Cls, [], [], At, FT), mgrp(Cls, Gs), mlbl(Gs, 0, PL), length(Gs, NP0), mggs(Gs, PL, At, FT, 0, NP0, AllIs, EPCs, Prs), keysort(Prs, SPrs), pairs_values(SPrs, XPCs), append(EPCs, XPCs, PCs), Cls = [C1|_], mhb(C1, H1, _), functor(H1, P1, A1), mname(P1, A1, NC), wzs(0, NC, 0, At, FT, PCs, AllIs, Out)), (mhb(C, H, B) :- (C =.. [F, X, Y], F == '':-'' -> H = X, B = Y ; H = C, B = true)), mwl([], A, F, A, F), (mwl([T|Ts], A0, F0, A, F) :- mwt(T, A0, F0, A1, F1), mwl(Ts, A1, F1, A, F)), (mwt(T, A0, F0, A, F) :- (var(T) -> A = A0, F = F0 ; T == [] -> madd(''[]'', A0, A1), A = A1, F = F0 ; integer(T) -> A = A0, F = F0 ; atom(T) -> madd(T, A0, A1), A = A1, F = F0 ; T =.. [Fn|Args], mwf(Fn, F0, F1), mwl(Args, A0, F1, A, F))), (mwf(Fn, F0, F1) :- (mskip(Fn) -> F1 = F0 ; madd(Fn, F0, F1))), (mskip(F) :- memberchk(F, ['':-'', '','', '';'', ''->'', ''is'', ''='', ''>'', ''<'', ''>='', ''=<'', ''=:='', ''=\\='', ''=='', ''\\==''])), (madd(X, L0, L1) :- (memberchk(X, L0) -> L1 = L0 ; append(L0, [X], L1))), mgrp([], []), (mgrp([C|Cs], [g(P, Ar, [C|Same])|Gs]) :- mhb(C, H, _), functor(H, P, Ar), mtks(Cs, P, Ar, Same, Rest), mgrp(Rest, Gs)), mtks([], _, _, [], []), (mtks([C|Cs], P, Ar, Same, Rest) :- mhb(C, H, _), functor(H, P2, A2), (P2 == P, A2 =:= Ar -> Same = [C|S1], mtks(Cs, P, Ar, S1, Rest) ; Same = [], Rest = [C|Cs])), mlbl([], _, []), (mlbl([g(P, Ar, _)|Gs], I, [pl(P, Ar, I)|R]) :- I1 is I + 1, mlbl(Gs, I1, R)), (mlook(P, Ar, [pl(P2, A2, L2)|Ps], L) :- (P == P2, Ar =:= A2 -> L = L2 ; mlook(P, Ar, Ps, L))), mggs([], _, _, _, _, _, [], [], []), (mggs([g(_, _, Cls)|Gs], PL, At, FT, PC, L0, AllIs, [PC|EPCs], Prs) :- mprd(Cls, PL, At, FT, PC, L0, Is, PC1, L1, Prs1), mggs(Gs, PL, At, FT, PC1, L1, RestIs, EPCs, Prs2), append(Is, RestIs, AllIs), append(Prs1, Prs2, Prs)), (mprd([C], PL, At, FT, PC, L0, Is, PCout, L, Prs) :- mone(C, PL, At, FT, PC, L0, L, Prs, Is), length(Is, N), PCout is PC + N), (mprd([C1, C2|Rest], PL, At, FT, PC, L0, Is, PCout, L, Prs) :- ChainL = L0, L1 is L0 + 1, PCc is PC + 1, mone(C1, PL, At, FT, PCc, L1, L2, Prs1, C1Is), length(C1Is, N1), PC1 is PCc + N1, malt([C2|Rest], PL, At, FT, PC1, ChainL, L2, AltIs, PCout, L, Prs2), append(Prs1, Prs2, Prs), append([enc(22, ChainL, 0, 0)|C1Is], AltIs, Is)), (malt([C], PL, At, FT, PC, MyL, L0, Is, PCout, L, [MyL-PC|Prs]) :- PCc is PC + 1, mone(C, PL, At, FT, PCc, L0, L, Prs, CIs), Is = [enc(24, 0, 0, 0)|CIs], length(Is, N), PCout is PC + N), (malt([C1, C2|Rest], PL, At, FT, PC, MyL, L0, Is, PCout, L, Prs) :- NextL = L0, L1 is L0 + 1, PCc is PC + 1, mone(C1, PL, At, FT, PCc, L1, L2, Prs1, C1Is), length(C1Is, N1), PC1 is PCc + N1, malt([C2|Rest], PL, At, FT, PC1, NextL, L2, RestIs, PCout, L, Prs2), append([MyL-PC|Prs1], Prs2, Prs), append([enc(23, NextL, 0, 0)|C1Is], RestIs, Is)), (mone(C, PL, At, FT, StartPC, L0, L, Prs, Is) :- copy_term(C, C2), numbervars(C2, 0, _), mhb(C2, H, B), functor(H, _, Ar), mhead(1, Ar, H, At, FT, 16, [], In1, HIs), (B == true -> append([enc(16, 0, 0, 0)|HIs], [enc(17, 0, 0, 0), enc(20, 0, 0, 0)], Is), L = L0, Prs = [] ; mgl(B, Gs), length(HIs, NH), PC0 is StartPC + 1 + NH, mgoals(Gs, PL, At, FT, PC0, _, L0, L, [], Prs, In1, _, GIs), append([enc(16, 0, 0, 0)|HIs], GIs, B0), append(B0, [enc(17, 0, 0, 0), enc(20, 0, 0, 0)], Is))), (mgl(G, L) :- (G =.. [F, X, Y], F == '','' -> mgl(X, L1), mgl(Y, L2), append(L1, L2, L) ; L = [G])), (mhead(I, Ar, H, At, FT, Xt0, In0, In, Is) :- (I > Ar -> In = In0, Is = [] ; arg(I, H, A), mharg(A, I, At, FT, Xt0, Xt1, In0, In1, AI), I1 is I + 1, mhead(I1, Ar, H, At, FT, Xt1, In1, In, RIs), append(AI, RIs, Is))), (mharg(''$VAR''(N), I, _, _, Xt, Xt, In0, In1, [enc(T, Y, Ai, 0)]) :- Y is 48 + N, Ai is I - 1, (memberchk(N, In0) -> T = 2, In1 = In0 ; T = 1, In1 = [N|In0])), (mharg([], I, At, _, Xt, Xt, In, In, [enc(0, Idx, Ai, 1)]) :- Ai is I - 1, fidx(''[]'', At, Idx)), (mharg([H|T2], I, At, FT, Xt0, Xt, In0, In2, [enc(4, Ai, 0, 0)|Rest]) :- Ai is I - 1, musq([H, T2], At, Xt0, Xt1, In0, In1, UIs, [], Defs), mdfr(Defs, At, FT, Xt1, Xt, In1, In2, DIs), append(UIs, DIs, Rest)), (mharg(V, I, _, _, Xt, Xt, In, In, [enc(0, V, O2, 0)]) :- integer(V), O2 is 65536 + I - 1), (mharg(A2, I, At, _, Xt, Xt, In, In, [enc(0, Idx, Ai, 1)]) :- atom(A2), Ai is I - 1, fidx(A2, At, Idx)), (mharg(T2, I, At, FT, Xt0, Xt, In0, In2, [enc(3, FI, O2, 2)|Rest]) :- compound(T2), functor(T2, F, N), fidx(F, FT, FI), Ai is I - 1, O2 is N * 65536 + Ai, T2 =.. [_|Args], musq(Args, At, Xt0, Xt1, In0, In1, UIs, [], Defs), mdfr(Defs, At, FT, Xt1, Xt, In1, In2, DIs), append(UIs, DIs, Rest)), musq([], _, Xt, Xt, In, In, [], D, D), (musq([A|As], At, Xt0, Xt, In0, In, Is, D0, D) :- muarg(A, At, Xt0, Xt1, In0, In1, AIs, D0, D1), musq(As, At, Xt1, Xt, In1, In, RIs, D1, D), append(AIs, RIs, Is)), (muarg(''$VAR''(N), _, Xt, Xt, In0, In1, [enc(T, Y, 0, 0)], D, D) :- Y is 48 + N, (memberchk(N, In0) -> T = 6, In1 = In0 ; T = 5, In1 = [N|In0])), (muarg([], At, Xt, Xt, In, In, [enc(7, Idx, 0, 1)], D, D) :- fidx(''[]'', At, Idx)), (muarg(V, _, Xt, Xt, In, In, [enc(7, V, 65536, 0)], D, D) :- integer(V)), (muarg(A2, At, Xt, Xt, In, In, [enc(7, Idx, 0, 1)], D, D) :- atom(A2), fidx(A2, At, Idx)), (muarg(C, _, Xt0, Xt1, In, In, [enc(5, Xt0, 0, 0)], D0, D) :- compound(C), Xt1 is Xt0 + 1, append(D0, [d(C, Xt0)], D)), mdfr([], _, _, Xt, Xt, In, In, []), (mdfr([d(C, Reg)|Ds], At, FT, Xt0, Xt, In0, In, Is) :- (C =.. [F, H, T2], F == ''[|]'' -> musq([H, T2], At, Xt0, Xt1, In0, In1, UIs, [], Defs), CIs = [enc(4, Reg, 0, 0)|UIs] ; functor(C, F2, N), fidx(F2, FT, FI), O2 is N * 65536 + Reg, C =.. [_|Args], musq(Args, At, Xt0, Xt1, In0, In1, UIs, [], Defs), CIs = [enc(3, FI, O2, 2)|UIs]), mdfr(Defs, At, FT, Xt1, Xt2, In1, In2, DIs), mdfr(Ds, At, FT, Xt2, Xt, In2, In, RIs), append(CIs, DIs, A1s), append(A1s, RIs, Is)), mgoals([], _, _, _, PC, PC, L, L, Prs, Prs, In, In, []), (mgoals([G|Gs], PL, At, FT, PC0, PC, L0, L, Prs0, Prs, In0, In, Is) :- mgoal(G, PL, At, FT, PC0, PC1, L0, L1, Prs0, Prs1, In0, In1, GI), mgoals(Gs, PL, At, FT, PC1, PC, L1, L, Prs1, Prs, In1, In, RI), append(GI, RI, Is)), (mgoal(G, PL, At, FT, PC0, PC, L0, L, Prs0, Prs, In0, In, Is) :- (G =.. [F0, X0, E0], F0 == '';'', X0 =.. [F1, C0, T0], F1 == ''->'' -> mite(C0, T0, E0, PL, At, FT, PC0, PC, L0, L, Prs0, Prs, In0, In, Is) ; L = L0, Prs = Prs0, mgd(G, PL, At, FT, PC0, PC, In0, In, Is))), (mgd(G, PL, At, FT, PC0, PC, In0, In, Is) :- (G =.. [F, L1a, E1], F == ''is'' -> mopd(L1a, 0, At, FT, In0, In1, IL), mopd(E1, 1, At, FT, In1, In, IE), append(IL, IE, LE), append(LE, [enc(21, 0, 2, 0)], Is), length(Is, N), PC is PC0 + N ; G =.. [F2, L2a, R2a], F2 == ''='' -> mopd(L2a, 0, At, FT, In0, Inb, IL2), mopd(R2a, 1, At, FT, Inb, In, IR2), append(IL2, IR2, LR), append(LR, [enc(21, 24, 2, 0)], Is), length(Is, N2), PC is PC0 + N2 ; G =.. [F3, L3a, R3a], mcmp(F3, Id) -> mopd(L3a, 0, At, FT, In0, Inc, IL3), mopd(R3a, 1, At, FT, Inc, In, IR3), append(IL3, IR3, LR3), append(LR3, [enc(21, Id, 2, 0)], Is), length(Is, N3), PC is PC0 + N3 ; functor(G, P, Ar), mbi(P, Ar, Bid) -> mcargs(1, Ar, G, At, FT, In0, In, SIs), append(SIs, [enc(21, Bid, Ar, 0)], Is), length(Is, N4), PC is PC0 + N4 ; functor(G, P2, Ar2), mlook(P2, Ar2, PL, Lbl), mcargs(1, Ar2, G, At, FT, In0, In, SIs2), append(SIs2, [enc(18, Lbl, Ar2, 0)], Is), length(Is, N5), PC is PC0 + N5)), (mite(C0, T0, E0, PL, At, FT, PC0, PC, L0, L, Prs0, Prs, In0, In, Is) :- ElseL = L0, JoinL is L0 + 1, L1 is L0 + 2, mgl(C0, CGs), mgl(T0, TGs), mgl(E0, EGs), PC1 is PC0 + 1, mgoals(CGs, PL, At, FT, PC1, PC2, L1, L2, Prs0, Prs1, In0, InC, CondIs), PC3 is PC2 + 1, mgoals(TGs, PL, At, FT, PC3, PC4, L2, L3, Prs1, Prs2, InC, InT, ThenIs), ElsePC is PC4 + 1, PC5 is ElsePC + 1, mgoals(EGs, PL, At, FT, PC5, JoinPC, L3, L, Prs2, Prs3, In0, InE, ElseIs), PC = JoinPC, append(Prs3, [ElseL-ElsePC, JoinL-JoinPC], Prs), minter(InT, InE, In), append([enc(22, ElseL, 0, 0)|CondIs], [enc(31, 0, 0, 0)|ThenIs], Front), append(Front, [enc(32, JoinL, 0, 0), enc(24, 0, 0, 0)|ElseIs], Is)), minter([], _, []), (minter([X|Xs], Ys, Out) :- (memberchk(X, Ys) -> Out = [X|R], minter(Xs, Ys, R) ; minter(Xs, Ys, Out))), mcmp(''>'', 1), mcmp(''<'', 2), mcmp(''>='', 3), mcmp(''=<'', 4), mcmp(''=:='', 5), mcmp(''=\\='', 6), mcmp(''=='', 7), mcmp(''\\=='', 21), mbi(functor, 3, 26), mbi(arg, 3, 27), mbi(''=..'', 2, 28), mbi(copy_term, 2, 29), mbi(atom_codes, 2, 36), mbi(number_codes, 2, 43), mbi(length, 2, 31), mbi(append, 3, 55), mbi(memberchk, 2, 56), mbi(sum_list, 2, 59), mbi(keysort, 2, 80), mbi(pairs_values, 2, 71), mbi(numbervars, 3, 124), mbi(read_term_from_atom, 2, 174), mbi(var, 1, 18), mbi(atom, 1, 13), mbi(integer, 1, 14), mbi(compound, 1, 17), (mcargs(I, Ar, G, At, FT, In0, In, Is) :- (I > Ar -> In = In0, Is = [] ; arg(I, G, A), Ai is I - 1, mopd(A, Ai, At, FT, In0, In1, AI), I1 is I + 1, mcargs(I1, Ar, G, At, FT, In1, In, R), append(AI, R, Is))), (mopd(''$VAR''(N), Ai, _, _, In0, In1, [enc(T, Y, Ai, 0)]) :- Y is 48 + N, (memberchk(N, In0) -> T = 10, In1 = In0 ; T = 9, In1 = [N|In0])), (mopd(V, Ai, _, _, In, In, [enc(8, V, O2, 0)]) :- integer(V), O2 is 65536 + Ai), (mopd([], Ai, At, _, In, In, [enc(8, Idx, Ai, 1)]) :- fidx(''[]'', At, Idx)), (mopd([H|T2], Ai, At, FT, In0, In1, Is) :- mbl([H|T2], Ai, first, At, FT, 16, _, In0, In1, Is)), (mopd(A2, Ai, At, _, In, In, [enc(8, Idx, Ai, 1)]) :- atom(A2), fidx(A2, At, Idx)), (mopd(T2, Ai, At, FT, In0, In1, Is) :- compound(T2), mbs(T2, Ai, At, FT, 16, _, In0, In1, Is)), (mbl(L0v, TR, Mode, At, FT, Xt0, Xt, In0, In2, Is) :- L0v =.. [_, E, Rest], (Mode == first -> Head = [enc(12, TR, 0, 0)] ; fidx(''[|]'', FT, CI), O2 is 131072 + TR, Head = [enc(11, CI, O2, 2)]), msarg(E, At, Xt0, Xta, In0, In1, SE, [], DefsE), (Rest == [] -> fidx(''[]'', At, NI), Tail = [enc(15, NI, 0, 1)], In1b = In1, Xtb = Xta, RestIs = [] ; Rest = ''$VAR''(_) -> msarg(Rest, At, Xta, Xtb, In1, In1b, Tail, [], []), RestIs = [] ; Tail = [enc(13, Xta, 0, 0)], Xtc is Xta + 1, mbl(Rest, Xta, nested, At, FT, Xtc, Xtb, In1, In1b, RestIs)), mdfb(DefsE, At, FT, Xtb, Xt, In1b, In2, DIs), append(Head, SE, HS), append(HS, Tail, HST), append(HST, RestIs, HSTR), append(HSTR, DIs, Is)), (mbs(T2, TR, At, FT, Xt0, Xt, In0, In2, [enc(11, FI, O2, 2)|Rest]) :- functor(T2, F, N), fidx(F, FT, FI), O2 is N * 65536 + TR, T2 =.. [_|Args], mssq(Args, At, Xt0, Xt1, In0, In1, SIs, [], Defs), mdfb(Defs, At, FT, Xt1, Xt, In1, In2, DIs), append(SIs, DIs, Rest)), mssq([], _, Xt, Xt, In, In, [], D, D), (mssq([A|As], At, Xt0, Xt, In0, In, Is, D0, D) :- msarg(A, At, Xt0, Xt1, In0, In1, AIs, D0, D1), mssq(As, At, Xt1, Xt, In1, In, RIs, D1, D), append(AIs, RIs, Is)), (msarg(''$VAR''(N), _, Xt, Xt, In0, In1, [enc(T, Y, 0, 0)], D, D) :- Y is 48 + N, (memberchk(N, In0) -> T = 14, In1 = In0 ; T = 13, In1 = [N|In0])), (msarg([], At, Xt, Xt, In, In, [enc(15, Idx, 0, 1)], D, D) :- fidx(''[]'', At, Idx)), (msarg(V, _, Xt, Xt, In, In, [enc(15, V, 1, 0)], D, D) :- integer(V)), (msarg(A2, At, Xt, Xt, In, In, [enc(15, Idx, 0, 1)], D, D) :- atom(A2), fidx(A2, At, Idx)), (msarg(C, _, Xt0, Xt1, In, In, [enc(13, Xt0, 0, 0)], D0, D) :- compound(C), Xt1 is Xt0 + 1, append(D0, [d(C, Xt0)], D)), mdfb([], _, _, Xt, Xt, In, In, []), (mdfb([d(C, Reg)|Ds], At, FT, Xt0, Xt, In0, In, Is) :- (C =.. [F, _, _], F == ''[|]'' -> mbl(C, Reg, nested, At, FT, Xt0, Xt1, In0, In1, CIs) ; mbs(C, Reg, At, FT, Xt0, Xt1, In0, In1, CIs)), mdfb(Ds, At, FT, Xt1, Xt, In1, In, RIs), append(CIs, RIs, Is)), (fidx(Op, [F|Fs], I) :- (Op == F -> I = 0 ; fidx(Op, Fs, I1), I is I1 + 1)), (mname(P, Ar, NC) :- atom_codes(P, PC), number_codes(Ar, AC), append(PC, [47|AC], NC)), (wzi(N, A0, A1) :- number_codes(N, Cs), append(Cs, [10|A1], A0)), (wza(X, A0, A1) :- atom_codes(X, Cs), append(Cs, [10|A1], A0)), (wzn(Cs, A0, A1) :- length(Cs, L), number_codes(L, LC), append(LC, [32|Mid], A0), append(Cs, [10|A1], Mid)), (wzsi(N, A0, A1) :- number_codes(N, Cs), append([32|Cs], A1, A0)), (wzs(EI, NC, LI, As, Fs, PCs, Is, Out) :- wzh(EI, NC, LI, Out, A6), length(As, NA), wzi(NA, A6, A7), wzat(As, A7, A8), length(Fs, NF), wzi(NF, A8, A9), wzat(Fs, A9, A10), wzp(PCs, A10, A11), wzc(Is, A11, A12), wzi(0, A12, [])), (wzh(EI, NC, LI, A0, Out) :- wza(''WAMO'', A0, A1), wzi(2, A1, A2), wzi(EI, A2, A3), wzi(1, A3, A4), wzn(NC, A4, A5), wzi(LI, A5, Out)), wzat([], A, A), (wzat([X|Xs], A0, A2) :- atom_codes(X, Cs), wzn(Cs, A0, A1), wzat(Xs, A1, A2)), (wzp(PCs, A0, A2) :- length(PCs, NL), wzi(NL, A0, A1), wzpr(PCs, A1, A2)), wzpr([], A, A), (wzpr([P|Ps], A0, A2) :- wzi(P, A0, A1), wzpr(Ps, A1, A2)), (wzc(Is, A0, A2) :- length(Is, NC2), wzi(NC2, A0, A1), wzcr(Is, A1, A2)), wzcr([], A, A), (wzcr([enc(T,O1,O2,Rl)|Is], A0, A2) :- wzr(T, O1, O2, Rl, A0, A1), wzcr(Is, A1, A2)), (wzr(T, O1, O2, Rl, A0, A5) :- number_codes(T, Tc), append(Tc, A1, A0), wzsi(O1, A1, A2), wzsi(O2, A2, A3), wzsi(Rl, A3, A4), A4 = [10|A5])]').

% The finding-no.-12 minimal-pair source (see the regression test).
finding12_pair_source('[(main0(R) :- read_term_from_atom(''[w(f(g(Z)), Z)]'', Cls), mgrp(Cls, Gs), mggs(Gs, 0, AllIs), length(AllIs, R)), (mhb(C, H, B) :- (C =.. [F, X, Y], F == '':-'' -> H = X, B = Y ; H = C, B = true)), mgrp([], []), (mgrp([C|Cs], [g(P, Ar, [C|Same])|Gs]) :- mhb(C, H, _), functor(H, P, Ar), mtks(Cs, P, Ar, Same, Rest), mgrp(Rest, Gs)), mtks([], _, _, [], []), (mtks([C|Cs], P, Ar, Same, Rest) :- mhb(C, H, _), functor(H, P2, A2), (P2 == P, A2 =:= Ar -> Same = [C|S1], mtks(Cs, P, Ar, S1, Rest) ; Same = [], Rest = [C|Cs])), mggs([], _, []), (mggs([g(_, _, [C])|Gs], PC, AllIs) :- mone(C, Is), length(Is, N), PC1 is PC + N, mggs(Gs, PC1, RestIs), append(Is, RestIs, AllIs)), (mone(C, Is) :- copy_term(C, C2), numbervars(C2, 0, _), mhb(C2, H, _), functor(H, _, Ar), mhead(1, Ar, H, [w, f, g], 16, [], _, HIs), append(HIs, [enc(20, 0, 0, 0)], Is)), (mhead(I, Ar, H, FT, Xt0, In0, In, Is) :- (I > Ar -> In = In0, Is = [] ; arg(I, H, A), mharg(A, I, FT, Xt0, Xt1, In0, In1, AI), I1 is I + 1, mhead(I1, Ar, H, FT, Xt1, In1, In, RIs), append(AI, RIs, Is))), (mharg(''$VAR''(N), I, _, Xt, Xt, In0, In1, [enc(T, Y, Ai, 0)]) :- Y is 48 + N, Ai is I - 1, (memberchk(N, In0) -> T = 2, In1 = In0 ; T = 1, In1 = [N|In0])), (mharg(V, I, _, Xt, Xt, In, In, [enc(0, V, O2, 0)]) :- integer(V), O2 is 65536 + I - 1), (mharg(T2, I, FT, Xt0, Xt, In0, In2, [enc(3, FI, O2, 2)|Rest]) :- compound(T2), functor(T2, F, N), fidx(F, FT, FI), Ai is I - 1, O2 is N * 65536 + Ai, T2 =.. [_|Args], musq(Args, Xt0, Xt1, In0, In1, UIs, [], Defs), mdfr(Defs, FT, Xt1, Xt, In1, In2, DIs), append(UIs, DIs, Rest)), musq([], Xt, Xt, In, In, [], D, D), (musq([A|As], Xt0, Xt, In0, In, Is, D0, D) :- muarg(A, Xt0, Xt1, In0, In1, AIs, D0, D1), musq(As, Xt1, Xt, In1, In, RIs, D1, D), append(AIs, RIs, Is)), (muarg(''$VAR''(N), Xt, Xt, In0, In1, [enc(T, Y, 0, 0)], D, D) :- Y is 48 + N, (memberchk(N, In0) -> T = 6, In1 = In0 ; T = 5, In1 = [N|In0])), (muarg(C, Xt0, Xt1, In, In, [enc(5, Xt0, 0, 0)], D0, D) :- compound(C), Xt1 is Xt0 + 1, append(D0, [d(C, Xt0)], D)), mdfr([], _, Xt, Xt, In, In, []), (mdfr([d(C, Reg)|Ds], FT, Xt0, Xt, In0, In, Is) :- (C =.. [F, H, T2], F == ''[|]'' -> musq([H, T2], Xt0, Xt1, In0, In1, UIs, [], Defs), CIs = [enc(4, Reg, 0, 0)|UIs] ; functor(C, F2, N), fidx(F2, FT, FI), O2 is N * 65536 + Reg, C =.. [_|Args], musq(Args, Xt0, Xt1, In0, In1, UIs, [], Defs), CIs = [enc(3, FI, O2, 2)|UIs]), mdfr(Defs, FT, Xt1, Xt2, In1, In2, DIs), mdfr(Ds, FT, Xt2, Xt, In2, In, RIs), append(CIs, DIs, A1s), append(A1s, RIs, Is)), (fidx(Op, [F|Fs], I) :- (Op == F -> I = 0 ; fidx(Op, Fs, I1), I is I1 + 1))]').

% Register-file ceiling regression: manyperm/1 has 20 variables all live
% across the mp_barrier call, so the compiler assigns them Y1..Y20. Before
% the register file was enlarged from [64 x %Value] to [128 x %Value], Y17+
% mapped to array index 64+ and wrote past the register array into adjacent
% %WamState fields -> memory corruption / segfault in loaded objects (and
% AOT). With the fix Y1..Y48 (48..95) have real backing; this must run to
% 1+2+...+20 = 210.
manyperm(R) :-
    N1 is 1, N2 is 2, N3 is 3, N4 is 4, N5 is 5,
    N6 is 6, N7 is 7, N8 is 8, N9 is 9, N10 is 10,
    N11 is 11, N12 is 12, N13 is 13, N14 is 14, N15 is 15,
    N16 is 16, N17 is 17, N18 is 18, N19 is 19, N20 is 20,
    mp_barrier,
    R is N1+N2+N3+N4+N5+N6+N7+N8+N9+N10
       + N11+N12+N13+N14+N15+N16+N17+N18+N19+N20.
mp_barrier.

% Campaign finding no. 9 regression: variable identity through get_value and
% the append/3 seed. Two runtime paths used the FULL deref on a value that
% could be an unbound variable, collapsing Ref-to-unbound-cell into the
% detached Unbound sentinel (no cell address) and silently severing the link:
%   (a) get_value var-var -- knot9(A, A) called with two unbound args left
%       them UNLINKED (the head "unification" was a no-op that still
%       succeeded), so a later binding through one side was invisible
%       through the other;
%   (b) builtin append(Cs, T, L) with T an unbound BARE variable seeded the
%       result tail with the collapsed sentinel instead of Ref{cell of T}.
% Both severed the difference-list serializer of the self-host fixpoint.
% Here: L = [1,2,3|H], H knotted to T via get_value, T extended with 4 and
% closed via the bare-var append seed; sum_list/length see the COMPLETE
% list only if both links held. R must be 10*100 + 4 = 1004 (on the broken
% runtime the entry FAILS: sum_list reaches an unbound tail).
dlknot(R) :-
    append([1,2], [3|H], L),
    knot9(H, T),
    em9(4, T, T2),
    T2 = [],
    sum_list(L, S), length(L, N),
    R is S*100 + N.
knot9(A, A).
em9(V, A0, A1) :- append([V], A1, A0).

% Loaded clause indexing: swkey/2 is an atom-keyed multi-clause predicate
% large enough for the tier-2 compiler to emit switch_on_constant --
% including writeq-QUOTED keys ('=:=', '\==') whose table entries only
% match if the writer unquotes them (switch_entry_unquote/2; a quoted key
% interns its quote characters into the atom name and NEVER matches).
% swmain probes: two quoted keys, a bare key, a MISSING key inside an
% if-then-else (a strict switch miss backtracks -- the call must fail
% cleanly into the else branch), and an UNBOUND first argument (the
% switch skips and the try_me_else chain runs -- first clause binds).
swkey(functor, 26).
swkey(arg, 27).
swkey('=:=', 5).
swkey('\\==', 21).
swkey(append, 55).
swkey(length, 31).
swkey(sort, 81).
swkey(between, 23).
swmain(R) :-
    swkey('=:=', A),
    swkey('\\==', B),
    swkey(append, C),
    ( swkey(nosuchkey, _) -> D = 1 ; D = 2 ),
    ( swkey(K0, 26), K0 == functor -> E = 3 ; E = 4 ),
    R is (((A * 100 + B) * 100 + C) * 10 + D) * 10 + E.  % 5215523

% Milestone 3b-db PR 3: rule bodies. assertz((H :- B)) stores a rule; calling
% its head runs the body (deterministic first solution). Bodies handle ,/2,
% builtins (is/2, comparisons, =/2), and predicate calls; head<->body variable
% sharing is preserved by a var-index durable copy instantiated per call.
:- dynamic rbdbl/2, rbinc/2, rbadd2/2, rbclamp/2, rbbase/1, rbcompute/1.
rb_double(R) :- assertz((rbdbl(X,Y) :- Y is X*2)), rbdbl(21, R).              % 42 (builtin body, shared X)
rb_chain(R)  :- assertz((rbinc(X,Y) :- Y is X+1)),
                assertz((rbadd2(X,Z) :- rbinc(X,Y), rbinc(Y,Z))),
                rbadd2(40, R).                                               % 42 (rule calls rule, var chain)
rb_clamp(R)  :- assertz((rbclamp(X,X) :- X =< 100)), rbclamp(42, R).         % 42 (head var in body guard)
rb_mix(R)    :- assertz(rbbase(10)),
                assertz((rbcompute(Y) :- rbbase(B), Y is B+32)),
                rbcompute(R).                                                % 42 (body: fact call + builtin)

% Rule-body control constructs (follow-up): if-then-else, disjunction, and
% negation-as-failure, all within the deterministic first-solution model.
:- dynamic rcsgn/2, rcpk/1, rcdj/1, rcnz/2.
rc_ite_t(R) :- assertz((rcsgn(N,X) :- (N > 0 -> X = 1 ; X = 0))), rcsgn(7, R).   % 1 (cond true)
rc_ite_f(R) :- assertz((rcsgn(N,X) :- (N > 0 -> X = 1 ; X = 0))), rcsgn(-7, R).  % 0 (cond false)
rc_disj1(R) :- assertz((rcpk(X) :- (X = 11 ; X = 22))), rcpk(R).                 % 11 (first branch)
rc_disj2(R) :- assertz((rcdj(X) :- (fail ; X = 22))), rcdj(R).                   % 22 (second branch)
rc_naf1(R)  :- assertz((rcnz(N,X) :- (\+ (N =:= 0) -> X = 1 ; X = 0))), rcnz(5, R). % 1
rc_naf0(R)  :- assertz((rcnz(N,X) :- (\+ (N =:= 0) -> X = 1 ; X = 0))), rcnz(0, R). % 0

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(wam_object).

% The writer emits a well-formed .wamo byte stream: "WAMO" magic, version
% 2, and the section counts we can recover by re-parsing the tokens.
test(encode_produces_well_formed_stream) :-
    wam_object_encode([user:answer/1, user:sum3/3],
        [wamo_entry(answer/1)], Codes),
    string_codes(Text, Codes),
    sub_string(Text, 0, 4, _, "WAMO"),
    split_string(Text, "\n \t", "\n \t", Parts0),
    exclude(==(""), Parts0, Parts),
    % tokens: WAMO 2 <entry> <natoms> ... ; version must be "2", entry "0"
    Parts = ["WAMO", "2", "0" | _],
    !.

% Float constants now compile in put/set_constant: the object carries the
% decimal text (in the C-string table) and the loader strtod's it. uses_float
% (X is 1.5 + 2.5) reaches both floats through set_constant.
test(float_constant_compiles) :-
    wam_object_encode([user:uses_float/1], [wamo_entry(uses_float/1)], Codes),
    string_codes(Text, Codes),
    sub_string(Text, 0, 4, _, "WAMO"),
    sub_string(Text, _, _, _, "1.5"),
    sub_string(Text, _, _, _, "2.5"),
    !.

% A requested entry predicate with no label is a hard error.
test(missing_entry_is_rejected,
        [throws(error(wamo_entry_not_found(_), _))]) :-
    wam_object_encode([user:answer/1, user:sum3/3],
        [wamo_entry(nonesuch/2)], _).

% Full round trip: write a grammar, build a host binary that carries the
% loader, load the object at runtime and run it. The host never saw the
% grammar at compile time.
test(host_loads_and_runs_object_at_runtime,
        [condition(clang_available)]) :-
    obj_dir(Dir),
    directory_file_path(Dir, 'grammar.wamo', Wamo),
    write_wam_object([user:answer/1, user:sum3/3],
        [wamo_entry(answer/1)], Wamo),
    build_host(Dir, Host),
    run_host(Host, Wamo, Out, 0),
    assertion(Out == "119\n"),
    !.

% Same host binary, a different grammar object -> different answer, with no
% host rebuild. This is the point of runtime-loadable objects.
test(same_host_runs_a_swapped_grammar,
        [condition(clang_available)]) :-
    obj_dir(Dir),
    directory_file_path(Dir, 'host_bin', Host),
    ( exists_file(Host) -> true ; build_host(Dir, Host) ),
    directory_file_path(Dir, 'grammar2.wamo', Wamo2),
    write_wam_object([user:answer_swapped/1, user:sum3/3],
        [wamo_entry(answer_swapped/1)], Wamo2),
    run_host(Host, Wamo2, Out, 0),
    assertion(Out == "1020\n"),
    !.

% The loader has an in-memory entry point: @wam_object_load_bytes parses a
% buffer directly, with no file. Embed a grammar's .wamo bytes as an LLVM
% constant in the host and load from memory -- this is the primitive that
% lets a grammar travel as a value rather than a path.
test(load_bytes_from_memory, [condition(clang_available)]) :-
    obj_dir(Dir),
    directory_file_path(Dir, 'embed.wamo', Wamo),
    write_wam_object([user:answer/1, user:sum3/3],
        [wamo_entry(answer/1)], Wamo),
    read_file_to_bytes(Wamo, Bytes),
    length(Bytes, NBytes),
    build_embed_host(Dir, Bytes, NBytes, Host),
    run_embed_host(Host, Out),
    assertion(Out == "119\n"),
    !.

% Multi-entry object: one .wamo exposes two named entries (answer/1 and
% answer_swapped/1, both over the shared sum3/3). The writer emits a
% name->label-index table; the encoded stream carries both names.
test(multi_entry_encodes_name_table) :-
    wam_object_encode([user:answer/1, user:answer_swapped/1, user:sum3/3],
        [wamo_entries([answer/1, answer_swapped/1])], Codes),
    string_codes(Text, Codes),
    split_string(Text, "\n \t", "\n \t", Parts0),
    exclude(==(""), Parts0, Parts),
    % WAMO 2 <default-entry> <E=2> 8 answer/1 <idx> 16 answer_swapped/1 <idx> ...
    Parts = ["WAMO", "2", _Default, "2" | _],
    memberchk("answer/1", Parts),
    memberchk("answer_swapped/1", Parts),
    !.

% Full round trip through the loader's name resolution: build one host that
% loads a two-entry object, resolves each entry by name to a label index
% (@wam_object_entry_index), turns each into a PC (@wam_label_pc) and calls
% it. Distinct names -> distinct results from the SAME object.
test(host_resolves_named_entries,
        [condition(clang_available)]) :-
    obj_dir(Dir),
    directory_file_path(Dir, 'multi.wamo', Wamo),
    write_wam_object([user:answer/1, user:answer_swapped/1, user:sum3/3],
        [wamo_entries([answer/1, answer_swapped/1])], Wamo),
    build_multi_host(Dir, Host),
    run_host(Host, Wamo, Out, 0),
    assertion(Out == "119\n1020\n"),
    !.

% A name that no entry exposes resolves to -1; the host exits with the
% resolve-fail code rather than calling a bogus PC.
test(unknown_entry_name_resolves_to_minus_one,
        [condition(clang_available)]) :-
    obj_dir(Dir),
    directory_file_path(Dir, 'multi.wamo', Wamo),
    ( exists_file(Wamo) -> true
    ; write_wam_object([user:answer/1, user:answer_swapped/1, user:sum3/3],
          [wamo_entries([answer/1, answer_swapped/1])], Wamo) ),
    directory_file_path(Dir, 'multi_miss_host', Host),
    ( exists_file(Host) -> true ; build_multi_miss_host(Dir, Host) ),
    run_host(Host, Wamo, _Out, 22),
    !.

% A grammar can return a Compound record; @wam_object_call_record
% deserializes its args into typed slots. makerec/1 returns
% rec2f(10, 10.5): field 0 as i64 (typecode 0) -> 10, field 1 as f64
% (typecode 1) -> 10.5. The compound + its arg cells are read before the
% arena rewind.
test(struct_return_deserializes_fields,
        [condition(clang_available)]) :-
    obj_dir(Dir),
    directory_file_path(Dir, 'rec.wamo', Wamo),
    write_wam_object([user:makerec/1], [wamo_entry(makerec/1)], Wamo),
    build_record_host(Dir, Host),
    run_host(Host, Wamo, Out, 0),
    assertion(Out == "10\n10.5\n"),
    !.

% A string field (typecode 2): makerecs/1 returns rs(7, hello). The record
% call writes field 0's i64 (7) into a slot and field 1's atom string into
% (out_slots[1] = ptr, out_lens[1] = 5). The pointer is into the persistent
% atom table, so it prints (%.*s) after the arena rewind.
test(struct_return_string_field,
        [condition(clang_available)]) :-
    obj_dir(Dir),
    directory_file_path(Dir, 'recs.wamo', Wamo),
    write_wam_object([user:makerecs/1], [wamo_entry(makerecs/1)], Wamo),
    build_record_str_host(Dir, Host),
    run_host(Host, Wamo, Out, 0),
    assertion(Out == "7\nhello\n"),
    !.

% Assoc-table variant: tally/1 returns [1-100, 2-200, 3-30];
% @wam_object_call_assoc inserts each pair into a fresh i64 table, and the
% host reads keys 1,2,3 back with @wam_assoc_i64_get.
test(assoc_return_populates_table,
        [condition(clang_available)]) :-
    obj_dir(Dir),
    directory_file_path(Dir, 'tally.wamo', Wamo),
    write_wam_object([user:tally/1], [wamo_entry(tally/1)], Wamo),
    build_record_assoc_host(Dir, Host),
    run_host(Host, Wamo, Out, 0),
    assertion(Out == "100\n200\n30\n"),
    !.

% Atom-first-argument indexing (switch_on_constant) is now loadable: the
% loader nops the switch and runs the clause chain unindexed. pick/1 over
% the atom-keyed coltab/2 returns 2 -- the first subset-expansion step for
% the eval bootstrap (item 5).
test(switch_on_constant_loads_and_runs,
        [condition(clang_available)]) :-
    obj_dir(Dir),
    directory_file_path(Dir, 'pick.wamo', Wamo),
    write_wam_object([user:pick/1, user:coltab/2], [wamo_entry(pick/1)], Wamo),
    directory_file_path(Dir, 'host_bin', Host),
    ( exists_file(Host) -> true ; build_host(Dir, Host) ),
    run_host(Host, Wamo, Out, 0),
    assertion(Out == "2\n"),
    !.

% get_structure functor-check regression: dispatch over a compound keyed by
% first-argument functor. Before get_structure verified the functor, calling
% dp_sel(three(7),R) mis-matched the first clause (one/1) and returned 8; now
% it correctly falls through to three/1 -> 42. Also validates that a tagged-
% union walker (a compiler's natural AST/token representation) dispatches
% correctly in loaded objects.
test(get_structure_functor_dispatch,
        [condition(clang_available)]) :-
    obj_dir(Dir),
    directory_file_path(Dir, 'dpgo.wamo', Wamo),
    write_wam_object([user:dp_go/1, user:dp_sel/2], [wamo_entry(dp_go/1)], Wamo),
    directory_file_path(Dir, 'host_bin', Host),
    ( exists_file(Host) -> true ; build_host(Dir, Host) ),
    run_host(Host, Wamo, Out, 0),
    assertion(Out == "42\n"),
    !.

% A loaded object meta-calls (call/N) one of its own predicates, the goal
% built at runtime. Atom goal: metaatom builds `mfoo` and calls it -> 100.
% Dispatch runs through the object's own meta-call table (fields 25/26).
test(meta_call_atom_goal_in_object,
        [condition(clang_available)]) :-
    obj_dir(Dir),
    directory_file_path(Dir, 'metaatom.wamo', Wamo),
    write_wam_object([user:metaatom/1, user:mfoo/1], [wamo_entry(metaatom/1)], Wamo),
    directory_file_path(Dir, 'host_bin', Host),
    ( exists_file(Host) -> true ; build_host(Dir, Host) ),
    run_host(Host, Wamo, Out, 0),
    assertion(Out == "100\n"),
    !.

% Compound goal: metacomp builds `maddk(10)` and calls it with one extra
% argument -> maddk(10, R) -> 42. Exercises functor-pointer matching against
% the object's own functor copies in the meta-call table.
test(meta_call_compound_goal_in_object,
        [condition(clang_available)]) :-
    obj_dir(Dir),
    directory_file_path(Dir, 'metacomp.wamo', Wamo),
    write_wam_object([user:metacomp/1, user:maddk/2], [wamo_entry(metacomp/1)], Wamo),
    directory_file_path(Dir, 'host_bin', Host),
    ( exists_file(Host) -> true ; build_host(Dir, Host) ),
    run_host(Host, Wamo, Out, 0),
    assertion(Out == "42\n"),
    !.

% findall over a user predicate loads and runs: begin_aggregate/end_aggregate
% are now in the loadable subset. collectsum sums agnum/1's four solutions.
test(findall_in_object,
        [condition(clang_available)]) :-
    obj_dir(Dir),
    directory_file_path(Dir, 'collectsum.wamo', Wamo),
    write_wam_object([user:collectsum/1, user:agnum/1], [wamo_entry(collectsum/1)], Wamo),
    directory_file_path(Dir, 'host_bin', Host),
    ( exists_file(Host) -> true ; build_host(Dir, Host) ),
    run_host(Host, Wamo, Out, 0),
    assertion(Out == "70\n"),
    !.

% setof (sort + dedup) and bagof (keep dups) load and run: the .wamo writer
% lowers them through the same aggregate path as findall (inline_bagof_setof).
test(setof_and_bagof_in_object,
        [condition(clang_available)]) :-
    obj_dir(Dir),
    directory_file_path(Dir, 'setcard.wamo', WSet),
    directory_file_path(Dir, 'bagcard.wamo', WBag),
    write_wam_object([user:setcard/1, user:agnum/1], [wamo_entry(setcard/1)], WSet),
    write_wam_object([user:bagcard/1, user:agnum/1], [wamo_entry(bagcard/1)], WBag),
    directory_file_path(Dir, 'host_bin', Host),
    ( exists_file(Host) -> true ; build_host(Dir, Host) ),
    run_host(Host, WSet, SetOut, 0),
    assertion(SetOut == "3\n"),
    run_host(Host, WBag, BagOut, 0),
    assertion(BagOut == "4\n"),
    !.

% Regression for the zero-solution aggregate crash: findall over a goal that
% never succeeds must finalize to the empty list and return, not loop forever.
test(empty_aggregate_terminates,
        [condition(clang_available)]) :-
    obj_dir(Dir),
    directory_file_path(Dir, 'emptycount.wamo', Wamo),
    write_wam_object([user:emptycount/1, user:noagg/1], [wamo_entry(emptycount/1)], Wamo),
    directory_file_path(Dir, 'host_bin', Host),
    ( exists_file(Host) -> true ; build_host(Dir, Host) ),
    run_host(Host, Wamo, Out, 0),
    assertion(Out == "0\n"),
    !.

% term_to_atom in a loaded object renders pt(3,[x,y,z]) -> 13 chars. Verifies
% the byte-based cons detection works with the object's own functor copies.
test(term_to_atom_in_object,
        [condition(clang_available)]) :-
    obj_dir(Dir),
    directory_file_path(Dir, 'ttalen.wamo', Wamo),
    write_wam_object([user:ttalen/1], [wamo_entry(ttalen/1)], Wamo),
    directory_file_path(Dir, 'host_bin', Host),
    ( exists_file(Host) -> true ; build_host(Dir, Host) ),
    run_host(Host, Wamo, Out, 0),
    assertion(Out == "13\n"),
    !.

% read_term_from_atom in a loaded object: parse "40" at runtime, add 2 -> 42.
% Proves the parse yields a genuine Integer usable in arithmetic.
test(read_term_from_atom_in_object,
        [condition(clang_available)]) :-
    obj_dir(Dir),
    directory_file_path(Dir, 'readint.wamo', Wamo),
    write_wam_object([user:readint_obj/1], [wamo_entry(readint_obj/1)], Wamo),
    directory_file_path(Dir, 'host_bin', Host),
    ( exists_file(Host) -> true ; build_host(Dir, Host) ),
    run_host(Host, Wamo, Out, 0),
    assertion(Out == "42\n"),
    !.

% Compound reader in a loaded object: a parsed compound unifies against a
% source literal (via @wam_functor_eq), decomposes, and computes. Also covers
% nested compounds and list parsing.
test(read_compound_in_object,
        [condition(clang_available)]) :-
    obj_dir(Dir),
    directory_file_path(Dir, 'readcompound.wamo', W1),
    directory_file_path(Dir, 'readnested.wamo', W2),
    directory_file_path(Dir, 'readlist.wamo', W3),
    write_wam_object([user:readcompound_obj/1], [wamo_entry(readcompound_obj/1)], W1),
    write_wam_object([user:readnested_obj/1], [wamo_entry(readnested_obj/1)], W2),
    write_wam_object([user:readlist_obj/1], [wamo_entry(readlist_obj/1)], W3),
    directory_file_path(Dir, 'host_bin', Host),
    ( exists_file(Host) -> true ; build_host(Dir, Host) ),
    run_host(Host, W1, O1, 0), assertion(O1 == "34\n"),
    run_host(Host, W2, O2, 0), assertion(O2 == "12\n"),
    run_host(Host, W3, O3, 0), assertion(O3 == "60\n"),
    !.

% Operator reader in a loaded object: infix arithmetic parsed with precedence
% and evaluated by is/2. Precedence (1+2*3=7), parens ((1+2)*3=9), and
% left-assoc + negatives + spaces (100 - 2 * -3 = 106).
test(read_operators_in_object,
        [condition(clang_available)]) :-
    obj_dir(Dir),
    directory_file_path(Dir, 'readop_prec.wamo', W1),
    directory_file_path(Dir, 'readop_paren.wamo', W2),
    directory_file_path(Dir, 'readop_assoc.wamo', W3),
    write_wam_object([user:readop_prec/1], [wamo_entry(readop_prec/1)], W1),
    write_wam_object([user:readop_paren/1], [wamo_entry(readop_paren/1)], W2),
    write_wam_object([user:readop_assoc/1], [wamo_entry(readop_assoc/1)], W3),
    directory_file_path(Dir, 'host_bin', Host),
    ( exists_file(Host) -> true ; build_host(Dir, Host) ),
    run_host(Host, W1, O1, 0), assertion(O1 == "7\n"),
    run_host(Host, W2, O2, 0), assertion(O2 == "9\n"),
    run_host(Host, W3, O3, 0), assertion(O3 == "106\n"),
    !.

% Variable reader in a loaded object: shared variables share a cell (bind once,
% see everywhere), anonymous _ are distinct.
test(read_variables_in_object,
        [condition(clang_available)]) :-
    obj_dir(Dir),
    directory_file_path(Dir, 'readvar_shared.wamo', W1),
    directory_file_path(Dir, 'readvar_arith.wamo', W2),
    directory_file_path(Dir, 'readvar_anon.wamo', W3),
    write_wam_object([user:readvar_shared/1], [wamo_entry(readvar_shared/1)], W1),
    write_wam_object([user:readvar_arith/1], [wamo_entry(readvar_arith/1)], W2),
    write_wam_object([user:readvar_anon/1], [wamo_entry(readvar_anon/1)], W3),
    directory_file_path(Dir, 'host_bin', Host),
    ( exists_file(Host) -> true ; build_host(Dir, Host) ),
    run_host(Host, W1, O1, 0), assertion(O1 == "9\n"),
    run_host(Host, W2, O2, 0), assertion(O2 == "42\n"),
    run_host(Host, W3, O3, 0), assertion(O3 == "1\n"),
    !.

% Control operators in a loaded object: a whole clause parses (:- , with a
% variable shared head-to-body), right-associative conjunction, and disjunction.
test(read_control_operators_in_object,
        [condition(clang_available)]) :-
    obj_dir(Dir),
    directory_file_path(Dir, 'readclause.wamo', W1),
    directory_file_path(Dir, 'readconj.wamo', W2),
    directory_file_path(Dir, 'readsemi.wamo', W3),
    write_wam_object([user:readclause/1], [wamo_entry(readclause/1)], W1),
    write_wam_object([user:readconj/1], [wamo_entry(readconj/1)], W2),
    write_wam_object([user:readsemi/1], [wamo_entry(readsemi/1)], W3),
    directory_file_path(Dir, 'host_bin', Host),
    ( exists_file(Host) -> true ; build_host(Dir, Host) ),
    run_host(Host, W1, O1, 0), assertion(O1 == "7\n"),
    run_host(Host, W2, O2, 0), assertion(O2 == "123\n"),
    run_host(Host, W3, O3, 0), assertion(O3 == "33\n"),
    !.

% Floats and quoted atoms in a loaded object: float arithmetic (3.5+1.5 -> 5),
% a float inside a compound (2.5+4.5 -> 7), and a quoted atom with a space
% (atom_length 'hello world' -> 11).
test(read_floats_and_quoted_in_object,
        [condition(clang_available)]) :-
    obj_dir(Dir),
    directory_file_path(Dir, 'readfloat.wamo', W1),
    directory_file_path(Dir, 'readfloatc.wamo', W2),
    directory_file_path(Dir, 'readquoted.wamo', W3),
    write_wam_object([user:readfloat/1], [wamo_entry(readfloat/1)], W1),
    write_wam_object([user:readfloatc/1], [wamo_entry(readfloatc/1)], W2),
    write_wam_object([user:readquoted/1], [wamo_entry(readquoted/1)], W3),
    directory_file_path(Dir, 'host_bin', Host),
    ( exists_file(Host) -> true ; build_host(Dir, Host) ),
    run_host(Host, W1, O1, 0), assertion(O1 == "5\n"),
    run_host(Host, W2, O2, 0), assertion(O2 == "7\n"),
    run_host(Host, W3, O3, 0), assertion(O3 == "11\n"),
    !.

% Byte-buffer output in a loaded object (eval bootstrap milestone 4): a grammar
% assembles a byte string at runtime and the host reads it back via
% @wam_object_call_bytes ({ptr, len, ok}), printing it with %.*s so embedded
% NULs or the absence of a trailing newline do not matter. A computed decimal
% ("42"), a ".wamo"-style header line ("WAMO 2"), and a literal code list
% ("hi"). No trailing newline is emitted by the host, so the bytes are exact.
test(emit_bytes_from_object,
        [condition(clang_available)]) :-
    obj_dir(Dir),
    directory_file_path(Dir, 'emitnum.wamo', W1),
    directory_file_path(Dir, 'emithdr.wamo', W2),
    directory_file_path(Dir, 'emitcodes.wamo', W3),
    write_wam_object([user:emitnum/1],   [wamo_entry(emitnum/1)],   W1),
    write_wam_object([user:emithdr/1],   [wamo_entry(emithdr/1)],   W2),
    write_wam_object([user:emitcodes/1], [wamo_entry(emitcodes/1)], W3),
    directory_file_path(Dir, 'bytes_host_bin', Host),
    ( exists_file(Host) -> true ; build_bytes_host(Dir, Host) ),
    run_host(Host, W1, O1, 0), assertion(O1 == "42"),
    run_host(Host, W2, O2, 0), assertion(O2 == "WAMO 2"),
    run_host(Host, W3, O3, 0), assertion(O3 == "hi"),
    !.

% Dynamic clause store in a loaded object (eval bootstrap milestone 3b-db):
% assertz/asserta/retractall build a process-global clause database at runtime,
% and call/1 consults it (with unification and backtracking) when the meta
% table misses. Each grammar runs in a fresh host process, so the store starts
% empty. Ground facts, called via call/1 (PR 1). See PLAWK_DYNAMIC_DB.md.
test(dynamic_clause_store_in_object,
        [condition(clang_available)]) :-
    obj_dir(Dir),
    directory_file_path(Dir, 'host_bin', Host),
    ( exists_file(Host) -> true ; build_host(Dir, Host) ),
    forall(member(Name-Expected,
               [ dsingle-"42\n",   % deterministic assert + call
                 dselect-"2\n",    % argument-directed clause selection
                 dfirst-"2\n",     % choice-point backtracking (continuation fails)
                 dorder-"9\n",     % asserta prepends
                 dret-"0\n",       % retractall clears, call fails
                 dretp-"110\n" ]), % partial retractall keeps the non-match
           ( PI = Name/1,
             directory_file_path(Dir, Name, Base),
             atom_concat(Base, '.wamo', W),
             write_wam_object([user:PI], [wamo_entry(PI)], W),
             run_host(Host, W, Out, 0),
             assertion(Out == Expected) )),
    !.

% Regression: findall/3 over a call/1 meta-call goal now collects correctly
% (was a template-variable aliasing bug in the aggregate lowering). Covers
% both a predicate compiled into the object (fanum/1 must be included so the
% meta table resolves it) and dynamically asserted facts consulted through
% the store.
test(findall_over_meta_call, [condition(clang_available)]) :-
    obj_dir(Dir),
    directory_file_path(Dir, 'faobjc.wamo', W1),
    directory_file_path(Dir, 'fadynsum.wamo', W2),
    write_wam_object([user:faobjc/1, user:fanum/1], [wamo_entry(faobjc/1)], W1),
    write_wam_object([user:fadynsum/1], [wamo_entry(fadynsum/1)], W2),
    directory_file_path(Dir, 'host_bin', Host),
    ( exists_file(Host) -> true ; build_host(Dir, Host) ),
    run_host(Host, W1, O1, 0), assertion(O1 == "6\n"),
    run_host(Host, W2, O2, 0), assertion(O2 == "60\n"),
    !.

% Milestone 3b-db PR 2: direct calls to :- dynamic predicates (rewritten to a
% call/1 store consult) and nondet retract/1 (a remove+unify+backtrack
% iterator). Each grammar runs in a fresh host process, so the store is empty.
test(dynamic_direct_calls_and_retract, [condition(clang_available)]) :-
    obj_dir(Dir),
    directory_file_path(Dir, 'host_bin', Host),
    ( exists_file(Host) -> true ; build_host(Dir, Host) ),
    forall(member(Name-Expected,
               [ ddirect-"5\n",    % direct call to a dynamic predicate
                 ddirbt-"2\n",     % direct call + choice-point backtracking
                 dtwoarg-"20\n",   % 2-arg direct call, argument-directed
                 dretract-"1\n",   % nondet retract, first match
                 dretbt-"2\n",     % retract + backtrack to the next clause
                 dretgone-"0\n",   % clauses actually removed
                 dretcount-"3\n" ]), % findall drives retract to remove all
           ( PI = Name/1,
             directory_file_path(Dir, Name, Base),
             atom_concat(Base, '.wamo', W),
             write_wam_object([user:PI], [wamo_entry(PI)], W),
             run_host(Host, W, Out, 0),
             assertion(Out == Expected) )),
    !.

% Milestone 3c: catch/3 and throw/1 in a loaded object. The helper predicates
% (Goal, Recovery, and any predicates they reach) must be in the object so the
% meta-call inside catch/throw can resolve them. ct_uncaught has no catch, so
% the throw escapes and the query fails (exit 21).
test(catch_and_throw_in_object, [condition(clang_available)]) :-
    obj_dir(Dir),
    directory_file_path(Dir, 'host_bin', Host),
    ( exists_file(Host) -> true ; build_host(Dir, Host) ),
    Cases = [ ct_catch-[ctrisky/1, cthandle/2]-"42\n"-0,
              ct_nothrow-[ctok/1, ctnorec/1]-"5\n"-0,
              ct_nested-[ctmid/1, ctthrower/1, cthia/1, cthb/2]-"9\n"-0,
              ct_ball-[ctcompute/1, ctgrab/2]-"42\n"-0,
              ct_uncaught-[]-""-21 ],
    forall(member(Name-Deps-Expected-ExpStatus, Cases),
           ( findall(user:P, member(P, Deps), DepPIs),
             directory_file_path(Dir, Name, Base),
             atom_concat(Base, '.wamo', W),
             write_wam_object([user:Name/1|DepPIs], [wamo_entry(Name/1)], W),
             run_host(Host, W, Out, ExpStatus),
             assertion(Out == Expected) )),
    !.

% Milestone 5: the eval/compile pipeline end to end. A compiler object runs on
% source text and emits .wamo bytes; @wam_object_eval loads those bytes into a
% fresh VM in the SAME process; running its entry yields the answer. The
% stand-in compiler (echocompile/2) echoes its source, so the "source" here is
% itself a valid .wamo (ea/1 -> 42); a real source-to-bytecode compiler is
% milestone 6. This closes the eval loop: emit bytes -> load -> run.
test(eval_compile_pipeline, [condition(clang_available)]) :-
    obj_dir(Dir),
    directory_file_path(Dir, 'ea.wamo', EaWamo),
    write_wam_object([user:ea/1], [wamo_entry(ea/1)], EaWamo),
    directory_file_path(Dir, 'compiler.wamo', CompWamo),
    write_wam_object([user:echocompile/2], [wamo_entry(echocompile/2)], CompWamo),
    directory_file_path(Dir, 'eval_host_bin', Host),
    ( exists_file(Host) -> true ; build_eval_host(Dir, Host) ),
    process_create(Host, [CompWamo, EaWamo],
        [stdout(pipe(S)), stderr(std), process(Pid)]),
    read_string(S, _, Out),
    close(S),
    process_wait(Pid, exit(Status)),
    assertion(Status == 0),
    assertion(Out == "42\n"),
    !.

% Milestone 6 (self-host) Stage A: the .wamo serializer runs AS a loaded
% compiler object. Two checks, low-cost first:
%  (1) differential -- the serializer grammar, run in-process, emits bytes
%      byte-identical to the host writer's golden .wamo for the same program.
%      This locks the format (operand encoding, section layout) before any
%      codegen, exactly as the self-host design's Stage A prescribes.
%  (2) end to end -- write the serializer to a .wamo, hand it to the eval host
%      as the "compiler"; @wam_object_eval runs it, loads the bytes it emits,
%      and runs the result -> 42. Proves a loaded grammar can assemble a valid
%      .wamo from ground instruction terms and that the loop closes on it.
test(selfhost_serializer_matches_golden) :-
    obj_dir(Dir),
    directory_file_path(Dir, 'golden_ea.wamo', Golden),
    write_wam_object([user:ea/1], [wamo_entry(ea/1)], Golden),
    read_file_to_string(Golden, GStr, [encoding(octet)]),
    string_codes(GStr, GoldenCodes),
    wamoserz(ignored_source, Wamo),
    atom_codes(Wamo, GotCodes),
    assertion(GotCodes == GoldenCodes),
    !.

test(selfhost_serializer_stage_a, [condition(clang_available)]) :-
    obj_dir(Dir),
    directory_file_path(Dir, 'serz.wamo', SerzWamo),
    write_wam_object([wam_bootstrap_compiler:wamoserz/2], [wamo_entry(wamoserz/2)], SerzWamo),
    % any source file works -- Stage A ignores its input
    directory_file_path(Dir, 'serz_src.txt', SrcPath),
    setup_call_cleanup(open(SrcPath, write, S0),
        write(S0, 'ignored'), close(S0)),
    directory_file_path(Dir, 'eval_host_bin', Host),
    ( exists_file(Host) -> true ; build_eval_host(Dir, Host) ),
    process_create(Host, [SerzWamo, SrcPath],
        [stdout(pipe(S)), stderr(std), process(Pid)]),
    read_string(S, _, Out),
    close(S),
    process_wait(Pid, exit(Status)),
    assertion(Status == 0),
    assertion(Out == "42\n"),
    !.

% Milestone 6 (self-host) Stage B: minimal codegen, source text to a running
% object end to end. cgcompile/2 is a REAL compiler grammar: it parses the
% source clause with the runtime reader, walks it to instructions, and
% serializes a .wamo. Hand it to the eval host as the "compiler"; for each
% source, @wam_object_eval runs cgcompile on the text, loads the .wamo it
% emits, and runs the result -> 42. Exercises both body forms (= and is) --
% the reader, the =/2-vs-is/2 functor dispatch, ground arithmetic evaluation,
% and the serializer, composed into the first source->bytecode compile.
test(selfhost_codegen_stage_b, [condition(clang_available)]) :-
    obj_dir(Dir),
    directory_file_path(Dir, 'cgcompile.wamo', CompWamo),
    write_wam_object([wam_bootstrap_compiler:cgcompile/2], [wamo_entry(cgcompile/2)], CompWamo),
    directory_file_path(Dir, 'eval_host_bin', Host),
    ( exists_file(Host) -> true ; build_eval_host(Dir, Host) ),
    forall(member(Source, ['p1(R) :- R = 42', 'p1(R) :- R is 6*7']),
        ( directory_file_path(Dir, 'cg_src.txt', SrcPath),
          setup_call_cleanup(open(SrcPath, write, S0),
              write(S0, Source), close(S0)),
          process_create(Host, [CompWamo, SrcPath],
              [stdout(pipe(S)), stderr(std), process(Pid)]),
          read_string(S, _, Out),
          close(S),
          process_wait(Pid, exit(Status)),
          assertion(Status == 0),
          assertion(Out == "42\n") )),
    !.

% Milestone 6 (self-host) Stage C: multi-clause program with a predicate call.
% cgcprog/2 compiles a list-of-clauses source into a multi-predicate .wamo: the
% entry clause tail-calls a second clause. Exercises label assignment, the
% label->PC table, and execute(CalleeLabel). "[(main0(R):-helper(R)),
% helper(42)]" compiles so main0 calls helper -> 42, end to end via the eval
% host. The codegen is also verified byte-identical to the host writer's golden
% for the same program (in SWI, where the /3 reader is available).
test(selfhost_codegen_stage_c, [condition(clang_available)]) :-
    obj_dir(Dir),
    directory_file_path(Dir, 'cgcprog.wamo', CompWamo),
    write_wam_object([wam_bootstrap_compiler:cgcprog/2], [wamo_entry(cgcprog/2)], CompWamo),
    directory_file_path(Dir, 'eval_host_bin', Host),
    ( exists_file(Host) -> true ; build_eval_host(Dir, Host) ),
    directory_file_path(Dir, 'cgc_src.txt', SrcPath),
    setup_call_cleanup(open(SrcPath, write, S0),
        write(S0, '[(main0(R):-helper(R)), helper(42)]'), close(S0)),
    process_create(Host, [CompWamo, SrcPath],
        [stdout(pipe(S)), stderr(std), process(Pid)]),
    read_string(S, _, Out),
    close(S),
    process_wait(Pid, exit(Status)),
    assertion(Status == 0),
    assertion(Out == "42\n"),
    !.

% Milestone 6 (self-host) Stage C (rest): conjunction + register allocation.
% cgconj/2 compiles a clause with a conjunction of unification goals, doing
% real register allocation (numbervars -> Y-registers, first/subsequent
% occurrence -> put_variable/put_value). "pconj(R) :- Y = 42, R = Y" -- Y is a
% temporary shared across the two goals -- compiles from source and runs to 42.
test(selfhost_codegen_stage_c_conjunction, [condition(clang_available)]) :-
    obj_dir(Dir),
    directory_file_path(Dir, 'cgconj.wamo', CompWamo),
    write_wam_object([wam_bootstrap_compiler:cgconj/2], [wamo_entry(cgconj/2)], CompWamo),
    directory_file_path(Dir, 'eval_host_bin', Host),
    ( exists_file(Host) -> true ; build_eval_host(Dir, Host) ),
    directory_file_path(Dir, 'cgconj_src.txt', SrcPath),
    setup_call_cleanup(open(SrcPath, write, S0),
        write(S0, 'pconj(R) :- Y = 42, R = Y'), close(S0)),
    process_create(Host, [CompWamo, SrcPath],
        [stdout(pipe(S)), stderr(std), process(Pid)]),
    read_string(S, _, Out),
    close(S),
    process_wait(Pid, exit(Status)),
    assertion(Status == 0),
    assertion(Out == "42\n"),
    !.

% Milestone 6 (self-host) Stage C (arithmetic): runtime is/2 with put_structure.
% cgarith/2 compiles "ca(R) :- X is 6*7, R = X" -- X is 6*7 builds *(6,7) on the
% heap (put_structure + set_constant), calls builtin is/2, binds temporary X;
% then R = X. Exercises the functor table (NF=1, functor "*"), put_structure
% with a functor reloc, and is/2 -- combined with conjunction and register
% allocation. Compiles from source in a loaded object and runs to 42.
test(selfhost_codegen_stage_c_arithmetic, [condition(clang_available)]) :-
    obj_dir(Dir),
    directory_file_path(Dir, 'cgarith.wamo', CompWamo),
    write_wam_object([wam_bootstrap_compiler:cgarith/2], [wamo_entry(cgarith/2)], CompWamo),
    directory_file_path(Dir, 'eval_host_bin', Host),
    ( exists_file(Host) -> true ; build_eval_host(Dir, Host) ),
    forall(member(Source, ['ca(R) :- X is 6*7, R = X',
                           'add(R) :- A = 40, B = 2, R is A+B']),
        ( directory_file_path(Dir, 'cgarith_src.txt', SrcPath),
          setup_call_cleanup(open(SrcPath, write, S0),
              write(S0, Source), close(S0)),
          process_create(Host, [CompWamo, SrcPath],
              [stdout(pipe(S)), stderr(std), process(Pid)]),
          read_string(S, _, Out),
          close(S),
          process_wait(Pid, exit(Status)),
          assertion(Status == 0),
          assertion(Out == "42\n") )),
    !.

% Milestone 6 (self-host) Stage C (non-tail calls): the unified compiler cgfull/2
% merges labels + register allocation + conjunction + arithmetic + predicate
% calls. Two multi-clause programs exercise a non-tail call whose result is used
% after it returns: a passthrough (mnt calls helper, then unifies) and a compute
% (main calls add1 with a computed arg, doubling via the callee). Both compile
% from a list-of-clauses source in a loaded object and run to 42.
test(selfhost_codegen_stage_c_nontail_call, [condition(clang_available)]) :-
    obj_dir(Dir),
    directory_file_path(Dir, 'cgfull.wamo', CompWamo),
    write_wam_object([wam_bootstrap_compiler:cgfull/2], [wamo_entry(cgfull/2)], CompWamo),
    directory_file_path(Dir, 'eval_host_bin', Host),
    ( exists_file(Host) -> true ; build_eval_host(Dir, Host) ),
    forall(member(Source,
               [ '[(mnt(R):-helper(A), R=A), helper(42)]',
                 '[(main0(R):-add1(41,V), R=V), (add1(X,Y):-Y is X+1)]' ]),
        ( directory_file_path(Dir, 'cgfull_src.txt', SrcPath),
          setup_call_cleanup(open(SrcPath, write, S0),
              write(S0, Source), close(S0)),
          process_create(Host, [CompWamo, SrcPath],
              [stdout(pipe(S)), stderr(std), process(Pid)]),
          read_string(S, _, Out),
          close(S),
          process_wait(Pid, exit(Status)),
          assertion(Status == 0),
          assertion(Out == "42\n") )),
    !.

% Milestone 6 (self-host) Stage D (multi-clause predicates): cgfull compiles
% predicates with MULTIPLE clauses into try_me_else/retry_me_else/trust_me
% chains -- backtracking dispatch and recursion from source text. Two
% programs: (1) dispatch -- picku/2 has two clauses; picku(2,R) must FAIL
% clause 1's head (get_constant 1 vs 2), backtrack through the chain, and
% match clause 2 -> 42. (2) recursion -- factorial with a base clause and a
% recursive clause (self-call by label, arithmetic on the way down and up);
% fact(3) -> 6. Both compile inside a loaded compiler object via the eval host.
test(selfhost_codegen_stage_d_multiclause, [condition(clang_available)]) :-
    obj_dir(Dir),
    directory_file_path(Dir, 'cgfull.wamo', CompWamo),
    write_wam_object([wam_bootstrap_compiler:cgfull/2], [wamo_entry(cgfull/2)], CompWamo),
    directory_file_path(Dir, 'eval_host_bin', Host),
    ( exists_file(Host) -> true ; build_eval_host(Dir, Host) ),
    forall(member(Source-Expected,
               [ '[(main0(R):-picku(2,R)), picku(1,10), picku(2,42)]' - "42\n",
                 '[(main0(R):-fact(3,R)), fact(0,1), (fact(N,R):- M is N-1, fact(M,F), R is N*F)]' - "6\n" ]),
        ( directory_file_path(Dir, 'cgmc_src.txt', SrcPath),
          setup_call_cleanup(open(SrcPath, write, S0),
              write(S0, Source), close(S0)),
          process_create(Host, [CompWamo, SrcPath],
              [stdout(pipe(S)), stderr(std), process(Pid)]),
          read_string(S, _, Out),
          close(S),
          process_wait(Pid, exit(Status)),
          assertion(Status == 0),
          assertion(Out == Expected) )),
    !.

% Milestone 6 (self-host) Stage D (lists): list patterns in heads, list
% literals in call args, and the atom table. The classic list-walking shape --
% a base clause on [] (atom constant, repeated head var A needing get_value)
% and a recursive clause destructuring [H|T] (get_list + unify_variable) --
% compiles from source and sums [10,20,12] to 42. This is the shape of every
% recursive helper in the compiler's own source.
test(selfhost_codegen_stage_d_lists, [condition(clang_available)]) :-
    obj_dir(Dir),
    directory_file_path(Dir, 'cgfull.wamo', CompWamo),
    write_wam_object([wam_bootstrap_compiler:cgfull/2], [wamo_entry(cgfull/2)], CompWamo),
    directory_file_path(Dir, 'eval_host_bin', Host),
    ( exists_file(Host) -> true ; build_eval_host(Dir, Host) ),
    directory_file_path(Dir, 'cgl_src.txt', SrcPath),
    setup_call_cleanup(open(SrcPath, write, S0),
        write(S0, '[(main0(R):-suml([10,20,12],0,R)), suml([],A,A), (suml([H|T],A,R):- A1 is A+H, suml(T,A1,R))]'),
        close(S0)),
    process_create(Host, [CompWamo, SrcPath],
        [stdout(pipe(S)), stderr(std), process(Pid)]),
    read_string(S, _, Out),
    close(S),
    process_wait(Pid, exit(Status)),
    assertion(Status == 0),
    assertion(Out == "42\n"),
    !.

% Milestone 6 (self-host) Stage D (guards): comparison guards and
% if-then-else. Three programs: max via ( A >= B -> R = A ; R = B ) taking
% the THEN branch (40>=2), the same taking the ELSE branch (2>=40 fails, the
% guard CP backtracks to trust_me), and a plain comparison goal in a
% conjunction (X > 10). All compile from source in a loaded compiler object.
test(selfhost_codegen_stage_d_guards, [condition(clang_available)]) :-
    obj_dir(Dir),
    directory_file_path(Dir, 'cgfull.wamo', CompWamo),
    write_wam_object([wam_bootstrap_compiler:cgfull/2], [wamo_entry(cgfull/2)], CompWamo),
    directory_file_path(Dir, 'eval_host_bin', Host),
    ( exists_file(Host) -> true ; build_eval_host(Dir, Host) ),
    forall(member(Source-Expected,
               [ '[(main0(R):-maxi(40,2,X), R is X+2), (maxi(A,B,R):-( A >= B -> R = A ; R = B ))]' - "42\n",
                 '[(main0(R):-maxi(2,40,X), R is X+2), (maxi(A,B,R):-( A >= B -> R = A ; R = B ))]' - "42\n",
                 '[(main0(R):- p(15,R)), (p(X,R):- X > 10, R = 42)]' - "42\n" ]),
        ( directory_file_path(Dir, 'cgg_src.txt', SrcPath),
          setup_call_cleanup(open(SrcPath, write, S0),
              write(S0, Source), close(S0)),
          process_create(Host, [CompWamo, SrcPath],
              [stdout(pipe(S)), stderr(std), process(Pid)]),
          read_string(S, _, Out),
          close(S),
          process_wait(Pid, exit(Status)),
          assertion(Status == 0),
          assertion(Out == Expected) )),
    !.

% Milestone 6 (self-host) Stage D (structures): general compound terms in
% heads and call arguments. Four programs: (1) flat -- mk(pt(40,2)) builds a
% structure in WRITE mode (fact head, unbound caller arg) and un(pt(X,Y),R)
% destructures it in READ mode (get_structure + unify_variable); (2) nested --
% f(g(40)) built and f(g(X)) matched via the X-temp deferral (unify_variable
% Xt then get_structure into Xt); (3) pair sugar 40-2 / A-B ('-'/2 is just a
% compound); (4) the compiler's own instruction-term shape enc(T,O1,O2,Rl)
% destructured in a head -- the exact pattern the fixpoint needs.
test(selfhost_codegen_stage_d_structures, [condition(clang_available)]) :-
    obj_dir(Dir),
    directory_file_path(Dir, 'cgfull.wamo', CompWamo),
    write_wam_object([wam_bootstrap_compiler:cgfull/2], [wamo_entry(cgfull/2)], CompWamo),
    directory_file_path(Dir, 'eval_host_bin', Host),
    ( exists_file(Host) -> true ; build_eval_host(Dir, Host) ),
    forall(member(Source-Expected,
               [ '[(main0(R):- mk(P), un(P,R)), mk(pt(40,2)), (un(pt(X,Y),R):- R is X+Y)]' - "42\n",
                 '[(main0(R):- wn(f(g(40)),R)), (wn(f(g(X)),R):- R is X+2)]' - "42\n",
                 '[(main0(R):- p(40-2,R)), (p(A-B,R):- R is A+B)]' - "42\n",
                 '[(main0(R):- d(enc(20,10,7,5),R)), (d(enc(T,O1,O2,Rl),R):- A is T+O1, B is O2+Rl, R is A+B)]' - "42\n" ]),
        ( directory_file_path(Dir, 'cgs_src.txt', SrcPath),
          setup_call_cleanup(open(SrcPath, write, S0),
              write(S0, Source), close(S0)),
          process_create(Host, [CompWamo, SrcPath],
              [stdout(pipe(S)), stderr(std), process(Pid)]),
          read_string(S, _, Out),
          close(S),
          process_wait(Pid, exit(Status)),
          assertion(Status == 0),
          assertion(Out == Expected) )),
    !.

% Milestone 6 (self-host) Stage D (builtin goals): the whitelisted builtins
% compile as staged-args + builtin_call. Four programs: (1) functor/3 + arg/3
% -- NB the grammar emits builtin_call arg/3 even for a constant index, so the
% host compiler's specialised-arg subset gap never arises; (2) =../2 building
% a term then unifying against a structure literal (the upgraded =/2 with
% full-term operands); (3) atom_codes/2 + length/2 over a data atom (exercises
% atoms-as-data in the atom table); (4) a type-check builtin (integer/1) as an
% if-then-else condition.
test(selfhost_codegen_stage_d_builtins, [condition(clang_available)]) :-
    obj_dir(Dir),
    directory_file_path(Dir, 'cgfull.wamo', CompWamo),
    write_wam_object([wam_bootstrap_compiler:cgfull/2], [wamo_entry(cgfull/2)], CompWamo),
    directory_file_path(Dir, 'eval_host_bin', Host),
    ( exists_file(Host) -> true ; build_eval_host(Dir, Host) ),
    forall(member(Source-Expected,
               [ '[(main0(R):- T = pt(40,2), functor(T,_,N), arg(1,T,X), R is X+N)]' - "42\n",
                 '[(main0(R):- T =.. [f,40,2], T = f(A,B), R is A+B)]' - "42\n",
                 '[(main0(R):- atom_codes(abc,Cs), length(Cs,N), R is N+39)]' - "42\n",
                 '[(main0(R):- T = 42, ( integer(T) -> R = T ; R = 0 ))]' - "42\n" ]),
        ( directory_file_path(Dir, 'cgb_src.txt', SrcPath),
          setup_call_cleanup(open(SrcPath, write, S0),
              write(S0, Source), close(S0)),
          process_create(Host, [CompWamo, SrcPath],
              [stdout(pipe(S)), stderr(std), process(Pid)]),
          read_string(S, _, Out),
          close(S),
          process_wait(Pid, exit(Status)),
          assertion(Status == 0),
          assertion(Out == Expected) )),
    !.

% Milestone 6 (self-host) Stage D: THE FIXPOINT SLICE. The loaded bootstrap
% compiler compiles the source text of its own Stage A serializer (the wz_*
% chain, restated cut-free in the accepted subset), and the doubly-compiled
% serializer then serializes the golden `ea(R):-R=42` program. The compiled
% program returns byte-sum + length of its output; the test computes the same
% checksum from the Stage A implementation (wamoserz) in SWI, so the assertion
% proves the twice-compiled serializer reproduces the golden byte stream
% exactly. This is the compiler compiling its own back end -- the first
% self-application of the self-host arc.
test(selfhost_codegen_stage_d_fixpoint, [condition(clang_available)]) :-
    obj_dir(Dir),
    directory_file_path(Dir, 'cgfull.wamo', CompWamo),
    write_wam_object([wam_bootstrap_compiler:cgfull/2], [wamo_entry(cgfull/2)], CompWamo),
    directory_file_path(Dir, 'eval_host_bin', Host),
    ( exists_file(Host) -> true ; build_eval_host(Dir, Host) ),
    % expected checksum from the Stage A serializer, computed in-process
    wamoserz(x, W), atom_codes(W, WCs),
    sum_list(WCs, WSum), length(WCs, WLen),
    ExpectedN is WSum + WLen,
    format(string(Expected), "~w\n", [ExpectedN]),
    fixpoint_serializer_source(Source),
    directory_file_path(Dir, 'cgfp_src.txt', SrcPath),
    setup_call_cleanup(open(SrcPath, write, S0),
        write(S0, Source), close(S0)),
    process_create(Host, [CompWamo, SrcPath],
        [stdout(pipe(S)), stderr(std), process(Pid)]),
    read_string(S, _, Out),
    close(S),
    process_wait(Pid, exit(Status)),
    assertion(Status == 0),
    assertion(Out == Expected),
    !.

% GEN 3: the loaded compiler compiles a COMPILER, and the doubly-compiled
% compiler compiles TWO golden programs byte-exactly. gen 1 = AOT cgfull;
% gen 2 = cmp2 (reader + =.. decomposition + a dispatching ITE codegen
% decision + atom-table emission + wz chain), compiled by gen 1 inside the
% eval host; gen 3 = gen 2 run on 'ea(R2) :- R2 = 42' (int constant, no
% tables) and 'eb(R2) :- R2 = foo' (atom constant, one atom-table row,
% reloc class 1). The combined byte checksum must equal what the Stage A
% serializers yield in-process (wamoserz for ea; wza_serialize for eb).
test(selfhost_codegen_stage_d_gen3, [condition(clang_available)]) :-
    obj_dir(Dir),
    directory_file_path(Dir, 'cgfull.wamo', CompWamo),
    write_wam_object([wam_bootstrap_compiler:cgfull/2], [wamo_entry(cgfull/2)], CompWamo),
    directory_file_path(Dir, 'eval_host_bin', Host),
    ( exists_file(Host) -> true ; build_eval_host(Dir, Host) ),
    wamoserz(x, W), atom_codes(W, WCs),
    sum_list(WCs, WSum), length(WCs, WLen),
    atom_codes('eb/1', EbName),
    wza_serialize(0, EbName, 0, [foo], [], [0],
        [enc(0,0,0,1), enc(20,0,0,0)], W2Cs),
    sum_list(W2Cs, W2Sum), length(W2Cs, W2Len),
    ExpectedN is WSum + WLen + W2Sum + W2Len,
    format(string(Expected), "~w\n", [ExpectedN]),
    fixpoint_compiler_source(Source),
    directory_file_path(Dir, 'cgc2_src.txt', SrcPath),
    setup_call_cleanup(open(SrcPath, write, S0),
        write(S0, Source), close(S0)),
    process_create(Host, [CompWamo, SrcPath],
        [stdout(pipe(S)), stderr(std), process(Pid)]),
    read_string(S, _, Out),
    close(S),
    process_wait(Pid, exit(Status)),
    assertion(Status == 0),
    assertion(Out == Expected),
    !.

% THE MIDDLE, first slice: the loaded compiler compiles its own CODEGEN.
% gen 1 = AOT cgfull; gen 2 = mid2 (fixpoint_middle_source/1 -- real
% register allocation: numbervars, first-occurrence init tracking,
% head/operand/expression compilation, functor-table collection, both
% serializer tables); gen 3 = mid2 run on the arithmetic clause
% 'sum3(A, B, R2) :- T is A + B, R2 is T + 1'. Byte-exactness target:
% the REAL cgfull middle (cgfull_term/2, reader split off) run on the
% same clause TERM in SWI -- the doubly-compiled codegen must reproduce
% the production compiler's bytes exactly, not just a fixed template.
test(selfhost_codegen_stage_d_middle, [condition(clang_available)]) :-
    obj_dir(Dir),
    directory_file_path(Dir, 'cgfull.wamo', CompWamo),
    write_wam_object([wam_bootstrap_compiler:cgfull/2], [wamo_entry(cgfull/2)], CompWamo),
    directory_file_path(Dir, 'eval_host_bin', Host),
    ( exists_file(Host) -> true ; build_eval_host(Dir, Host) ),
    cgfull_term([(sum3(A, B, R2) :- T is A + B, R2 is T + 1)], W),
    atom_codes(W, WCs),
    sum_list(WCs, WSum), length(WCs, WLen),
    ExpectedN is WSum + WLen,
    format(string(Expected), "~w\n", [ExpectedN]),
    fixpoint_middle_source(Source),
    directory_file_path(Dir, 'cgmid_src.txt', SrcPath),
    setup_call_cleanup(open(SrcPath, write, S0),
        write(S0, Source), close(S0)),
    process_create(Host, [CompWamo, SrcPath],
        [stdout(pipe(S)), stderr(std), process(Pid)]),
    read_string(S, _, Out),
    close(S),
    process_wait(Pid, exit(Status)),
    assertion(Status == 0),
    assertion(Out == Expected),
    !.

% THE FRONT: the loaded compiler compiles its own clause-grouping, label,
% and chain machinery (plus the middle), and the doubly-compiled compiler
% compiles a MULTI-PREDICATE program -- facts, a two-clause chain with
% try/trust and a Label-PC pair, a predicate call, integer/atom head
% constants, and arithmetic -- byte-identically to the production cgfull.
test(selfhost_codegen_stage_d_front, [condition(clang_available)]) :-
    obj_dir(Dir),
    directory_file_path(Dir, 'cgfull.wamo', CompWamo),
    write_wam_object([wam_bootstrap_compiler:cgfull/2], [wamo_entry(cgfull/2)], CompWamo),
    directory_file_path(Dir, 'eval_host_bin', Host),
    ( exists_file(Host) -> true ; build_eval_host(Dir, Host) ),
    cgfull_term([(dbl(X, Y) :- Y is X + X),
                 (tst(R2) :- dbl(4, R2)),
                 pick(1, a), pick(2, b)], W),
    atom_codes(W, WCs),
    sum_list(WCs, WSum), length(WCs, WLen),
    ExpectedN is WSum + WLen,
    format(string(Expected), "~w\n", [ExpectedN]),
    fixpoint_front_source(Source),
    directory_file_path(Dir, 'cgfront_src.txt', SrcPath),
    setup_call_cleanup(open(SrcPath, write, S0),
        write(S0, Source), close(S0)),
    process_create(Host, [CompWamo, SrcPath],
        [stdout(pipe(S)), stderr(std), process(Pid)]),
    read_string(S, _, Out),
    close(S),
    process_wait(Pid, exit(Status)),
    assertion(Status == 0),
    assertion(Out == Expected),
    !.

% Finding no. 11 regression: the reader var-dict must GROW past 128
% distinct names. The source below has 130 names: main0's R, 128 filler
% facts d1(V1)..d128(V128), then eq(Z, Z) whose repeated Z is the 130th.
% With the old fixed cap, Z stopped being deduplicated -- eq compiled as
% eq(_, _) with two independent variables -- and main0 printed an
% unbound result instead of 5.
test(reader_var_dict_grows_past_128, [condition(clang_available)]) :-
    obj_dir(Dir),
    directory_file_path(Dir, 'cgfull.wamo', CompWamo),
    write_wam_object([wam_bootstrap_compiler:cgfull/2], [wamo_entry(cgfull/2)], CompWamo),
    directory_file_path(Dir, 'eval_host_bin', Host),
    ( exists_file(Host) -> true ; build_eval_host(Dir, Host) ),
    numlist(1, 128, Ns),
    findall(F, ( member(I, Ns),
                 format(atom(F), 'd~w(V~w)', [I, I]) ), Fillers),
    atomic_list_concat(Fillers, ', ', FillerText),
    format(atom(Src), '[(main0(R) :- eq(5, R)), ~w, eq(Z, Z)]',
           [FillerText]),
    directory_file_path(Dir, 'cgvd_src.txt', SrcPath),
    setup_call_cleanup(open(SrcPath, write, S0),
        write(S0, Src), close(S0)),
    process_create(Host, [CompWamo, SrcPath],
        [stdout(pipe(S)), stderr(std), process(Pid)]),
    read_string(S, _, Out),
    close(S),
    process_wait(Pid, exit(Status)),
    assertion(Status == 0),
    assertion(Out == "5\n"),
    !.

% THE WALKERS (loaded, COMPLETE): the doubly-compiled compiler handles
% ITE codegen with labels and init-set intersection, comparison guards,
% the builtin whitelist, list-literal operands, and structure/list head
% patterns INCLUDING the X-temp deferral paths (nested pattern
% w(f(g(Z)), Z), nested build (A + B) * 2) -- byte-identically to the
% production cgfull. Deferral loaded was unblocked by closing finding
% no. 12 (monotonic heap) and the fact Y-window clobber (facts now
% allocate).
test(selfhost_codegen_stage_d_walkers, [condition(clang_available)]) :-
    obj_dir(Dir),
    directory_file_path(Dir, 'cgfull.wamo', CompWamo),
    write_wam_object([wam_bootstrap_compiler:cgfull/2], [wamo_entry(cgfull/2)], CompWamo),
    directory_file_path(Dir, 'eval_host_bin', Host),
    ( exists_file(Host) -> true ; build_eval_host(Dir, Host) ),
    cgfull_term([(cls(N, R2) :- (N >= 10 -> R2 = big ; R2 = small)),
                 (sum2(Xs, R3) :- append(Xs, [7], Ys), sum_list(Ys, R3)),
                 swap(p(X, Y), p(Y, X)),
                 w(f(g(Z)), Z),
                 (tot([A, B], R4) :- R4 is (A + B) * 2)], W),
    atom_codes(W, WCs),
    sum_list(WCs, WSum), length(WCs, WLen),
    ExpectedN is WSum + WLen,
    format(string(Expected), "~w\n", [ExpectedN]),
    fixpoint_walkers_source(Source),
    directory_file_path(Dir, 'cgwalk_src.txt', SrcPath),
    setup_call_cleanup(open(SrcPath, write, S0),
        write(S0, Source), close(S0)),
    process_create(Host, [CompWamo, SrcPath],
        [stdout(pipe(S)), stderr(std), process(Pid)]),
    read_string(S, _, Out),
    close(S),
    process_wait(Pid, exit(Status)),
    assertion(Status == 0),
    assertion(Out == Expected),
    !.

% THE WALKERS (SWI semantics): interpret the walkers source AS PROLOG in
% SWI and run it on the FULL golden -- including the X-temp deferral
% cases (nested head pattern w(f(g(Z)), Z) and nested expression build
% (A + B) * 2). The output must be byte-identical to the production
% cgfull middle. This pins the walkers' LOGIC as correct; the loaded
% deferral gap is runtime finding no. 12, not a compiler-logic gap.
test(selfhost_walkers_logic_in_swi) :-
    fixpoint_walkers_source(Source),
    % the source uses reader-verbatim quoting; double backslashes for SWI
    atom_codes(Source, Cs0),
    findall(C2, ( member(C, Cs0),
                  ( C =:= 0'\\ -> member(C2, [C, C]) ; C2 = C ) ), Cs1),
    atom_codes(Source2, Cs1),
    term_to_atom(Clauses, Source2),
    forall(member(C, Clauses),
           ( C = (H :- B) -> assertz(mid4w:(H :- B)) ; assertz(mid4w:C) )),
    Golden = [(cls(N, R2) :- (N >= 10 -> R2 = big ; R2 = small)),
              (sum2(Xs, R3) :- append(Xs, [7], Ys), sum_list(Ys, R3)),
              swap(p(X, Y), p(Y, X)),
              w(f(g(Z)), Z),
              (tot([A, B], R4) :- R4 is (A + B) * 2)],
    mid4w:mwl(Golden, [], [], At, FT),
    mid4w:mgrp(Golden, Gs),
    mid4w:mlbl(Gs, 0, PL),
    length(Gs, NP0),
    mid4w:mggs(Gs, PL, At, FT, 0, NP0, AllIs, EPCs, Prs),
    keysort(Prs, SPrs), pairs_values(SPrs, XPCs),
    append(EPCs, XPCs, PCs),
    Golden = [C1|_], mid4w:mhb(C1, H1, _), functor(H1, P1, A1),
    mid4w:mname(P1, A1, NC),
    mid4w:wzs(0, NC, 0, At, FT, PCs, AllIs, Codes),
    cgfull_term(Golden, WGold),
    atom_codes(WGold, GoldCodes),
    assertion(Codes == GoldCodes),
    !.

% Finding no. 12 regression (the minimal pair): a deferral walker whose
% clause wraps its body in an if-then-else. Before the monotonic-heap fix,
% a post-join failure backtracked into a completed call's stale chain CP,
% and re-execution followed a dangling Ref into the rewound heap region --
% FATAL heap oob read at heap_top. With backtrack no longer rewinding
% heap_top (cells persist unreferenced, like the arena), re-satisfaction
% runs on consistent state and the compile COMPLETES. NOTE: the completed
% value still reflects a divergent re-satisfaction derivation (the
% loaded-vs-SWI semantic difference that triggers the initial failure is
% the open remainder of finding no. 12) -- when that is fixed, tighten
% the assertion to the exact output.
test(selfhost_finding12_no_heap_oob, [condition(clang_available)]) :-
    obj_dir(Dir),
    directory_file_path(Dir, 'cgfull.wamo', CompWamo),
    write_wam_object([wam_bootstrap_compiler:cgfull/2], [wamo_entry(cgfull/2)], CompWamo),
    directory_file_path(Dir, 'eval_host_bin', Host),
    ( exists_file(Host) -> true ; build_eval_host(Dir, Host) ),
    directory_file_path(Dir, 'cgf12_src.txt', SrcPath),
    finding12_pair_source(Source),
    setup_call_cleanup(open(SrcPath, write, S0),
        write(S0, Source), close(S0)),
    process_create(Host, [CompWamo, SrcPath],
        [stdout(pipe(S)), stderr(std), process(Pid)]),
    read_string(S, _, Out),
    close(S),
    process_wait(Pid, exit(Status)),
    assertion(Status == 0),
    assertion(Out == "6\n"),
    !.

% ============================ THE CAPSTONE ============================
% compile(SelfSource): the fixpoint. gen 1 = the AOT-compiled cgfull.
% The SELF-SOURCE is the walkers compiler itself, re-entered through
% main2(Src, W) :- cm3(Src, Cs), atom_codes(W, Cs) -- the source text
% arrives as a runtime argument, so no quine trick is needed. gen 2 =
% gen 1 compiling the self-source (a working compiler object, proven
% byte-exact against cgfull all campaign). gen 3 = GEN 2 COMPILING ITS
% OWN SOURCE. Because gen 2 reproduces cgfull's bytes exactly, gen 3
% must equal gen 2 BYTE-FOR-BYTE: F(F) = F -- the compiler compiles
% itself and the self-compiled compiler is the same object. The
% triangle is closed by gen 3 compiling the full walkers golden
% byte-identically to the production cgfull middle.
test(selfhost_capstone_fixpoint, [condition(clang_available)]) :-
    obj_dir(Dir),
    directory_file_path(Dir, 'cgfull.wamo', CompWamo),
    write_wam_object([wam_bootstrap_compiler:cgfull/2], [wamo_entry(cgfull/2)], CompWamo),
    directory_file_path(Dir, 'dump_host_bin', Dump),
    ( exists_file(Dump) -> true ; build_dump_host(Dir, Dump) ),
    fixpoint_walkers_source(Walkers),
    once(sub_atom(Walkers, BPos, _, _, '), (cm3(')),
    Skip is BPos + 2,
    sub_atom(Walkers, Skip, _, 0, Rest),
    atom_concat('[(main2(Src2, W2) :- cm3(Src2, Cs2), atom_codes(W2, Cs2)),',
                Rest, SelfSrc),
    directory_file_path(Dir, 'self_src.txt', SelfPath),
    setup_call_cleanup(open(SelfPath, write, S0),
        write(S0, SelfSrc), close(S0)),
    run_dump(Dump, CompWamo, SelfPath, Gen2Bytes),
    directory_file_path(Dir, 'gen2.wamo', Gen2Path),
    setup_call_cleanup(open(Gen2Path, write, S1),
        write(S1, Gen2Bytes), close(S1)),
    run_dump(Dump, Gen2Path, SelfPath, Gen3Bytes),
    assertion(Gen2Bytes == Gen3Bytes),
    directory_file_path(Dir, 'gen3.wamo', Gen3Path),
    setup_call_cleanup(open(Gen3Path, write, S2),
        write(S2, Gen3Bytes), close(S2)),
    directory_file_path(Dir, 'cap_gold.txt', GoldPath),
    setup_call_cleanup(open(GoldPath, write, S3),
        write(S3, '[(cls(N, R2) :- (N >= 10 -> R2 = big ; R2 = small)), (sum2(Xs, R3) :- append(Xs, [7], Ys), sum_list(Ys, R3)), swap(p(X, Y), p(Y, X)), w(f(g(Z)), Z), (tot([A, B], R4) :- R4 is (A + B) * 2)]'),
        close(S3)),
    run_dump(Dump, Gen3Path, GoldPath, GoldBytes),
    cgfull_term([(cls(N, R2) :- (N >= 10 -> R2 = big ; R2 = small)),
                 (sum2(Xs, R3) :- append(Xs, [7], Ys), sum_list(Ys, R3)),
                 swap(p(X, Y), p(Y, X)),
                 w(f(g(Z)), Z),
                 (tot([A, B], R4) :- R4 is (A + B) * 2)], WGold),
    atom_string(GoldAtom, GoldBytes),
    assertion(GoldAtom == WGold),
    !.

% Capstone finding regression: loaded arithmetic on a NON-ARITHMETIC
% compound must FAIL (SWI raises a type error) rather than silently
% evaluating the unknown functor to 0. Before the @wam_arith_err flag,
% X is 1 + f(2) succeeded with X = 1, so any dispatch clause guarded by
% an is-failure wrongly committed -- the self-compile diverged from the
% production compiler at exactly its numbervarred-marker clauses.
test(selfhost_arith_error_fails_cleanly, [condition(clang_available)]) :-
    obj_dir(Dir),
    directory_file_path(Dir, 'cgfull.wamo', CompWamo),
    write_wam_object([wam_bootstrap_compiler:cgfull/2], [wamo_entry(cgfull/2)], CompWamo),
    directory_file_path(Dir, 'eval_host_bin', Host),
    ( exists_file(Host) -> true ; build_eval_host(Dir, Host) ),
    directory_file_path(Dir, 'cgae_src.txt', SrcPath),
    setup_call_cleanup(open(SrcPath, write, S0),
        write(S0, '[(main0(R) :- (X is 1 + f(2) -> R = X ; R = 2))]'),
        close(S0)),
    process_create(Host, [CompWamo, SrcPath],
        [stdout(pipe(S)), stderr(std), process(Pid)]),
    read_string(S, _, Out),
    close(S),
    process_wait(Pid, exit(Status)),
    assertion(Status == 0),
    assertion(Out == "2\n"),
    !.

% Loaded truncating integer division (deferred small item): // used to
% alias float-capable / (7 // 2 evaluated to 3.5 where SWI yields 3).
% It now truncates toward zero (sdiv, matching SWI's default
% integer_rounding_function), and division by zero fails through the
% arith-error channel into the else branch. 7//2=3, -7//2=-3,
% (3*100+3)*10+4 = 3034.
test(selfhost_intdiv_truncates, [condition(clang_available)]) :-
    obj_dir(Dir),
    directory_file_path(Dir, 'cgfull.wamo', CompWamo),
    write_wam_object([wam_bootstrap_compiler:cgfull/2], [wamo_entry(cgfull/2)], CompWamo),
    directory_file_path(Dir, 'eval_host_bin', Host),
    ( exists_file(Host) -> true ; build_eval_host(Dir, Host) ),
    directory_file_path(Dir, 'cgid_src.txt', SrcPath),
    setup_call_cleanup(open(SrcPath, write, S0),
        write(S0, '[(main0(R) :- A is 7 // 2, B is 0 - 7, C is B // 2, (D is 5 // 0 -> E = D ; E = 4), R is (A * 100 + (0 - C)) * 10 + E)]'),
        close(S0)),
    process_create(Host, [CompWamo, SrcPath],
        [stdout(pipe(S)), stderr(std), process(Pid)]),
    read_string(S, _, Out),
    close(S),
    process_wait(Pid, exit(Status)),
    assertion(Status == 0),
    assertion(Out == "3034\n"),
    !.

% Demand-driven subset growth: plain CUT compiles from source and runs
% loaded with committed-choice semantics. pick(7) commits to the first
% clause (guard passed, cut fired -> 1); pick(3) fails the guard BEFORE
% the cut and falls to the second clause (-> 2). R = 12.
test(selfhost_cut_commits, [condition(clang_available)]) :-
    obj_dir(Dir),
    directory_file_path(Dir, 'cgfull.wamo', CompWamo),
    write_wam_object([wam_bootstrap_compiler:cgfull/2], [wamo_entry(cgfull/2)], CompWamo),
    directory_file_path(Dir, 'eval_host_bin', Host),
    ( exists_file(Host) -> true ; build_eval_host(Dir, Host) ),
    directory_file_path(Dir, 'cgcut_src.txt', SrcPath),
    setup_call_cleanup(open(SrcPath, write, S0),
        write(S0, '[(main0(R) :- pick(7, A), pick(3, B), R is A * 10 + B), (pick(X, R2) :- X > 5, !, R2 = 1), pick(_, 2)]'),
        close(S0)),
    process_create(Host, [CompWamo, SrcPath],
        [stdout(pipe(S)), stderr(std), process(Pid)]),
    read_string(S, _, Out),
    close(S),
    process_wait(Pid, exit(Status)),
    assertion(Status == 0),
    assertion(Out == "12\n"),
    !.

% BARE disjunction stays backtrackable: d(a) takes the first branch;
% d(b) enters it, fails the ==, and BACKTRACKS into the second (the
% try_me_else CP is live -- no soft cut). R = 12.
test(selfhost_bare_disjunction_backtracks, [condition(clang_available)]) :-
    obj_dir(Dir),
    directory_file_path(Dir, 'cgfull.wamo', CompWamo),
    write_wam_object([wam_bootstrap_compiler:cgfull/2], [wamo_entry(cgfull/2)], CompWamo),
    directory_file_path(Dir, 'eval_host_bin', Host),
    ( exists_file(Host) -> true ; build_eval_host(Dir, Host) ),
    directory_file_path(Dir, 'cgdisj_src.txt', SrcPath),
    setup_call_cleanup(open(SrcPath, write, S0),
        write(S0, '[(main0(R) :- d(a, A), d(b, B), R is A * 10 + B), (d(X, R2) :- (X == a, R2 = 1 ; R2 = 2))]'),
        close(S0)),
    process_create(Host, [CompWamo, SrcPath],
        [stdout(pipe(S)), stderr(std), process(Pid)]),
    read_string(S, _, Out),
    close(S),
    process_wait(Pid, exit(Status)),
    assertion(Status == 0),
    assertion(Out == "12\n"),
    !.

% Negation as failure: \+ G desugars to ( G -> fail ; true ). good is
% not bad (1); bad is (2). R = 12.
test(selfhost_negation_as_failure, [condition(clang_available)]) :-
    obj_dir(Dir),
    directory_file_path(Dir, 'cgfull.wamo', CompWamo),
    write_wam_object([wam_bootstrap_compiler:cgfull/2], [wamo_entry(cgfull/2)], CompWamo),
    directory_file_path(Dir, 'eval_host_bin', Host),
    ( exists_file(Host) -> true ; build_eval_host(Dir, Host) ),
    directory_file_path(Dir, 'cgneg_src.txt', SrcPath),
    setup_call_cleanup(open(SrcPath, write, S0),
        write(S0, '[(main0(R) :- n(good, A), n(bad, B), R is A * 10 + B), (n(X, R2) :- (\\+ X == bad -> R2 = 1 ; R2 = 2))]'),
        close(S0)),
    process_create(Host, [CompWamo, SrcPath],
        [stdout(pipe(S)), stderr(std), process(Pid)]),
    read_string(S, _, Out),
    close(S),
    process_wait(Pid, exit(Status)),
    assertion(Status == 0),
    assertion(Out == "12\n"),
    !.

% Text-family whitelist: sub_atom slices, the slice compares, and its
% length feeds arithmetic. sub_atom(hello, 1, 3, _, ell) -> 3 + 10.
test(selfhost_sub_atom_slices, [condition(clang_available)]) :-
    obj_dir(Dir),
    directory_file_path(Dir, 'cgfull.wamo', CompWamo),
    write_wam_object([wam_bootstrap_compiler:cgfull/2], [wamo_entry(cgfull/2)], CompWamo),
    directory_file_path(Dir, 'eval_host_bin', Host),
    ( exists_file(Host) -> true ; build_eval_host(Dir, Host) ),
    directory_file_path(Dir, 'cgsub_src.txt', SrcPath),
    setup_call_cleanup(open(SrcPath, write, S0),
        write(S0, '[(main0(R) :- sub_atom(hello, 1, 3, _, S), atom_length(S, L), (S == ell -> R is L + 10 ; R = 0))]'),
        close(S0)),
    process_create(Host, [CompWamo, SrcPath],
        [stdout(pipe(S)), stderr(std), process(Pid)]),
    read_string(S, _, Out),
    close(S),
    process_wait(Pid, exit(Status)),
    assertion(Status == 0),
    assertion(Out == "13\n"),
    !.

% call/N meta-call emission: the goal stages into A1 (extras A2..) with
% the meta sentinel (op1 = -1), and the serializer emits the meta-call
% table so dispatch resolves the object's OWN predicates -- a compound
% goal built at runtime (add1(5) -> 6) and an atom goal with an extra
% arg (seven -> 7). R = 67.
test(selfhost_call_n_meta, [condition(clang_available)]) :-
    obj_dir(Dir),
    directory_file_path(Dir, 'cgfull.wamo', CompWamo),
    write_wam_object([wam_bootstrap_compiler:cgfull/2], [wamo_entry(cgfull/2)], CompWamo),
    directory_file_path(Dir, 'eval_host_bin', Host),
    ( exists_file(Host) -> true ; build_eval_host(Dir, Host) ),
    directory_file_path(Dir, 'cgcall_src.txt', SrcPath),
    setup_call_cleanup(open(SrcPath, write, S0),
        write(S0, '[(main0(R) :- G = add1(5), call(G, V), call(seven, W), R is V * 10 + W), (add1(X, Y) :- Y is X + 1), seven(7)]'),
        close(S0)),
    process_create(Host, [CompWamo, SrcPath],
        [stdout(pipe(S)), stderr(std), process(Pid)]),
    read_string(S, _, Out),
    close(S),
    process_wait(Pid, exit(Status)),
    assertion(Status == 0),
    assertion(Out == "67\n"),
    !.

% The assert family reads back through call/1's dynamic-store fallback:
% k/1 is not in the object's meta table, so dispatch consults the
% process-global clause store. The COMPLETE-goal form is the supported
% idiom (the store iterator reads the goal from reg0; a call/N closure
% with extra args fails cleanly there by design -- see the dispatcher's
% atom_consult note). assertz(k(41)), G = k(V), call(G) -> 42.
test(selfhost_assertz_call_readback, [condition(clang_available)]) :-
    obj_dir(Dir),
    directory_file_path(Dir, 'cgfull.wamo', CompWamo),
    write_wam_object([wam_bootstrap_compiler:cgfull/2], [wamo_entry(cgfull/2)], CompWamo),
    directory_file_path(Dir, 'eval_host_bin', Host),
    ( exists_file(Host) -> true ; build_eval_host(Dir, Host) ),
    directory_file_path(Dir, 'cgaz_src.txt', SrcPath),
    setup_call_cleanup(open(SrcPath, write, S0),
        write(S0, '[(main0(R) :- assertz(k(41)), G = k(V), call(G), R is V + 1)]'),
        close(S0)),
    process_create(Host, [CompWamo, SrcPath],
        [stdout(pipe(S)), stderr(std), process(Pid)]),
    read_string(S, _, Out),
    close(S),
    process_wait(Pid, exit(Status)),
    assertion(Status == 0),
    assertion(Out == "42\n"),
    !.

% findall emission: begin_aggregate(collect) / goal / end_aggregate
% compile from source and collect over a fact table. Sum 6, length 3
% -> 63.
test(selfhost_findall_collects, [condition(clang_available)]) :-
    obj_dir(Dir),
    directory_file_path(Dir, 'cgfull.wamo', CompWamo),
    write_wam_object([wam_bootstrap_compiler:cgfull/2], [wamo_entry(cgfull/2)], CompWamo),
    directory_file_path(Dir, 'eval_host_bin', Host),
    ( exists_file(Host) -> true ; build_eval_host(Dir, Host) ),
    directory_file_path(Dir, 'cgfa_src.txt', SrcPath),
    setup_call_cleanup(open(SrcPath, write, S0),
        write(S0, '[(main0(R) :- findall(X, m(X), L), sum_list(L, S), length(L, N2), R is S * 10 + N2), m(1), m(2), m(3)]'),
        close(S0)),
    process_create(Host, [CompWamo, SrcPath],
        [stdout(pipe(S)), stderr(std), process(Pid)]),
    read_string(S, _, Out),
    close(S),
    process_wait(Pid, exit(Status)),
    assertion(Status == 0),
    assertion(Out == "63\n"),
    !.

% Zero solutions is begin_aggregate's edge case (the continuation PC is
% computed by the forward scan at begin time, since end_aggregate never
% runs): an empty findall binds [] and execution continues. -> 100.
test(selfhost_findall_empty, [condition(clang_available)]) :-
    obj_dir(Dir),
    directory_file_path(Dir, 'cgfull.wamo', CompWamo),
    write_wam_object([wam_bootstrap_compiler:cgfull/2], [wamo_entry(cgfull/2)], CompWamo),
    directory_file_path(Dir, 'eval_host_bin', Host),
    ( exists_file(Host) -> true ; build_eval_host(Dir, Host) ),
    directory_file_path(Dir, 'cgfe_src.txt', SrcPath),
    setup_call_cleanup(open(SrcPath, write, S0),
        write(S0, '[(main0(R) :- findall(X, m(X), L), length(L, N2), (L == [] -> R is N2 + 100 ; R = 0)), (m(1) :- fail)]'),
        close(S0)),
    process_create(Host, [CompWamo, SrcPath],
        [stdout(pipe(S)), stderr(std), process(Pid)]),
    read_string(S, _, Out),
    close(S),
    process_wait(Pid, exit(Status)),
    assertion(Status == 0),
    assertion(Out == "100\n"),
    !.

% cgfullm: cgfull + a MULTI-ENTRY name table (the eval compiler the
% plawk CLI ships). Byte-compat guard: a single-predicate source
% serializes IDENTICALLY through both (NE=1, first predicate named,
% label 0), so content-dedup handles and every existing single-grammar
% compile are unchanged -- and cgfull stays the self-host oracle.
test(cgfullm_single_pred_byte_identical) :-
    Clauses = [(sq(X, R) :- R is X * X)],
    cgfull_term(Clauses, W1),
    cgfullm_term(Clauses, W2),
    assertion(W1 == W2),
    !.

% A multi-predicate source gains one entry row per predicate group
% ("name/arity" -> its group label), which is what lets dyncall_at@name
% resolve ANY predicate of a runtime-compiled object, not just the
% first.
test(cgfullm_multi_pred_entry_table) :-
    Clauses = [(sq(X, R) :- R is X * X), (dbl(X2, R2) :- R2 is X2 * 2)],
    cgfullm_term(Clauses, Wamo),
    atomic_list_concat(Lines, '\n', Wamo),
    Lines = ['WAMO', '2', '0', NE, E1Name, E1Lbl, E2Name, E2Lbl | _],
    assertion(NE == '2'),
    assertion(E1Name == '4 sq/2'),
    assertion(E1Lbl == '0'),
    assertion(E2Name == '5 dbl/2'),
    assertion(E2Lbl == '1'),
    !.

% Nested arithmetic in the loaded compiler: the is-expression is staged
% with c_operand (build_struct + X-temp deferral), so arbitrarily nested
% expressions compile. (X + Y) * (X - 1) + 100 with X=3, Y=4 -> 114.
test(selfhost_codegen_stage_d_nested_arith, [condition(clang_available)]) :-
    obj_dir(Dir),
    directory_file_path(Dir, 'cgfull.wamo', CompWamo),
    write_wam_object([wam_bootstrap_compiler:cgfull/2], [wamo_entry(cgfull/2)], CompWamo),
    directory_file_path(Dir, 'eval_host_bin', Host),
    ( exists_file(Host) -> true ; build_eval_host(Dir, Host) ),
    directory_file_path(Dir, 'cgna_src.txt', SrcPath),
    setup_call_cleanup(open(SrcPath, write, S0),
        write(S0, '[(main0(R) :- X = 3, Y = 4, R is (X + Y) * (X - 1) + 100)]'),
        close(S0)),
    process_create(Host, [CompWamo, SrcPath],
        [stdout(pipe(S)), stderr(std), process(Pid)]),
    read_string(S, _, Out),
    close(S),
    process_wait(Pid, exit(Status)),
    assertion(Status == 0),
    assertion(Out == "114\n"),
    !.

% Fail-fast compile diagnostic: an unsupported goal (findall/3 is neither
% a whitelisted builtin nor a defined predicate) must make the loaded
% compile ABORT immediately via the f_goal catch-all throw -- not fail
% into catastrophic backtracking through the compile's stale choice
% points (a silent multi-minute hang before this guard; this test would
% time the suite out on the old behavior).
test(selfhost_codegen_fail_fast_on_unsupported, [condition(clang_available)]) :-
    obj_dir(Dir),
    directory_file_path(Dir, 'cgfull.wamo', CompWamo),
    write_wam_object([wam_bootstrap_compiler:cgfull/2], [wamo_entry(cgfull/2)], CompWamo),
    directory_file_path(Dir, 'eval_host_bin', Host),
    ( exists_file(Host) -> true ; build_eval_host(Dir, Host) ),
    directory_file_path(Dir, 'cgff_src.txt', SrcPath),
    setup_call_cleanup(open(SrcPath, write, S0),
        write(S0, '[(main0(R) :- findall(X, foo(X), L), length(L, R))]'),
        close(S0)),
    process_create(Host, [CompWamo, SrcPath],
        [stdout(pipe(S)), stderr(std), process(Pid)]),
    read_string(S, _, Out),
    close(S),
    process_wait(Pid, exit(Status)),
    assertion(Status \== 0),
    assertion(Out == ""),
    !.

% Finding no. 10 regression: an UNCAUGHT throw must ABORT the machine, not
% act as a failure. Before the @wam_uncaught flag, the run loop backtracked
% into live choice points (clearing halted) and re-executed over the
% half-unwound state; with enough choice-point structure (the mgl clause
% below adds an ITE-recursive predicate compiled before the throwing one)
% the corrupted state spun forever inside the append builtin. The compile
% of this source throws cg_unsupported_goal(madd, 3) -- madd is undefined
% here -- and the eval host must exit nonzero IMMEDIATELY (this test hung
% for minutes on the old behavior).
test(selfhost_uncaught_throw_aborts, [condition(clang_available)]) :-
    obj_dir(Dir),
    directory_file_path(Dir, 'cgfull.wamo', CompWamo),
    write_wam_object([wam_bootstrap_compiler:cgfull/2], [wamo_entry(cgfull/2)], CompWamo),
    directory_file_path(Dir, 'eval_host_bin', Host),
    ( exists_file(Host) -> true ; build_eval_host(Dir, Host) ),
    directory_file_path(Dir, 'cgut_src.txt', SrcPath),
    setup_call_cleanup(open(SrcPath, write, S0),
        write(S0, '[(main0(R) :- R = 5), (mgl(G, L) :- (G =.. [F, X, Y], F == '','' -> mgl(X, L1), mgl(Y, L2), append(L1, L2, L) ; L = [G])), mcol([], A, A), (mcol([G|Gs], A0, A2) :- (G =.. [F, _, E], F == is, functor(E, Op, 2) -> madd(Op, A0, A1) ; A1 = A0), mcol(Gs, A1, A2))]'),
        close(S0)),
    process_create(Host, [CompWamo, SrcPath],
        [stdout(pipe(S)), stderr(std), process(Pid)]),
    read_string(S, _, Out),
    close(S),
    process_wait(Pid, exit(Status)),
    assertion(Status \== 0),
    assertion(Out == ""),
    !.

% Register-file ceiling fix: a clause with 20 permanent variables (Y1..Y20)
% loads and runs. Before enlarging the register file to [128 x %Value], Y17+
% overflowed the 64-slot array and corrupted memory. Must yield 210.
test(register_file_many_permanents, [condition(clang_available)]) :-
    obj_dir(Dir),
    directory_file_path(Dir, 'host_bin', Host),
    ( exists_file(Host) -> true ; build_host(Dir, Host) ),
    directory_file_path(Dir, 'manyperm.wamo', W),
    write_wam_object([user:manyperm/1], [wamo_entry(manyperm/1)], W),
    run_host(Host, W, Out, 0),
    assertion(Out == "210\n"),
    !.

% Loaded clause indexing: switch_on_constant is REAL dispatch in loaded
% objects now (tag 25 + a trailing key->label table the loader turns
% into the same %SwitchEntry shape the AOT dispatcher uses). swmain/1
% exercises quoted-key entries ('=:=', '\==' -- only match if the writer
% unquoted them), a strict switch MISS failing cleanly into an ITE else
% branch, and an unbound first argument skipping the switch into the
% try_me_else chain. The structural assertions pin the encoding: a real
% tag-25 row (not the old nop 26) and the trailing section present.
test(loaded_switch_on_constant_dispatch, [condition(clang_available)]) :-
    obj_dir(Dir),
    directory_file_path(Dir, 'host_bin', Host),
    ( exists_file(Host) -> true ; build_host(Dir, Host) ),
    directory_file_path(Dir, 'swkey.wamo', W),
    write_wam_object([user:swmain/1, user:swkey/2],
                     [wamo_entry(swmain/1)], W),
    read_file_to_string(W, WS, [encoding(octet)]),
    assertion(sub_string(WS, _, _, _, "\n25 0 0 0\n")),
    run_host(Host, W, Out, 0),
    assertion(Out == "5215523\n"),
    !.

% Finding no. 9: get_value var-var linking + append bare-var tail seed
% (see dlknot/1 above). 1004 proves both variable links held through the
% difference-list chain; the broken runtime fails the entry instead.
test(get_value_var_var_and_append_var_seed, [condition(clang_available)]) :-
    obj_dir(Dir),
    directory_file_path(Dir, 'host_bin', Host),
    ( exists_file(Host) -> true ; build_host(Dir, Host) ),
    directory_file_path(Dir, 'dlknot.wamo', W),
    write_wam_object([user:dlknot/1, user:knot9/2, user:em9/3],
                     [wamo_entry(dlknot/1)], W),
    run_host(Host, W, Out, 0),
    assertion(Out == "1004\n"),
    !.

% Milestone 3b-db PR 3: rule bodies in a loaded object. asserted (H :- B)
% clauses run their bodies when the head is called -- builtin bodies with
% head<->body var sharing (rb_double, rb_clamp), a rule whose body calls
% another rule with a chained variable (rb_chain), and a body mixing a fact
% call with a builtin (rb_mix).
test(rule_bodies_in_object, [condition(clang_available)]) :-
    obj_dir(Dir),
    directory_file_path(Dir, 'host_bin', Host),
    ( exists_file(Host) -> true ; build_host(Dir, Host) ),
    forall(member(Name, [rb_double, rb_chain, rb_clamp, rb_mix]),
           ( PI = Name/1,
             directory_file_path(Dir, Name, Base),
             atom_concat(Base, '.wamo', W),
             write_wam_object([user:PI], [wamo_entry(PI)], W),
             run_host(Host, W, Out, 0),
             assertion(Out == "42\n") )),
    !.

% Rule-body control constructs: if-then-else (both directions), disjunction
% (both branches), and negation-as-failure (both ways), deterministic.
test(rule_body_control_in_object, [condition(clang_available)]) :-
    obj_dir(Dir),
    directory_file_path(Dir, 'host_bin', Host),
    ( exists_file(Host) -> true ; build_host(Dir, Host) ),
    forall(member(Name-Expected,
               [ rc_ite_t-"1\n",   % if-then-else, condition true
                 rc_ite_f-"0\n",   % if-then-else, condition false -> else
                 rc_disj1-"11\n",  % disjunction, first branch
                 rc_disj2-"22\n",  % disjunction, first fails -> second
                 rc_naf1-"1\n",    % \+ of a false goal -> succeeds
                 rc_naf0-"0\n" ]), % \+ of a true goal -> else
           ( PI = Name/1,
             directory_file_path(Dir, Name, Base),
             atom_concat(Base, '.wamo', W),
             write_wam_object([user:PI], [wamo_entry(PI)], W),
             run_host(Host, W, Out, 0),
             assertion(Out == Expected) )),
    !.

:- end_tests(wam_object).

% --- helpers ---------------------------------------------------------------

obj_dir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_wam_object', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

% Build a host binary: a trivial module carrying the .wamo loader, plus a
% main() that loads argv[1], runs the entry, and prints the integer.
build_host(Dir, Host) :-
    directory_file_path(Dir, 'host.ll', LL),
    with_output_to(string(_),
        write_wam_llvm_project([user:answer/1],
            [module_name(wam_object_host), emit_wamo_loader(true)], LL)),
    host_main_ir(MainIR),
    setup_call_cleanup(
        open(LL, append, S, [encoding(utf8)]),
        write(S, MainIR),
        close(S)),
    directory_file_path(Dir, 'host_bin', Host),
    format(atom(Cmd), 'clang -w -O2 ~w -o ~w -lm 2>&1', [LL, Host]),
    process_create(path(sh), ['-c', Cmd],
        [stdout(pipe(CS)), stderr(std), process(Pid)]),
    read_string(CS, _, ClangOut),
    close(CS),
    process_wait(Pid, Status),
    ( Status == exit(0) -> true
    ; throw(error(clang_failed(ClangOut), _)) ).

host_main_ir(
'\n@.wam_object_fmt = private constant [6 x i8] c"%lld\\0A\\00"\n\n\c
define i32 @main(i32 %argc, i8** %argv) {\n\c
entry:\n\c
  %p1 = getelementptr i8*, i8** %argv, i64 1\n\c
  %path = load i8*, i8** %p1\n\c
  %obj = call { %WamState*, i32 } @wam_object_load(i8* %path)\n\c
  %vm = extractvalue { %WamState*, i32 } %obj, 0\n\c
  %pc = extractvalue { %WamState*, i32 } %obj, 1\n\c
  %vm_null = icmp eq %WamState* %vm, null\n\c
  br i1 %vm_null, label %load_fail, label %run\n\c
run:\n\c
  %r = call { i64, i1 } @wam_object_call_i64(%WamState* %vm, i32 %pc, i32 0, %Value* null, i32 0)\n\c
  %val = extractvalue { i64, i1 } %r, 0\n\c
  %ok = extractvalue { i64, i1 } %r, 1\n\c
  br i1 %ok, label %print, label %run_fail\n\c
print:\n\c
  %fmt = getelementptr [6 x i8], [6 x i8]* @.wam_object_fmt, i32 0, i32 0\n\c
  %pr = call i32 (i8*, ...) @printf(i8* %fmt, i64 %val)\n\c
  ret i32 0\n\c
load_fail:\n\c
  ret i32 20\n\c
run_fail:\n\c
  ret i32 21\n\c
}\n').

% Build a host that loads argv[1], resolves "answer/1" and "answer_swapped/1"
% by name against the object's entry table, and calls each in turn.
build_multi_host(Dir, Host) :-
    directory_file_path(Dir, 'multi_host.ll', LL),
    with_output_to(string(_),
        write_wam_llvm_project([user:answer/1],
            [module_name(wam_object_multi_host), emit_wamo_loader(true)], LL)),
    multi_host_main_ir(MainIR),
    setup_call_cleanup(
        open(LL, append, S, [encoding(utf8)]),
        write(S, MainIR),
        close(S)),
    directory_file_path(Dir, 'multi_host_bin', Host),
    clang_link(LL, Host).

% A host that resolves a name absent from the table -> -1 -> resolve_fail (22).
build_multi_miss_host(Dir, Host) :-
    directory_file_path(Dir, 'multi_miss.ll', LL),
    with_output_to(string(_),
        write_wam_llvm_project([user:answer/1],
            [module_name(wam_object_multi_miss), emit_wamo_loader(true)], LL)),
    multi_miss_main_ir(MainIR),
    setup_call_cleanup(
        open(LL, append, S, [encoding(utf8)]),
        write(S, MainIR),
        close(S)),
    clang_link(LL, Host).

% Build a host that loads argv[1], calls the entry via
% @wam_object_call_record with a 2-field shape (i64, f64), and prints the
% deserialized fields.
build_record_host(Dir, Host) :-
    directory_file_path(Dir, 'record_host.ll', LL),
    with_output_to(string(_),
        write_wam_llvm_project([user:answer/1],
            [module_name(wam_object_record_host), emit_wamo_loader(true)], LL)),
    record_host_main_ir(MainIR),
    setup_call_cleanup(
        open(LL, append, S, [encoding(utf8)]),
        write(S, MainIR),
        close(S)),
    directory_file_path(Dir, 'record_host_bin', Host),
    clang_link(LL, Host).

record_host_main_ir(
'\n@.rec_ifmt = private constant [6 x i8] c"%lld\\0A\\00"\n\c
@.rec_ffmt = private constant [6 x i8] c"%.1f\\0A\\00"\n\c
@.rec_types = private constant [2 x i8] c"\\00\\01"\n\n\c
define i32 @main(i32 %argc, i8** %argv) {\n\c
entry:\n\c
  %p1 = getelementptr i8*, i8** %argv, i64 1\n\c
  %path = load i8*, i8** %p1\n\c
  %obj = call { %WamState*, i32 } @wam_object_load(i8* %path)\n\c
  %vm = extractvalue { %WamState*, i32 } %obj, 0\n\c
  %pc = extractvalue { %WamState*, i32 } %obj, 1\n\c
  %vm_null = icmp eq %WamState* %vm, null\n\c
  br i1 %vm_null, label %load_fail, label %run\n\c
run:\n\c
  %slots = alloca i64, i32 2\n\c
  %lens = alloca i64, i32 2\n\c
  %tc = getelementptr [2 x i8], [2 x i8]* @.rec_types, i32 0, i32 0\n\c
  %ok = call i1 @wam_object_call_record(%WamState* %vm, i32 %pc, i32 0, %Value* null, i32 0, i32 2, i8* %tc, i64* %slots, i64* %lens)\n\c
  br i1 %ok, label %print, label %run_fail\n\c
print:\n\c
  %s0 = getelementptr i64, i64* %slots, i64 0\n\c
  %v0 = load i64, i64* %s0\n\c
  %ifmt = getelementptr [6 x i8], [6 x i8]* @.rec_ifmt, i32 0, i32 0\n\c
  %pi = call i32 (i8*, ...) @printf(i8* %ifmt, i64 %v0)\n\c
  %s1 = getelementptr i64, i64* %slots, i64 1\n\c
  %v1bits = load i64, i64* %s1\n\c
  %v1 = bitcast i64 %v1bits to double\n\c
  %ffmt = getelementptr [6 x i8], [6 x i8]* @.rec_ffmt, i32 0, i32 0\n\c
  %pf = call i32 (i8*, ...) @printf(i8* %ffmt, double %v1)\n\c
  ret i32 0\n\c
load_fail:\n\c
  ret i32 20\n\c
run_fail:\n\c
  ret i32 21\n\c
}\n').

% Build a host that calls @wam_object_call_record with a 2-field shape
% (i64, string), then prints field 0 as an integer and field 1 as a
% length-counted string (%.*s from out_slots[1] + out_lens[1]).
build_record_str_host(Dir, Host) :-
    directory_file_path(Dir, 'record_str_host.ll', LL),
    with_output_to(string(_),
        write_wam_llvm_project([user:answer/1],
            [module_name(wam_object_record_str_host), emit_wamo_loader(true)], LL)),
    record_str_host_main_ir(MainIR),
    setup_call_cleanup(
        open(LL, append, S, [encoding(utf8)]),
        write(S, MainIR),
        close(S)),
    directory_file_path(Dir, 'record_str_host_bin', Host),
    clang_link(LL, Host).

record_str_host_main_ir(
'\n@.rs_ifmt = private constant [6 x i8] c"%lld\\0A\\00"\n\c
@.rs_sfmt = private constant [6 x i8] c"%.*s\\0A\\00"\n\c
@.rs_types = private constant [2 x i8] c"\\00\\02"\n\n\c
define i32 @main(i32 %argc, i8** %argv) {\n\c
entry:\n\c
  %p1 = getelementptr i8*, i8** %argv, i64 1\n\c
  %path = load i8*, i8** %p1\n\c
  %obj = call { %WamState*, i32 } @wam_object_load(i8* %path)\n\c
  %vm = extractvalue { %WamState*, i32 } %obj, 0\n\c
  %pc = extractvalue { %WamState*, i32 } %obj, 1\n\c
  %vm_null = icmp eq %WamState* %vm, null\n\c
  br i1 %vm_null, label %load_fail, label %run\n\c
run:\n\c
  %slots = alloca i64, i32 2\n\c
  %lens = alloca i64, i32 2\n\c
  %tc = getelementptr [2 x i8], [2 x i8]* @.rs_types, i32 0, i32 0\n\c
  %ok = call i1 @wam_object_call_record(%WamState* %vm, i32 %pc, i32 0, %Value* null, i32 0, i32 2, i8* %tc, i64* %slots, i64* %lens)\n\c
  br i1 %ok, label %print, label %run_fail\n\c
print:\n\c
  %s0 = getelementptr i64, i64* %slots, i64 0\n\c
  %v0 = load i64, i64* %s0\n\c
  %ifmt = getelementptr [6 x i8], [6 x i8]* @.rs_ifmt, i32 0, i32 0\n\c
  %pi = call i32 (i8*, ...) @printf(i8* %ifmt, i64 %v0)\n\c
  %s1 = getelementptr i64, i64* %slots, i64 1\n\c
  %v1 = load i64, i64* %s1\n\c
  %ptr = inttoptr i64 %v1 to i8*\n\c
  %l1 = getelementptr i64, i64* %lens, i64 1\n\c
  %len = load i64, i64* %l1\n\c
  %len32 = trunc i64 %len to i32\n\c
  %sfmt = getelementptr [6 x i8], [6 x i8]* @.rs_sfmt, i32 0, i32 0\n\c
  %ps = call i32 (i8*, ...) @printf(i8* %sfmt, i32 %len32, i8* %ptr)\n\c
  ret i32 0\n\c
load_fail:\n\c
  ret i32 20\n\c
run_fail:\n\c
  ret i32 21\n\c
}\n').

% Build a host that allocates an i64 assoc table, calls
% @wam_object_call_assoc to populate it from the grammar's returned pairs,
% then reads keys 1,2,3 back and prints their values.
build_record_assoc_host(Dir, Host) :-
    directory_file_path(Dir, 'record_assoc_host.ll', LL),
    with_output_to(string(_),
        write_wam_llvm_project([user:answer/1],
            [module_name(wam_object_record_assoc_host), emit_wamo_loader(true)], LL)),
    record_assoc_host_main_ir(MainIR),
    setup_call_cleanup(
        open(LL, append, S, [encoding(utf8)]),
        write(S, MainIR),
        close(S)),
    directory_file_path(Dir, 'record_assoc_host_bin', Host),
    clang_link(LL, Host).

record_assoc_host_main_ir(
'\n@.ra_ifmt = private constant [6 x i8] c"%lld\\0A\\00"\n\n\c
define i32 @main(i32 %argc, i8** %argv) {\n\c
entry:\n\c
  %p1 = getelementptr i8*, i8** %argv, i64 1\n\c
  %path = load i8*, i8** %p1\n\c
  %obj = call { %WamState*, i32 } @wam_object_load(i8* %path)\n\c
  %vm = extractvalue { %WamState*, i32 } %obj, 0\n\c
  %pc = extractvalue { %WamState*, i32 } %obj, 1\n\c
  %vm_null = icmp eq %WamState* %vm, null\n\c
  br i1 %vm_null, label %load_fail, label %run\n\c
run:\n\c
  %table = call %WamAssocI64Table* @wam_assoc_i64_new(i64 64)\n\c
  %ok = call i1 @wam_object_call_assoc(%WamState* %vm, i32 %pc, i32 0, %Value* null, i32 0, %WamAssocI64Table* %table)\n\c
  br i1 %ok, label %print, label %run_fail\n\c
print:\n\c
  %v1 = call i64 @wam_assoc_i64_get(%WamAssocI64Table* %table, i64 1)\n\c
  %v2 = call i64 @wam_assoc_i64_get(%WamAssocI64Table* %table, i64 2)\n\c
  %v3 = call i64 @wam_assoc_i64_get(%WamAssocI64Table* %table, i64 3)\n\c
  %fmt = getelementptr [6 x i8], [6 x i8]* @.ra_ifmt, i32 0, i32 0\n\c
  %p_1 = call i32 (i8*, ...) @printf(i8* %fmt, i64 %v1)\n\c
  %p_2 = call i32 (i8*, ...) @printf(i8* %fmt, i64 %v2)\n\c
  %p_3 = call i32 (i8*, ...) @printf(i8* %fmt, i64 %v3)\n\c
  ret i32 0\n\c
load_fail:\n\c
  ret i32 20\n\c
run_fail:\n\c
  ret i32 21\n\c
}\n').

% Build a host that loads argv[1], runs the entry via @wam_object_call_bytes,
% and writes the returned bytes verbatim (length-counted, %.*s -- no trailing
% newline, so the captured output equals the grammar's byte string exactly).
build_bytes_host(Dir, Host) :-
    directory_file_path(Dir, 'bytes_host.ll', LL),
    with_output_to(string(_),
        write_wam_llvm_project([user:answer/1],
            [module_name(wam_object_bytes_host), emit_wamo_loader(true)], LL)),
    bytes_host_main_ir(MainIR),
    setup_call_cleanup(
        open(LL, append, S, [encoding(utf8)]),
        write(S, MainIR),
        close(S)),
    directory_file_path(Dir, 'bytes_host_bin', Host),
    clang_link(LL, Host).

bytes_host_main_ir(
'\n@.by_sfmt = private constant [5 x i8] c"%.*s\\00"\n\n\c
define i32 @main(i32 %argc, i8** %argv) {\n\c
entry:\n\c
  %p1 = getelementptr i8*, i8** %argv, i64 1\n\c
  %path = load i8*, i8** %p1\n\c
  %obj = call { %WamState*, i32 } @wam_object_load(i8* %path)\n\c
  %vm = extractvalue { %WamState*, i32 } %obj, 0\n\c
  %pc = extractvalue { %WamState*, i32 } %obj, 1\n\c
  %vm_null = icmp eq %WamState* %vm, null\n\c
  br i1 %vm_null, label %load_fail, label %run\n\c
run:\n\c
  %r = call { i8*, i64, i1 } @wam_object_call_bytes(%WamState* %vm, i32 %pc, i32 0, %Value* null, i32 0)\n\c
  %ptr = extractvalue { i8*, i64, i1 } %r, 0\n\c
  %len = extractvalue { i8*, i64, i1 } %r, 1\n\c
  %ok = extractvalue { i8*, i64, i1 } %r, 2\n\c
  br i1 %ok, label %print, label %run_fail\n\c
print:\n\c
  %len32 = trunc i64 %len to i32\n\c
  %fmt = getelementptr [5 x i8], [5 x i8]* @.by_sfmt, i32 0, i32 0\n\c
  %ps = call i32 (i8*, ...) @printf(i8* %fmt, i32 %len32, i8* %ptr)\n\c
  ret i32 0\n\c
load_fail:\n\c
  ret i32 20\n\c
run_fail:\n\c
  ret i32 21\n\c
}\n').

% Build a host for the eval/compile pipeline: argv[1] = compiler .wamo,
% argv[2] = source (here itself a .wamo). It lazy-loads the compiler
% (@wam_object_load_cached), reads the source, runs @wam_object_eval to
% compile+load it, then runs the resulting object and prints the integer.
build_eval_host(Dir, Host) :-
    directory_file_path(Dir, 'eval_host.ll', LL),
    with_output_to(string(_),
        write_wam_llvm_project([user:answer/1],
            [module_name(wam_object_eval_host), emit_wamo_loader(true)], LL)),
    eval_host_main_ir(MainIR),
    setup_call_cleanup(
        open(LL, append, S, [encoding(utf8)]),
        write(S, MainIR),
        close(S)),
    directory_file_path(Dir, 'eval_host_bin', Host),
    clang_link(LL, Host).

eval_host_main_ir(
'\n@.ev_fmt = private constant [6 x i8] c"%lld\\0A\\00"\n\n\c
define i32 @main(i32 %argc, i8** %argv) {\n\c
entry:\n\c
  %p1 = getelementptr i8*, i8** %argv, i64 1\n  %cpath = load i8*, i8** %p1\n\c
  %p2 = getelementptr i8*, i8** %argv, i64 2\n  %spath = load i8*, i8** %p2\n\c
  %cobj = call { %WamState*, i32 } @wam_object_load_cached(i8* %cpath)\n\c
  %cvm = extractvalue { %WamState*, i32 } %cobj, 0\n\c
  %cpc = extractvalue { %WamState*, i32 } %cobj, 1\n\c
  %cnull = icmp eq %WamState* %cvm, null\n  br i1 %cnull, label %fail, label %readsrc\n\c
readsrc:\n\c
  %totp = alloca i64\n\c
  %sbuf = call i8* @wamo_read_file(i8* %spath, i64* %totp)\n\c
  %sbnull = icmp eq i8* %sbuf, null\n  br i1 %sbnull, label %fail, label %doeval\n\c
doeval:\n\c
  %slen = load i64, i64* %totp\n\c
  %robj = call { %WamState*, i32 } @wam_object_eval(%WamState* %cvm, i32 %cpc, i8* %sbuf, i64 %slen)\n\c
  %rvm = extractvalue { %WamState*, i32 } %robj, 0\n\c
  %rpc = extractvalue { %WamState*, i32 } %robj, 1\n\c
  %rnull = icmp eq %WamState* %rvm, null\n  br i1 %rnull, label %fail, label %runit\n\c
runit:\n\c
  %res = call { i64, i1 } @wam_object_call_i64(%WamState* %rvm, i32 %rpc, i32 0, %Value* null, i32 0)\n\c
  %val = extractvalue { i64, i1 } %res, 0\n\c
  %ok = extractvalue { i64, i1 } %res, 1\n  br i1 %ok, label %prn, label %fail\n\c
prn:\n\c
  %fmt = getelementptr [6 x i8], [6 x i8]* @.ev_fmt, i32 0, i32 0\n\c
  %pr = call i32 (i8*, ...) @printf(i8* %fmt, i64 %val)\n  ret i32 0\n\c
fail:\n  ret i32 21\n}\n').

run_dump(Dump, Comp, Src, Bytes) :-
    process_create(Dump, [Comp, Src],
        [stdout(pipe(S)), stderr(std), process(Pid)]),
    read_string(S, _, Bytes),
    close(S),
    process_wait(Pid, exit(Status)),
    Status == 0.

% A bytes-dump host: load a compiler object, run it on a source file via
% @wam_object_call_bytes, print the emitted object text. This is stage 1
% of the eval host in isolation -- it lets the capstone test CHAIN
% compilers: dump(gen_n, src) emits gen_{n+1}, which is itself loadable.
build_dump_host(Dir, Host) :-
    directory_file_path(Dir, 'dump_host.ll', LL),
    with_output_to(string(_),
        write_wam_llvm_project([user:answer/1],
            [module_name(wam_object_dump_host), emit_wamo_loader(true)], LL)),
    dump_host_main_ir(MainIR),
    setup_call_cleanup(
        open(LL, append, S, [encoding(utf8)]),
        write(S, MainIR),
        close(S)),
    directory_file_path(Dir, 'dump_host_bin', Host),
    clang_link(LL, Host).

dump_host_main_ir(
'\n@.dh_fmt = private constant [5 x i8] c"%.*s\\00"\n\n\c
define i32 @main(i32 %argc, i8** %argv) {\n\c
entry:\n\c
  %p1 = getelementptr i8*, i8** %argv, i64 1\n  %cp = load i8*, i8** %p1\n\c
  %p2 = getelementptr i8*, i8** %argv, i64 2\n  %sp = load i8*, i8** %p2\n\c
  %obj = call { %WamState*, i32 } @wam_object_load(i8* %cp)\n\c
  %vm = extractvalue { %WamState*, i32 } %obj, 0\n\c
  %pc = extractvalue { %WamState*, i32 } %obj, 1\n\c
  %vmn = icmp eq %WamState* %vm, null\n  br i1 %vmn, label %lf, label %rd\n\c
rd:\n\c
  %totp = alloca i64\n\c
  %sbuf = call i8* @wamo_read_file(i8* %sp, i64* %totp)\n\c
  %sbn = icmp eq i8* %sbuf, null\n  br i1 %sbn, label %lf, label %go\n\c
go:\n\c
  %slen = load i64, i64* %totp\n\c
  %sid = call i64 @wam_intern_atom(i8* %sbuf, i64 %slen)\n\c
  %v0 = insertvalue %Value undef, i32 0, 0\n\c
  %sv = insertvalue %Value %v0, i64 %sid, 1\n\c
  %args = alloca %Value, i32 1\n\c
  store %Value %sv, %Value* %args\n\c
  %r = call { i8*, i64, i1 } @wam_object_call_bytes(%WamState* %vm, i32 %pc, i32 1, %Value* %args, i32 1)\n\c
  %ptr = extractvalue { i8*, i64, i1 } %r, 0\n\c
  %len = extractvalue { i8*, i64, i1 } %r, 1\n\c
  %ok = extractvalue { i8*, i64, i1 } %r, 2\n\c
  br i1 %ok, label %pr, label %rf\n\c
pr:\n\c
  %l32 = trunc i64 %len to i32\n\c
  %f = getelementptr [5 x i8], [5 x i8]* @.dh_fmt, i32 0, i32 0\n\c
  %x = call i32 (i8*, ...) @printf(i8* %f, i32 %l32, i8* %ptr)\n  ret i32 0\n\c
lf:\n  ret i32 20\n\c
rf:\n  ret i32 21\n}\n').

clang_link(LL, Bin) :-
    format(atom(Cmd), 'clang -w -O2 ~w -o ~w -lm 2>&1', [LL, Bin]),
    process_create(path(sh), ['-c', Cmd],
        [stdout(pipe(CS)), stderr(std), process(Pid)]),
    read_string(CS, _, ClangOut),
    close(CS),
    process_wait(Pid, Status),
    ( Status == exit(0) -> true ; throw(error(clang_failed(ClangOut), _)) ).

multi_host_main_ir(
'\n@.me_fmt = private constant [6 x i8] c"%lld\\0A\\00"\n\c
@.me_n1 = private constant [8 x i8] c"answer/1"\n\c
@.me_n2 = private constant [16 x i8] c"answer_swapped/1"\n\n\c
define i32 @main(i32 %argc, i8** %argv) {\n\c
entry:\n\c
  %p1 = getelementptr i8*, i8** %argv, i64 1\n\c
  %path = load i8*, i8** %p1\n\c
  %obj = call { %WamState*, i32 } @wam_object_load(i8* %path)\n\c
  %vm = extractvalue { %WamState*, i32 } %obj, 0\n\c
  %vm_null = icmp eq %WamState* %vm, null\n\c
  br i1 %vm_null, label %load_fail, label %e1\n\c
e1:\n\c
  %n1 = getelementptr [8 x i8], [8 x i8]* @.me_n1, i32 0, i32 0\n\c
  %idx1 = call i32 @wam_object_entry_index(i8* %path, i8* %n1, i64 8)\n\c
  %bad1 = icmp slt i32 %idx1, 0\n\c
  br i1 %bad1, label %resolve_fail, label %run1\n\c
run1:\n\c
  %pc1 = call i32 @wam_label_pc(%WamState* %vm, i32 %idx1)\n\c
  %r1 = call { i64, i1 } @wam_object_call_i64(%WamState* %vm, i32 %pc1, i32 0, %Value* null, i32 0)\n\c
  %v1 = extractvalue { i64, i1 } %r1, 0\n\c
  %ok1 = extractvalue { i64, i1 } %r1, 1\n\c
  br i1 %ok1, label %print1, label %run_fail\n\c
print1:\n\c
  %fmt1 = getelementptr [6 x i8], [6 x i8]* @.me_fmt, i32 0, i32 0\n\c
  %pr1 = call i32 (i8*, ...) @printf(i8* %fmt1, i64 %v1)\n\c
  br label %e2\n\c
e2:\n\c
  %n2 = getelementptr [16 x i8], [16 x i8]* @.me_n2, i32 0, i32 0\n\c
  %idx2 = call i32 @wam_object_entry_index(i8* %path, i8* %n2, i64 16)\n\c
  %bad2 = icmp slt i32 %idx2, 0\n\c
  br i1 %bad2, label %resolve_fail, label %run2\n\c
run2:\n\c
  %pc2 = call i32 @wam_label_pc(%WamState* %vm, i32 %idx2)\n\c
  %r2 = call { i64, i1 } @wam_object_call_i64(%WamState* %vm, i32 %pc2, i32 0, %Value* null, i32 0)\n\c
  %v2 = extractvalue { i64, i1 } %r2, 0\n\c
  %ok2 = extractvalue { i64, i1 } %r2, 1\n\c
  br i1 %ok2, label %print2, label %run_fail\n\c
print2:\n\c
  %fmt2 = getelementptr [6 x i8], [6 x i8]* @.me_fmt, i32 0, i32 0\n\c
  %pr2 = call i32 (i8*, ...) @printf(i8* %fmt2, i64 %v2)\n\c
  ret i32 0\n\c
load_fail:\n\c
  ret i32 20\n\c
run_fail:\n\c
  ret i32 21\n\c
resolve_fail:\n\c
  ret i32 22\n\c
}\n').

multi_miss_main_ir(
'\n@.mm_name = private constant [7 x i8] c"nope/99"\n\n\c
define i32 @main(i32 %argc, i8** %argv) {\n\c
entry:\n\c
  %p1 = getelementptr i8*, i8** %argv, i64 1\n\c
  %path = load i8*, i8** %p1\n\c
  %obj = call { %WamState*, i32 } @wam_object_load(i8* %path)\n\c
  %vm = extractvalue { %WamState*, i32 } %obj, 0\n\c
  %vm_null = icmp eq %WamState* %vm, null\n\c
  br i1 %vm_null, label %load_fail, label %resolve\n\c
resolve:\n\c
  %n = getelementptr [7 x i8], [7 x i8]* @.mm_name, i32 0, i32 0\n\c
  %idx = call i32 @wam_object_entry_index(i8* %path, i8* %n, i64 7)\n\c
  %bad = icmp slt i32 %idx, 0\n\c
  br i1 %bad, label %resolve_fail, label %ok\n\c
ok:\n\c
  ret i32 0\n\c
load_fail:\n\c
  ret i32 20\n\c
resolve_fail:\n\c
  ret i32 22\n\c
}\n').

run_host(Host, Wamo, Out, ExpectedStatus) :-
    process_create(Host, [Wamo],
        [stdout(pipe(S)), stderr(std), process(Pid)]),
    read_string(S, _, Out),
    close(S),
    process_wait(Pid, exit(Status)),
    assertion(Status == ExpectedStatus).

% Build a host whose main() embeds the .wamo bytes as a constant and loads
% them via @wam_object_load_bytes (no file open), then runs the entry.
build_embed_host(Dir, Bytes, NBytes, Host) :-
    directory_file_path(Dir, 'embed_host.ll', LL),
    with_output_to(string(_),
        write_wam_llvm_project([user:answer/1],
            [module_name(wam_object_embed_host), emit_wamo_loader(true)], LL)),
    llvm_bytes_escape(Bytes, Escaped),
    format(atom(MainIR),
'\n@.embedded_wamo = private constant [~w x i8] c"~w"\n\n\c
define i32 @main() {\n\c
entry:\n\c
  %p = getelementptr [~w x i8], [~w x i8]* @.embedded_wamo, i32 0, i32 0\n\c
  %obj = call { %WamState*, i32 } @wam_object_load_bytes(i8* %p, i64 ~w)\n\c
  %vm = extractvalue { %WamState*, i32 } %obj, 0\n\c
  %pc = extractvalue { %WamState*, i32 } %obj, 1\n\c
  %vm_null = icmp eq %WamState* %vm, null\n\c
  br i1 %vm_null, label %load_fail, label %run\n\c
run:\n\c
  %r = call { i64, i1 } @wam_object_call_i64(%WamState* %vm, i32 %pc, i32 0, %Value* null, i32 0)\n\c
  %val = extractvalue { i64, i1 } %r, 0\n\c
  %fmt = getelementptr [6 x i8], [6 x i8]* @.wam_object_embed_fmt, i32 0, i32 0\n\c
  %pr = call i32 (i8*, ...) @printf(i8* %fmt, i64 %val)\n\c
  ret i32 0\n\c
load_fail:\n\c
  ret i32 20\n\c
}\n@.wam_object_embed_fmt = private constant [6 x i8] c"%lld\\0A\\00"\n',
        [NBytes, Escaped, NBytes, NBytes, NBytes]),
    setup_call_cleanup(
        open(LL, append, S, [encoding(utf8)]),
        write(S, MainIR),
        close(S)),
    directory_file_path(Dir, 'embed_host_bin', Host),
    format(atom(Cmd), 'clang -w -O2 ~w -o ~w -lm 2>&1', [LL, Host]),
    process_create(path(sh), ['-c', Cmd],
        [stdout(pipe(CS)), stderr(std), process(Pid)]),
    read_string(CS, _, ClangOut),
    close(CS),
    process_wait(Pid, Status),
    ( Status == exit(0) -> true ; throw(error(clang_failed(ClangOut), _)) ).

run_embed_host(Host, Out) :-
    process_create(Host, [],
        [stdout(pipe(S)), stderr(std), process(Pid)]),
    read_string(S, _, Out),
    close(S),
    process_wait(Pid, exit(Status)),
    assertion(Status == 0).

read_file_to_bytes(Path, Bytes) :-
    setup_call_cleanup(
        open(Path, read, S, [type(binary)]),
        read_stream_bytes(S, Bytes),
        close(S)).

read_stream_bytes(S, Bytes) :-
    get_byte(S, B),
    ( B == -1 -> Bytes = []
    ; Bytes = [B | Rest], read_stream_bytes(S, Rest) ).

% Escape a byte list for an LLVM c"..." string constant: printable ASCII
% (except " and backslash) verbatim, everything else as \XX hex.
llvm_bytes_escape(Bytes, Escaped) :-
    foldl(llvm_byte_escape, Bytes, [], RevCodes),
    reverse(RevCodes, Codes),
    string_codes(Escaped, Codes).

llvm_byte_escape(B, Acc, NewAcc) :-
    (   B >= 32, B =< 126, B =\= 0'", B =\= 0'\\
    ->  NewAcc = [B | Acc]
    ;   Hi is B >> 4, Lo is B /\ 0xF,
        hex_digit(Hi, HiC), hex_digit(Lo, LoC),
        NewAcc = [LoC, HiC, 0'\\ | Acc]
    ).

hex_digit(N, C) :- N < 10, !, C is 0'0 + N.
hex_digit(N, C) :- C is 0'A + (N - 10).
