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

% Milestone 6 (self-host) Stage A: a .wamo serializer written in the loadable
% subset. wamoserz/2 is a real eval-pipeline compiler entry -- compile(Src,
% Wamo) -- but Stage A ignores Src and serializes a FIXED ground instruction
% list for a 42-returning program (`ea(R):-R=42`), exercising the string-
% assembly back end end to end. It reproduces wamo_serialize/8's byte format
% using only loadable builtins (atom_codes/number_codes/append/length), so it
% runs as a loaded object; the emitted bytes load via @wam_object_eval and run
% to 42. NA/NF are 0 here (no atom/functor tables); the PC and instruction
% lists drive the rest, so this is a genuine serializer, not a string literal.
%
% Written as SMALL accumulator-threaded clauses (a code list is threaded
% through, each clause holding only a few call-spanning variables) with only
% list / enc(...) head dispatch -- deliberately avoiding two loaded-runtime
% limitations this stage surfaced: (1) a clause holding ~16+ call-spanning
% variables emits register indices past the 64-slot register file and corrupts
% memory; (2) multi-way first-argument functor dispatch across >=4 tagged
% variants (with a list-carrying variant present) mis-dispatches in loaded
% objects. Both are recorded in PLAWK_SELFHOST.md as prerequisites for later
% stages; the "small clauses, correctness over register reuse" style the design
% doc prescribes routes around both.
wamoserz(_Src, Wamo) :-
    atom_codes('ea/1', NameCodes),
    wz_serialize(0, NameCodes, 0, 0, 0, [0],
        [enc(0,42,65536,0), enc(20,0,0,0)], Codes),
    atom_codes(Wamo, Codes).

% accumulator appenders: each threads a code list, holding few call-spanning
% vars. wz_i "<int>\n", wz_a "<atom>\n", wz_n "<len> <bytes>\n", wz_si " <int>".
wz_i(N, A0, A1)  :- number_codes(N, Cs), append(A0, Cs, B), append(B, [10], A1).
wz_a(X, A0, A1)  :- atom_codes(X, Cs), append(A0, Cs, B), append(B, [10], A1).
wz_n(Cs, A0, A1) :- length(Cs, Len), number_codes(Len, LC),
    append(A0, LC, B), append(B, [32|Cs], C), append(C, [10], A1).
wz_si(N, A0, A1) :- number_codes(N, Cs), append(A0, [32|Cs], A1).

wz_serialize(EntryIdx, NameCodes, LabelIdx, NA, NF, PCs, Instrs, Out) :-
    wz_header(EntryIdx, NameCodes, LabelIdx, NA, NF, [], Hdr),
    wz_body(PCs, Instrs, Hdr, Out).

% header: WAMO / version 2 / entry index / NE=1 / name / label / NA / NF,
% split across two clauses so neither holds too many call-spanning vars.
wz_header(EntryIdx, NameCodes, LabelIdx, NA, NF, A0, Out) :-
    wz_a('WAMO', A0, A1), wz_i(2, A1, A2), wz_i(EntryIdx, A2, A3), wz_i(1, A3, A4),
    wz_header2(NameCodes, LabelIdx, NA, NF, A4, Out).
wz_header2(NameCodes, LabelIdx, NA, NF, A0, Out) :-
    wz_n(NameCodes, A0, A1), wz_i(LabelIdx, A1, A2), wz_i(NA, A2, A3), wz_i(NF, A3, Out).

% body: PC section, instruction section, then NM=0.
wz_body(PCs, Instrs, A0, Out) :-
    wz_pcs_sec(PCs, A0, A1),
    wz_instr_sec(Instrs, A1, A2),
    wz_i(0, A2, Out).

wz_pcs_sec(PCs, A0, A2) :- length(PCs, NL), wz_i(NL, A0, A1), wz_pcs_rows(PCs, A1, A2).
wz_pcs_rows([], A, A).
wz_pcs_rows([P|Ps], A0, A2) :- wz_i(P, A0, A1), wz_pcs_rows(Ps, A1, A2).

wz_instr_sec(Instrs, A0, A2) :-
    length(Instrs, NC), wz_i(NC, A0, A1), wz_instr_rows(Instrs, A1, A2).
wz_instr_rows([], A, A).
wz_instr_rows([enc(T,O1,O2,R)|Is], A0, A2) :-
    wz_row(T, O1, O2, R, A0, A1), wz_instr_rows(Is, A1, A2).
wz_row(T, O1, O2, R, A0, A5) :-
    number_codes(T, Tc), append(A0, Tc, A1),
    wz_si(O1, A1, A2), wz_si(O2, A2, A3), wz_si(R, A3, A4), append(A4, [10], A5).

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
    write_wam_object([user:wamoserz/2], [wamo_entry(wamoserz/2)], SerzWamo),
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
