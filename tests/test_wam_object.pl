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

% read_term_from_atom/2 (eval bootstrap milestone 3b, reader -- first
% increment: atomic terms). Parse an integer from text at runtime and use it
% arithmetically -- proving the parse yields a real Integer, in a loaded
% object. Compounds/lists/floats/operators are follow-up increments.
readint_obj(R) :- read_term_from_atom('40', T), R is T + 2.  % -> 42

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
