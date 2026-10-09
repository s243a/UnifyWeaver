# Rust WAM: streams and lazy input, review and design

Status: review and gap analysis with a design. No runtime or template code
changes. Companion test: `tests/test_wam_rust_stream_swi_parity.pl`.

Baseline: `origin/main` at `5f08375`. Reference system: SWI-Prolog 9.0.4
(`swipl`). File abbreviations used below:

| key | file |
| --- | --- |
| `T` | `src/unifyweaver/targets/wam_rust_target.pl` (emits the runtime's `impl WamState` arms) |
| `WT` | `src/unifyweaver/targets/wam_target.pl` (shared compiler) |
| `ST` | `templates/targets/rust_wam/state.rs.mustache` |
| `SB` | `templates/targets/rust_wam/stream_builtin.rs.mustache` |
| `DB` | `templates/targets/rust_wam/dynamic_db_methods.rs.mustache` |
| `V` | `templates/targets/rust_wam/value.rs.mustache` |
| `PA` | `templates/targets/rust_wam/par_aggregate.rs.mustache` |
| `GO` | `templates/targets/go_wam/state.go.mustache` |
| `CPP` | `templates/targets/cpp_wam/runtime/builtin_dispatch.cpp.mustache` |
| `HC` | `docs/proposals/wam_rust_heap_cell_rewrite_design.md` (the heap-cell rewrite) |

## 1. Summary

The goal is SWI-consistent stream handling in the Rust WAM target: ordinary
stream I/O is **not** undone on backtracking, and SWI's opt-in replayable
input (lazy lists: `phrase_from_file/2,3`, `stream_to_lazy_list/2`,
`lazy_list/2,3`) lets a grammar backtrack over input that has already been
read without reading it twice.

Findings:

- **The part that exists is already SWI-consistent.** `read/1`,
  `read_term/1,2` (stdin), `write/1`, `format/1,2,3`, `put_char/1`, `nl/0`
  and the Rust-specific line reader (`stream_open/2`, `read_line/2`,
  `stream_close/1`) all keep their side effects across failure and
  backtracking, exactly as SWI does (§4, rows S1–S9 match in both emit
  modes). Stream state lives outside the heap, trail and choice points
  (`ST:2211-2215`, `SB:2-9`), which is the right invariant to keep.
- **Most of ISO/SWI stream I/O is missing.** The shared compiler registers
  `open/3`, `close/1`, `get_char/1,2`, `peek_char/1,2`, `get_code/1,2`,
  `put_char/2`, `read_line_to_string/2`, `read_string/5`,
  `at_end_of_stream/1`, `with_output_to/2` as builtins (`WT:2767-2773`,
  `WT:2860`, `WT:2871-2882`), the Go and C++ runtimes implement them
  (`GO:3056-3155`, `CPP:965-1041`, `CPP:2386-2547`), but the Rust runtime
  has no arm, so each call reaches `execute_builtin`'s default
  (`T:1554-1562`) and **fails silently**. `stream_property/2`,
  `set_stream_position/2`, `current_output/1`, `write/2`, `nl/1`,
  `at_end_of_stream/0`, `read/2`, `read_term/3` are not registered
  anywhere and compile to unresolved calls that also fail silently
  (`T:740-795`, `T:7410`).
- **Two silent failures are wrong answers**, not just missing features:
  `( at_end_of_stream -> A ; B )` runs `B` (row G6), and `read/1` on a
  syntax error fails instead of throwing (G5). Every bad-stream call fails
  where SWI throws an ISO error term (G4).
- **Lazy lists need attributed variables, and the runtime has none.** There
  is no `put_attr`, `freeze`, `when`, `dif` or wakeup mechanism in the
  runtime or the shared compiler (grep of `T`, `WT`, `ST`, all `rust_wam`
  templates and `HC`: no hits). SWI builds lazy lists from attributed
  variables plus non-backtrackable assignment (`nb_setarg`) and block reads
  (§6).
- **The design (§7)** adds a real stream layer first (handles, positions,
  peek, errors), then a minimal attributed-variable facility (an attribute
  table, a trailed `put_attr`, a hook point in `unify`, and a wakeup queue),
  then a runtime-native lazy tail whose materialized blocks live in the
  stream object, not the heap. That makes the materialized prefix stable
  across backtracking and re-materializable after a heap-top reset in both
  today's `Value` runtime and the heap-cell rewrite, without a frozen heap
  bar. §7.6 maps each piece onto the rewrite's cells, trail, obligations,
  detached terms and forks; §8 gives the phased plan and what should land
  before the rewrite.

## 2. Method

Each probe is a small program whose 0-arity driver prints with `write/1`
and `nl/0`. The same clauses are (a) asserted into `user:` and compiled with
`write_wam_rust_project/3` (`runtime_parser(compiled)`, both
`emit_mode(interpreter)` and `emit_mode(functions)`), then run by WAM label
from a probe binary with a fixed stdin, and (b) written to a file and run by
a **separate `swipl` process** with the same stdin, so SWI's stdin
semantics are real (consumption across backtracking, `end_of_file`,
`peek_char/1`), not an in-process emulation. The harness follows
`tests/test_wam_rust_baseline_fixes.pl`. The committed subset is
`tests/test_wam_rust_stream_swi_parity.pl`: 9 supported programs pass in
interpreter mode, 12 unsupported ones are `blocked(...)` with the gap id.

## 3. Inventory: what the Rust WAM target has today

### 3.1 Prolog-visible I/O builtins in the Rust runtime

| builtin | where | semantics today | SWI difference |
| --- | --- | --- | --- |
| `read/1`, `read_term/1`, `read_term/2` | `DB:9-44`, routed from the `Call`/`Execute` arms `T:762-764`, `T:859-864`, and `T:2813-2817` | On the first call **slurps all of stdin** into `term_input` (`DB:13`), then cuts dotted terms with `next_dotted_term_text` and parses each on a separate parser machine running the compiled `prolog_term_parser` (`T:3238-3298`). The cursor `term_input_pos` is a plain field, never trailed or saved in a choice point (`ST:2211-2215`), and survives `reset_query` (`ST:4297-4316` does not touch it) | Consumption semantics match SWI (S1–S4). Differences: whole-stdin read blocks interactive use; a syntax error fails instead of throwing `syntax_error` and resyncing (G5); `variable_names`, `variables`, `singletons` supported, no other options; no stream argument form |
| `read_term_from_atom/2,3`, `atom_to_term/3`, `term_to_atom/2` | `T:3175-3230` | parse via the compiled parser; `syntax_errors(error)` throws (`DB:234`) | ok for this review |
| `write/1`, `print/1`, `display/1` | `T:1841-1848` | `print!("{}", deref_heap(v))` through `Value`'s `Display` (`V:788-816`) | operators not honoured, `", "` separators (C1); unbuffered-flush not guaranteed (stdout `LineWriter`) |
| `writeln/1`, `write_canonical/1`, `tab/1`, `put_char/1`, `put_code/1`, `nl/0` | `T:1850-1925`, `T:2394` | stdout only | no stream forms |
| `format/1,2,3` | `T:1775-1839` | sinks: stdout, `user_output`, `user_error`, `atom(_)`, `string(_)`, `codes(_)`; flushes | any other stream argument fails |
| `stream_open/2`, `read_line/2`, `stream_close/1` | `SB:62-113`; registered `WT:2834-2836` | UnifyWeaver-specific buffered line reader. Handles are **integers** indexing a **process-global** `OnceLock<Mutex<Vec<...>>>` table (`SB:2-9`, max 4096). Reads are plain side effects (S8 matches SWI's `open/3`+`read_line_to_string/2`) | not SWI API; missing file is a plain failure, not `existence_error(source_sink, _)`; an integer handle unifies with the integer |
| `read_file_to_atom/2`, `write_atom_to_file/2`, `append_atom_to_file/2`, `copy_file/2`, … | `T:2122-2160` | whole-file helpers | not streams |

### 3.2 Registered by the shared compiler but absent in Rust

These compile to `BuiltinCall` (`WT:2717`-`2900`, `T:989-990`) and fail in
`execute_builtin`'s default arm (`T:1554-1562`), which tries the six family
dispatchers and returns `false` with no diagnostic:

`open/3`, `close/1`, `read_line_to_string/2`, `read_string/5`,
`at_end_of_stream/1`, `write_to_stream/2`, `nl_to_stream/1`
(`WT:2767-2773`); `get_char/1,2`, `get_code/1,2`, `peek_char/1,2`,
`put_char/2`, `put_code/2` (`WT:2873-2882`); `with_output_to/2`
(`WT:2860`).

The Go runtime implements all of these (`GO:3056-3155`) with a dedicated
`StreamHandle` value (`GO:53-66`: id, `*os.File`, `bufio.Reader`, mode,
closed flag) and a default-input reader with peek. The C++ runtime does too
(`CPP:965-1041` for stdin char I/O with `ungetc` peek, `CPP:2386-2547` for
the stream family). So this is a Rust-lane inconsistency inside the WAM
family, not a shared-compiler gap.

### 3.3 Not registered anywhere (compile to unresolved calls)

`at_end_of_stream/0`, `stream_property/2`, `set_stream_position/2`,
`stream_position_data/3`, `current_input/1`, `current_output/1`,
`set_input/1`, `set_output/1`, `flush_output/0,1`, `read/2`,
`read_term/3`, `write/2`, `writeq/1,2`, `print/2`, `nl/1`, `see/1`,
`tell/1`, `phrase/2,3`, `phrase_from_file/2,3`, `stream_to_lazy_list/2`,
`lazy_list/2,3`, `put_attr/3`, `get_attr/3`, `freeze/2`, `when/2`,
`dif/2`. The `Call` arm (`T:740-795`) finds no label, no builtin, and
calls `warn_unresolved_goal` (`T:7410-7414`), which prints only when
`UW_WAM_WARN_UNKNOWN` is set. The goal fails.

### 3.4 Lazy or stream-backed data in the runtime (not Prolog streams)

| component | what it is | relevance |
| --- | --- | --- |
| `seek_fact_source.rs.mustache` | seek reader over UWFI/UWIX fact stores (`indexed(Prefix)`) and an LMDB tier; rows delivered as ground `Value`s to `fact_table_attempt` / `finish_foreign_results` with a shared decoded-row cache (`HC` §9.3) | a model for "data outside the heap, cells created at delivery": the same rule the lazy-list block cache follows (§7.4) |
| `csr_fact_source.rs.mustache`, `lmdb_fact_source_*.rs.mustache`, `lazy_category_parents.rs.mustache` | `LookupSource` implementations for the kernels (int ids, atom maps) | no Prolog-visible stream; no backtracking interaction beyond result delivery |
| `boundary_cache.rs.mustache` | histogram cache on `u32` node ids | none |
| `term_input` / `term_input_pos` (`ST:2211-2215`) | the only input buffer with Prolog-visible position; deliberately outside snapshots | becomes stream 0's buffer in the design (§7.2) |

Nondeterministic builtins keep redo data in `BuiltinState { name, args,
data }` on the choice point (`ST:794`, `T:5050`); none of them is a stream
reader, so no stream state is in any choice point or trail entry today
(`ST:573-620` trail kinds: `Binding`, `Register` only).

### 3.5 Attributed variables and coroutining

None. `grep -i 'attvar|put_attr|get_attr|freeze|frozen|when/|attributed'`
over `T`, `WT`, `ST`, every `rust_wam` template and `HC` finds only the
Stage-2 "frozen resolver shape" comments and the cache-attribution tests.
SWI's lazy lists are built entirely on this facility (§6), so it is a
prerequisite, not an option.

## 4. SWI vs Rust comparison

Outputs are the trimmed stdout of each driver. "fails" means the Rust
driver's `run()` returned `false` and printed nothing (the builtin failed
or the goal was unresolved). Both Rust emit modes gave the same verdict on
every row. Files: `in_lines.txt` = `one\ntwo\n`, `in_aaab.txt` = `aaab\n`.

### 4.1 Supported behaviour (committed as tests)

| id | program | stdin / file | SWI 9.0.4 | Rust WAM | verdict |
| --- | --- | --- | --- | --- | --- |
| S1 | `( read(_), fail ; read(Y) ), write(Y)` | `a. b. c.` | `b` | `b` | match: input stays consumed |
| S2 | `sp_rr(X) :- read(p(X)), X > 5.` `sp_rr(X) :- read(p(X)).` then `write(X)` | `p(1). p(2). p(3).` | `2` | `2` | match: read behind a clause CP |
| S3 | `( sp_rm(X), read(T), write(X-T), nl, fail ; true )` over `a,b,c` | `t(1). t(2).` | `a-t(1)` `b-t(2)` `c-end_of_file` | `-(a, t(1))` `-(b, t(2))` `-(c, end_of_file)` | semantics match; printing C1 |
| S4 | `read(X), read(Y), read(Z)` | `a.` | `[a,end_of_file,end_of_file]` | `[a, end_of_file, end_of_file]` | semantics match; printing C1 |
| S5 | `read_term(T,[variable_names(Vs)])`, bind each `N=V` with `V=N`, print | `f(X,Y,X).` | `f(X,Y,X)-[X=X,Y=Y]` | `-(f(X, Y, X), [=(X, X), =(Y, Y)])` | semantics match (sharing of `X` kept); printing C1 |
| S6 | `( write(hello), fail ; true ), write(world)` | | `helloworld` | `helloworld` | match: output not undone |
| S7 | `( format("~w-~w",[a,b]), put_char(c), fail ; true ), nl` | | `a-bc` | `a-bc` | match |
| S8 | `stream_open(F,H), ( read_line(H,_), fail ; read_line(H,L2) ), stream_close(H), write(L2)` (SWI via `open/3` + `read_line_to_string/2` shims) | `in_lines.txt` | `two` | `two` | match: line stays consumed |
| S9 | SWI's translation of `g(N) --> as(N), "b", "\n".` `as(0) --> [].` `as(N) --> "a", as(M), {N is M+1}.` run on `[97,97,97,98,10]` | | `3` | `3` | match: the grammar backtracks over an in-memory list |

### 4.2 Unsupported or divergent

| id | program | stdin / file | SWI 9.0.4 | Rust WAM | class |
| --- | --- | --- | --- | --- | --- |
| G1 | `peek_char(A), get_char(B), get_char(C), get_char(D), peek_char(E)` | `xy` | `[x,x,y,end_of_file,end_of_file]` | fails | missing builtin |
| G2 | `open(F,read,S), read_line_to_string(S,L1), (at_end_of_stream(S)->E=yes;E=no), read_line_to_string(S,L2), (at_end_of_stream(S)->E2=yes;E2=no), close(S)` | `in_lines.txt` | `one-no-two-yes` | fails | missing builtin |
| G3 | `open(F,read,S), stream_property(S,position(P0)), get_char(S,C1), get_char(S,C2), set_stream_position(S,P0), get_char(S,C3), stream_property(S,position(P1)), stream_position_data(char_count,P1,N)` | `in_aaab.txt` | `[a,a,a,1]` | fails | missing builtin |
| G4 | `catch(G, error(E,_), ...)` for seven bad-stream goals | | `existence_error(stream,nostream)` (`get_char/2`), `existence_error(stream,foo)` (`close/1`), `existence_error(source_sink,'/nonexistent/x')` (`open/3`), `domain_error(stream_or_alias,42)` (`peek_char/2`), `instantiation_error` (`get_char(_,_)`), `existence_error(stream,foo)` (`at_end_of_stream/1`), `domain_error(stream_position,bad)` (`set_stream_position/2`) | every goal **fails**, nothing thrown | error-term mismatch |
| G5 | `catch(read(T), error(syntax_error(_),_), T=syntax), write(T), nl, read(T2), write(T2)` | `foo(. ok.` | `syntax` then `ok` (SWI throws and resyncs after the `.`) | fails | wrong answer |
| G6 | `( at_end_of_stream -> write(eof) ; write(more) )` | empty | `eof` | `more` | **wrong answer** (unresolved goal fails silently; else branch runs) |
| G7 | `current_output(S), write(S,hi), nl(S)` | | `hi` | fails | missing builtin |
| G8 | `with_output_to(string(S), (write(a), fail ; write(b))), write(S)` | | `ab` (output inside the capture is not undone either) | fails | missing builtin (registered `WT:2860`, no arm) |
| L1 | `phrase_from_file(sp_g(N), F), write(N)` | `in_aaab.txt` | `3` | fails | missing (lazy lists) |
| L2 | `lazy_list(sp_nxt, 1, L), L=[A,B,C|_], (L=[1,2,4|_]->wrong ; L=[1,2,3,4|_]->replay_ok), count(L)` with `sp_nxt(S0,S1,S0) :- S0<6, S1 is S0+1` | | `[1,2,3]` `replay_ok` `5` | fails | missing (lazy lists) |
| L3 | `open(F,read,S), stream_to_lazy_list(S,L), (L=[0'x|_]->R=x ; L=[0'a,0'a|_]->R=aa ; R=none), (sp_g(N,L,[])->true ; N=nog), close(S)` | `in_aaab.txt` | `aa-3` | fails | missing (lazy lists) |
| C1 | `write(a-b)`, `write([1,2])`, `write(1+2*3)` | | `a-b` `[1,2]` `1+2*3` | `-(a, b)` `[1, 2]` `+(1, *(2, 3))` | cosmetic |

Side finding, not stream-related: `atom_length('ab\n', L)` (a **quoted atom
constant containing a newline** in program text) fails in Rust, while
`atom_codes(X,[97,10]), atom_length(X,L)` gives `2` as in SWI. The program
text is line-based (`parse_instructions`, `ST:6894-6905`) and
`escape_rust_string` (`T:8781-8786`) escapes only `\` and `"`, so the
constant breaks the instruction line. Wrong answer; out of scope here, but
it bit the first version of S9 (`atom_codes('aaab\n', Cs)`).

## 5. Gap list, ranked

Severity classes: wrong answer > missing builtin > error-term mismatch >
cosmetic. Within a class, ranked by how much SWI-consistent code it blocks.

| rank | id | gap | class | where |
| ---: | --- | --- | --- | --- |
| 1 | G6 | `at_end_of_stream/0` (and every other unregistered I/O goal, §3.3) fails **silently**; `->`/`\+` turn that into the wrong branch | wrong answer | `T:740-795` (`Call` arm falls through), `T:7410-7414` (diagnostic gated on an env var), `WT:2771` (only `/1` registered) |
| 2 | G5 | `read/1` syntax error: silent failure instead of `error(syntax_error(_),_)`; no resync to the next `.`; subsequent reads are also lost | wrong answer | `DB:9-44` (no `syntax_errors` default), `DB:234-247` (raise only on the atom path) |
| 3 | G1 | `get_char/1,2`, `peek_char/1,2`, `get_code/1,2` missing; also stdin has no char-level buffer shared with `read/1` (`DB:13` slurps everything), so char and term reads cannot interleave | missing builtin | `WT:2873-2878` registers them; `T:1554-1562` default arm; `SB` has no stdin reader |
| 4 | G2 | `open/3`, `close/1`, `read_line_to_string/2`, `read_string/5`, `at_end_of_stream/1`, `put_char/2`, `put_code/2`, `write_to_stream/2`, `nl_to_stream/1` missing; Go and C++ have them | missing builtin | `WT:2767-2773`, `WT:2880-2882`; `GO:3056-3155`, `CPP:2386-2547` |
| 5 | L1–L3 | no lazy lists: `stream_to_lazy_list/2`, `phrase_from_file/2,3`, `phrase_from_stream/2`, `lazy_list/2,3`, `lazy_list_materialize/1`, `lazy_list_length/2`; no `phrase/2,3` either | missing builtin | nothing in `T`, `WT`, `ST` (§3.5) |
| 6 | G3 | `stream_property/2` (`position`, `file_name`, `end_of_stream`, `alias`, `mode`), `set_stream_position/2`, `stream_position_data/3`, `line_count/2`, `character_count/2`: needed by `lazy_list_location//1` and by any repositioning reader | missing builtin | not registered |
| 7 | G4 | ISO error terms for bad streams: `existence_error(stream, S)`, `existence_error(source_sink, F)`, `domain_error(stream_or_alias, X)`, `permission_error(input|output, stream, S)`, `instantiation_error`; `stream_open/2` on a missing file fails (`SB:68-71`) | error-term mismatch | `SB:62-113`; the runtime already has `raise_iso_error` (`DB:222`) and `raise_builtin_error` (`T:5731`) to build them |
| 8 | G7 | `current_output/1`, `current_input/1`, `set_input/1`, `set_output/1`, `write/2`, `nl/1`, `print/2`, `flush_output/0,1`, `format/3` to a stream handle (only `user_output`/`user_error`/sinks at `T:1801-1830`) | missing builtin | not registered / `T:1801-1830` |
| 9 | G8 | `with_output_to/2` registered, no arm; needs a redirectable current-output stack | missing builtin | `WT:2860`, no Rust arm |
| 10 | G9 | stream handles are bare integers (`SB:18`, `SB:62-80`): `write(H)` prints `1`, `H == 1` succeeds, `integer(H)` succeeds; Go uses an opaque `StreamHandle` (`GO:53-66`) | cosmetic (type-test semantics) | `SB` |
| 11 | C1 | `write/1`/`print/1` ignore operators and print `", "` between arguments and list elements | cosmetic, but it defeats every output-based parity test | `V:788-816`, `T:1841-1848` |
| 12 | — | `read/1` reads all of stdin at the first call, so an interactive program blocks until EOF; SWI reads incrementally | behavioural | `DB:11-16` |

Two notes on things that are **not** gaps: the stream table being
process-global (`SB:2-9`) matches SWI, where streams are process objects
shared by threads; and `term_input_pos` surviving `reset_query` matches a
stream's position surviving a new query.

## 6. How SWI implements replayable input

Ordinary streams are plain side effects: the read pointer moves, nothing is
trailed. Rows S1–S8 and G2 show it. Replayable input is layered on top with
three mechanisms (`library(pure_input)`, `library(lazy_lists)`, SWI
9.0.4):

1. **An attributed variable as the list tail.** `stream_to_lazy_list/2`
   does `stream_property(Stream, position(Pos))` and
   `put_attr(List, pure_input, lazy_input(Stream, PrevPos, Pos, Read))`.
   `lazy_list/2` does `put_attr(List, lazy_lists, lazy_list(Next, Read))`.
   The list is an unbound variable until something unifies with it.
2. **`attr_unify_hook/2` materializes one block.** Unifying the tail with
   a term binds the variable first (that binding is trailed like any
   other) and then runs the hook with the attribute and the value. For
   `pure_input`:

   ```prolog
   attr_unify_hook_ndebug(State, Value) :-
       State = lazy_input(Stream, _PrevPos, Pos, Read),
       (   var(Read)
       ->  fill_buffer(Stream),
           read_pending_codes(Stream, NewList, Tail),   % one buffer, 4096 bytes by default
           (   Tail == []
           ->  nb_setarg(4, State, []), Value = []
           ;   stream_to_lazy_list(Stream, Pos, Tail),  % the new tail is the next lazy cell
               nb_linkarg(4, State, NewList),
               Value = NewList
           )
       ;   Value = Read                                  % already read: replay, no I/O
       ).
   ```

   `lazy_lists` is the same shape with `call(Next, NewList, Tail)` in place
   of the block read and a `dummy` attribute on `Tail` whose `Next` is
   linked in with `nb_linkarg`. `lazy_list/3` wraps a state in `s(State0)`
   and advances it with `nb_setarg` after each successful `call(Pred,
   State0, State1, Head)`.
3. **Non-backtrackable assignment keeps the block.** `nb_setarg/3` and
   `nb_linkarg/3` write the fourth argument of the attribute term in
   place and are **not undone on backtracking**; SWI also freezes the
   global stack below the assigned term so backtracking does not reclaim
   it. When the program backtracks past the unification that forced the
   block, the variable's binding is undone (it is an ordinary trailed
   bind) but `Read` stays set. The next unification takes the `Value =
   Read` branch: the same cells, no stream read. That is the whole
   "read once, replay many" guarantee. The stream pointer only ever
   moves forward; `PrevPos`/`Pos` are kept for `lazy_list_location//1`
   (error positions), which repositions the stream temporarily with
   `set_stream_position/2`.

Consequences the design must reproduce:

- forcing is permanent for the lifetime of the list; a forced block is
  never re-read, even after backtracking (L2's `replay_ok`, L3's `aa`
  after `[0'x|_]` failed);
- the stream is consumed exactly once overall, regardless of how many
  times the grammar backtracks over it (SWI's hook runs at most one
  `read_pending_codes` per cell);
- blocks are the buffer size (4096 here), so a grammar's lookahead across
  a block boundary forces the next cell transparently;
- memory: everything forced stays reachable from the list head for as long
  as the head is live. `phrase_from_file/2` over a 1 GB file holds 1 GB of
  code cells (SWI documents this); the runtime has no GC anyway (`HC`
  §3.3), so this is no worse than SWI's non-GC behaviour;
- a lazy list is an unbound variable to `var/1`, `nonvar/1`, type tests
  and `write/1` until forced; `copy_term/2`, `findall/3` and `assert/1`
  copy attributes (SWI copies attributed variables with their attributes
  by default);
- closing the stream while a lazy tail is still unforced makes the next
  force throw `existence_error(stream, S)`; `stream_to_lazy_list/2` on an
  unbuffered stream throws `permission_error(create, lazy_list, S)`.

## 7. Design

The design is in layers so that each can land and be tested alone. Every
layer has a "today" form (the `Value`-based runtime: `Value::Unbound(Sym)`,
`bindings: HashMap<Sym, Value>`, `heap: Vec<Value>` of `put_structure`
placeholders, `TrailEntry { key: Binding|Register, old_value }`,
`ChoicePoint { trail_len, heap_len, saved_args, stack, ... }`,
`T:1443-1500` backtrack) and a "rewrite" form (`HC`: 8-byte tagged cells,
conditional trailing with HB, rollback obligations, `Detached`, forks).

### 7.1 Invariant: ordinary stream state is never part of the machine state

Keep what the runtime already does for `term_input` (`ST:2211-2215`) and
the line-reader table (`SB:2-9`): stream objects live in a table outside
`heap`, `trail`, `bindings`, `choice_points` and `ScopeRec`s; a choice point
stores no stream position; `backtrack`, `rewind`, `reset_query` and the
dynamic-DB snapshot/adopt path (`HC` §9.4) never touch it. A redo record
may hold a stream **id** under `HC` §5.6's "other ids" row (valid while the
table generation is unchanged). This gives S1–S8 by construction and is
what SWI does. The only replay mechanism is the lazy list of §7.4, and it
replays from its own block cache, never by moving the stream pointer.

### 7.2 Stream layer (prerequisite for everything else)

**Objects.** One `Stream` per open stream, in a process-global table (as
`SB:2-9` today, so forks and the parser machine see the same streams, as
SWI threads do):

```rust
struct Stream {
    id: u32, gen: u32,                       // gen bumps on close; a stale handle is existence_error
    kind: Input(InputStream) | Output(OutputStream),
    alias: Option<AtomId>,                   // user_input, user_output, user_error
    file_name: Option<String>, mode: Mode,
    eof_action: EofAction, encoding: Utf8,   // ISO properties, minimal set
}
struct InputStream {
    src: Box<dyn Read>,                      // File, stdin, or a memory buffer
    seekable: bool,
    buf: Vec<u8>, buf_pos: usize,            // one 4096-byte block, like SWI
    pos: Position,                           // byte_count, char_count, line_count, line_position
    peeked: Option<char>,                    // peek_char without consuming
    lazy: Option<LazyBlocks>,                // §7.4: append-only block cache while a lazy list is live
}
```

`user_input` is stream 0 over stdin and **replaces `term_input`**:
`read/1` reads from the stream's buffer incrementally (term text up to the
next end `.` plus layout, then parse as today), so `get_char/1`, `peek_char/1`,
`read/1` and `at_end_of_stream/0` share one position. `set_term_input`
(`DB:4`) stays as a test seam by installing a memory-backed `user_input`.
Output streams wrap `stdout`/`stderr`/`File` and a redirect stack for
`with_output_to/2` and `format(atom(_), …)`.

**Handles.** Today: `Value::Str("$stream/1", [Integer(id)])` (prints as
`$stream(3)`, is a compound to type tests, and `write/1`, `==`, `copy_term`
need no new arm). Rewrite: a `BOX` with `Boxed::Stream(id, gen)`; `BOX` is
atomic to unify, which is what SWI's blob gives. `write/1` prints
`<stream>(0x…)`-style text; nothing in the project compares that text.
Aliases are atoms resolved at call time. The Go `StreamHandle` value
(`GO:53-66`) is the same idea. The integer handles of `stream_open/2`
stay accepted by the three legacy builtins for compatibility and are
documented as deprecated.

**Builtins (all in a new `stream_builtin.rs.mustache` section, routed from
`execute_io_builtin`).** `open/3,4` (read, write, append; `alias(_)`,
`eof_action(_)`), `close/1,2`, `current_input/1`, `current_output/1`,
`set_input/1`, `set_output/1`, `get_char/1,2`, `peek_char/1,2`,
`get_code/1,2`, `peek_code/1,2`, `put_char/2`, `put_code/2`, `nl/1`,
`write/2`, `print/2`, `write_canonical/2`, `writeq/1,2`, `format/3` to any
output stream, `flush_output/0,1`, `at_end_of_stream/0,1`,
`read_line_to_string/2`, `read_line_to_codes/2`, `read_string/3,5`,
`read_term/3`, `read/2`, `stream_property/2` (`position`, `file_name`,
`alias`, `mode`, `end_of_stream`, `buffer`, `buffer_size`, `input`,
`output`), `set_stream_position/2`, `stream_position_data/3`,
`line_count/2`, `character_count/2`, `with_output_to/2`. Register each in
`WT:is_builtin_pred/2` so the shared compiler emits `BuiltinCall` (which
also fixes G6's silent unresolved call for `at_end_of_stream/0`), and make
sure Go and C++ keep parity or document the difference (they already have
most of G1/G2).

**Position terms.** `'$stream_position'(CharCount, LineNo, LinePos,
ByteCount)` as in SWI, so `stream_position_data/3` and
`set_stream_position/2` interoperate textually. Repositioning: `seek` for
files (byte offset, then re-fill), `domain_error(stream_position, _)` for a
malformed term, `permission_error(reposition, stream, S)` for pipes and
stdin, as SWI.

**Errors.** Build the ISO terms with the existing `raise_iso_error`
(`DB:222`): unbound stream → `instantiation_error`; not a handle or alias →
`domain_error(stream_or_alias, X)`; closed or unknown → `existence_error(stream,
X)`; wrong direction → `permission_error(input|output, stream, S)`;
missing file → `existence_error(source_sink, F)`; `open/3` with a bad mode →
`domain_error(io_mode, M)`. `read/1,2` and `read_term/2,3` throw
`error(syntax_error(Kind), stream(S, Line, LinePos, CharNo))` by default
and **resync** by consuming through the end `.` (G5 shows SWI reading `ok`
after the bad term); `syntax_errors(fail|quiet)` keep today's silent form.

**Silent-failure diagnostics (G6).** Independent of streams: make
`warn_unresolved_goal` (`T:7410`) print by default (to stderr, once per
name) and keep `UW_WAM_WARN_UNKNOWN=0` to silence it; or throw
`existence_error(procedure, PI)` as SWI does with the `unknown` flag set to
`error`. The owner decides; the review recommends the throw behind a
build-time flag, with the warning as the default until the full builtin
sweep lands.

### 7.3 Attributed variables, minimal form

Only what lazy lists need, designed so `freeze/2`, `when/2`, `dif/2` can
reuse it.

**Today's runtime.** Add `attvars: HashMap<Sym, Attr>` next to `bindings`
(`ST:2344`). A variable is attributed if it is unbound and has an entry.
`put_attr` inserts and pushes a new trail kind `TrailKey::Attr(Sym)` with
`old_value: Option<Attr>` so `unwind_trail_to` (`ST:4275`) and
`unwind_trail_bindings_only` restore or remove it on backtracking (SWI
trails `put_attr`; the review's probe `( stream_to_lazy_list(S, L), fail ;
true ), L = [a]` must leave `L` plain). This is the one new trail kind and
the one place that touches the trail; it must be coordinated with the
concurrent register-trail audit (the audit's question "is stream state ever
in a register trail entry" stays answered "no": the entry carries an
attribute, not a stream position). `var/1`, `nonvar/1`, type tests,
`deref_heap`, `write/1`, standard order need no change: an attvar is still
`Value::Unbound(name)`.

**Hook point.** `unify` (`ST:6443`) binds in exactly two arms,
`(Unbound(n1), other)` and `(other, Unbound(n2))`. Before `bind_var`, check
`attvars.get(n)`: for a **native** attribute (`LazyTail`, §7.4) run the
native hook inline; for a **Prolog** attribute (`lazy_list/2,3`'s `Next`,
later `freeze/2`) bind as usual and push `(attr, other)` onto a
`wakeup: Vec<(Attr, Value)>` queue. Attvar–attvar unification: bind the
right one to the left and queue the right one's hook with the left variable
as the value (SWI's behaviour for a single attribute module).

**Wakeup.** The step loop runs the queue before dispatching the next
`Call`, `Execute`, `Proceed` or `BuiltinCall` (SWI runs `$wakeup/1` at the
next call port). Each entry runs its hook goal through `call_goal_once`
(`T:7311`) with the backtrack floor set (`HC` R−1a); a failing hook fails
the instruction that bound the variable, so control backtracks exactly as
if the unification had failed. The queue is cleared on `backtrack` and
`reset_query` (an entry queued by a unification that is being undone must
not run). The lowered tier (`emit_mode(functions)`) has no step loop
between head unification and the first body goal; a lowered predicate that
can bind an attributed variable must either drain the queue after its head
unification or decline lowering for predicates whose arguments can be
attributed. Since lowering is shape-gated and the resolver never sees
attvars, the first version drains the queue at `lowered_call`'s return and
at every `call_goal_once` boundary, and adds a paranoid-build assertion that
the queue is empty at `Proceed`.

**Rewrite.** `HC` §3.1 has no free tag. Use the `VAR` cell's 4-bit `kind`:
reserve `kind = 14` (`ATTV`) whose `n` indexes `attvars: Vec<AttrRec {
name: (kind, n), attr: Attr }>`, a per-machine arena saved in every
obligation and truncated on rollback like `boxes` (`HC` §5.2). The record
carries the variable's real printed name, so text and standard order are
byte-identical. `deref` stops at any `VAR`, so every "is it a variable"
site keeps working; only `bind(addr, c)` (`HC` §7.1) gains
`if kind(heap[addr]) == ATTV { hook }`. `put_attr` on a plain `VAR` is a
heap write of the cell at `addr` and is conditionally trailed like a bind:
with the "trail stores the old cell" option of `HC` §5.1/§16, undo
restores the plain `VAR` with its name for free; the address-only
alternative would need the name re-derived from a side table, which is a
concrete argument for the old-cell form. An `ATTV` record created after an
obligation disappears with the arena truncation; one created before
survives, and its `attr` payload may carry non-backtrackable state (§7.4),
exactly like SWI's `nb_setarg` on an older term. Export (`HC` §9.2): `DVar`
gains `attr: Option<DAttr>`; import re-creates the attvar (`copy_term/2`,
`findall/3`, `assert/1`, thrown balls, par results all copy attributes, as
SWI does by default).

### 7.4 Runtime-native lazy tail (`stream_to_lazy_list/2`, `phrase_from_file/2,3`)

**Data placement rule.** Materialized input is **owned by the stream
object, as bytes**, never by the heap:

```rust
struct LazyBlocks { base: Position, blocks: Vec<Arc<[u8]>>, live: usize }  // in InputStream.lazy
enum Attr { LazyTail { stream: u32, gen: u32, block: u32 }, Prolog(Value /* Detached in the rewrite */) }
```

`stream_to_lazy_list(S, L)`: if `S.lazy` is `None`, set it to
`LazyBlocks { base: S.pos, blocks: [] }`; put `LazyTail { stream, gen,
block: blocks.len() }` on `L`. Blocks are appended by reading one buffer
(4096 bytes, `read_pending_codes` semantics: whatever is buffered, at least
one byte, split at a UTF-8 boundary) when a tail whose `block` equals
`blocks.len()` is forced. A tail whose `block < blocks.len()` is a
**replay**: its block is already there. So forcing block `k` reads the
stream iff `k == blocks.len()`, and the stream is read exactly once per
block no matter how often the program backtracks over it.

**Forcing (the native hook, inline in `unify`).** For `LazyTail { stream,
gen, block }` on variable `v` unified with `other`:

1. look the stream up by `(stream, gen)`; a closed stream throws
   `existence_error(stream, S)` from the unification (SWI does);
2. if `block == blocks.len()`, read one block and append it; if the read
   returns 0 bytes, the cell is `[]`: bind `v := []` (trailed) and continue
   unifying `[]` with `other`;
3. otherwise build the cell list for block `k`: the block's codes as a
   list whose tail is a **fresh** attributed variable `v'` with
   `LazyTail { block: k + 1 }` (same stream, same gen); bind `v := list`
   (trailed) and continue the `unify` loop with `(list, other)`.

Today: the list is a `Value::Str("[|]/2", …)` chain (the runtime's partial
list form, `ST:6643` `rebuild_partial`) of `Value::Integer` codes ending in
`Value::Unbound(v')`, with `attvars[v'] = LazyTail{..}`. All of it is `Arc`
memory, so it survives `heap.truncate` (`T:1471`); only the `bindings`
entry for `v` is trailed and undone. Backtracking past the force and
forcing again rebuilds a new chain for the same block from the cached
bytes; the old chain is garbage once unreachable. The chain is O(block)
allocations per force; a `Value::List` with an explicit tail (a
`Value::Partial(Args, Box<Value>)`) is a later optimization, as is caching
the built chain in the attvar record (`Option<Value>`, not trailed) so that
a replay after backtracking is O(1) as in SWI; that cache is safe today
because `Value`s do not live in the truncated heap.

Rewrite: step 3 allocates the block's `LIS` cells and the `ATTV` cell on
the heap above the current H, so a rollback below them truncates them. That
is intended: the **bytes** survive in `LazyBlocks`, the cells are
re-materializable, and nothing retains a cell address across the rollback
(`HC` §5.6 retention rules; the `LazyTail` record carries ids only). After
the rollback the tail variable `v` is unbound again (its bind was trailed
because `addr(v) < HB` whenever an older obligation protects it) and
carries the same `LazyTail { block: k }`, so the next force rebuilds block
`k` from the cache. A tail allocated *after* an obligation vanishes with
the heap and its `ATTV` record with the arena; the block it pointed to
stays in `LazyBlocks`, so if the list head itself survived, a later force
finds the block by index. The alternative, a frozen heap bar (SWI's
mechanism: rollback truncates to `max(o.h, frozen_bar)` so forced cells
survive), gives O(1) replay but breaks `HC`'s invariant I1 (`h` monotone,
HB derivable from the top obligations) and makes bindings into the frozen
region unconditional; the review does not recommend it for R2. It can be
revisited as a measured optimization once a paranoid build can check it.

**Position bookkeeping.** Each block records the stream `Position` at its
start (char, line, line position, byte). `lazy_list_location//1` and
`lazy_list_character_count//1` then compute positions from `(block,
offset)` without repositioning the stream, which is better than SWI's
temporary `set_stream_position/2` and works for pipes and stdin too.
`'$skip_list'/3` is a native helper (walk cells to the first non-cons).

**Memory.** `LazyBlocks` grows to the whole input while the stream is open
and holds the cache; it is dropped at `close/1` (forcing after close throws,
as SWI) or when the stream object is dropped. Cells are the same order of
memory as SWI's. No GC today and none in the rewrite (`HC` §3.3), so
`phrase_from_file/2` on a very large file is bounded by memory in both.

**`phrase_from_file/2,3`, `phrase_from_stream/2`, `phrase/2,3`.** Compile
them as library predicates in Prolog on top of the native pieces, the way
`prolog_term_parser` is pulled into a project (`T:9346-9353`): `phrase/2,3`
is `call(G, L, R)` with the DCG body translation the shared compiler already
performs; `phrase_from_file/3` is `setup_call_cleanup(open/4,
phrase_from_stream/2, close/1)`. `setup_call_cleanup/3` is not in the
runtime either; a first version uses `catch/3` plus explicit close on both
exits (first-solution `catch/3` is fine here because `phrase_from_file/2`
is `once`-like in SWI too: the cleanup closes the stream after the first
solution's exit).

### 7.5 `lazy_list/2,3` with a Prolog fetch predicate

`lazy_list(Next, L)`: `put_attr(L, lazy_lists, lazy_list(Next, Read))` with
`Read` unbound. The hook is Prolog, run from the wakeup queue (§7.3):

```prolog
'$lazy_list_hook'(State, Value) :-
    State = lazy_list(Next, Read),
    (   var(Read)
    ->  call(Next, NewList, Tail),                  % Tail == [] or unbound
        (   Tail == []
        ->  '$nb_set_attr_arg'(2, State, NewList)
        ;   put_attr(Tail, lazy_lists, lazy_list(Next, _)),
            '$nb_set_attr_arg'(2, State, NewList)
        ),
        Value = NewList
    ;   Value = Read
    ).
```

`'$nb_set_attr_arg'/3` is the one new primitive: a non-backtrackable write
into the attribute record (today: replace the `Attr::Prolog(Value)` payload
in `attvars` without a trail entry; rewrite: write the `AttrRec.attr`
`Detached` in the arena, no trail). SWI's `nb_setarg/3` is general; the
review scopes it to attribute payloads so no heap term ever becomes
non-backtrackable, which keeps `HC`'s rollback model intact. In the rewrite
`NewList` is exported to a `Detached` at the time of the write (it is ground
except for `Tail`, which is exported as an attributed `DVar`), and the
`Value = Read` branch imports it with `Anchored` so a surviving `Tail` cell
is reused and a truncated one is re-created with its attribute. The
`lazy_list/3` state cell is the same primitive with a `s(State0)` payload.
`lazy_read_lines/4`, `lazy_read_terms/4`, `lazy_get_codes/4` are plain
Prolog over the stream layer and ship with the library.

### 7.6 Interaction with the heap-cell rewrite, point by point

| rewrite concern (`HC` §) | lazy-list answer |
| --- | --- |
| cell representation (§3.1) | attvar = `VAR` with `kind = ATTV`, `n` = index into a per-machine `attvars` arena; the record holds the printed name and the `Attr`; stream handle = `BOX` `Boxed::Stream(id, gen)`; `LazyTail` holds ids only, never cells |
| heap layout, no GC (§3.3) | forced cells are ordinary heap cells; bytes live in the stream object; nothing is pinned, so `reset_query` is still O(1) and `H = base` |
| variable names and order (§3.4) | `ATTV` records keep `(kind, n)`; printing and `compare_std` class 0 are unchanged |
| conditional trailing, HB (§5.1–5.2) | binding an attvar and `put_attr` are heap writes at `addr`, trailed iff `addr < HB`; `'$nb_set_attr_arg'` writes the arena record and is deliberately not trailed; the `attvars` arena top joins `h`, `tr`, `boxes`, `e_top` in every obligation and is truncated on rollback |
| heap-top reset on backtrack (§5.5) | forced cells above `o.h` are truncated and re-materialized from the block cache on the next force; a prefix shared with an older obligation is below `o.h` and survives; no frozen bar |
| retained cells and builtin redo (§5.6) | a redo record never holds a lazy cell; `LazyTail` is an id-only record under the "other ids" row with the stream `gen` as its table generation |
| builtin API (§7.1, §7.4) | stream builtins start bridged, as §7.4 already says for "stream, read_term, format"; the force path is native because it runs inside `unify` |
| detached terms (§9.2) | `DVar.attr: Option<DAttr>`; `DAttr::LazyTail{stream, gen, block}` is re-importable on any machine of the process (the stream table is global); `DAttr::Prolog(Detached)` for `lazy_list/2,3`; anchors behave as today |
| fact sources (§9.3) | same pattern: data outside the heap, cells at delivery |
| snapshot and adoption (§9.4) | `snapshot` copies the `attvars` arena and the wakeup queue with the machine (same logical machine); `adopt` keeps them |
| fork, `par_aggregate` (§9.4, `PA:29-226`) | a fork copies the heap and the arena, so both parent and worker would hold `LazyTail{block: k}` records on the same stream; forcing in both is safe for already-cached blocks and a data race on the `blocks.len()` append (two readers of one stream). Rule: `parallel_gate` marks a body ineligible for `par_aggregate` when it reaches a stream builtin or `phrase_from_*` (it already requires a pure body); the runtime additionally guards `LazyBlocks` with a mutex and makes the append idempotent per block index, so a fork that does slip through reads each block once. Workers' results leave as `Detached` and can carry `DAttr::LazyTail` |
| parser machine (§3.3 "reset-surviving holders") | `read_term` from a stream reads the term text through the stream layer on the main machine, then parses on the parser machine as today; no attvar crosses machines |
| `var_counter` (§5.7) | fresh tails take `_LZ<n>` names (a new kind in the §3.4 table, pre-increment); CP backtrack does not restore the counter, so a re-forced block gets new names, which is unobservable (the old tail was unbound and is gone) |
| open question "old cell vs address-only trail" (§16) | `put_attr` undo and attvar bind undo both want the old cell back; this review counts as one more input to that decision in favour of the old-cell form |

## 8. Phased plan

Small D-rows, numbered provisionally as D-S1… (renumber into the project's
D sequence when landing). Each row has a test gate against SWI through
`tests/test_wam_rust_stream_swi_parity.pl`: a row lands by turning its
`blocked(...)` tests into supported ones without changing any S-row.

| row | content | gate | before or during the rewrite |
| --- | --- | --- | --- |
| D-S1 | Diagnostics for unresolved goals: `warn_unresolved_goal` on by default; optional `existence_error(procedure, PI)` behind a flag. Register `at_end_of_stream/0` so it is a builtin (G6) | G6 unblocked; the baseline-fixes and parity suites unchanged | **before**: tiny, catches every later gap loudly |
| D-S2 | Stream layer core: `Stream` table, `user_input`/`user_output`/`user_error`, handles as `$stream(Id)`, `open/3,4`, `close/1,2`, `current_*`, `set_*`, `get_char/1,2`, `peek_char/1,2`, `get_code/1,2`, `peek_code/1,2`, `put_char/2`, `put_code/2`, `nl/1`, `write/2`, `print/2`, `flush_output/0,1`, `at_end_of_stream/0,1`, `read_line_to_string/2`, `read_line_to_codes/2`, `read_string/3,5`; ISO error terms; `stream_open/2` family rebased on it. `user_input` replaces `term_input` (`read/1` reads incrementally from stream 0) | G1, G2, G4, G7 unblocked; S1–S9 still pass; `tests/test_wam_rust_dynamic_builtins.pl` (`set_term_input` seam) still passes | **before**: it is a bridged, cold builtin family (`HC` §7.4) and touches no trail or heap invariant; porting it later costs the same, having it earlier gives the rewrite's gates real stream programs |
| D-S3 | `read/1,2`, `read_term/2,3` on streams: syntax error throw with `stream(S, L, LP, C)` context and resync; `syntax_errors(_)` option; `end_of_file` at EOF | G5 unblocked; the `read_syntax` row prints `syntax` then `ok` | **before** (part of D-S2's family) |
| D-S4 | `stream_property/2`, `set_stream_position/2`, `stream_position_data/3`, `line_count/2`, `character_count/2`, `'$stream_position'/4` | G3 unblocked | before |
| D-S5 | Output redirection stack: `with_output_to/2`, `format/3` to any stream, `writeq/1,2`, `write_canonical/2`, `write_term/2,3` with `quoted(true)` and operator rendering (fixes C1 for `write/1` too, behind a byte-identity gate because `HC` §10 freezes program output) | G8 unblocked; C1 unblocked only if the resolver gates stay byte-identical, otherwise `write/1` keeps today's text and only `writeq`/`print` get operators | before; the C1 part is an owner decision |
| D-S6 | Attributed variables, minimal: `attvars` table, `TrailKey::Attr`, hook point in `unify`, wakeup queue, `put_attr/3`, `get_attr/3`, `del_attr/2`, `attvar/1`, `copy_term` and `findall` attribute copying. Lowered tier drains the queue at `lowered_call` return | new unit tests: `put_attr` undone on backtracking; a hook that fails fails the unification; `( put_attr(X,m,a), fail ; var(X), \+ attvar(X) )` | **during** or just before R2: it adds a trail kind and a per-obligation arena top, which is exactly what the rewrite's `ScopeRec`/`ChoicePoint` redesign (`HC` §5.2, §5.5) is changing. Landing it on today's runtime first is possible (the `Value` form is small) but means porting it twice; recommended: land the `Value` form only if D-S7 is wanted before the rewrite ships |
| D-S7 | Native lazy tail: `LazyBlocks`, `Attr::LazyTail`, `stream_to_lazy_list/2`, `'$skip_list'/3`, `lazy_list_materialize/1`, `lazy_list_length/2`, `lazy_list_location//1`, `lazy_list_character_count//1`; `phrase/2,3`, `phrase_from_stream/2`, `phrase_from_file/2,3` as compiled library predicates | L1, L3 unblocked; plus new SWI-oracled rows: a grammar with lookahead across a 4096-byte boundary; `( L=[0'x|_] -> … ; … )` replay; force, backtrack below the force, force again (stream read count stays 1 per block, checked through a byte counter on the stream, the way `seek_fact_source` counts bytes for D43); force after `close/1` throws `existence_error(stream, S)`; `phrase_from_file` on an empty file; `findall` over a lazy list copies the attribute | with D-S6 |
| D-S8 | `lazy_list/2,3` with a Prolog `Next`, `'$nb_set_attr_arg'/3`, `lazy_read_lines/4`, `lazy_read_terms/4`, `lazy_get_codes/4` | L2 unblocked; the `replay_ok` and length rows; a `Next` that throws propagates from the unification site | after D-S7 |
| D-S9 | `freeze/2`, `when/2`, `dif/2` on the same wakeup queue; `frozen/2`; `copy_term/3` with attribute goals | SWI-oracled rows | optional, after the rewrite |

**Test strategy.** Keep the two-process oracle (SWI subprocess with the
same stdin and files) for every row; add a `--features paranoid` counter
for "bytes read per stream" and "blocks appended" so replay tests assert
exactly-once reading, not just equal output; run every supported row in
`emit_mode(interpreter)` and `emit_mode(functions)` (both agree today);
once D-S5 lands, drop the operator-free restriction on the drivers and
compare raw `write/1` text. For the rewrite's gates (`HC` §11), add the
S-rows and the D-S7 rows to the reference-model tests so `unify`'s new
hook point is covered by the obligation-model property tests (a forced
tail below and above an obligation's `h`, then rollback, then force again).

**Ordering summary.** D-S1–D-S5 before the rewrite (cold bridged family,
no invariant changes, immediate SWI parity for ordinary streams). D-S6–D-S8
are designed against the rewrite's obligations and arenas and are cheapest
to land inside R2–R4; if the owner wants lazy lists sooner, the `Value`
form in §7.3/§7.4 is self-contained (one trail kind, one hash map, one
`unify` check) and the data-placement rule means the second port is
mechanical.

## 9. Open questions

1. **Unresolved-goal policy (D-S1).** Warn by default, or throw
   `existence_error(procedure, PI)` like SWI's `unknown=error`? Throwing
   changes the result of today's silently-failing programs (G6 becomes an
   error instead of a wrong branch) and may surface in the resolver gates.
2. **`write/1` text (C1, D-S5).** `HC` §10 requires byte-identical program
   output across the rewrite. Fixing operators in `write/1` is a behaviour
   change that must be gated separately; the alternative is to fix only
   `writeq/print/write_term` and leave `write/1` as is until the rewrite
   is frozen.
3. **Replay cost.** Re-materializing a block after a rollback is O(block)
   per force; SWI's frozen global stack makes it O(1). Is a measured
   frozen-bar variant worth its invariant cost later, or is caching the
   built chain in the attvar record (safe today, not in the rewrite)
   enough?
4. **Lowered tier and wakeups.** Should `emit_mode(functions)` decline to
   lower any predicate whose head can bind an attributed variable
   (shape-gated, as today's lowering is), or drain the queue after head
   unification inside lowered code? The review proposes draining at
   `lowered_call` return plus a paranoid assertion, but the F11/T4 loops
   may need an explicit drain point.
5. **`nb_setarg` scope.** The review limits non-backtrackable assignment
   to attribute payloads. Does any planned feature (global variables
   `b_setval`/`nb_setval`, `HC` §9.2 holders table) want the general form,
   and should both share one "non-backtrackable store outside the heap"?
6. **Fork policy.** Make stream builtins and `phrase_from_*` disqualify a
   body from `par_aggregate` at compile time (recommended), or allow it with
   the mutex-guarded idempotent block append?
7. **`stream_open/2` family.** Keep the integer-handle builtins as thin
   aliases over the new layer, or deprecate them once `open/3` lands in
   Rust (they exist only because Rust lacked `open/3`; Go and C++ have
   both)?
8. **Shared-compiler registration.** Registering the full stream family in
   `WT:is_builtin_pred/2` makes every WAM target emit `BuiltinCall` for
   them; targets without an arm (Haskell, Lua, …) move from "unresolved
   call" to "builtin fails". Same observable result, but worth a note in
   `WAM_BACKEND_CONVENTIONS` §7.
9. **Side finding.** Quoted atom constants containing a newline break the
   line-based program text (§4.2). Fix in `escape_rust_string` plus
   `parse_instructions`, or switch constants to the pre-encoded cells of
   `HC` §4, which removes the text round-trip.
