:- encoding(utf8).
%% End-to-end demo of the WAM R hybrid target.
%%
%% Genealogy with a small family tree, a recursive ancestor relation,
%% and a foreign handler that returns ages from a lookup table. The
%% demo exercises every major mechanism of the target in one ~80-line
%% Prolog file:
%%
%%   - WAM-compiled facts (`parent_of/2`)
%%   - Recursive Prolog (`ancestor/2`)
%%   - Mixed WAM + foreign body (`ancestor_age_above/2`)
%%   - Foreign handler returning bindings (`age_of/2`)
%%   - `findall/3` aggregation over the dynamic store
%%
%% Run:
%%   swipl -g build_demo -t halt examples/wam_r_demo/genealogy.pl
%%
%% That writes a self-contained R project to ./_genealogy_out_r/.
%% Query it (no compile step -- Rscript loads the generated source
%% directly):
%%
%%   cd _genealogy_out_r
%%   Rscript R/generated_program.R 'ancestor/2' alice carol
%%   # -> true (alice -> bob -> carol)
%%   Rscript R/generated_program.R 'ancestor_age_above/2' alice 50
%%   # -> true (bob is 60)

:- use_module('../../src/unifyweaver/targets/wam_r_target').

% ---------------- Family tree ----------------
% A small chain: alice -> bob -> carol -> dan, plus a cousin branch.

:- dynamic user:parent_of/2.
user:parent_of(alice, bob).
user:parent_of(bob,   carol).
user:parent_of(carol, dan).
user:parent_of(alice, eve).
user:parent_of(eve,   frank).

% ---------------- Recursive ancestor ----------------
% Standard transitive-closure pattern. Compiles to WAM bytecode.

:- dynamic user:ancestor/2.
user:ancestor(X, Y) :- user:parent_of(X, Y).
user:ancestor(X, Y) :- user:parent_of(X, Z), user:ancestor(Z, Y).

% ---------------- Foreign-side computation ----------------
% `age_of/2` is intentionally NOT a Prolog clause -- it's served by
% a foreign handler injected at codegen time. The handler holds a
% map from atom names to integer ages and binds reg(2) on a hit.
%
% `ancestor_age_above(Person, MinAge)` is a regular Prolog predicate
% that mixes the WAM-compiled `ancestor/2` with the foreign `age_of/2`.

:- dynamic user:age_of/2.
user:age_of(_, _).   % stub clause -- body replaced by CallForeign
:- dynamic user:ancestor_age_above/2.
user:ancestor_age_above(Person, MinAge) :-
    user:ancestor(Person, A),
    user:age_of(A, Age),
    Age > MinAge.

% ---------------- All-ancestors aggregation ----------------
% Demonstrates the compiled findall/3 path with a recursive goal.

:- dynamic user:all_ancestors/2.
user:all_ancestors(Person, As) :-
    findall(A, user:ancestor(Person, A), As).

% ---------------- R foreign handler ----------------
% R-language source for the age_of/2 handler. Returns
% list(ok = TRUE, bindings = list(list(idx = 2L, val = IntTerm(N))))
% on a hit, list(ok = FALSE) on a miss.

age_handler_code(C) :-
    C = "function(state, args, table) {\n  ages <- list(bob=60L, carol=35L, dan=12L, eve=58L, frank=28L)\n  v <- WamRuntime$deref(state, args[[1]])\n  if (is.null(v) || is.null(v$tag) || v$tag != \"atom\") return(list(ok = FALSE))\n  name <- WamRuntime$string_of(table, v$id)\n  if (!(name %in% names(ages))) return(list(ok = FALSE))\n  list(ok = TRUE, bindings = list(list(idx = 2L, val = IntTerm(ages[[name]]))))\n}".

% ---------------- Build the project ----------------

build_demo :-
    age_handler_code(Handler),
    write_wam_r_project(
        [ user:parent_of/2,
          user:ancestor/2,
          user:age_of/2,
          user:ancestor_age_above/2,
          user:all_ancestors/2 ],
        [ module_name('genealogy.demo'),
          intern_atoms([alice, bob, carol, dan, eve, frank]),
          foreign_predicates([age_of/2]),
          r_foreign_handlers([handler(age_of/2, Handler)])
        ],
        '_genealogy_out_r'),
    format("[OK] Wrote project to ./_genealogy_out_r/~n"),
    format("Try: cd _genealogy_out_r && Rscript R/generated_program.R 'ancestor/2' alice carol~n").
