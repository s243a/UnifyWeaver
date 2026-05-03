:- encoding(utf8).
%% End-to-end demo of the WAM Scala hybrid target.
%%
%% Genealogy with a small family tree, a recursive ancestor relation,
%% and a foreign handler that counts the depth of an ancestor chain
%% (purely to demonstrate the foreign-handler path side-by-side with
%% WAM-compiled clauses).
%%
%% Run:
%%   swipl -g build_demo -t halt examples/wam_scala_demo/genealogy.pl
%%
%% That writes a Scala project to ./_genealogy_out/. Build and query it:
%%
%%   cd _genealogy_out
%%   mkdir classes
%%   scalac -d classes src/main/scala/demo/genealogy/*.scala
%%   scala -classpath classes demo.genealogy.GeneratedProgram \
%%     'ancestor/2' alice charlie       # → true (great-grand)
%%   scala -classpath classes demo.genealogy.GeneratedProgram \
%%     'ancestor_age_above/2' alice 50  # → true via foreign handler

:- use_module('../../src/unifyweaver/targets/wam_scala_target').

% ---------------- Family tree ----------------
% A small chain: alice → bob → carol → dan, plus a cousin branch.

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
% `age_of/2` is intentionally NOT a Prolog clause — it's served by
% a foreign handler injected at codegen time. The handler holds a
% map from atoms to integer ages and answers (?+) queries.
%
% `ancestor_age_above(Person, MinAge)` is a regular Prolog predicate
% that mixes the WAM-compiled `ancestor/2` with the foreign `age_of/2`.

:- dynamic user:age_of/2.
user:age_of(_, _).   % stub clause — body replaced by CallForeign
:- dynamic user:ancestor_age_above/2.
user:ancestor_age_above(Person, MinAge) :-
    user:ancestor(Person, A),
    user:age_of(A, Age),
    Age > MinAge.

% ---------------- Build the project ----------------

age_handler_code(C) :-
    %  Maps atom IDs (from the program's intern table) to ages and
    %  binds reg(2) on a hit. Lookup misses backtrack/fail.
    C = "new ForeignHandler { def apply(args: Array[WamTerm]): ForeignResult = {\
\n  val ages = Map(\
\n    \"bob\"   -> 60,\
\n    \"carol\" -> 35,\
\n    \"dan\"   -> 12,\
\n    \"eve\"   -> 58,\
\n    \"frank\" -> 28\
\n  )\
\n  args(0) match {\
\n    case Atom(id) =>\
\n      val name = internTable.stringAt(id)\
\n      ages.get(name) match {\
\n        case Some(n) => ForeignBindings(Map(2 -> IntTerm(n)))\
\n        case None    => ForeignFail\
\n      }\
\n    case _ => ForeignFail\
\n  }\
\n}\
\n}".

build_demo :-
    age_handler_code(Handler),
    write_wam_scala_project(
        [ user:parent_of/2,
          user:ancestor/2,
          user:age_of/2,
          user:ancestor_age_above/2 ],
        [ package('demo.genealogy'),
          runtime_package('demo.genealogy'),
          module_name('genealogy-demo'),
          intern_atoms([alice, bob, carol, dan, eve, frank]),
          foreign_predicates([age_of/2]),
          scala_foreign_handlers([handler(age_of/2, Handler)])
        ],
        '_genealogy_out'),
    format("[OK] Wrote project to ./_genealogy_out/~n").
