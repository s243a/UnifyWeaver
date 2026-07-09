% =============================================================================
% cpp_runtime_parser_wrappers.pl
%
% Thin Prolog wrappers that map standard SWI parsing builtin names
% (read_term_from_atom/2, /3) onto the portable Prolog term parser
% in src/unifyweaver/core/prolog_term_parser.pl.
%
% Used by the WAM-cpp target when runtime_parser(compiled) is
% requested: write_wam_cpp_project auto-includes both the parser
% predicates AND these wrappers so a generated program can call
% read_term_from_atom/2 directly without having to learn the
% portable-parser API.
%
% Lives next to prolog_term_parser.pl rather than under
% targets/wam_cpp_* because the wrappers themselves are
% target-agnostic Prolog. Any target that compiles the portable
% parser in via the same hook can reuse them; only the codegen
% integration is target-specific.
%
% See:
%   docs/design/RUNTIME_PARSER_TRANSPILATION_*.md
%   wam_cpp_target:expand_cpp_runtime_parser_predicates/3
% =============================================================================

:- module(cpp_runtime_parser_wrappers, [
    read_term_from_atom/2,
    read_term_from_atom/3,
    parse_atom_to_term/2,
    parse_term_to_atom/2
]).

:- use_module(prolog_term_parser, [
    parse_term_from_atom/3,
    canonical_op_table/1
]).

%% read_term_from_atom(+Atom, -Term) is semidet.
%
% Parse Atom into a Prolog Term using the canonical operator
% table. Mirrors SWI's read_term_from_atom/2 surface for targets
% that compile the portable parser in.
read_term_from_atom(Atom, Term) :-
    canonical_op_table(Ops),
    parse_term_from_atom(Atom, Ops, Term).

%% read_term_from_atom(+Atom, -Term, +Options) is semidet.
%
% This portable wrapper silently ignores options. Targets may intercept the
% builtin before wrapper dispatch and implement options around the parser's
% /4 environment result; the Rust runtime currently supports
% variable_names/1 this way. Keeping the fallback wrapper permissive preserves
% a stable surface for targets that have not adopted option handling yet.
read_term_from_atom(Atom, Term, _Options) :-
    read_term_from_atom(Atom, Term).

%% parse_atom_to_term(+Atom, -Term) is semidet.
%
% Operator-aware atom-to-term conversion. Equivalent in shape to
% SWI's atom_to_term/3 but drops the bindings argument -- the
% portable parser does not return a separate bindings list; it
% preserves variable identity within one parse internally.
%
% Available only in compiled mode. Uses a different name from
% the SWI standard atom_to_term/3 because that name is registered
% as a WAM-cpp builtin (is_builtin_pred(atom_to_term, 3)) and the
% compiler emits builtin_call for it, which bypasses the label
% dispatch where a wrapper would live. See the PR #2334 commit
% discussion for the three ways one could route the standard
% name through (option C -- explicit names -- chosen here).
parse_atom_to_term(Atom, Term) :-
    canonical_op_table(Ops),
    parse_term_from_atom(Atom, Ops, Term).

%% parse_term_to_atom(?Term, ?Atom) is semidet.
%
% Bidirectional term/atom conversion with operator support on the
% parse side. Mirrors SWI's term_to_atom/2:
%
%   - (+Term, ?Atom): render Term to an atom; unify with Atom.
%   - (-Term, +Atom): parse Atom into Term using the canonical op
%     table (operator notation supported -- this is what you cannot
%     get from the native term_to_atom/2 builtin).
%   - (+Term, +Atom): render and check via unification.
%
% Forward mode delegates to the WAM-cpp native term_to_atom/2
% builtin (rendering does not need operator support and the
% native path is faster). Reverse mode goes through
% parse_term_from_atom/3.
parse_term_to_atom(Term, Atom) :-
    (   nonvar(Term)
    ->  term_to_atom(Term, Atom)
    ;   canonical_op_table(Ops),
        parse_term_from_atom(Atom, Ops, Term)
    ).
