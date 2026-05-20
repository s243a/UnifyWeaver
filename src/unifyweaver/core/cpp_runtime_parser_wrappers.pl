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
    read_term_from_atom/3
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
% Options are silently ignored in v1 -- the portable parser does
% not yet support variable_names/1, character_escapes/1, etc.
% Documenting this here so the wrapper has a stable surface and
% the option-handling work has an obvious home for a follow-up.
read_term_from_atom(Atom, Term, _Options) :-
    read_term_from_atom(Atom, Term).
