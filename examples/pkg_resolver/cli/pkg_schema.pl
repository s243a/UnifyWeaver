:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% pkg_schema.pl -- the `pkg` command grammar, expressed as a cli_args REGISTRY.
%
% This file is the ONLY place the CLI's command surface is defined. It is a
% plain term in the shape `examples/cli_args/cli_args.pl` documents for
% `parse_args/3`:
%
%   Registry ::= list of Name-Entry
%   Entry    ::= schema(Options, Positionals) | group(Actions)
%   Options  ::= list of OptionName-Kind,  Kind ∈ boolean | string | optional
%   Positionals: "name" required, "[name]" optional, "...name" variadic.
%
% `cli/derive.pl` renders this term to `generated/pkg_registry.json` in the
% object shape `examples/cli_args/wamjs/cliArgs.mjs` already converts back into
% the Prolog term (`toRegistryTerm/toEntryTerm`), so `cli/pkg.mjs` never has to
% know what a registry means -- it hands the object to the transpiled
% `parse_args/3` and reads the answer.
%
% NOTE on globals: cli_args' GLOBAL_OPTIONS are fixed (`--state`, `--name`) and
% belong to the parser being mirrored, not to `pkg`. `pkg`'s own cross-cutting
% options (`--catalog`, `--json`) are therefore declared per command via
% pkg_common_options/1 and must appear AFTER the command word.
%
%   swipl -q -g "use_module('examples/pkg_resolver/cli/pkg_schema.pl'), \
%                pkg_registry(R), print_term(R,[]), nl" -t halt

:- module(pkg_schema, [
    pkg_registry/1,             % -Registry
    pkg_common_options/1        % -Options
]).

%!  pkg_common_options(-Options) is det.
%
%   Accepted by every `pkg` command. `--catalog` names the catalog file the
%   query runs against; `--json` switches the machine-readable form on.
pkg_common_options([
    "catalog"-string,
    "json"-boolean
]).

%!  pkg_registry(-Registry) is det.
%
%   The `pkg` command vocabulary. Names follow Pkg (Puppy Linux) heritage
%   where Pkg had a command for the same question -- `deps`, `what-needs`,
%   `orphans` -- and are spelled as questions where uw-resolve answers
%   something Pkg could not ask (`why-blocked`, `why-frozen`, `safe-upgrade`).
pkg_registry(Registry) :-
    pkg_common_options(C),
    Registry = [
        % pkg resolve <name> [more...]      -- classic closure (ignores the base)
        "resolve"-schema(C, ["name", "...more"]),

        % pkg install-plan <name> [more...] -- layered closure + install order
        "install-plan"-schema(C, ["name", "...more"]),

        % pkg layer <name> [more...]        -- Pkg's `sfs-combine` spelling
        "layer"-schema(C, ["name", "...more"]),

        % pkg why-blocked <name>            -- the frozen-base ceilings
        "why-blocked"-schema(C, ["name"]),

        % pkg deps <name>                   -- Pkg `deps|e` / `list-deps|le`
        "deps"-schema(C, ["name"]),

        % pkg what-needs <name> [--installed]  -- Pkg `what-needs|wn`
        "what-needs"-schema(["installed"-boolean|C], ["name"]),

        % pkg orphans <name>                -- Pkg `remove`'s orphan trim
        "orphans"-schema(C, ["name"]),

        % pkg why-frozen <name>             -- one hold's freeze reason
        "why-frozen"-schema(C, ["name"]),

        % pkg audit                         -- every hold's freeze reason
        "audit"-schema(C, []),

        % pkg safe-upgrade <name> <version> -- verdict + coordinated set
        "safe-upgrade"-schema(C, ["name", "version"])
    ].
