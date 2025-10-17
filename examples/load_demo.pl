% Helper script to load data sources and demo
% Usage: ?- [examples/load_demo].

% Load the public sources interface (provides source/3)
:- use_module('src/unifyweaver/sources').

% Load plugin modules for side-effects only (no imports)
% Using load_files/2 with imports([]) to explicitly prevent imports
:- load_files('src/unifyweaver/sources/csv_source', [imports([])]).
:- load_files('src/unifyweaver/sources/http_source', [imports([])]).
:- load_files('src/unifyweaver/sources/json_source', [imports([])]).
:- load_files('src/unifyweaver/sources/python_source', [imports([])]).

% Load the demo
:- [examples/data_sources_demo].

:- format('~n✅ Demo loaded! Run: ?- main.~n~n', []).
