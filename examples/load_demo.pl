% Helper script to load data sources and demo
% Usage: ?- [examples/load_demo].

% Load modules only if not already loaded
:- (current_module(csv_source) -> true ; use_module('src/unifyweaver/sources/csv_source')).
:- (current_module(http_source) -> true ; use_module('src/unifyweaver/sources/http_source')).
:- (current_module(json_source) -> true ; use_module('src/unifyweaver/sources/json_source')).
:- (current_module(python_source) -> true ; use_module('src/unifyweaver/sources/python_source')).

% Load the demo
:- [examples/data_sources_demo].

:- format('~n✅ Demo loaded! Run: ?- main.~n~n', []).
