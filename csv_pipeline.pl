:- module(csv_pipeline, [get_user_age/2]).
:- use_module('src/unifyweaver/sources').
:- use_module('src/unifyweaver/sources/csv_source').

% 1. Define the data source from the CSV file
% Name should be just 'users' not 'users/3', arity is detected from CSV header
% Option should be 'csv_file' not 'file'
:- source(csv, users, [csv_file('test_data/test_users.csv'), has_header(true)]).

% 2. Define the processing logic
get_user_age(Name, Age) :-
    users(_, Name, Age).
