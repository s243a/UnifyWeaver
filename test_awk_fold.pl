:- encoding(utf8).
% Test AWK target with fold/reduce patterns

% Simple linear recursive sum (fold pattern)
sum_list([], 0).
sum_list([H|T], Sum) :-
    sum_list(T, RestSum),
    Sum is H + RestSum.

% Count pattern
count_list([], 0).
count_list([_|T], Count) :-
    count_list(T, RestCount),
    Count is RestCount + 1.

% For AWK, we want these to compile to aggregation:
% sum: { sum += $1 } END { print sum }
% count: { count++ } END { print count }

% Test: manually show what we want
test_concept :-
    write('Desired AWK output for sum:'), nl,
    write('{ sum += $1 }'), nl,
    write('END { print sum }'), nl, nl,

    write('Desired AWK output for count:'), nl,
    write('{ count++ }'), nl,
    write('END { print count }'), nl.

:- initialization(test_concept).
