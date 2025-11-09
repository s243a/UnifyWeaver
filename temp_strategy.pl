% choose_strategy/3 - Determines the best context-gathering strategy.
% Signature: choose_strategy(+FileCount, +Budget, -Strategy)

choose_strategy(1, _, single_file_precision).

choose_strategy(FileCount, Budget, balanced_deep_dive) :-
    FileCount > 1,
    EstimatedCost is 0.002 * FileCount,
    EstimatedCost =< Budget.

choose_strategy(FileCount, Budget, quick_triage) :-
    FileCount > 1,
    EstimatedCost is 0.002 * FileCount,
    EstimatedCost > Budget.
