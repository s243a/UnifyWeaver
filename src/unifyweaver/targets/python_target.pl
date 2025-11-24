:- module(python_target, [
    compile_predicate_to_python/3
]).

:- meta_predicate compile_predicate_to_python(:, +, -).

% Conditional import of call_graph for mutual recursion detection
% Falls back gracefully if module not available
:- catch(use_module('../core/advanced/call_graph'), _, true).

/** <module> Python Target Compiler
 *
 * Compiles Prolog predicates to Python scripts using a generator-based pipeline.
 *
 * @author John William Creighton
 * @license MIT
 */

%% compile_predicate_to_python(+Predicate, +Options, -PythonCode)
%
% Compiles the given Predicate to a complete Python script.
%
% Options:
%   * record_format(Format) - 'jsonl' (default) or 'nul_json'
%   * mode(Mode) - 'procedural' (default) or 'generator'
%
compile_predicate_to_python(PredicateIndicator, Options, PythonCode) :-
    % Handle module expansion (meta_predicate ensures M:Name/Arity)
    (   PredicateIndicator = Module:Name/Arity
    ->  true
    ;   PredicateIndicator = Name/Arity, Module = user
    ),
    
    % Determine ordering constraint
    (   member(ordered(_Order), Options) % Changed AllOptions to Options
    ->  true
    ;   _Order = true  % Default: ordered
    ),
    
    % Determine evaluation mode
    option(mode(Mode), Options, procedural),
    
    % Dispatch to appropriate compiler
    (   Mode == generator
    ->  compile_generator_mode(Name, Arity, Module, Options, PythonCode)
    ;   compile_procedural_mode(Name, Arity, Module, Options, PythonCode)
    ).

%% compile_procedural_mode(+Name, +Arity, +Module, +Options, -PythonCode)
%  Current implementation (renamed for clarity)
compile_procedural_mode(Name, Arity, Module, Options, PythonCode) :-
    functor(Head, Name, Arity),
    findall((Head, Body), clause(Module:Head, Body), Clauses),
    (   Clauses == []
    ->  throw(error(clause_not_found(Module:Head), _))
    ;   true
    ),
    
    % Check if predicate is recursive
    (   is_recursive_predicate(Name, Clauses)
    ->  compile_recursive_predicate(Name, Arity, Clauses, Options, PythonCode)
    ;   compile_non_recursive_predicate(Name, Arity, Clauses, Options, PythonCode)
    ).

%% compile_non_recursive_predicate(+Name, +Arity, +Clauses, +Options, -PythonCode)
compile_non_recursive_predicate(_Name, Arity, Clauses, Options, PythonCode) :-
    % Generate clause functions
    findall(ClauseCode, (
        nth0(Index, Clauses, (ClauseHead, ClauseBody)),
        translate_clause(ClauseHead, ClauseBody, Index, Arity, ClauseCode)
    ), ClauseCodes),
    atomic_list_concat(ClauseCodes, "\n", AllClausesCode),
    
    % Generate process_stream calls
    findall(Call, (
        nth0(Index, Clauses, _),
        format(string(Call), "        yield from _clause_~d(record)", [Index])
    ), Calls),
    atomic_list_concat(Calls, "\n", CallsCode),
    
    header(Header),
    helpers(Helpers),
    
    % Select format
    option(record_format(Format), Options, jsonl),
    (   Format == nul_json
    ->  Reader = "read_nul_json", Writer = "write_nul_json"
    ;   Reader = "read_jsonl", Writer = "write_jsonl"
    ),
    
    format(string(Main), 
"
def main():
    records = ~w(sys.stdin)
    results = process_stream(records)
    ~w(results, sys.stdout)

if __name__ == '__main__':
    main()
", [Reader, Writer]),
    
    format(string(Logic), 
"
~s

def process_stream(records: Iterator[Dict]) -> Iterator[Dict]:
    \"\"\"Generated predicate logic.\"\"\"
    for record in records:
~s
\n", [AllClausesCode, CallsCode]),

    format(string(PythonCode), "~s~s~s~s", [Header, Helpers, Logic, Main]).

%% compile_recursive_predicate(+Name, +Arity, +Clauses, +Options, -PythonCode)
compile_recursive_predicate(Name, Arity, Clauses, Options, PythonCode) :-
    % Separate base and recursive clauses
    partition(is_recursive_clause_for(Name), Clauses, RecClauses, BaseClauses),
    
    % Check if part of mutual recursion group
    (   is_mutually_recursive(Name/Arity, MutualGroup),
        length(MutualGroup, GroupSize),
        GroupSize > 1
    ->  % Mutual recursion - compile entire group together
        compile_mutual_recursive_group(MutualGroup, Options, PythonCode)
    ;   % Single predicate recursion
        % Check if this is tail recursion (can be optimized to a loop)
        (   is_tail_recursive(Name, RecClauses)
        ->  compile_tail_recursive(Name, Arity, BaseClauses, RecClauses, Options, PythonCode)
        ;   compile_general_recursive(Name, Arity, BaseClauses, RecClauses, Options, PythonCode)
        )
    ).

%% is_mutually_recursive(+Pred, -MutualGroup)
%  Check if predicate is part of a mutual recursion group
%  Uses call graph analysis from advanced recursion modules
is_mutually_recursive(Pred, MutualGroup) :-
    % Try to use call_graph:predicates_in_group if module is loaded
    catch(
        (   call_graph:predicates_in_group(Pred, Group),
            length(Group, Len),
            Len > 1,
            MutualGroup = Group
        ),
        _Error,
        fail  % Silently fail if module not available or predicates not found
    ).

%% is_tail_recursive(+Name, +RecClauses)
%  Check if recursive call is in tail position
is_tail_recursive(Name, RecClauses) :-
    member((_, Body), RecClauses),
    % Get the last goal in the body
    get_last_goal(Body, LastGoal),
    functor(LastGoal, Name, _).

%% get_last_goal(+Body, -LastGoal)
get_last_goal((_, B), LastGoal) :- !, get_last_goal(B, LastGoal).
get_last_goal(Goal, Goal).

%% get_last_goal(+Body, -LastGoal)
get_last_goal((_, B), LastGoal) :- !, get_last_goal(B, LastGoal).
get_last_goal(Goal, Goal).

%% compile_mutual_recursive_group(+Predicates, +Options, -PythonCode)
%  Compile a group of mutually recursive predicates together
%  Example: [is_even/1, is_odd/1]
compile_mutual_recursive_group(Predicates, Options, PythonCode) :-
    % Generate worker functions for each predicate in the group
    findall(WorkerCode,
        (   member(Pred/Arity, Predicates),
            atom_string(Pred, PredStr),
            functor(Head, Pred, Arity),
            findall((Head, Body), clause(Head, Body), Clauses),
            partition(is_recursive_clause_for(Pred), Clauses, RecClauses, BaseClauses),
            generate_mutual_worker(PredStr, Arity, BaseClauses, RecClauses, Predicates, WorkerCode)
        ),
        WorkerCodes
    ),
    atomic_list_concat(WorkerCodes, "\n\n", AllWorkersCode),
    
    % Generate wrappers for each predicate
    findall(WrapperCode,
        (   member(Pred/Arity, Predicates),
            atom_string(Pred, PredStr),
            generate_mutual_wrapper(PredStr, Arity, WrapperCode)
        ),
        WrapperCodes
    ),
    atomic_list_concat(WrapperCodes, "\n\n", AllWrappersCode),
    
    header_with_functools(Header),
    helpers(Helpers),
    
    % Select format
    option(record_format(Format), Options, jsonl),
    (   Format == nul_json
    ->  Reader = "read_nul_json", Writer = "write_nul_json"
    ;   Reader = "read_jsonl", Writer = "write_jsonl"
    ),
    
    % For mutual recursion, generate a dispatcher that handles all predicates
    findall(Pred/Arity, member(Pred/Arity, Predicates), PredList),
    generate_mutual_dispatcher(PredList, DispatcherCode),
    
    format(string(Main), 
"
def main():
    records = ~w(sys.stdin)
    results = process_stream(records)
    ~w(results, sys.stdout)

if __name__ == '__main__':
    main()
", [Reader, Writer]),
    
    format(string(Logic), 
"
~s

~s

~s
", [AllWorkersCode, AllWrappersCode, DispatcherCode]),

    format(string(PythonCode), "~s~s~s~s", [Header, Helpers, Logic, Main]).

%% generate_mutual_worker(+PredStr, +Arity, +BaseClauses, +RecClauses, +AllPredicates, -WorkerCode)
%  Generate worker function for one predicate in mutual group
generate_mutual_worker(PredStr, Arity, BaseClauses, RecClauses, _AllPredicates, WorkerCode) :-
    (   Arity =:= 1
    ->  % Binary predicate with single argument
        (   BaseClauses = [(BaseHead, _BaseBody)|_]
        ->  BaseHead =.. [_, BaseValue],
            (   number(BaseValue)
            ->  format(string(BaseCondition), "arg == ~w", [BaseValue])
            ;   BaseCondition = "False"
            ),
            BaseReturn = "True"
        ;   BaseCondition = "False", BaseReturn = "False"
        ),
        
        % Extract recursive case - find which function it calls
        (   RecClauses = [(_RecHead, RecBody)|_]
        ->  extract_mutual_call(RecBody, CalledPred, CalledArg),
            atom_string(CalledPred, CalledPredStr),
            translate_call_arg_simple(CalledArg, PyArg)
        ;   CalledPredStr = "unknown", PyArg = "arg"
        ),
        
        format(string(WorkerCode),
"@functools.cache
def _~w_worker(arg):
    # Base case
    if ~s:
        return ~s
    
    # Mutual recursive case
    return _~w_worker(~s)
", [PredStr, BaseCondition, BaseReturn, CalledPredStr, PyArg])
    ;   % Unsupported arity
        format(string(WorkerCode), "# ERROR: Mutual recursion only supports arity 1, got arity ~d\n", [Arity])
    ).

%% extract_mutual_call(+Body, -CalledPred, -CalledArg)
%  Extract the predicate call and its argument from recursive clause
extract_mutual_call(Body, CalledPred, CalledArg) :-
    extract_goals_list(Body, Goals),
    member(Call, Goals),
    compound(Call),
    Call =.. [CalledPred, CalledArg],
    \+ member(CalledPred, [is, '>', '<', '>=', '=<', '=:=', '==']).

%% translate_call_arg_simple(+Arg, -PyArg)
translate_call_arg_simple(Expr, PyArg) :-
    (   Expr = (_ - Const),
        number(Const)
    ->  format(string(PyArg), "arg - ~w", [Const])
    ;   Expr = (_ + Const),
        number(Const)
    ->  format(string(PyArg), "arg + ~w", [Const])
    ;   PyArg = "arg"
    ).

%% generate_mutual_wrapper(+PredStr, +Arity, -WrapperCode)
generate_mutual_wrapper(PredStr, Arity, WrapperCode) :-
    (   Arity =:= 1
    ->  format(string(WrapperCode),
"def _~w_clause(v_0: Dict) -> Iterator[Dict]:
    # Extract input
    keys = list(v_0.keys())
    if not keys:
        return
    input_key = keys[0]
    input_val = v_0[input_key]
    
    # Call worker
    result = _~w_worker(input_val)
    
    # Yield result dict
    output_key = 'result'
    yield {input_key: input_val, output_key: result}
", [PredStr, PredStr])
    ;   WrapperCode = "# ERROR: Unsupported arity for mutual recursion wrapper"
    ).

%% generate_mutual_dispatcher(+Predicates, -DispatcherCode)
%  Generate process_stream that dispatches to appropriate predicate
generate_mutual_dispatcher(Predicates, DispatcherCode) :-
    findall(Case,
        (   member(Pred/_Arity, Predicates),
            atom_string(Pred, PredStr),
            format(string(Case), "    if 'predicate' in record and record['predicate'] == '~w':\n        yield from _~w_clause(record)", [PredStr, PredStr])
        ),
        Cases
    ),
    atomic_list_concat(Cases, "\n", CasesCode),
    format(string(DispatcherCode),
"def process_stream(records: Iterator[Dict]) -> Iterator[Dict]:
    \"\"\"Generated mutual recursion dispatcher.\"\"\"
    for record in records:
~s
", [CasesCode]).

%% compile_tail_recursive(+Name, +Arity, +BaseClauses, +RecClauses, +Options, -PythonCode)
compile_tail_recursive(Name, Arity, BaseClauses, RecClauses, Options, PythonCode) :-
    % Generate iterative code with while loop
    generate_tail_recursive_code(Name, Arity, BaseClauses, RecClauses, WorkerCode),
    
    % Generate streaming wrapper
    generate_recursive_wrapper(Name, Arity, WrapperCode),
    
    header_with_functools(Header),
    helpers(Helpers),
    
    % Select format
    option(record_format(Format), Options, jsonl),
    (   Format == nul_json
    ->  Reader = "read_nul_json", Writer = "write_nul_json"
    ;   Reader = "read_jsonl", Writer = "write_jsonl"
    ),
    
    format(string(Main), 
"
def main():
    records = ~w(sys.stdin)
    results = process_stream(records)
    ~w(results, sys.stdout)

if __name__ == '__main__':
    main()
", [Reader, Writer]),
    
    format(string(Logic), 
"
~s

~s

def process_stream(records: Iterator[Dict]) -> Iterator[Dict]:
    \"\"\"Generated predicate logic - tail recursive (optimized).\"\"\"
    for record in records:
        yield from _clause_0(record)
\n", [WorkerCode, WrapperCode]),

    format(string(PythonCode), "~s~s~s~s", [Header, Helpers, Logic, Main]).

%% compile_general_recursive(+Name, +Arity, +BaseClauses, +RecClauses, +Options, -PythonCode)
compile_general_recursive(Name, Arity, BaseClauses, RecClauses, Options, PythonCode) :-
    % Generate worker function with memoization
    generate_worker_function(Name, Arity, BaseClauses, RecClauses, WorkerCode),
    
    % Generate streaming wrapper
    generate_recursive_wrapper(Name, Arity, WrapperCode),
    
    header_with_functools(Header),
    helpers(Helpers),
    
    % Select format
    option(record_format(Format), Options, jsonl),
    (   Format == nul_json
    ->  Reader = "read_nul_json", Writer = "write_nul_json"
    ;   Reader = "read_jsonl", Writer = "write_jsonl"
    ),
    
    format(string(Main), 
"
def main():
    records = ~w(sys.stdin)
    results = process_stream(records)
    ~w(results, sys.stdout)

if __name__ == '__main__':
    main()
", [Reader, Writer]),
    
    format(string(Logic), 
"
~s

~s

def process_stream(records: Iterator[Dict]) -> Iterator[Dict]:
    \"\"\"Generated predicate logic - recursive wrapper.\"\"\"
    for record in records:
        yield from _clause_0(record)
\n", [WorkerCode, WrapperCode]),

    format(string(PythonCode), "~s~s~s~s", [Header, Helpers, Logic, Main]).

%% is_recursive_predicate(+Name, +Clauses)
is_recursive_predicate(Name, Clauses) :-
    member((_, Body), Clauses),
    contains_recursive_call(Body, Name).

%% is_recursive_clause_for(+Name, +Clause)
is_recursive_clause_for(Name, (_, Body)) :-
    contains_recursive_call(Body, Name).

%% contains_recursive_call(+Body, +Name)
contains_recursive_call(Body, Name) :-
    extract_goal(Body, Goal),
    functor(Goal, Name, _),
    !.

%% extract_goal(+Body, -Goal)
extract_goal(Goal, Goal) :-
    compound(Goal),
    \+ Goal = (_,_),
    \+ Goal = (_;_).
extract_goal((A, _), Goal) :- extract_goal(A, Goal).
extract_goal((_, B), Goal) :- extract_goal(B, Goal).

translate_clause(Head, Body, Index, Arity, Code) :-
    % Instanciate variables
    numbervars((Head, Body), 0, _),
    
    % Assume first argument is the input record
    arg(1, Head, RecordVar),
    var_to_python(RecordVar, PyRecordVar),
    
    % Determine output variable (Last argument if Arity > 1, else Input)
    (   Arity > 1
    ->  arg(Arity, Head, OutputVar)
    ;   OutputVar = RecordVar
    ),
    var_to_python(OutputVar, PyOutputVar),
    
    translate_body(Body, BodyCode),
    
    format(string(Code),
"def _clause_~d(~w: Dict) -> Iterator[Dict]:
~s
    yield ~w
", [Index, PyRecordVar, BodyCode, PyOutputVar]).

translate_body((Goal, Rest), Code) :-
    !,
    translate_goal(Goal, Code1),
    translate_body(Rest, Code2),
    string_concat(Code1, Code2, Code).
translate_body(Goal, Code) :-
    translate_goal(Goal, Code).

translate_goal(get_dict(Key, Record, Value), Code) :-
    !,
    var_to_python(Record, PyRecord),
    (   var(Value)
    ->  var_to_python(Value, PyValue),
        format(string(Code), "    ~w = ~w.get('~w')\n", [PyValue, PyRecord, Key])
    ;   % Value is a constant/bound
        var_to_python(Value, PyValue),
        % Check if key exists and equals value
        format(string(Code), "    if ~w.get('~w') != ~w: return\n", [PyRecord, Key, PyValue])
    ).

translate_goal(=(Var, Dict), Code) :-
    is_dict(Dict),
    !,
    var_to_python(Var, PyVar),
    dict_pairs(Dict, _Tag, Pairs),
    maplist(pair_to_python, Pairs, PyPairList),
    atomic_list_concat(PyPairList, ', ', PairsStr),
    format(string(Code), "    ~w = {~s}\n", [PyVar, PairsStr]).

translate_goal(>(Var, Value), Code) :-
    !,
    var_to_python(Var, PyVar),
    var_to_python(Value, PyValue),
    format(string(Code), "    if not (~w > ~w): return\n", [PyVar, PyValue]).

translate_goal(true, Code) :-
    !,
    Code = "    pass\n".

translate_goal(Goal, "") :-
    format(string(Msg), "Warning: Unsupported goal ~w", [Goal]),
    print_message(warning, Msg).

pair_to_python(Key-Value, Str) :-
    var_to_python(Value, PyValue),
    format(string(Str), "'~w': ~w", [Key, PyValue]).

var_to_python('$VAR'(I), PyVar) :- 
    !, 
    format(string(PyVar), "v_~d", [I]).
var_to_python(Atom, Quoted) :- 
    atom(Atom), 
    !,
    format(string(Quoted), "\"~w\"", [Atom]).
var_to_python(Number, Number) :- 
    number(Number), 
    !.
var_to_python(Term, String) :-
    term_string(Term, String).

%% generate_tail_recursive_code(+Name, +Arity, +BaseClauses, +RecClauses, -WorkerCode)
%  Generate iterative code with while loop for tail recursion
generate_tail_recursive_code(Name, Arity, BaseClauses, RecClauses, WorkerCode) :-
    (   Arity =:= 2
    ->  generate_binary_tail_loop(Name, BaseClauses, RecClauses, WorkerCode)
    ;   Arity =:= 3
    ->  generate_ternary_tail_loop(Name, BaseClauses, RecClauses, WorkerCode)
    ;   % Fallback: generate error message
        format(string(WorkerCode), "# ERROR: Tail recursion only supported for arity 2-3, got arity ~d\n", [Arity])
    ).

%% generate_binary_tail_loop(+Name, +BaseClauses, +RecClauses, -WorkerCode)
%  Generate while loop for binary tail recursion
generate_binary_tail_loop(Name, BaseClauses, RecClauses, WorkerCode) :-
    % Extract base case pattern
    (   BaseClauses = [(BaseHead, _BaseBody)|_]
    ->  BaseHead =.. [_, BaseInput, BaseOutput],
        translate_base_case(BaseInput, BaseOutput, BaseCondition, BaseReturn)
    ;   BaseCondition = "False", BaseReturn = "None"
    ),
    
    % Extract step operation from recursive clause
    (   RecClauses = [(_RecHead, RecBody)|_]
    ->  extract_step_operation(RecBody, _StepOp)
    ;   _StepOp = "arg - 1"  % Default decrement
    ),
    
    format(string(WorkerCode),
"def _~w_worker(arg):
    # Tail recursion optimized to while loop
    current = arg
    
    # Base case check
    if ~s:
        return ~s
    
    # Iterative loop (tail recursion optimization)
    result = 1  # Initialize accumulator
    while current > 0:
        result = result * current
        current = current - 1
    
    return result
", [Name, BaseCondition, BaseReturn]).

%% extract_step_operation(+Body, -StepOp)
%  Extract the step operation for the loop
extract_step_operation(Body, StepOp) :-
    % Find 'is' expression for decrement: N1 is N - 1
    extract_goals_list(Body, Goals),
    (   member((_ is Expr), Goals),
        Expr = (_ - _)
    ->  StepOp = "arg - 1"
    ;   StepOp = "arg - 1"  % Default
    ).

%% generate_ternary_tail_loop(+Name, +BaseClauses, +RecClauses, -WorkerCode)
%  Generate while loop for ternary tail recursion with accumulator
%  Pattern: sum(0, Acc, Acc). sum(N, Acc, S) :- N > 0, ..., sum(N1, Acc1, S).
generate_ternary_tail_loop(Name, BaseClauses, RecClauses, WorkerCode) :-
    % Extract base case: pred(BaseInput, Acc, Acc)
    (   BaseClauses = [(BaseHead, _BaseBody)|_]
    ->  BaseHead =.. [_, BaseInput, _Acc, _Result],
        translate_base_case_ternary(BaseInput, BaseCondition)
    ;   BaseCondition = "False"
    ),
    
    % Extract accumulator update from recursive clause
    % sum(N, Acc, S) :- N > 0, N1 is N - 1, Acc1 is Acc + N, sum(N1, Acc1, S)
    (   RecClauses = [(_RecHead, RecBody)|_]
    ->  extract_accumulator_update(RecBody, AccUpdate)
    ;   AccUpdate = "acc + n"  % Default
    ),
    
    format(string(WorkerCode),
"def _~w_worker(n, acc):
    # Tail recursion (arity 3) optimized to while loop
    current = n
    result = acc
    
    # Base case check
    if ~s:
        return result
    
    # Iterative loop (tail recursion optimization)
    while current > 0:
        result = ~s
        current = current - 1
    
    return result
", [Name, BaseCondition, AccUpdate]).

%% translate_base_case_ternary(+Input, -Condition)
translate_base_case_ternary(Input, Condition) :-
    (   Input == []
    ->  Condition = "not current or current == []"
    ;   number(Input)
    ->  format(string(Condition), "current == ~w", [Input])
    ;   Condition = "False"
    ).

%% extract_accumulator_update(+Body, -Update)
%  Extract accumulator update expression
%  From: Acc1 is Acc + N → "result + current"
extract_accumulator_update(Body, Update) :-
    extract_goals_list(Body, Goals),
    % Find the accumulator update: Acc1 is Acc + N (or Acc * N, etc.)
    findall(Expr, member((_ is Expr), Goals), Exprs),
    % The second 'is' expression (if present) is usually the accumulator update
    (   length(Exprs, Len), Len >= 2,
        nth1(2, Exprs, AccExpr)
    ->  translate_acc_expr(AccExpr, Update)
    ;   Update = "result + current"  % Default
    ).

%% translate_acc_expr(+Expr, -PyExpr)
translate_acc_expr(Expr, PyExpr) :-
    functor(Expr, Op, 2),
    (   Op = '+'
    ->  PyOp = "+", _Order = normal
    ;   Op = '*'
    ->  PyOp = "*", _Order = normal
    ;   Op = '-'
    ->  PyOp = "-", _Order = normal
    ;   PyOp = "+", _Order = normal
    ),
    % Determine order: is it Acc + N or N + Acc?
    Expr =.. [_, Arg1, Arg2],
    (   var(Arg1), \+ var(Arg2)  % Acc + N
    ->  format(string(PyExpr), "result ~s current", [PyOp])
    ;   var(Arg2), \+ var(Arg1)  % N + Acc (reverse)
    ->  format(string(PyExpr), "current ~s result", [PyOp])
    ;   % Both vars or both ground, default
        format(string(PyExpr), "result ~s current", [PyOp])
    ).

%% generate_worker_function(+Name, +Arity, +BaseClauses, +RecClauses, -WorkerCode)
generate_worker_function(Name, Arity, BaseClauses, RecClauses, WorkerCode) :-
    % For now, only support binary recursion (Input, Output)
    (   Arity =:= 2
    ->  generate_binary_worker(Name, BaseClauses, RecClauses, WorkerCode)
    ;   % Fallback: generate error message
        format(string(WorkerCode), "# ERROR: Recursion only supported for arity 2, got arity ~d\n", [Arity])
    ).

%% generate_binary_worker(+Name, +BaseClauses, +RecClauses, -WorkerCode)
generate_binary_worker(Name, BaseClauses, RecClauses, WorkerCode) :-
    % Extract base case pattern
    (   BaseClauses = [(BaseHead, _BaseBody)|_]
    ->  BaseHead =.. [_, BaseInput, BaseOutput],
        translate_base_case(BaseInput, BaseOutput, BaseCondition, BaseReturn)
    ;   BaseCondition = "False", BaseReturn = "None"
    ),
    
    % Extract recursive case
    (   RecClauses = [(RecHead, RecBody)|_]
    ->  RecHead =.. [_, RecInput, RecOutput],
        translate_recursive_case(Name, RecInput, RecOutput, RecBody, RecCode)
    ;   RecCode = "    pass"
    ),
    
    format(string(WorkerCode),
"@functools.cache
def _~w_worker(arg):
    # Base case
    if ~s:
        return ~s
    
    # Recursive case
~s
", [Name, BaseCondition, BaseReturn, RecCode]).

%% translate_base_case(+Input, +Output, -Condition, -Return)
translate_base_case(Input, Output, Condition, Return) :-
    (   Input == []
    ->  Condition = "not arg or arg == []"
    ;   number(Input)
    ->  format(string(Condition), "arg == ~w", [Input])
    ;   Condition = "False"  % Unknown pattern
    ),
    (   number(Output)
    ->  format(string(Return), "~w", [Output])
    ;   atom(Output)
    ->  format(string(Return), "\"~w\"", [Output])
    ;   var_to_python(Output, Return)
    ).

%% translate_recursive_case(+Name, +Input, +Output, +Body, -Code)
translate_recursive_case(Name, _Input, _Output, Body, Code) :-
    % Find the recursive call and arithmetic operations
    extract_recursive_pattern(Body, Name, Pattern),
    translate_pattern_to_python(Name, Pattern, Code).

%% extract_recursive_pattern(+Body, +Name, -Pattern)
extract_recursive_pattern(Body, Name, Pattern) :-
    % factorial: N > 0, N1 is N - 1, factorial(N1, F1), F is N * F1
    % The LAST 'is' expression is the one that computes the result
    extract_goals_list(Body, Goals),
    findall(Expr, member((_ is Expr), Goals), Exprs),
    % Take the last expression (the result computation)
    (   Exprs \= [],
        last(Exprs, RecExpr),
        functor(RecExpr, Op, 2),
        member(Op, ['*', '+', '-', '/'])
    ->  member(RecCall, Goals),
        functor(RecCall, Name, _),
        Pattern = arithmetic(RecExpr, RecCall)
    ;   Pattern = unknown
    ).

extract_goals_list((A, B), [A|Rest]) :- !, extract_goals_list(B, Rest).
extract_goals_list(Goal, [Goal]).

%% translate_pattern_to_python(+Name, +Pattern, -Code)
translate_pattern_to_python(Name, arithmetic(Expr, _RecCall), Code) :-
    % For factorial: F is N * F1 → return arg * _factorial_worker(arg - 1)
    % Extract operator using functor
    functor(Expr, Op, _),
    (   Op = '*'
    ->  PyOp = "*"
    ;   Op = '+'
    ->  PyOp = "+"
    ;   Op = '-'
    ->  PyOp = "-"
    ;   PyOp = "*"  % Default to multiplication
    ),
    format(string(Code), "    return arg ~s _~w_worker(arg - 1)", [PyOp, Name]).
translate_pattern_to_python(_, unknown, "    pass  # Unknown recursion pattern").

%% generate_recursive_wrapper(+Name, +Arity, -WrapperCode)
generate_recursive_wrapper(Name, Arity, WrapperCode) :-
    % Generate _clause_0 that extracts input from dict, calls worker, yields result
    (   Arity =:= 2
    ->  format(string(WrapperCode),
"def _clause_0(v_0: Dict) -> Iterator[Dict]:
    # Extract input
    keys = list(v_0.keys())
    if not keys:
        return
    input_key = keys[0]
    input_val = v_0[input_key]
    
    # Call worker
    result = _~w_worker(input_val)
    
    # Yield result dict
    output_key = keys[1] if len(keys) > 1 else 'result'
    yield {input_key: input_val, output_key: result}
", [Name])
    ;   Arity =:= 3
    ->  format(string(WrapperCode),
"def _clause_0(v_0: Dict) -> Iterator[Dict]:
    # Extract input and accumulator from dict
    keys = list(v_0.keys())
    if len(keys) < 2:
        return
    input_key = keys[0]
    acc_key = keys[1]
    input_val = v_0[input_key]
    acc_val = v_0.get(acc_key, 0)  # Default accumulator to 0
    
    # Call worker
    result = _~w_worker(input_val, acc_val)
    
    # Yield result dict
    output_key = keys[2] if len(keys) > 2 else 'result'
    yield {input_key: input_val, acc_key: acc_val, output_key: result}
", [Name])
    ;   WrapperCode = "# ERROR: Unsupported arity for recursion wrapper"
    ).

header("import sys\nimport json\nfrom typing import Iterator, Dict, Any\n\n").

header_with_functools("import sys\nimport json\nimport functools\nfrom typing import Iterator, Dict, Any\n\n").

helpers("
def read_jsonl(stream) -> Iterator[Dict[str, Any]]:
    \"\"\"Read JSONL from stream.\"\"\"
    for line in stream:
        if line.strip():
            yield json.loads(line)

def write_jsonl(records: Iterator[Dict], stream) -> None:
    \"\"\"Write JSONL to stream.\"\"\"
    for record in records:
        stream.write(json.dumps(record) + '\\n')

def read_nul_json(stream) -> Iterator[Dict[str, Any]]:
    \"\"\"Read NUL-delimited JSON from stream.\"\"\"
    buff = ''
    while True:
        chunk = stream.read(4096)
        if not chunk:
            break
        buff += chunk
        while '\\0' in buff:
            line, buff = buff.split('\\0', 1)
            if line:
                yield json.loads(line)
    if buff and buff.strip('\\0'):
        yield json.loads(buff)

def write_nul_json(records: Iterator[Dict], stream) -> None:
    \"\"\"Write NUL-delimited JSON to stream.\"\"\"
    for record in records:
        stream.write(json.dumps(record) + '\\0')
\n").

%% ============================================
%% GENERATOR MODE (Semi-Naive Evaluation)
%% ============================================

%% compile_generator_mode(+Name, +Arity, +Module, +Options, -PythonCode)
%  Compile using generator-based semi-naive fixpoint iteration
%  Similar to C# query engine approach
compile_generator_mode(Name, Arity, Module, Options, PythonCode) :-
    functor(Head, Name, Arity),
    findall((Head, Body), clause(Module:Head, Body), Clauses),
    (   Clauses == []
    ->  format(string(PythonCode), "# ERROR: No clauses found for ~w/~w\n", [Name, Arity])
    ;   generate_generator_code(Name, Arity, Clauses, Options, PythonCode)
    ).

%% generate_generator_code(+Name, +Arity, +Clauses, +Options, -PythonCode)
generate_generator_code(_Name, _Arity, Clauses, Options, PythonCode) :-
    % Generate components
    generator_header(Header),
    generator_helpers(Options, Helpers),
    generate_rule_functions(Name, Clauses, RuleFunctions),
    generate_fixpoint_loop(Name, Clauses, FixpointLoop),
    
    % Main function
    option(record_format(Format), Options, jsonl),
    (   Format == nul_json
    ->  Reader = "read_nul_json", Writer = "write_nul_json"
    ;   Reader = "read_jsonl", Writer = "write_jsonl"
    ),
    
    format(string(Main),
"
def main():
    records = ~w(sys.stdin)
    results = process_stream_generator(records)
    ~w(results, sys.stdout)

if __name__ == '__main__':
    main()
", [Reader, Writer]),
    
    atomic_list_concat([Header, Helpers, RuleFunctions, FixpointLoop, Main], "\n", PythonCode).

%% generator_header(-Header)
generator_header(Header) :-
    Header = "import sys
import json
from typing import Iterator, Dict, Any, Set
from dataclasses import dataclass

# FrozenDict - hashable dictionary for use in sets
@dataclass(frozen=True)
class FrozenDict:
    '''Immutable dictionary that can be used in sets.'''
    items: tuple
    
    @staticmethod
    def from_dict(d: Dict) -> 'FrozenDict':
        return FrozenDict(tuple(sorted(d.items())))
    
    def to_dict(self) -> Dict:
        return dict(self.items)
    
    def get(self, key, default=None):
        for k, v in self.items:
            if k == key:
                return v
        return default
    
    def __contains__(self, key):
        return any(k == key for k, _ in self.items)
    
    def __repr__(self):
        return f'FrozenDict({dict(self.items)})'
".

%% generator_helpers(+Options, -Helpers)
generator_helpers(Options, Helpers) :-
    option(record_format(Format), Options, jsonl),
    (   Format == nul_json
    ->  NulReader = "
def read_nul_json(stream: Any) -> Iterator[Dict]:
    '''Read NUL-separated JSON records.'''
    buffer = ''
    while True:
        char = stream.read(1)
        if not char:
            if buffer.strip():
                yield json.loads(buffer)
            break
        if char == '\\0':
            if buffer.strip():
                yield json.loads(buffer)
                buffer = ''
        else:
            buffer += char

def write_nul_json(records: Iterator[Dict], stream: Any):
    '''Write NUL-separated JSON records.'''
    for record in records:
        stream.write(json.dumps(record) + '\\0')
",
        JsonlReader = ""
    ;   JsonlReader = "
def read_jsonl(stream: Any) -> Iterator[Dict]:
    '''Read JSONL records.'''
    for line in stream:
        line = line.strip()
        if line:
            yield json.loads(line)

def write_jsonl(records: Iterator[Dict], stream: Any):
    '''Write JSONL records.'''
    for record in records:
        stream.write(json.dumps(record) + '\\n')
",
        NulReader = ""
    ),
    atomic_list_concat([JsonlReader, NulReader], "", Helpers).

%% generate_rule_functions(+Name, +Clauses, -RuleFunctions)
generate_rule_functions(Name, Clauses, RuleFunctions) :-
    findall(RuleFunc,
        (   nth1(RuleNum, Clauses, (Head, Body)),
            generate_rule_function(Name, RuleNum, Head, Body, RuleFunc)
        ),
        RuleFuncs),
    atomic_list_concat(RuleFuncs, "\n\n", RuleFunctions).

%% generate_rule_function(+Name, +RuleNum, +Head, +Body, -RuleFunc)
generate_rule_function(_Name, RuleNum, Head, Body, RuleFunc) :-
    (   Body == true
    ->  % Fact (no body) - emit constant
        translate_fact_rule(_Name, RuleNum, Head, RuleFunc)
    ;   extract_goals_list(Body, Goals),
        length(Goals, NumGoals),
        (   NumGoals == 1
        ->  % Single goal - copy/transform rule
            Goals = [SingleGoal],
            translate_copy_rule(_Name, RuleNum, Head, SingleGoal, RuleFunc)
        ;   % Multiple goals - join rule
            translate_join_rule(_Name, RuleNum, Head, Goals, RuleFunc)
        )
    ).

%% translate_fact_rule(+Name, +RuleNum, +Head, -RuleFunc)
%  Translate a fact (constant) into a rule that emits it once
translate_fact_rule(_Name, RuleNum, Head, RuleFunc) :-
    Head =.. [_Pred | Args],
    extract_constants(Args, ConstPairs),
    format_dict_pairs(ConstPairs, DictStr),
    format(string(RuleFunc),
"def _apply_rule_~w(fact: FrozenDict, total: Set[FrozenDict]) -> Iterator[FrozenDict]:
    '''Fact: ~w'''
    # Emit constant fact once (only if not already in total)
    result = FrozenDict.from_dict({~w})
    if result not in total:
        yield result
", [RuleNum, Head, DictStr]).

%% extract_constants(+Args, -Pairs)
%  Extract constant values from arguments
extract_constants(Args, Pairs) :-
    findall(Key-Value,
        (   nth1(Idx, Args, Arg),
            atom(Arg),
            \+ var(Arg),
            Key is Idx - 1,  % 0-indexed
            atom_string(Arg, Value)
        ),
        Pairs).

%% format_dict_pairs(+Pairs, -DictStr)
format_dict_pairs(Pairs, DictStr) :-
    findall(Pair,
        (   member(Key-Value, Pairs),
            format(string(Pair), "'arg~w': '~w'", [Key, Value])
        ),
        PairStrs),
    atomic_list_concat(PairStrs, ", ", DictStr).

%% translate_copy_rule(+Name, +RuleNum, +Head, +Goal, -RuleFunc)
%  Translate a simple copy/transform rule (single goal in body)
translate_copy_rule(_Name, RuleNum, Head, Goal, RuleFunc) :-
    % Extract variable mappings
    Head =.. [_HeadPred | HeadArgs],
    Goal =.. [GoalPred | GoalArgs],
    
    % Build pattern match condition
    length(GoalArgs, GoalArity),
    findall(Check,
        (   between(0, GoalArity, Idx),
            Idx < GoalArity,
            format(string(Check), "'arg~w' in fact", [Idx])
        ),
        Checks),
    atomic_list_concat(Checks, " and ", ConditionStr),
    
    % Build output dict
    findall(Assign,
        (   nth0(HeadIdx, HeadArgs, HeadArg),
            nth0(GoalIdx, GoalArgs, GoalArg),
            HeadArg == GoalArg,  % Same variable
            format(string(Assign), "'arg~w': fact.get('arg~w')", [HeadIdx, GoalIdx])
        ),
        Assigns),
    atomic_list_concat(Assigns, ", ", OutputStr),
    
    % GoalPred not currently used but keep for future reference
    atom_string(GoalPred, _GoalPredStr),
    format(string(RuleFunc),
"def _apply_rule_~w(fact: FrozenDict, total: Set[FrozenDict]) -> Iterator[FrozenDict]:
    '''Copy rule: ~w -> ~w'''
    # Match pattern and copy variables
    if ~w:
        yield FrozenDict.from_dict({~w})
", [RuleNum, Goal, Name, ConditionStr, OutputStr]).

%% translate_join_rule(+Name, +RuleNum, +Head, +Goals, -RuleFunc)
%  Translate a join rule (multiple goals in body)
translate_join_rule(_Name, RuleNum, Head, Goals, RuleFunc) :-
    length(Goals, NumGoals),
    (   NumGoals == 2
    ->  % Binary join (existing fast path)
        Goals = [Goal1, Goal2],
        translate_binary_join(_Name, RuleNum, Head, Goal1, Goal2, RuleFunc)
    ;   NumGoals >= 3
    ->  % N-way join (new!)
        translate_nway_join(RuleNum, Head, Goals, RuleFunc)
    ;   % Single goal shouldn't reach here
        format(string(RuleFunc),
"def _apply_rule_~w(fact: FrozenDict, total: Set[FrozenDict]) -> Iterator[FrozenDict]:
    '''ERROR: Invalid join - single goal should use copy rule'''
    return iter([])
", [RuleNum])
    ).

%% translate_binary_join(+Name, +RuleNum, +Head, +Goal1, +Goal2, -RuleFunc)
translate_binary_join(_Name, RuleNum, Head, Goal1, Goal2, RuleFunc) :-
    % Extract predicates and arguments
    Goal1 =.. [Pred1 | Args1],
    Goal2 =.. [Pred2 | Args2],
    Head =.. [_HeadPred | HeadArgs],
    
    % Find join variable (appears in both goals)
    findall(Var-Idx1-Idx2,
        (   nth0(Idx1, Args1, Var),
            nth0(Idx2, Args2, Var),
            \+ atom(Var)  % Variable, not constant
        ),
        JoinVars),
    
    % Build join condition
    (   JoinVars = [_Var-JIdx1-JIdx2|_]
    ->  format(string(JoinCond), "other.get('arg~w') == fact.get('arg~w')", [JIdx2, JIdx1])
    ;   JoinCond = "True"  % No explicit join
    ),
    
    % Build output mapping
    findall(OutAssign,
        (   nth0(HIdx, HeadArgs, HVar),
            (   nth0(G1Idx, Args1, HVar)
            ->  format(string(OutAssign), "'arg~w': fact.get('arg~w')", [HIdx, G1Idx])
            ;   nth0(G2Idx, Args2, HVar),
                format(string(OutAssign), "'arg~w': other.get('arg~w')", [HIdx, G2Idx])
            )
        ),
        OutAssigns),
    atomic_list_concat(OutAssigns, ", ", OutputMapping),
    
    % Pattern match for first goal
    length(Args1, Arity1),
    findall(PatCheck,
        (   between(0, Arity1, Idx),
            Idx < Arity1,
            format(string(PatCheck), "'arg~w' in fact", [Idx])
        ),
        PatChecks),
    atomic_list_concat(PatChecks, " and ", Pattern1),
    
    atom_string(Pred1, _Pred1Str),
    atom_string(Pred2, _Pred2Str),
    
    format(string(RuleFunc),
"def _apply_rule_~w(fact: FrozenDict, total: Set[FrozenDict]) -> Iterator[FrozenDict]:
    '''Join rule: ~w :- ~w, ~w'''
    # Match first goal pattern
    if ~w:
        # Join with second predicate facts
        for other in total:
            # Check join condition
            if ~w:
                yield FrozenDict.from_dict({~w})
", [RuleNum, Head, Goal1, Goal2, Pattern1, JoinCond, OutputMapping]).

%% translate_nway_join(+RuleNum, +Head, +Goals, -RuleFunc)
%  Translate N-way join (3+ goals)
translate_nway_join(RuleNum, Head, Goals, RuleFunc) :-
    % Strategy: First goal from fact, rest joined from total
    Goals = [FirstGoal | RestGoals],
    
    % Build pattern match for first goal
    FirstGoal =.. [_Pred1 | Args1],
    length(Args1, Arity1),
    findall(Check,
        (   between(0, Arity1, Idx),
            Idx < Arity1,
            format(string(Check), "'arg~w' in fact", [Idx])
        ),
        Checks),
    atomic_list_concat(Checks, " and ", Pattern1),
    
    % Build nested joins for remaining goals
    build_nested_joins(RestGoals, 1, JoinCode, _FinalIdx),
    
    % Build output mapping from head
    Head =.. [_HeadPred | HeadArgs],
    collect_all_goal_args([FirstGoal | RestGoals], AllGoalArgs),
    build_output_mapping(HeadArgs, FirstGoal, RestGoals, AllGoalArgs, OutputMapping),
    
    format(string(RuleFunc),
"def _apply_rule_~w(fact: FrozenDict, total: Set[FrozenDict]) -> Iterator[FrozenDict]:
    '''N-way join: ~w'''
    # Match first goal
    if ~w:
~w
                yield FrozenDict.from_dict({~w})
", [RuleNum, Head, Pattern1, JoinCode, OutputMapping]).

%%build_nested_joins(+Goals, +StartIdx, -JoinCode, -FinalIdx)
%  Build nested for loops for N-way joins
build_nested_joins([], Idx, "", Idx) :- !.
build_nested_joins([Goal | RestGoals], Idx, JoinCode, FinalIdx) :-
    Indent is (Idx + 1) * 4,
    format(string(Spaces), "~*c", [Indent, 32]),  % 32 = space char
    
    % Detect join conditions with previous goals
    detect_join_condition(Goal, Idx, JoinCond),
    
    format(string(ThisJoin),
"~wfor join_~w in total:
~w    if ~w:",
        [Spaces, Idx, Spaces, JoinCond]),
    
    NextIdx is Idx + 1,
    build_nested_joins(RestGoals, NextIdx, RestCode, FinalIdx),
    
    (   RestCode == ""
    ->  JoinCode = ThisJoin
    ;   format(string(JoinCode), "~w\n~w", [ThisJoin, RestCode])
    ).

%% detect_join_condition(+Goal, +Idx, -JoinCond)
%  Find variables that join with previous goals
detect_join_condition(_Goal, Idx, JoinCond) :-
    % Simplified: assume join on matching arg indices
    % TODO: Smarter variable tracking
    _PrevIdx is Idx - 1,
    format(string(JoinCond), "join_~w.get('arg0') == fact.get('arg1')", [Idx]).

%% collect_all_goal_args(+Goals, -AllArgs)
collect_all_goal_args(Goals, AllArgs) :-
    findall(Args,
        (   member(Goal, Goals),
            Goal =.. [_ | Args]
        ),
        ArgLists),
    append(ArgLists, AllArgs).

%% build_output_mapping(+HeadArgs, +FirstGoal, +RestGoals, +AllGoalArgs, -Mapping)
build_output_mapping(HeadArgs, FirstGoal, _RestGoals, _AllGoalArgs, Mapping) :-
    % Simplified: map from first goal args
    FirstGoal =.. [_ | FirstArgs],
    findall(Assign,
        (   nth0(HIdx, HeadArgs, HVar),
            nth0(GIdx, FirstArgs, HVar),
            format(string(Assign), "'arg~w': fact.get('arg~w')", [HIdx, GIdx])
        ),
        Assigns),
    atomic_list_concat(Assigns, ", ", Mapping).


%% generate_fixpoint_loop(+Name, +Clauses, -FixpointLoop)
generate_fixpoint_loop(_Name, Clauses, FixpointLoop) :-
    length(Clauses, NumRules),
    findall(RuleCall,
        (   between(1, NumRules, RuleNum),
            format(string(RuleCall), "            for new_fact in _apply_rule_~w(fact, total):", [RuleNum])
        ),
        RuleCalls),
    
    % Build nested rule application
    findall(IndentedCall,
        (   nth1(Idx, RuleCalls, Call),
            Indent is Idx * 4,
            format(string(Spaces), "~*c", [Indent, 32]),  % 32 = space
            format(string(IndentedCall), "~w~w", [Spaces, Call])
        ),
        IndentedCalls),
    atomic_list_concat(IndentedCalls, "\n", RuleCallsStr),
    
    format(string(FixpointLoop),
"
def process_stream_generator(records: Iterator[Dict]) -> Iterator[Dict]:
    '''Semi-naive fixpoint evaluation.
    
    Maintains two sets:
    - total: All facts discovered so far
    - delta: New facts from last iteration
    
    Iterates until no new facts are discovered (fixpoint reached).
    '''
    total: Set[FrozenDict] = set()
    delta: Set[FrozenDict] = set()
    
    # Initialize delta with input records
    for record in records:
        frozen = FrozenDict.from_dict(record)
        delta.add(frozen)
        total.add(frozen)
        yield record  # Yield initial facts
    
    # Fixpoint iteration (semi-naive evaluation)
    iteration = 0
    while delta:
        iteration += 1
        new_delta: Set[FrozenDict] = set()
        
        # Apply rules to facts in delta
        for fact in delta:
~w
                if new_fact not in total:
                    total.add(new_fact)
                    new_delta.add(new_fact)
                    yield new_fact.to_dict()
        
        delta = new_delta
", [RuleCallsStr]).

