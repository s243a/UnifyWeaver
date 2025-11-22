:- module(python_target, [
    compile_predicate_to_python/3
]).

:- meta_predicate compile_predicate_to_python(:, +, -).

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
%
compile_predicate_to_python(PredicateIndicator, Options, PythonCode) :-
    % Handle module expansion (meta_predicate ensures M:Name/Arity)
    (   PredicateIndicator = Module:Name/Arity
    ->  true
    ;   PredicateIndicator = Name/Arity, Module = user
    ),
    
    functor(Head, Name, Arity),
    findall((Head, Body), clause(Module:Head, Body), Clauses),
    (   Clauses == []
    ->  throw(error(clause_not_found(Module:Head), _))
    ;   true
    ),
    
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

header("import sys\nimport json\nfrom typing import Iterator, Dict, Any\n\n").

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
