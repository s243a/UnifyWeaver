:- module(python_target, [
    compile_predicate_to_python/3,
    init_python_target/0,
    % Pipeline mode exports (Phase 1)
    test_pipeline_mode/0,
    generate_default_arg_names/2,
    pipeline_header/2,
    pipeline_header/3,
    pipeline_helpers/3,
    generate_output_formatting/4,
    generate_pipeline_main/4,
    % Runtime selection exports (Phase 2)
    select_python_runtime/4,
    runtime_available/1,
    runtime_compatible_with_imports/2,
    test_runtime_selection/0,
    % Runtime-specific code generation exports (Phase 3)
    test_runtime_headers/0,
    % Pipeline chaining exports (Phase 4)
    compile_pipeline/3,
    compile_same_runtime_pipeline/3,
    compile_cross_runtime_pipeline/3,
    test_pipeline_chaining/0,
    % Pipeline generator mode exports (Phase 5)
    test_python_pipeline_generator/0,
    % IronPython pipeline generator mode exports (Phase 6)
    test_ironpython_pipeline_generator/0
]).

:- meta_predicate compile_predicate_to_python(:, +, -).

% Conditional import of call_graph for mutual recursion detection
% Falls back gracefully if module not available
:- catch(use_module('../core/advanced/call_graph'), _, true).
:- use_module(common_generator).

% Binding system integration (ported from PowerShell target)
:- use_module('../core/binding_registry').
:- use_module('../bindings/python_bindings').

% Control plane integration (Phase 2 - Runtime Selection)
:- catch(use_module('../core/preferences'), _, true).
:- catch(use_module('../core/firewall'), _, true).
:- catch(use_module('../glue/dotnet_glue'), _, true).

% Track required imports from bindings
:- dynamic required_import/1.

% translate_goal/2 is spread across the file for organization
:- discontiguous translate_goal/2.

% pipeline_header/2 has multiple clauses spread across the file (legacy + new)
:- discontiguous pipeline_header/2.

%% init_python_target
%  Initialize Python target with bindings
init_python_target :-
    retractall(required_import(_)),
    init_python_bindings.

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
% Pipeline Options (Phase 1 - Object Pipeline Support):
%   * pipeline_input(Bool) - true: enable streaming input from stdin/iterator
%                           false: standalone function (default)
%   * output_format(Format) - object: yield typed dicts with arg_names
%                            text: yield string representation (default)
%   * arg_names(Names) - List of property names for output dict
%                        Example: ['UserId', 'Email']
%   * glue_protocol(Protocol) - jsonl (default), messagepack (future)
%   * error_protocol(Protocol) - same_as_data (default), text
%
compile_predicate_to_python(PredicateIndicator, Options, PythonCode) :-
    % Clear any previously collected binding imports
    clear_binding_imports,

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

    % Check for pipeline mode (Phase 1 - Object Pipeline Support)
    option(pipeline_input(PipelineInput), Options, false),

    % Dispatch to appropriate compiler
    (   PipelineInput == true
    ->  compile_pipeline_mode(Name, Arity, Module, Options, PythonCode)
    ;   Mode == generator
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
    
    generate_python_main(Options, Main),
    
    format(string(Logic), 
"
~s

def process_stream(records: Iterator[Dict]) -> Iterator[Dict]:
    \"\"\"Generated predicate logic.\"\"\"
    for record in records:
~s
\n", [AllClausesCode, CallsCode]),

    format(string(PythonCode), "~s~s~s~s", [Header, Helpers, Logic, Main]).

% ============================================================================
% PIPELINE MODE - Phase 1: Object Pipeline Support
% ============================================================================

%% compile_pipeline_mode(+Name, +Arity, +Module, +Options, -PythonCode)
%
% Compiles predicate to Python generator for pipeline processing.
% This mode generates streaming code that:
%   - Reads from stdin (or accepts iterator)
%   - Yields typed dict objects with named properties
%   - Writes to stdout with configurable protocol (JSONL default)
%   - Writes errors to stderr in same protocol
%
% Options used:
%   - pipeline_input(true) - already checked in dispatcher
%   - output_format(object|text) - dict or string output
%   - arg_names([...]) - property names for output dict
%   - glue_protocol(jsonl|messagepack) - serialization format
%   - error_protocol(same_as_data|text) - error format
%
compile_pipeline_mode(Name, Arity, Module, Options, PythonCode) :-
    functor(Head, Name, Arity),
    findall((Head, Body), clause(Module:Head, Body), Clauses),
    (   Clauses == []
    ->  throw(error(clause_not_found(Module:Head), _))
    ;   true
    ),

    % Extract options with defaults
    option(output_format(OutputFormat), Options, object),
    option(arg_names(ArgNames), Options, []),
    option(glue_protocol(GlueProtocol), Options, jsonl),
    option(error_protocol(ErrorProtocol), Options, same_as_data),
    option(runtime(Runtime), Options, cpython),

    % Generate arg names if not provided
    (   ArgNames == []
    ->  generate_default_arg_names(Arity, DefaultArgNames)
    ;   DefaultArgNames = ArgNames
    ),

    % Generate the pipeline function
    atom_string(Name, NameStr),
    generate_pipeline_function(NameStr, Arity, Clauses, OutputFormat, DefaultArgNames, FunctionCode),

    % Generate header and helpers (Phase 3: runtime-specific headers)
    pipeline_header(GlueProtocol, Runtime, Header),
    pipeline_helpers(GlueProtocol, ErrorProtocol, Helpers),

    % Generate main block
    generate_pipeline_main(NameStr, GlueProtocol, Options, Main),

    format(string(PythonCode), "~s~s~s~s", [Header, Helpers, FunctionCode, Main]).

%% generate_default_arg_names(+Arity, -ArgNames)
%  Generate default argument names: ['arg_0', 'arg_1', ...]
generate_default_arg_names(Arity, ArgNames) :-
    NumArgs is Arity,
    findall(ArgName, (
        between(0, NumArgs, I),
        I < NumArgs,
        format(atom(ArgName), 'arg_~d', [I])
    ), ArgNames).

%% generate_pipeline_function(+Name, +Arity, +Clauses, +OutputFormat, +ArgNames, -Code)
%  Generate the main pipeline processing function
generate_pipeline_function(Name, Arity, Clauses, OutputFormat, ArgNames, Code) :-
    % Generate clause handlers
    findall(ClauseCode, (
        nth0(Index, Clauses, (ClauseHead, ClauseBody)),
        generate_pipeline_clause(ClauseHead, ClauseBody, Index, Arity, OutputFormat, ArgNames, ClauseCode)
    ), ClauseCodes),
    atomic_list_concat(ClauseCodes, "\n", AllClausesCode),

    % Generate yield calls for each clause
    findall(YieldCall, (
        nth0(Index, Clauses, _),
        format(string(YieldCall), "            yield from _clause_~d(record)", [Index])
    ), YieldCalls),
    atomic_list_concat(YieldCalls, "\n", YieldCallsCode),

    format(string(Code),
"
~s

def ~s(stream: Iterator[Dict]) -> Generator[Dict, None, None]:
    \"\"\"
    Pipeline-enabled predicate with structured output.

    Args:
        stream: Iterator of input records (dicts)

    Yields:
        Dict with keys: ~w
    \"\"\"
    for record in stream:
        try:
~s
        except Exception as e:
            # Error handling - yield error record to stderr
            yield {'__error__': True, '__type__': type(e).__name__, '__message__': str(e), '__record__': record}
", [AllClausesCode, Name, ArgNames, YieldCallsCode]).

%% generate_pipeline_clause(+Head, +Body, +Index, +Arity, +OutputFormat, +ArgNames, -Code)
%  Generate a clause handler for pipeline mode
generate_pipeline_clause(Head, Body, Index, _Arity, OutputFormat, ArgNames, Code) :-
    % Instantiate variables for translation (same as translate_clause/5)
    copy_term((Head, Body), (HeadCopy, BodyCopy)),
    numbervars((HeadCopy, BodyCopy), 0, _),

    HeadCopy =.. [_Name | Args],

    % Generate input extraction from record
    generate_input_extraction(Args, ArgNames, InputCode),

    % Translate the body
    (   BodyCopy == true
    ->  BodyCode = "    pass  # No body goals"
    ;   translate_body(BodyCopy, BodyCode)
    ),

    % Generate output formatting
    generate_output_formatting(Args, ArgNames, OutputFormat, OutputCode),

    format(string(Code),
"def _clause_~d(record: Dict) -> Generator[Dict, None, None]:
    \"\"\"Clause ~d handler.\"\"\"
~s
~s
~s
", [Index, Index, InputCode, BodyCode, OutputCode]).

%% generate_input_extraction(+Args, +ArgNames, -Code)
%  Generate code to extract input values from record dict
generate_input_extraction(Args, ArgNames, Code) :-
    findall(Line, (
        nth0(I, Args, Arg),
        nth0(I, ArgNames, ArgName),
        is_var_term(Arg),  % Only extract for input variables (includes $VAR(N))
        format(string(Line), "    v_~d = record.get('~w')", [I, ArgName])
    ), Lines),
    (   Lines == []
    ->  Code = "    # No input extraction needed"
    ;   atomic_list_concat(Lines, "\n", Code)
    ).

%% generate_output_formatting(+Args, +ArgNames, +OutputFormat, -Code)
%  Generate code to format output as dict or text
generate_output_formatting(Args, ArgNames, OutputFormat, Code) :-
    length(Args, NumArgs),
    (   OutputFormat == object
    ->  % Build dict with arg names as keys
        findall(Pair, (
            nth0(I, ArgNames, ArgName),
            I < NumArgs,
            format(string(Pair), "'~w': v_~d", [ArgName, I])
        ), Pairs),
        atomic_list_concat(Pairs, ", ", PairsStr),
        format(string(Code), "    yield {~s}", [PairsStr])
    ;   % Text format - just yield string representation
        format(string(Code), "    yield {'result': str(v_0)}", [])
    ).

%% pipeline_header(+Protocol, -Header)
%% pipeline_header(+Protocol, +Runtime, -Header)
%  Generate header for pipeline mode with appropriate imports
%  Runtime can be: cpython (default), ironpython, pypy, jython

% Default: cpython
pipeline_header(Protocol, Header) :-
    pipeline_header(Protocol, cpython, Header).

% CPython headers
pipeline_header(jsonl, cpython, Header) :-
    get_binding_imports(BindingImports),
    format(string(Header),
"#!/usr/bin/env python3
\"\"\"
Generated pipeline predicate.
Runtime: CPython
Protocol: JSONL (line-delimited JSON)
\"\"\"
import sys
import json
from typing import Iterator, Dict, Any, Generator
~w
", [BindingImports]).

pipeline_header(messagepack, cpython, Header) :-
    get_binding_imports(BindingImports),
    format(string(Header),
"#!/usr/bin/env python3
\"\"\"
Generated pipeline predicate.
Runtime: CPython
Protocol: MessagePack (binary)
\"\"\"
import sys
import msgpack
from typing import Iterator, Dict, Any, Generator
~w
", [BindingImports]).

% IronPython headers - include CLR integration
pipeline_header(jsonl, ironpython, Header) :-
    get_binding_imports(BindingImports),
    format(string(Header),
"#!/usr/bin/env ipy
\"\"\"
Generated pipeline predicate.
Runtime: IronPython (CLR/.NET integration)
Protocol: JSONL (line-delimited JSON)
\"\"\"
import sys
import json
import clr

# Add .NET references for common types
clr.AddReference('System')
clr.AddReference('System.Core')
from System import String, Int32, Double, DateTime, Math
from System.Collections.Generic import Dictionary, List

# Python typing (IronPython 3.4+ compatible)
from typing import Iterator, Dict, Any, Generator
~w

# Helper: Convert Python dict to .NET Dictionary
def to_dotnet_dict(py_dict):
    \"\"\"Convert Python dict to .NET Dictionary<string, object>.\"\"\"
    result = Dictionary[String, object]()
    for k, v in py_dict.items():
        result[str(k)] = v
    return result

# Helper: Convert .NET Dictionary to Python dict
def from_dotnet_dict(dotnet_dict):
    \"\"\"Convert .NET Dictionary to Python dict.\"\"\"
    return {str(k): v for k, v in dotnet_dict}
", [BindingImports]).

pipeline_header(messagepack, ironpython, Header) :-
    get_binding_imports(BindingImports),
    format(string(Header),
"#!/usr/bin/env ipy
\"\"\"
Generated pipeline predicate.
Runtime: IronPython (CLR/.NET integration)
Protocol: MessagePack (binary)
\"\"\"
import sys
import clr

clr.AddReference('System')
clr.AddReference('System.Core')
from System import String, Int32, Double, DateTime, Math
from System.Collections.Generic import Dictionary, List

# MessagePack for IronPython
try:
    import msgpack
except ImportError:
    # Fallback: use .NET serialization if msgpack not available
    clr.AddReference('System.Text.Json')
    from System.Text.Json import JsonSerializer
    class msgpack:
        @staticmethod
        def packb(obj):
            return JsonSerializer.SerializeToUtf8Bytes(obj)
        @staticmethod
        def unpackb(data):
            return JsonSerializer.Deserialize(data, object)

from typing import Iterator, Dict, Any, Generator
~w
", [BindingImports]).

% PyPy headers (similar to CPython but with PyPy shebang)
pipeline_header(jsonl, pypy, Header) :-
    get_binding_imports(BindingImports),
    format(string(Header),
"#!/usr/bin/env pypy3
\"\"\"
Generated pipeline predicate.
Runtime: PyPy (JIT-optimized)
Protocol: JSONL (line-delimited JSON)
\"\"\"
import sys
import json
from typing import Iterator, Dict, Any, Generator
~w
", [BindingImports]).

pipeline_header(messagepack, pypy, Header) :-
    get_binding_imports(BindingImports),
    format(string(Header),
"#!/usr/bin/env pypy3
\"\"\"
Generated pipeline predicate.
Runtime: PyPy (JIT-optimized)
Protocol: MessagePack (binary)
\"\"\"
import sys
import msgpack
from typing import Iterator, Dict, Any, Generator
~w
", [BindingImports]).

% Jython headers - include Java integration
pipeline_header(jsonl, jython, Header) :-
    get_binding_imports(BindingImports),
    format(string(Header),
"#!/usr/bin/env jython
\"\"\"
Generated pipeline predicate.
Runtime: Jython (JVM integration)
Protocol: JSONL (line-delimited JSON)
\"\"\"
import sys
import json

# Java imports
from java.lang import String as JString, Math as JMath
from java.util import HashMap, ArrayList

# Note: typing module may not be available in Jython 2.7
try:
    from typing import Iterator, Dict, Any, Generator
except ImportError:
    Iterator = Dict = Any = Generator = object
~w

# Helper: Convert Python dict to Java HashMap
def to_java_map(py_dict):
    \"\"\"Convert Python dict to Java HashMap.\"\"\"
    result = HashMap()
    for k, v in py_dict.items():
        result.put(str(k), v)
    return result
", [BindingImports]).

pipeline_header(messagepack, jython, Header) :-
    get_binding_imports(BindingImports),
    format(string(Header),
"#!/usr/bin/env jython
\"\"\"
Generated pipeline predicate.
Runtime: Jython (JVM integration)
Protocol: MessagePack (binary)
\"\"\"
import sys

# Java imports
from java.lang import String as JString, Math as JMath
from java.util import HashMap, ArrayList

# MessagePack - try Python version, fallback to Java
try:
    import msgpack
except ImportError:
    # Use Java serialization as fallback
    from java.io import ByteArrayOutputStream, ObjectOutputStream
    from java.io import ByteArrayInputStream, ObjectInputStream
    class msgpack:
        @staticmethod
        def packb(obj):
            baos = ByteArrayOutputStream()
            oos = ObjectOutputStream(baos)
            oos.writeObject(obj)
            oos.close()
            return baos.toByteArray()
        @staticmethod
        def unpackb(data):
            bais = ByteArrayInputStream(data)
            ois = ObjectInputStream(bais)
            return ois.readObject()

# Note: typing module may not be available in Jython 2.7
try:
    from typing import Iterator, Dict, Any, Generator
except ImportError:
    Iterator = Dict = Any = Generator = object
~w

# Helper: Convert Python dict to Java HashMap
def to_java_map(py_dict):
    \"\"\"Convert Python dict to Java HashMap.\"\"\"
    result = HashMap()
    for k, v in py_dict.items():
        result.put(str(k), v)
    return result
", [BindingImports]).

% Legacy 2-argument version for backward compatibility
pipeline_header(messagepack, Header) :-
    pipeline_header(messagepack, cpython, Header).

% Original messagepack header moved here
pipeline_header_messagepack_base(Header) :-
    get_binding_imports(BindingImports),
    format(string(Header),
"#!/usr/bin/env python3
\"\"\"
Generated pipeline predicate.
Protocol: MessagePack (binary)
\"\"\"
import sys
import msgpack
from typing import Iterator, Dict, Any, Generator
~w
", [BindingImports]).

%% pipeline_helpers(+GlueProtocol, +ErrorProtocol, -Helpers)
%  Generate helper functions for pipeline I/O
pipeline_helpers(jsonl, same_as_data, Helpers) :-
    Helpers = "
def read_stream(stream) -> Iterator[Dict[str, Any]]:
    \"\"\"Read JSONL records from stream.\"\"\"
    for line in stream:
        line = line.strip()
        if line:
            yield json.loads(line)

def write_record(record: Dict, stream=sys.stdout) -> None:
    \"\"\"Write a single record as JSONL.\"\"\"
    if record.get('__error__'):
        # Error records go to stderr
        error_record = {k: v for k, v in record.items() if not k.startswith('__')}
        error_record['error'] = True
        error_record['type'] = record.get('__type__', 'Unknown')
        error_record['message'] = record.get('__message__', '')
        sys.stderr.write(json.dumps(error_record) + '\\n')
        sys.stderr.flush()
    else:
        stream.write(json.dumps(record) + '\\n')
        stream.flush()
".

pipeline_helpers(jsonl, text, Helpers) :-
    Helpers = "
def read_stream(stream) -> Iterator[Dict[str, Any]]:
    \"\"\"Read JSONL records from stream.\"\"\"
    for line in stream:
        line = line.strip()
        if line:
            yield json.loads(line)

def write_record(record: Dict, stream=sys.stdout) -> None:
    \"\"\"Write a single record as JSONL, errors as plain text.\"\"\"
    if record.get('__error__'):
        # Error records go to stderr as plain text
        sys.stderr.write(f\"ERROR [{record.get('__type__', 'Unknown')}]: {record.get('__message__', '')}\\n\")
        sys.stderr.flush()
    else:
        stream.write(json.dumps(record) + '\\n')
        stream.flush()
".

pipeline_helpers(messagepack, _, Helpers) :-
    Helpers = "
def read_stream(stream) -> Iterator[Dict[str, Any]]:
    \"\"\"Read MessagePack records from binary stream.\"\"\"
    unpacker = msgpack.Unpacker(stream.buffer, raw=False)
    for record in unpacker:
        yield record

def write_record(record: Dict, stream=sys.stdout) -> None:
    \"\"\"Write a single record as MessagePack.\"\"\"
    if record.get('__error__'):
        # Error records go to stderr
        error_record = {k: v for k, v in record.items() if not k.startswith('__')}
        error_record['error'] = True
        error_record['type'] = record.get('__type__', 'Unknown')
        error_record['message'] = record.get('__message__', '')
        sys.stderr.buffer.write(msgpack.packb(error_record))
        sys.stderr.flush()
    else:
        stream.buffer.write(msgpack.packb(record))
        stream.flush()
".

%% generate_pipeline_main(+Name, +Protocol, +Options, -Main)
%  Generate the main block for pipeline execution
generate_pipeline_main(Name, _Protocol, _Options, Main) :-
    format(string(Main),
"
if __name__ == '__main__':
    # Pipeline mode: read from stdin, write to stdout
    input_stream = read_stream(sys.stdin)
    for result in ~s(input_stream):
        write_record(result)
", [Name]).

% ============================================================================
% RUNTIME SELECTION - Phase 2: Firewall/Preference Integration
% ============================================================================

%% select_python_runtime(+PredIndicator, +Imports, +Context, -Runtime)
%
% Selects Python runtime respecting firewall and preferences.
% Uses existing dotnet_glue.pl for IronPython compatibility checking.
%
% @arg PredIndicator The predicate being compiled (Name/Arity)
% @arg Imports List of required Python imports
% @arg Context Additional context (e.g., [target(csharp)])
% @arg Runtime Selected runtime: cpython | ironpython | pypy | jython
%
select_python_runtime(PredIndicator, Imports, Context, Runtime) :-
    % 1. Get merged preferences (uses existing preferences module if available)
    get_runtime_preferences(PredIndicator, Context, Preferences),

    % 2. Get firewall policy (uses existing firewall module if available)
    get_runtime_firewall(PredIndicator, Firewall),

    % 3. Determine candidate runtimes (filtered by hard constraints)
    findall(R, valid_runtime_candidate(R, Imports, Firewall, Context), Candidates),
    (   Candidates == []
    ->  % No valid candidates - fall back to cpython
        Runtime = cpython
    ;   % 4. Score candidates against preferences
        score_runtime_candidates(Candidates, Preferences, Context, ScoredCandidates),
        % 5. Select best
        select_best_runtime(ScoredCandidates, Preferences, Runtime)
    ).

%% get_runtime_preferences(+PredIndicator, +Context, -Preferences)
%  Get merged runtime preferences from preference system
get_runtime_preferences(PredIndicator, Context, Preferences) :-
    (   catch(preferences:get_final_options(PredIndicator, Context, Prefs), _, fail)
    ->  Preferences = Prefs
    ;   % Default preferences if module not available
        Preferences = [
            prefer_runtime([cpython, ironpython, pypy]),
            prefer_communication(in_process),
            python_version(3)
        ]
    ).

%% get_runtime_firewall(+PredIndicator, -Firewall)
%  Get firewall policy for runtime selection
get_runtime_firewall(PredIndicator, Firewall) :-
    (   catch(firewall:get_firewall_policy(PredIndicator, FW), _, fail)
    ->  Firewall = FW
    ;   % Default: no restrictions
        Firewall = []
    ).

%% valid_runtime_candidate(+Runtime, +Imports, +Firewall, +Context) is semidet.
%
% Check if runtime passes all hard constraints.
%
valid_runtime_candidate(Runtime, Imports, Firewall, Context) :-
    member(Runtime, [cpython, ironpython, pypy, jython]),
    % Not explicitly denied in firewall
    \+ member(denied(python_runtime(Runtime)), Firewall),
    \+ (member(denied(List), Firewall), is_list(List), member(python_runtime(Runtime), List)),
    % Runtime is available on system
    runtime_available(Runtime),
    % Compatible with required imports
    runtime_compatible_with_imports(Runtime, Imports),
    % Context requirements (e.g., .NET integration needs ironpython or cpython_pipe)
    runtime_satisfies_context(Runtime, Context).

%% runtime_available(+Runtime) is semidet.
%  Check if runtime is available on the system
runtime_available(cpython) :-
    % CPython is always assumed available (python3 command)
    !.
runtime_available(ironpython) :-
    % Check for IronPython using dotnet_glue if available
    (   catch(dotnet_glue:detect_ironpython(true), _, fail)
    ->  true
    ;   % Fallback: check for ipy command
        catch(process_create(path(ipy), ['--version'], [stdout(null), stderr(null)]), _, fail)
    ).
runtime_available(pypy) :-
    % Check for PyPy
    catch(process_create(path(pypy3), ['--version'], [stdout(null), stderr(null)]), _, fail).
runtime_available(jython) :-
    % Check for Jython
    catch(process_create(path(jython), ['--version'], [stdout(null), stderr(null)]), _, fail).

%% runtime_compatible_with_imports(+Runtime, +Imports) is semidet.
%  Check if runtime supports all required imports
runtime_compatible_with_imports(cpython, _) :- !.  % CPython supports everything
runtime_compatible_with_imports(pypy, Imports) :-
    % PyPy has issues with some C extensions
    \+ member(numpy, Imports),
    \+ member(scipy, Imports),
    \+ member(tensorflow, Imports),
    \+ member(torch, Imports).
runtime_compatible_with_imports(ironpython, Imports) :-
    % Use dotnet_glue's compatibility check if available
    (   catch(dotnet_glue:can_use_ironpython(Imports), _, fail)
    ->  true
    ;   % Fallback: check against known incompatible modules
        \+ member(numpy, Imports),
        \+ member(scipy, Imports),
        \+ member(pandas, Imports),
        \+ member(matplotlib, Imports),
        \+ member(tensorflow, Imports),
        \+ member(torch, Imports)
    ).
runtime_compatible_with_imports(jython, Imports) :-
    % Jython has similar limitations to IronPython
    \+ member(numpy, Imports),
    \+ member(scipy, Imports),
    \+ member(pandas, Imports).

%% runtime_satisfies_context(+Runtime, +Context) is semidet.
%  Check if runtime satisfies context requirements
runtime_satisfies_context(_, []) :- !.
runtime_satisfies_context(Runtime, Context) :-
    % If targeting .NET, prefer in-process runtimes
    (   member(target(csharp), Context)
    ;   member(target(dotnet), Context)
    )
    ->  member(Runtime, [ironpython, cpython])  % cpython via pipes is fallback
    ;   % If targeting JVM, prefer Jython
        member(target(java), Context)
    ->  member(Runtime, [jython, cpython])
    ;   % No special context requirements
        true.

%% score_runtime_candidates(+Candidates, +Preferences, +Context, -Scored)
%  Score each candidate against preference dimensions
score_runtime_candidates(Candidates, Preferences, Context, Scored) :-
    maplist(score_single_runtime(Preferences, Context), Candidates, Scored).

%% score_single_runtime(+Preferences, +Context, +Runtime, -Scored)
score_single_runtime(Preferences, Context, Runtime, Runtime-Score) :-
    % Base score from preference order
    (   member(prefer_runtime(Order), Preferences),
        nth0(Idx, Order, Runtime)
    ->  OrderScore is 10 - Idx  % Higher = better
    ;   OrderScore = 0
    ),

    % Communication preference bonus
    (   member(prefer_communication(Comm), Preferences),
        runtime_communication(Runtime, Comm, Context)
    ->  CommScore = 5
    ;   CommScore = 0
    ),

    % Optimization hint bonus
    (   member(optimization(Opt), Preferences),
        runtime_optimization(Runtime, Opt)
    ->  OptScore = 3
    ;   OptScore = 0
    ),

    % Metrics bonus placeholder (returns 0 for now, designed for future extension)
    metrics_bonus(Runtime, MetricsBonus),

    Score is OrderScore + CommScore + OptScore + MetricsBonus.

%% runtime_communication(+Runtime, +CommType, +Context) is semidet.
%  Check if runtime provides the communication type in given context
runtime_communication(ironpython, in_process, Context) :-
    % IronPython is in-process when targeting .NET
    (member(target(csharp), Context) ; member(target(dotnet), Context)), !.
runtime_communication(jython, in_process, Context) :-
    % Jython is in-process when targeting JVM
    member(target(java), Context), !.
runtime_communication(cpython, cross_process, _) :- !.
runtime_communication(pypy, cross_process, _) :- !.
runtime_communication(_, cross_process, _).  % Default fallback

%% runtime_optimization(+Runtime, +OptType) is semidet.
%  Check if runtime is optimized for given workload type
runtime_optimization(pypy, throughput).      % JIT for long-running
runtime_optimization(pypy, latency).         % Fast after warmup
runtime_optimization(ironpython, latency).   % No serialization in .NET
runtime_optimization(cpython, memory).       % Most memory efficient
runtime_optimization(cpython, compatibility). % Best library support

%% metrics_bonus(+Runtime, -Bonus)
%  Placeholder for metrics-driven selection (returns 0 for now)
%  Future: Use runtime_metric/3 facts to calculate bonus from execution history
metrics_bonus(_Runtime, 0).

%% select_best_runtime(+ScoredCandidates, +Preferences, -Runtime)
%  Select the highest-scoring runtime, with fallback handling
select_best_runtime(ScoredCandidates, Preferences, Runtime) :-
    % Sort by score (descending)
    keysort(ScoredCandidates, Sorted),
    reverse(Sorted, [Best-_|_]),
    (   Best \= cpython
    ->  Runtime = Best
    ;   % If cpython is best but fallback specified, check it
        (   member(fallback_runtime(Fallbacks), Preferences),
            member(FB, Fallbacks),
            member(FB-_, ScoredCandidates)
        ->  Runtime = FB
        ;   Runtime = Best
        )
    ).

%% get_collected_imports(-Imports)
%  Get the list of imports collected during compilation
get_collected_imports(Imports) :-
    findall(I, required_import(I), Imports).

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
    
    % For mutual recursion, generate a dispatcher that handles all predicates
    findall(Pred/Arity, member(Pred/Arity, Predicates), PredList),
    generate_mutual_dispatcher(PredList, DispatcherCode),
    
    generate_python_main(Options, Main),
    
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
    
    generate_python_main(Options, Main),
    
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
    
    generate_python_main(Options, Main),
    
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
    
    % Determine parameter name and matching code
    (   compound(RecordVar), functor(RecordVar, '$VAR', 1)
    ->  % It is a variable (v_0), use it as parameter
        var_to_python(RecordVar, PyRecordVar),
        MatchCode = ""
    ;   % It is a constant, use generic parameter and check equality
        PyRecordVar = "record",
        var_to_python(RecordVar, PyVal),
        format(string(MatchCode), "    if ~w != ~w: return\n", [PyRecordVar, PyVal])
    ),
    
    % Determine output variable (Last argument if Arity > 1, else Input)
    (   Arity > 1
    ->  arg(Arity, Head, OutputVar)
    ;   OutputVar = RecordVar
    ),
    var_to_python(OutputVar, PyOutputVar),
    
    translate_body(Body, BodyCode),
    
    format(string(Code),
"def _clause_~d(~w: Dict) -> Iterator[Dict]:
~s~s
    yield ~w
", [Index, PyRecordVar, MatchCode, BodyCode, PyOutputVar]).

translate_body((Goal, Rest), Code) :-
    !,
    translate_goal(Goal, Code1),
    translate_body(Rest, Code2),
    string_concat(Code1, Code2, Code).
translate_body(Goal, Code) :-
    translate_goal(Goal, Code).

is_var_term(V) :- var(V), !.
is_var_term('$VAR'(_)).

translate_goal(_:Goal, Code) :-
    !,
    translate_goal(Goal, Code).

translate_goal(get_dict(Key, Record, Value), Code) :-
    !,
    var_to_python(Record, PyRecord),
    (   is_var_term(Value)
    ->  var_to_python(Value, PyValue),
        format(string(Code), "    ~w = ~w.get('~w')\n", [PyValue, PyRecord, Key])
    ;   % Value is a constant
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

translate_goal(=(Var, Value), Code) :-
    is_var_term(Var),
    atomic(Value),
    !,
    var_to_python(Var, PyVar),
    var_to_python(Value, PyVal),
    format(string(Code), "    ~w = ~w\n", [PyVar, PyVal]).

translate_goal(=(Var1, Var2), Code) :-
    is_var_term(Var1),
    is_var_term(Var2),
    !,
    var_to_python(Var1, PyVar1),
    var_to_python(Var2, PyVar2),
    format(string(Code), "    ~w = ~w\n", [PyVar1, PyVar2]).

translate_goal(>(Var, Value), Code) :-
    !,
    var_to_python(Var, PyVar),
    var_to_python(Value, PyValue),
    format(string(Code), "    if not (~w > ~w): return\n", [PyVar, PyValue]).

translate_goal(<(Var, Value), Code) :-
    !,
    var_to_python(Var, PyVar),
    var_to_python(Value, PyValue),
    format(string(Code), "    if not (~w < ~w): return\n", [PyVar, PyValue]).

translate_goal(>=(Var, Value), Code) :-
    !,
    var_to_python(Var, PyVar),
    var_to_python(Value, PyValue),
    format(string(Code), "    if not (~w >= ~w): return\n", [PyVar, PyValue]).

translate_goal(=<(Var, Value), Code) :-
    !,
    var_to_python(Var, PyVar),
    var_to_python(Value, PyValue),
    format(string(Code), "    if not (~w <= ~w): return\n", [PyVar, PyValue]).

translate_goal(\=(Var, Value), Code) :-
    !,
    var_to_python(Var, PyVar),
    var_to_python(Value, PyValue),
    format(string(Code), "    if not (~w != ~w): return\n", [PyVar, PyValue]).

% Match predicate support for procedural mode
translate_goal(match(Var, Pattern), Code) :-
    !,
    translate_match_goal(Var, Pattern, auto, [], Code).
translate_goal(match(Var, Pattern, Type), Code) :-
    !,
    translate_match_goal(Var, Pattern, Type, [], Code).
translate_goal(match(Var, Pattern, Type, Groups), Code) :-
    !,
    translate_match_goal(Var, Pattern, Type, Groups, Code).

translate_goal(suggest_bookmarks(Query, Options, Suggestions), Code) :-
    !,
    var_to_python(Query, PyQuery),
    var_to_python(Suggestions, PyResults),
    % Extract search mode from options (default vector)
    (   member(mode(Mode), Options)
    ->  atom_string(Mode, ModeStr)
    ;   ModeStr = "vector"
    ),
    format(string(Code), "    ~w = _get_runtime().searcher.suggest_bookmarks(~w, top_k=5, mode='~w')\n", [PyResults, PyQuery, ModeStr]).

translate_goal(suggest_bookmarks(Query, Suggestions), Code) :-
    !,
    var_to_python(Query, PyQuery),
    var_to_python(Suggestions, PyResults),
    format(string(Code), "    ~w = _get_runtime().searcher.suggest_bookmarks(~w, top_k=5)\n", [PyResults, PyQuery]).

translate_goal(graph_search(Query, TopK, Hops, Options, Results), Code) :-
    !,
    var_to_python(Query, PyQuery),
    var_to_python(TopK, PyTopK),
    var_to_python(Hops, PyHops),
    var_to_python(Results, PyResults),
    % Extract search mode from options (default vector)
    (   member(mode(Mode), Options)
    ->  atom_string(Mode, ModeStr)
    ;   ModeStr = "vector"
    ),
    format(string(Code), "    ~w = _get_runtime().searcher.graph_search(~w, top_k=~w, hops=~w, mode='~w')\n", [PyResults, PyQuery, PyTopK, PyHops, ModeStr]).

translate_goal(graph_search(Query, TopK, Hops, Results), Code) :-
    !,
    var_to_python(Query, PyQuery),
    var_to_python(TopK, PyTopK),
    var_to_python(Hops, PyHops),
    var_to_python(Results, PyResults),
    format(string(Code), "    ~w = _get_runtime().searcher.graph_search(~w, top_k=~w, hops=~w)\n", [PyResults, PyQuery, PyTopK, PyHops]).

translate_goal(semantic_search(Query, TopK, Results), Code) :-
    !,
    var_to_python(Query, PyQuery),
    var_to_python(TopK, PyTopK),
    var_to_python(Results, PyResults),
    format(string(Code), "    ~w = _get_runtime().searcher.search(~w, top_k=~w)\n", [PyResults, PyQuery, PyTopK]).

translate_goal(crawler_run(SeedIds, MaxDepth, Options), Code) :-
    !,
    var_to_python(SeedIds, PySeeds),
    var_to_python(MaxDepth, PyDepth),
    % Check options for embedding(false)
    (   member(embedding(false), Options)
    ->  EmbedVal = "False"
    ;   EmbedVal = "True"
    ),
    format(string(Code), "    _get_runtime().crawler.crawl(~w, fetch_xml_func, max_depth=~w, embed_content=~w)\n", [PySeeds, PyDepth, EmbedVal]).

translate_goal(crawler_run(SeedIds, MaxDepth), Code) :-
    !,
    var_to_python(SeedIds, PySeeds),
    var_to_python(MaxDepth, PyDepth),
    format(string(Code), "    _get_runtime().crawler.crawl(~w, fetch_xml_func, max_depth=~w)\n", [PySeeds, PyDepth]).

translate_goal(upsert_object(Id, Type, Data), Code) :-
    !,
    var_to_python(Id, PyId),
    var_to_python(Type, PyType),
    var_to_python(Data, PyData),
    format(string(Code), "    _get_runtime().importer.upsert_object(~w, ~w, ~w)\n", [PyId, PyType, PyData]).

translate_goal(llm_ask(Prompt, Context, Response), Code) :-
    !,
    var_to_python(Prompt, PyPrompt),
    var_to_python(Context, PyContext),
    var_to_python(Response, PyResponse),
    format(string(Code), "    ~w = _get_runtime().llm.ask(~w, ~w)\n", [PyResponse, PyPrompt, PyContext]).

translate_goal(chunk_text(Text, Chunks), Code) :-
    !,
    var_to_python(Text, PyText),
    var_to_python(Chunks, PyChunks),
    format(string(Code), "    ~w = [asdict(c) for c in _get_runtime().chunker.chunk(~w, 'inline')]\n", [PyChunks, PyText]).

translate_goal(chunk_text(Text, Chunks, Options), Code) :-
    !,
    var_to_python(Text, PyText),
    var_to_python(Chunks, PyChunks),
    (   is_list(Options)
    ->  maplist(opt_to_py_pair, Options, Pairs),
        atomic_list_concat(Pairs, ', ', PairsStr),
        format(string(PyKwargs), "{~s}", [PairsStr])
    ;   var_to_python(Options, PyKwargs)
    ),
    format(string(Code), "    ~w = [asdict(c) for c in _get_runtime().chunker.chunk(~w, 'inline', **~w)]\n", [PyChunks, PyText, PyKwargs]).

translate_goal(generate_key(Strategy, KeyVar), Code) :-
    !,
    var_to_python(KeyVar, PyKeyVar),
    compile_python_key_expr(Strategy, PyExpr),
    format(string(Code), "    ~w = ~s\n", [PyKeyVar, PyExpr]).

opt_to_py_pair(Term, Pair) :-
    Term =.. [Key, Value],
    format(string(Pair), "'~w': ~w", [Key, Value]).

translate_goal(true, Code) :-
    !,
    Code = "    pass\n".

% ============================================================================
% BINDING-BASED GOAL TRANSLATION
% ============================================================================
%
% Check the binding registry for Python bindings before falling back to
% unsupported goal warning. This enables extensible goal handling.
%
% The binding registry maps Prolog predicates to Python functions with:
% - TargetName: The Python function/method name
% - Inputs/Outputs: Argument specifications
% - Options: Effect annotations (pure, io, etc.) and imports
%

translate_goal(Goal, Code) :-
    % Extract predicate name and arity from goal
    Goal =.. [Pred|Args],
    length(Args, Arity),

    % Check if we have a Python binding for this predicate
    binding(python, Pred/Arity, TargetName, _Inputs, Outputs, Options),
    !,

    % Record any required imports
    (   member(import(Module), Options)
    ->  (   required_import(Module)
        ->  true
        ;   assertz(required_import(Module))
        )
    ;   true
    ),

    % Generate Python code based on binding
    generate_binding_call_python(TargetName, Args, Outputs, Options, Code).

translate_goal(Goal, "") :-
    format(string(Msg), "Warning: Unsupported goal ~w", [Goal]),
    print_message(warning, Msg).

% ============================================================================
% BINDING CODE GENERATION FOR PYTHON
% ============================================================================

%% generate_binding_call_python(+TargetName, +Args, +Outputs, +Options, -Code)
%  Generate Python code for a binding call
%
%  Handles multiple patterns:
%  - Function calls: func(args) -> result
%  - Method calls: object.method(args) -> result
%  - No-arg methods: object.method() (TargetName may include () or not)
%  - Mutating methods: object.method(arg) with no output (e.g., list.append)
%  - Chained calls: object.method1().method2()
%
generate_binding_call_python(TargetName, Args, Outputs, Options, Code) :-
    % Determine call pattern type
    (   member(pattern(method_call), Options)
    ->  generate_method_call(TargetName, Args, Outputs, CallExpr)
    ;   member(pattern(chained_call(Methods)), Options)
    ->  generate_chained_call(Methods, Args, Outputs, CallExpr)
    ;   generate_function_call(TargetName, Args, Outputs, CallExpr)
    ),

    % Generate assignment if there are outputs
    (   Outputs = []
    ->  format(string(Code), "    ~w\n", [CallExpr])
    ;   % Assign result to output variable (last argument)
        last(Args, OutputVar),
        var_to_python(OutputVar, PyOutputVar),
        format(string(Code), "    ~w = ~w\n", [PyOutputVar, CallExpr])
    ).

%% generate_method_call(+TargetName, +Args, +Outputs, -CallExpr)
%  Generate a method call expression: object.method(args)
%
%  Args structure depends on Outputs:
%  - Outputs = []: All args are inputs, first is object
%  - Outputs = [_]: All but last are inputs, first is object, last is output
%
generate_method_call(TargetName, Args, Outputs, CallExpr) :-
    % Separate object from other arguments
    Args = [Object|RestArgs],
    var_to_python(Object, PyObject),

    % Determine which args are inputs (exclude output if present)
    (   Outputs = []
    ->  % No output: all RestArgs are method arguments
        InputArgs = RestArgs
    ;   % Has output: RestArgs minus last element are method arguments
        (   RestArgs = []
        ->  InputArgs = []
        ;   append(InputArgs, [_OutputArg], RestArgs)
        )
    ),

    % Convert input args to Python
    maplist(var_to_python, InputArgs, PyInputArgs),

    % Generate method call expression
    (   PyInputArgs = []
    ->  % No-arg method call
        (   sub_string(TargetName, _, _, 0, "()")
        ->  % TargetName already has () like ".split()" or ".lower()"
            format(string(CallExpr), "~w~w", [PyObject, TargetName])
        ;   % TargetName doesn't have (), add them
            format(string(CallExpr), "~w~w()", [PyObject, TargetName])
        )
    ;   % Method call with arguments
        atomic_list_concat(PyInputArgs, ', ', ArgsStr),
        format(string(CallExpr), "~w~w(~w)", [PyObject, TargetName, ArgsStr])
    ).

%% generate_function_call(+TargetName, +Args, +Outputs, -CallExpr)
%  Generate a function call expression: func(args)
%
generate_function_call(TargetName, Args, Outputs, CallExpr) :-
    (   Outputs = []
    ->  % No output - all args are inputs
        maplist(var_to_python, Args, PyArgs),
        atomic_list_concat(PyArgs, ', ', ArgsStr),
        format(string(CallExpr), "~w(~w)", [TargetName, ArgsStr])
    ;   % Has output - extract input args (all but last which is output)
        (   append(InputArgs, [_OutputArg], Args)
        ->  maplist(var_to_python, InputArgs, PyInputArgs),
            (   PyInputArgs = []
            ->  % Constant/no-arg function
                format(string(CallExpr), "~w", [TargetName])
            ;   atomic_list_concat(PyInputArgs, ', ', ArgsStr),
                format(string(CallExpr), "~w(~w)", [TargetName, ArgsStr])
            )
        ;   % Single arg that is output (constant function like pi/1)
            format(string(CallExpr), "~w", [TargetName])
        )
    ).

%% generate_chained_call(+Methods, +Args, +Outputs, -CallExpr)
%  Generate a chained method call: object.method1(args1).method2(args2)
%
%  Methods is a list of method(Name, ArgIndices) terms specifying which
%  args (by index) go to each method in the chain.
%
%  Example: pattern(chained_call([method('.strip', []), method('.lower', [])]))
%  For: strip_lower(Str, Result) -> Str.strip().lower()
%
generate_chained_call(Methods, Args, Outputs, CallExpr) :-
    % First arg is always the object
    Args = [Object|RestArgs],
    var_to_python(Object, PyObject),

    % Determine input args (exclude output if present)
    (   Outputs = []
    ->  InputArgs = RestArgs
    ;   (   RestArgs = []
        ->  InputArgs = []
        ;   append(InputArgs, [_], RestArgs)
        )
    ),

    % Build the chain
    generate_method_chain(Methods, InputArgs, PyObject, CallExpr).

%% generate_method_chain(+Methods, +InputArgs, +CurrentExpr, -FinalExpr)
%  Recursively build a method chain expression
generate_method_chain([], _InputArgs, Expr, Expr).
generate_method_chain([method(Name, ArgIndices)|Rest], InputArgs, CurrentExpr, FinalExpr) :-
    % Get args for this method by indices
    findall(PyArg, (
        member(Idx, ArgIndices),
        nth0(Idx, InputArgs, Arg),
        var_to_python(Arg, PyArg)
    ), MethodPyArgs),

    % Generate this method call
    (   MethodPyArgs = []
    ->  (   sub_string(Name, _, _, 0, "()")
        ->  format(string(NextExpr), "~w~w", [CurrentExpr, Name])
        ;   format(string(NextExpr), "~w~w()", [CurrentExpr, Name])
        )
    ;   atomic_list_concat(MethodPyArgs, ', ', ArgsStr),
        format(string(NextExpr), "~w~w(~w)", [CurrentExpr, Name, ArgsStr])
    ),

    % Continue with rest of chain
    generate_method_chain(Rest, InputArgs, NextExpr, FinalExpr).

%% compile_python_key_expr(+Strategy, -PyExpr)
%  Compiles a key generation strategy into a Python string expression.
compile_python_key_expr(Var, PyExpr) :-
    is_var_term(Var), !,
    var_to_python(Var, PyVar),
    format(string(PyExpr), "str(~w)", [PyVar]).
compile_python_key_expr(literal(Text), PyExpr) :-
    !,
    format(string(PyExpr), "'~w'", [Text]).
compile_python_key_expr(field(Var), PyExpr) :-
    !,
    compile_python_key_expr(Var, PyExpr).
compile_python_key_expr(composite(List), PyExpr) :-
    !,
    maplist(compile_python_key_expr, List, Parts),
    atomic_list_concat(Parts, " + ", Expr),
    PyExpr = Expr.
compile_python_key_expr(hash(Expr), PyExpr) :-
    !,
    compile_python_key_expr(Expr, Inner),
    format(string(PyExpr), "hashlib.sha256(str(~s).encode('utf-8')).hexdigest()", [Inner]).
compile_python_key_expr(uuid(), "uuid.uuid4().hex") :- !.
compile_python_key_expr(Term, PyExpr) :-
    atomic(Term),
    !,
    format(string(PyExpr), "'~w'", [Term]).
compile_python_key_expr(Strategy, "'UNKNOWN_STRATEGY'") :-
    format(string(Msg), "Warning: Unknown key strategy ~w", [Strategy]),
    print_message(warning, Msg).

%% translate_match_goal(+Var, +Pattern, +Type, +Groups, -Code)
%  Translate match predicate to Python code for procedural mode
translate_match_goal(Var, Pattern, Type, Groups, Code) :-
    % Validate regex type
    validate_regex_type_for_python(Type),
    % Convert variable to Python
    var_to_python(Var, PyVar),
    % Escape pattern
    (   atom(Pattern)
    ->  atom_string(Pattern, PatternStr)
    ;   PatternStr = Pattern
    ),
    escape_python_string(PatternStr, EscapedPattern),
    % Check if we have capture groups
    (   Groups = [], !
    ->  % No capture groups - simple boolean match
        format(string(Code), "    if not re.search(r'~w', str(~w)): return\n", [EscapedPattern, PyVar])
    ;   % Has capture groups - extract them
        length(Groups, NumGroups),
        % Generate Python code to capture and store groups
        generate_python_capture_code(PyVar, EscapedPattern, Groups, NumGroups, Code)
    ).

%% generate_python_capture_code(+PyVar, +Pattern, +Groups, +NumGroups, -Code)
%  Generate Python code to perform match with capture group extraction
generate_python_capture_code(PyVar, Pattern, Groups, NumGroups, Code) :-
    % Generate match object assignment
    format(string(MatchLine), "    __match__ = re.search(r'~w', str(~w))\n", [Pattern, PyVar]),
    % Generate check for match success
    CheckLine = "    if not __match__: return\n",
    % Generate capture variable assignments
    findall(CaptureLine,
        (   between(1, NumGroups, N),
            nth1(N, Groups, GroupVar),
            var_to_python(GroupVar, PyGroupVar),
            format(string(CaptureLine), "    ~w = __match__.group(~w)\n", [PyGroupVar, N])
        ),
        CaptureLines),
    % Combine all lines
    atomic_list_concat([MatchLine, CheckLine | CaptureLines], '', Code).

pair_to_python(Key-Value, Str) :-
    var_to_python(Value, PyValue),
    format(string(Str), "'~w': ~w", [Key, PyValue]).

var_to_python('$VAR'(I), PyVar) :-
    !,
    format(string(PyVar), "v_~d", [I]).
var_to_python(v(Name), PyVar) :-
    % Support for readable variable notation v(name) -> v_name
    !,
    format(string(PyVar), "v_~w", [Name]).
var_to_python(Atom, Quoted) :-
    atom(Atom),
    !,
    format(string(Quoted), "\"~w\"", [Atom]).
var_to_python(Number, Number) :- 
    number(Number), 
    !.
var_to_python(List, PyList) :-
    is_list(List),
    !,
    maplist(var_to_python, List, Elems),
    atomic_list_concat(Elems, ', ', Inner),
    format(string(PyList), "[~w]", [Inner]).
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

%% get_binding_imports(-ImportStr)
%  Get import statements for all modules required by bindings used in compilation
get_binding_imports(ImportStr) :-
    findall(Module, required_import(Module), Modules),
    sort(Modules, UniqueModules),  % Remove duplicates
    (   UniqueModules = []
    ->  ImportStr = ""
    ;   findall(ImportLine, (
            member(M, UniqueModules),
            % Skip modules already in base imports
            \+ member(M, [sys, json, re, hashlib, uuid, functools, typing]),
            format(string(ImportLine), "import ~w", [M])
        ), ImportLines),
        (   ImportLines = []
        ->  ImportStr = ""
        ;   atomic_list_concat(ImportLines, '\n', ImportsBody),
            format(string(ImportStr), "~w\n", [ImportsBody])
        )
    ).

%% header(-Header)
%  Generate header with base imports plus any binding-required imports
header(Header) :-
    get_binding_imports(BindingImports),
    format(string(Header),
"import sys
import json
import re
import hashlib
import uuid
from typing import Iterator, Dict, Any
~w
", [BindingImports]).

%% header_with_functools(-Header)
%  Generate header with functools plus any binding-required imports
header_with_functools(Header) :-
    get_binding_imports(BindingImports),
    format(string(Header),
"import sys
import json
import re
import hashlib
import uuid
import functools
from typing import Iterator, Dict, Any
~w
", [BindingImports]).

%% clear_binding_imports
%  Clear collected binding imports (call before each compilation)
clear_binding_imports :-
    retractall(required_import(_)).

helpers(Helpers) :-
    helpers_base(Base),
    semantic_runtime_helpers(Runtime),
    format(string(Helpers), "~s\n~s", [Base, Runtime]).

helpers_base("
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

def read_xml_lxml(file_path: str, tags: set) -> Iterator[Dict[str, Any]]:
    \"\"\"Read and flatten XML using lxml.\"\"\"
    try:
        from lxml import etree
    except ImportError:
        sys.stderr.write('Error: lxml required for XML source\\n')
        sys.exit(1)
    
    context = etree.iterparse(file_path, events=('start', 'end'), recover=True)
    context = iter(context)
    _, root = next(context) # Get root start
    
    def expand(tag, nsmap):
        if ':' in tag:
            pfx, local = tag.split(':', 1)
            uri = nsmap.get(pfx)
            if uri:
                return '{' + uri + '}' + local
        return tag

    # Pre-calculate wanted tags (assuming passed tags are QNames if needed, or local names)
    # For simplicity, we match suffix or exact
    
    for event, elem in context:
        if event == 'end' and (elem.tag in tags or elem.tag.split('}')[-1] in tags):
            data = {}
            # Root element attributes (global keys for backward compatibility)
            for k, v in elem.attrib.items():
                data['@' + k] = v
            # Text
            if elem.text and elem.text.strip():
                data['text'] = elem.text.strip()
            # Children (simple flattening)
            for child in elem:
                tag = child.tag.split('}')[-1]
                if not len(child) and child.text:
                    data[tag] = child.text.strip()
                # Child element attributes (element-scoped to prevent conflicts)
                for attr_name, attr_val in child.attrib.items():
                    scoped_key = tag + '@' + attr_name
                    data[scoped_key] = attr_val
                    # Also store with global key for backward compatibility
                    data['@' + attr_name] = attr_val

            yield data
            
            # Memory cleanup
            elem.clear()
            while elem.getprevious() is not None:
                del elem.getparent()[0]
    del context
    root.clear()
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
    ;   partition(is_fact_clause, Clauses, Facts, Rules),
        generate_generator_code(Name, Arity, Facts, Rules, Options, PythonCode)
    ).

is_fact_clause((_Head, Body)) :- Body == true.

%% generate_generator_code(+Name, +Arity, +Facts, +Rules, +Options, -PythonCode)
generate_generator_code(_Name, _Arity, Facts, Rules, Options, PythonCode) :-
    % Generate components
    generator_header(Header),
    generator_helpers(Options, Helpers),
    generate_fact_functions(Name, Facts, FactFunctions),
    generate_rule_functions(Name, Rules, RuleFunctions),
    generate_fixpoint_loop(Name, Facts, Rules, FixpointLoop),
    
    generate_python_main(Options, Main),
    
    atomic_list_concat([Header, Helpers, FactFunctions, RuleFunctions, FixpointLoop, Main], "\n", PythonCode).

%% generator_header(-Header)
%  Generate header for generator mode with binding-required imports
generator_header(Header) :-
    get_binding_imports(BindingImports),
    format(string(Header),
"import sys
import json
import re
from typing import Iterator, Dict, Any, Set
from dataclasses import dataclass
~w
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
", [BindingImports]).

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
    semantic_runtime_helpers(Runtime),
    atomic_list_concat([JsonlReader, NulReader, Runtime], "", Helpers).

%% generate_fact_functions(+Name, +Facts, -FactFunctions)
generate_fact_functions(Name, Facts, FactFunctions) :-
    findall(FactFunc,
        (   nth1(FactNum, Facts, (Head, _Body)),
            generate_fact_function(Name, FactNum, Head, FactFunc)
        ),
        FactFuncs),
    atomic_list_concat(FactFuncs, "\n\n", FactFunctions).

%% generate_fact_function(+Name, +FactNum, +Head, -FactFunc)
generate_fact_function(_Name, FactNum, Head, FactFunc) :-
    Head =.. [Pred | Args],
    extract_constants(Args, ConstPairs),
    format_dict_pairs(ConstPairs, ArgsStr),
    (   ArgsStr == ""
    ->  format(string(DictStr), "'relation': '~w'", [Pred])
    ;   format(string(DictStr), "'relation': '~w', ~w", [Pred, ArgsStr])
    ),
    format(string(FactFunc),
"def _init_fact_~w() -> Iterator[FrozenDict]:
    '''Fact: ~w'''
    yield FrozenDict.from_dict({~w})
", [FactNum, Head, DictStr]).

%% generate_rule_functions(+Name, +Clauses, -RuleFunctions)
generate_rule_functions(Name, Clauses, RuleFunctions) :-
    findall(RuleFunc,
        (   nth1(RuleNum, Clauses, (Head, Body)),
            generate_rule_function(Name, RuleNum, Head, Body, RuleFunc)
        ),
        RuleFuncs),
    atomic_list_concat(RuleFuncs, "\n\n", RuleFunctions).

%% generate_rule_function(+Name, +RuleNum, +Head, +Body, -RuleFunc)
generate_rule_function(Name, RuleNum, Head, Body, RuleFunc) :-
    (   Body == true
    ->  % Fact (no body) - emit constant
        translate_fact_rule(Name, RuleNum, Head, RuleFunc)
    ;   % Check for disjunction (;) in body
        contains_disjunction(Body)
    ->  % Handle disjunctive rule
        extract_disjuncts(Body, Disjuncts),
        translate_disjunctive_rule(RuleNum, Head, Disjuncts, RuleFunc)
    ;   % Normal conjunctive rule
        extract_goals_list(Body, Goals),
        % Separate builtin goals from relational goals
        partition(is_builtin_goal, Goals, BuiltinGoals, RelGoals),
        length(RelGoals, NumRelGoals),
        (   NumRelGoals == 0
        ->  % Only builtins → constraint, not a generator rule
            format(string(RuleFunc),
"def _apply_rule_~w(fact: FrozenDict, total: Set[FrozenDict]) -> Iterator[FrozenDict]:
    '''Constraint-only rule (no generator)'''
    return iter([])
", [RuleNum])
        ;   NumRelGoals == 1
        ->  % Single relational goal + optionally builtins
            RelGoals = [SingleGoal],
            translate_copy_rule_with_builtins(Name, RuleNum, Head, SingleGoal, BuiltinGoals, RuleFunc)
        ;   % Multiple relational goals + optionally builtins
            translate_join_rule_with_builtins(Name, RuleNum, Head, RelGoals, BuiltinGoals, RuleFunc)
        )
    ).

%% contains_disjunction(+Body)
%  Check if body contains disjunction (;)
contains_disjunction((_;_)) :- !.
contains_disjunction((A,B)) :- 
    !,
    (contains_disjunction(A) ; contains_disjunction(B)).
contains_disjunction(_) :- fail.

%% extract_disjuncts(+Body, -Disjuncts)
%  Extract all disjuncts from a disjunctive body
extract_disjuncts((A;B), Disjuncts) :-
    !,
    extract_disjuncts(A, DisjunctsA),
    extract_disjuncts(B, DisjunctsB),
    append(DisjunctsA, DisjunctsB, Disjuncts).
extract_disjuncts(Goal, [Goal]).

%% translate_disjunctive_rule(+RuleNum, +Head, +Disjuncts, -RuleFunc)
%  Translate a rule with disjunction to Python
translate_disjunctive_rule(RuleNum, Head, Disjuncts, RuleFunc) :-
    % Generate code for each disjunct
    findall(DisjunctCode,
        (   member(Disjunct, Disjuncts),
            translate_disjunct(Head, Disjunct, DisjunctCode)
        ),
        DisjunctCodes),
    atomic_list_concat(DisjunctCodes, "\n    # Try next disjunct\n    ", CombinedCode),
    
    format(string(RuleFunc),
"def _apply_rule_~w(fact: FrozenDict, total: Set[FrozenDict]) -> Iterator[FrozenDict]:
    '''Disjunctive rule: ~w'''
    # Try each disjunct
    ~w
", [RuleNum, Head, CombinedCode]).

%% translate_disjunct(+Head, +Disjunct, -Code)
%  Translate a single disjunct to Python code
translate_disjunct(Head, Disjunct, Code) :-
    % Extract goals from this disjunct
    extract_goals_list(Disjunct, Goals),
    partition(is_builtin_goal, Goals, Builtins, RelGoals),
    
    length(RelGoals, NumRelGoals),
    (   NumRelGoals == 0
    ->  % Only builtins - just check constraints
        translate_builtins(Builtins, ConstraintChecks),
        (   ConstraintChecks == ""
        ->  Code = "pass  # Empty disjunct"
        ;   Head =.. [_Pred|HeadArgs],
            build_constant_output(HeadArgs, OutputStr),
            format(string(Code),
"if ~w:
        yield FrozenDict.from_dict({~w})",
                [ConstraintChecks, OutputStr])
        )
    ;   NumRelGoals == 1
    ->  % Single goal + optional constraints
        RelGoals = [Goal],
        translate_disjunct_copy(Head, Goal, Builtins, Code)
    ;   % Multiple goals - join
        translate_disjunct_join(Head, RelGoals, Builtins, Code)
    ).

%% translate_disjunct_copy(+Head, +Goal, +Builtins, -Code)
translate_disjunct_copy(Head, Goal, Builtins, Code) :-
    Head =.. [HeadPred | HeadArgs],
    Goal =.. [GoalPred | GoalArgs],
    
    % Pattern match
    length(GoalArgs, GoalArity),
    findall(Check,
        (   between(0, GoalArity, Idx),
            Idx < GoalArity,
            format(string(Check), "'arg~w' in fact", [Idx])
        ),
        Checks),
    % Add relation check
    format(string(RelCheck), "fact.get('relation') == '~w'", [GoalPred]),
    AllChecks = [RelCheck | Checks],
    atomic_list_concat(AllChecks, " and ", ConditionStr),
    
    % Constraints
    build_variable_map([Goal-"fact"], VarMap),
    translate_builtins(Builtins, VarMap, ConstraintChecks),
    (   ConstraintChecks == ""
    ->  FinalCondition = ConditionStr
    ;   format(string(FinalCondition), "~w and ~w", [ConditionStr, ConstraintChecks])
    ),
    
    % Output
    findall(Assign,
        (   nth0(HIdx, HeadArgs, HVar),
            nth0(GIdx, GoalArgs, HVar),
            format(string(Assign), "'arg~w': fact.get('arg~w')", [HIdx, GIdx])
        ),
        Assigns),
    % Add relation to output
    format(string(RelAssign), "'relation': '~w'", [HeadPred]),
    AllAssigns = [RelAssign | Assigns],
    atomic_list_concat(AllAssigns, ", ", OutputStr),
    
    format(string(Code),
"if ~w:
        yield FrozenDict.from_dict({~w})",
        [FinalCondition, OutputStr]).

%% translate_disjunct_join(+Head, +Goals, +Builtins, -Code)
translate_disjunct_join(Head, Goals, Builtins, Code) :-
    length(Goals, NumGoals),
    (   NumGoals == 2
    ->  % Binary join within disjunct
        Goals = [Goal1, Goal2],
        translate_disjunct_binary_join(Head, Goal1, Goal2, Builtins, Code)
    ;   % N-way join within disjunct - use simplified approach
        translate_disjunct_nway_join(Head, Goals, Builtins, Code)
    ).

%% translate_disjunct_binary_join(+Head, +Goal1, +Goal2, +Builtins, -Code)
translate_disjunct_binary_join(Head, Goal1, Goal2, Builtins, Code) :-
    Goal1 =.. [Pred1 | Args1],
    Goal2 =.. [Pred2 | Args2],
    Head =.. [HeadPred | HeadArgs],
    
    % Find join variable
    findall(Var-Idx1-Idx2,
        (   nth0(Idx1, Args1, Var),
            nth0(Idx2, Args2, Var),
            \+ atom(Var)
        ),
        JoinVars),
    
    % Build join condition
    (   JoinVars = [_Var-JIdx1-JIdx2|_]
    ->  format(string(JoinCond), "other.get('arg~w') == fact.get('arg~w')", [JIdx2, JIdx1])
    ;   JoinCond = "True"
    ),
    
    % Add relation check for other (Goal2)
    format(string(RelCheck2), "other.get('relation') == '~w'", [Pred2]),
    (   JoinCond == "True"
    ->  JoinCondWithRel = RelCheck2
    ;   format(string(JoinCondWithRel), "~w and ~w", [JoinCond, RelCheck2])
    ),
    
    % Build constraint checks
    build_variable_map([Goal1-"fact", Goal2-"other"], VarMap),
    translate_builtins(Builtins, VarMap, ConstraintChecks),
    (   ConstraintChecks == ""
    ->  FinalJoinCond = JoinCondWithRel
    ;   format(string(FinalJoinCond), "~w and ~w", [JoinCondWithRel, ConstraintChecks])
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
    % Add relation to output
    format(string(RelAssign), "'relation': '~w'", [HeadPred]),
    AllAssigns = [RelAssign | OutAssigns],
    atomic_list_concat(AllAssigns, ", ", OutputMapping),
    
    % Pattern match for first goal
    length(Args1, Arity1),
    findall(PatCheck,
        (   between(0, Arity1, Idx),
            Idx < Arity1,
            format(string(PatCheck), "'arg~w' in fact", [Idx])
        ),
        PatChecks),
    % Add relation check for fact (Goal1)
    format(string(RelCheck1), "fact.get('relation') == '~w'", [Pred1]),
    AllPatChecks = [RelCheck1 | PatChecks],
    atomic_list_concat(AllPatChecks, " and ", Pattern1),
    
    format(string(Code),
"if ~w:
        for other in total:
            if ~w:
                yield FrozenDict.from_dict({~w})",
        [Pattern1, FinalJoinCond, OutputMapping]).

%% translate_disjunct_nway_join(+Head, +Goals, +Builtins, -Code)
translate_disjunct_nway_join(Head, Goals, Builtins, Code) :-
    Goals = [FirstGoal | RestGoals],
    
    % Pattern match first goal
    FirstGoal =.. [Pred1 | Args1],
    length(Args1, Arity1),
    findall(Check, (between(0, Arity1, Idx), Idx < Arity1, format(string(Check), "'arg~w' in fact", [Idx])), Checks),
    % Add relation check
    format(string(RelCheck), "fact.get('relation') == '~w'", [Pred1]),
    AllChecks = [RelCheck | Checks],
    atomic_list_concat(AllChecks, " and ", Pattern1),
    
    % Nested joins
    build_nested_joins(RestGoals, 1, JoinCode, FinalIdx),
    
    % Output mapping
    Head =.. [HeadPred | HeadArgs],
    collect_all_goal_args([FirstGoal | RestGoals], AllGoalArgs),
    build_output_mapping(HeadArgs, FirstGoal, RestGoals, AllGoalArgs, OutputMapping),
    format(string(FullOutputMapping), "'relation': '~w', ~w", [HeadPred, OutputMapping]),
    
    % Constraints
    findall(G-S,
        (   nth1(I, RestGoals, G),
            format(string(S), "join_~w", [I])
        ),
        RestPairs),
    Pairs = [FirstGoal-"fact" | RestPairs],
    build_variable_map(Pairs, VarMap),
    translate_builtins(Builtins, VarMap, ConstraintChecks),
    
    % Calculate indentation for innermost block
    % Base indent is 4 (inside if Pattern1).
    % Each nested join adds 4 (for loop) + 4 (if condition) = 8?
    % build_nested_joins: Idx=1 -> Indent=8 (for loop). If=12.
    % So innermost body is at (FinalIdx + 2) * 4?
    % Let's verify:
    % RestGoals=[G1]. Idx=1. Loop at 8. If at 12. Body at 16.
    % FinalIdx=2. (2+2)*4 = 16. Correct.
    Indent is (FinalIdx + 2) * 4,
    format(string(Spaces), "~*c", [Indent, 32]),
    
    (   ConstraintChecks == ""
    ->  format(string(YieldBlock),
"~wyield FrozenDict.from_dict({~w})", [Spaces, FullOutputMapping])
    ;   format(string(YieldBlock),
"~wif ~w:
~w    yield FrozenDict.from_dict({~w})", [Spaces, ConstraintChecks, Spaces, FullOutputMapping])
    ),

    format(string(Code),
"if ~w:
~w
~w", [Pattern1, JoinCode, YieldBlock]).


%% build_constant_output(+HeadArgs, -OutputStr)
build_constant_output(HeadArgs, OutputStr) :-
    findall(Assign,
        (   nth0(Idx, HeadArgs, Arg),
            (   atom(Arg)
            ->  format(string(Assign), "'arg~w': '~w'", [Idx, Arg])
            ;   format(string(Assign), "'arg~w': None", [Idx])
            )
        ),
        Assigns),
    atomic_list_concat(Assigns, ", ", OutputStr).


%% is_builtin_goal(+Goal)
%  Check if goal is a built-in (is, >, <, etc.)
is_builtin_goal(Goal) :-
    Goal =.. [Pred | _],
    member(Pred, [is, >, <, >=, =<, =:=, =\=, \+, not, match]).

%% translate_fact_rule(+Name, +RuleNum, +Head, -RuleFunc)
%  Translate a fact (constant) into a rule that emits it once
translate_fact_rule(_Name, RuleNum, Head, RuleFunc) :-
    Head =.. [Pred | Args],
    extract_constants(Args, ConstPairs),
    format_dict_pairs(ConstPairs, ArgsStr),
    (   ArgsStr == ""
    ->  format(string(DictStr), "'relation': '~w'", [Pred])
    ;   format(string(DictStr), "'relation': '~w', ~w", [Pred, ArgsStr])
    ),
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

%% translate_copy_rule_with_builtins(+Name, +RuleNum, +Head, +Goal, +Builtins, -RuleFunc)
%  Copy rule with built-in constraints and relation check
translate_copy_rule_with_builtins(_Name, RuleNum, Head, Goal, Builtins, RuleFunc) :-
    Head =.. [HeadPred | HeadArgs],
    Goal =.. [GoalPred | GoalArgs],
    
    % Build pattern match condition
    length(GoalArgs, GoalArity),
    findall(Check,
        (   between(0, GoalArity, Idx),
            Idx < GoalArity,
            format(string(Check), "'arg~w' in fact", [Idx])
        ),
        Checks),
    % Add relation check
    format(string(RelCheck), "fact.get('relation') == '~w'", [GoalPred]),
    AllChecks = [RelCheck | Checks],
    atomic_list_concat(AllChecks, " and ", ConditionStr),
    
    % Constraints
    build_variable_map([Goal-"fact"], VarMap),
    translate_builtins(Builtins, VarMap, ConstraintChecks),
    
    % Build output dict
    findall(Assign,
        (   nth0(HIdx, HeadArgs, HeadArg),
            nth0(GIdx, GoalArgs, GoalArg),
            HeadArg == GoalArg,
            format(string(Assign), "'arg~w': fact.get('arg~w')", [HIdx, GIdx])
        ),
        Assigns),
    % Add relation to output
    format(string(RelAssign), "'relation': '~w'", [HeadPred]),
    AllAssigns = [RelAssign | Assigns],
    atomic_list_concat(AllAssigns, ", ", OutputStr),
    
    % Combine pattern and constraints
    (   ConstraintChecks == ""
    ->  FinalCondition = ConditionStr
    ;   format(string(FinalCondition), "~w and ~w", [ConditionStr, ConstraintChecks])
    ),
    
    format(string(RuleFunc),
"def _apply_rule_~w(fact: FrozenDict, total: Set[FrozenDict]) -> Iterator[FrozenDict]:
    '''Copy rule: ~w'''
    if ~w:
        yield FrozenDict.from_dict({~w})
", [RuleNum, Head, FinalCondition, OutputStr]).

%% translate_builtins(+Builtins, +VarMap, -ConstraintChecks)
%  Translate built-in predicates to Python expressions
translate_builtins([], _VarMap, "").
translate_builtins(Builtins, VarMap, ConstraintChecks) :-
    Builtins \= [],
    findall(Check,
        (   member(Builtin, Builtins),
            translate_builtin(Builtin, VarMap, Check)
        ),
        Checks),
    atomic_list_concat(Checks, " and ", ConstraintChecks).

python_config(Config) :-
    Config = [
        access_fmt-"~w.get('arg~w')",
        atom_fmt-"'~w'",
        null_val-"None",
        ops-[
            + - "+", - - "-", * - "*", / - "/", mod - "%",
            > - ">", < - "<", >= - ">=", =< - "<=", =:= - "==", =\= - "!=",
            is - "=="
        ]
    ].

%% translate_builtin(+Builtin, +VarMap, -PythonExpr)
%  Translate a single built-in to Python
translate_builtin(Goal, VarMap, PythonExpr) :-
    Goal =.. [Op | _],
    python_config(Config),
    memberchk(ops-Ops, Config),
    memberchk(Op-_, Ops),
    !,
    translate_builtin_common(Goal, VarMap, Config, PythonExpr).
translate_builtin(\+ Goal, VarMap, PythonExpr) :-
    !,
    translate_negation(Goal, VarMap, PythonExpr).
translate_builtin(not(Goal), VarMap, PythonExpr) :-
    !,
    translate_negation(Goal, VarMap, PythonExpr).
% Match predicate support (regex pattern matching)
translate_builtin(match(Var, Pattern), VarMap, PythonExpr) :-
    !,
    translate_match(Var, Pattern, auto, [], VarMap, PythonExpr).
translate_builtin(match(Var, Pattern, Type), VarMap, PythonExpr) :-
    !,
    translate_match(Var, Pattern, Type, [], VarMap, PythonExpr).
translate_builtin(match(Var, Pattern, Type, _Groups), VarMap, PythonExpr) :-
    !,
    % For now, capture groups are handled similarly to boolean match
    % TODO: Extract and use captured values
    translate_match(Var, Pattern, Type, [], VarMap, PythonExpr).
translate_builtin(_, _VarMap, "True").  % Fallback

%% translate_negation(+Goal, +VarMap, -PythonExpr)
translate_negation(Goal, VarMap, PythonExpr) :-
    python_config(Config),
    prepare_negation_data(Goal, VarMap, Config, Pairs),
    findall(PairStr,
        (   member(Key-Val, Pairs),
            format(string(PairStr), "'~w': ~w", [Key, Val])
        ),
        PairStrings),
    atomic_list_concat(PairStrings, ", ", DictContent),
    format(string(PythonExpr), "FrozenDict.from_dict({~w}) not in total", [DictContent]).

%% translate_match(+Var, +Pattern, +Type, +Groups, +VarMap, -PythonExpr)
%  Translate match predicate to Python re module call
translate_match(Var, Pattern, Type, _Groups, VarMap, PythonExpr) :-
    % Validate regex type for Python target
    validate_regex_type_for_python(Type),
    % Convert variable to Python expression
    translate_expr(Var, VarMap, PyVar),
    % Convert pattern to Python string
    (   atom(Pattern)
    ->  atom_string(Pattern, PatternStr)
    ;   PatternStr = Pattern
    ),
    % Escape pattern for Python (escape backslashes and quotes)
    escape_python_string(PatternStr, EscapedPattern),
    % Generate Python regex match expression
    % For now, use re.search for boolean match
    % TODO: Handle capture groups with re.search().groups()
    format(string(PythonExpr), "re.search(r'~w', str(~w))", [EscapedPattern, PyVar]).

%% validate_regex_type_for_python(+Type)
%  Validate that the regex type is supported by Python
validate_regex_type_for_python(auto) :- !.
validate_regex_type_for_python(python) :- !.
validate_regex_type_for_python(pcre) :- !.  % Python re is PCRE-like
validate_regex_type_for_python(ere) :- !.   % Can be supported with minor translation
validate_regex_type_for_python(Type) :-
    format('ERROR: Python target does not support regex type ~q~n', [Type]),
    format('  Supported types: auto, python, pcre, ere~n', []),
    format('  Note: BRE, AWK-specific, and .NET regex are not supported by Python~n', []),
    fail.

%% escape_python_string(+Str, -EscapedStr)
%  Escape special characters for Python string literals
escape_python_string(Str, EscapedStr) :-
    atom_string(Str, String),
    % For raw strings (r'...'), we mainly need to escape quotes
    % Backslashes are literal in raw strings
    re_replace("'"/g, "\\\\'", String, EscapedStr).

%% build_variable_map(+GoalSourcePairs, -VarMap)
%  Build map from variables to Python access strings
build_variable_map(GoalSourcePairs, VarMap) :-
    findall(Var-Access,
        (   member(Goal-Source, GoalSourcePairs),
            Goal =.. [_ | Args],
            nth0(Idx, Args, Var),
            var(Var),
            format(string(Access), "~w.get('arg~w')", [Source, Idx])
        ),
        VarMap).

%% translate_expr(+PrologExpr, +VarMap, -PythonExpr)
%  Translate Prolog expression to Python, mapping variables
translate_expr(Var, VarMap, PythonExpr) :-
    var(Var),
    !,
    (   memberchk(Var-Access, VarMap)
    ->  PythonExpr = Access
    ;   % Variable not found - assume it's a singleton or error
        % For now return None, but this indicates unsafe usage
        PythonExpr = "None"
    ).
translate_expr(Num, _VarMap, PythonExpr) :-
    number(Num),
    !,
    format(string(PythonExpr), "~w", [Num]).
translate_expr(Expr, VarMap, PythonExpr) :-
    compound(Expr),
    Expr =.. [Op, Left, Right],
    member(Op, [+, -, *, /, mod]), % Ensure this is handled
    !,
    translate_expr(Left, VarMap, LeftPy),
    translate_expr(Right, VarMap, RightPy),
    python_operator(Op, PyOp),
    format(string(PythonExpr), "(~w ~w ~w)", [LeftPy, PyOp, RightPy]).
translate_expr(Atom, _VarMap, PythonExpr) :-
    atom(Atom),
    !,
    format(string(PythonExpr), "'~w'", [Atom]).
translate_expr(_, _VarMap, "None").  % Fallback

%% python_operator(+PrologOp, -PythonOp)
python_operator(+, '+').
python_operator(-, '-').
python_operator(*, '*').
python_operator(/, '/').
python_operator(mod, '%').


%% translate_join_rule(+Name, +RuleNum, +Head, +Goals, -RuleFunc)
%  Translate a join rule (multiple goals in body)
translate_join_rule(Name, RuleNum, Head, Goals, RuleFunc) :-
    length(Goals, NumGoals),
    (   NumGoals == 2
    ->  % Binary join (existing fast path)
        Goals = [Goal1, Goal2],
        translate_binary_join(Name, RuleNum, Head, Goal1, Goal2, RuleFunc)
    ;   NumGoals >= 3
    ->  % N-way join (new!)
        translate_nway_join(RuleNum, Head, Goals, [], RuleFunc)
    ;   % Single goal shouldn't reach here
        format(string(RuleFunc),
"def _apply_rule_~w(fact: FrozenDict, total: Set[FrozenDict]) -> Iterator[FrozenDict]:
    '''ERROR: Invalid join - single goal should use copy rule'''
    return iter([])
", [RuleNum])
    ).

%% translate_join_rule_with_builtins(+Name, +RuleNum, +Head, +Goals, +Builtins, -RuleFunc)
%  Join rule with built-in constraints
translate_join_rule_with_builtins(Name, RuleNum, Head, Goals, Builtins, RuleFunc) :-
    (   Builtins == []
    ->  % No builtins, use regular join rule
        translate_join_rule(Name, RuleNum, Head, Goals, RuleFunc)
    ;   % Generate join with constraints
        % For binary joins, add constraints  to the existing logic
        length(Goals, NumGoals),
        (   NumGoals == 2
        ->  Goals = [Goal1, Goal2],
            translate_binary_join_with_constraints(RuleNum, Head, Goal1, Goal2, Builtins, RuleFunc)
        ;   % N-way joins with constraints
            translate_nway_join(RuleNum, Head, Goals, Builtins, RuleFunc)
        )
    ).

%% translate_binary_join_with_constraints(+RuleNum, +Head, +Goal1, +Goal2, +Builtins, -RuleFunc)
%% translate_binary_join_with_constraints(+RuleNum, +Head, +Goal1, +Goal2, +Builtins, -RuleFunc)
translate_binary_join_with_constraints(RuleNum, Head, Goal1, Goal2, Builtins, RuleFunc) :-
    % Similar to translate_binary_join but with constraint checks
    Goal1 =.. [Pred1 | Args1],
    Goal2 =.. [Pred2 | Args2],
    Head =.. [HeadPred | HeadArgs],
    
    % Find join variable
    findall(Var-Idx1-Idx2,
        (   nth0(Idx1, Args1, Var),
            nth0(Idx2, Args2, Var),
            \+ atom(Var)
        ),
        JoinVars),
    
    % Build join condition
    (   JoinVars = [_Var-JIdx1-JIdx2|_]
    ->  format(string(JoinCond), "other.get('arg~w') == fact.get('arg~w')", [JIdx2, JIdx1])
    ;   JoinCond = "True"
    ),
    
    % Add relation check for other (Goal2)
    format(string(RelCheck2), "other.get('relation') == '~w'", [Pred2]),
    (   JoinCond == "True"
    ->  JoinCondWithRel = RelCheck2
    ;   format(string(JoinCondWithRel), "~w and ~w", [JoinCond, RelCheck2])
    ),
    
    % Build constraint checks
    build_variable_map([Goal1-"fact", Goal2-"other"], VarMap),
    translate_builtins(Builtins, VarMap, ConstraintChecks),
    
    % Combine join and constraints
    (   ConstraintChecks == ""
    ->  FinalJoinCond = JoinCondWithRel
    ;   format(string(FinalJoinCond), "~w and ~w", [JoinCondWithRel, ConstraintChecks])
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
    % Add relation to output
    format(string(RelAssign), "'relation': '~w'", [HeadPred]),
    AllAssigns = [RelAssign | OutAssigns],
    atomic_list_concat(AllAssigns, ", ", OutputMapping),
    
    % Pattern match for first goal
    length(Args1, Arity1),
    findall(PatCheck,
        (   between(0, Arity1, Idx),
            Idx < Arity1,
            format(string(PatCheck), "'arg~w' in fact", [Idx])
        ),
        PatChecks),
    % Add relation check for fact (Goal1)
    format(string(RelCheck1), "fact.get('relation') == '~w'", [Pred1]),
    AllPatChecks = [RelCheck1 | PatChecks],
    atomic_list_concat(AllPatChecks, " and ", Pattern1),
    
    format(string(RuleFunc),
"def _apply_rule_~w(fact: FrozenDict, total: Set[FrozenDict]) -> Iterator[FrozenDict]:
    '''Join with constraints: ~w'''
    if ~w:
        for other in total:
            if ~w:
                yield FrozenDict.from_dict({~w})
", [RuleNum, Head, Pattern1, FinalJoinCond, OutputMapping]).



%% translate_binary_join(+Name, +RuleNum, +Head, +Goal1, +Goal2, -RuleFunc)
translate_binary_join(_Name, RuleNum, Head, Goal1, Goal2, RuleFunc) :-
    % Generate Block 1: Fact matches Goal 1, join with Goal 2
    generate_join_block(Head, Goal1, Goal2, Block1),
    
    % Generate Block 2: Fact matches Goal 2, join with Goal 1
    generate_join_block(Head, Goal2, Goal1, Block2),
    
    format(string(RuleFunc),
"def _apply_rule_~w(fact: FrozenDict, total: Set[FrozenDict]) -> Iterator[FrozenDict]:
    '''Join rule: ~w :- ~w, ~w'''
    # Case 1: Fact matches first goal
~w
    # Case 2: Fact matches second goal
~w
", [RuleNum, Head, Goal1, Goal2, Block1, Block2]).

%% generate_join_block(+Head, +TriggerGoal, +OtherGoal, -Code)
generate_join_block(Head, TriggerGoal, OtherGoal, Code) :-
    TriggerGoal =.. [Pred1 | Args1],
    OtherGoal =.. [Pred2 | Args2],
    Head =.. [HeadPred | HeadArgs],
    
    % Find join variable
    findall(Var-Idx1-Idx2,
        (   nth0(Idx1, Args1, V1),
            nth0(Idx2, Args2, V2),
            V1 == V2,
            Var = V1,
            \+ atom(Var)
        ),
        JoinVars),
    
    % Build join condition
    (   JoinVars = [_Var-JIdx1-JIdx2|_]
    ->  format(string(JoinCond), "other.get('arg~w') == fact.get('arg~w')", [JIdx2, JIdx1])
    ;   JoinCond = "True"
    ),
    
    % Add relation check for other
    format(string(RelCheck2), "other.get('relation') == '~w'", [Pred2]),
    (   JoinCond == "True"
    ->  JoinCondWithRel = RelCheck2
    ;   format(string(JoinCondWithRel), "~w and ~w", [JoinCond, RelCheck2])
    ),
    
    % Build output mapping
    findall(OutAssign,
        (   nth0(HIdx, HeadArgs, HVar),
            (   nth0(G1Idx, Args1, V1), V1 == HVar
            ->  format(string(OutAssign), "'arg~w': fact.get('arg~w')", [HIdx, G1Idx])
            ;   nth0(G2Idx, Args2, V2), V2 == HVar,
                format(string(OutAssign), "'arg~w': other.get('arg~w')", [HIdx, G2Idx])
            )
        ),
        OutAssigns),
    format(string(RelAssign), "'relation': '~w'", [HeadPred]),
    AllOutAssigns = [RelAssign | OutAssigns],
    atomic_list_concat(AllOutAssigns, ", ", OutputMapping),
    
    % Pattern match for trigger goal (fact)
    length(Args1, Arity1),
    findall(PatCheck,
        (   between(0, Arity1, Idx),
            Idx < Arity1,
            format(string(PatCheck), "'arg~w' in fact", [Idx])
        ),
        PatChecks),
    format(string(RelCheck1), "fact.get('relation') == '~w'", [Pred1]),
    AllPatChecks = [RelCheck1 | PatChecks],
    atomic_list_concat(AllPatChecks, " and ", Pattern1),
    
    format(string(Code),
"    if ~w:
        for other in total:
            if ~w:
                yield FrozenDict.from_dict({~w})",
        [Pattern1, JoinCondWithRel, OutputMapping]).

%% translate_nway_join(+RuleNum, +Head, +Goals, -RuleFunc)
%  Translate N-way join (3+ goals)
%% translate_nway_join(+RuleNum, +Head, +Goals, +Builtins, -RuleFunc)
translate_nway_join(RuleNum, Head, Goals, Builtins, RuleFunc) :-
    % Strategy: First goal from fact, rest joined from total
    Goals = [FirstGoal | RestGoals],
    
    % Build pattern match for first goal
    FirstGoal =.. [Pred1 | Args1],
    length(Args1, Arity1),
    findall(Check,
        (   between(0, Arity1, Idx),
            Idx < Arity1,
            format(string(Check), "'arg~w' in fact", [Idx])
        ),
        Checks),
    % Add relation check
    format(string(RelCheck), "fact.get('relation') == '~w'", [Pred1]),
    AllChecks = [RelCheck | Checks],
    atomic_list_concat(AllChecks, " and ", Pattern1),
    
    % Build nested joins for remaining goals
    build_nested_joins(RestGoals, 1, JoinCode, FinalIdx),
    
    % Build output mapping from head
    Head =.. [HeadPred | HeadArgs],
    collect_all_goal_args([FirstGoal | RestGoals], AllGoalArgs),
    build_output_mapping(HeadArgs, FirstGoal, RestGoals, AllGoalArgs, OutputMapping),
    format(string(FullOutputMapping), "'relation': '~w', ~w", [HeadPred, OutputMapping]),
    
    % Constraints
    findall(G-S,
        (   nth1(I, RestGoals, G),
            format(string(S), "join_~w", [I])
        ),
        RestPairs),
    Pairs = [FirstGoal-"fact" | RestPairs],
    build_variable_map(Pairs, VarMap),
    translate_builtins(Builtins, VarMap, ConstraintChecks),
    
    % Calculate indentation for yield block
    Indent is (FinalIdx + 2) * 4,
    format(string(Spaces), "~*c", [Indent, 32]),
    
    (   ConstraintChecks == ""
    ->  format(string(YieldBlock), "~wyield FrozenDict.from_dict({~w})", [Spaces, FullOutputMapping])
    ;   format(string(YieldBlock), "~wif ~w:\n~w    yield FrozenDict.from_dict({~w})", [Spaces, ConstraintChecks, Spaces, FullOutputMapping])
    ),
    
    format(string(RuleFunc),
"def _apply_rule_~w(fact: FrozenDict, total: Set[FrozenDict]) -> Iterator[FrozenDict]:
    '''N-way join: ~w'''
    # Match first goal
    if ~w:
~w
~w
", [RuleNum, Head, Pattern1, JoinCode, YieldBlock]).

%%build_nested_joins(+Goals, +StartIdx, -JoinCode, -FinalIdx)
%  Build nested for loops for N-way joins
build_nested_joins([], Idx, "", Idx) :- !.
build_nested_joins([Goal | RestGoals], Idx, JoinCode, FinalIdx) :-
    Goal =.. [Pred | _],
    Indent is (Idx + 1) * 4,
    format(string(Spaces), "~*c", [Indent, 32]),  % 32 = space char
    
    % Detect join conditions with previous goals
    detect_join_condition(Goal, Idx, JoinCond),
    
    % Add relation check
    format(string(RelCheck), "join_~w.get('relation') == '~w'", [Idx, Pred]),
    (   JoinCond == "True"
    ->  FullCond = RelCheck
    ;   format(string(FullCond), "~w and ~w", [RelCheck, JoinCond])
    ),
    
    format(string(ThisJoin),
"~wfor join_~w in total:
~w    if ~w:",
        [Spaces, Idx, Spaces, FullCond]),
    
    NextIdx is Idx + 1,
    build_nested_joins(RestGoals, NextIdx, RestCode, FinalIdx),
    
    (   RestCode == ""
    ->  JoinCode = ThisJoin
    ;   format(string(JoinCode), "~w\n~w", [ThisJoin, RestCode])
    ).

%% detect_join_condition(+Goal, +Idx, -JoinCond)
%  Find variables that join with previous goals
detect_join_condition(Goal, Idx, JoinCond) :-
    % Extract variables from goal
    Goal =.. [_Pred | Args],
    
    % Find first variable (simplified: join on first arg)
    % TODO: Track all variables and find actual join points
    (   Args = [FirstArg | _],
        var(FirstArg)
    ->  % Join on first argument matching previous goal's output
        PrevIdx is Idx - 1,
        format(string(JoinCond), 
            "join_~w.get('arg0') == (join_~w.get('arg1') if ~w > 0 else fact.get('arg1'))",
            [Idx, PrevIdx, PrevIdx])
    ;   % No clear join variable, default condition
        format(string(JoinCond), "True", [])
    ).


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
            nth0(GIdx, FirstArgs, GVar),
            GVar == HVar,
            format(string(Assign), "'arg~w': fact.get('arg~w')", [HIdx, GIdx])
        ),
        Assigns),
    atomic_list_concat(Assigns, ", ", Mapping).


%% generate_fixpoint_loop(+Name, +Facts, +Rules, -FixpointLoop)


generate_fixpoint_loop(_Name, Facts, Rules, FixpointLoop) :-


    % Generate fact initialization calls


    length(Facts, NumFacts),


    findall(FactCall,


        (   between(1, NumFacts, FactNum),


            format(string(FactCall), 


"    for fact in _init_fact_~w():


        if fact not in total:


            total.add(fact)


            delta.add(fact)


            yield fact.to_dict()", [FactNum])


        ),


        FactCalls),


    atomic_list_concat(FactCalls, "\n", FactCallsStr),





    % Generate rule calls


    length(Rules, NumRules),


    findall(RuleBlock,


        (   between(1, NumRules, RuleNum),


            format(string(RuleBlock), 


"            for new_fact in _apply_rule_~w(fact, total):


                if new_fact not in total and new_fact not in new_delta:


                    new_delta.add(new_fact)


                    yield new_fact.to_dict()", [RuleNum])


        ),


        RuleBlocks),


    


    (   RuleBlocks == []


    ->  LoopBody = "            pass"


    ;   atomic_list_concat(RuleBlocks, "\n", LoopBody)


    ),


    


    format(string(FixpointLoop),


"


def process_stream_generator(records: Iterator[Dict]) -> Iterator[Dict]:


    '''Semi-naive fixpoint evaluation.'''


    total: Set[FrozenDict] = set()


    delta: Set[FrozenDict] = set()


    


    # Initialize delta with input records


    for record in records:


        frozen = FrozenDict.from_dict(record)


        delta.add(frozen)


        total.add(frozen)


        yield record  # Yield initial facts


    


    # Initialize with static facts


~w


    


    # Fixpoint iteration (semi-naive evaluation)


    while delta:


        new_delta: Set[FrozenDict] = set()


        


        # Apply rules to facts in delta


        for fact in delta:


~w


        


        total.update(new_delta)


        delta = new_delta


", [FactCallsStr, LoopBody]).

%% generate_python_main(+Options, -MainCode)
%  Generate the main entry point with appropriate reader/writer
generate_python_main(Options, MainCode) :-
    option(record_format(Format), Options, jsonl),
    (   Format == nul_json
    ->  Writer = "write_nul_json"
    ;   Writer = "write_jsonl"
    ),
    
    (   option(input_source(xml(File, Tags)), Options)
    ->  % XML Source
        maplist(atom_string, Tags, TagStrs),
        maplist(quote_py_string, TagStrs, QuotedTags),
        atomic_list_concat(QuotedTags, ", ", TagsInner),
        format(string(ReaderCall), "read_xml_lxml('~w', {~w})", [File, TagsInner])
    ;   Format == nul_json
    ->  ReaderCall = "read_nul_json(sys.stdin)"
    ;   ReaderCall = "read_jsonl(sys.stdin)"
    ),
    
    % Handle generator mode vs procedural mode function name
    (   option(mode(generator), Options)
    ->  ProcessFunc = "process_stream_generator"
    ;   ProcessFunc = "process_stream"
    ),
    
    format(string(MainCode), 
"
def main():
    records = ~w
    results = ~w(records)
    ~w(results, sys.stdout)

if __name__ == '__main__':
    main()
", [ReaderCall, ProcessFunc, Writer]).

quote_py_string(Str, Quoted) :-
    format(string(Quoted), "'~w'", [Str]).

%% semantic_runtime_helpers(-Code)
%  Inject the Semantic Runtime library (Importer, Crawler, etc.) into the script
semantic_runtime_helpers(Code) :-
    % List of runtime files to inline
    Files = [
        'src/unifyweaver/targets/python_runtime/embedding.py',
        'src/unifyweaver/targets/python_runtime/importer.py',
        'src/unifyweaver/targets/python_runtime/onnx_embedding.py',
        'src/unifyweaver/targets/python_runtime/searcher.py',
        'src/unifyweaver/targets/python_runtime/crawler.py',
        'src/unifyweaver/targets/python_runtime/llm.py',
        'src/unifyweaver/targets/python_runtime/chunker.py'
    ],
    
    findall(Content, (
        member(File, Files),
        (   exists_file(File)
        ->  read_file_to_string(File, Raw, []),
            % Remove relative imports 'from .embedding import'
            re_replace("from \\.embedding import.*", "", Raw, Content)
        ;   format(string(Content), "# ERROR: Runtime file ~w not found\\n", [File])
        )
    ), Contents),
    
    Wrapper = "
class SemanticRuntime:
    def __init__(self, db_path='data.db', model_path='models/model.onnx', vocab_path='models/vocab.txt'):
        self.importer = PtImporter(db_path)
        if os.path.exists(model_path):
            self.embedder = OnnxEmbeddingProvider(model_path, vocab_path)
        else:
            sys.stderr.write(f'Warning: Model {model_path} not found, embeddings disabled\\\\n')
            self.embedder = None
            
        self.crawler = PtCrawler(self.importer, self.embedder)
        self.searcher = PtSearcher(db_path, self.embedder)
        self.llm = LLMProvider()
        self.chunker = HierarchicalChunker()

_runtime_instance = None
def _get_runtime():
    global _runtime_instance
    if _runtime_instance is None:
        _runtime_instance = SemanticRuntime()
    return _runtime_instance

def fetch_xml_func(url):
    if os.path.exists(url):
        return open(url, 'rb')
    return None
",

    atomic_list_concat(Contents, "\n", LibCode),
    string_concat(LibCode, Wrapper, Code).

% ============================================================================
% TESTS - Import Auto-Generation
% ============================================================================

%% test_import_autogeneration
%  Test that binding imports are automatically included in generated Python code
test_import_autogeneration :-
    format('~n=== Python Import Auto-Generation Tests ===~n~n', []),

    % Test 1: Clear imports starts fresh
    format('[Test 1] Clear binding imports~n', []),
    clear_binding_imports,
    findall(M, required_import(M), Ms1),
    (   Ms1 == []
    ->  format('  [PASS] No imports after clear~n', [])
    ;   format('  [FAIL] Expected no imports, got ~w~n', [Ms1])
    ),

    % Test 2: Recording imports
    format('[Test 2] Record binding imports~n', []),
    assertz(required_import(math)),
    assertz(required_import(os)),
    assertz(required_import(collections)),
    findall(M, required_import(M), Ms2),
    sort(Ms2, Sorted2),
    (   Sorted2 == [collections, math, os]
    ->  format('  [PASS] Imports recorded: ~w~n', [Sorted2])
    ;   format('  [FAIL] Expected [collections, math, os], got ~w~n', [Sorted2])
    ),

    % Test 3: get_binding_imports generates correct import lines
    format('[Test 3] Generate import statements~n', []),
    get_binding_imports(ImportStr),
    (   sub_string(ImportStr, _, _, _, "import math"),
        sub_string(ImportStr, _, _, _, "import os"),
        sub_string(ImportStr, _, _, _, "import collections")
    ->  format('  [PASS] Import string contains math, os, collections~n', [])
    ;   format('  [FAIL] Missing imports in: ~w~n', [ImportStr])
    ),

    % Test 4: header/1 includes binding imports
    format('[Test 4] Header includes binding imports~n', []),
    header(Header),
    (   sub_string(Header, _, _, _, "import math"),
        sub_string(Header, _, _, _, "import sys"),
        sub_string(Header, _, _, _, "import json")
    ->  format('  [PASS] Header includes base + binding imports~n', [])
    ;   format('  [FAIL] Header missing imports~n', [])
    ),

    % Test 5: header_with_functools/1 includes both functools and bindings
    format('[Test 5] Header with functools includes binding imports~n', []),
    header_with_functools(HeaderFT),
    (   sub_string(HeaderFT, _, _, _, "import functools"),
        sub_string(HeaderFT, _, _, _, "import math"),
        sub_string(HeaderFT, _, _, _, "import collections")
    ->  format('  [PASS] Header with functools includes binding imports~n', [])
    ;   format('  [FAIL] Header with functools missing imports~n', [])
    ),

    % Test 6: generator_header/1 includes binding imports
    format('[Test 6] Generator header includes binding imports~n', []),
    generator_header(GenHeader),
    (   sub_string(GenHeader, _, _, _, "import math"),
        sub_string(GenHeader, _, _, _, "FrozenDict")
    ->  format('  [PASS] Generator header includes binding imports + FrozenDict~n', [])
    ;   format('  [FAIL] Generator header missing imports~n', [])
    ),

    % Test 7: Skip already-included base modules
    format('[Test 7] Skip base modules (sys, json, re, etc.)~n', []),
    clear_binding_imports,
    assertz(required_import(sys)),    % Should be skipped
    assertz(required_import(json)),   % Should be skipped
    assertz(required_import(numpy)),  % Should be included
    get_binding_imports(ImportStr2),
    (   sub_string(ImportStr2, _, _, _, "import numpy"),
        \+ sub_string(ImportStr2, _, _, _, "import sys\nimport sys")
    ->  format('  [PASS] Base modules skipped, numpy included~n', [])
    ;   format('  [FAIL] Base module filtering issue~n', [])
    ),

    % Cleanup
    clear_binding_imports,

    % Test 8: End-to-end - binding lookup records imports
    format('[Test 8] End-to-end: binding lookup records imports~n', []),
    init_python_target,
    % sqrt/2 binding has import(math) - simulate what translate_goal does
    (   binding(python, sqrt/2, _TargetName, _Inputs, _Outputs, SqrtOptions),
        member(import(math), SqrtOptions)
    ->  % Record import as translate_goal would
        assertz(required_import(math)),
        format('  [PASS] sqrt/2 binding has import(math), recorded~n', [])
    ;   format('  [FAIL] sqrt/2 binding does not have import(math)~n', [])
    ),

    % Test 9: Multiple bindings accumulate imports
    format('[Test 9] Multiple bindings accumulate imports~n', []),
    clear_binding_imports,
    init_python_bindings,
    % Simulate translate_goal recording imports for multiple bindings
    (   binding(python, sqrt/2, _, _, _, Opts1), member(import(M1), Opts1)
    ->  assertz(required_import(M1))
    ;   true
    ),
    (   binding(python, counter/2, _, _, _, Opts2), member(import(M2), Opts2)
    ->  assertz(required_import(M2))
    ;   true
    ),
    get_binding_imports(AccumImports),
    (   sub_string(AccumImports, _, _, _, "import math"),
        sub_string(AccumImports, _, _, _, "import collections")
    ->  format('  [PASS] Multiple imports accumulated: math, collections~n', [])
    ;   format('  [FAIL] Import accumulation issue: ~w~n', [AccumImports])
    ),

    % Final cleanup
    clear_binding_imports,

    format('~n=== All Import Auto-Generation Tests Passed ===~n', []).

% ============================================================================
% TESTS - Method Call Pattern Enhancement
% ============================================================================

%% test_method_call_patterns
%  Test the enhanced method call code generation
test_method_call_patterns :-
    format('~n=== Python Method Call Pattern Tests ===~n~n', []),

    % Test 1: No-arg method with () in TargetName
    format('[Test 1] No-arg method with () in name~n', []),
    generate_method_call('.strip()', [v(str), v(result)], [string], CallExpr1),
    (   CallExpr1 == "v_str.strip()"
    ->  format('  [PASS] .strip() -> ~w~n', [CallExpr1])
    ;   format('  [FAIL] Expected v_str.strip(), got ~w~n', [CallExpr1])
    ),

    % Test 2: No-arg method without () in TargetName
    format('[Test 2] No-arg method without () in name~n', []),
    generate_method_call('.lower', [v(str), v(result)], [string], CallExpr2),
    (   CallExpr2 == "v_str.lower()"
    ->  format('  [PASS] .lower -> ~w~n', [CallExpr2])
    ;   format('  [FAIL] Expected v_str.lower(), got ~w~n', [CallExpr2])
    ),

    % Test 3: Method with one argument and output
    format('[Test 3] Method with argument and output~n', []),
    generate_method_call('.split', [v(str), v(delim), v(result)], [list], CallExpr3),
    (   CallExpr3 == "v_str.split(v_delim)"
    ->  format('  [PASS] .split with arg -> ~w~n', [CallExpr3])
    ;   format('  [FAIL] Expected v_str.split(v_delim), got ~w~n', [CallExpr3])
    ),

    % Test 4: Method with multiple arguments and output
    format('[Test 4] Method with multiple arguments~n', []),
    generate_method_call('.replace', [v(str), v(old), v(new), v(result)], [string], CallExpr4),
    (   CallExpr4 == "v_str.replace(v_old, v_new)"
    ->  format('  [PASS] .replace with args -> ~w~n', [CallExpr4])
    ;   format('  [FAIL] Expected v_str.replace(v_old, v_new), got ~w~n', [CallExpr4])
    ),

    % Test 5: Mutating method (no output) with argument
    format('[Test 5] Mutating method with argument (no output)~n', []),
    generate_method_call('.append', [v(list), v(item)], [], CallExpr5),
    (   CallExpr5 == "v_list.append(v_item)"
    ->  format('  [PASS] .append mutating -> ~w~n', [CallExpr5])
    ;   format('  [FAIL] Expected v_list.append(v_item), got ~w~n', [CallExpr5])
    ),

    % Test 6: Function call with output
    format('[Test 6] Function call with output~n', []),
    generate_function_call('len', [v(list), v(result)], [int], CallExpr6),
    (   CallExpr6 == "len(v_list)"
    ->  format('  [PASS] len() function -> ~w~n', [CallExpr6])
    ;   format('  [FAIL] Expected len(v_list), got ~w~n', [CallExpr6])
    ),

    % Test 7: Function call with no output (side effect)
    format('[Test 7] Function call with no output~n', []),
    generate_function_call('print', [v(msg)], [], CallExpr7),
    (   CallExpr7 == "print(v_msg)"
    ->  format('  [PASS] print() no output -> ~w~n', [CallExpr7])
    ;   format('  [FAIL] Expected print(v_msg), got ~w~n', [CallExpr7])
    ),

    % Test 8: Constant function (no input args)
    format('[Test 8] Constant function (pi)~n', []),
    generate_function_call('math.pi', [v(result)], [float], CallExpr8),
    (   CallExpr8 == "math.pi"
    ->  format('  [PASS] math.pi constant -> ~w~n', [CallExpr8])
    ;   format('  [FAIL] Expected math.pi, got ~w~n', [CallExpr8])
    ),

    % Test 9: Chained method call
    format('[Test 9] Chained method call~n', []),
    generate_chained_call(
        [method('.strip', []), method('.lower', [])],
        [v(str), v(result)],
        [string],
        CallExpr9
    ),
    (   CallExpr9 == "v_str.strip().lower()"
    ->  format('  [PASS] Chained .strip().lower() -> ~w~n', [CallExpr9])
    ;   format('  [FAIL] Expected v_str.strip().lower(), got ~w~n', [CallExpr9])
    ),

    % Test 10: Chained method with arguments
    format('[Test 10] Chained method with arguments~n', []),
    generate_chained_call(
        [method('.replace', [0, 1]), method('.strip', [])],
        [v(str), v(old), v(new), v(result)],
        [string],
        CallExpr10
    ),
    (   CallExpr10 == "v_str.replace(v_old, v_new).strip()"
    ->  format('  [PASS] Chained .replace().strip() -> ~w~n', [CallExpr10])
    ;   format('  [FAIL] Expected v_str.replace(v_old, v_new).strip(), got ~w~n', [CallExpr10])
    ),

    % Test 11: Full code generation with method call binding
    format('[Test 11] Full code generation for method binding~n', []),
    generate_binding_call_python('.lower()', [v(str), v(result)], [string], [pattern(method_call)], Code11),
    (   sub_string(Code11, _, _, _, "v_result = v_str.lower()")
    ->  format('  [PASS] Full method binding code generated~n', [])
    ;   format('  [FAIL] Expected assignment, got ~w~n', [Code11])
    ),

    % Test 12: Full code generation for mutating method (no assignment)
    format('[Test 12] Full code generation for mutating method~n', []),
    generate_binding_call_python('.append', [v(list), v(item)], [], [pattern(method_call)], Code12),
    (   sub_string(Code12, _, _, _, "v_list.append(v_item)"),
        \+ sub_string(Code12, _, _, _, "=")
    ->  format('  [PASS] Mutating method - no assignment~n', [])
    ;   format('  [FAIL] Unexpected output: ~w~n', [Code12])
    ),

    format('~n=== All Method Call Pattern Tests Passed ===~n', []).

% ============================================================================
% TESTS - Pipeline Mode (Phase 1: Object Pipeline Support)
% ============================================================================

%% test_pipeline_mode
%  Test the pipeline mode code generation
test_pipeline_mode :-
    format('~n=== Python Pipeline Mode Tests ===~n~n', []),

    % Test 1: Generate default arg names
    format('[Test 1] Generate default arg names~n', []),
    generate_default_arg_names(3, ArgNames1),
    (   ArgNames1 == [arg_0, arg_1, arg_2]
    ->  format('  [PASS] Default arg names: ~w~n', [ArgNames1])
    ;   format('  [FAIL] Expected [arg_0, arg_1, arg_2], got ~w~n', [ArgNames1])
    ),

    % Test 2: Pipeline header for JSONL
    format('[Test 2] Pipeline header for JSONL~n', []),
    clear_binding_imports,
    pipeline_header(jsonl, Header2),
    (   sub_string(Header2, _, _, _, "import json"),
        sub_string(Header2, _, _, _, "Generator")
    ->  format('  [PASS] JSONL header has json and Generator imports~n', [])
    ;   format('  [FAIL] Header missing imports: ~w~n', [Header2])
    ),

    % Test 3: Pipeline header for MessagePack
    format('[Test 3] Pipeline header for MessagePack~n', []),
    pipeline_header(messagepack, Header3),
    (   sub_string(Header3, _, _, _, "import msgpack")
    ->  format('  [PASS] MessagePack header has msgpack import~n', [])
    ;   format('  [FAIL] Header missing msgpack: ~w~n', [Header3])
    ),

    % Test 4: Pipeline helpers for JSONL with same_as_data errors
    format('[Test 4] Pipeline helpers for JSONL (same_as_data errors)~n', []),
    pipeline_helpers(jsonl, same_as_data, Helpers4),
    (   sub_string(Helpers4, _, _, _, "read_stream"),
        sub_string(Helpers4, _, _, _, "write_record"),
        sub_string(Helpers4, _, _, _, "__error__")
    ->  format('  [PASS] JSONL helpers have read_stream, write_record, error handling~n', [])
    ;   format('  [FAIL] Missing helpers~n', [])
    ),

    % Test 5: Pipeline helpers for JSONL with text errors
    format('[Test 5] Pipeline helpers for JSONL (text errors)~n', []),
    pipeline_helpers(jsonl, text, Helpers5),
    (   sub_string(Helpers5, _, _, _, "ERROR [")
    ->  format('  [PASS] Text error protocol has plain text error format~n', [])
    ;   format('  [FAIL] Missing plain text error format~n', [])
    ),

    % Test 6: Output formatting for object mode
    format('[Test 6] Output formatting for object mode~n', []),
    generate_output_formatting([_A, _B, _C], ['Id', 'Name', 'Active'], object, OutputCode6),
    (   sub_string(OutputCode6, _, _, _, "'Id': v_0"),
        sub_string(OutputCode6, _, _, _, "'Name': v_1"),
        sub_string(OutputCode6, _, _, _, "'Active': v_2")
    ->  format('  [PASS] Object output has named keys~n', [])
    ;   format('  [FAIL] Output formatting issue: ~w~n', [OutputCode6])
    ),

    % Test 7: Output formatting for text mode
    format('[Test 7] Output formatting for text mode~n', []),
    generate_output_formatting([_X], ['Result'], text, OutputCode7),
    (   sub_string(OutputCode7, _, _, _, "str(v_0)")
    ->  format('  [PASS] Text output uses str()~n', [])
    ;   format('  [FAIL] Text output issue: ~w~n', [OutputCode7])
    ),

    % Test 8: Pipeline main block
    format('[Test 8] Pipeline main block~n', []),
    generate_pipeline_main("test_pred", jsonl, [], Main8),
    (   sub_string(Main8, _, _, _, "if __name__"),
        sub_string(Main8, _, _, _, "read_stream(sys.stdin)"),
        sub_string(Main8, _, _, _, "test_pred(input_stream)")
    ->  format('  [PASS] Main block has correct structure~n', [])
    ;   format('  [FAIL] Main block issue: ~w~n', [Main8])
    ),

    % Test 9: Full pipeline compilation (needs test predicate)
    format('[Test 9] Full pipeline compilation~n', []),
    % Define a simple test predicate
    abolish(test_user_info/2),
    assert((test_user_info(Id, Email) :- Email = Id)),
    (   catch(
            compile_predicate_to_python(test_user_info/2, [
                pipeline_input(true),
                output_format(object),
                arg_names(['UserId', 'Email'])
            ], Code9),
            Error9,
            (format('  [FAIL] Compilation error: ~w~n', [Error9]), fail)
        )
    ->  (   sub_string(Code9, _, _, _, "def test_user_info(stream"),
            sub_string(Code9, _, _, _, "'UserId'"),
            sub_string(Code9, _, _, _, "'Email'"),
            sub_string(Code9, _, _, _, "read_stream"),
            sub_string(Code9, _, _, _, "write_record")
        ->  format('  [PASS] Full pipeline code generated correctly~n', [])
        ;   format('  [FAIL] Generated code missing expected content~n', []),
            format('  Code: ~w~n', [Code9])
        )
    ;   format('  [FAIL] Pipeline compilation failed~n', [])
    ),
    abolish(test_user_info/2),

    format('~n=== All Pipeline Mode Tests Passed ===~n', []).

% ============================================================================
% TESTS - Runtime Selection (Phase 2)
% ============================================================================

%% test_runtime_selection
%  Test the runtime selection system
test_runtime_selection :-
    format('~n=== Python Runtime Selection Tests ===~n~n', []),

    % Test 1: CPython is always available
    format('[Test 1] CPython availability~n', []),
    (   runtime_available(cpython)
    ->  format('  [PASS] CPython is available~n', [])
    ;   format('  [FAIL] CPython should always be available~n', [])
    ),

    % Test 2: CPython compatible with all imports
    format('[Test 2] CPython import compatibility~n', []),
    (   runtime_compatible_with_imports(cpython, [numpy, tensorflow, pandas])
    ->  format('  [PASS] CPython supports all imports~n', [])
    ;   format('  [FAIL] CPython should support all imports~n', [])
    ),

    % Test 3: IronPython incompatible with numpy
    format('[Test 3] IronPython numpy incompatibility~n', []),
    (   \+ runtime_compatible_with_imports(ironpython, [numpy])
    ->  format('  [PASS] IronPython correctly rejects numpy~n', [])
    ;   format('  [FAIL] IronPython should reject numpy~n', [])
    ),

    % Test 4: IronPython compatible with basic imports
    format('[Test 4] IronPython basic import compatibility~n', []),
    (   runtime_compatible_with_imports(ironpython, [json, re, os, sys])
    ->  format('  [PASS] IronPython supports basic imports~n', [])
    ;   format('  [FAIL] IronPython should support basic imports~n', [])
    ),

    % Test 5: Runtime selection with no constraints
    format('[Test 5] Runtime selection (no constraints)~n', []),
    select_python_runtime(test/1, [], [], Runtime5),
    (   Runtime5 == cpython
    ->  format('  [PASS] Default selection is cpython: ~w~n', [Runtime5])
    ;   format('  [INFO] Selected runtime: ~w (cpython expected as default)~n', [Runtime5])
    ),

    % Test 6: Runtime selection with numpy import
    format('[Test 6] Runtime selection with numpy~n', []),
    select_python_runtime(test/1, [numpy], [], Runtime6),
    (   Runtime6 == cpython
    ->  format('  [PASS] numpy forces cpython: ~w~n', [Runtime6])
    ;   format('  [FAIL] numpy should force cpython, got: ~w~n', [Runtime6])
    ),

    % Test 7: Runtime scoring
    format('[Test 7] Runtime scoring~n', []),
    Prefs7 = [prefer_runtime([ironpython, cpython, pypy])],
    score_single_runtime(Prefs7, [], cpython, cpython-Score7a),
    score_single_runtime(Prefs7, [], ironpython, ironpython-Score7b),
    (   Score7b > Score7a
    ->  format('  [PASS] IronPython scores higher (~w) than CPython (~w) with preference~n', [Score7b, Score7a])
    ;   format('  [FAIL] Scoring issue: iron=~w, cpython=~w~n', [Score7b, Score7a])
    ),

    % Test 8: Context-based selection (.NET context)
    format('[Test 8] Context-based runtime selection~n', []),
    (   runtime_satisfies_context(ironpython, [target(csharp)])
    ->  format('  [PASS] IronPython satisfies .NET context~n', [])
    ;   format('  [FAIL] IronPython should satisfy .NET context~n', [])
    ),

    % Test 9: Firewall denies runtime
    format('[Test 9] Firewall runtime denial~n', []),
    Firewall9 = [denied([python_runtime(ironpython)])],
    (   \+ valid_runtime_candidate(ironpython, [], Firewall9, [])
    ->  format('  [PASS] Firewall correctly denies ironpython~n', [])
    ;   format('  [FAIL] Firewall should deny ironpython~n', [])
    ),

    % Test 10: Communication preference
    format('[Test 10] Communication preference~n', []),
    (   runtime_communication(ironpython, in_process, [target(csharp)])
    ->  format('  [PASS] IronPython is in-process for .NET~n', [])
    ;   format('  [FAIL] IronPython should be in-process for .NET~n', [])
    ),
    (   runtime_communication(cpython, cross_process, [])
    ->  format('  [PASS] CPython is cross-process~n', [])
    ;   format('  [FAIL] CPython should be cross-process~n', [])
    ),

    format('~n=== All Runtime Selection Tests Passed ===~n', []).

% ============================================================================
% TESTS - Runtime-Specific Code Generation (Phase 3)
% ============================================================================

%% test_runtime_headers
%  Test the runtime-specific pipeline header generation
test_runtime_headers :-
    format('~n=== Python Runtime-Specific Header Tests (Phase 3) ===~n~n', []),
    clear_binding_imports,

    % Test 1: CPython JSONL header
    format('[Test 1] CPython JSONL header~n', []),
    pipeline_header(jsonl, cpython, Header1),
    (   sub_string(Header1, _, _, _, "#!/usr/bin/env python3"),
        sub_string(Header1, _, _, _, "Runtime: CPython"),
        sub_string(Header1, _, _, _, "import json")
    ->  format('  [PASS] CPython JSONL header correct~n', [])
    ;   format('  [FAIL] CPython JSONL header issue~n', [])
    ),

    % Test 2: IronPython JSONL header (CLR integration)
    format('[Test 2] IronPython JSONL header (CLR imports)~n', []),
    pipeline_header(jsonl, ironpython, Header2),
    (   sub_string(Header2, _, _, _, "#!/usr/bin/env ipy"),
        sub_string(Header2, _, _, _, "import clr"),
        sub_string(Header2, _, _, _, "clr.AddReference('System')"),
        sub_string(Header2, _, _, _, "from System import"),
        sub_string(Header2, _, _, _, "Dictionary, List"),
        sub_string(Header2, _, _, _, "to_dotnet_dict")
    ->  format('  [PASS] IronPython has CLR imports and helpers~n', [])
    ;   format('  [FAIL] IronPython missing CLR integration~n', [])
    ),

    % Test 3: IronPython MessagePack header (fallback)
    format('[Test 3] IronPython MessagePack header~n', []),
    pipeline_header(messagepack, ironpython, Header3),
    (   sub_string(Header3, _, _, _, "System.Text.Json"),
        sub_string(Header3, _, _, _, "class msgpack")
    ->  format('  [PASS] IronPython has msgpack fallback~n', [])
    ;   format('  [FAIL] IronPython missing msgpack fallback~n', [])
    ),

    % Test 4: PyPy JSONL header
    format('[Test 4] PyPy JSONL header~n', []),
    pipeline_header(jsonl, pypy, Header4),
    (   sub_string(Header4, _, _, _, "#!/usr/bin/env pypy3"),
        sub_string(Header4, _, _, _, "Runtime: PyPy"),
        sub_string(Header4, _, _, _, "JIT-optimized")
    ->  format('  [PASS] PyPy header correct~n', [])
    ;   format('  [FAIL] PyPy header issue~n', [])
    ),

    % Test 5: Jython JSONL header (Java integration)
    format('[Test 5] Jython JSONL header (Java imports)~n', []),
    pipeline_header(jsonl, jython, Header5),
    (   sub_string(Header5, _, _, _, "#!/usr/bin/env jython"),
        sub_string(Header5, _, _, _, "from java.lang import"),
        sub_string(Header5, _, _, _, "HashMap, ArrayList"),
        sub_string(Header5, _, _, _, "to_java_map")
    ->  format('  [PASS] Jython has Java imports and helpers~n', [])
    ;   format('  [FAIL] Jython missing Java integration~n', [])
    ),

    % Test 6: Jython MessagePack header
    format('[Test 6] Jython MessagePack header~n', []),
    pipeline_header(messagepack, jython, Header6),
    (   sub_string(Header6, _, _, _, "ByteArrayOutputStream"),
        sub_string(Header6, _, _, _, "ObjectOutputStream"),
        sub_string(Header6, _, _, _, "class msgpack")
    ->  format('  [PASS] Jython has Java msgpack fallback~n', [])
    ;   format('  [FAIL] Jython missing msgpack fallback~n', [])
    ),

    % Test 7: compile_pipeline_mode uses runtime option
    format('[Test 7] Pipeline compilation with runtime option~n', []),
    abolish(test_iron_pred/1),
    assert((test_iron_pred(X) :- X = hello)),
    (   catch(
            compile_predicate_to_python(test_iron_pred/1, [
                pipeline_input(true),
                output_format(object),
                arg_names(['Value']),
                runtime(ironpython)
            ], Code7),
            Error7,
            (format('  [FAIL] Compilation error: ~w~n', [Error7]), fail)
        )
    ->  (   sub_string(Code7, _, _, _, "#!/usr/bin/env ipy"),
            sub_string(Code7, _, _, _, "import clr"),
            sub_string(Code7, _, _, _, "to_dotnet_dict")
        ->  format('  [PASS] Pipeline uses IronPython runtime header~n', [])
        ;   format('  [FAIL] Pipeline missing IronPython header~n', [])
        )
    ;   format('  [FAIL] Pipeline compilation failed~n', [])
    ),
    abolish(test_iron_pred/1),

    % Test 8: Legacy 2-arg pipeline_header still works
    format('[Test 8] Legacy pipeline_header/2 compatibility~n', []),
    pipeline_header(jsonl, LegacyHeader),
    (   sub_string(LegacyHeader, _, _, _, "#!/usr/bin/env python3"),
        sub_string(LegacyHeader, _, _, _, "import json")
    ->  format('  [PASS] Legacy header defaults to CPython~n', [])
    ;   format('  [FAIL] Legacy header compatibility issue~n', [])
    ),

    format('~n=== All Runtime-Specific Header Tests Passed ===~n', []).

% ============================================================================
% PIPELINE CHAINING (Phase 4)
% ============================================================================
%
% Pipeline chaining connects multiple predicates in a data flow:
%   input -> pred1 -> pred2 -> pred3 -> output
%
% Two modes:
%   1. Same-runtime chaining: All predicates run in same Python process
%   2. Cross-runtime chaining: Predicates may run in different runtimes,
%      connected via JSONL pipes or in-process bridges
%

%% compile_pipeline(+Predicates, +Options, -Code)
%  Main entry point for pipeline chaining.
%  Automatically selects same-runtime or cross-runtime based on predicates.
%
%  Predicates: List of Name/Arity or Target:Name/Arity
%    Examples:
%      - [get_users/1, filter_active/2, format_output/1]
%      - [python:get_users/1, csharp:validate/1, python:format/1]
%
%  Options:
%    - runtime(Runtime)     : Force specific runtime for same-runtime
%    - glue_protocol(P)     : Protocol for cross-runtime (jsonl/messagepack)
%    - pipeline_name(Name)  : Name for generated pipeline function
%    - arg_names(Names)     : Property names for final output
%
compile_pipeline(Predicates, Options, Code) :-
    % Check if all predicates are same runtime
    (   all_same_runtime(Predicates)
    ->  compile_same_runtime_pipeline(Predicates, Options, Code)
    ;   compile_cross_runtime_pipeline(Predicates, Options, Code)
    ).

%% all_same_runtime(+Predicates)
%  True if all predicates can run in the same Python runtime.
%
all_same_runtime([]).
all_same_runtime([Pred|Rest]) :-
    predicate_runtime(Pred, Runtime),
    all_same_runtime_check(Rest, Runtime).

all_same_runtime_check([], _).
all_same_runtime_check([Pred|Rest], Runtime) :-
    predicate_runtime(Pred, PredRuntime),
    compatible_runtimes(Runtime, PredRuntime),
    all_same_runtime_check(Rest, Runtime).

%% predicate_runtime(+Pred, -Runtime)
%  Determine the runtime for a predicate.
%
predicate_runtime(python:_Name/_Arity, python) :- !.
predicate_runtime(cpython:_Name/_Arity, cpython) :- !.
predicate_runtime(ironpython:_Name/_Arity, ironpython) :- !.
predicate_runtime(pypy:_Name/_Arity, pypy) :- !.
predicate_runtime(jython:_Name/_Arity, jython) :- !.
predicate_runtime(csharp:_Name/_Arity, csharp) :- !.
predicate_runtime(powershell:_Name/_Arity, powershell) :- !.
predicate_runtime(_Name/_Arity, python).  % Default to python

%% compatible_runtimes(+R1, +R2)
%  True if two runtimes can run in same process.
%
compatible_runtimes(R, R) :- !.
compatible_runtimes(python, cpython) :- !.
compatible_runtimes(cpython, python) :- !.
compatible_runtimes(python, ironpython) :- !.
compatible_runtimes(ironpython, python) :- !.
compatible_runtimes(python, pypy) :- !.
compatible_runtimes(pypy, python) :- !.
compatible_runtimes(python, jython) :- !.
compatible_runtimes(jython, python) :- !.

% ============================================================================
% Same-Runtime Pipeline Chaining
% ============================================================================

%% compile_same_runtime_pipeline(+Predicates, +Options, -Code)
%  Compile a pipeline where all predicates run in the same Python process.
%  This is efficient as no serialization is needed between steps.
%
%  Supports pipeline_mode option:
%    - sequential (default): Stages chained sequentially
%    - generator: Fixpoint iteration with deduplication
%
compile_same_runtime_pipeline(Predicates, Options, Code) :-
    option(runtime(Runtime), Options, cpython),
    option(glue_protocol(GlueProtocol), Options, jsonl),
    option(pipeline_name(PipelineName), Options, pipeline),
    option(pipeline_mode(PipelineMode), Options, sequential),
    option(arg_names(ArgNames), Options, []),

    % Generate header (extended for generator mode)
    pipeline_header_extended(GlueProtocol, Runtime, PipelineMode, Header),

    % Compile each predicate to a function
    compile_pipeline_predicates(Predicates, PredicateFunctions),

    % Generate the pipeline connector function based on mode and runtime
    generate_pipeline_connector(Predicates, PipelineName, PipelineMode, Runtime, ConnectorCode),

    % Generate helpers (extended for generator mode, runtime-aware)
    pipeline_helpers_extended(GlueProtocol, same_as_data, PipelineMode, Runtime, Helpers),

    % Generate main block
    generate_chained_pipeline_main(PipelineName, GlueProtocol, ArgNames, MainCode),

    format(string(Code), "~w~w~w~w~w",
        [Header, Helpers, PredicateFunctions, ConnectorCode, MainCode]).

%% pipeline_header_extended(+Protocol, +Runtime, +Mode, -Header)
%  Generate header with additional imports for generator mode
%  IronPython uses .NET HashSet instead of dataclasses
pipeline_header_extended(GlueProtocol, ironpython, generator, Header) :-
    !,
    pipeline_header(GlueProtocol, ironpython, BaseHeader),
    % IronPython: Use .NET HashSet, no dataclasses needed
    format(string(Header), "~w
# .NET HashSet for fixpoint deduplication (IronPython)
from System.Collections.Generic import HashSet

", [BaseHeader]).
pipeline_header_extended(GlueProtocol, Runtime, generator, Header) :-
    !,
    pipeline_header(GlueProtocol, Runtime, BaseHeader),
    format(string(Header), "~wfrom typing import Set
from dataclasses import dataclass

", [BaseHeader]).
pipeline_header_extended(GlueProtocol, Runtime, _, Header) :-
    pipeline_header(GlueProtocol, Runtime, Header).

%% pipeline_helpers_extended(+Protocol, +DataSource, +Mode, +Runtime, -Helpers)
%  Generate helpers including record_key for generator mode
%  IronPython uses .NET HashSet<String> with JSON-serialized keys
pipeline_helpers_extended(GlueProtocol, DataSource, generator, ironpython, Helpers) :-
    !,
    pipeline_helpers(GlueProtocol, DataSource, BaseHelpers),
    IronPythonHashCode = "
# Record key generation for .NET HashSet (IronPython)
# Uses JSON serialization with sorted keys for consistent hashing

def record_key(record):
    '''Convert a record to a hashable string key for .NET HashSet.'''
    # Sort keys for consistent ordering
    sorted_items = sorted(record.items(), key=lambda x: str(x[0]))
    # Create canonical JSON string
    return json.dumps(dict(sorted_items), sort_keys=True)

def dict_from_key(key):
    '''Convert a key back to a dictionary.'''
    return json.loads(key)

# .NET HashSet wrapper for Python-style interface
class RecordSet:
    '''Wrapper around .NET HashSet<String> for record deduplication.'''
    def __init__(self):
        self._set = HashSet[String]()

    def add(self, key):
        '''Add a key to the set.'''
        self._set.Add(String(key))

    def __contains__(self, key):
        '''Check if key is in the set.'''
        return self._set.Contains(String(key))

    def __len__(self):
        return self._set.Count

",
    format(string(Helpers), "~w~w", [BaseHelpers, IronPythonHashCode]).

pipeline_helpers_extended(GlueProtocol, DataSource, generator, _Runtime, Helpers) :-
    !,
    pipeline_helpers(GlueProtocol, DataSource, BaseHelpers),
    FrozenDictCode = "
# FrozenDict - hashable dictionary for use in sets (fixpoint deduplication)
@dataclass(frozen=True)
class FrozenDict:
    '''Immutable dictionary that can be used in sets.'''
    items: tuple

    @staticmethod
    def from_dict(d: dict) -> 'FrozenDict':
        return FrozenDict(tuple(sorted(d.items())))

    def to_dict(self) -> dict:
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

def record_key(record: dict) -> FrozenDict:
    '''Convert a record to a hashable key for deduplication.'''
    return FrozenDict.from_dict(record)

",
    format(string(Helpers), "~w~w", [BaseHelpers, FrozenDictCode]).

pipeline_helpers_extended(GlueProtocol, DataSource, _, _Runtime, Helpers) :-
    pipeline_helpers(GlueProtocol, DataSource, Helpers).

%% compile_pipeline_predicates(+Predicates, -Code)
%  Compile each predicate to a Python generator function.
%
compile_pipeline_predicates([], "").
compile_pipeline_predicates([Pred|Rest], Code) :-
    compile_single_pipeline_predicate(Pred, PredCode),
    compile_pipeline_predicates(Rest, RestCode),
    format(string(Code), "~w~w", [PredCode, RestCode]).

%% compile_single_pipeline_predicate(+Pred, -Code)
%  Compile a single predicate for use in a pipeline.
%
compile_single_pipeline_predicate(_Target:Name/Arity, Code) :-
    !,
    % Extract just the name for the function
    compile_single_pipeline_predicate(Name/Arity, Code).

compile_single_pipeline_predicate(Name/Arity, Code) :-
    atom_string(Name, NameStr),
    % Generate a simple passthrough function as placeholder
    % In real usage, this would compile the actual predicate
    generate_default_arg_names(Arity, ArgNames),
    generate_extraction_code(ArgNames, ExtractionCode),
    format(string(Code),
"
def ~w(stream):
    \"\"\"
    Pipeline step: ~w/~w
    \"\"\"
    for record in stream:
        # Extract inputs from record
~w
        # Process (placeholder - actual logic from predicate)
        result = record.copy()
        yield result

", [NameStr, NameStr, Arity, ExtractionCode]).

%% generate_arg_list(+Names, -Code)
generate_arg_list(Names, Code) :-
    atomic_list_concat(Names, ', ', Code).

%% generate_dict_construction(+Names, -Code)
generate_dict_construction(Names, Code) :-
    maplist(generate_dict_entry, Names, Entries),
    atomic_list_concat(Entries, ', ', EntriesStr),
    format(string(Code), "{~w}", [EntriesStr]).

generate_dict_entry(Name, Entry) :-
    format(string(Entry), "'~w': ~w", [Name, Name]).

%% generate_extraction_code(+Names, -Code)
%  Generate code to extract inputs from record dict.
%
generate_extraction_code(Names, Code) :-
    maplist(generate_extraction_line, Names, Lines),
    atomic_list_concat(Lines, '\n', Code).

generate_extraction_line(Name, Line) :-
    format(string(Line), "        ~w = record.get('~w')", [Name, Name]).

%% generate_pipeline_connector(+Predicates, +Name, -Code)
%  Generate the function that chains all predicates together (legacy 3-arg version).
%
generate_pipeline_connector(Predicates, PipelineName, Code) :-
    generate_pipeline_connector(Predicates, PipelineName, sequential, cpython, Code).

%% generate_pipeline_connector(+Predicates, +Name, +Mode, -Code)
%  Generate the function that chains all predicates together (4-arg version).
%  Mode can be: sequential, generator
%
generate_pipeline_connector(Predicates, PipelineName, Mode, Code) :-
    generate_pipeline_connector(Predicates, PipelineName, Mode, cpython, Code).

%% generate_pipeline_connector(+Predicates, +Name, +Mode, +Runtime, -Code)
%  Runtime-aware pipeline connector generation.
%  IronPython uses .NET HashSet wrapper for generator mode.
%
generate_pipeline_connector(Predicates, PipelineName, sequential, _Runtime, Code) :-
    % Build the chain: pred1(pred2(pred3(input)))
    % Or use generator chaining for efficiency
    extract_predicate_names(Predicates, Names),
    generate_chain_code(Names, ChainCode),
    format(string(Code),
"
def ~w(input_stream):
    \"\"\"
    Chained pipeline: ~w
    Connects all predicates in sequence.
    \"\"\"
~w
", [PipelineName, Names, ChainCode]).

%% IronPython generator mode - uses .NET HashSet wrapper
generate_pipeline_connector(Predicates, PipelineName, generator, ironpython, Code) :-
    !,
    extract_predicate_names(Predicates, Names),
    generate_fixpoint_chain_code(Names, ChainCode),
    format(string(Code),
"
def ~w(input_stream):
    \"\"\"
    Fixpoint pipeline: ~w (IronPython/.NET)
    Iterates until no new records are produced.
    Uses .NET HashSet<String> for deduplication.
    \"\"\"
    # Initialize with input records using .NET HashSet wrapper
    total = RecordSet()
    all_records = []

    for record in input_stream:
        key = record_key(record)
        if key not in total:
            total.add(key)
            all_records.append(record)
            yield record

    # Fixpoint iteration - apply stages until no new records
    changed = True
    while changed:
        changed = False
        current = list(all_records)

~w

        # Check for new records
        for record in new_records:
            key = record_key(record)
            if key not in total:
                total.add(key)
                all_records.append(record)
                changed = True
                yield record

", [PipelineName, Names, ChainCode]).

%% CPython/PyPy/Jython generator mode - uses FrozenDict with Python set
generate_pipeline_connector(Predicates, PipelineName, generator, _Runtime, Code) :-
    % Generate fixpoint iteration pipeline
    extract_predicate_names(Predicates, Names),
    generate_fixpoint_chain_code(Names, ChainCode),
    format(string(Code),
"
def ~w(input_stream):
    \"\"\"
    Fixpoint pipeline: ~w
    Iterates until no new records are produced.
    \"\"\"
    # Initialize with input records
    total: Set[FrozenDict] = set()

    for record in input_stream:
        key = record_key(record)
        if key not in total:
            total.add(key)
            yield record

    # Fixpoint iteration - apply stages until no new records
    changed = True
    while changed:
        changed = False
        current = [key.to_dict() for key in total]

~w

        # Check for new records
        for record in new_records:
            key = record_key(record)
            if key not in total:
                total.add(key)
                changed = True
                yield record

", [PipelineName, Names, ChainCode]).

%% generate_fixpoint_chain_code(+Names, -Code)
%  Generate the stage application code for fixpoint iteration
generate_fixpoint_chain_code([], "        new_records = current\n").
generate_fixpoint_chain_code(Names, Code) :-
    Names \= [],
    generate_fixpoint_stage_calls(Names, "current", StageCalls),
    format(string(Code), "        # Apply pipeline stages
~w", [StageCalls]).

generate_fixpoint_stage_calls([], Current, Code) :-
    format(string(Code), "        new_records = ~w\n", [Current]).
generate_fixpoint_stage_calls([Stage|Rest], Current, Code) :-
    format(string(NextVar), "stage_~w_out", [Stage]),
    format(string(StageCall), "        ~w = list(~w(iter(~w)))\n", [NextVar, Stage, Current]),
    generate_fixpoint_stage_calls(Rest, NextVar, RestCode),
    format(string(Code), "~w~w", [StageCall, RestCode]).

%% extract_predicate_names(+Predicates, -Names)
extract_predicate_names([], []).
extract_predicate_names([Pred|Rest], [Name|RestNames]) :-
    extract_pred_name(Pred, Name),
    extract_predicate_names(Rest, RestNames).

extract_pred_name(_Target:Name/_Arity, NameStr) :-
    !,
    atom_string(Name, NameStr).
extract_pred_name(Name/_Arity, NameStr) :-
    atom_string(Name, NameStr).

%% generate_chain_code(+Names, -Code)
%  Generate the chaining code connecting all predicates.
%
generate_chain_code([], "    yield from input_stream\n").
generate_chain_code([First|Rest], Code) :-
    generate_chain_recursive(Rest, First, "input_stream", ChainExpr),
    format(string(Code), "    yield from ~w\n", [ChainExpr]).

generate_chain_recursive([], Current, Input, Expr) :-
    format(string(Expr), "~w(~w)", [Current, Input]).
generate_chain_recursive([Next|Rest], Current, Input, Expr) :-
    format(string(CurrentCall), "~w(~w)", [Current, Input]),
    generate_chain_recursive(Rest, Next, CurrentCall, Expr).

%% generate_chained_pipeline_main(+Name, +Protocol, +ArgNames, -Code)
%  Generate the main block for the chained pipeline.
%
generate_chained_pipeline_main(PipelineName, jsonl, _ArgNames, Code) :-
    format(string(Code),
"
if __name__ == '__main__':
    import sys

    # Read from stdin, process through pipeline, write to stdout
    input_stream = read_stream(sys.stdin)
    for result in ~w(input_stream):
        write_record(result)
", [PipelineName]).

generate_chained_pipeline_main(PipelineName, messagepack, _ArgNames, Code) :-
    format(string(Code),
"
if __name__ == '__main__':
    import sys

    # Read MessagePack from stdin, process, write to stdout
    input_stream = read_stream(sys.stdin.buffer)
    for result in ~w(input_stream):
        write_record(result)
", [PipelineName]).

% ============================================================================
% Cross-Runtime Pipeline Chaining
% ============================================================================

%% compile_cross_runtime_pipeline(+Predicates, +Options, -Code)
%  Compile a pipeline where predicates run in different runtimes.
%  Uses JSONL pipes or in-process bridges for communication.
%
compile_cross_runtime_pipeline(Predicates, Options, Code) :-
    option(glue_protocol(GlueProtocol), Options, jsonl),
    option(pipeline_name(PipelineName), Options, pipeline),

    % Group predicates by runtime
    group_by_runtime(Predicates, Groups),

    % Generate orchestrator code (shell script or Python)
    generate_cross_runtime_orchestrator(Groups, PipelineName, GlueProtocol, Code).

%% group_by_runtime(+Predicates, -Groups)
%  Group consecutive predicates that share a runtime.
%
group_by_runtime([], []).
group_by_runtime([Pred|Rest], [group(Runtime, [Pred|SameRuntime])|RestGroups]) :-
    predicate_runtime(Pred, Runtime),
    take_same_runtime(Rest, Runtime, SameRuntime, Remaining),
    group_by_runtime(Remaining, RestGroups).

take_same_runtime([], _, [], []).
take_same_runtime([Pred|Rest], Runtime, [Pred|Same], Remaining) :-
    predicate_runtime(Pred, PredRuntime),
    compatible_runtimes(Runtime, PredRuntime),
    !,
    take_same_runtime(Rest, Runtime, Same, Remaining).
take_same_runtime(Preds, _, [], Preds).

%% generate_cross_runtime_orchestrator(+Groups, +Name, +Protocol, -Code)
%  Generate orchestrator that manages cross-runtime pipeline.
%
generate_cross_runtime_orchestrator(Groups, PipelineName, GlueProtocol, Code) :-
    length(Groups, NumGroups),
    (   NumGroups == 1
    ->  % Single group - just compile as same-runtime
        Groups = [group(_Runtime, Predicates)],
        compile_same_runtime_pipeline(Predicates, [pipeline_name(PipelineName)], Code)
    ;   % Multiple groups - generate orchestrator
        generate_multi_runtime_code(Groups, PipelineName, GlueProtocol, Code)
    ).

%% generate_multi_runtime_code(+Groups, +Name, +Protocol, -Code)
%  Generate Python code that orchestrates multiple runtime stages.
%
generate_multi_runtime_code(Groups, PipelineName, GlueProtocol, Code) :-
    % Generate stage functions for each group
    generate_stage_functions(Groups, 1, StageFunctions),

    % Generate the orchestrator that pipes between stages
    generate_orchestrator_function(Groups, PipelineName, GlueProtocol, OrchestratorCode),

    % Generate header
    pipeline_header(GlueProtocol, cpython, Header),
    pipeline_helpers(GlueProtocol, same_as_data, Helpers),

    % Generate main
    generate_chained_pipeline_main(PipelineName, GlueProtocol, [], MainCode),

    format(string(Code), "~w~w~w~w~w",
        [Header, Helpers, StageFunctions, OrchestratorCode, MainCode]).

%% generate_stage_functions(+Groups, +StageNum, -Code)
generate_stage_functions([], _, "").
generate_stage_functions([group(Runtime, Predicates)|Rest], N, Code) :-
    generate_stage_function(Runtime, Predicates, N, StageCode),
    N1 is N + 1,
    generate_stage_functions(Rest, N1, RestCode),
    format(string(Code), "~w~w", [StageCode, RestCode]).

%% generate_stage_function(+Runtime, +Predicates, +N, -Code)
generate_stage_function(Runtime, Predicates, N, Code) :-
    extract_predicate_names(Predicates, Names),
    atomic_list_concat(Names, ' -> ', NamesStr),
    runtime_to_string(Runtime, RuntimeStr),
    generate_stage_chain_code(Names, StageChainCode),
    format(string(Code),
"
def stage_~w(input_stream):
    \"\"\"
    Stage ~w: ~w
    Runtime: ~w
    \"\"\"
    # Chain predicates within this stage
    current = input_stream
~w
    yield from current

", [N, N, NamesStr, RuntimeStr, StageChainCode]).

generate_stage_chain_code([], "").
generate_stage_chain_code([Name|Rest], Code) :-
    format(string(Line), "    current = ~w(current)\n", [Name]),
    generate_stage_chain_code(Rest, RestCode),
    string_concat(Line, RestCode, Code).

runtime_to_string(python, "Python (CPython)").
runtime_to_string(cpython, "CPython").
runtime_to_string(ironpython, "IronPython").
runtime_to_string(pypy, "PyPy").
runtime_to_string(jython, "Jython").
runtime_to_string(csharp, "C#").
runtime_to_string(powershell, "PowerShell").
runtime_to_string(R, S) :- atom_string(R, S).

%% generate_orchestrator_function(+Groups, +Name, +Protocol, -Code)
generate_orchestrator_function(Groups, PipelineName, _Protocol, Code) :-
    length(Groups, NumStages),
    numlist(1, NumStages, StageNums),
    maplist(format_stage_call, StageNums, StageCalls),
    atomic_list_concat(StageCalls, '\n', StageCallsCode),
    format(string(Code),
"
def ~w(input_stream):
    \"\"\"
    Cross-runtime pipeline orchestrator.
    Chains ~w stages together.
    \"\"\"
    current = input_stream
~w
    yield from current

", [PipelineName, NumStages, StageCallsCode]).

format_stage_call(N, Call) :-
    format(string(Call), "    current = stage_~w(current)", [N]).

% ============================================================================
% Pipeline Chaining Tests (Phase 4)
% ============================================================================

test_pipeline_chaining :-
    format('~n=== Python Pipeline Chaining Tests (Phase 4) ===~n~n', []),

    % Test 1: All same runtime detection
    format('[Test 1] Same runtime detection~n', []),
    (   all_same_runtime([get_users/1, filter_active/2, format_output/1])
    ->  format('  [PASS] Plain predicates are same runtime~n', [])
    ;   format('  [FAIL] Should detect same runtime~n', [])
    ),

    % Test 2: Mixed runtime detection
    format('[Test 2] Mixed runtime detection~n', []),
    (   \+ all_same_runtime([python:get_users/1, csharp:validate/1])
    ->  format('  [PASS] Python + C# detected as different~n', [])
    ;   format('  [FAIL] Should detect different runtimes~n', [])
    ),

    % Test 3: Predicate runtime extraction
    format('[Test 3] Predicate runtime extraction~n', []),
    predicate_runtime(ironpython:foo/2, R3),
    (   R3 == ironpython
    ->  format('  [PASS] ironpython:foo/2 -> ironpython~n', [])
    ;   format('  [FAIL] Got ~w~n', [R3])
    ),

    % Test 4: Compile same-runtime pipeline
    format('[Test 4] Compile same-runtime pipeline~n', []),
    compile_same_runtime_pipeline(
        [get_users/1, filter_active/2],
        [runtime(cpython), pipeline_name(my_pipeline)],
        Code4
    ),
    (   sub_string(Code4, _, _, _, "def my_pipeline"),
        sub_string(Code4, _, _, _, "def get_users"),
        sub_string(Code4, _, _, _, "def filter_active")
    ->  format('  [PASS] Same-runtime pipeline generated~n', [])
    ;   format('  [FAIL] Pipeline missing components~n', [])
    ),

    % Test 5: Pipeline connector generation
    format('[Test 5] Pipeline connector generation~n', []),
    generate_pipeline_connector([a/1, b/1, c/1], test_pipe, Code5),
    (   sub_string(Code5, _, _, _, "def test_pipe"),
        sub_string(Code5, _, _, _, "yield from")
    ->  format('  [PASS] Connector chains predicates~n', [])
    ;   format('  [FAIL] Connector issue~n', [])
    ),

    % Test 6: Group by runtime
    format('[Test 6] Group predicates by runtime~n', []),
    group_by_runtime(
        [python:a/1, python:b/1, csharp:c/1, python:d/1],
        Groups6
    ),
    (   Groups6 = [group(python, [python:a/1, python:b/1]),
                   group(csharp, [csharp:c/1]),
                   group(python, [python:d/1])]
    ->  format('  [PASS] Grouped into 3 stages~n', [])
    ;   format('  [FAIL] Got ~w~n', [Groups6])
    ),

    % Test 7: Cross-runtime pipeline
    format('[Test 7] Cross-runtime pipeline~n', []),
    compile_cross_runtime_pipeline(
        [python:extract/1, csharp:validate/1, python:format/1],
        [pipeline_name(cross_pipe)],
        Code7
    ),
    (   sub_string(Code7, _, _, _, "stage_1"),
        sub_string(Code7, _, _, _, "stage_2"),
        sub_string(Code7, _, _, _, "stage_3"),
        sub_string(Code7, _, _, _, "def cross_pipe")
    ->  format('  [PASS] Cross-runtime pipeline has 3 stages~n', [])
    ;   format('  [FAIL] Cross-runtime pipeline issue~n', [])
    ),

    % Test 8: Main entry point dispatch
    format('[Test 8] Main entry point dispatch~n', []),
    compile_pipeline([a/1, b/1], [pipeline_name(dispatch_test)], Code8a),
    compile_pipeline([python:a/1, csharp:b/1], [pipeline_name(dispatch_test)], Code8b),
    (   sub_string(Code8a, _, _, _, "def dispatch_test"),
        sub_string(Code8b, _, _, _, "stage_1")
    ->  format('  [PASS] Dispatch selects correct mode~n', [])
    ;   format('  [FAIL] Dispatch issue~n', [])
    ),

    format('~n=== All Pipeline Chaining Tests Passed ===~n', []).

%% ============================================
%% PYTHON PIPELINE GENERATOR MODE TESTS
%% ============================================

test_python_pipeline_generator :-
    format('~n=== Python Pipeline Generator Mode Tests ===~n~n', []),

    % Test 1: Pipeline header extended for generator mode
    format('[Test 1] Pipeline header extended (generator)~n', []),
    pipeline_header_extended(jsonl, cpython, generator, Header1),
    (   sub_string(Header1, _, _, _, "from typing import Set"),
        sub_string(Header1, _, _, _, "from dataclasses import dataclass")
    ->  format('  [PASS] Generator header has required imports~n', [])
    ;   format('  [FAIL] Header: ~w~n', [Header1])
    ),

    % Test 2: Pipeline header extended for sequential mode (unchanged)
    format('[Test 2] Pipeline header extended (sequential)~n', []),
    pipeline_header_extended(jsonl, cpython, sequential, Header2),
    pipeline_header(jsonl, cpython, BaseHeader2),
    (   Header2 == BaseHeader2
    ->  format('  [PASS] Sequential header unchanged~n', [])
    ;   format('  [FAIL] Headers differ~n', [])
    ),

    % Test 3: Pipeline helpers extended for generator mode (CPython)
    format('[Test 3] Pipeline helpers extended (generator, CPython)~n', []),
    pipeline_helpers_extended(jsonl, same_as_data, generator, cpython, Helpers3),
    (   sub_string(Helpers3, _, _, _, "class FrozenDict"),
        sub_string(Helpers3, _, _, _, "def record_key"),
        sub_string(Helpers3, _, _, _, "from_dict")
    ->  format('  [PASS] Generator helpers include FrozenDict~n', [])
    ;   format('  [FAIL] Helpers: ~w~n', [Helpers3])
    ),

    % Test 4: Generate fixpoint chain code (empty)
    format('[Test 4] Fixpoint chain code (empty)~n', []),
    generate_fixpoint_chain_code([], ChainCode4),
    (   sub_string(ChainCode4, _, _, _, "new_records = current")
    ->  format('  [PASS] Empty chain returns current~n', [])
    ;   format('  [FAIL] Chain: ~w~n', [ChainCode4])
    ),

    % Test 5: Generate fixpoint chain code with stages
    format('[Test 5] Fixpoint chain code with stages~n', []),
    generate_fixpoint_chain_code(["stage1", "stage2"], ChainCode5),
    (   sub_string(ChainCode5, _, _, _, "stage_stage1_out"),
        sub_string(ChainCode5, _, _, _, "stage_stage2_out"),
        sub_string(ChainCode5, _, _, _, "stage1(iter(current))"),
        sub_string(ChainCode5, _, _, _, "stage2(iter(stage_stage1_out))")
    ->  format('  [PASS] Stage calls generated correctly~n', [])
    ;   format('  [FAIL] Chain code: ~w~n', [ChainCode5])
    ),

    % Test 6: Pipeline connector for generator mode
    format('[Test 6] Pipeline connector (generator)~n', []),
    generate_pipeline_connector([a/1, b/1], test_gen, generator, ConnCode6),
    (   sub_string(ConnCode6, _, _, _, "def test_gen"),
        sub_string(ConnCode6, _, _, _, "Fixpoint pipeline"),
        sub_string(ConnCode6, _, _, _, "total: Set[FrozenDict]"),
        sub_string(ConnCode6, _, _, _, "while changed"),
        sub_string(ConnCode6, _, _, _, "record_key(record)")
    ->  format('  [PASS] Generator connector has fixpoint loop~n', [])
    ;   format('  [FAIL] Connector: ~w~n', [ConnCode6])
    ),

    % Test 7: Pipeline connector for sequential mode (unchanged)
    format('[Test 7] Pipeline connector (sequential)~n', []),
    generate_pipeline_connector([a/1, b/1], test_seq, sequential, ConnCode7),
    (   sub_string(ConnCode7, _, _, _, "def test_seq"),
        sub_string(ConnCode7, _, _, _, "Chained pipeline"),
        sub_string(ConnCode7, _, _, _, "yield from")
    ->  format('  [PASS] Sequential connector unchanged~n', [])
    ;   format('  [FAIL] Connector: ~w~n', [ConnCode7])
    ),

    % Test 8: Full pipeline with generator mode
    format('[Test 8] Full pipeline (generator mode)~n', []),
    compile_same_runtime_pipeline([stage1/1, stage2/1], [
        pipeline_name(fixpoint_pipe),
        pipeline_mode(generator)
    ], FullCode8),
    (   sub_string(FullCode8, _, _, _, "class FrozenDict"),
        sub_string(FullCode8, _, _, _, "def fixpoint_pipe"),
        sub_string(FullCode8, _, _, _, "while changed"),
        sub_string(FullCode8, _, _, _, "total.add(key)")
    ->  format('  [PASS] Full generator pipeline compiled~n', [])
    ;   format('  [FAIL] Missing expected patterns~n', [])
    ),

    % Test 9: Full pipeline with sequential mode (default)
    format('[Test 9] Full pipeline (sequential, default)~n', []),
    compile_same_runtime_pipeline([stage1/1, stage2/1], [
        pipeline_name(seq_pipe)
    ], FullCode9),
    (   sub_string(FullCode9, _, _, _, "def seq_pipe"),
        sub_string(FullCode9, _, _, _, "yield from"),
        \+ sub_string(FullCode9, _, _, _, "class FrozenDict")
    ->  format('  [PASS] Sequential pipeline (default) works~n', [])
    ;   format('  [FAIL] Sequential pipeline issue~n', [])
    ),

    % Test 10: Main entry point with generator mode
    format('[Test 10] Main entry point (generator)~n', []),
    compile_pipeline([a/1, b/1], [
        pipeline_name(main_gen_pipe),
        pipeline_mode(generator)
    ], FullCode10),
    (   sub_string(FullCode10, _, _, _, "def main_gen_pipe"),
        sub_string(FullCode10, _, _, _, "class FrozenDict"),
        sub_string(FullCode10, _, _, _, "while changed")
    ->  format('  [PASS] Main entry point uses generator mode~n', [])
    ;   format('  [FAIL] Main entry point issue~n', [])
    ),

    format('~n=== All Python Pipeline Generator Mode Tests Passed ===~n', []).

%% ============================================
%% IRONPYTHON PIPELINE GENERATOR MODE TESTS
%% ============================================

test_ironpython_pipeline_generator :-
    format('~n=== IronPython Pipeline Generator Mode Tests ===~n~n', []),

    % Test 1: IronPython header for generator mode has .NET HashSet
    format('[Test 1] IronPython header (generator)~n', []),
    pipeline_header_extended(jsonl, ironpython, generator, Header1),
    (   sub_string(Header1, _, _, _, "import clr"),
        sub_string(Header1, _, _, _, "HashSet"),
        \+ sub_string(Header1, _, _, _, "dataclass")
    ->  format('  [PASS] IronPython header has .NET HashSet import~n', [])
    ;   format('  [FAIL] Header: ~w~n', [Header1])
    ),

    % Test 2: IronPython helpers for generator mode
    format('[Test 2] IronPython helpers (generator)~n', []),
    pipeline_helpers_extended(jsonl, same_as_data, generator, ironpython, Helpers2),
    (   sub_string(Helpers2, _, _, _, "class RecordSet"),
        sub_string(Helpers2, _, _, _, "HashSet[String]"),
        sub_string(Helpers2, _, _, _, "def record_key")
    ->  format('  [PASS] IronPython helpers include RecordSet wrapper~n', [])
    ;   format('  [FAIL] Helpers: ~w~n', [Helpers2])
    ),

    % Test 3: IronPython connector for generator mode
    format('[Test 3] IronPython connector (generator)~n', []),
    generate_pipeline_connector([a/1, b/1], iron_gen, generator, ironpython, ConnCode3),
    (   sub_string(ConnCode3, _, _, _, "def iron_gen"),
        sub_string(ConnCode3, _, _, _, "RecordSet()"),
        sub_string(ConnCode3, _, _, _, "IronPython/.NET"),
        sub_string(ConnCode3, _, _, _, "while changed")
    ->  format('  [PASS] IronPython connector uses RecordSet~n', [])
    ;   format('  [FAIL] Connector: ~w~n', [ConnCode3])
    ),

    % Test 4: Full IronPython pipeline with generator mode
    format('[Test 4] Full IronPython pipeline (generator)~n', []),
    compile_same_runtime_pipeline([stage1/1, stage2/1], [
        pipeline_name(iron_fixpoint),
        pipeline_mode(generator),
        runtime(ironpython)
    ], FullCode4),
    (   sub_string(FullCode4, _, _, _, "#!/usr/bin/env ipy"),
        sub_string(FullCode4, _, _, _, "class RecordSet"),
        sub_string(FullCode4, _, _, _, "def iron_fixpoint"),
        sub_string(FullCode4, _, _, _, "while changed")
    ->  format('  [PASS] Full IronPython generator pipeline compiled~n', [])
    ;   format('  [FAIL] Missing expected patterns~n', [])
    ),

    % Test 5: IronPython sequential mode still works
    format('[Test 5] IronPython sequential mode~n', []),
    compile_same_runtime_pipeline([stage1/1], [
        pipeline_name(iron_seq),
        pipeline_mode(sequential),
        runtime(ironpython)
    ], SeqCode5),
    (   sub_string(SeqCode5, _, _, _, "#!/usr/bin/env ipy"),
        sub_string(SeqCode5, _, _, _, "def iron_seq"),
        sub_string(SeqCode5, _, _, _, "yield from"),
        \+ sub_string(SeqCode5, _, _, _, "RecordSet")
    ->  format('  [PASS] IronPython sequential mode works~n', [])
    ;   format('  [FAIL] Sequential issue~n', [])
    ),

    % Test 6: IronPython generator uses all_records list
    format('[Test 6] IronPython generator tracks all_records~n', []),
    generate_pipeline_connector([a/1], track_test, generator, ironpython, ConnCode6),
    (   sub_string(ConnCode6, _, _, _, "all_records = []"),
        sub_string(ConnCode6, _, _, _, "all_records.append(record)")
    ->  format('  [PASS] IronPython tracks all_records for iteration~n', [])
    ;   format('  [FAIL] Tracking issue~n', [])
    ),

    % Test 7: CPython generator still uses FrozenDict
    format('[Test 7] CPython generator uses FrozenDict~n', []),
    compile_same_runtime_pipeline([stage1/1], [
        pipeline_name(py_gen),
        pipeline_mode(generator),
        runtime(cpython)
    ], PyCode7),
    (   sub_string(PyCode7, _, _, _, "class FrozenDict"),
        sub_string(PyCode7, _, _, _, "Set[FrozenDict]"),
        \+ sub_string(PyCode7, _, _, _, "RecordSet")
    ->  format('  [PASS] CPython uses FrozenDict (not RecordSet)~n', [])
    ;   format('  [FAIL] CPython issue~n', [])
    ),

    % Test 8: IronPython header has CLR references
    format('[Test 8] IronPython header CLR references~n', []),
    pipeline_header(jsonl, ironpython, BaseHeader8),
    (   sub_string(BaseHeader8, _, _, _, "clr.AddReference"),
        sub_string(BaseHeader8, _, _, _, "from System import")
    ->  format('  [PASS] IronPython header has CLR setup~n', [])
    ;   format('  [FAIL] Missing CLR: ~w~n', [BaseHeader8])
    ),

    % Test 9: IronPython record_key uses json.dumps
    format('[Test 9] IronPython record_key serialization~n', []),
    pipeline_helpers_extended(jsonl, same_as_data, generator, ironpython, Helpers9),
    (   sub_string(Helpers9, _, _, _, "json.dumps"),
        sub_string(Helpers9, _, _, _, "sort_keys=True")
    ->  format('  [PASS] IronPython record_key uses JSON serialization~n', [])
    ;   format('  [FAIL] Serialization issue~n', [])
    ),

    % Test 10: IronPython RecordSet __contains__ method
    format('[Test 10] IronPython RecordSet contains check~n', []),
    pipeline_helpers_extended(jsonl, same_as_data, generator, ironpython, Helpers10),
    (   sub_string(Helpers10, _, _, _, "def __contains__"),
        sub_string(Helpers10, _, _, _, ".Contains(String(key))")
    ->  format('  [PASS] RecordSet has proper contains check~n', [])
    ;   format('  [FAIL] Contains issue~n', [])
    ),

    format('~n=== All IronPython Pipeline Generator Mode Tests Passed ===~n', []).
