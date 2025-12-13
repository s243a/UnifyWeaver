% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% JVM Glue - Cross-target communication for JVM ecosystem
%
% This module generates glue code for:
% - Java ↔ Jython in-process communication
% - Java ↔ Scala direct method calls (future)
% - Java ↔ Kotlin direct method calls (future)
% - JVM process management for pipeline steps

:- encoding(utf8).

:- module(jvm_glue, [
    % Runtime detection
    detect_jvm_runtime/1,           % detect_jvm_runtime(-Runtime)
    detect_java_version/1,          % detect_java_version(-Version)
    detect_jython/1,                % detect_jython(-Available)
    
    % Transport selection
    jvm_transport_type/3,           % jvm_transport_type(+From, +To, -Transport)
    can_use_direct/2,               % can_use_direct(+From, +To)
    
    % Bridge generation
    generate_java_jython_bridge/3,  % generate_java_jython_bridge(+Steps, +Options, -Code)
    generate_jython_java_bridge/3,  % generate_jython_java_bridge(+Steps, +Options, -Code)
    
    % JVM process management
    generate_jvm_launcher/3,        % generate_jvm_launcher(+Steps, +Options, -ShellScript)
    generate_classpath/2,           % generate_classpath(+Options, -Classpath)
    
    % Pipeline orchestration
    generate_jvm_pipeline/3,        % generate_jvm_pipeline(+Steps, +Options, -Code)
    
    % Testing
    test_jvm_glue/0
]).

:- use_module(library(lists)).

% ============================================================================
% RUNTIME DETECTION
% ============================================================================

%% detect_jvm_runtime(-Runtime)
%  Detect available JVM runtime.
%  Runtime = jdk | jre | graalvm | none
detect_jvm_runtime(Runtime) :-
    % Try java command
    (   catch(
            (process_create(path(java), ['--version'], [stdout(pipe(S))]),
             read_line_to_string(S, VersionStr),
             close(S)),
            _, fail)
    ->  (   sub_string(VersionStr, _, _, _, 'GraalVM')
        ->  Runtime = graalvm
        ;   sub_string(VersionStr, _, _, _, 'openjdk')
        ->  Runtime = jdk
        ;   Runtime = jre
        )
    ;   Runtime = none
    ).

%% detect_java_version(-Version)
%  Detect Java version number (e.g., 17, 21, 25).
detect_java_version(Version) :-
    (   catch(
            (process_create(path(java), ['--version'], [stdout(pipe(S))]),
             read_line_to_string(S, VersionStr),
             close(S)),
            _, fail)
    ->  % Parse version from string like "openjdk 25 2025-03-18"
        (   sub_string(VersionStr, Before, _, _, ' '),
            Before > 0,
            sub_string(VersionStr, Before, _, _, Rest),
            sub_string(Rest, 1, _, _, Rest2),
            (   sub_string(Rest2, N, _, _, ' '),
                sub_string(Rest2, 0, N, _, VerAtom)
            ;   VerAtom = Rest2
            ),
            atom_number(VerAtom, Version)
        ->  true
        ;   Version = unknown
        )
    ;   Version = none
    ).

%% detect_jython(-Available)
%  Check if Jython is available.
detect_jython(Available) :-
    (   catch(
            (process_create(path(jython), ['--version'], [stdout(pipe(S))]),
             read_line_to_string(S, _),
             close(S)),
            _, fail)
    ->  Available = true
    ;   Available = false
    ).

% ============================================================================
% TRANSPORT SELECTION
% ============================================================================

%% jvm_transport_type(+From, +To, -Transport)
%  Determine transport type between two JVM targets.
%  Transport = direct | pipe | http
jvm_transport_type(From, To, Transport) :-
    (   can_use_direct(From, To)
    ->  Transport = direct
    ;   Transport = pipe
    ).

%% can_use_direct(+From, +To)
%  Check if direct in-process communication is possible.
%  All JVM languages can communicate directly via shared JVM.
can_use_direct(From, To) :-
    jvm_target(From),
    jvm_target(To).

%% jvm_target(+Target)
jvm_target(java).
jvm_target(jython).
jvm_target(scala).
jvm_target(kotlin).
jvm_target(clojure).

% ============================================================================
% JAVA → JYTHON BRIDGE
% ============================================================================

%% generate_java_jython_bridge(+Steps, +Options, -Code)
%  Generate Java code that calls Jython steps directly.
generate_java_jython_bridge(Steps, Options, Code) :-
    option(package(Package), Options, generated),
    option(class_name(ClassName), Options, 'JythonBridge'),
    
    % Generate step invocation code
    findall(StepCode,
        (   member(step(Name, jython, Script, _StepOpts), Steps),
            format(string(StepCode),
"    /**
     * Execute Jython step: ~w
     */
    public Map<String, Object> call_~w(Map<String, Object> record) {
        try {
            PythonInterpreter interp = new PythonInterpreter();
            interp.exec(loadScript(\"~w\"));
            PyObject processFunc = interp.get(\"process\");
            PyObject pyRecord = Py.java2py(record);
            PyObject result = processFunc.__call__(pyRecord);
            return (Map<String, Object>) result.__tojava__(Map.class);
        } catch (Exception e) {
            System.err.println(\"Jython step ~w failed: \" + e.getMessage());
            return record;
        }
    }
", [Name, Name, Script, Name])
        ),
        StepCodes),
    atomic_list_concat(StepCodes, '\n', AllStepCode),
    
    format(string(Code),
"// Generated by UnifyWeaver JVM Glue
// Java → Jython Bridge

package ~w;

import org.python.util.PythonInterpreter;
import org.python.core.*;
import java.util.*;
import java.io.*;
import java.nio.file.*;

/**
 * Bridge for calling Jython steps from Java.
 * Uses Jython's embedded interpreter for direct JVM communication.
 */
public class ~w {

    private static String loadScript(String path) throws IOException {
        return new String(Files.readAllBytes(Paths.get(path)));
    }

~w

    /**
     * Execute full pipeline with Jython steps.
     */
    public Map<String, Object> runPipeline(Map<String, Object> record) {
        Map<String, Object> current = new HashMap<>(record);
        // TODO: Chain step calls
        return current;
    }
}
", [Package, ClassName, AllStepCode]).

% ============================================================================
% JYTHON → JAVA BRIDGE
% ============================================================================

%% generate_jython_java_bridge(+Steps, +Options, -Code)
%  Generate Jython code that calls Java steps directly.
generate_jython_java_bridge(Steps, Options, Code) :-
    option(java_class(JavaClass), Options, 'generated.Pipeline'),
    
    % Generate step invocation code
    findall(StepCode,
        (   member(step(Name, java, _Script, _StepOpts), Steps),
            format(string(StepCode),
"def call_~w(record):
    '''Call Java step ~w via direct JVM bridge.'''
    try:
        from ~w import ~w
        java_processor = ~w()
        result = java_processor.process(record)
        if result.isPresent():
            return result.get()
        return None
    except Exception as e:
        print >> sys.stderr, 'Java step ~w failed:', str(e)
        return record
", [Name, Name, JavaClass, Name, Name, Name])
        ),
        StepCodes),
    atomic_list_concat(StepCodes, '\n', AllStepCode),
    
    format(string(Code),
"#!/usr/bin/env jython
# -*- coding: utf-8 -*-
# Generated by UnifyWeaver JVM Glue
# Jython → Java Bridge

from __future__ import print_function
import sys
import json

~w

def run_pipeline(record):
    '''Execute full pipeline with Java steps.'''
    current = dict(record)
    # TODO: Chain step calls
    return current

if __name__ == '__main__':
    # Test bridge
    test_record = {'test': 'data'}
    result = run_pipeline(test_record)
    print(json.dumps(result))
", [AllStepCode]).

% ============================================================================
% JVM PROCESS MANAGEMENT
% ============================================================================

%% generate_jvm_launcher(+Steps, +Options, -ShellScript)
%  Generate shell script to launch JVM pipeline.
generate_jvm_launcher(Steps, Options, ShellScript) :-
    option(main_class(MainClass), Options, 'generated.Pipeline'),
    option(java_opts(JavaOpts), Options, '-Xmx512m'),
    
    generate_classpath(Options, Classpath),
    
    % Count step types
    findall(T, member(step(_, T, _, _), Steps), Types),
    length(Types, NumSteps),
    
    format(string(ShellScript),
"#!/bin/bash
# Generated by UnifyWeaver JVM Glue
# JVM Pipeline Launcher

# Configuration
JAVA_OPTS=\"~w\"
CLASSPATH=\"~w\"
MAIN_CLASS=\"~w\"

# Number of steps: ~w

# Check Java
if ! command -v java &> /dev/null; then
    echo \"Error: Java not found\" >&2
    exit 1
fi

# Check Jython (if needed)
HAS_JYTHON_STEPS=false
# TODO: Detect Jython steps

if [ \"$HAS_JYTHON_STEPS\" = true ]; then
    if ! command -v jython &> /dev/null; then
        echo \"Warning: Jython not found, will use embedded interpreter\" >&2
    fi
fi

# Run pipeline
java $JAVA_OPTS -cp \"$CLASSPATH\" \"$MAIN_CLASS\" \"$@\"
", [JavaOpts, Classpath, MainClass, NumSteps]).

%% generate_classpath(+Options, -Classpath)
%  Generate classpath string from options.
generate_classpath(Options, Classpath) :-
    option(jars(Jars), Options, []),
    option(classes_dir(ClassesDir), Options, 'build/classes'),
    
    % Default dependencies
    DefaultJars = ['lib/gson.jar', 'lib/jython-standalone.jar'],
    append(DefaultJars, Jars, AllJars),
    
    atomic_list_concat([ClassesDir|AllJars], ':', Classpath).

% ============================================================================
% PIPELINE ORCHESTRATION
% ============================================================================

%% generate_jvm_pipeline(+Steps, +Options, -Code)
%  Generate orchestration code for mixed JVM pipeline.
generate_jvm_pipeline(Steps, Options, Code) :-
    % Separate steps by target
    partition(is_java_step, Steps, JavaSteps, OtherSteps),
    partition(is_jython_step, OtherSteps, JythonSteps, _),
    
    % Generate appropriate code based on mix
    (   JavaSteps \= [], JythonSteps \= []
    ->  % Mixed pipeline - generate Java host with Jython bridges
        generate_mixed_jvm_pipeline(JavaSteps, JythonSteps, Options, Code)
    ;   JavaSteps \= []
    ->  % Java only
        generate_java_only_pipeline(JavaSteps, Options, Code)
    ;   JythonSteps \= []
    ->  % Jython only
        generate_jython_only_pipeline(JythonSteps, Options, Code)
    ;   Code = "// No JVM steps found"
    ).

is_java_step(step(_, java, _, _)).
is_jython_step(step(_, jython, _, _)).

%% generate_mixed_jvm_pipeline(+JavaSteps, +JythonSteps, +Options, -Code)
generate_mixed_jvm_pipeline(JavaSteps, JythonSteps, Options, Code) :-
    length(JavaSteps, NumJava),
    length(JythonSteps, NumJython),
    
    option(package(Package), Options, generated),
    
    format(string(Code),
"// Generated by UnifyWeaver JVM Glue
// Mixed JVM Pipeline: ~w Java steps, ~w Jython steps

package ~w;

import org.python.util.PythonInterpreter;
import org.python.core.*;
import java.util.*;
import java.io.*;

/**
 * Mixed JVM pipeline orchestrator.
 * Uses direct JVM calls for all step communication.
 */
public class MixedPipeline {

    private final PythonInterpreter jython = new PythonInterpreter();

    public MixedPipeline() {
        // Initialize Jython interpreter
        jython.exec(\"import sys\");
        jython.exec(\"import json\");
    }

    /**
     * Run the full mixed pipeline.
     */
    public Optional<Map<String, Object>> process(Map<String, Object> record) {
        Map<String, Object> current = new HashMap<>(record);
        
        // TODO: Execute steps in order
        // Each step can be Java or Jython, both run in same JVM
        
        return Optional.of(current);
    }

    public static void main(String[] args) throws Exception {
        MixedPipeline pipeline = new MixedPipeline();
        
        BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
        String line;
        while ((line = reader.readLine()) != null) {
            if (line.trim().isEmpty()) continue;
            
            @SuppressWarnings(\"unchecked\")
            Map<String, Object> record = new com.google.gson.Gson()
                .fromJson(line, Map.class);
            
            Optional<Map<String, Object>> result = pipeline.process(record);
            result.ifPresent(r -> System.out.println(
                new com.google.gson.Gson().toJson(r)));
        }
    }
}
", [NumJava, NumJython, Package]).

%% generate_java_only_pipeline(+Steps, +Options, -Code)
generate_java_only_pipeline(Steps, _Options, Code) :-
    length(Steps, NumSteps),
    format(string(Code), "// Java-only pipeline with ~w steps\n// Use java_target.pl directly", [NumSteps]).

%% generate_jython_only_pipeline(+Steps, +Options, -Code)
generate_jython_only_pipeline(Steps, _Options, Code) :-
    length(Steps, NumSteps),
    format(string(Code), "# Jython-only pipeline with ~w steps\n# Use jython_target.pl directly", [NumSteps]).

% ============================================================================
% UTILITY PREDICATES
% ============================================================================

%% option(+Option, +Options, +Default)
option(Option, Options, _Default) :-
    member(Option, Options),
    !.
option(Option, _Options, Default) :-
    Option =.. [_, Default].

% ============================================================================
% TESTS
% ============================================================================

test_jvm_glue :-
    format('~n=== Testing JVM Glue ===~n~n'),
    
    % Test 1: Runtime detection
    format('[Test 1] JVM runtime detection~n'),
    (   detect_jvm_runtime(Runtime)
    ->  format('  [INFO] Runtime: ~w~n', [Runtime])
    ;   format('  [INFO] No JVM detected~n')
    ),
    
    % Test 2: Java version
    format('~n[Test 2] Java version detection~n'),
    (   detect_java_version(Version)
    ->  format('  [INFO] Version: ~w~n', [Version])
    ;   format('  [INFO] Could not detect~n')
    ),
    
    % Test 3: Transport selection  
    format('~n[Test 3] Transport selection~n'),
    jvm_transport_type(java, jython, T1),
    format('  [INFO] java → jython: ~w~n', [T1]),
    jvm_transport_type(java, python, T2),
    format('  [INFO] java → python: ~w~n', [T2]),
    
    % Test 4: Bridge generation
    format('~n[Test 4] Java → Jython bridge generation~n'),
    Steps = [step(process_data, jython, 'process.py', [])],
    generate_java_jython_bridge(Steps, [package(test)], BridgeCode),
    (   sub_atom(BridgeCode, _, _, _, 'PythonInterpreter')
    ->  format('  [PASS] Bridge uses PythonInterpreter~n')
    ;   format('  [FAIL] Missing PythonInterpreter~n')
    ),
    
    % Test 5: Launcher generation
    format('~n[Test 5] Launcher generation~n'),
    generate_jvm_launcher(Steps, [main_class('test.Main')], LauncherCode),
    (   sub_atom(LauncherCode, _, _, _, '#!/bin/bash')
    ->  format('  [PASS] Generated bash launcher~n')
    ;   format('  [FAIL] Invalid launcher~n')
    ),
    
    format('~n=== JVM Glue Tests Complete ===~n').
