% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% JVM Glue - Cross-target communication for JVM ecosystem
%
% This module generates glue code for in-process JVM communication:
% - Java ↔ Jython (embedded Python interpreter)
% - Java ↔ Scala (direct interop - same bytecode)
% - Java ↔ Kotlin (direct interop - same bytecode)
% - Java ↔ Clojure (via clojure.java.api)
% - Scala ↔ Kotlin (direct interop)
% - JVM process management for pipeline steps

:- encoding(utf8).

:- module(jvm_glue, [
    % Runtime detection
    detect_jvm_runtime/1,           % detect_jvm_runtime(-Runtime)
    detect_java_version/1,          % detect_java_version(-Version)
    detect_jython/1,                % detect_jython(-Available)
    detect_scala/1,                 % detect_scala(-Available)
    detect_kotlin/1,                % detect_kotlin(-Available)
    detect_clojure/1,               % detect_clojure(-Available)

    % Transport selection
    jvm_transport_type/3,           % jvm_transport_type(+From, +To, -Transport)
    can_use_direct/2,               % can_use_direct(+From, +To)

    % Bridge generation - Java ↔ Jython
    generate_java_jython_bridge/3,  % generate_java_jython_bridge(+Steps, +Options, -Code)
    generate_jython_java_bridge/3,  % generate_jython_java_bridge(+Steps, +Options, -Code)

    % Bridge generation - Java ↔ Scala
    generate_java_scala_bridge/3,   % generate_java_scala_bridge(+Steps, +Options, -Code)
    generate_scala_java_bridge/3,   % generate_scala_java_bridge(+Steps, +Options, -Code)

    % Bridge generation - Java ↔ Kotlin
    generate_java_kotlin_bridge/3,  % generate_java_kotlin_bridge(+Steps, +Options, -Code)
    generate_kotlin_java_bridge/3,  % generate_kotlin_java_bridge(+Steps, +Options, -Code)

    % Bridge generation - Java ↔ Clojure
    generate_java_clojure_bridge/3, % generate_java_clojure_bridge(+Steps, +Options, -Code)
    generate_clojure_java_bridge/3, % generate_clojure_java_bridge(+Steps, +Options, -Code)

    % Bridge generation - Scala ↔ Kotlin
    generate_scala_kotlin_bridge/3, % generate_scala_kotlin_bridge(+Steps, +Options, -Code)
    generate_kotlin_scala_bridge/3, % generate_kotlin_scala_bridge(+Steps, +Options, -Code)

    % Generic bridge generator
    generate_jvm_bridge/4,          % generate_jvm_bridge(+From, +To, +Options, -Code)

    % JVM process management
    generate_jvm_launcher/3,        % generate_jvm_launcher(+Steps, +Options, -ShellScript)
    generate_classpath/2,           % generate_classpath(+Options, -Classpath)

    % Pipeline orchestration
    generate_jvm_pipeline/3,        % generate_jvm_pipeline(+Steps, +Options, -Code)
    generate_multi_jvm_pipeline/3,  % generate_multi_jvm_pipeline(+Steps, +Options, -Code)

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
    (   catch(
            (process_create(path(java), ['--version'], [stdout(pipe(S)), stderr(null)]),
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
            (process_create(path(java), ['--version'], [stdout(pipe(S)), stderr(null)]),
             read_line_to_string(S, VersionStr),
             close(S)),
            _, fail)
    ->  (   sub_string(VersionStr, Before, _, _, ' '),
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
            (process_create(path(jython), ['--version'], [stdout(null), stderr(null), process(PID)]),
             process_wait(PID, exit(0))),
            _, fail)
    ->  Available = true
    ;   Available = false
    ).

%% detect_scala(-Available)
%  Check if Scala compiler is available.
detect_scala(Available) :-
    (   catch(
            (process_create(path(scalac), ['-version'], [stdout(null), stderr(null), process(PID)]),
             process_wait(PID, exit(_))),
            _, fail)
    ->  Available = true
    ;   Available = false
    ).

%% detect_kotlin(-Available)
%  Check if Kotlin compiler is available.
detect_kotlin(Available) :-
    (   catch(
            (process_create(path(kotlinc), ['-version'], [stdout(null), stderr(null), process(PID)]),
             process_wait(PID, exit(_))),
            _, fail)
    ->  Available = true
    ;   Available = false
    ).

%% detect_clojure(-Available)
%  Check if Clojure is available (via clj or clojure command).
detect_clojure(Available) :-
    (   catch(
            (process_create(path(clj), ['--version'], [stdout(null), stderr(null), process(PID)]),
             process_wait(PID, exit(_))),
            _, fail)
    ->  Available = true
    ;   catch(
            (process_create(path(clojure), ['--version'], [stdout(null), stderr(null), process(PID2)]),
             process_wait(PID2, exit(_))),
            _, fail)
    ->  Available = true
    ;   Available = false
    ).

% ============================================================================
% TRANSPORT SELECTION
% ============================================================================

%% jvm_transport_type(+From, +To, -Transport)
%  Determine transport type between two targets.
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
% GENERIC BRIDGE GENERATOR
% ============================================================================

%% generate_jvm_bridge(+From, +To, +Options, -Code)
%  Generate bridge code for any JVM language pair.
generate_jvm_bridge(java, jython, Options, Code) :- !,
    generate_java_jython_bridge([], Options, Code).
generate_jvm_bridge(jython, java, Options, Code) :- !,
    generate_jython_java_bridge([], Options, Code).
generate_jvm_bridge(java, scala, Options, Code) :- !,
    generate_java_scala_bridge([], Options, Code).
generate_jvm_bridge(scala, java, Options, Code) :- !,
    generate_scala_java_bridge([], Options, Code).
generate_jvm_bridge(java, kotlin, Options, Code) :- !,
    generate_java_kotlin_bridge([], Options, Code).
generate_jvm_bridge(kotlin, java, Options, Code) :- !,
    generate_kotlin_java_bridge([], Options, Code).
generate_jvm_bridge(java, clojure, Options, Code) :- !,
    generate_java_clojure_bridge([], Options, Code).
generate_jvm_bridge(clojure, java, Options, Code) :- !,
    generate_clojure_java_bridge([], Options, Code).
generate_jvm_bridge(scala, kotlin, Options, Code) :- !,
    generate_scala_kotlin_bridge([], Options, Code).
generate_jvm_bridge(kotlin, scala, Options, Code) :- !,
    generate_kotlin_scala_bridge([], Options, Code).
generate_jvm_bridge(From, To, _, Code) :-
    format(string(Code), "// Bridge ~w → ~w not yet implemented", [From, To]).

% ============================================================================
% JAVA ↔ JYTHON BRIDGE
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
    (   StepCodes = []
    ->  AllStepCode = "    // No Jython steps defined"
    ;   atomic_list_concat(StepCodes, '\n', AllStepCode)
    ),

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
 * All calls run in the same JVM - no subprocess overhead.
 */
public class ~w {

    private final PythonInterpreter interp;

    public ~w() {
        this.interp = new PythonInterpreter();
        interp.exec(\"import sys\");
        interp.exec(\"import json\");
    }

    private static String loadScript(String path) throws IOException {
        return new String(Files.readAllBytes(Paths.get(path)));
    }

~w

    /**
     * Execute full pipeline with Jython steps.
     */
    public Map<String, Object> runPipeline(Map<String, Object> record) {
        Map<String, Object> current = new HashMap<>(record);
        // Chain step calls here
        return current;
    }

    public void close() {
        if (interp != null) {
            interp.close();
        }
    }
}
", [Package, ClassName, ClassName, AllStepCode]).

%% generate_jython_java_bridge(+Steps, +Options, -Code)
%  Generate Jython code that calls Java steps directly.
generate_jython_java_bridge(Steps, Options, Code) :-
    option(java_class(JavaClass), Options, 'generated.Pipeline'),

    findall(StepCode,
        (   member(step(Name, java, _Script, _StepOpts), Steps),
            format(string(StepCode),
"def call_~w(record):
    '''Call Java step ~w via direct JVM bridge.'''
    try:
        from ~w import ~w
        java_processor = ~w()
        result = java_processor.process(record)
        if hasattr(result, 'isPresent') and result.isPresent():
            return result.get()
        return result
    except Exception as e:
        print('Java step ~w failed: ' + str(e), file=sys.stderr)
        return record
", [Name, Name, JavaClass, Name, Name, Name])
        ),
        StepCodes),
    (   StepCodes = []
    ->  AllStepCode = "# No Java steps defined\npass"
    ;   atomic_list_concat(StepCodes, '\n', AllStepCode)
    ),

    format(string(Code),
"#!/usr/bin/env jython
# -*- coding: utf-8 -*-
# Generated by UnifyWeaver JVM Glue
# Jython → Java Bridge
#
# Direct JVM interop - no subprocess overhead

from __future__ import print_function
import sys
import json

~w

def run_pipeline(record):
    '''Execute full pipeline with Java steps.'''
    current = dict(record)
    # Chain step calls here
    return current

if __name__ == '__main__':
    test_record = {'test': 'data'}
    result = run_pipeline(test_record)
    print(json.dumps(result))
", [AllStepCode]).

% ============================================================================
% JAVA ↔ SCALA BRIDGE
% ============================================================================

%% generate_java_scala_bridge(+Steps, +Options, -Code)
%  Generate Java code that calls Scala objects/functions directly.
%  Scala compiles to JVM bytecode, so interop is seamless.
generate_java_scala_bridge(Steps, Options, Code) :-
    option(package(Package), Options, generated),
    option(class_name(ClassName), Options, 'ScalaBridge'),
    option(scala_package(ScalaPackage), Options, 'generated.scala'),

    findall(StepCode,
        (   member(step(Name, scala, _Script, _StepOpts), Steps),
            format(string(StepCode),
"    /**
     * Call Scala object: ~w
     */
    public Map<String, Object> call_~w(Map<String, Object> record) {
        try {
            // Scala objects are accessed via MODULE$ singleton
            Object scalaObj = Class.forName(\"~w.~w$\")
                .getField(\"MODULE$\").get(null);

            // Convert Java Map to Scala Map
            scala.collection.immutable.Map<String, Object> scalaMap =
                scala.jdk.CollectionConverters.MapHasAsScala(record).asScala().toMap(
                    scala.Predef.<scala.Tuple2<String, Object>>conforms());

            // Call process method
            java.lang.reflect.Method processMethod = scalaObj.getClass()
                .getMethod(\"process\", scala.collection.immutable.Map.class);

            @SuppressWarnings(\"unchecked\")
            scala.collection.immutable.Map<String, Object> result =
                (scala.collection.immutable.Map<String, Object>)
                processMethod.invoke(scalaObj, scalaMap);

            // Convert back to Java Map
            return scala.jdk.CollectionConverters.MapHasAsJava(result).asJava();
        } catch (Exception e) {
            System.err.println(\"Scala step ~w failed: \" + e.getMessage());
            e.printStackTrace();
            return record;
        }
    }
", [Name, Name, ScalaPackage, Name, Name])
        ),
        StepCodes),
    (   StepCodes = []
    ->  AllStepCode = "    // No Scala steps defined"
    ;   atomic_list_concat(StepCodes, '\n', AllStepCode)
    ),

    format(string(Code),
"// Generated by UnifyWeaver JVM Glue
// Java → Scala Bridge
//
// Scala compiles to JVM bytecode, enabling seamless interop.
// Scala objects (singletons) are accessed via MODULE$ field.

package ~w;

import java.util.*;

/**
 * Bridge for calling Scala code from Java.
 * Uses reflection to access Scala objects and handle Map conversions.
 */
public class ~w {

~w

    /**
     * Execute pipeline with Scala steps.
     */
    public Map<String, Object> runPipeline(Map<String, Object> record) {
        Map<String, Object> current = new HashMap<>(record);
        // Chain step calls here
        return current;
    }
}
", [Package, ClassName, AllStepCode]).

%% generate_scala_java_bridge(+Steps, +Options, -Code)
%  Generate Scala code that calls Java classes directly.
generate_scala_java_bridge(Steps, Options, Code) :-
    option(package(Package), Options, 'generated.scala'),
    option(java_package(JavaPackage), Options, 'generated'),

    findall(StepCode,
        (   member(step(Name, java, _Script, _StepOpts), Steps),
            format(string(StepCode),
"  /**
   * Call Java class: ~w
   */
  def call_~w(record: Map[String, Any]): Map[String, Any] = {
    try {
      import ~w.~w
      import scala.jdk.CollectionConverters._

      val javaMap: java.util.Map[String, Object] =
        record.view.mapValues(_.asInstanceOf[Object]).toMap.asJava

      val processor = new ~w()
      val result = processor.process(javaMap)

      result.asScala.toMap.view.mapValues(_.asInstanceOf[Any]).toMap
    } catch {
      case e: Exception =>
        System.err.println(s\"Java step ~w failed: $${e.getMessage}\")
        record
    }
  }
", [Name, Name, JavaPackage, Name, Name, Name])
        ),
        StepCodes),
    (   StepCodes = []
    ->  AllStepCode = "  // No Java steps defined"
    ;   atomic_list_concat(StepCodes, '\n', AllStepCode)
    ),

    format(string(Code),
"// Generated by UnifyWeaver JVM Glue
// Scala → Java Bridge
//
// Direct JVM interop - Java classes are directly callable from Scala.

package ~w

import scala.jdk.CollectionConverters._

/**
 * Bridge for calling Java code from Scala.
 * Uses Scala's Java interop with collection converters.
 */
object JavaBridge {

~w

  /**
   * Execute pipeline with Java steps.
   */
  def runPipeline(record: Map[String, Any]): Map[String, Any] = {
    var current = record
    // Chain step calls here
    current
  }
}
", [Package, AllStepCode]).

% ============================================================================
% JAVA ↔ KOTLIN BRIDGE
% ============================================================================

%% generate_java_kotlin_bridge(+Steps, +Options, -Code)
%  Generate Java code that calls Kotlin classes/functions directly.
generate_java_kotlin_bridge(Steps, Options, Code) :-
    option(package(Package), Options, generated),
    option(class_name(ClassName), Options, 'KotlinBridge'),
    option(kotlin_package(KotlinPackage), Options, 'generated.kotlin'),

    findall(StepCode,
        (   member(step(Name, kotlin, _Script, _StepOpts), Steps),
            format(string(StepCode),
"    /**
     * Call Kotlin class: ~w
     */
    public Map<String, Object> call_~w(Map<String, Object> record) {
        try {
            // Kotlin classes are directly accessible
            Class<?> kotlinClass = Class.forName(\"~w.~w\");
            Object instance = kotlinClass.getDeclaredConstructor().newInstance();

            java.lang.reflect.Method processMethod = kotlinClass.getMethod(
                \"process\", Map.class);

            @SuppressWarnings(\"unchecked\")
            Map<String, Object> result = (Map<String, Object>)
                processMethod.invoke(instance, record);

            return result;
        } catch (Exception e) {
            System.err.println(\"Kotlin step ~w failed: \" + e.getMessage());
            return record;
        }
    }
", [Name, Name, KotlinPackage, Name, Name])
        ),
        StepCodes),
    (   StepCodes = []
    ->  AllStepCode = "    // No Kotlin steps defined"
    ;   atomic_list_concat(StepCodes, '\n', AllStepCode)
    ),

    format(string(Code),
"// Generated by UnifyWeaver JVM Glue
// Java → Kotlin Bridge
//
// Kotlin compiles to JVM bytecode with full Java interop.

package ~w;

import java.util.*;

/**
 * Bridge for calling Kotlin code from Java.
 * Kotlin classes are directly accessible as Java classes.
 */
public class ~w {

~w

    /**
     * Execute pipeline with Kotlin steps.
     */
    public Map<String, Object> runPipeline(Map<String, Object> record) {
        Map<String, Object> current = new HashMap<>(record);
        // Chain step calls here
        return current;
    }
}
", [Package, ClassName, AllStepCode]).

%% generate_kotlin_java_bridge(+Steps, +Options, -Code)
%  Generate Kotlin code that calls Java classes directly.
generate_kotlin_java_bridge(Steps, Options, Code) :-
    option(package(Package), Options, 'generated.kotlin'),
    option(java_package(JavaPackage), Options, 'generated'),

    findall(StepCode,
        (   member(step(Name, java, _Script, _StepOpts), Steps),
            format(string(StepCode),
"    /**
     * Call Java class: ~w
     */
    fun call_~w(record: Map<String, Any>): Map<String, Any> {
        return try {
            val processor = ~w.~w()

            @Suppress(\"UNCHECKED_CAST\")
            val javaMap = record as Map<String, Any?>

            val result = processor.process(javaMap)

            @Suppress(\"UNCHECKED_CAST\")
            result as Map<String, Any>
        } catch (e: Exception) {
            System.err.println(\"Java step ~w failed: $${e.message}\")
            record
        }
    }
", [Name, Name, JavaPackage, Name, Name])
        ),
        StepCodes),
    (   StepCodes = []
    ->  AllStepCode = "    // No Java steps defined"
    ;   atomic_list_concat(StepCodes, '\n', AllStepCode)
    ),

    format(string(Code),
"// Generated by UnifyWeaver JVM Glue
// Kotlin → Java Bridge
//
// Kotlin has seamless Java interop - Java classes are directly callable.

package ~w

/**
 * Bridge for calling Java code from Kotlin.
 * Java classes work directly in Kotlin with no conversion needed.
 */
class JavaBridge {

~w

    /**
     * Execute pipeline with Java steps.
     */
    fun runPipeline(record: Map<String, Any>): Map<String, Any> {
        var current = record
        // Chain step calls here
        return current
    }
}
", [Package, AllStepCode]).

% ============================================================================
% JAVA ↔ CLOJURE BRIDGE
% ============================================================================

%% generate_java_clojure_bridge(+Steps, +Options, -Code)
%  Generate Java code that calls Clojure functions via clojure.java.api.
generate_java_clojure_bridge(Steps, Options, Code) :-
    option(package(Package), Options, generated),
    option(class_name(ClassName), Options, 'ClojureBridge'),
    option(clojure_namespace(ClojureNS), Options, 'generated.clj'),

    findall(StepCode,
        (   member(step(Name, clojure, _Script, _StepOpts), Steps),
            format(string(StepCode),
"    /**
     * Call Clojure function: ~w
     */
    public Map<String, Object> call_~w(Map<String, Object> record) {
        try {
            // Load the Clojure namespace
            IFn require = Clojure.var(\"clojure.core\", \"require\");
            require.invoke(Clojure.read(\"~w\"));

            // Get the process function
            IFn processFunc = Clojure.var(\"~w\", \"process\");

            // Convert Java Map to Clojure map
            Object clojureMap = Clojure.var(\"clojure.core\", \"into\")
                .invoke(Clojure.read(\"{}\"), record);

            // Call the function
            Object result = processFunc.invoke(clojureMap);

            // Convert back to Java Map
            @SuppressWarnings(\"unchecked\")
            Map<String, Object> resultMap = (Map<String, Object>) result;
            return new HashMap<>(resultMap);
        } catch (Exception e) {
            System.err.println(\"Clojure step ~w failed: \" + e.getMessage());
            return record;
        }
    }
", [Name, Name, ClojureNS, ClojureNS, Name])
        ),
        StepCodes),
    (   StepCodes = []
    ->  AllStepCode = "    // No Clojure steps defined"
    ;   atomic_list_concat(StepCodes, '\n', AllStepCode)
    ),

    format(string(Code),
"// Generated by UnifyWeaver JVM Glue
// Java → Clojure Bridge
//
// Uses clojure.java.api for invoking Clojure from Java.

package ~w;

import clojure.java.api.Clojure;
import clojure.lang.IFn;
import java.util.*;

/**
 * Bridge for calling Clojure code from Java.
 * Uses Clojure's Java API for runtime function invocation.
 */
public class ~w {

    public ~w() {
        // Ensure Clojure runtime is initialized
        Clojure.var(\"clojure.core\", \"identity\");
    }

~w

    /**
     * Execute pipeline with Clojure steps.
     */
    public Map<String, Object> runPipeline(Map<String, Object> record) {
        Map<String, Object> current = new HashMap<>(record);
        // Chain step calls here
        return current;
    }
}
", [Package, ClassName, ClassName, AllStepCode]).

%% generate_clojure_java_bridge(+Steps, +Options, -Code)
%  Generate Clojure code that calls Java classes directly.
generate_clojure_java_bridge(Steps, Options, Code) :-
    option(namespace(Namespace), Options, 'generated.clj'),
    option(java_package(JavaPackage), Options, 'generated'),

    findall(StepCode,
        (   member(step(Name, java, _Script, _StepOpts), Steps),
            format(string(StepCode),
"(defn call-~w
  \"Call Java class: ~w\"
  [record]
  (try
    (let [processor (new ~w.~w)
          result (.process processor record)]
      (into {} result))
    (catch Exception e
      (binding [*out* *err*]
        (println (str \"Java step ~w failed: \" (.getMessage e))))
      record)))
", [Name, Name, JavaPackage, Name, Name])
        ),
        StepCodes),
    (   StepCodes = []
    ->  AllStepCode = ";; No Java steps defined"
    ;   atomic_list_concat(StepCodes, '\n', AllStepCode)
    ),

    format(string(Code),
";; Generated by UnifyWeaver JVM Glue
;; Clojure → Java Bridge
;;
;; Clojure has excellent Java interop via (new Class) and (.method obj).

(ns ~w
  (:import [java.util Map HashMap]))

~w

(defn run-pipeline
  \"Execute pipeline with Java steps.\"
  [record]
  (let [current (into {} record)]
    ;; Chain step calls here
    current))

;; Entry point for testing
(defn -main [& args]
  (let [test-record {\"test\" \"data\"}
        result (run-pipeline test-record)]
    (println result)))
", [Namespace, AllStepCode]).

% ============================================================================
% SCALA ↔ KOTLIN BRIDGE
% ============================================================================

%% generate_scala_kotlin_bridge(+Steps, +Options, -Code)
%  Generate Scala code that calls Kotlin classes.
generate_scala_kotlin_bridge(Steps, Options, Code) :-
    option(package(Package), Options, 'generated.scala'),
    option(kotlin_package(KotlinPackage), Options, 'generated.kotlin'),

    findall(StepCode,
        (   member(step(Name, kotlin, _Script, _StepOpts), Steps),
            format(string(StepCode),
"  /**
   * Call Kotlin class: ~w
   */
  def call_~w(record: Map[String, Any]): Map[String, Any] = {
    try {
      import scala.jdk.CollectionConverters._

      // Kotlin classes are directly accessible
      val processor = new ~w.~w()

      val javaMap: java.util.Map[String, Object] =
        record.view.mapValues(_.asInstanceOf[Object]).toMap.asJava

      val result = processor.process(javaMap)

      result.asScala.toMap.view.mapValues(_.asInstanceOf[Any]).toMap
    } catch {
      case e: Exception =>
        System.err.println(s\"Kotlin step ~w failed: $${e.getMessage}\")
        record
    }
  }
", [Name, Name, KotlinPackage, Name, Name])
        ),
        StepCodes),
    (   StepCodes = []
    ->  AllStepCode = "  // No Kotlin steps defined"
    ;   atomic_list_concat(StepCodes, '\n', AllStepCode)
    ),

    format(string(Code),
"// Generated by UnifyWeaver JVM Glue
// Scala → Kotlin Bridge
//
// Both compile to JVM bytecode - interop is straightforward.

package ~w

import scala.jdk.CollectionConverters._

/**
 * Bridge for calling Kotlin code from Scala.
 */
object KotlinBridge {

~w

  /**
   * Execute pipeline with Kotlin steps.
   */
  def runPipeline(record: Map[String, Any]): Map[String, Any] = {
    var current = record
    // Chain step calls here
    current
  }
}
", [Package, AllStepCode]).

%% generate_kotlin_scala_bridge(+Steps, +Options, -Code)
%  Generate Kotlin code that calls Scala objects.
generate_kotlin_scala_bridge(Steps, Options, Code) :-
    option(package(Package), Options, 'generated.kotlin'),
    option(scala_package(ScalaPackage), Options, 'generated.scala'),

    findall(StepCode,
        (   member(step(Name, scala, _Script, _StepOpts), Steps),
            format(string(StepCode),
"    /**
     * Call Scala object: ~w
     */
    fun call_~w(record: Map<String, Any>): Map<String, Any> {
        return try {
            // Access Scala object via MODULE$ companion
            val scalaObjClass = Class.forName(\"~w.~w$$\")
            val moduleField = scalaObjClass.getField(\"MODULE$$\")
            val scalaObj = moduleField.get(null)

            // Convert to scala.collection.immutable.Map
            val toScalaMap = Class.forName(\"scala.jdk.CollectionConverters$$\")
                .getMethod(\"MapHasAsScala\", java.util.Map::class.java)

            val processMethod = scalaObj.javaClass.getMethod(\"process\", Any::class.java)

            @Suppress(\"UNCHECKED_CAST\")
            val result = processMethod.invoke(scalaObj, record) as Map<String, Any>
            result
        } catch (e: Exception) {
            System.err.println(\"Scala step ~w failed: $${e.message}\")
            record
        }
    }
", [Name, Name, ScalaPackage, Name, Name])
        ),
        StepCodes),
    (   StepCodes = []
    ->  AllStepCode = "    // No Scala steps defined"
    ;   atomic_list_concat(StepCodes, '\n', AllStepCode)
    ),

    format(string(Code),
"// Generated by UnifyWeaver JVM Glue
// Kotlin → Scala Bridge
//
// Access Scala objects via MODULE$ field (singleton pattern).

package ~w

/**
 * Bridge for calling Scala code from Kotlin.
 */
class ScalaBridge {

~w

    /**
     * Execute pipeline with Scala steps.
     */
    fun runPipeline(record: Map<String, Any>): Map<String, Any> {
        var current = record
        // Chain step calls here
        return current
    }
}
", [Package, AllStepCode]).

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

    % Detect which JVM languages are used
    (   member(jython, Types) -> HasJython = "true" ; HasJython = "false" ),
    (   member(scala, Types) -> HasScala = "true" ; HasScala = "false" ),
    (   member(kotlin, Types) -> HasKotlin = "true" ; HasKotlin = "false" ),
    (   member(clojure, Types) -> HasClojure = "true" ; HasClojure = "false" ),

    format(string(ShellScript),
"#!/bin/bash
# Generated by UnifyWeaver JVM Glue
# JVM Pipeline Launcher

set -e

# Configuration
JAVA_OPTS=\"~w\"
CLASSPATH=\"~w\"
MAIN_CLASS=\"~w\"

# Number of steps: ~w
# Languages used: Java~w~w~w~w

# Check Java
if ! command -v java &> /dev/null; then
    echo \"Error: Java not found\" >&2
    exit 1
fi

# Check optional runtimes
HAS_JYTHON_STEPS=~w
HAS_SCALA_STEPS=~w
HAS_KOTLIN_STEPS=~w
HAS_CLOJURE_STEPS=~w

if [ \"$HAS_JYTHON_STEPS\" = \"true\" ]; then
    if ! java -cp \"$CLASSPATH\" org.python.util.PythonInterpreter --version 2>/dev/null; then
        echo \"Warning: Jython may not be in classpath\" >&2
    fi
fi

if [ \"$HAS_CLOJURE_STEPS\" = \"true\" ]; then
    if ! java -cp \"$CLASSPATH\" clojure.main --version 2>/dev/null; then
        echo \"Warning: Clojure may not be in classpath\" >&2
    fi
fi

# Run pipeline
echo \"Starting JVM pipeline...\" >&2
java $JAVA_OPTS -cp \"$CLASSPATH\" \"$MAIN_CLASS\" \"$@\"
", [JavaOpts, Classpath, MainClass, NumSteps,
    (HasJython = "true" -> ", Jython" ; ""),
    (HasScala = "true" -> ", Scala" ; ""),
    (HasKotlin = "true" -> ", Kotlin" ; ""),
    (HasClojure = "true" -> ", Clojure" ; ""),
    HasJython, HasScala, HasKotlin, HasClojure]).

%% generate_classpath(+Options, -Classpath)
%  Generate classpath string from options.
generate_classpath(Options, Classpath) :-
    option(jars(Jars), Options, []),
    option(classes_dir(ClassesDir), Options, 'build/classes'),

    % Default dependencies for each language
    DefaultJars = [
        'lib/gson.jar',
        'lib/jython-standalone.jar',
        'lib/scala-library.jar',
        'lib/kotlin-stdlib.jar',
        'lib/clojure.jar'
    ],
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
    partition(is_jython_step, OtherSteps, JythonSteps, OtherSteps2),
    partition(is_scala_step, OtherSteps2, ScalaSteps, OtherSteps3),
    partition(is_kotlin_step, OtherSteps3, KotlinSteps, OtherSteps4),
    partition(is_clojure_step, OtherSteps4, ClojureSteps, _),

    % Count languages
    AllLanguages = [java-JavaSteps, jython-JythonSteps, scala-ScalaSteps,
                    kotlin-KotlinSteps, clojure-ClojureSteps],
    include(has_steps, AllLanguages, UsedLanguages),
    length(UsedLanguages, NumLanguages),

    (   NumLanguages > 1
    ->  generate_multi_jvm_pipeline(Steps, Options, Code)
    ;   NumLanguages =:= 1, UsedLanguages = [Lang-_]
    ->  generate_single_language_pipeline(Lang, Steps, Options, Code)
    ;   Code = "// No JVM steps found"
    ).

has_steps(_-Steps) :- Steps \= [].

is_java_step(step(_, java, _, _)).
is_jython_step(step(_, jython, _, _)).
is_scala_step(step(_, scala, _, _)).
is_kotlin_step(step(_, kotlin, _, _)).
is_clojure_step(step(_, clojure, _, _)).

%% generate_single_language_pipeline(+Lang, +Steps, +Options, -Code)
generate_single_language_pipeline(java, Steps, _Options, Code) :-
    length(Steps, N),
    format(string(Code), "// Java-only pipeline with ~w steps\\n// Use java_target.pl directly", [N]).
generate_single_language_pipeline(jython, Steps, _Options, Code) :-
    length(Steps, N),
    format(string(Code), "# Jython-only pipeline with ~w steps\\n# Use jython_target.pl directly", [N]).
generate_single_language_pipeline(scala, Steps, _Options, Code) :-
    length(Steps, N),
    format(string(Code), "// Scala-only pipeline with ~w steps\\n// Use scala_target.pl directly", [N]).
generate_single_language_pipeline(kotlin, Steps, _Options, Code) :-
    length(Steps, N),
    format(string(Code), "// Kotlin-only pipeline with ~w steps\\n// Use kotlin_target.pl directly", [N]).
generate_single_language_pipeline(clojure, Steps, _Options, Code) :-
    length(Steps, N),
    format(string(Code), ";; Clojure-only pipeline with ~w steps\\n;; Use clojure_target.pl directly", [N]).

%% generate_multi_jvm_pipeline(+Steps, +Options, -Code)
%  Generate a Java-based orchestrator for multi-language pipelines.
generate_multi_jvm_pipeline(Steps, Options, Code) :-
    option(package(Package), Options, generated),

    % Count each language
    include(is_java_step, Steps, JavaSteps),
    include(is_jython_step, Steps, JythonSteps),
    include(is_scala_step, Steps, ScalaSteps),
    include(is_kotlin_step, Steps, KotlinSteps),
    include(is_clojure_step, Steps, ClojureSteps),

    length(JavaSteps, NumJava),
    length(JythonSteps, NumJython),
    length(ScalaSteps, NumScala),
    length(KotlinSteps, NumKotlin),
    length(ClojureSteps, NumClojure),

    % Generate imports
    (   NumJython > 0
    ->  JythonImports = "import org.python.util.PythonInterpreter;\nimport org.python.core.*;"
    ;   JythonImports = ""
    ),
    (   NumClojure > 0
    ->  ClojureImports = "import clojure.java.api.Clojure;\nimport clojure.lang.IFn;"
    ;   ClojureImports = ""
    ),

    format(string(Code),
"// Generated by UnifyWeaver JVM Glue
// Multi-Language JVM Pipeline Orchestrator
// Languages: Java(~w), Jython(~w), Scala(~w), Kotlin(~w), Clojure(~w)

package ~w;

import java.util.*;
import java.io.*;
~w
~w

/**
 * Orchestrates a pipeline with multiple JVM languages.
 * All languages run in the same JVM for maximum performance.
 */
public class MultiLanguagePipeline {

    // Language-specific bridges would be initialized here

    public MultiLanguagePipeline() {
        // Initialize any embedded interpreters (Jython, Clojure)
    }

    /**
     * Run the complete multi-language pipeline.
     */
    public Optional<Map<String, Object>> process(Map<String, Object> record) {
        Map<String, Object> current = new HashMap<>(record);

        // TODO: Execute steps in order, calling appropriate bridges
        // Each step stays in the same JVM - no subprocess overhead

        return Optional.of(current);
    }

    public static void main(String[] args) throws Exception {
        MultiLanguagePipeline pipeline = new MultiLanguagePipeline();

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
", [NumJava, NumJython, NumScala, NumKotlin, NumClojure,
    Package, JythonImports, ClojureImports]).

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
    format('~n=== Testing JVM Glue (Extended) ===~n~n'),

    % Test 1: Runtime detection
    format('[Test 1] JVM runtime detection~n'),
    detect_jvm_runtime(Runtime),
    format('  Runtime: ~w~n', [Runtime]),

    % Test 2: Java version
    format('~n[Test 2] Java version detection~n'),
    detect_java_version(Version),
    format('  Version: ~w~n', [Version]),

    % Test 3: Language availability
    format('~n[Test 3] JVM language availability~n'),
    detect_scala(ScalaAvail),
    detect_kotlin(KotlinAvail),
    detect_clojure(ClojureAvail),
    detect_jython(JythonAvail),
    format('  Scala: ~w~n', [ScalaAvail]),
    format('  Kotlin: ~w~n', [KotlinAvail]),
    format('  Clojure: ~w~n', [ClojureAvail]),
    format('  Jython: ~w~n', [JythonAvail]),

    % Test 4: Transport selection
    format('~n[Test 4] Transport selection~n'),
    forall(
        (   member(From, [java, scala, kotlin, clojure]),
            member(To, [java, scala, kotlin, clojure]),
            From \= To
        ),
        (   jvm_transport_type(From, To, Transport),
            format('  ~w → ~w: ~w~n', [From, To, Transport])
        )
    ),

    % Test 5: Bridge generation
    format('~n[Test 5] Bridge generation~n'),

    % Java → Scala
    generate_java_scala_bridge([], [package(test)], ScalaBridge),
    (   sub_atom(ScalaBridge, _, _, _, 'ScalaBridge')
    ->  format('  [PASS] Java → Scala bridge~n')
    ;   format('  [FAIL] Java → Scala bridge~n')
    ),

    % Java → Kotlin
    generate_java_kotlin_bridge([], [package(test)], KotlinBridge),
    (   sub_atom(KotlinBridge, _, _, _, 'KotlinBridge')
    ->  format('  [PASS] Java → Kotlin bridge~n')
    ;   format('  [FAIL] Java → Kotlin bridge~n')
    ),

    % Java → Clojure
    generate_java_clojure_bridge([], [package(test)], ClojureBridge),
    (   sub_atom(ClojureBridge, _, _, _, 'ClojureBridge')
    ->  format('  [PASS] Java → Clojure bridge~n')
    ;   format('  [FAIL] Java → Clojure bridge~n')
    ),

    % Scala → Java
    generate_scala_java_bridge([], [package(test)], ScalaJavaBridge),
    (   sub_atom(ScalaJavaBridge, _, _, _, 'JavaBridge')
    ->  format('  [PASS] Scala → Java bridge~n')
    ;   format('  [FAIL] Scala → Java bridge~n')
    ),

    % Kotlin → Java
    generate_kotlin_java_bridge([], [package(test)], KotlinJavaBridge),
    (   sub_atom(KotlinJavaBridge, _, _, _, 'JavaBridge')
    ->  format('  [PASS] Kotlin → Java bridge~n')
    ;   format('  [FAIL] Kotlin → Java bridge~n')
    ),

    % Clojure → Java
    generate_clojure_java_bridge([], [namespace(test)], ClojureJavaBridge),
    (   sub_atom(ClojureJavaBridge, _, _, _, 'run-pipeline')
    ->  format('  [PASS] Clojure → Java bridge~n')
    ;   format('  [FAIL] Clojure → Java bridge~n')
    ),

    format('~n=== JVM Glue Tests Complete ===~n').
