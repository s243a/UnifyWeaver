:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% hadoop_native.pl - Native Hadoop backend using JVM glue
% Uses Hadoop's native Java/Scala API for in-process MapReduce
% Best for JVM language targets (Java, Scala, Kotlin, Clojure)

:- module(hadoop_native_backend, [
    backend_init_impl/2,
    backend_execute_impl/4,
    backend_cleanup_impl/1,

    % Code generation exports
    generate_hadoop_mapper/4,
    generate_hadoop_reducer/4,
    generate_hadoop_driver/4
]).

:- use_module(library(process)).
:- use_module(library(filesex)).
:- use_module(library(lists)).

% Import JVM glue for cross-target communication
:- use_module(unifyweaver(glue/jvm_glue)).

%% ============================================
%% BACKEND IMPLEMENTATION
%% ============================================

%% backend_init_impl(+Config, -State)
%  Initialize native Hadoop backend
%
%  Configuration options:
%  - backend_args([target_language(L)]) - java | scala | kotlin (default: java)
%  - backend_args([reducers(N)]) - Number of reduce tasks
%  - backend_args([combiner(true/false)]) - Use combiner optimization
%  - backend_args([speculative(true/false)]) - Enable speculative execution
%  - backend_args([compression(C)]) - none | gzip | snappy | lz4
%  - backend_args([memory_mb(N)]) - Map/reduce task memory
%  - backend_args([hdfs_replication(N)]) - HDFS replication factor
%  - backend_args([jar_output(Path)]) - Output JAR path
%
%  @example Initialize with Scala target
%    ?- backend_init_impl([backend_args([target_language(scala), reducers(10)])], State).
backend_init_impl(Config, State) :-
    % Check JVM and Hadoop are available
    check_jvm_hadoop_available(HadoopHome, JavaVersion),

    % Extract configuration
    (   member(backend_args(Args), Config)
    ->  true
    ;   Args = []
    ),

    % Parse target language
    (   member(target_language(TargetLang), Args)
    ->  true
    ;   TargetLang = java
    ),
    validate_target_language(TargetLang),

    % Parse number of reducers
    (   member(reducers(NumReducers), Args)
    ->  true
    ;   NumReducers = 1
    ),

    % Parse combiner option
    (   member(combiner(UseCombiner), Args)
    ->  true
    ;   UseCombiner = false
    ),

    % Parse compression
    (   member(compression(Compression), Args)
    ->  true
    ;   Compression = none
    ),

    % Parse memory settings
    (   member(memory_mb(MemoryMB), Args)
    ->  true
    ;   MemoryMB = 1024
    ),

    % Parse JAR output path
    (   member(jar_output(JarPath), Args)
    ->  true
    ;   get_time(Ts),
        format(atom(JarPath), '/tmp/unifyweaver_hadoop_~w.jar', [Ts])
    ),

    % Create temporary directory
    create_temp_directory(TempDir),

    % Initialize state
    State = state(
        hadoop_home(HadoopHome),
        java_version(JavaVersion),
        target_language(TargetLang),
        num_reducers(NumReducers),
        use_combiner(UseCombiner),
        compression(Compression),
        memory_mb(MemoryMB),
        jar_path(JarPath),
        temp_dir(TempDir)
    ),

    format('[HadoopNative] Initialized: lang=~w, reducers=~w, java=~w~n',
           [TargetLang, NumReducers, JavaVersion]).

%% validate_target_language(+Lang)
validate_target_language(java) :- !.
validate_target_language(scala) :- !.
validate_target_language(kotlin) :- !.
validate_target_language(clojure) :- !.
validate_target_language(Lang) :-
    throw(error(domain_error(jvm_language, Lang),
                context(validate_target_language/1,
                       'Language must be: java, scala, kotlin, or clojure'))).

%% backend_execute_impl(+State, +Partitions, +ScriptPath, -Results)
%  Execute using native Hadoop MapReduce
backend_execute_impl(State, Partitions, ScriptPath, Results) :-
    State = state(
        hadoop_home(HadoopHome),
        java_version(_),
        target_language(TargetLang),
        num_reducers(NumReducers),
        use_combiner(UseCombiner),
        compression(Compression),
        memory_mb(MemoryMB),
        jar_path(JarPath),
        temp_dir(TempDir)
    ),

    length(Partitions, NumPartitions),
    format('[HadoopNative] Executing ~w partitions as ~w MapReduce job~n',
           [NumPartitions, TargetLang]),

    % Write partition data to input files
    write_input_data(Partitions, TempDir, InputPath),

    % Generate MapReduce code
    format('[HadoopNative] Generating ~w MapReduce code...~n', [TargetLang]),
    generate_mapreduce_code(TargetLang, ScriptPath, TempDir, UseCombiner,
                            Compression, MapperClass, ReducerClass, DriverClass),

    % Compile and package JAR
    format('[HadoopNative] Compiling and packaging JAR...~n', []),
    compile_and_package(TargetLang, TempDir, JarPath, HadoopHome,
                        MapperClass, ReducerClass, DriverClass),

    % Prepare output path
    get_time(Ts),
    format(atom(OutputPath), '~w/output_~w', [TempDir, Ts]),

    % Build and execute Hadoop command
    build_hadoop_jar_command(HadoopHome, JarPath, DriverClass, InputPath,
                             OutputPath, NumReducers, MemoryMB, Command),

    format('[HadoopNative] Executing: ~w~n', [Command]),
    execute_hadoop_command(Command, ExitCode),

    (   ExitCode =:= 0
    ->  format('[HadoopNative] MapReduce job completed successfully~n', [])
    ;   format('[HadoopNative] Warning: Exit code ~w~n', [ExitCode])
    ),

    % Collect results
    collect_results(OutputPath, Partitions, TempDir, Results),
    format('[HadoopNative] Collected results~n', []).

%% backend_cleanup_impl(+State)
%  Clean up native Hadoop backend resources
backend_cleanup_impl(state(_, _, _, _, _, _, _, jar_path(JarPath), temp_dir(TempDir))) :-
    format('[HadoopNative] Cleaning up...~n', []),

    % Clean temp directory
    (   exists_directory(TempDir)
    ->  delete_directory_and_contents(TempDir)
    ;   true
    ),

    % Optionally remove JAR
    (   exists_file(JarPath)
    ->  delete_file(JarPath)
    ;   true
    ),

    format('[HadoopNative] Cleanup complete~n', []).

%% ============================================
%% CODE GENERATION
%% ============================================

%% generate_mapreduce_code(+Lang, +ScriptPath, +TempDir, +UseCombiner,
%%                         +Compression, -MapperClass, -ReducerClass, -DriverClass)
generate_mapreduce_code(java, ScriptPath, TempDir, UseCombiner, Compression,
                        MapperClass, ReducerClass, DriverClass) :-
    MapperClass = 'UnifyWeaverMapper',
    ReducerClass = 'UnifyWeaverReducer',
    DriverClass = 'UnifyWeaverDriver',

    % Generate Mapper
    generate_hadoop_mapper(java, ScriptPath, [], MapperCode),
    format(atom(MapperFile), '~w/UnifyWeaverMapper.java', [TempDir]),
    write_file(MapperFile, MapperCode),

    % Generate Reducer
    generate_hadoop_reducer(java, [], ReducerCode),
    format(atom(ReducerFile), '~w/UnifyWeaverReducer.java', [TempDir]),
    write_file(ReducerFile, ReducerCode),

    % Generate Driver
    generate_hadoop_driver(java, [combiner(UseCombiner), compression(Compression)], DriverCode),
    format(atom(DriverFile), '~w/UnifyWeaverDriver.java', [TempDir]),
    write_file(DriverFile, DriverCode).

generate_mapreduce_code(scala, ScriptPath, TempDir, UseCombiner, Compression,
                        MapperClass, ReducerClass, DriverClass) :-
    MapperClass = 'UnifyWeaverMapper',
    ReducerClass = 'UnifyWeaverReducer',
    DriverClass = 'UnifyWeaverDriver',

    % Generate Scala code
    generate_scala_mapreduce(ScriptPath, UseCombiner, Compression, ScalaCode),
    format(atom(ScalaFile), '~w/UnifyWeaver.scala', [TempDir]),
    write_file(ScalaFile, ScalaCode).

generate_mapreduce_code(kotlin, ScriptPath, TempDir, UseCombiner, Compression,
                        MapperClass, ReducerClass, DriverClass) :-
    MapperClass = 'UnifyWeaverMapper',
    ReducerClass = 'UnifyWeaverReducer',
    DriverClass = 'UnifyWeaverDriverKt',

    % Generate Kotlin code
    generate_kotlin_mapreduce(ScriptPath, UseCombiner, Compression, KotlinCode),
    format(atom(KotlinFile), '~w/UnifyWeaver.kt', [TempDir]),
    write_file(KotlinFile, KotlinCode).

generate_mapreduce_code(clojure, ScriptPath, TempDir, UseCombiner, Compression,
                        MapperClass, ReducerClass, DriverClass) :-
    MapperClass = 'unifyweaver.hadoop.mapper',
    ReducerClass = 'unifyweaver.hadoop.reducer',
    DriverClass = 'unifyweaver.hadoop.driver',

    % Generate Clojure code
    generate_clojure_mapreduce(ScriptPath, UseCombiner, Compression, ClojureCode),
    format(atom(ClojureDir), '~w/src/unifyweaver/hadoop', [TempDir]),
    process_create(path(mkdir), ['-p', ClojureDir], [stdout(null), stderr(null), process(MkPID)]),
    process_wait(MkPID, _),
    format(atom(ClojureFile), '~w/driver.clj', [ClojureDir]),
    write_file(ClojureFile, ClojureCode).

%% generate_hadoop_mapper(+Lang, +ScriptPath, +Options, -Code)
%  Generate Hadoop Mapper code
generate_hadoop_mapper(java, ScriptPath, _Options, Code) :-
    format(string(Code),
'// Generated by UnifyWeaver Hadoop Native Backend
import java.io.*;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.*;

/**
 * UnifyWeaver Mapper - processes input and emits key-value pairs.
 * Wraps script: ~w
 */
public class UnifyWeaverMapper extends Mapper<LongWritable, Text, Text, Text> {

    private Text outputKey = new Text();
    private Text outputValue = new Text();

    @Override
    protected void map(LongWritable key, Text value, Context context)
            throws IOException, InterruptedException {
        String line = value.toString();

        // Process line through embedded logic or script
        String result = processLine(line);

        if (result != null && !result.isEmpty()) {
            // Emit with line number as key for ordering
            outputKey.set(String.valueOf(key.get()));
            outputValue.set(result);
            context.write(outputKey, outputValue);
        }
    }

    /**
     * Process a single input line.
     * Override this method or use script execution.
     */
    protected String processLine(String line) {
        // Default: pass through (identity mapper)
        // In production, this would execute the Prolog-compiled logic
        try {
            // Simple transformation: double numbers
            if (line.matches("^\\\\d+$")) {
                int num = Integer.parseInt(line.trim());
                return String.valueOf(num * 2);
            }
            return line;
        } catch (Exception e) {
            return line;
        }
    }
}
', [ScriptPath]).

%% generate_hadoop_reducer(+Lang, +Options, -Code)
%  Generate Hadoop Reducer code
generate_hadoop_reducer(java, _Options, Code) :-
    Code =
'// Generated by UnifyWeaver Hadoop Native Backend
import java.io.*;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.*;

/**
 * UnifyWeaver Reducer - aggregates mapper output.
 */
public class UnifyWeaverReducer extends Reducer<Text, Text, Text, Text> {

    private Text result = new Text();

    @Override
    protected void reduce(Text key, Iterable<Text> values, Context context)
            throws IOException, InterruptedException {

        StringBuilder sb = new StringBuilder();
        for (Text val : values) {
            if (sb.length() > 0) sb.append("\\n");
            sb.append(val.toString());
        }

        result.set(sb.toString());
        context.write(key, result);
    }
}
'.

%% generate_hadoop_driver(+Lang, +Options, -Code)
%  Generate Hadoop Driver (Job configuration) code
generate_hadoop_driver(java, Options, Code) :-
    (   member(combiner(true), Options)
    ->  CombinerLine = '        job.setCombinerClass(UnifyWeaverReducer.class);'
    ;   CombinerLine = '        // No combiner configured'
    ),
    (   member(compression(gzip), Options)
    ->  CompressionLines = '        FileOutputFormat.setCompressOutput(job, true);
        FileOutputFormat.setOutputCompressorClass(job, org.apache.hadoop.io.compress.GzipCodec.class);'
    ;   member(compression(snappy), Options)
    ->  CompressionLines = '        FileOutputFormat.setCompressOutput(job, true);
        FileOutputFormat.setOutputCompressorClass(job, org.apache.hadoop.io.compress.SnappyCodec.class);'
    ;   CompressionLines = '        // No compression configured'
    ),

    format(string(Code),
'// Generated by UnifyWeaver Hadoop Native Backend
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.*;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

/**
 * UnifyWeaver Hadoop Driver - configures and runs the MapReduce job.
 */
public class UnifyWeaverDriver {

    public static void main(String[] args) throws Exception {
        if (args.length < 2) {
            System.err.println("Usage: UnifyWeaverDriver <input> <output> [reducers]");
            System.exit(1);
        }

        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "UnifyWeaver MapReduce");

        job.setJarByClass(UnifyWeaverDriver.class);
        job.setMapperClass(UnifyWeaverMapper.class);
        job.setReducerClass(UnifyWeaverReducer.class);

~w

        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);

        // Set number of reducers
        if (args.length > 2) {
            job.setNumReduceTasks(Integer.parseInt(args[2]));
        }

~w

        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));

        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}
', [CombinerLine, CompressionLines]).

%% generate_scala_mapreduce(+ScriptPath, +UseCombiner, +Compression, -Code)
generate_scala_mapreduce(ScriptPath, UseCombiner, Compression, Code) :-
    (   UseCombiner = true -> CombinerStr = 'true' ; CombinerStr = 'false' ),
    (   Compression = none -> CompressionStr = 'None'
    ;   format(atom(CompressionStr), 'Some("~w")', [Compression])
    ),

    format(string(Code),
'// Generated by UnifyWeaver Hadoop Native Backend - Scala
// Script reference: ~w

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.Path
import org.apache.hadoop.io.{LongWritable, Text}
import org.apache.hadoop.mapreduce.{Job, Mapper, Reducer}
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat
import scala.collection.JavaConverters._

/**
 * UnifyWeaver Scala Mapper
 */
class UnifyWeaverMapper extends Mapper[LongWritable, Text, Text, Text] {
  private val outputKey = new Text()
  private val outputValue = new Text()

  override def map(key: LongWritable, value: Text, context: Mapper[LongWritable, Text, Text, Text]#Context): Unit = {
    val line = value.toString
    val result = processLine(line)

    if (result.nonEmpty) {
      outputKey.set(key.get.toString)
      outputValue.set(result)
      context.write(outputKey, outputValue)
    }
  }

  def processLine(line: String): String = {
    // Transform: double numbers
    if (line.matches("^\\\\d+$")) {
      (line.trim.toInt * 2).toString
    } else {
      line
    }
  }
}

/**
 * UnifyWeaver Scala Reducer
 */
class UnifyWeaverReducer extends Reducer[Text, Text, Text, Text] {
  private val result = new Text()

  override def reduce(key: Text, values: java.lang.Iterable[Text],
                      context: Reducer[Text, Text, Text, Text]#Context): Unit = {
    val combined = values.asScala.map(_.toString).mkString("\\n")
    result.set(combined)
    context.write(key, result)
  }
}

/**
 * UnifyWeaver Scala Driver
 */
object UnifyWeaverDriver {
  val useCombiner: Boolean = ~w
  val compression: Option[String] = ~w

  def main(args: Array[String]): Unit = {
    if (args.length < 2) {
      System.err.println("Usage: UnifyWeaverDriver <input> <output> [reducers]")
      System.exit(1)
    }

    val conf = new Configuration()
    val job = Job.getInstance(conf, "UnifyWeaver Scala MapReduce")

    job.setJarByClass(this.getClass)
    job.setMapperClass(classOf[UnifyWeaverMapper])
    job.setReducerClass(classOf[UnifyWeaverReducer])

    if (useCombiner) {
      job.setCombinerClass(classOf[UnifyWeaverReducer])
    }

    job.setOutputKeyClass(classOf[Text])
    job.setOutputValueClass(classOf[Text])

    if (args.length > 2) {
      job.setNumReduceTasks(args(2).toInt)
    }

    FileInputFormat.addInputPath(job, new Path(args(0)))
    FileOutputFormat.setOutputPath(job, new Path(args(1)))

    System.exit(if (job.waitForCompletion(true)) 0 else 1)
  }
}
', [ScriptPath, CombinerStr, CompressionStr]).

%% generate_kotlin_mapreduce(+ScriptPath, +UseCombiner, +Compression, -Code)
generate_kotlin_mapreduce(ScriptPath, UseCombiner, Compression, Code) :-
    (   UseCombiner = true -> CombinerStr = 'true' ; CombinerStr = 'false' ),
    (   Compression = none -> CompressionStr = 'null'
    ;   format(atom(CompressionStr), '"~w"', [Compression])
    ),

    format(string(Code),
'// Generated by UnifyWeaver Hadoop Native Backend - Kotlin
// Script reference: ~w

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.Path
import org.apache.hadoop.io.LongWritable
import org.apache.hadoop.io.Text
import org.apache.hadoop.mapreduce.Job
import org.apache.hadoop.mapreduce.Mapper
import org.apache.hadoop.mapreduce.Reducer
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat

/**
 * UnifyWeaver Kotlin Mapper
 */
class UnifyWeaverMapper : Mapper<LongWritable, Text, Text, Text>() {
    private val outputKey = Text()
    private val outputValue = Text()

    override fun map(key: LongWritable, value: Text, context: Context) {
        val line = value.toString()
        val result = processLine(line)

        if (result.isNotEmpty()) {
            outputKey.set(key.get().toString())
            outputValue.set(result)
            context.write(outputKey, outputValue)
        }
    }

    private fun processLine(line: String): String {
        return if (line.matches(Regex("^\\\\d+$"))) {
            (line.trim().toInt() * 2).toString()
        } else {
            line
        }
    }
}

/**
 * UnifyWeaver Kotlin Reducer
 */
class UnifyWeaverReducer : Reducer<Text, Text, Text, Text>() {
    private val result = Text()

    override fun reduce(key: Text, values: Iterable<Text>, context: Context) {
        val combined = values.joinToString("\\n") { it.toString() }
        result.set(combined)
        context.write(key, result)
    }
}

/**
 * UnifyWeaver Kotlin Driver
 */
fun main(args: Array<String>) {
    val useCombiner = ~w
    val compression: String? = ~w

    if (args.size < 2) {
        System.err.println("Usage: UnifyWeaverDriver <input> <output> [reducers]")
        System.exit(1)
    }

    val conf = Configuration()
    val job = Job.getInstance(conf, "UnifyWeaver Kotlin MapReduce")

    job.setJarByClass(UnifyWeaverMapper::class.java)
    job.mapperClass = UnifyWeaverMapper::class.java
    job.reducerClass = UnifyWeaverReducer::class.java

    if (useCombiner) {
        job.combinerClass = UnifyWeaverReducer::class.java
    }

    job.setOutputKeyClass(Text::class.java)
    job.setOutputValueClass(Text::class.java)

    if (args.size > 2) {
        job.numReduceTasks = args[2].toInt()
    }

    FileInputFormat.addInputPath(job, Path(args[0]))
    FileOutputFormat.setOutputPath(job, Path(args[1]))

    System.exit(if (job.waitForCompletion(true)) 0 else 1)
}
', [ScriptPath, CombinerStr, CompressionStr]).

%% generate_clojure_mapreduce(+ScriptPath, +UseCombiner, +Compression, -Code)
generate_clojure_mapreduce(ScriptPath, UseCombiner, Compression, Code) :-
    (   UseCombiner = true -> CombinerStr = 'true' ; CombinerStr = 'false' ),
    (   Compression = none -> CompressionStr = 'nil'
    ;   format(atom(CompressionStr), '"~w"', [Compression])
    ),

    format(string(Code),
';; Generated by UnifyWeaver Hadoop Native Backend - Clojure
;; Script reference: ~w

(ns unifyweaver.hadoop.driver
  (:import [org.apache.hadoop.conf Configuration]
           [org.apache.hadoop.fs Path]
           [org.apache.hadoop.io LongWritable Text]
           [org.apache.hadoop.mapreduce Job Mapper Reducer]
           [org.apache.hadoop.mapreduce.lib.input FileInputFormat]
           [org.apache.hadoop.mapreduce.lib.output FileOutputFormat])
  (:gen-class))

;; Configuration
(def use-combiner ~w)
(def compression ~w)

;;; ============================================
;;; MAPPER
;;; ============================================

(gen-class
  :name unifyweaver.hadoop.mapper
  :extends org.apache.hadoop.mapreduce.Mapper
  :prefix "mapper-")

(defn mapper-map
  "Map function: process each input line."
  [this key value context]
  (let [line (.toString value)
        result (process-line line)]
    (when (and result (not (empty? result)))
      (.write context
              (Text. (str (.get key)))
              (Text. result)))))

(defn process-line
  "Process a single input line - identity transform with number doubling."
  [line]
  (try
    (if (re-matches #"^\\d+$" (.trim line))
      (str (* 2 (Long/parseLong (.trim line))))
      line)
    (catch Exception e line)))

;;; ============================================
;;; REDUCER
;;; ============================================

(gen-class
  :name unifyweaver.hadoop.reducer
  :extends org.apache.hadoop.mapreduce.Reducer
  :prefix "reducer-")

(defn reducer-reduce
  "Reduce function: aggregate mapper output."
  [this key values context]
  (let [combined (->> values
                      iterator-seq
                      (map #(.toString %))
                      (clojure.string/join "\\n"))]
    (.write context key (Text. combined))))

;;; ============================================
;;; DRIVER
;;; ============================================

(defn -main
  "Main driver: configure and run the MapReduce job."
  [& args]
  (when (< (count args) 2)
    (binding [*out* *err*]
      (println "Usage: unifyweaver.hadoop.driver <input> <output> [reducers]"))
    (System/exit 1))

  (let [conf (Configuration.)
        job (Job/getInstance conf "UnifyWeaver Clojure MapReduce")
        input-path (first args)
        output-path (second args)
        num-reducers (if (>= (count args) 3)
                       (Integer/parseInt (nth args 2))
                       1)]

    ;; Configure job
    (.setJarByClass job (class -main))
    (.setMapperClass job unifyweaver.hadoop.mapper)
    (.setReducerClass job unifyweaver.hadoop.reducer)

    ;; Combiner
    (when use-combiner
      (.setCombinerClass job unifyweaver.hadoop.reducer))

    ;; Output types
    (.setOutputKeyClass job Text)
    (.setOutputValueClass job Text)

    ;; Reducers
    (.setNumReduceTasks job num-reducers)

    ;; Compression
    (when compression
      (FileOutputFormat/setCompressOutput job true)
      (cond
        (= compression "gzip")
        (FileOutputFormat/setOutputCompressorClass job
          org.apache.hadoop.io.compress.GzipCodec)
        (= compression "snappy")
        (FileOutputFormat/setOutputCompressorClass job
          org.apache.hadoop.io.compress.SnappyCodec)))

    ;; Input/Output paths
    (FileInputFormat/addInputPath job (Path. input-path))
    (FileOutputFormat/setOutputPath job (Path. output-path))

    ;; Run job
    (System/exit (if (.waitForCompletion job true) 0 1))))
', [ScriptPath, CombinerStr, CompressionStr]).

%% ============================================
%% COMPILATION AND PACKAGING
%% ============================================

%% compile_and_package(+Lang, +TempDir, +JarPath, +HadoopHome, +Mapper, +Reducer, +Driver)
compile_and_package(java, TempDir, JarPath, HadoopHome, _, _, _) :-
    % Get Hadoop classpath
    get_hadoop_classpath(HadoopHome, Classpath),

    % Compile Java files
    format(atom(CompileCmd),
           'javac -cp "~w" -d "~w" ~w/*.java 2>&1',
           [Classpath, TempDir, TempDir]),
    execute_shell(CompileCmd, CompileExit),
    (   CompileExit =:= 0
    ->  true
    ;   throw(error(compilation_failed, context(compile_and_package/7, 'Java compilation failed')))
    ),

    % Create JAR
    format(atom(JarCmd),
           'jar -cvf "~w" -C "~w" . 2>&1',
           [JarPath, TempDir]),
    execute_shell(JarCmd, JarExit),
    (   JarExit =:= 0
    ->  format('[HadoopNative] Created JAR: ~w~n', [JarPath])
    ;   throw(error(jar_failed, context(compile_and_package/7, 'JAR creation failed')))
    ).

compile_and_package(scala, TempDir, JarPath, HadoopHome, _, _, _) :-
    get_hadoop_classpath(HadoopHome, Classpath),

    % Compile Scala files
    format(atom(CompileCmd),
           'scalac -cp "~w" -d "~w" ~w/*.scala 2>&1',
           [Classpath, TempDir, TempDir]),
    execute_shell(CompileCmd, CompileExit),
    (   CompileExit =:= 0
    ->  true
    ;   throw(error(compilation_failed, context(compile_and_package/7, 'Scala compilation failed')))
    ),

    % Create JAR
    format(atom(JarCmd), 'jar -cvf "~w" -C "~w" . 2>&1', [JarPath, TempDir]),
    execute_shell(JarCmd, _).

compile_and_package(kotlin, TempDir, JarPath, HadoopHome, _, _, _) :-
    get_hadoop_classpath(HadoopHome, Classpath),

    % Compile Kotlin files
    format(atom(CompileCmd),
           'kotlinc -cp "~w" -d "~w" ~w/*.kt 2>&1',
           [Classpath, TempDir, TempDir]),
    execute_shell(CompileCmd, CompileExit),
    (   CompileExit =:= 0
    ->  true
    ;   throw(error(compilation_failed, context(compile_and_package/7, 'Kotlin compilation failed')))
    ),

    % Create JAR
    format(atom(JarCmd), 'jar -cvf "~w" -C "~w" . 2>&1', [JarPath, TempDir]),
    execute_shell(JarCmd, _).

compile_and_package(clojure, TempDir, JarPath, HadoopHome, _, _, _) :-
    get_hadoop_classpath(HadoopHome, Classpath),

    % Create classes directory
    format(atom(ClassDir), '~w/classes', [TempDir]),
    process_create(path(mkdir), ['-p', ClassDir], [stdout(null), stderr(null), process(MkPID)]),
    process_wait(MkPID, _),

    % Compile Clojure files using AOT compilation
    % First try with clojure CLI tools
    format(atom(SrcDir), '~w/src', [TempDir]),
    format(atom(CompileCmd),
           'java -cp "~w:~w:~w" clojure.main -e "(binding [*compile-path* \\"~w\\"] (compile \\'unifyweaver.hadoop.driver))" 2>&1',
           [Classpath, SrcDir, ClassDir, ClassDir]),
    execute_shell(CompileCmd, CompileExit),
    (   CompileExit =:= 0
    ->  format('[HadoopNative] Clojure AOT compilation succeeded~n', [])
    ;   % Fallback: include source files in JAR for interpreted execution
        format('[HadoopNative] Note: Running Clojure in interpreted mode~n', []),
        format(atom(CopyCmd), 'cp -r "~w"/* "~w/"', [SrcDir, ClassDir]),
        execute_shell(CopyCmd, _)
    ),

    % Create JAR with Clojure dependencies info
    format(atom(ManifestFile), '~w/MANIFEST.MF', [TempDir]),
    open(ManifestFile, write, MStream, [encoding(utf8)]),
    write(MStream, 'Manifest-Version: 1.0\n'),
    write(MStream, 'Main-Class: unifyweaver.hadoop.driver\n'),
    close(MStream),

    format(atom(JarCmd), 'jar -cvfm "~w" "~w" -C "~w" . 2>&1', [JarPath, ManifestFile, ClassDir]),
    execute_shell(JarCmd, _).

%% ============================================
%% HELPER PREDICATES
%% ============================================

%% check_jvm_hadoop_available(-HadoopHome, -JavaVersion)
check_jvm_hadoop_available(HadoopHome, JavaVersion) :-
    % Check Java
    detect_java_version(JavaVersion),
    (   JavaVersion = none
    ->  throw(error(existence_error(runtime, java),
                   context(check_jvm_hadoop_available/2, 'Java not found')))
    ;   true
    ),

    % Check Hadoop
    (   getenv('HADOOP_HOME', HadoopHome)
    ->  true
    ;   catch(
            (   process_create(path(hadoop), ['classpath'],
                              [stdout(pipe(S)), stderr(null), process(PID)]),
                read_string(S, _, _),
                close(S),
                process_wait(PID, exit(0)),
                HadoopHome = ''
            ),
            _,
            throw(error(existence_error(runtime, hadoop),
                       context(check_jvm_hadoop_available/2,
                              'Hadoop not found. Set HADOOP_HOME')))
        )
    ),
    format('[HadoopNative] Using Hadoop: ~w, Java: ~w~n',
           [HadoopHome = '' -> 'PATH' ; HadoopHome, JavaVersion]).

%% get_hadoop_classpath(+HadoopHome, -Classpath)
get_hadoop_classpath(HadoopHome, Classpath) :-
    (   HadoopHome = ''
    ->  HadoopCmd = 'hadoop classpath'
    ;   format(atom(HadoopCmd), '~w/bin/hadoop classpath', [HadoopHome])
    ),
    process_create('/bin/bash', ['-c', HadoopCmd],
                   [stdout(pipe(S)), stderr(null), process(PID)]),
    read_string(S, _, ClasspathStr),
    close(S),
    process_wait(PID, exit(_)),
    normalize_space(atom(Classpath), ClasspathStr).

%% create_temp_directory(-TempDir)
create_temp_directory(TempDir) :-
    get_time(Timestamp),
    format(atom(TempDir), '/tmp/unifyweaver_hadoop_native_~w', [Timestamp]),
    make_directory(TempDir).

%% write_input_data(+Partitions, +TempDir, -InputPath)
write_input_data(Partitions, TempDir, InputPath) :-
    format(atom(InputPath), '~w/input', [TempDir]),
    make_directory(InputPath),
    forall(member(partition(ID, Data), Partitions),
           (   format(atom(FilePath), '~w/part_~w.txt', [InputPath, ID]),
               open(FilePath, write, Stream),
               forall(member(Item, Data),
                      (   (atom(Item) ; string(Item))
                      ->  writeln(Stream, Item)
                      ;   write_term(Stream, Item, [quoted(true), nl(true)])
                      )),
               close(Stream)
           )).

%% build_hadoop_jar_command(+HadoopHome, +JarPath, +DriverClass, +Input,
%%                          +Output, +NumReducers, +MemoryMB, -Command)
build_hadoop_jar_command(HadoopHome, JarPath, DriverClass, Input,
                         Output, NumReducers, MemoryMB, Command) :-
    (   HadoopHome = ''
    ->  HadoopCmd = 'hadoop'
    ;   format(atom(HadoopCmd), '~w/bin/hadoop', [HadoopHome])
    ),
    format(atom(Command),
           '~w jar "~w" ~w "~w" "~w" ~w -D mapreduce.map.memory.mb=~w -D mapreduce.reduce.memory.mb=~w',
           [HadoopCmd, JarPath, DriverClass, Input, Output, NumReducers, MemoryMB, MemoryMB]).

%% execute_hadoop_command(+Command, -ExitCode)
execute_hadoop_command(Command, ExitCode) :-
    process_create('/bin/bash', ['-c', Command],
                   [stdout(pipe(Out)), stderr(pipe(Err)), process(PID)]),
    read_string(Out, _, StdOut),
    read_string(Err, _, StdErr),
    close(Out),
    close(Err),
    process_wait(PID, exit(ExitCode)),
    (   StdOut \= "" -> format('[HadoopNative] ~w~n', [StdOut]) ; true ),
    (   StdErr \= "" -> format('[HadoopNative] ~w~n', [StdErr]) ; true ).

%% execute_shell(+Command, -ExitCode)
execute_shell(Command, ExitCode) :-
    process_create('/bin/bash', ['-c', Command],
                   [stdout(null), stderr(null), process(PID)]),
    process_wait(PID, exit(ExitCode)).

%% write_file(+Path, +Content)
write_file(Path, Content) :-
    open(Path, write, Stream, [encoding(utf8)]),
    write(Stream, Content),
    close(Stream).

%% collect_results(+OutputPath, +Partitions, +TempDir, -Results)
collect_results(OutputPath, Partitions, _TempDir, Results) :-
    % Find and merge all part files
    format(atom(Pattern), '~w/part-*', [OutputPath]),
    expand_file_name(Pattern, PartFiles),
    (   PartFiles = []
    ->  AllOutput = ""
    ;   maplist(read_file_safe, PartFiles, PartOutputs),
        atomic_list_concat(PartOutputs, AllOutput)
    ),
    % Create result for each partition
    maplist(create_partition_result(AllOutput), Partitions, Results).

read_file_safe(File, Content) :-
    (   exists_file(File)
    ->  read_file_to_string(File, Content, [])
    ;   Content = ""
    ).

create_partition_result(Output, partition(ID, _), result(ID, Output)).
