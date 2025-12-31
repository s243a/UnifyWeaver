:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% hadoop_streaming.pl - Hadoop Streaming backend for distributed parallel execution
% Uses Hadoop Streaming for MapReduce-style batch processing across clusters
% Works with any language that reads stdin and writes stdout

:- module(hadoop_streaming_backend, [
    backend_init_impl/2,
    backend_execute_impl/4,
    backend_cleanup_impl/1
]).

:- use_module(library(process)).
:- use_module(library(filesex)).
:- use_module(library(lists)).

%% ============================================
%% BACKEND IMPLEMENTATION
%% ============================================

%% backend_init_impl(+Config, -State)
%  Initialize Hadoop Streaming backend
%
%  Configuration options:
%  - backend_args([reducers(N)]) - Number of reduce tasks
%  - backend_args([mapper(Script)]) - Custom mapper script path
%  - backend_args([reducer(Script)]) - Custom reducer script (default: cat)
%  - backend_args([hadoop_home(Path)]) - HADOOP_HOME directory
%  - backend_args([input_format(F)]) - Input format (text | sequence)
%  - backend_args([output_format(F)]) - Output format (text | sequence)
%  - backend_args([combiner(Script)]) - Optional combiner script
%  - backend_args([files(FileList)]) - Additional files to distribute
%  - backend_args([conf(ConfList)]) - Additional Hadoop config options
%  - backend_args([use_hdfs(Bool)]) - Use HDFS for input/output (default: false)
%  - backend_args([hdfs_input(Path)]) - HDFS input path
%  - backend_args([hdfs_output(Path)]) - HDFS output path
%
%  @example Initialize with 10 reducers
%    ?- backend_init_impl([backend_args([reducers(10)])], State).
%
%  @example Initialize with custom mapper/reducer
%    ?- backend_init_impl([backend_args([
%           mapper('scripts/my_mapper.sh'),
%           reducer('scripts/my_reducer.sh'),
%           reducers(5)
%       ])], State).
backend_init_impl(Config, State) :-
    % Check Hadoop is installed
    check_hadoop_installed(HadoopHome),

    % Extract configuration
    (   member(backend_args(Args), Config)
    ->  true
    ;   Args = []
    ),

    % Parse number of reducers
    (   member(reducers(NumReducers), Args)
    ->  true
    ;   NumReducers = 1  % Default: 1 reducer
    ),

    % Parse custom mapper (optional - ScriptPath used if not specified)
    (   member(mapper(MapperScript), Args)
    ->  true
    ;   MapperScript = none
    ),

    % Parse custom reducer (default: cat for identity)
    (   member(reducer(ReducerScript), Args)
    ->  true
    ;   ReducerScript = '/bin/cat'
    ),

    % Parse combiner (optional)
    (   member(combiner(CombinerScript), Args)
    ->  true
    ;   CombinerScript = none
    ),

    % Parse additional files to distribute
    (   member(files(DistFiles), Args)
    ->  true
    ;   DistFiles = []
    ),

    % Parse additional Hadoop configuration
    (   member(conf(HadoopConf), Args)
    ->  true
    ;   HadoopConf = []
    ),

    % Parse HDFS options
    (   member(use_hdfs(UseHDFS), Args)
    ->  true
    ;   UseHDFS = false
    ),
    (   member(hdfs_input(HDFSInput), Args)
    ->  true
    ;   HDFSInput = none
    ),
    (   member(hdfs_output(HDFSOutput), Args)
    ->  true
    ;   HDFSOutput = none
    ),

    % Validate
    (   integer(NumReducers), NumReducers >= 0
    ->  true
    ;   throw(error(domain_error(non_negative_integer, NumReducers),
                    context(backend_init_impl/2, 'reducers must be non-negative integer')))
    ),

    % Create temporary directory for batch files and scripts
    create_temp_directory(TempDir),

    % Initialize state
    State = state(
        hadoop_home(HadoopHome),
        num_reducers(NumReducers),
        mapper(MapperScript),
        reducer(ReducerScript),
        combiner(CombinerScript),
        dist_files(DistFiles),
        hadoop_conf(HadoopConf),
        use_hdfs(UseHDFS),
        hdfs_input(HDFSInput),
        hdfs_output(HDFSOutput),
        temp_dir(TempDir)
    ),

    format('[HadoopStreaming] Initialized: reducers=~w, hadoop_home=~w, temp_dir=~w~n',
           [NumReducers, HadoopHome, TempDir]).

%% backend_execute_impl(+State, +Partitions, +ScriptPath, -Results)
%  Execute script on partitions using Hadoop Streaming
%
%  @arg State Backend state from backend_init_impl/2
%  @arg Partitions List of partition(ID, Data) terms
%  @arg ScriptPath Path to bash script to execute (used as mapper)
%  @arg Results List of result(PartitionID, Output) terms
backend_execute_impl(State, Partitions, ScriptPath, Results) :-
    State = state(
        hadoop_home(HadoopHome),
        num_reducers(NumReducers),
        mapper(CustomMapper),
        reducer(ReducerScript),
        combiner(CombinerScript),
        dist_files(DistFiles),
        hadoop_conf(HadoopConf),
        use_hdfs(UseHDFS),
        hdfs_input(HDFSInput),
        hdfs_output(HDFSOutput),
        temp_dir(TempDir)
    ),

    % Determine actual mapper script
    (   CustomMapper = none
    ->  MapperScript = ScriptPath
    ;   MapperScript = CustomMapper
    ),

    length(Partitions, NumPartitions),
    format('[HadoopStreaming] Executing ~w partitions with ~w reducers~n',
           [NumPartitions, NumReducers]),

    % Prepare input data
    (   UseHDFS = true, HDFSInput \= none
    ->  % Use HDFS input directly
        InputPath = HDFSInput,
        format('[HadoopStreaming] Using HDFS input: ~w~n', [InputPath])
    ;   % Write partition data to local files and upload to HDFS or use local
        write_input_data(Partitions, TempDir, InputPath, UseHDFS),
        format('[HadoopStreaming] Created input at: ~w~n', [InputPath])
    ),

    % Determine output path
    (   UseHDFS = true, HDFSOutput \= none
    ->  OutputPath = HDFSOutput
    ;   UseHDFS = true
    ->  get_time(Ts),
        format(atom(OutputPath), '/tmp/unifyweaver_output_~w', [Ts])
    ;   format(atom(OutputPath), '~w/output', [TempDir])
    ),

    % Build and execute Hadoop Streaming command
    build_hadoop_command(HadoopHome, InputPath, OutputPath, MapperScript,
                         ReducerScript, CombinerScript, NumReducers,
                         DistFiles, HadoopConf, TempDir, Command),

    format('[HadoopStreaming] Executing: ~w~n', [Command]),
    execute_hadoop_command(Command, ExitCode),

    (   ExitCode =:= 0
    ->  format('[HadoopStreaming] Execution completed successfully~n', [])
    ;   format('[HadoopStreaming] Warning: Exit code ~w~n', [ExitCode])
    ),

    % Collect results
    collect_hadoop_results(OutputPath, UseHDFS, Partitions, TempDir, Results),
    format('[HadoopStreaming] Collected results~n', []).

%% backend_cleanup_impl(+State)
%  Clean up Hadoop Streaming backend resources
backend_cleanup_impl(State) :-
    State = state(_, _, _, _, _, _, _, use_hdfs(UseHDFS), _, hdfs_output(HDFSOutput), temp_dir(TempDir)),

    format('[HadoopStreaming] Cleaning up...~n', []),

    % Clean local temp directory
    (   exists_directory(TempDir)
    ->  delete_directory_and_contents(TempDir),
        format('[HadoopStreaming] Cleaned local temp: ~w~n', [TempDir])
    ;   true
    ),

    % Optionally clean HDFS output (commented out for safety)
    % (   UseHDFS = true, HDFSOutput \= none
    % ->  format(atom(Cmd), 'hdfs dfs -rm -r ~w', [HDFSOutput]),
    %     process_create('/bin/bash', ['-c', Cmd], [])
    % ;   true
    % ),

    format('[HadoopStreaming] Cleanup complete~n', []).

%% ============================================
%% HELPER PREDICATES
%% ============================================

%% check_hadoop_installed(-HadoopHome)
%  Verify Hadoop is installed and accessible
check_hadoop_installed(HadoopHome) :-
    % First try HADOOP_HOME environment variable
    (   getenv('HADOOP_HOME', HadoopHome)
    ->  true
    ;   % Try common locations
        member(HadoopHome, ['/usr/local/hadoop', '/opt/hadoop', '/usr/lib/hadoop']),
        exists_directory(HadoopHome)
    ->  true
    ;   % Try to find hadoop command in PATH
        catch(
            (   process_create(path(hadoop), ['version'],
                              [stdout(pipe(Stream)), stderr(null), process(PID)]),
                read_string(Stream, _, Output),
                close(Stream),
                process_wait(PID, exit(ExitCode)),
                (   ExitCode =:= 0
                ->  format('[HadoopStreaming] Detected: ~w', [Output]),
                    % Use empty string to indicate PATH-based access
                    HadoopHome = ''
                ;   fail
                )
            ),
            _,
            fail
        )
    ),
    !,
    format('[HadoopStreaming] Using HADOOP_HOME: ~w~n',
           [HadoopHome = '' -> 'PATH' ; HadoopHome]).

check_hadoop_installed(_) :-
    format('[HadoopStreaming] ERROR: Hadoop not found~n', []),
    format('[HadoopStreaming] Set HADOOP_HOME or install Hadoop~n', []),
    throw(error(existence_error(system_command, hadoop),
               context(check_hadoop_installed/1,
                      'Hadoop is not installed or HADOOP_HOME not set'))).

%% create_temp_directory(-TempDir)
%  Create temporary directory for batch files
create_temp_directory(TempDir) :-
    get_time(Timestamp),
    format(atom(TempDir), '/tmp/unifyweaver_hadoop_~w', [Timestamp]),
    make_directory(TempDir).

%% write_input_data(+Partitions, +TempDir, -InputPath, +UseHDFS)
%  Write partition data to input file(s)
write_input_data(Partitions, TempDir, InputPath, UseHDFS) :-
    % Write all partition data to a single input file
    format(atom(LocalInputPath), '~w/input.txt', [TempDir]),
    open(LocalInputPath, write, Stream, [encoding(utf8)]),
    forall(member(partition(_ID, Data), Partitions),
           maplist(write_item(Stream), Data)),
    close(Stream),

    (   UseHDFS = true
    ->  % Upload to HDFS
        get_time(Ts),
        format(atom(HDFSInputPath), '/tmp/unifyweaver_input_~w', [Ts]),
        format(atom(PutCmd), 'hdfs dfs -put ~w ~w', [LocalInputPath, HDFSInputPath]),
        process_create('/bin/bash', ['-c', PutCmd], [process(PID)]),
        process_wait(PID, exit(_)),
        InputPath = HDFSInputPath
    ;   InputPath = LocalInputPath
    ).

write_item(Stream, Item) :-
    (   atom(Item)
    ->  writeln(Stream, Item)
    ;   string(Item)
    ->  writeln(Stream, Item)
    ;   write_term(Stream, Item, [quoted(true), nl(true)])
    ).

%% build_hadoop_command(+HadoopHome, +Input, +Output, +Mapper, +Reducer,
%%                      +Combiner, +NumReducers, +DistFiles, +Conf, +TempDir, -Command)
%  Build Hadoop Streaming command
build_hadoop_command(HadoopHome, Input, Output, Mapper, Reducer,
                     Combiner, NumReducers, DistFiles, Conf, TempDir, Command) :-
    % Build hadoop command prefix
    (   HadoopHome = ''
    ->  HadoopCmd = 'hadoop'
    ;   format(atom(HadoopCmd), '~w/bin/hadoop', [HadoopHome])
    ),

    % Find streaming jar
    find_streaming_jar(HadoopHome, StreamingJar),

    % Build base command
    format(string(BaseCmd), '~w jar ~w', [HadoopCmd, StreamingJar]),

    % Add input/output
    format(string(IOCmd), '-input "~w" -output "~w"', [Input, Output]),

    % Create wrapper scripts for mapper/reducer (to handle shebang issues)
    create_wrapper_script(Mapper, TempDir, WrappedMapper),
    create_wrapper_script(Reducer, TempDir, WrappedReducer),

    % Add mapper and reducer
    format(string(MRCmd), '-mapper "bash ~w" -reducer "bash ~w"',
           [WrappedMapper, WrappedReducer]),

    % Add combiner if specified
    (   Combiner = none
    ->  CombinerCmd = ''
    ;   create_wrapper_script(Combiner, TempDir, WrappedCombiner),
        format(string(CombinerCmd), '-combiner "bash ~w"', [WrappedCombiner])
    ),

    % Add number of reducers
    format(string(ReducersCmd), '-D mapreduce.job.reduces=~w', [NumReducers]),

    % Add distributed files
    (   DistFiles = []
    ->  FilesCmd = ''
    ;   maplist(format_file_arg, DistFiles, FileArgs),
        atomic_list_concat(FileArgs, ' ', FilesCmd)
    ),

    % Add additional configuration
    (   Conf = []
    ->  ConfCmd = ''
    ;   maplist(format_conf_arg, Conf, ConfArgs),
        atomic_list_concat(ConfArgs, ' ', ConfCmd)
    ),

    % Combine all parts
    atomic_list_concat([BaseCmd, IOCmd, MRCmd, CombinerCmd, ReducersCmd, FilesCmd, ConfCmd],
                       ' ', Command).

%% find_streaming_jar(+HadoopHome, -JarPath)
%  Find the Hadoop streaming JAR file
find_streaming_jar(HadoopHome, JarPath) :-
    % Common locations to check
    (   HadoopHome = ''
    ->  % Try to find via hadoop classpath
        catch(
            (   process_create(path(hadoop), ['classpath'],
                              [stdout(pipe(Stream)), stderr(null), process(PID)]),
                read_string(Stream, _, Classpath),
                close(Stream),
                process_wait(PID, exit(0)),
                % Look for streaming jar in classpath
                split_string(Classpath, ":", "", Paths),
                member(Path, Paths),
                sub_string(Path, _, _, _, "streaming"),
                exists_file(Path),
                JarPath = Path
            ),
            _,
            fail
        )
    ;   % Look in HADOOP_HOME
        member(SubPath, [
            'share/hadoop/tools/lib/hadoop-streaming-*.jar',
            'share/hadoop/tools/lib/hadoop-streaming.jar',
            'contrib/streaming/hadoop-streaming-*.jar',
            'lib/hadoop-streaming-*.jar'
        ]),
        format(atom(Pattern), '~w/~w', [HadoopHome, SubPath]),
        expand_file_name(Pattern, [JarPath|_]),
        exists_file(JarPath)
    ),
    !,
    format('[HadoopStreaming] Using streaming jar: ~w~n', [JarPath]).

find_streaming_jar(_, _) :-
    throw(error(existence_error(file, 'hadoop-streaming.jar'),
               context(find_streaming_jar/2,
                      'Could not find Hadoop streaming JAR'))).

%% create_wrapper_script(+Script, +TempDir, -WrapperPath)
%  Create a wrapper script for the mapper/reducer
create_wrapper_script(Script, TempDir, WrapperPath) :-
    % Get base name
    file_base_name(Script, BaseName),
    format(atom(WrapperPath), '~w/wrapper_~w', [TempDir, BaseName]),

    % Create wrapper that sources the script
    format(string(WrapperContent), '#!/bin/bash
# Wrapper script for Hadoop Streaming
# Ensures proper execution of: ~w

# Source the actual script and process stdin
cat | bash "~w"
', [Script, Script]),

    open(WrapperPath, write, Stream, [encoding(utf8)]),
    write(Stream, WrapperContent),
    close(Stream),

    % Make executable
    process_create(path(chmod), ['+x', WrapperPath], []).

%% format_file_arg(+File, -Arg)
%  Format distributed file argument
format_file_arg(File, Arg) :-
    format(atom(Arg), '-file "~w"', [File]).

%% format_conf_arg(+Conf, -Arg)
%  Format configuration argument
format_conf_arg(Key=Value, Arg) :-
    format(atom(Arg), '-D ~w=~w', [Key, Value]).
format_conf_arg(Key-Value, Arg) :-
    format(atom(Arg), '-D ~w=~w', [Key, Value]).

%% execute_hadoop_command(+Command, -ExitCode)
%  Execute Hadoop command via bash
execute_hadoop_command(Command, ExitCode) :-
    process_create('/bin/bash', ['-c', Command],
                   [stdout(pipe(OutStream)),
                    stderr(pipe(ErrStream)),
                    process(PID)]),

    % Read output streams
    read_string(OutStream, _, StdOut),
    read_string(ErrStream, _, StdErr),
    close(OutStream),
    close(ErrStream),

    % Wait for completion
    process_wait(PID, exit(ExitCode)),

    % Log output
    (   StdOut \= ""
    ->  format('[HadoopStreaming] stdout: ~w~n', [StdOut])
    ;   true
    ),
    (   StdErr \= ""
    ->  format('[HadoopStreaming] stderr: ~w~n', [StdErr])
    ;   true
    ).

%% collect_hadoop_results(+OutputPath, +UseHDFS, +Partitions, +TempDir, -Results)
%  Collect results from Hadoop output
collect_hadoop_results(OutputPath, UseHDFS, Partitions, TempDir, Results) :-
    % Get output from HDFS or local
    (   UseHDFS = true
    ->  % Download from HDFS
        format(atom(LocalOutput), '~w/hadoop_output', [TempDir]),
        format(atom(GetCmd), 'hdfs dfs -getmerge ~w ~w', [OutputPath, LocalOutput]),
        process_create('/bin/bash', ['-c', GetCmd], [process(PID)]),
        process_wait(PID, exit(_)),
        OutputFile = LocalOutput
    ;   % Local output - merge part files
        format(atom(OutputFile), '~w/part-*', [OutputPath]),
        expand_file_name(OutputFile, PartFiles),
        (   PartFiles = []
        ->  AllOutput = ""
        ;   maplist(read_file_to_string_safe, PartFiles, PartOutputs),
            atomic_list_concat(PartOutputs, AllOutput)
        )
    ),

    % For Hadoop streaming, we get merged output
    % Distribute to partition results (simplified - all partitions get same merged result)
    (   var(AllOutput)
    ->  (   exists_file(OutputFile)
        ->  read_file_to_string(OutputFile, AllOutput, [])
        ;   AllOutput = ""
        )
    ;   true
    ),

    % Create result for each partition
    maplist(create_partition_result(AllOutput), Partitions, Results).

read_file_to_string_safe(File, Content) :-
    (   exists_file(File)
    ->  read_file_to_string(File, Content, [])
    ;   Content = ""
    ).

create_partition_result(Output, partition(ID, _), result(ID, Output)).
