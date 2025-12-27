% test_connection.pl - Test RPyC connection from Prolog
:- use_module('../../src/unifyweaver/glue/rpyc_glue').

test_connection :-
    format('Testing RPyC connection from Prolog...~n'),
    catch(
        (   rpyc_connect(localhost, [
                security(unsecured),
                acknowledge_risk(true),
                remote_port(18812)
            ], Proxy),
            format('Connected to RPyC server.~n'),

            % Test module access
            rpyc_import(Proxy, math, Math),
            py_call(Math:sqrt(25), Sqrt),
            format('math.sqrt(25) = ~w~n', [Sqrt]),

            % Test proxy layers
            rpyc_root(Proxy, Root),
            format('Root proxy obtained: ~w~n', [Root]),

            rpyc_disconnect(Proxy),
            format('Disconnected successfully.~n')
        ),
        Error,
        format('Error: ~w~n', [Error])
    ).

:- initialization(test_connection).
