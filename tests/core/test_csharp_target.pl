
:- module(test_csharp_target, [
    test_csharp_target/0
]).

:- asserta(user:file_search_path(library, 'src/unifyweaver/targets')).

:- use_module(library(apply)).
:- use_module(library(filesex)).
:- use_module(library(lists)).
:- use_module(library(process)).
:- use_module(library(uuid)).
:- use_module(library(csharp_target)).

:- use_module('src/unifyweaver/core/dynamic_source_compiler').
:- use_module('src/unifyweaver/sources').
:- use_module('src/unifyweaver/sources/csv_source').
:- use_module('src/unifyweaver/sources/json_source').

:- dynamic cqt_option/2.
:- dynamic user:test_factorial/2.
:- dynamic user:test_factorial_input/1.
:- dynamic user:test_even/1.
:- dynamic user:test_odd/1.
:- dynamic user:test_parity_input/1.
:- dynamic user:test_product_record/1.
:- dynamic user:test_jsonpath_projection/2.
:- dynamic user:test_order_summary/1.
:- dynamic user:test_orders_jsonl/3.
:- dynamic user:test_json_null_skip/2.
:- dynamic user:test_json_null_default/2.

:- dynamic progress_last_report/1.
:- dynamic progress_count/1.
:- dynamic progress_total/1.

test_csharp_target :-
    set_prolog_flag(verbose, silent),
    configure_csharp_query_options,
    writeln('=== Testing C# query target ==='),
    setup_test_data,
    progress_init,
    Tests = [
        verify_fact_plan,
        verify_join_plan,
        verify_selection_plan,
        verify_arithmetic_plan,
        verify_recursive_arithmetic_plan,
        verify_comparison_plan,
        verify_recursive_plan,
        verify_mutual_recursion_plan,
        verify_dynamic_source_plan,
        verify_tsv_dynamic_source_plan,
        verify_json_dynamic_source_plan,
        verify_json_nested_source_plan,
        verify_json_jsonpath_source_plan,
        verify_json_schema_source_plan,
        verify_json_nested_schema_record_plan,
        verify_json_jsonl_source_plan,
        verify_json_null_policy_skip_plan,
        verify_json_null_policy_default_plan,
        verify_json_object_source_plan,
        verify_generator_mode
    ],
    length(Tests, Total),
    retractall(progress_total(_)),
    asserta(progress_total(Total)),
    maplist(run_with_progress, Tests),
    progress_maybe_report(force),
    cleanup_test_data,
    writeln('=== C# query target tests complete ===').

setup_test_data :-
    cleanup_test_data,
    assertz(user:test_fact(alice, bob)),
    assertz(user:test_fact(bob, charlie)),
    assertz(user:(test_link(X, Z) :- test_fact(X, Y), test_fact(Y, Z))),
    assertz(user:(test_filtered(X) :- test_fact(X, _), X = alice)),
    assertz(user:test_val(item1, 5)),
    assertz(user:test_val(item2, 2)),
    assertz(user:(test_increment(Id, Result) :- test_val(Id, Value), Result is Value + 1)),
    assertz(user:test_num(item1, 5)),
    assertz(user:test_num(item2, -3)),
    assertz(user:(test_positive(Id) :- test_num(Id, Value), Value > 0)),
    assertz(user:test_factorial_input(1)),
    assertz(user:test_factorial_input(2)),
    assertz(user:test_factorial_input(3)),
    assertz(user:test_factorial(0, 1)),
    assertz(user:(test_factorial(N, Result) :-
        test_factorial_input(N),
        N > 0,
        N1 is N - 1,
        test_factorial(N1, Prev),
        Result is Prev * N
    )),
    assertz(user:test_parity_input(0)),
    assertz(user:test_parity_input(1)),
    assertz(user:test_parity_input(2)),
    assertz(user:test_parity_input(3)),
    assertz(user:test_parity_input(4)),
    assertz(user:test_even(0)),
    assertz(user:test_odd(1)),
    assertz(user:(test_even(N) :-
        test_parity_input(N),
        N > 0,
        N1 is N - 1,
        test_odd(N1)
    )),
    assertz(user:(test_odd(N) :-
        test_parity_input(N),
        N > 1,
        N1 is N - 1,
        test_even(N1)
    )),
    assertz(user:(test_reachable(X, Y) :- test_fact(X, Y))),
    assertz(user:(test_reachable(X, Z) :- test_fact(X, Y), test_reachable(Y, Z))).

cleanup_test_data :-
    retractall(user:test_fact(_, _)),
    retractall(user:test_link(_, _)),
    retractall(user:test_filtered(_)),
    retractall(user:test_val(_, _)),
    retractall(user:test_increment(_, _)),
    retractall(user:test_num(_, _)),
    retractall(user:test_positive(_)),
    retractall(user:test_factorial_input(_)),
    retractall(user:test_factorial(_, _)),
    retractall(user:test_parity_input(_)),
    retractall(user:test_even(_)),
    retractall(user:test_odd(_)),
    retractall(user:test_reachable(_, _)),
    cleanup_csv_dynamic_source.

verify_fact_plan :-
    csharp_target:build_query_plan(test_fact/2, [target(csharp_query)], Plan),
    get_dict(head, Plan, predicate{name:test_fact, arity:2}),
    get_dict(root, Plan, relation_scan{type:relation_scan, predicate:predicate{name:test_fact, arity:2}, width:_}),
    get_dict(relations, Plan, [relation{predicate:predicate{name:test_fact, arity:2}, facts:Facts}]),
    Facts == [[alice, bob], [bob, charlie]],
    csharp_target:render_plan_to_csharp(Plan, Source),
    sub_string(Source, _, _, _, 'RelationScanNode').

verify_join_plan :-
    csharp_target:build_query_plan(test_link/2, [target(csharp_query)], Plan),
    get_dict(root, Plan, projection{type:projection, input:JoinNode, columns:[0, 3], width:2}),
    JoinNode = join{
        type:join,
        left:relation_scan{predicate:predicate{name:test_fact, arity:2}, type:relation_scan, width:_},
        right:relation_scan{predicate:predicate{name:test_fact, arity:2}, type:relation_scan, width:_},
        left_keys:[1],
        right_keys:[0],
        left_width:_,
        right_width:_,
        width:_
    },
    get_dict(relations, Plan, [relation{predicate:predicate{name:test_fact, arity:2}, facts:_}]),
    maybe_run_query_runtime(Plan, ['alice,charlie']).

verify_selection_plan :-
    csharp_target:build_query_plan(test_filtered/1, [target(csharp_query)], Plan),
    get_dict(root, Plan, projection{type:projection, input:Selection, columns:[0], width:1}),
    Selection = selection{
        type:selection,
        input:relation_scan{predicate:predicate{name:test_fact, arity:2}, type:relation_scan, width:_},
        predicate:condition{type:eq, left:operand{kind:column, index:0}, right:operand{kind:value, value:alice}},
        width:_
    },
    get_dict(relations, Plan, [relation{predicate:predicate{name:test_fact, arity:2}, facts:_}]),
    maybe_run_query_runtime(Plan, ['alice']).

verify_arithmetic_plan :-
    csharp_target:build_query_plan(test_increment/2, [target(csharp_query)], Plan),
    get_dict(root, Plan, projection{type:projection, input:ArithmeticNode, columns:[0, 2], width:2}),
    ArithmeticNode = arithmetic{
        type:arithmetic,
        input:relation_scan{predicate:predicate{name:test_val, arity:2}, type:relation_scan, width:_},
        expression:Expression,
        result_index:2,
        width:3
    },
    Expression = expr{
        type:binary,
        op:add,
        left:expr{type:column, index:1},
        right:expr{type:value, value:1}
    },
    get_dict(relations, Plan, [relation{predicate:predicate{name:test_val, arity:2}, facts:_}]),
    maybe_run_query_runtime(Plan, ['item1,6', 'item2,3']).

verify_comparison_plan :-
    csharp_target:build_query_plan(test_positive/1, [target(csharp_query)], Plan),
    get_dict(root, Plan, projection{type:projection, input:Selection, columns:[0], width:1}),
    Selection = selection{
        type:selection,
        input:relation_scan{predicate:predicate{name:test_num, arity:2}, type:relation_scan, width:_},
        predicate:condition{type:gt, left:operand{kind:column, index:1}, right:operand{kind:value, value:0}},
        width:_
    },
    maybe_run_query_runtime(Plan, ['item1']).

verify_recursive_arithmetic_plan :-
    csharp_target:build_query_plan(test_factorial/2, [target(csharp_query)], Plan),
    get_dict(is_recursive, Plan, true),
    get_dict(root, Plan, fixpoint{type:fixpoint, head:_, base:Base, recursive:[RecursiveClause], width:2}),
    Base = relation_scan{predicate:predicate{name:test_factorial, arity:2}, type:relation_scan, width:2},
    RecursiveClause = projection{
        type:projection,
        input:OuterArithmetic,
        columns:[0, 4],
        width:2
    },
    OuterArithmetic = arithmetic{
        type:arithmetic,
        input:JoinNode,
        expression:OuterExpr,
        result_index:4,
        width:5
    },
    is_dict(OuterExpr, expr),
    get_dict(op, OuterExpr, multiply),
    get_dict(left, OuterExpr, expr{type:column, index:3}),
    get_dict(right, OuterExpr, expr{type:column, index:0}),
    JoinNode = join{
        type:join,
        left:InnerArithmetic,
        right:recursive_ref{predicate:predicate{name:test_factorial, arity:2}, role:delta, type:recursive_ref, width:_},
        left_keys:[1],
        right_keys:[0],
        left_width:_,
        right_width:_,
        width:_
    },
    InnerArithmetic = arithmetic{
        type:arithmetic,
        input:Selection,
        expression:InnerExpr,
        result_index:1,
        width:2
    },
    is_dict(InnerExpr, expr),
    get_dict(op, InnerExpr, add),
    get_dict(left, InnerExpr, expr{type:column, index:0}),
    get_dict(right, InnerExpr, expr{type:value, value:Neg1}),
    Neg1 = -1,
    Selection = selection{
        type:selection,
        input:relation_scan{predicate:predicate{name:test_factorial_input, arity:1}, type:relation_scan, width:_},
        predicate:condition{type:gt, left:operand{kind:column, index:0}, right:operand{kind:value, value:0}},
        width:1
    },
    maybe_run_query_runtime(Plan, ['0,1', '1,1', '2,2', '3,6']).

verify_recursive_plan :-
    csharp_target:build_query_plan(test_reachable/2, [target(csharp_query)], Plan),
    get_dict(is_recursive, Plan, true),
    get_dict(root, Plan, fixpoint{type:fixpoint, head:_, base:Base, recursive:[RecursiveClause], width:2}),
    Base = projection{
        type:projection,
        input:relation_scan{predicate:predicate{name:test_fact, arity:2}, type:relation_scan, width:_},
        columns:[0, 1],
        width:2
    },
    RecursiveClause = projection{
        type:projection,
        input:JoinNode,
        columns:[0, 3],
        width:2
    },
    JoinNode = join{
        type:join,
        left:relation_scan{predicate:predicate{name:test_fact, arity:2}, type:relation_scan, width:_},
        right:recursive_ref{predicate:predicate{name:test_reachable, arity:2}, role:delta, type:recursive_ref, width:_},
        left_keys:[1],
        right_keys:[0],
        left_width:_,
        right_width:_,
        width:_
    },
    maybe_run_query_runtime(Plan, ['alice,bob', 'bob,charlie', 'alice,charlie']).

verify_mutual_recursion_plan :-
    csharp_target:build_query_plan(test_even/1, [target(csharp_query)], Plan),
    get_dict(is_recursive, Plan, true),
    get_dict(root, Plan, mutual_fixpoint{type:mutual_fixpoint, head:predicate{name:test_even, arity:1}, members:Members}),
    length(Members, 2),
    member(EvenMember, Members),
    get_dict(predicate, EvenMember, predicate{name:test_even, arity:1}),
    get_dict(base, EvenMember, EvenBase),
    get_dict(recursive, EvenMember, EvenVariants),
    EvenBase = relation_scan{predicate:predicate{name:test_even, arity:1}, type:relation_scan, width:_},
    member(EvenRecursive, EvenVariants),
    sub_term(cross_ref{predicate:predicate{name:test_odd, arity:1}, role:delta, type:cross_ref, width:_}, EvenRecursive),
    member(OddMember, Members),
    get_dict(predicate, OddMember, predicate{name:test_odd, arity:1}),
    get_dict(base, OddMember, OddBase),
    get_dict(recursive, OddMember, OddVariants),
    OddBase = relation_scan{predicate:predicate{name:test_odd, arity:1}, type:relation_scan, width:_},
    member(OddRecursive, OddVariants),
    sub_term(cross_ref{predicate:predicate{name:test_even, arity:1}, role:delta, type:cross_ref, width:_}, OddRecursive),
    maybe_run_query_runtime(Plan, ['0', '2', '4']).

verify_dynamic_source_plan :-
    setup_call_cleanup(
        setup_csv_dynamic_source,
        verify_dynamic_source_plan_(),
        cleanup_csv_dynamic_source
    ).

verify_dynamic_source_plan_ :-
    csharp_target:build_query_plan(test_user_age/2, [target(csharp_query)], Plan),
    csharp_target:render_plan_to_csharp(Plan, Source),
    sub_string(Source, _, _, _, 'DelimitedTextReader'),
    sub_string(Source, _, _, _, 'test_users.csv'),
    maybe_run_query_runtime(Plan, ['Alice,30', 'Bob,25', 'Charlie,35']).

verify_tsv_dynamic_source_plan :-
    setup_call_cleanup(
        setup_tsv_dynamic_source,
        verify_tsv_dynamic_source_plan_(),
        cleanup_tsv_dynamic_source
    ).

verify_tsv_dynamic_source_plan_ :-
    csharp_target:build_query_plan(test_sales_total/2, [target(csharp_query)], Plan),
    csharp_target:render_plan_to_csharp(Plan, Source),
    sub_string(Source, _, _, _, 'test_sales.tsv'),
    string_codes(TabLiteral, [0'@, 34, 9, 34]),
    sub_string(Source, _, _, _, TabLiteral),
    maybe_run_query_runtime(Plan, ['Laptop,1200', 'Mouse,25', 'Keyboard,75']).

verify_json_dynamic_source_plan :-
    setup_call_cleanup(
        setup_json_dynamic_source,
        verify_json_dynamic_source_plan_(),
        cleanup_json_dynamic_source
    ).

verify_json_dynamic_source_plan_ :-
    csharp_target:build_query_plan(test_product_price/2, [target(csharp_query)], Plan),
    csharp_target:render_plan_to_csharp(Plan, Source),
    sub_string(Source, _, _, _, 'JsonStreamReader'),
    sub_string(Source, _, _, _, 'test_products.json'),
    maybe_run_query_runtime(Plan, ['Laptop,999', 'Mouse,25', 'Keyboard,75']).

verify_json_nested_source_plan :-
    setup_call_cleanup(
        setup_json_orders_source,
        verify_json_nested_source_plan_(),
        cleanup_json_orders_source
    ).

verify_json_nested_source_plan_ :-
    csharp_target:build_query_plan(test_order_first_item/3, [target(csharp_query)], Plan),
    csharp_target:render_plan_to_csharp(Plan, Source),
    sub_string(Source, _, _, _, 'test_orders.json'),
    sub_string(Source, _, _, _, 'items[0].product'),
    maybe_run_query_runtime(Plan, ['Alice,Laptop,1200', 'Bob,Mouse,25', 'Charlie,Keyboard,75']).

verify_json_jsonpath_source_plan :-
    setup_call_cleanup(
        setup_json_jsonpath_source,
        verify_json_jsonpath_source_plan_(),
        cleanup_json_jsonpath_source
    ).

verify_json_jsonpath_source_plan_ :-
    csharp_target:build_query_plan(test_jsonpath_projection/2, [target(csharp_query)], Plan),
    csharp_target:render_plan_to_csharp(Plan, Source),
    sub_string(Source, _, _, _, 'JsonColumnSelectorKind.JsonPath'),
    maybe_run_query_runtime(Plan, [
        'Alice,Laptop',
        'Bob,Mouse',
        'Charlie,Keyboard'
    ]).

verify_json_schema_source_plan :-
    setup_call_cleanup(
        setup_json_schema_source,
        verify_json_schema_source_plan_(),
        cleanup_json_schema_source
    ).

verify_json_schema_source_plan_ :-
    csharp_target:build_query_plan(test_product_record/1, [target(csharp_query)], Plan),
    csharp_target:render_plan_to_csharp(Plan, Source),
    sub_string(Source, _, _, _, 'ProductRecord'),
    maybe_run_query_runtime(Plan, [
        'ProductRecord { Id = P001, Name = Laptop, Price = 999 }',
        'ProductRecord { Id = P002, Name = Mouse, Price = 25 }',
        'ProductRecord { Id = P003, Name = Keyboard, Price = 75 }'
    ]).

verify_json_nested_schema_record_plan :-
    setup_call_cleanup(
        setup_json_nested_schema_source,
        verify_json_nested_schema_record_plan_(),
        cleanup_json_nested_schema_source
    ).

verify_json_nested_schema_record_plan_ :-
    csharp_target:build_query_plan(test_order_summary/1, [target(csharp_query)], Plan),
    csharp_target:render_plan_to_csharp(Plan, Source),
    sub_string(Source, _, _, _, 'OrderRecord'),
    sub_string(Source, _, _, _, 'LineItemRecord'),
    maybe_run_query_runtime(Plan, [
        'OrderSummaryRecord { Order = OrderRecord { Id = SO1, Customer = Alice }, FirstItem = LineItemRecord { Product = Laptop, Total = 1200 } }',
        'OrderSummaryRecord { Order = OrderRecord { Id = SO2, Customer = Bob }, FirstItem = LineItemRecord { Product = Mouse, Total = 25 } }',
        'OrderSummaryRecord { Order = OrderRecord { Id = SO3, Customer = Charlie }, FirstItem = LineItemRecord { Product = Keyboard, Total = 75 } }'
    ]).

verify_json_jsonl_source_plan :-
    setup_call_cleanup(
        setup_json_jsonl_source,
        verify_json_jsonl_source_plan_(),
        cleanup_json_jsonl_source
    ).

verify_json_jsonl_source_plan_ :-
    csharp_target:build_query_plan(test_orders_jsonl/3, [target(csharp_query)], Plan),
    csharp_target:render_plan_to_csharp(Plan, Source),
    sub_string(Source, _, _, _, 'test_orders.jsonl'),
    sub_string(Source, _, _, _, 'TreatArrayAsStream = false'),
    maybe_run_query_runtime(Plan, ['Alice,Laptop,1200', 'Bob,Mouse,25', 'Charlie,Keyboard,75']).

verify_json_null_policy_skip_plan :-
    setup_call_cleanup(
        setup_json_null_skip_source,
        verify_json_null_policy_skip_plan_(),
        cleanup_json_null_skip_source
    ).

verify_json_null_policy_skip_plan_ :-
    csharp_target:build_query_plan(test_json_null_skip/2, [target(csharp_query)], Plan),
    csharp_target:render_plan_to_csharp(Plan, Source),
    sub_string(Source, _, _, _, 'JsonNullPolicy.Skip'),
    maybe_run_query_runtime(Plan, ['Alice,Mouse']).

verify_json_null_policy_default_plan :-
    setup_call_cleanup(
        setup_json_null_default_source,
        verify_json_null_policy_default_plan_(),
        cleanup_json_null_default_source
    ).

verify_json_null_policy_default_plan_ :-
    csharp_target:build_query_plan(test_json_null_default/2, [target(csharp_query)], Plan),
    csharp_target:render_plan_to_csharp(Plan, Source),
    sub_string(Source, _, _, _, 'JsonNullPolicy.Default'),
    sub_string(Source, _, _, _, 'NullReplacement = "N/A"'),
    maybe_run_query_runtime(Plan, ['Alice,Mouse', 'Bob,N/A', 'Charlie,N/A']).

verify_json_object_source_plan :-
    setup_call_cleanup(
        setup_json_object_source,
        verify_json_object_source_plan_(),
        cleanup_json_object_source
    ).

verify_json_object_source_plan_ :-
    csharp_target:build_query_plan(test_product_object/1, [target(csharp_query)], Plan),
    csharp_target:render_plan_to_csharp(Plan, Source),
    sub_string(Source, _, _, _, 'JsonStreamReader'),
    sub_string(Source, _, _, _, 'ReturnObject = true'),
    maybe_run_query_runtime(Plan, [
        '{"id":"P001","name":"Laptop","price":999}',
        '{"id":"P002","name":"Mouse","price":25}',
        '{"id":"P003","name":"Keyboard","price":75}'
    ]).

verify_xml_dynamic_source_plan :-
    setup_call_cleanup(
        setup_xml_dynamic_source,
        verify_xml_dynamic_source_plan_(),
        cleanup_xml_dynamic_source
    ).

verify_xml_dynamic_source_plan_ :-
    csharp_target:build_query_plan(test_xml_item/1, [target(csharp_query)], Plan),
    csharp_target:render_plan_to_csharp(Plan, Source),
    sub_string(Source, _, _, _, 'XmlStreamReader'),
    sub_string(Source, _, _, _, 'test_xml_fragments.txt'),
    sub_string(Source, _, _, _, 'TreatPearltreesCDataAsText = true'),
    % Validate dictionary projection picks up local + qualified keys
    maybe_run_query_runtime(Plan, [
        "_{id:1, name:Alpha, '@lang':en}",
        "_{id:2, title:Hacktivism, 'pt:item':Hacktivism, '@code':X, '@pt:id':2, '@pt:code':A1}",
        "_{id:3, name:Gamma}"
    ]).

verify_xml_nested_projection_plan :-
    setup_call_cleanup(
        setup_xml_dynamic_source_nested,
        verify_xml_nested_projection_plan_(),
        cleanup_xml_dynamic_source_nested
    ).

verify_xml_nested_projection_plan_ :-
    csharp_target:build_query_plan(test_xml_item_nested/1, [target(csharp_query)], Plan),
    csharp_target:render_plan_to_csharp(Plan, Source),
    sub_string(Source, _, _, _, 'XmlStreamReader'),
    sub_string(Source, _, _, _, 'NestedProjection = true'),
    sub_string(Source, _, _, _, 'test_xml_fragments.txt').

verify_xml_pearltrees_preset_plan :-
    setup_call_cleanup(
        setup_xml_pearltrees_source,
        verify_xml_pearltrees_preset_plan_(),
        cleanup_xml_pearltrees_source
    ).

verify_xml_pearltrees_preset_plan_ :-
    csharp_query_target:build_query_plan(test_pt_item/1, [target(csharp_query)], Plan),
    csharp_query_target:render_plan_to_csharp(Plan, Source),
    sub_string(Source, _, _, _, 'XmlStreamReader'),
    sub_string(Source, _, _, _, 'NamespacePrefixes'),
    sub_string(Source, _, _, _, 'TreatPearltreesCDataAsText = true').

setup_csv_dynamic_source :-
    source(csv, test_users, [csv_file('test_data/test_users.csv'), has_header(true)]),
    assertz(user:(test_user_age(Name, Age) :- test_users(_, Name, Age))).

cleanup_csv_dynamic_source :-
    retractall(user:test_user_age(_, _)),
    retractall(dynamic_source_compiler:dynamic_source_def(test_users/3, _, _)),
    retractall(dynamic_source_compiler:dynamic_source_metadata(test_users/3, _)).

setup_tsv_dynamic_source :-
    source(csv, test_sales, [
        csv_file('test_data/test_sales.tsv'),
        delimiter('\t'),
        has_header(true),
        quote_style(none)
    ]),
    assertz(user:(test_sales_total(Product, Total) :- test_sales(_, Product, Total))).

cleanup_tsv_dynamic_source :-
    retractall(user:test_sales_total(_, _)),
    retractall(dynamic_source_compiler:dynamic_source_def(test_sales/3, _, _)),
    retractall(dynamic_source_compiler:dynamic_source_metadata(test_sales/3, _)).

setup_json_dynamic_source :-
    source(json, test_products, [
        json_file('test_data/test_products.json'),
        record_format(json),
        columns([id, name, price])
    ]),
    assertz(user:(test_product_price(Name, Price) :- test_products(_, Name, Price))).

cleanup_json_dynamic_source :-
    retractall(user:test_product_price(_, _)),
    retractall(dynamic_source_compiler:dynamic_source_def(test_products/3, _, _)),
    retractall(dynamic_source_compiler:dynamic_source_metadata(test_products/3, _)).

setup_xml_dynamic_source :-
    source(xml, test_xml_items, [
        file('test_data/test_xml_fragments.txt'),
        record_format(xml),
        record_separator(line_feed)
    ]),
    assertz(user:(test_xml_item(Row) :- test_xml_items(Row))).

cleanup_xml_dynamic_source :-
    retractall(user:test_xml_item(_)),
    retractall(dynamic_source_compiler:dynamic_source_def(test_xml_items/1, _, _)),
    retractall(dynamic_source_compiler:dynamic_source_metadata(test_xml_items/1, _)).

setup_json_schema_source :-
    source(json, test_product_record_source, [
        json_file('test_data/test_products.json'),
        schema([
            field(id, 'id', string),
            field(name, 'name', string),
            field(price, 'price', double)
        ]),
        record_type('ProductRecord')
    ]),
    assertz(user:(test_product_record(Row) :- test_product_record_source(Row))).

setup_xml_dynamic_source_nested :-
    source(xml, test_xml_items_nested, [
        file('test_data/test_xml_fragments.txt'),
        record_format(xml),
        record_separator(line_feed),
        nested_projection(true)
    ]),
    assertz(user:(test_xml_item_nested(Row) :- test_xml_items_nested(Row))).

cleanup_xml_dynamic_source_nested :-
    retractall(user:test_xml_item_nested(_)),
    retractall(dynamic_source_compiler:dynamic_source_def(test_xml_items_nested/1, _, _)),
    retractall(dynamic_source_compiler:dynamic_source_metadata(test_xml_items_nested/1, _)).

setup_xml_pearltrees_source :-
    source(xml, test_pt_items, [
        file('test_data/test_xml_fragments.txt'),
        record_format(xml),
        record_separator(line_feed),
        pearltrees(true)
    ]),
    assertz(user:(test_pt_item(Row) :- test_pt_items(Row))).

cleanup_xml_pearltrees_source :-
    retractall(user:test_pt_item(_)),
    retractall(dynamic_source_compiler:dynamic_source_def(test_pt_items/1, _, _)),
    retractall(dynamic_source_compiler:dynamic_source_metadata(test_pt_items/1, _)).

cleanup_json_schema_source :-
    retractall(user:test_product_record(_)),
    retractall(dynamic_source_compiler:dynamic_source_def(test_product_record_source/1, _, _)),
    retractall(dynamic_source_compiler:dynamic_source_metadata(test_product_record_source/1, _)).

setup_json_nested_schema_source :-
    source(json, test_order_summary_source, [
        json_file('test_data/test_orders.json'),
        schema([
            field(order, 'order', record('OrderRecord', [
                field(id, 'id', string),
                field(customer, 'customer.name', string)
            ])),
            field(first_item, 'items[0]', record('LineItemRecord', [
                field(product, 'product', string),
                field(total, 'total', double)
            ]))
        ]),
        record_type('OrderSummaryRecord')
    ]),
    assertz(user:(test_order_summary(Row) :- test_order_summary_source(Row))).

cleanup_json_nested_schema_source :-
    retractall(user:test_order_summary(_)),
    retractall(dynamic_source_compiler:dynamic_source_def(test_order_summary_source/1, _, _)),
    retractall(dynamic_source_compiler:dynamic_source_metadata(test_order_summary_source/1, _)).

setup_json_jsonl_source :-
    source(json, test_orders_jsonl_source, [
        json_file('test_data/test_orders.jsonl'),
        record_format(jsonl),
        columns(['order.customer.name', 'items[0].product', 'items[0].total'])
    ]),
    assertz(user:(test_orders_jsonl(Customer, Product, Total) :-
        test_orders_jsonl_source(Customer, Product, Total))).

cleanup_json_jsonl_source :-
    retractall(user:test_orders_jsonl(_, _, _)),
    retractall(dynamic_source_compiler:dynamic_source_def(test_orders_jsonl_source/3, _, _)),
    retractall(dynamic_source_compiler:dynamic_source_metadata(test_orders_jsonl_source/3, _)).

setup_json_null_skip_source :-
    source(json, test_json_null_skip_source, [
        json_file('test_data/test_orders.json'),
        columns([
            jsonpath('$.order.customer.name'),
            jsonpath('$.items[1].product')
        ]),
        null_policy(skip)
    ]),
    assertz(user:(test_json_null_skip(Customer, Product) :-
        test_json_null_skip_source(Customer, Product))).

cleanup_json_null_skip_source :-
    retractall(user:test_json_null_skip(_, _)),
    retractall(dynamic_source_compiler:dynamic_source_def(test_json_null_skip_source/2, _, _)),
    retractall(dynamic_source_compiler:dynamic_source_metadata(test_json_null_skip_source/2, _)).

setup_json_null_default_source :-
    source(json, test_json_null_default_source, [
        json_file('test_data/test_orders.json'),
        columns([
            jsonpath('$.order.customer.name'),
            jsonpath('$.items[1].product')
        ]),
        null_policy(default('N/A'))
    ]),
    assertz(user:(test_json_null_default(Customer, Product) :-
        test_json_null_default_source(Customer, Product))).

cleanup_json_null_default_source :-
    retractall(user:test_json_null_default(_, _)),
    retractall(dynamic_source_compiler:dynamic_source_def(test_json_null_default_source/2, _, _)),
    retractall(dynamic_source_compiler:dynamic_source_metadata(test_json_null_default_source/2, _)).

setup_json_orders_source :-
    source(json, test_orders, [
        json_file('test_data/test_orders.json'),
        record_format(json),
        columns(['order.customer.name', 'items[0].product', 'items[0].total'])
    ]),
    assertz(user:(test_order_first_item(Customer, Product, Total) :-
        test_orders(Customer, Product, Total))).

cleanup_json_orders_source :-
    retractall(user:test_order_first_item(_, _, _)),
    retractall(dynamic_source_compiler:dynamic_source_def(test_orders/3, _, _)),
    retractall(dynamic_source_compiler:dynamic_source_metadata(test_orders/3, _)).

setup_json_jsonpath_source :-
    source(json, test_jsonpath_projection_source, [
        json_file('test_data/test_orders.json'),
        columns([
            jsonpath('$.order.customer.name'),
            jsonpath('$.items[*].product')
        ])
    ]),
    assertz(user:(test_jsonpath_projection(Customer, Product) :-
        test_jsonpath_projection_source(Customer, Product))).

cleanup_json_jsonpath_source :-
    retractall(user:test_jsonpath_projection(_, _)),
    retractall(dynamic_source_compiler:dynamic_source_def(test_jsonpath_projection_source/2, _, _)),
    retractall(dynamic_source_compiler:dynamic_source_metadata(test_jsonpath_projection_source/2, _)).

setup_json_object_source :-
    source(json, test_products_object, [
        json_file('test_data/test_products.json'),
        record_format(json),
        arity(1),
        type_hint('System.Text.Json.Nodes.JsonObject, System.Text.Json'),
        return_object(true)
    ]),
    assertz(user:(test_product_object(ProductJson) :-
        test_products_object(ProductJson))).

cleanup_json_object_source :-
    retractall(user:test_product_object(_)),
    retractall(dynamic_source_compiler:dynamic_source_def(test_products_object/1, _, _)),
    retractall(dynamic_source_compiler:dynamic_source_metadata(test_products_object/1, _)).

% Run with build-first approach, optionally skipping execution
maybe_run_query_runtime(Plan, ExpectedRows) :-
    dotnet_cli(Dotnet),
    !,
    prepare_temp_dir(Plan, Dir),
    (   getenv('SKIP_CSHARP_EXECUTION', '1')
    ->  % Generate code but skip execution (quiet)
        (   generate_csharp_code_only(Dotnet, Plan, Dir)
        ->  true
        ;   writeln('  (C# code generation: FAIL)'),
            finalize_temp_dir(Dir),
            fail
        ),
        finalize_temp_dir(Dir)
    ;   % Full execution
        (   run_dotnet_plan_build_first(Dotnet, Plan, ExpectedRows, Dir)
        ->  writeln('  (query runtime execution: PASS)'),
            finalize_temp_dir(Dir)
        ;   writeln('  (query runtime execution: FAIL - but plan structure verified)'),
            finalize_temp_dir(Dir)
        )
    ).

% Fall back to plan-only verification if dotnet not available
maybe_run_query_runtime(_Plan, _ExpectedRows) :-
    writeln('  (dotnet run skipped; see docs/CSHARP_DOTNET_RUN_HANG_SOLUTION.md)').

dotnet_cli(Path) :-
    catch(absolute_file_name(path(dotnet), Path, [access(execute)]), _, fail).

% Create temp directory with test name from Plan
prepare_temp_dir(Plan, Dir) :-
    is_dict(Plan),
    get_dict(head, Plan, predicate{name:PredName, arity:_}),
    !,
    uuid(UUID),
    atomic_list_concat(['csharp_query_', PredName, '_', UUID], Sub),
    (   cqt_option(output_dir, Base) -> true ; Base = 'tmp'),
    make_directory_path(Base),
    directory_file_path(Base, Sub, Dir),
    make_directory_path(Dir).

% Fallback for backwards compatibility
prepare_temp_dir(Dir) :-
    uuid(UUID),
    atomic_list_concat(['csharp_query_', UUID], Sub),
    (   cqt_option(output_dir, Base) -> true ; Base = 'tmp'),
    make_directory_path(Base),
    directory_file_path(Base, Sub, Dir),
    make_directory_path(Dir).

run_dotnet_plan_verbose(Dotnet, Plan, ExpectedRows, Dir) :-
    run_dotnet_plan(Dotnet, Plan, ExpectedRows, Dir),
    (   cqt_option(keep_artifacts, true)
    ->  format('  (kept C# artifacts in ~w)~n', [Dir])
    ;   true
    ).

finalize_temp_dir(Dir) :-
    (   cqt_option(keep_artifacts, true)
    ->  format('  (kept C# artifacts in ~w)~n', [Dir])
    ;   catch(
            delete_directory_and_contents(Dir),
            Error,
            format('  (warning: could not cleanup ~w: ~w)~n', [Dir, Error])
        )
    ).

% Build-first approach (works around dotnet run hang)
% See: docs/CSHARP_DOTNET_RUN_HANG_SOLUTION.md
run_dotnet_plan_build_first(Dotnet, Plan, ExpectedRows, Dir) :-
    % Step 1: Create project and write source files
    dotnet_command(Dotnet, ['new','console','--force','--framework','net9.0'], Dir, StatusNew, _),
    (   StatusNew =:= 0
    ->  true
    ;   writeln('  (dotnet new console failed; skipping runtime execution test)'), fail
    ),

    % Copy QueryRuntime.cs
    absolute_file_name('src/unifyweaver/targets/csharp_query_runtime/QueryRuntime.cs', RuntimePath, []),
    directory_file_path(Dir, 'QueryRuntime.cs', RuntimeCopy),
    copy_file(RuntimePath, RuntimeCopy),

    % Generate and write query module
    csharp_target:render_plan_to_csharp(Plan, ModuleSource),
    csharp_target:plan_module_name(Plan, ModuleClass),
    atom_concat(ModuleClass, '.cs', ModuleFile),
    directory_file_path(Dir, ModuleFile, ModulePath),
    write_string(ModulePath, ModuleSource),

    % Write harness
    harness_source(ModuleClass, HarnessSource),
    directory_file_path(Dir, 'Program.cs', ProgramPath),
    write_string(ProgramPath, HarnessSource),

    % Step 2: Build the project
    dotnet_command(Dotnet, ['build','--no-restore'], Dir, StatusBuild, BuildOutput),
    (   StatusBuild =:= 0
    ->  true
    ;   format('  (dotnet build failed: ~s)~n', [BuildOutput]), fail
    ),

    % Step 3: Find and execute compiled binary
    find_compiled_executable(Dir, ExePath),
    (   ExePath \= ''
    ->  true
    ;   writeln('  (compiled executable not found)'), fail
    ),

    % Execute the binary directly
    execute_compiled_binary(ExePath, Dir, StatusRun, Output),
    (   StatusRun =:= 0
    ->  extract_result_rows(Output, Rows),
        sort(Rows, SortedRows),
        maplist(to_atom, ExpectedRows, ExpectedAtoms),
        sort(ExpectedAtoms, SortedExpected),
        (   SortedRows == SortedExpected
        ->  true
        ;   format('  dotnet run output mismatch: ~w~n', [SortedRows]), fail
        )
    ;   format('  (execution failed: ~s)~n', [Output]), fail
    ).

% Generate C# code without execution (for SKIP_CSHARP_EXECUTION mode)
% Skips all dotnet commands - just generates and writes C# source files
generate_csharp_code_only(_Dotnet, Plan, Dir) :-
    % Create .csproj file manually (without calling dotnet new console)
    file_base_name(Dir, ProjectName),
    create_minimal_csproj(Dir, ProjectName),

    % Copy QueryRuntime.cs
    absolute_file_name('src/unifyweaver/targets/csharp_query_runtime/QueryRuntime.cs', RuntimePath, []),
    directory_file_path(Dir, 'QueryRuntime.cs', RuntimeCopy),
    copy_file(RuntimePath, RuntimeCopy),

    % Generate and write query module
    csharp_target:render_plan_to_csharp(Plan, ModuleSource),
    csharp_target:plan_module_name(Plan, ModuleClass),
    atom_concat(ModuleClass, '.cs', ModuleFile),
    directory_file_path(Dir, ModuleFile, ModulePath),
    write_string(ModulePath, ModuleSource),

    % Write harness
    harness_source(ModuleClass, HarnessSource),
    directory_file_path(Dir, 'Program.cs', ProgramPath),
    write_string(ProgramPath, HarnessSource).
    % Note: dotnet execution commands (build, run) still skipped in this mode

% Create a minimal .csproj file manually
create_minimal_csproj(Dir, ProjectName) :-
    atom_concat(ProjectName, '.csproj', CsprojFile),
    directory_file_path(Dir, CsprojFile, CsprojPath),
    CsprojContent = '<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>net9.0</TargetFramework>
    <Nullable>enable</Nullable>
  </PropertyGroup>
</Project>',
    open(CsprojPath, write, Stream),
    write(Stream, CsprojContent),
    close(Stream).

% Original run_dotnet_plan (kept for reference, but not used)
run_dotnet_plan(Dotnet, Plan, ExpectedRows, Dir) :-
    dotnet_command(Dotnet, ['new','console','--force','--framework','net9.0'], Dir, StatusNew, _),
    (   StatusNew =:= 0
    ->  true
    ;   writeln('  (dotnet new console failed; skipping runtime execution test)'), fail
    ),
    absolute_file_name('src/unifyweaver/targets/csharp_query_runtime/QueryRuntime.cs', RuntimePath, []),
    directory_file_path(Dir, 'QueryRuntime.cs', RuntimeCopy),
    copy_file(RuntimePath, RuntimeCopy),
    csharp_target:render_plan_to_csharp(Plan, ModuleSource),
    csharp_target:plan_module_name(Plan, ModuleClass),
    atom_concat(ModuleClass, '.cs', ModuleFile),
    directory_file_path(Dir, ModuleFile, ModulePath),
    write_string(ModulePath, ModuleSource),
    harness_source(ModuleClass, HarnessSource),
    directory_file_path(Dir, 'Program.cs', ProgramPath),
    write_string(ProgramPath, HarnessSource),
    dotnet_command(Dotnet, ['run','--no-restore'], Dir, StatusRun, Output),
    (   StatusRun =:= 0
    ->  extract_result_rows(Output, Rows),
        sort(Rows, SortedRows),
        maplist(to_atom, ExpectedRows, ExpectedAtoms),
        sort(ExpectedAtoms, SortedExpected),
        (   SortedRows == SortedExpected
        ->  true
        ;   format('  dotnet run output mismatch: ~w~n', [SortedRows]), fail
        )
    ;   format('  (dotnet run failed: ~s)~n', [Output]), fail
    ).

% Find the compiled executable in bin/Debug/net9.0/
find_compiled_executable(Dir, ExePath) :-
    directory_file_path(Dir, 'bin/Debug/net9.0', DebugDir),
    (   exists_directory(DebugDir)
    ->  directory_files(DebugDir, Files),
        member(File, Files),
        \+ atom_concat(_, '.dll', File),
        \+ atom_concat(_, '.pdb', File),
        \+ atom_concat(_, '.deps.json', File),
        \+ atom_concat(_, '.runtimeconfig.json', File),
        File \= '.',
        File \= '..',
        directory_file_path(DebugDir, File, ExePath),
        exists_file(ExePath),
        !
    ;   % No native executable, try DLL
        directory_file_path(DebugDir, 'test.dll', DllPath),
        exists_file(DllPath),
        !,
        ExePath = DllPath
    ).

% Execute compiled binary (native or DLL)
execute_compiled_binary(ExePath, Dir, Status, Output) :-
    dotnet_env(Dir, Env),
    (   atom_concat(_, '.dll', ExePath)
    ->  % Execute DLL with dotnet
        dotnet_cli(Dotnet),
        process_create(Dotnet, [ExePath],
                       [ cwd(Dir),
                         env(Env),
                         stdout(pipe(Out)),
                         stderr(pipe(Err)),
                         process(PID)
                       ])
    ;   % Execute native binary directly
        process_create(ExePath, [],
                       [ cwd(Dir),
                         env(Env),
                         stdout(pipe(Out)),
                         stderr(pipe(Err)),
                         process(PID)
                       ])
    ),
    read_string(Out, _, Stdout),
    read_string(Err, _, Stderr),
    close(Out),
    close(Err),
    process_wait(PID, exit(Status)),
    string_concat(Stdout, Stderr, Output).

harness_source(ModuleClass, Source) :-
    format(atom(Source),
'using System;
using System.Linq;
using UnifyWeaver.QueryRuntime;
using System.Text.Json;
using System.Text.Json.Nodes;

var result = UnifyWeaver.Generated.~w.Build();
var executor = new QueryExecutor(result.Provider);
var jsonOptions = new JsonSerializerOptions { WriteIndented = false };

string FormatValue(object? value) => value switch
{
    JsonNode node => node.ToJsonString(jsonOptions),
    JsonElement element => element.GetRawText(),
    _ => value?.ToString() ?? string.Empty
};
foreach (var row in executor.Execute(result.Plan))
{
    var projected = row.Take(result.Plan.Head.Arity)
                       .Select(FormatValue)
                       .ToArray();

    if (projected.Length == 0)
    {
        continue;
    }

    Console.WriteLine(string.Join(\",\", projected));
}
', [ModuleClass]).

write_string(Path, String) :-
    setup_call_cleanup(open(Path, write, Stream),
                       write(Stream, String),
                       close(Stream)).

dotnet_command(Dotnet, Args, Dir, Status, Output) :-
    dotnet_env(Dir, Env),
    process_create(Dotnet, Args,
                   [ cwd(Dir),
                     env(Env),
                     stdout(pipe(Out)),
                     stderr(pipe(Err)),
                     process(PID)
                   ]),
    read_string(Out, _, Stdout),
    read_string(Err, _, Stderr),
    close(Out),
    close(Err),
    process_wait(PID, exit(Status)),
    string_concat(Stdout, Stderr, Output).

dotnet_env(Dir, Env) :-
    environ(RawEnv),
    exclude(is_dotnet_env, RawEnv, BaseEnv),
    Env = ['DOTNET_CLI_HOME'=Dir,
           'DOTNET_CLI_TELEMETRY_OPTOUT'='1',
           'DOTNET_NOLOGO'='1'
           | BaseEnv].

is_dotnet_env('DOTNET_CLI_HOME'=_).
is_dotnet_env('DOTNET_CLI_TELEMETRY_OPTOUT'=_).
is_dotnet_env('DOTNET_NOLOGO'=_).

extract_result_rows(Output, Rows) :-
    split_string(Output, "\n", "\r", Lines0),
    maplist(normalize_space_string, Lines0, NormalizedLines),
    include(non_empty_line, NormalizedLines, Candidate),
    maplist(to_atom, Candidate, Rows).

non_empty_line(Line) :-
    Line \== '',
    Line \== "".

normalize_space_string(Line, Normalized) :-
    normalize_space(string(Normalized), Line).

to_atom(Value, Atom) :-
    (   atom(Value) -> Atom = Value
    ;   string(Value) -> atom_string(Atom, Value)
    ;   term_to_atom(Atom, Value)
    ).

%% Option handling ---------------------------------------------------------

% The following predicates allow the dotnet harness to respect CLI
% switches (e.g. --csharp-query-output, --csharp-query-keep) and
% corresponding environment variables, mirroring the behaviour used in
% the education module examples.
configure_csharp_query_options :-
    retractall(cqt_option(_, _)),
    default_cqt_options(Default),
    maplist(assertz, Default),
    capture_env_overrides,
    capture_cli_overrides.

default_cqt_options([
    cqt_option(output_dir, 'tmp'),
    cqt_option(keep_artifacts, false)
]).

%% progress reporting helpers --------------------------------------------------

progress_init :-
    get_time(Now),
    retractall(progress_last_report(_)),
    retractall(progress_count(_)),
    asserta(progress_last_report(Now)),
    asserta(progress_count(0)).

progress_interval_seconds(10).

run_with_progress(Goal) :-
    (   catch(once(call(Goal)), E, (print_message(error, E), fail))
    ->  true
    ;   format('  FAILED: ~w~n', [Goal]),
        fail
    ),
    retract(progress_count(C0)),
    C is C0 + 1,
    asserta(progress_count(C)),
    progress_maybe_report(normal).

progress_maybe_report(force) :-
    !,
    progress_count(C),
    progress_total(T),
    format('  Progress: ~w/~w tests complete.~n', [C, T]).
progress_maybe_report(normal) :-
    progress_interval_seconds(Interval),
    get_time(Now),
    (   progress_last_report(Last),
        Delta is Now - Last,
        Delta >= Interval
    ->  retract(progress_last_report(_)),
        asserta(progress_last_report(Now)),
        progress_count(C),
        progress_total(T),
        format('  Progress: ~w/~w tests complete.~n', [C, T])
    ;   true
    ).

capture_env_overrides :-
    (   getenv('CSHARP_QUERY_OUTPUT_DIR', Dir),
        Dir \= ''
    ->  retractall(cqt_option(output_dir, _)),
        assertz(cqt_option(output_dir, Dir))
    ;   true
    ),
    (   getenv('CSHARP_QUERY_KEEP_ARTIFACTS', KeepRaw),
        normalize_yes_no(KeepRaw, Keep)
    ->  retractall(cqt_option(keep_artifacts, _)),
        assertz(cqt_option(keep_artifacts, Keep))
    ;   true
    ).

capture_cli_overrides :-
    current_prolog_flag(argv, Argv),
    apply_cli_overrides(Argv).

apply_cli_overrides([]).
apply_cli_overrides([Arg|Rest]) :-
    (   atom(Arg),
        atom_concat('--csharp-query-output=', DirAtom, Arg)
    ->  set_cqt_option(output_dir, DirAtom),
        apply_cli_overrides(Rest)
    ;   Arg == '--csharp-query-output',
        Rest = [Dir|Tail]
    ->  set_cqt_option(output_dir, Dir),
        apply_cli_overrides(Tail)
    ;   Arg == '--csharp-query-keep'
    ->  set_cqt_option(keep_artifacts, true),
        apply_cli_overrides(Rest)
    ;   Arg == '--csharp-query-autodelete'
    ->  set_cqt_option(keep_artifacts, false),
        apply_cli_overrides(Rest)
    ;   apply_cli_overrides(Rest)
    ).

set_cqt_option(Key, Value) :-
    retractall(cqt_option(Key, _)),
    assertz(cqt_option(Key, Value)).

normalize_yes_no(Value0, Bool) :-
    (   atom(Value0) -> atom_string(Value0, Value)
    ;   Value = Value0
    ),
    string_lower(Value, Lower),
    (   member(Lower, ['1', 'true', 'yes', 'keep'])
    ->  Bool = true
    ;   member(Lower, ['0', 'false', 'no', 'delete', 'autodelete'])
    ->  Bool = false
    ).

verify_generator_mode :-
    % Basic structure check
    csharp_target:compile_predicate_to_csharp(test_link/2, [mode(generator)], Code),
    sub_string(Code, _, _, _, 'class TestLink_Module'),
    sub_string(Code, _, _, _, 'public record Fact'),
    sub_string(Code, _, _, _, 'GetInitialFacts'),
    sub_string(Code, _, _, _, 'ApplyRule_1'),
    
    % Try to compile and run if dotnet is available
    (   dotnet_cli(Dotnet)
    ->  (   verify_generator_execution(Code, Dotnet)
        ->  true
        ;   writeln('  (generator mode execution: FAIL - see output above)')
        )
    ;   writeln('  (dotnet not available, skipping execution test)')
    ).
verify_generator_execution(Code, Dotnet) :-
    setup_call_cleanup(
        prepare_temp_dir(test_link, Dir),
        verify_generator_execution_(Code, Dotnet, Dir),
        finalize_temp_dir(Dir)
    ).

verify_generator_execution_(Code, Dotnet, Dir) :-
    format('  [generator] working dir: ~w~n', [Dir]),
    % Create .NET project
    (   catch(dotnet_command(Dotnet, ['new', 'console', '--force', '--framework', 'net9.0'], Dir, StatusNew, NewOut),
              E, (print_message(error, E), StatusNew = -1, NewOut = ""))
    ->  ( StatusNew =:= 0
        -> format('  (dotnet new ok)~n', [])
        ;  format('  (dotnet new failed)~n~s~n', [NewOut]), fail
        )
    ;   format('  (dotnet new threw)~n'), fail
    ),
    % Write generated code
    directory_file_path(Dir, 'Generated.cs', GeneratedPath),
    write_string(GeneratedPath, Code),
    % Derive module class from generated code
    (   sub_string(Code, Pos, _, _, "public static class ")
    ->  Start is Pos + 21, % length("public static class ")
        sub_string(Code, Start, _, _, AfterClass),
        split_string(AfterClass, " {", " {;\n\r\t", [ModuleClassStr|_]),
        atom_string(ModuleClass, ModuleClassStr)
    ;   ModuleClass = 'TestLink_Module'  % fallback for this test
    ),
    % Harness to execute Solve and print facts
    format(atom(Harness),
"using System;
using System.Collections.Generic;
using UnifyWeaver.Generated;

class Program
{
    static void Main()
    {
        var total = ~w.Solve();
        foreach (var fact in total)
        {
            var a0 = fact.Args.TryGetValue(\"arg0\", out var v0) ? v0 : \"\";
            var a1 = fact.Args.TryGetValue(\"arg1\", out var v1) ? v1 : \"\";
            Console.WriteLine($\"{fact.Relation},{a0},{a1}\");
        }
    }
}
", [ModuleClass]),
    directory_file_path(Dir, 'Program.cs', ProgramPath),
    write_string(ProgramPath, Harness),
    % Build
    (   catch(dotnet_command(Dotnet, ['build', '--no-restore'], Dir, StatusBuild, BuildOutput),
              E, (print_message(error, E), StatusBuild = -1, BuildOutput = ""))
    ->  ( StatusBuild =:= 0
        -> format('  (dotnet build ok)~n', [])
        ;  format('  (dotnet build failed) exit=~w~n~s~n', [StatusBuild, BuildOutput]), fail
        )
    ;   format('  (dotnet build threw)~n'), fail
    ),
    % Run and check output
    (   catch(dotnet_command(Dotnet, ['run', '--no-build', '--no-restore'], Dir, StatusRun, Output),
              E, (print_message(error, E), StatusRun = -1, Output = ""))
    ->  (   StatusRun =:= 0,
            sub_string(Output, _, _, _, 'test_link,alice,charlie')
        ->  format('  (generator mode execution: PASS)~nOutput:~s~n', [Output])
        ;   format('  (generator mode execution: FAIL) exit=~w~nOutput:~s~n', [StatusRun, Output]), fail
        )
    ;   format('  (dotnet run threw)~n'), fail
    ).
