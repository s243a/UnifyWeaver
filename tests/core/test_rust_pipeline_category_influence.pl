:- module(test_rust_pipeline_category_influence, [test_rust_pipeline_category_influence/0]).

:- use_module(library(plunit)).
:- use_module('../../src/unifyweaver/targets/rust_target').

cleanup_category_influence_benchmark :-
    catch(abolish(user:max_depth/1), _, true),
    catch(abolish(user:influence_dimension/1), _, true),
    catch(abolish(user:category_parent/2), _, true),
    catch(abolish(user:article_category/2), _, true),
    catch(abolish(user:root_category/1), _, true),
    catch(abolish(user:category_ancestor/3), _, true),
    catch(abolish(user:root_category_set/1), _, true),
    catch(abolish(user:article_root_weight/3), _, true),
    catch(abolish(user:category_influence/2), _, true),
    catch(abolish(user:run/0), _, true).

:- begin_tests(rust_pipeline_category_influence).

test(compile_category_influence_stage, [
    setup(user:consult('examples/benchmark/category_influence.pl')),
    cleanup(cleanup_category_influence_benchmark)
]) :-
    compile_rust_pipeline([category_influence/2], [pipeline_mode(generator)], Code),
    sub_string(Code, _, _, _, 'fn stage_category_influence'),
    \+ sub_string(Code, _, _, _, '// TODO: Implement stage logic'),
    sub_string(Code, _, _, _, 'matches!(fact.get("relation"), Some(Value::String(rel)) if rel == "root_category")'),
    sub_string(Code, _, _, _, 'matches!(f.get("relation"), Some(Value::String(rel)) if rel == "article_root_weight")'),
    sub_string(Code, _, _, _, 'let agg = values.iter().copied().sum::<f64>()'),
    sub_string(Code, _, _, _, 'if agg > 0'),
    sub_string(Code, _, _, _, 'Value::String("category_influence".to_string())').

test(compile_full_category_influence_pipeline, [
    setup(user:consult('examples/benchmark/category_influence.pl')),
    cleanup(cleanup_category_influence_benchmark)
]) :-
    compile_rust_pipeline([category_ancestor/3, article_root_weight/3, category_influence/2], [pipeline_mode(generator)], Code),
    sub_string(Code, _, _, _, 'fn stage_category_ancestor'),
    sub_string(Code, _, _, _, 'fn stage_article_root_weight'),
    sub_string(Code, _, _, _, 'fn stage_category_influence'),
    \+ sub_string(Code, _, _, _, 'fn stage_category_ancestor(input: Vec<HashMap<String, Value>>) -> Vec<HashMap<String, Value>> {\n    // TODO: Implement stage logic'),
    \+ sub_string(Code, _, _, _, 'fn stage_article_root_weight(input: Vec<HashMap<String, Value>>) -> Vec<HashMap<String, Value>> {\n    // TODO: Implement stage logic'),
    \+ sub_string(Code, _, _, _, 'fn stage_category_influence(input: Vec<HashMap<String, Value>>) -> Vec<HashMap<String, Value>> {\n    // TODO: Implement stage logic'),
    sub_string(Code, _, _, _, 'new_records.extend(stage_category_ancestor(current.clone()));'),
    sub_string(Code, _, _, _, 'new_records.extend(stage_article_root_weight(current.clone()));'),
    sub_string(Code, _, _, _, 'new_records.extend(stage_category_influence(current.clone()));'),
    sub_string(Code, _, _, _, 'matches!(f3.get("relation"), Some(Value::String(rel)) if rel == "category_ancestor")'),
    sub_string(Code, _, _, _, 'let rn_3 = (rn_2).powf((-value_to_f64(Some(&rv_4))));').

:- end_tests(rust_pipeline_category_influence).

test_rust_pipeline_category_influence :-
    run_tests([rust_pipeline_category_influence]).
