:- encoding(utf8).

:- use_module('../src/unifyweaver/targets/wam_ir_mode').
:- use_module(library(plunit)).

:- begin_tests(wam_ir_mode).

test(python_interpreter_defaults_to_items_bridge) :-
    wam_ir_mode(wam_python, interpreter, [], Mode),
    assertion(Mode == wam_items_bridge).

test(python_lowered_defaults_to_text) :-
    wam_ir_mode(wam_python, lowered, [], Mode),
    assertion(Mode == wam_text).

test(lua_r_and_elixir_interpreter_default_to_items_bridge) :-
    wam_ir_mode(wam_lua, interpreter, [], LuaMode),
    wam_ir_mode(wam_r, interpreter, [], RMode),
    wam_ir_mode(wam_elixir, interpreter, [], ElixirMode),
    assertion(LuaMode == wam_items_bridge),
    assertion(RMode == wam_items_bridge),
    assertion(ElixirMode == wam_items_bridge).

test(elixir_lowered_defaults_to_text) :-
    wam_ir_mode(wam_elixir, lowered, [], Mode),
    assertion(Mode == wam_text).

test(unknown_target_defaults_to_text) :-
    wam_ir_mode(wam_unknown, interpreter, [], Mode),
    assertion(Mode == wam_text).

test(explicit_option_overrides_default) :-
    wam_ir_mode(wam_python, interpreter, [wam_ir(wam_text)], Mode),
    assertion(Mode == wam_text).

test(rejects_unknown_mode, [throws(error(domain_error(wam_ir_mode, nonsense), _))]) :-
    wam_ir_mode(wam_python, interpreter, [wam_ir(nonsense)], _).

test(mode_predicates_classify_wam_and_text_skip) :-
    assertion(wam_ir_mode_uses_wam(wam_text)),
    assertion(wam_ir_mode_uses_wam(wam_items_bridge)),
    assertion(wam_ir_mode_uses_wam(wam_items_native)),
    assertion(\+ wam_ir_mode_uses_wam(direct_target)),
    assertion(wam_ir_mode_skips_text(wam_items_native)),
    assertion(wam_ir_mode_skips_text(direct_target)),
    assertion(\+ wam_ir_mode_skips_text(wam_items_bridge)).

:- end_tests(wam_ir_mode).
