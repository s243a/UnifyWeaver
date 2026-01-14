% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% data_binding.pl - Cross-platform reactive data binding system
%
% Provides declarative state and binding primitives that compile across
% React Native, Vue, Flutter, and SwiftUI targets.
%
% Usage:
%   use_module('src/unifyweaver/binding/data_binding').
%   generate_state(state(counter, 0, []), react_native, Code).

:- module(data_binding, [
    % State declarations
    state/3,                        % state(Name, Initial, Options)
    store/3,                        % store(Name, Slices, Options)
    computed/3,                     % computed(Name, Deps, Expression)
    effect/3,                       % effect(Name, Deps, Body)

    % Binding types
    bind/2,                         % bind(Target, Source)
    two_way/2,                      % two_way(Target, Source)
    on/3,                           % on(Element, Event, Handler)

    % Actions
    action/3,                       % action(Name, State, Updater)
    store_action/4,                 % store_action(Store, Name, Param, Updates)

    % Code generation - main entry points
    generate_state/3,               % generate_state(StateSpec, Target, Code)
    generate_store/3,               % generate_store(StoreSpec, Target, Code)
    generate_binding/3,             % generate_binding(BindingSpec, Target, Code)
    generate_computed/3,            % generate_computed(ComputedSpec, Target, Code)
    generate_effect/3,              % generate_effect(EffectSpec, Target, Code)
    generate_action/3,              % generate_action(ActionSpec, Target, Code)

    % Target-specific generators
    generate_react_native_state/2,
    generate_react_native_store/2,
    generate_react_native_computed/2,
    generate_react_native_effect/2,
    generate_react_native_binding/2,
    generate_react_native_action/2,

    generate_vue_state/2,
    generate_vue_store/2,
    generate_vue_computed/2,
    generate_vue_effect/2,
    generate_vue_binding/2,
    generate_vue_action/2,

    generate_flutter_state/2,
    generate_flutter_store/2,
    generate_flutter_computed/2,
    generate_flutter_effect/2,
    generate_flutter_binding/2,
    generate_flutter_action/2,

    generate_swiftui_state/2,
    generate_swiftui_store/2,
    generate_swiftui_computed/2,
    generate_swiftui_effect/2,
    generate_swiftui_binding/2,
    generate_swiftui_action/2,

    % Validation
    validate_state/2,               % validate_state(Spec, Errors)
    validate_binding/2,             % validate_binding(Spec, Errors)

    % Testing
    test_data_binding/0
]).

:- use_module(library(lists)).

% ============================================================================
% State Declarations - Term Constructors
% ============================================================================

%! state(+Name, +Initial, +Options) is det
%  Create a local state specification.
%  Options: type(Type), persist(bool), readonly(bool)
state(Name, Initial, Options) :-
    atom(Name),
    is_list(Options),
    state_spec(Name, Initial, Options).

state_spec(Name, Initial, Options) :-
    atom(Name),
    is_list(Options),
    \+ member(type(_), Options)
    ->  infer_type(Initial, Type),
        state_spec(Name, Initial, [type(Type)|Options])
    ;   true.

%! store(+Name, +Slices, +Options) is det
%  Create a global store specification.
store(Name, Slices, Options) :-
    atom(Name),
    is_list(Slices),
    is_list(Options).

%! computed(+Name, +Deps, +Expression) is det
%  Create a computed/derived value specification.
computed(Name, Deps, Expression) :-
    atom(Name),
    is_list(Deps),
    (atom(Expression) ; string(Expression)).

%! effect(+Name, +Deps, +Body) is det
%  Create an effect/watcher specification.
effect(Name, Deps, Body) :-
    atom(Name),
    is_list(Deps),
    (atom(Body) ; string(Body)).

% ============================================================================
% Binding Types - Term Constructors
% ============================================================================

%! bind(+Target, +Source) is det
%  Create a one-way binding (read only).
bind(Target, Source) :-
    atom(Target),
    (atom(Source) ; string(Source)).

%! two_way(+Target, +Source) is det
%  Create a two-way binding (read/write).
two_way(Target, Source) :-
    atom(Target),
    atom(Source).

%! on(+Element, +Event, +Handler) is det
%  Create an event binding.
on(Element, Event, Handler) :-
    atom(Element),
    atom(Event),
    (atom(Handler) ; string(Handler)).

% ============================================================================
% Actions - Term Constructors
% ============================================================================

%! action(+Name, +State, +Updater) is det
%  Create a state action/mutation.
action(Name, State, Updater) :-
    atom(Name),
    atom(State),
    (atom(Updater) ; string(Updater)).

%! store_action(+Store, +Name, +Param, +Updates) is det
%  Create a store action with multiple updates.
store_action(Store, Name, Param, Updates) :-
    atom(Store),
    atom(Name),
    atom(Param),
    is_list(Updates).

% ============================================================================
% Type Inference
% ============================================================================

infer_type(Value, number) :- number(Value), !.
infer_type(Value, string) :- (atom(Value) ; string(Value)), !.
infer_type(Value, boolean) :- (Value = true ; Value = false), !.
infer_type(Value, list) :- is_list(Value), !.
infer_type(_, any).

% ============================================================================
% Code Generation - Main Dispatchers
% ============================================================================

%! generate_state(+StateSpec, +Target, -Code) is det
generate_state(state(Name, Initial, Options), Target, Code) :-
    (   Target = react_native
    ->  generate_react_native_state(state(Name, Initial, Options), Code)
    ;   Target = vue
    ->  generate_vue_state(state(Name, Initial, Options), Code)
    ;   Target = flutter
    ->  generate_flutter_state(state(Name, Initial, Options), Code)
    ;   Target = swiftui
    ->  generate_swiftui_state(state(Name, Initial, Options), Code)
    ;   Code = ""
    ).

%! generate_store(+StoreSpec, +Target, -Code) is det
generate_store(store(Name, Slices, Options), Target, Code) :-
    (   Target = react_native
    ->  generate_react_native_store(store(Name, Slices, Options), Code)
    ;   Target = vue
    ->  generate_vue_store(store(Name, Slices, Options), Code)
    ;   Target = flutter
    ->  generate_flutter_store(store(Name, Slices, Options), Code)
    ;   Target = swiftui
    ->  generate_swiftui_store(store(Name, Slices, Options), Code)
    ;   Code = ""
    ).

%! generate_computed(+ComputedSpec, +Target, -Code) is det
generate_computed(computed(Name, Deps, Expr), Target, Code) :-
    (   Target = react_native
    ->  generate_react_native_computed(computed(Name, Deps, Expr), Code)
    ;   Target = vue
    ->  generate_vue_computed(computed(Name, Deps, Expr), Code)
    ;   Target = flutter
    ->  generate_flutter_computed(computed(Name, Deps, Expr), Code)
    ;   Target = swiftui
    ->  generate_swiftui_computed(computed(Name, Deps, Expr), Code)
    ;   Code = ""
    ).

%! generate_effect(+EffectSpec, +Target, -Code) is det
generate_effect(effect(Name, Deps, Body), Target, Code) :-
    (   Target = react_native
    ->  generate_react_native_effect(effect(Name, Deps, Body), Code)
    ;   Target = vue
    ->  generate_vue_effect(effect(Name, Deps, Body), Code)
    ;   Target = flutter
    ->  generate_flutter_effect(effect(Name, Deps, Body), Code)
    ;   Target = swiftui
    ->  generate_swiftui_effect(effect(Name, Deps, Body), Code)
    ;   Code = ""
    ).

%! generate_binding(+BindingSpec, +Target, -Code) is det
generate_binding(Binding, Target, Code) :-
    (   Target = react_native
    ->  generate_react_native_binding(Binding, Code)
    ;   Target = vue
    ->  generate_vue_binding(Binding, Code)
    ;   Target = flutter
    ->  generate_flutter_binding(Binding, Code)
    ;   Target = swiftui
    ->  generate_swiftui_binding(Binding, Code)
    ;   Code = ""
    ).

%! generate_action(+ActionSpec, +Target, -Code) is det
generate_action(Action, Target, Code) :-
    (   Target = react_native
    ->  generate_react_native_action(Action, Code)
    ;   Target = vue
    ->  generate_vue_action(Action, Code)
    ;   Target = flutter
    ->  generate_flutter_action(Action, Code)
    ;   Target = swiftui
    ->  generate_swiftui_action(Action, Code)
    ;   Code = ""
    ).

% ============================================================================
% React Native Code Generation
% ============================================================================

%! generate_react_native_state(+StateSpec, -Code) is det
generate_react_native_state(state(Name, Initial, _Options), Code) :-
    capitalize_first(Name, SetterName),
    format_initial_value(Initial, react_native, InitialStr),
    format(atom(Code), 'const [~w, set~w] = useState(~w);', [Name, SetterName, InitialStr]).

%! generate_react_native_store(+StoreSpec, -Code) is det
generate_react_native_store(store(Name, Slices, Options), Code) :-
    capitalize_first(Name, StoreName),
    generate_rn_store_types(Slices, TypeDefs),
    generate_rn_store_state(Slices, StateDefs),
    generate_rn_store_actions(Slices, ActionDefs),
    (   member(persist(true), Options)
    ->  PersistImport = "import { persist, createJSONStorage } from 'zustand/middleware';\nimport AsyncStorage from '@react-native-async-storage/async-storage';\n",
        format(atom(Code), "import { create } from 'zustand';
~w
~w

export const use~wStore = create(
  persist(
    (set, get) => ({
~w
~w
    }),
    {
      name: '~w-storage',
      storage: createJSONStorage(() => AsyncStorage),
    }
  )
);", [PersistImport, TypeDefs, StoreName, StateDefs, ActionDefs, Name])
    ;   format(atom(Code), "import { create } from 'zustand';

~w

export const use~wStore = create((set, get) => ({
~w
~w
}));", [TypeDefs, StoreName, StateDefs, ActionDefs])
    ).

generate_rn_store_types(Slices, TypeDefs) :-
    findall(TypeDef, (
        member(slice(SliceName, Fields, _), Slices),
        generate_rn_slice_type(SliceName, Fields, TypeDef)
    ), TypeDefList),
    atomic_list_concat(TypeDefList, '\n', TypeDefs).

generate_rn_slice_type(SliceName, Fields, TypeDef) :-
    findall(FieldDef, (
        member(field(FName, FType, _), Fields),
        ts_type(FType, TSType),
        format(atom(FieldDef), '  ~w: ~w;', [FName, TSType])
    ), FieldDefs),
    atomic_list_concat(FieldDefs, '\n', FieldsStr),
    format(atom(TypeDef), 'interface ~wSlice {\n~w\n}', [SliceName, FieldsStr]).

generate_rn_store_state(Slices, StateDefs) :-
    findall(StateDef, (
        member(slice(_, Fields, _), Slices),
        member(field(FName, _, Default), Fields),
        format_initial_value(Default, react_native, DefaultStr),
        format(atom(StateDef), '  ~w: ~w,', [FName, DefaultStr])
    ), StateDefList),
    atomic_list_concat(StateDefList, '\n', StateDefs).

generate_rn_store_actions(Slices, ActionDefs) :-
    findall(ActionDef, (
        member(slice(_, _, Actions), Slices),
        member(action(AName, ABody), Actions),
        format(atom(ActionDef), '  ~w: ~w,', [AName, ABody])
    ), ActionDefList),
    atomic_list_concat(ActionDefList, '\n', ActionDefs).

%! generate_react_native_computed(+ComputedSpec, -Code) is det
generate_react_native_computed(computed(Name, Deps, Expr), Code) :-
    atomic_list_concat(Deps, ', ', DepsStr),
    format(atom(Code), 'const ~w = useMemo(() => ~w, [~w]);', [Name, Expr, DepsStr]).

%! generate_react_native_effect(+EffectSpec, -Code) is det
generate_react_native_effect(effect(_Name, Deps, Body), Code) :-
    atomic_list_concat(Deps, ', ', DepsStr),
    format(atom(Code), 'useEffect(() => {\n  ~w\n}, [~w]);', [Body, DepsStr]).

%! generate_react_native_binding(+BindingSpec, -Code) is det
generate_react_native_binding(bind(Target, Source), Code) :-
    format(atom(Code), '~w={~w}', [Target, Source]).
generate_react_native_binding(two_way(Target, Source), Code) :-
    capitalize_first(Source, SetterName),
    format(atom(Code), '~w={~w} onChangeText={set~w}', [Target, Source, SetterName]).
generate_react_native_binding(on(Element, Event, Handler), Code) :-
    rn_event_prop(Event, Prop),
    format(atom(Code), '<~w ~w={~w} />', [Element, Prop, Handler]).

rn_event_prop(press, 'onPress').
rn_event_prop(click, 'onPress').
rn_event_prop(change, 'onChangeText').
rn_event_prop(submit, 'onSubmitEditing').
rn_event_prop(focus, 'onFocus').
rn_event_prop(blur, 'onBlur').

%! generate_react_native_action(+ActionSpec, -Code) is det
generate_react_native_action(action(Name, State, Updater), Code) :-
    capitalize_first(State, SetterName),
    format(atom(Code), 'const ~w = useCallback(() => set~w(~w), []);', [Name, SetterName, Updater]).

% ============================================================================
% Vue Code Generation
% ============================================================================

%! generate_vue_state(+StateSpec, -Code) is det
generate_vue_state(state(Name, Initial, Options), Code) :-
    format_initial_value(Initial, vue, InitialStr),
    (   member(reactive(true), Options)
    ->  format(atom(Code), 'const ~w = reactive(~w);', [Name, InitialStr])
    ;   format(atom(Code), 'const ~w = ref(~w);', [Name, InitialStr])
    ).

%! generate_vue_store(+StoreSpec, -Code) is det
generate_vue_store(store(Name, Slices, Options), Code) :-
    generate_vue_store_state(Slices, StateDefs),
    generate_vue_store_getters(Slices, GetterDefs),
    generate_vue_store_actions(Slices, ActionDefs),
    (   member(persist(true), Options)
    ->  PersistPlugin = ",\n  persist: true"
    ;   PersistPlugin = ""
    ),
    format(atom(Code), "import { defineStore } from 'pinia';

export const use~wStore = defineStore('~w', {
  state: () => ({
~w
  }),
  getters: {
~w
  },
  actions: {
~w
  }~w
});", [Name, Name, StateDefs, GetterDefs, ActionDefs, PersistPlugin]).

generate_vue_store_state(Slices, StateDefs) :-
    findall(StateDef, (
        member(slice(_, Fields, _), Slices),
        member(field(FName, _, Default), Fields),
        format_initial_value(Default, vue, DefaultStr),
        format(atom(StateDef), '    ~w: ~w,', [FName, DefaultStr])
    ), StateDefList),
    atomic_list_concat(StateDefList, '\n', StateDefs).

generate_vue_store_getters(Slices, GetterDefs) :-
    findall(GetterDef, (
        member(slice(_, _, Actions), Slices),
        member(getter(GName, GBody), Actions),
        format(atom(GetterDef), '    ~w() { return ~w; },', [GName, GBody])
    ), GetterDefList),
    (   GetterDefList = []
    ->  GetterDefs = ""
    ;   atomic_list_concat(GetterDefList, '\n', GetterDefs)
    ).

generate_vue_store_actions(Slices, ActionDefs) :-
    findall(ActionDef, (
        member(slice(_, _, Actions), Slices),
        member(action(AName, ABody), Actions),
        format(atom(ActionDef), '    ~w() { ~w },', [AName, ABody])
    ), ActionDefList),
    atomic_list_concat(ActionDefList, '\n', ActionDefs).

%! generate_vue_computed(+ComputedSpec, -Code) is det
generate_vue_computed(computed(Name, _Deps, Expr), Code) :-
    format(atom(Code), 'const ~w = computed(() => ~w);', [Name, Expr]).

%! generate_vue_effect(+EffectSpec, -Code) is det
generate_vue_effect(effect(_Name, Deps, Body), Code) :-
    (   Deps = [SingleDep]
    ->  format(atom(Code), 'watch(~w, () => {\n  ~w\n});', [SingleDep, Body])
    ;   atomic_list_concat(Deps, ', ', DepsStr),
        format(atom(Code), 'watch([~w], () => {\n  ~w\n});', [DepsStr, Body])
    ).

%! generate_vue_binding(+BindingSpec, -Code) is det
generate_vue_binding(bind(Target, Source), Code) :-
    format(atom(Code), ':~w="~w"', [Target, Source]).
generate_vue_binding(two_way(_Target, Source), Code) :-
    format(atom(Code), 'v-model="~w"', [Source]).
generate_vue_binding(on(Element, Event, Handler), Code) :-
    format(atom(Code), '<~w @~w="~w" />', [Element, Event, Handler]).

%! generate_vue_action(+ActionSpec, -Code) is det
generate_vue_action(action(Name, State, Updater), Code) :-
    format(atom(Code), 'const ~w = () => { ~w.value = ~w; };', [Name, State, Updater]).

% ============================================================================
% Flutter Code Generation
% ============================================================================

%! generate_flutter_state(+StateSpec, -Code) is det
generate_flutter_state(state(Name, Initial, Options), Code) :-
    (   member(type(Type), Options)
    ->  dart_type(Type, DartType)
    ;   infer_type(Initial, InferredType),
        dart_type(InferredType, DartType)
    ),
    format_initial_value(Initial, flutter, InitialStr),
    format(atom(Code), '~w _~w = ~w;', [DartType, Name, InitialStr]).

%! generate_flutter_store(+StoreSpec, -Code) is det
generate_flutter_store(store(Name, Slices, _Options), Code) :-
    capitalize_first(Name, StoreName),
    generate_flutter_store_fields(Slices, FieldDefs),
    generate_flutter_store_getters(Slices, GetterDefs),
    generate_flutter_store_setters(Slices, SetterDefs),
    format(atom(Code), "class ~wStore extends ChangeNotifier {
~w

~w

~w
}", [StoreName, FieldDefs, GetterDefs, SetterDefs]).

generate_flutter_store_fields(Slices, FieldDefs) :-
    findall(FieldDef, (
        member(slice(_, Fields, _), Slices),
        member(field(FName, FType, Default), Fields),
        dart_type(FType, DartType),
        format_initial_value(Default, flutter, DefaultStr),
        format(atom(FieldDef), '  ~w _~w = ~w;', [DartType, FName, DefaultStr])
    ), FieldDefList),
    atomic_list_concat(FieldDefList, '\n', FieldDefs).

generate_flutter_store_getters(Slices, GetterDefs) :-
    findall(GetterDef, (
        member(slice(_, Fields, _), Slices),
        member(field(FName, FType, _), Fields),
        dart_type(FType, DartType),
        format(atom(GetterDef), '  ~w get ~w => _~w;', [DartType, FName, FName])
    ), GetterDefList),
    atomic_list_concat(GetterDefList, '\n', GetterDefs).

generate_flutter_store_setters(Slices, SetterDefs) :-
    findall(SetterDef, (
        member(slice(_, Fields, _), Slices),
        member(field(FName, FType, _), Fields),
        dart_type(FType, DartType),
        format(atom(SetterDef), '  set ~w(~w value) {\n    _~w = value;\n    notifyListeners();\n  }', [FName, DartType, FName])
    ), SetterDefList),
    atomic_list_concat(SetterDefList, '\n\n', SetterDefs).

%! generate_flutter_computed(+ComputedSpec, -Code) is det
generate_flutter_computed(computed(Name, _Deps, Expr), Code) :-
    format(atom(Code), 'get ~w => ~w;', [Name, Expr]).

%! generate_flutter_effect(+EffectSpec, -Code) is det
generate_flutter_effect(effect(Name, _Deps, Body), Code) :-
    format(atom(Code), 'void _~w() {\n  ~w\n}', [Name, Body]).

%! generate_flutter_binding(+BindingSpec, -Code) is det
generate_flutter_binding(bind(Target, Source), Code) :-
    format(atom(Code), '~w: ~w', [Target, Source]).
generate_flutter_binding(two_way(Target, Source), Code) :-
    format(atom(Code), 'TextField(\n  controller: _~wController,\n  onChanged: (value) => setState(() => _~w = value),\n)', [Target, Source]).
generate_flutter_binding(on(Element, Event, Handler), Code) :-
    flutter_event_prop(Event, Prop),
    format(atom(Code), '~w(~w: ~w)', [Element, Prop, Handler]).

flutter_event_prop(press, 'onPressed').
flutter_event_prop(click, 'onPressed').
flutter_event_prop(tap, 'onTap').
flutter_event_prop(change, 'onChanged').
flutter_event_prop(submit, 'onSubmitted').

%! generate_flutter_action(+ActionSpec, -Code) is det
generate_flutter_action(action(Name, State, Updater), Code) :-
    format(atom(Code), 'void ~w() {\n  setState(() {\n    _~w = ~w;\n  });\n}', [Name, State, Updater]).

% ============================================================================
% SwiftUI Code Generation
% ============================================================================

%! generate_swiftui_state(+StateSpec, -Code) is det
generate_swiftui_state(state(Name, Initial, Options), Code) :-
    (   member(type(Type), Options)
    ->  swift_type(Type, SwiftType)
    ;   infer_type(Initial, InferredType),
        swift_type(InferredType, SwiftType)
    ),
    format_initial_value(Initial, swiftui, InitialStr),
    (   member(binding(true), Options)
    ->  format(atom(Code), '@Binding var ~w: ~w', [Name, SwiftType])
    ;   format(atom(Code), '@State private var ~w: ~w = ~w', [Name, SwiftType, InitialStr])
    ).

%! generate_swiftui_store(+StoreSpec, -Code) is det
generate_swiftui_store(store(Name, Slices, _Options), Code) :-
    capitalize_first(Name, StoreName),
    generate_swift_store_fields(Slices, FieldDefs),
    format(atom(Code), "class ~wStore: ObservableObject {
~w
}", [StoreName, FieldDefs]).

generate_swift_store_fields(Slices, FieldDefs) :-
    findall(FieldDef, (
        member(slice(_, Fields, _), Slices),
        member(field(FName, FType, Default), Fields),
        swift_type(FType, SwiftType),
        format_initial_value(Default, swiftui, DefaultStr),
        format(atom(FieldDef), '  @Published var ~w: ~w = ~w', [FName, SwiftType, DefaultStr])
    ), FieldDefList),
    atomic_list_concat(FieldDefList, '\n', FieldDefs).

%! generate_swiftui_computed(+ComputedSpec, -Code) is det
generate_swiftui_computed(computed(Name, _Deps, Expr), Code) :-
    format(atom(Code), 'var ~w: String {\n  ~w\n}', [Name, Expr]).

%! generate_swiftui_effect(+EffectSpec, -Code) is det
generate_swiftui_effect(effect(_Name, Deps, Body), Code) :-
    (   Deps = [SingleDep]
    ->  format(atom(Code), '.onChange(of: ~w) { _ in\n  ~w\n}', [SingleDep, Body])
    ;   Deps = [Dep1, Dep2|_]
    ->  format(atom(Code), '.onChange(of: ~w) { _ in\n  ~w\n}\n.onChange(of: ~w) { _ in\n  ~w\n}', [Dep1, Body, Dep2, Body])
    ;   Code = ""
    ).

%! generate_swiftui_binding(+BindingSpec, -Code) is det
generate_swiftui_binding(bind(Target, Source), Code) :-
    format(atom(Code), '~w: ~w', [Target, Source]).
generate_swiftui_binding(two_way(Target, Source), Code) :-
    format(atom(Code), '~w("", text: $~w)', [Target, Source]).
generate_swiftui_binding(on(Element, Event, Handler), Code) :-
    swift_event_handler(Event, Element, Handler, Code).

swift_event_handler(press, Element, Handler, Code) :-
    format(atom(Code), '~w { ~w() }', [Element, Handler]).
swift_event_handler(click, Element, Handler, Code) :-
    format(atom(Code), '~w { ~w() }', [Element, Handler]).
swift_event_handler(tap, Element, Handler, Code) :-
    format(atom(Code), '~w.onTapGesture { ~w() }', [Element, Handler]).

%! generate_swiftui_action(+ActionSpec, -Code) is det
generate_swiftui_action(action(Name, State, Updater), Code) :-
    format(atom(Code), 'func ~w() {\n  ~w = ~w\n}', [Name, State, Updater]).

% ============================================================================
% Type Mappings
% ============================================================================

ts_type(string, 'string').
ts_type(number, 'number').
ts_type(boolean, 'boolean').
ts_type(list, 'any[]').
ts_type(any, 'any').

dart_type(string, 'String').
dart_type(number, 'int').
dart_type(boolean, 'bool').
dart_type(list, 'List<dynamic>').
dart_type(any, 'dynamic').

swift_type(string, 'String').
swift_type(number, 'Int').
swift_type(boolean, 'Bool').
swift_type(list, '[Any]').
swift_type(any, 'Any').

% ============================================================================
% Value Formatting
% ============================================================================

format_initial_value(Value, _Target, Str) :-
    number(Value), !,
    format(atom(Str), '~w', [Value]).
format_initial_value(Value, Target, Str) :-
    (Value = true ; Value = false), !,
    (   Target = swiftui
    ->  (Value = true -> Str = 'true' ; Str = 'false')
    ;   format(atom(Str), '~w', [Value])
    ).
format_initial_value(Value, _, Str) :-
    Value = [], !,
    Str = '[]'.
format_initial_value(Value, Target, Str) :-
    atom(Value), !,
    (   Value = ''
    ->  Str = '""'
    ;   Target = swiftui
    ->  format(atom(Str), '"~w"', [Value])
    ;   format(atom(Str), '\'~w\'', [Value])
    ).
format_initial_value(Value, Target, Str) :-
    string(Value), !,
    (   Value = ""
    ->  Str = '""'
    ;   Target = swiftui
    ->  format(atom(Str), '"~w"', [Value])
    ;   format(atom(Str), '\'~w\'', [Value])
    ).
format_initial_value(Value, _, Str) :-
    format(atom(Str), '~w', [Value]).

% ============================================================================
% Helper Predicates
% ============================================================================

capitalize_first(Atom, Capitalized) :-
    atom_chars(Atom, [First|Rest]),
    upcase_atom(First, Upper),
    atom_chars(Capitalized, [Upper|Rest]).

% ============================================================================
% Validation
% ============================================================================

%! validate_state(+Spec, -Errors) is det
validate_state(state(Name, _Initial, _Options), Errors) :-
    (   atom(Name)
    ->  Errors = []
    ;   Errors = [invalid_state_name(Name)]
    ).

%! validate_binding(+Spec, -Errors) is det
validate_binding(bind(Target, Source), Errors) :-
    findall(Error, (
        (\+ atom(Target) -> Error = invalid_target(Target) ; fail)
    ;   (\+ (atom(Source) ; string(Source)) -> Error = invalid_source(Source) ; fail)
    ), Errors).
validate_binding(two_way(Target, Source), Errors) :-
    findall(Error, (
        (\+ atom(Target) -> Error = invalid_target(Target) ; fail)
    ;   (\+ atom(Source) -> Error = invalid_source(Source) ; fail)
    ), Errors).
validate_binding(on(Element, Event, Handler), Errors) :-
    findall(Error, (
        (\+ atom(Element) -> Error = invalid_element(Element) ; fail)
    ;   (\+ atom(Event) -> Error = invalid_event(Event) ; fail)
    ;   (\+ (atom(Handler) ; string(Handler)) -> Error = invalid_handler(Handler) ; fail)
    ), Errors).

% ============================================================================
% Testing
% ============================================================================

test_data_binding :-
    format('Running data binding tests...~n', []),

    % Test 1: State generation for React Native
    generate_state(state(counter, 0, []), react_native, RNState),
    (   sub_atom(RNState, _, _, _, 'useState')
    ->  format('  Test 1 passed: React Native state~n', [])
    ;   format('  Test 1 FAILED: React Native state~n', [])
    ),

    % Test 2: State generation for Vue
    generate_state(state(counter, 0, []), vue, VueState),
    (   sub_atom(VueState, _, _, _, 'ref')
    ->  format('  Test 2 passed: Vue state~n', [])
    ;   format('  Test 2 FAILED: Vue state~n', [])
    ),

    % Test 3: State generation for Flutter
    generate_state(state(counter, 0, [type(number)]), flutter, FlutterState),
    (   sub_atom(FlutterState, _, _, _, 'int')
    ->  format('  Test 3 passed: Flutter state~n', [])
    ;   format('  Test 3 FAILED: Flutter state~n', [])
    ),

    % Test 4: State generation for SwiftUI
    generate_state(state(counter, 0, [type(number)]), swiftui, SwiftState),
    (   sub_atom(SwiftState, _, _, _, '@State')
    ->  format('  Test 4 passed: SwiftUI state~n', [])
    ;   format('  Test 4 FAILED: SwiftUI state~n', [])
    ),

    % Test 5: Computed for React Native
    generate_computed(computed(doubled, [count], 'count * 2'), react_native, RNComputed),
    (   sub_atom(RNComputed, _, _, _, 'useMemo')
    ->  format('  Test 5 passed: React Native computed~n', [])
    ;   format('  Test 5 FAILED: React Native computed~n', [])
    ),

    % Test 6: Computed for Vue
    generate_computed(computed(doubled, [count], 'count.value * 2'), vue, VueComputed),
    (   sub_atom(VueComputed, _, _, _, 'computed')
    ->  format('  Test 6 passed: Vue computed~n', [])
    ;   format('  Test 6 FAILED: Vue computed~n', [])
    ),

    % Test 7: Effect for React Native
    generate_effect(effect(logChange, [user], 'console.log(user)'), react_native, RNEffect),
    (   sub_atom(RNEffect, _, _, _, 'useEffect')
    ->  format('  Test 7 passed: React Native effect~n', [])
    ;   format('  Test 7 FAILED: React Native effect~n', [])
    ),

    % Test 8: Effect for Vue
    generate_effect(effect(logChange, [user], 'console.log(user)'), vue, VueEffect),
    (   sub_atom(VueEffect, _, _, _, 'watch')
    ->  format('  Test 8 passed: Vue effect~n', [])
    ;   format('  Test 8 FAILED: Vue effect~n', [])
    ),

    % Test 9: Two-way binding for React Native
    generate_binding(two_way(value, name), react_native, RNBinding),
    (   sub_atom(RNBinding, _, _, _, 'onChangeText')
    ->  format('  Test 9 passed: React Native two-way binding~n', [])
    ;   format('  Test 9 FAILED: React Native two-way binding~n', [])
    ),

    % Test 10: Two-way binding for Vue
    generate_binding(two_way(value, name), vue, VueBinding),
    (   sub_atom(VueBinding, _, _, _, 'v-model')
    ->  format('  Test 10 passed: Vue two-way binding~n', [])
    ;   format('  Test 10 FAILED: Vue two-way binding~n', [])
    ),

    format('All 10 data binding tests completed!~n', []).

:- initialization(test_data_binding, program).
