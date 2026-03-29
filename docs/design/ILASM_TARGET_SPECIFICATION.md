# ILAsm Target — Specification

## Module Structure

```
src/unifyweaver/
  targets/
    ilasm_target.pl              — Target module, compilation entry point
  core/
    cil_bytecode.pl              — Shared CIL instruction generation
  bindings/
    cil_asm_bindings.pl          — .NET BCL instruction bindings

templates/targets/ilasm/
    transitive_closure.mustache  — TC template (composable)
    tc_definitions.mustache      — BFS functions in IL
    tc_input_*.mustache          — Input mode variants
    tc_interface_cli.mustache    — CLI entry point
```

## CIL Instruction Generation (`cil_bytecode.pl`)

### Expression Compilation

```prolog
%% cil_expr_to_instructions(+Expr, +VarMap, -Instructions)
%  Compile a Prolog arithmetic expression to CIL stack instructions.
cil_expr_to_instructions(Var, VarMap, ['ldloc', VarName]) :-
    var(Var), lookup_var(Var, VarMap, VarName).
cil_expr_to_instructions(N, _, ['ldc.i8', N]) :-
    integer(N).
cil_expr_to_instructions(A + B, VarMap, Instrs) :-
    cil_expr_to_instructions(A, VarMap, AI),
    cil_expr_to_instructions(B, VarMap, BI),
    append(AI, BI, AB),
    append(AB, ['add'], Instrs).
%% ... similar for -, *, /, mod
```

### Guard Compilation

```prolog
%% cil_guard_to_instructions(+Guard, +VarMap, +FalseLabel, -Instructions)
cil_guard_to_instructions(A > B, VarMap, Label, Instrs) :-
    cil_expr_to_instructions(A, VarMap, AI),
    cil_expr_to_instructions(B, VarMap, BI),
    append(AI, BI, AB),
    append(AB, [format_atom('ble ~w', [Label])], Instrs).
```

### If-Chain Assembly

```prolog
%% cil_if_chain(+Branches, +LabelPrefix, -Instructions)
%  Assemble pre-compiled branches into CIL if/else chain with labels.
```

## ILAsm Output Format

### Function Template

```il
.method public static int64 classify_sign(int64 arg1) cil managed {
    .maxstack 4
    .locals init (int64 v1)

    // Guard: arg1 == 0
    ldarg.0
    ldc.i8 0
    beq CLAUSE_0

    // Guard: arg1 > 0
    ldarg.0
    ldc.i8 0
    bgt CLAUSE_1

    // Else: negative
    ldstr "negative"
    ret

CLAUSE_0:
    ldstr "zero"
    ret

CLAUSE_1:
    ldstr "positive"
    ret
}
```

### Complete Assembly

```il
.assembly extern mscorlib {}
.assembly PrologGenerated {}

.class public auto ansi PrologGenerated.Program extends [mscorlib]System.Object {

    // Generated functions here

    .method public static void Main(string[] args) cil managed {
        .entrypoint
        // CLI dispatch
    }
}
```

## compile_expression Hooks

```prolog
%% render_output_goal(ilasm, Goal, VarMap, Line, VarName, VarMapOut)
clause_body_analysis:render_output_goal(ilasm, Goal, VarMap, Line, VarName, VarMapOut) :-
    (   Goal = (Var = Expr), var(Var)
    ->  ensure_var(VarMap, Var, VarName, VarMapOut),
        cil_expr_to_instructions(Expr, VarMap, ExprInstrs),
        cil_instructions_to_text(ExprInstrs, ExprText),
        format(string(Line), '~w\n    stloc ~w', [ExprText, VarName])
    ;   ...
    ).

%% render_guard_condition(ilasm, Goal, VarMap, CondStr)
clause_body_analysis:render_guard_condition(ilasm, Goal, VarMap, CondStr) :-
    cil_guard_to_instructions(Goal, VarMap, "L_false", Instrs),
    cil_instructions_to_text(Instrs, CondStr).

%% render_ite_block(ilasm, Cond, ThenLines, ElseLines, ...)
clause_body_analysis:render_ite_block(ilasm, Cond, ThenLines, ElseLines, _Indent, _RetVars, Lines) :-
    %% CIL uses label-based branching, not structured if/else
    append([Cond, 'L_then:'|ThenLines], ['br L_end', 'L_else:'|ElseLines], Pre),
    append(Pre, ['L_end:'], Lines).
```

## Component Integration

```prolog
%% Register ILAsm as a component type
:- initialization((
    catch(
        register_component_type(source, custom_ilasm, custom_ilasm, [
            description('Injects custom CIL assembly as a component')
        ]),
        _, true
    )
)).

%% compile_component for custom IL injection
ilasm_compile_component(Name, Config, _Options, Code) :-
    member(code(ILCode), Config),
    format(string(Code), '// Component: ~w\n~w', [Name, ILCode]).
```

## Binding Registry

```prolog
%% cil_asm_bindings.pl — .NET BCL instruction bindings

init_cil_asm_bindings :-
    %% Arithmetic
    declare_binding(ilasm, add/3, 'add', [int64, int64], [int64], [pure]),
    declare_binding(ilasm, sub/3, 'sub', [int64, int64], [int64], [pure]),
    declare_binding(ilasm, mul/3, 'mul', [int64, int64], [int64], [pure]),
    declare_binding(ilasm, div/3, 'div', [int64, int64], [int64], [pure]),
    declare_binding(ilasm, mod/3, 'rem', [int64, int64], [int64], [pure]),

    %% Math (System.Math calls)
    declare_binding(ilasm, sqrt/2, 'call float64 [mscorlib]System.Math::Sqrt(float64)',
        [float64], [float64], [pure]),
    declare_binding(ilasm, abs/2, 'call int64 [mscorlib]System.Math::Abs(int64)',
        [int64], [int64], [pure]),

    %% String operations
    declare_binding(ilasm, length/2, 'callvirt instance int32 [mscorlib]System.String::get_Length()',
        [string], [int32], [pure]),

    %% Console I/O
    declare_binding(ilasm, print/1, 'call void [mscorlib]System.Console::WriteLine(string)',
        [string], [], [effect(io)]).
```

## Type Mapping

```prolog
%% type_declarations.pl additions
resolve_type(atom, ilasm, "string").
resolve_type(string, ilasm, "string").
resolve_type(integer, ilasm, "int64").
resolve_type(float, ilasm, "float64").
resolve_type(number, ilasm, "float64").
resolve_type(boolean, ilasm, "bool").
resolve_type(any, ilasm, "object").
resolve_type(list(Type), ilasm, Concrete) :-
    resolve_type(Type, ilasm, Inner),
    format(string(Concrete), "class [mscorlib]System.Collections.Generic.List`1<~w>", [Inner]).
```

## Seed Statements (Input Source)

```prolog
seed_statement(ilasm, _BasePred, From, To, Statement) :-
    format(string(Statement),
        '    ldstr "~w"\n    ldstr "~w"\n    call void PrologGenerated.Program::AddFact(string, string)',
        [From, To]).
```
