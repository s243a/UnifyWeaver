# Skill: UnifyWeaver Prolog-to-Bash Transpiler

## 1. Purpose

This skill enables an agent to take a source file containing Prolog predicates and transpile it into an executable bash script using the core UnifyWeaver compilation engine.

This is the primary mechanism for converting declarative Prolog logic into an efficient, portable script.

## 2. When to Use

Use this skill when you have a set of Prolog rules in a file (e.g., one you have just generated) and you need to convert them into an executable format. This is a necessary step for any logic that needs to be run as part of a larger workflow.

## 3. Tool: `unifyweaver.compile`

This skill is conceptually exposed as a single tool, `unifyweaver.compile`. The implementation is a call to the SWI-Prolog interpreter with the correct arguments to invoke the `compiler_driver`.

### 3.1. Interface

The agent should construct a shell command in the following format.

**Command Template:**
```bash
swipl -g "use_module('src/unifyweaver/core/compiler_driver'), consult('<prolog_file>'), compile(<predicate_indicator>, [<options>])" -t halt
```

### 3.2. Parameters

| Parameter | Description | Example |
| :--- | :--- | :--- |
| `<prolog_file>` | The path to the `.pl` file containing the Prolog source code to be compiled. | `'my_rules.pl'` |
| `<predicate_indicator>` | The main predicate to compile, in the format `functor/arity`. The compiler will automatically handle its dependencies. | `choose_strategy/3` |
| `<options>` | A comma-separated list of Prolog options to pass to the compiler. | `output_dir('.')` |

### 3.3. Common Options

- **`output_dir(Directory)`:** Specifies where to save the compiled `.sh` file. The filename will be derived from the predicate name. It is highly recommended to always set this to a known location, like `'.'`. If not provided, it may default to an unexpected path.

## 4. Example Invocation

**Goal:** Compile the predicate `choose_strategy/3` from the file `temp_strategy.pl` and save the output to the current directory.

**Agent's Action (Shell Command):**
```bash
swipl -g "use_module('src/unifyweaver/core/compiler_driver'), consult('temp_strategy.pl'), compile(choose_strategy/3, [output_dir('.')])" -t halt
```

## 5. Important Considerations

- **Arithmetic Predicates:** The current version of the `compiler_driver` has a known limitation and cannot transpile predicates that directly use arithmetic comparison operators (e.g., `>`, `<`, `=<`). This will cause the compilation to fail.
- **Dependencies:** The compiler will automatically analyze and compile any user-defined predicates that your main predicate depends on.
- **Output:** The result of a successful compilation is a new, executable `.sh` file in the specified output directory (e.g., `choose_strategy.sh`).
