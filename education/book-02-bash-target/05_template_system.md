<!--
SPDX-License-Identifier: MIT AND CC-BY-4.0
Copyright (c) 2025 John William Creighton (s243a)

This documentation is dual-licensed under MIT and CC-BY-4.0.
-->

# Chapter 8: The Modern Template System

## Introduction: Why Templates?

UnifyWeaver generates bash code using a **template system** rather than building strings with concatenation. This approach keeps code generation clean, maintainable, and readable.

**Bad approach (concatenation):**
```prolog
% ❌ Hard to read and maintain
generate_function(Name, Body, Code) :-
    format(string(Code), '#!/bin/bash\n~w() {\n~w\n}\n', [Name, Body]).
```

**Good approach (templates):**
```prolog
% ✅ Clear, readable, maintainable
Template = '#!/bin/bash\n{{name}}() {\n{{body}}\n}',
render_template(Template, [name=Name, body=Body], Code).
```

The template system separates the structure of the code (the template) from the data (the predicate names and logic), which is a core principle of good software design.

## Basic vs. Named Templates

The original UnifyWeaver system used a simple `render_template/3` predicate that took a string as input. The modern system revolves around **named templates**.

Instead of passing a long string to a predicate, you refer to a template by its name (an atom like `bash_header` or `function_definition`). The system then finds and renders that template according to a configurable strategy.

**The main predicate you will use is `render_named_template/3`:**
```prolog
render_named_template(+TemplateName, +Dict, -Result)
```

## Template Source Strategies

The most powerful feature of the new template system is its ability to load templates from different sources. The system is governed by the `source_order` option, which is a list that tells UnifyWeaver where to look for a template and in what order.

The three possible sources are:

1.  **`file`**: Looks for a template file in the `templates/` directory (e.g., `templates/my_template.tmpl.sh`). This is the most flexible option and is recommended for customization.
2.  **`cached`**: Looks for a template in an in-memory cache. The system can be configured to automatically cache templates after they are loaded from a file or generated.
3.  **`generated`**: Looks for a hardcoded template defined inside `template_system.pl` using the `template/2` fact. This is the fallback and provides the default behavior.

### Configuring the Source Order

You can control the source order globally or for a specific template.

**Example: Preferring file-based templates globally**

By default, the system only looks for `generated` templates. You can change this by setting a new global default.

```prolog
% In your Prolog session
?- use_module(unifyweaver(core/template_system)).
?- set_template_config_default([source_order([file, cached, generated])]).
```

Now, when you call `render_named_template('my_template', Dict, Result)`, the system will:
1.  First, look for a file named `templates/my_template.tmpl.sh`.
2.  If not found, it will look for a `my_template` in the in-memory cache.
3.  If still not found, it will fall back to the `generated` `template(my_template, ...)` fact inside `template_system.pl`.

## A Practical Example: Overriding a Template

Let's see how you can use this system to customize the generated code without touching the compiler itself.

UnifyWeaver has a built-in, `generated` template named `bash_header`.

**Goal:** We want to add a custom comment to the header of all our generated scripts.

### Step 1: Configure the Template System

At the start of your Prolog session, tell the system to prefer `file`-based templates.

```prolog
?- set_template_config_default([source_order([file, generated])]).
```

### Step 2: Create a Custom Template File

Create a new file named `templates/bash_header.tmpl.sh`. The name is important: it must match the name of the template you want to override (`bash_header`) and be in the configured `templates` directory.

**File content for `templates/bash_header.tmpl.sh`:**
```bash
#!/bin/bash
# ----------------------------------------
# Custom header for My Project
# Generated on: $(date)
# Description: {{description}}
# ----------------------------------------
```

### Step 3: Render the Template

Now, when any part of the UnifyWeaver compiler renders the `bash_header` template, it will use your custom file instead of the built-in one.

```prolog
?- render_named_template(bash_header, [description='My Ancestor Script'], Code).
```

**Resulting `Code`:**
```bash
#!/bin/bash
# ----------------------------------------
# Custom header for My Project
# Generated on: $(date)
# Description: My Ancestor Script
# ----------------------------------------
```

You have successfully customized the output without modifying the compiler's source code. This powerful feature allows for extensive customization and theming of the generated Bash scripts.

## Caching

The template system also includes an in-memory and on-disk caching layer. When configured, it can automatically save generated templates to a `templates/cache` directory. This is useful for two reasons:

1.  **Performance:** It avoids the overhead of generating or reading a template from a file on every compilation.
2.  **Inspection:** It allows you to easily see the exact template code that was used during a compilation, which is very helpful for debugging.

Caching can be enabled in the configuration:

```prolog
% Enable caching and file-based templates
?- set_template_config_default([
    source_order([file, cached, generated]),
    auto_cache(true)
]).
```

## Summary

The modern template system is a significant enhancement that gives you fine-grained control over the generated code.

**Key Takeaways:**
-   The system uses **named templates** (e.g., `bash_header`).
-   The **`source_order`** option controls where the system looks for templates.
-   You can easily **override** built-in templates by creating files with matching names in the `templates/` directory.
-   This provides a powerful mechanism for **customization and theming** without altering the core compiler logic.

```

---

## Navigation

**←** [Previous: Chapter 7: Variable Scope and Process Substitution](04_variable_scope_and_process_substitution) | [📖 Book 2: Bash Target](README) | [Next: Chapter 9: Advanced Recursion Patterns →](06_advanced_recursion)
