# Tool Usage Examples

This file contains examples of how to use various UnifyWeaver utility tools.

---

### Example: Extract Full Record

This example shows how to extract the full, original Markdown content of a specific record by its name.

> [!example-record]
> id: 2025-11-11-extract-full-record
> name: unifyweaver.tool_usage.extract_full_record
> compile_instruction: |
>   scripts/utils/extract_records.pl --query "prolog.recursion.ancestor" "education/UnifyWeaver_Education/book-workflow/examples_library/recursion_examples.md"
> transpiled_output: |
>   ### Example: Ancestor Predicate
>
>   > [!example-record]
>   > id: 20251026-141000-001
>   > name: prolog.recursion.ancestor
>
>   ```prolog
>   % parent(Parent, Child)
>   parent(a, b).
>   parent(b, c).
>
>   % ancestor(Ancestor, Descendant)
>   ancestor(A, B) :- parent(A, B).
>   ancestor(A, C) :- parent(A, B), ancestor(B, C).
>   ```

---

### Example: Extract Content as JSON

This example shows how to extract just the code content of a record and format the output as a JSON object.

> [!example-record]
> id: 2025-11-11-extract-content-as-json
> name: unifyweaver.tool_usage.extract_content_as_json
> compile_instruction: |
>   scripts/utils/extract_records.pl --format json --query "prolog.recursion.ancestor" "education/UnifyWeaver_Education/book-workflow/examples_library/recursion_examples.md"
> transpiled_output: |
>   {"id":"20251026-141000-001","name":"prolog.recursion.ancestor","content":"% parent(Parent, Child)\nparent(a, b).\nparent(b, c).\n\n% ancestor(Ancestor, Descendant)\nancestor(A, B) :- parent(A, B).\nancestor(A, C) :- parent(A, B), ancestor(B, C).\n"}
