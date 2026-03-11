# Fix: Add CLI Main Blocks to R Recursion Templates

## Summary

- Add `if (!interactive())` main blocks to 4 R templates that were missing them: tail ternary, tail binary, linear numeric fold, and linear list fold
- Generated R scripts (e.g., `factorial.R`, `sum_list.R`, `count_items.R`) can now be run directly with `Rscript <file> <args>` instead of requiring `source()` and manual function calls

## Before

```
$ Rscript output/advanced/factorial.R 6
(no output)

$ Rscript -e "source('output/advanced/factorial.R'); cat(factorial(6), '\n')"
720
```

## After

```
$ Rscript output/advanced/factorial.R 6
720

$ Rscript output/advanced/sum_list.R 1,2,3,4,5
15

$ Rscript output/advanced/count_items.R 10,20,30
3
```

## Test plan

- [x] `factorial.R 6` → 720
- [x] `sum_list.R 1,2,3,4,5` → 15
- [x] `count_items.R 10,20,30` → 3
- [x] `list_length.R 1,2,3,4` → 4
- [x] Existing scripts unaffected: tree_sum→6, even_odd→TRUE, fib_multicall(10)→55

🤖 Generated with [Claude Code](https://claude.com/claude-code)
