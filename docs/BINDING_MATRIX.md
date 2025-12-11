# Binding Tracking Matrix

**Status:** Living Document
**Last Updated:** 2025-12-10

This document tracks which bindings are implemented for each target language.

| Symbol | Meaning |
|--------|---------|
| ✓ | Implemented |
| ✗ | Not implemented |

---

## Summary

| Target | Bindings | Categories |
|--------|----------|------------|
| **Python** | 106 | 9 (Built-ins, String, List, Dict, I/O, Regex, Math, Collections, Itertools) |
| **PowerShell** | 68 | 4 (Cmdlets, Automation, .NET, C# Hosting) |

---

## Core Operations

| Binding | Python | PowerShell | Notes |
|---------|:------:|:----------:|-------|
| `abs/2` | ✓ | ✓ | Absolute value |
| `sqrt/2` | ✓ | ✓ | Square root |
| `to_string/2` | ✓ | ✓ | Convert to string |
| `to_int/2` | ✓ | ✓ | Convert to integer |
| `to_float/2` | ✓ | ✗ | Convert to float |
| `to_bool/2` | ✓ | ✗ | Convert to boolean |
| `to_list/2` | ✓ | ✗ | Convert to list |
| `to_array/2` | ✗ | ✓ | Convert to array |
| `to_set/2` | ✓ | ✗ | Convert to set |
| `to_tuple/2` | ✓ | ✗ | Convert to tuple |
| `to_dict/2` | ✓ | ✗ | Convert to dict |
| `to_hashtable/2` | ✗ | ✓ | Convert to hashtable |
| `length/2` | ✓ | ✗ | Get length/count |
| `type_of/2` | ✓ | ✗ | Get type |
| `is_instance/2` | ✓ | ✗ | Type check |
| `is_type/2` | ✗ | ✓ | Type check (-is) |
| `object_id/2` | ✓ | ✗ | Get unique identifier |
| `round/2` | ✗ | ✓ | Round number |

---

## Math Operations

| Binding | Python | PowerShell | Notes |
|---------|:------:|:----------:|-------|
| `sqrt/2` | ✓ | ✓ | Square root |
| `abs/2` | ✓ | ✓ | Absolute value |
| `floor/2` | ✓ | ✗ | Floor |
| `ceil/2` | ✓ | ✗ | Ceiling |
| `round/2` | ✗ | ✓ | Round |
| `pow/3` | ✓ | ✗ | Power |
| `log/2` | ✓ | ✗ | Natural logarithm |
| `log10/2` | ✓ | ✗ | Base 10 logarithm |
| `sin/2` | ✓ | ✗ | Sine |
| `cos/2` | ✓ | ✗ | Cosine |
| `tan/2` | ✓ | ✗ | Tangent |
| `pi/1` | ✓ | ✗ | Pi constant |
| `e/1` | ✓ | ✗ | Euler's number |
| `min/2` | ✓ | ✗ | Minimum value |
| `max/2` | ✓ | ✗ | Maximum value |
| `sum/2` | ✓ | ✗ | Sum of sequence |

---

## String Operations

| Binding | Python | PowerShell | Notes |
|---------|:------:|:----------:|-------|
| `string_split/2` | ✓ | ✗ | Split (no delimiter) |
| `string_split/3` | ✓ | ✓ | Split with delimiter |
| `string_join/3` | ✓ | ✗ | Join strings |
| `string_replace/4` | ✓ | ✓ | Replace substring |
| `string_strip/2` | ✓ | ✗ | Strip whitespace |
| `string_trim/2` | ✗ | ✓ | Trim whitespace |
| `string_lower/2` | ✓ | ✗ | Lowercase |
| `string_upper/2` | ✓ | ✗ | Uppercase |
| `string_starts_with/2` | ✓ | ✗ | Check prefix |
| `string_ends_with/2` | ✓ | ✗ | Check suffix |
| `string_find/3` | ✓ | ✗ | Find substring |
| `string_format/3` | ✓ | ✗ | Format string |
| `fstring/2` | ✓ | ✗ | f-string |

---

## List/Array Operations

| Binding | Python | PowerShell | Notes |
|---------|:------:|:----------:|-------|
| `list_append/2` | ✓ | ✗ | Append element |
| `list_extend/2` | ✓ | ✗ | Extend with iterable |
| `list_insert/3` | ✓ | ✗ | Insert at index |
| `list_remove/2` | ✓ | ✗ | Remove first occurrence |
| `list_pop/2` | ✓ | ✗ | Pop last element |
| `list_pop/3` | ✓ | ✗ | Pop at index |
| `list_index/3` | ✓ | ✗ | Find index of element |
| `list_count/3` | ✓ | ✗ | Count occurrences |
| `list_sort/1` | ✓ | ✗ | Sort in place |
| `list_reverse/1` | ✓ | ✗ | Reverse in place |
| `list_copy/2` | ✓ | ✗ | Shallow copy |
| `sorted/2` | ✓ | ✗ | Return sorted copy |
| `reversed/2` | ✓ | ✗ | Return reversed iterator |

---

## Dict/Hashtable Operations

| Binding | Python | PowerShell | Notes |
|---------|:------:|:----------:|-------|
| `dict_get/3` | ✓ | ✗ | Get value |
| `dict_get/4` | ✓ | ✗ | Get value with default |
| `dict_keys/2` | ✓ | ✗ | Get keys |
| `dict_values/2` | ✓ | ✗ | Get values |
| `dict_items/2` | ✓ | ✗ | Get key-value pairs |
| `dict_update/2` | ✓ | ✗ | Merge dicts |
| `dict_pop/3` | ✓ | ✗ | Remove and return |
| `dict_setdefault/4` | ✓ | ✗ | Get or set default |
| `dict_copy/2` | ✓ | ✗ | Shallow copy |

---

## File I/O Operations

| Binding | Python | PowerShell | Notes |
|---------|:------:|:----------:|-------|
| `file_open/3` | ✓ | ✗ | Open file |
| `file_read/2` | ✓ | ✗ | Read file content |
| `file_readlines/2` | ✓ | ✗ | Read file lines |
| `file_write/2` | ✓ | ✗ | Write to file |
| `file_close/1` | ✓ | ✗ | Close file |
| `file_exists/1` | ✗ | ✓ | Check file exists |
| `file_read_all_text/2` | ✗ | ✓ | Read all text |
| `file_write_all_text/2` | ✗ | ✓ | Write all text |
| `with_open/3` | ✓ | ✗ | Context manager |
| `get_content/2` | ✗ | ✓ | Get-Content |
| `set_content/2` | ✗ | ✓ | Set-Content |
| `get_child_item/2` | ✗ | ✓ | List files/dirs |
| `test_path/1` | ✗ | ✓ | Test-Path |
| `new_item/2` | ✗ | ✓ | Create file/dir |
| `remove_item/1` | ✗ | ✓ | Delete file/dir |

---

## Path Operations

| Binding | Python | PowerShell | Notes |
|---------|:------:|:----------:|-------|
| `path_exists/1` | ✓ | ✗ | Check path exists |
| `path_join/3` | ✓ | ✗ | Join paths |
| `path_combine/3` | ✗ | ✓ | Combine paths |
| `path_dirname/2` | ✓ | ✗ | Get directory name |
| `path_basename/2` | ✓ | ✗ | Get base name |
| `path_get_full/2` | ✗ | ✓ | Get full path |

---

## JSON Operations

| Binding | Python | PowerShell | Notes |
|---------|:------:|:----------:|-------|
| `json_load/2` | ✓ | ✗ | Load JSON from file |
| `json_loads/2` | ✓ | ✗ | Parse JSON string |
| `json_dump/2` | ✓ | ✗ | Write JSON to file |
| `json_dumps/2` | ✓ | ✗ | Serialize to JSON |

---

## Regex Operations

| Binding | Python | PowerShell | Notes |
|---------|:------:|:----------:|-------|
| `regex_search/3` | ✓ | ✗ | Search for pattern |
| `regex_match/3` | ✓ | ✗ | Match at beginning |
| `regex_fullmatch/3` | ✓ | ✗ | Full string match |
| `regex_findall/3` | ✓ | ✗ | Find all matches |
| `regex_finditer/3` | ✓ | ✗ | Find all as iterator |
| `regex_sub/4` | ✓ | ✗ | Substitute pattern |
| `regex_split/3` | ✓ | ✗ | Split by pattern |
| `regex_compile/2` | ✓ | ✗ | Compile pattern |

---

## Sequence/Iterator Operations

| Binding | Python | PowerShell | Notes |
|---------|:------:|:----------:|-------|
| `range/2` | ✓ | ✗ | Generate range |
| `range/3` | ✓ | ✗ | Range with start/stop |
| `range/4` | ✓ | ✗ | Range with step |
| `enumerate/2` | ✓ | ✗ | Add index to iterable |
| `zip/3` | ✓ | ✗ | Pair iterables |

---

## Collections Module (Python)

| Binding | Python | PowerShell | Notes |
|---------|:------:|:----------:|-------|
| `deque/2` | ✓ | ✗ | Double-ended queue |
| `counter/2` | ✓ | ✗ | Counter |
| `defaultdict/2` | ✓ | ✗ | Default dict |
| `namedtuple/3` | ✓ | ✗ | Named tuple factory |
| `ordereddict/2` | ✓ | ✗ | Ordered dict |

---

## Itertools Module (Python)

| Binding | Python | PowerShell | Notes |
|---------|:------:|:----------:|-------|
| `chain/2` | ✓ | ✗ | Chain iterables |
| `combinations/3` | ✓ | ✗ | Combinations |
| `permutations/3` | ✓ | ✗ | Permutations |
| `product/2` | ✓ | ✗ | Cartesian product |
| `groupby/3` | ✓ | ✗ | Group consecutive |
| `islice/4` | ✓ | ✗ | Slice iterator |
| `takewhile/3` | ✓ | ✗ | Take while condition |
| `dropwhile/3` | ✓ | ✗ | Drop while condition |
| `cycle/2` | ✓ | ✗ | Cycle through |
| `repeat/3` | ✓ | ✗ | Repeat value |

---

## I/O Output Operations

| Binding | Python | PowerShell | Notes |
|---------|:------:|:----------:|-------|
| `print/1` | ✓ | ✗ | Print to stdout |
| `input/1` | ✓ | ✗ | Read from stdin |
| `input/2` | ✓ | ✗ | Read with prompt |
| `write_output/1` | ✗ | ✓ | Write-Output |
| `write_host/1` | ✗ | ✓ | Write-Host |
| `write_verbose/1` | ✗ | ✓ | Write-Verbose |
| `write_debug/1` | ✗ | ✓ | Write-Debug |
| `write_warning/1` | ✗ | ✓ | Write-Warning |
| `write_error/1` | ✗ | ✓ | Write-Error |

---

## PowerShell Pipeline Operations

| Binding | Python | PowerShell | Notes |
|---------|:------:|:----------:|-------|
| `foreach_object/2` | ✗ | ✓ | ForEach-Object |
| `where_object/2` | ✗ | ✓ | Where-Object |
| `sort_object/2` | ✗ | ✓ | Sort-Object |
| `group_object/2` | ✗ | ✓ | Group-Object |
| `measure_object/2` | ✗ | ✓ | Measure-Object |
| `select_object/2` | ✗ | ✓ | Select-Object |
| `ps_object/2` | ✗ | ✓ | [PSCustomObject] |

---

## PowerShell Windows Services

| Binding | Python | PowerShell | Notes |
|---------|:------:|:----------:|-------|
| `get_service/1` | ✗ | ✓ | Get all services |
| `get_service/2` | ✗ | ✓ | Get service by name |
| `start_service/1` | ✗ | ✓ | Start service |
| `stop_service/1` | ✗ | ✓ | Stop service |
| `restart_service/1` | ✗ | ✓ | Restart service |

---

## PowerShell Process Operations

| Binding | Python | PowerShell | Notes |
|---------|:------:|:----------:|-------|
| `get_process/1` | ✗ | ✓ | Get all processes |
| `get_process/2` | ✗ | ✓ | Get process by name |
| `start_process/2` | ✗ | ✓ | Start process |
| `stop_process/1` | ✗ | ✓ | Stop process |

---

## PowerShell Registry Operations

| Binding | Python | PowerShell | Notes |
|---------|:------:|:----------:|-------|
| `get_item_property/3` | ✗ | ✓ | Read registry value |
| `set_item_property/3` | ✗ | ✓ | Write registry value |
| `new_registry_key/1` | ✗ | ✓ | Create registry key |

---

## PowerShell WMI/CIM Operations

| Binding | Python | PowerShell | Notes |
|---------|:------:|:----------:|-------|
| `get_cim_instance/2` | ✗ | ✓ | Query WMI/CIM |
| `invoke_cim_method/4` | ✗ | ✓ | Call WMI method |
| `get_win_event/2` | ✗ | ✓ | Query event logs |
| `write_event_log/4` | ✗ | ✓ | Write to event log |

---

## PowerShell XML Operations

| Binding | Python | PowerShell | Notes |
|---------|:------:|:----------:|-------|
| `xml_reader_create/2` | ✗ | ✓ | Create XmlReader |
| `xml_document_load/2` | ✗ | ✓ | Load XmlDocument |

---

## PowerShell C# Hosting

| Binding | Python | PowerShell | Notes |
|---------|:------:|:----------:|-------|
| `add_type/1` | ✗ | ✓ | Compile inline C# |
| `load_assembly/1` | ✗ | ✓ | Load assembly |
| `load_dll/1` | ✗ | ✓ | Load DLL |
| `new_object/2` | ✗ | ✓ | Create .NET object |
| `new_object/3` | ✗ | ✓ | Create with args |
| `create_runspace/1` | ✗ | ✓ | Create runspace |
| `open_runspace/1` | ✗ | ✓ | Open runspace |
| `create_powershell/1` | ✗ | ✓ | Create PS instance |
| `add_script/2` | ✗ | ✓ | Add script |
| `add_command/2` | ✗ | ✓ | Add command |
| `add_parameter/3` | ✗ | ✓ | Add parameter |
| `invoke_powershell/2` | ✗ | ✓ | Invoke and get results |
| `cast_type/3` | ✗ | ✓ | Cast to .NET type |
| `get_assembly_types/2` | ✗ | ✓ | Get types from assembly |
| `get_type_assembly/2` | ✗ | ✓ | Get assembly from type |

---

## Cross-Target Bindings (Shared)

These bindings have the same Prolog predicate name but different implementations:

| Binding | Python Target | PowerShell Target |
|---------|---------------|-------------------|
| `abs/2` | `abs()` | `[Math]::Abs()` |
| `sqrt/2` | `math.sqrt()` | `[Math]::Sqrt()` |
| `to_string/2` | `str()` | `[string]` |
| `to_int/2` | `int()` | `[int]` |
| `string_split/3` | `.split()` | `.Split()` |
| `string_replace/4` | `.replace()` | `.Replace()` |

---

## Future Targets

Placeholder for future target languages:

| Target | Status | Notes |
|--------|--------|-------|
| Go | Planned | Native bindings for Go standard library |
| Rust | Planned | Bindings for Rust std crate |
| C# | Planned | LINQ and .NET bindings |
| Bash | Planned | Shell command bindings |
| SQL | N/A | SQL uses declarative generation, not bindings |

---

## Adding New Bindings

To add a binding for a target:

1. Edit the target's binding file (e.g., `src/unifyweaver/bindings/python_bindings.pl`)
2. Add `declare_binding/6` in the appropriate registration function
3. Update the binding matrix in this document
4. Run tests to verify

Example:
```prolog
declare_binding(python, new_predicate/2, 'target_function',
    [input_type], [output_type],
    [pure, deterministic, total]).
```

---

**Maintained by:** UnifyWeaver Team
