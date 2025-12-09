# Add PowerShell Data Partitioning Target

## Summary

Ports `bash_partitioning_target.pl` functionality to pure PowerShell, completing Phase 3 of the PowerShell pure implementation roadmap.

## New File

`src/unifyweaver/targets/powershell_partitioning_target.pl` (466 lines)

## Strategies Implemented

| Strategy | Bash Equivalent | PowerShell Approach |
|----------|-----------------|---------------------|
| `fixed_size(rows(N))` | `split -l` | `Get-Content` + array slicing |
| `fixed_size(bytes(N))` | `split -b` | `ReadAllBytes` + `Array.Copy` |
| `hash_based([...])` | AWK hash function | Character code sum % N |
| `key_based([...])` | AWK GROUP BY | Hashtable key tracking |

## Usage

```prolog
% Fixed-size by rows
generate_powershell_partitioner(fixed_size(rows(1000)), [], Code)

% Hash-based on column 1 into 8 partitions
generate_powershell_partitioner(hash_based([column(1), num_partitions(8)]), [], Code)

% Key-based (GROUP BY) on column 0
generate_powershell_partitioner(key_based([column(0)]), [], Code)
```

## Generated Functions

Each strategy generates:
- `Split-Data` - Main partitioning function
- `Get-Partitions` - List partition files
- `Get-PartitionCount` - Get count
- `metadata.json` - Partition metadata

## Example Output (fixed_size)

```powershell
function Split-Data {
    param(
        [string]$InputFile,
        [string]$OutputDir,
        [int]$RowsPerPartition = 1000
    )
    
    $lines = Get-Content -Path $InputFile
    for ($i = 0; $i -lt $partitionCount; $i++) {
        $lines[$start..$end] | Out-File -FilePath $partitionFile
    }
    
    $metadata | ConvertTo-Json | Out-File 'metadata.json'
}
```

## Roadmap Progress

- Phase 1: CSV/JSON/HTTP ✅
- Phase 2: Recursion patterns ✅
- **Phase 3: Data partitioning ✅**
- Phase 4: PowerShell object pipeline (pending)
