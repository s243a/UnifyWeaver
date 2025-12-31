# Cross-Target Glue Examples

Examples demonstrating UnifyWeaver's cross-target communication capabilities.

## Log Analysis Pipeline

A three-stage pipeline that processes Apache-style access logs:

```
AWK (parse) → Python (analyze) → AWK (summarize)
```

### Files

| File | Description |
|------|-------------|
| `log_pipeline.pl` | Pipeline definition and script generator |
| `sample_access.log` | Sample log data for testing |

### Usage

```bash
# Generate the pipeline scripts
swipl log_pipeline.pl

# Make executable
chmod +x run_pipeline.sh

# Run on sample data
cat sample_access.log | ./run_pipeline.sh

# Or specify input file
./run_pipeline.sh < sample_access.log
```

### Pipeline Stages

**Stage 1: Parse (AWK)**
- Parses Apache Combined Log Format
- Extracts: IP, timestamp, path, status code
- Filters: Only error responses (4xx, 5xx)

**Stage 2: Analyze (Python)**
- Categorizes error type (client_error, server_error)
- Assigns severity (warning, critical)
- Extracts endpoint category (api, static, page)

**Stage 3: Summarize (AWK)**
- Counts errors by type, category, status code
- Reports unique IPs
- Produces human-readable summary

### Example Output

```
=== Error Summary ===

Total errors: 8
Unique IPs: 5

By Error Type:
  client_error: 4
  server_error: 4

By Category:
  api: 6
  static: 2

By Status Code:
  404: 3
  500: 2
  401: 1
  403: 1
  502: 1
  503: 1
```

## How It Works

The pipeline uses TSV (tab-separated values) as the intermediate format between stages:

1. AWK outputs TSV to stdout
2. Python reads TSV from stdin, processes, outputs TSV to stdout
3. AWK reads TSV from stdin, generates summary

This follows the Unix philosophy of small tools connected by pipes.
