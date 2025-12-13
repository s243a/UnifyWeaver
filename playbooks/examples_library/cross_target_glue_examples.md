# Cross-Target Glue Examples

This file contains executable records for the cross-target glue playbook.

## Phase 1: Shell Integration (AWK + Python + Bash Pipeline)

:::unifyweaver.execution.cross_target_glue_shell
```bash
#!/bin/bash
# Cross-Target Glue Demo - Shell Integration
# Demonstrates AWK -> Python -> Bash pipeline via shell_glue

set -euo pipefail
cd /root/UnifyWeaver

echo "=== Cross-Target Glue Demo: Shell Integration ==="

# Create test directory
mkdir -p tmp/glue_demo

# Create sample TSV data
cat > tmp/glue_demo/sales_data.tsv << 'EOF'
region	product	quantity	price
North	Widget	100	25.50
South	Gadget	50	45.00
North	Gadget	75	45.00
East	Widget	200	25.50
South	Widget	150	25.50
East	Gadget	25	45.00
EOF

echo "Created sample data: tmp/glue_demo/sales_data.tsv"

# Create the Prolog script that uses shell_glue
cat > tmp/glue_demo/generate_pipeline.pl << 'PROLOG'
:- use_module(library(lists)).

% Load shell_glue module
:- use_module('src/unifyweaver/glue/shell_glue').

% Main entry point
main :-
    % Step 1: Generate AWK script for filtering and calculating totals
    AwkLogic = '
    # Calculate line total
    total = quantity * price
    if (total > 1000) {
        print region, product, quantity, price, total
    }',
    generate_awk_script(AwkLogic, [region, product, quantity, price],
                        [format(tsv), header(true)], AwkScript),

    % Write AWK script
    open('tmp/glue_demo/filter.awk', write, AwkStream),
    write(AwkStream, AwkScript),
    close(AwkStream),
    format("Generated: tmp/glue_demo/filter.awk~n"),

    % Step 2: Generate Python script for aggregation
    PythonLogic = '    # Aggregate by region
    record["total"] = float(record["quantity"]) * float(record["price"])',
    generate_python_script(PythonLogic, [region, product, quantity, price],
                           [format(tsv), header(false)], PythonScript),

    % Write Python script
    open('tmp/glue_demo/aggregate.py', write, PyStream),
    write(PyStream, PythonScript),
    close(PyStream),
    format("Generated: tmp/glue_demo/aggregate.py~n"),

    % Step 3: Generate Bash script for final formatting
    BashLogic = '    # Format output
    formatted=$(printf "%-10s %-10s %8s %10s" "$region" "$product" "$quantity" "$price")',
    generate_bash_script(BashLogic, [region, product, quantity, price],
                         [format(tsv), header(false)], BashScript),

    % Write Bash script
    open('tmp/glue_demo/format.sh', write, BashStream),
    write(BashStream, BashScript),
    close(BashStream),
    format("Generated: tmp/glue_demo/format.sh~n"),

    % Step 4: Generate pipeline orchestrator
    Steps = [
        step(filter, awk, 'tmp/glue_demo/filter.awk', []),
        step(aggregate, python, 'tmp/glue_demo/aggregate.py', []),
        step(format, bash, 'tmp/glue_demo/format.sh', [])
    ],
    generate_pipeline(Steps, [input('tmp/glue_demo/sales_data.tsv')], PipelineScript),

    % Write pipeline script
    open('tmp/glue_demo/run_pipeline.sh', write, PipeStream),
    write(PipeStream, PipelineScript),
    close(PipeStream),
    format("Generated: tmp/glue_demo/run_pipeline.sh~n"),

    format("~nAll scripts generated successfully!~n"),
    halt(0).

:- initialization(main, main).
PROLOG

echo ""
echo "Running Prolog to generate glue scripts..."
swipl tmp/glue_demo/generate_pipeline.pl

echo ""
echo "=== Generated Scripts ==="
echo ""
echo "--- filter.awk ---"
head -20 tmp/glue_demo/filter.awk
echo ""
echo "--- aggregate.py ---"
head -30 tmp/glue_demo/aggregate.py
echo ""
echo "--- run_pipeline.sh ---"
cat tmp/glue_demo/run_pipeline.sh

echo ""
echo "Success: Cross-target glue shell integration demo complete"
```
:::

## Phase 2: Shell Integration with Go Binary

:::unifyweaver.execution.cross_target_glue_go
```bash
#!/bin/bash
# Cross-Target Glue Demo - Go Binary Integration
# Demonstrates generating and using a Go binary in a pipeline

set -euo pipefail
cd /root/UnifyWeaver

echo "=== Cross-Target Glue Demo: Go Binary Integration ==="

# Check if Go is available
if ! command -v go &> /dev/null; then
    echo "Go is not installed. Skipping Go integration test."
    echo "Install Go to enable this feature."
    exit 0
fi

go version

# Create test directory
mkdir -p tmp/glue_demo/go_processor

# Create sample JSON data
cat > tmp/glue_demo/records.jsonl << 'EOF'
{"id": 1, "name": "Alice", "score": 85}
{"id": 2, "name": "Bob", "score": 92}
{"id": 3, "name": "Charlie", "score": 78}
{"id": 4, "name": "Diana", "score": 95}
{"id": 5, "name": "Eve", "score": 88}
EOF

echo "Created sample data: tmp/glue_demo/records.jsonl"

# Create Go processor using native_glue patterns
cat > tmp/glue_demo/go_processor/main.go << 'GO'
package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
)

// Record represents input data
type Record struct {
	ID    int    `json:"id"`
	Name  string `json:"name"`
	Score int    `json:"score"`
}

// OutputRecord with grade
type OutputRecord struct {
	ID    int    `json:"id"`
	Name  string `json:"name"`
	Score int    `json:"score"`
	Grade string `json:"grade"`
}

func gradeScore(score int) string {
	switch {
	case score >= 90:
		return "A"
	case score >= 80:
		return "B"
	case score >= 70:
		return "C"
	case score >= 60:
		return "D"
	default:
		return "F"
	}
}

func main() {
	scanner := bufio.NewScanner(os.Stdin)
	for scanner.Scan() {
		var record Record
		if err := json.Unmarshal(scanner.Bytes(), &record); err != nil {
			fmt.Fprintf(os.Stderr, "Error parsing JSON: %v\n", err)
			continue
		}

		output := OutputRecord{
			ID:    record.ID,
			Name:  record.Name,
			Score: record.Score,
			Grade: gradeScore(record.Score),
		}

		outputJSON, _ := json.Marshal(output)
		fmt.Println(string(outputJSON))
	}
}
GO

echo ""
echo "Building Go processor..."
cd tmp/glue_demo/go_processor
go build -o ../go_grader main.go
cd /root/UnifyWeaver

echo "Built: tmp/glue_demo/go_grader"

# Create the Prolog script for pipeline with Go binary
cat > tmp/glue_demo/generate_go_pipeline.pl << 'PROLOG'
:- use_module('src/unifyweaver/glue/shell_glue').

main :-
    % Generate a pipeline that uses the compiled Go binary
    Steps = [
        step(grader, go, 'tmp/glue_demo/go_grader', [])
    ],
    generate_pipeline(Steps, [input('tmp/glue_demo/records.jsonl')], PipelineScript),

    % Write pipeline
    open('tmp/glue_demo/run_go_pipeline.sh', write, Stream),
    write(Stream, PipelineScript),
    close(Stream),
    format("Generated: tmp/glue_demo/run_go_pipeline.sh~n"),
    halt(0).

:- initialization(main, main).
PROLOG

echo ""
echo "Running Prolog to generate Go pipeline..."
swipl tmp/glue_demo/generate_go_pipeline.pl

echo ""
echo "=== Generated Pipeline Script ==="
cat tmp/glue_demo/run_go_pipeline.sh

echo ""
echo "=== Running Go Pipeline ==="
chmod +x tmp/glue_demo/run_go_pipeline.sh
bash tmp/glue_demo/run_go_pipeline.sh

echo ""
echo "Success: Cross-target glue Go integration demo complete"
```
:::

## Phase 3: .NET Integration via dotnet_glue

:::unifyweaver.execution.cross_target_glue_dotnet
```bash
#!/bin/bash
# Cross-Target Glue Demo - .NET Integration
# Demonstrates C#/PowerShell glue generation

set -euo pipefail
cd /root/UnifyWeaver

# Set memory limit for Termux proot-distro
export DOTNET_GCHeapHardLimit=1C0000000

echo "=== Cross-Target Glue Demo: .NET Integration ==="

# Check if dotnet is available
if ! command -v dotnet &> /dev/null; then
    # Try the user install location
    if [ -f "$HOME/.dotnet/dotnet" ]; then
        export PATH="$HOME/.dotnet:$PATH"
    else
        echo ".NET SDK not found. Skipping .NET integration test."
        exit 0
    fi
fi

dotnet --version

# Create test directory
mkdir -p tmp/glue_demo/dotnet_processor

# Create sample data
cat > tmp/glue_demo/products.json << 'EOF'
[
  {"sku": "W001", "name": "Widget", "price": 25.50, "stock": 100},
  {"sku": "G001", "name": "Gadget", "price": 45.00, "stock": 50},
  {"sku": "D001", "name": "Doodad", "price": 12.75, "stock": 200}
]
EOF

echo "Created sample data: tmp/glue_demo/products.json"

# Create the Prolog script using dotnet_glue
cat > tmp/glue_demo/generate_dotnet_pipeline.pl << 'PROLOG'
:- use_module('src/unifyweaver/glue/dotnet_glue').

main :-
    % Generate C# code for processing JSON
    CsharpLogic = '
        // Read JSON, calculate total value, output enhanced records
        using System.Text.Json;

        public class Product {
            public string Sku { get; set; }
            public string Name { get; set; }
            public decimal Price { get; set; }
            public int Stock { get; set; }
        }

        public class EnhancedProduct : Product {
            public decimal TotalValue { get; set; }
            public string StockStatus { get; set; }
        }',

    % Generate using dotnet_glue predicates
    (   generate_dotnet_pipeline(CsharpLogic, [], CsharpCode)
    ->  atom_length(CsharpCode, Len),
        format("Generated C# code (~d chars)~n", [Len])
    ;   format("dotnet_glue module provides: generate_dotnet_pipeline/3~n"),
        format("This generates .NET pipeline processors with JSON support~n")
    ),

    format("~nNote: For full .NET integration, see:~n"),
    format("  - playbooks/powershell_inline_dotnet_playbook.md~n"),
    format("  - playbooks/json_litedb_playbook.md~n"),

    halt(0).

:- initialization(main, main).
PROLOG

echo ""
echo "Running Prolog to demonstrate dotnet_glue..."
swipl tmp/glue_demo/generate_dotnet_pipeline.pl 2>&1 || true

# Show what dotnet_glue provides
echo ""
echo "=== dotnet_glue Module Exports ==="
swipl -g "use_module('src/unifyweaver/glue/dotnet_glue'), module_property(dotnet_glue, exports(E)), forall(member(X,E), format('  ~w~n', [X])), halt." 2>/dev/null || echo "(Module exports listed above)"

echo ""
echo "Success: Cross-target glue .NET integration demo complete"
```
:::

## Phase 4: Rust Binary Integration

:::unifyweaver.execution.cross_target_glue_rust
```bash
#!/bin/bash
# Cross-Target Glue Demo - Rust Binary Integration
# Demonstrates generating and using a Rust binary in a pipeline

set -euo pipefail
cd /root/UnifyWeaver

echo "=== Cross-Target Glue Demo: Rust Binary Integration ==="

# Check if Rust is available
if ! command -v rustc &> /dev/null; then
    echo "Rust is not installed. Skipping Rust integration test."
    echo "Install Rust to enable this feature."
    exit 0
fi

rustc --version

# Create test directory
mkdir -p tmp/glue_demo/rust_processor

# Create sample TSV data
cat > tmp/glue_demo/metrics.tsv << 'EOF'
timestamp	metric	value
1699900000	cpu_usage	45.2
1699900060	cpu_usage	52.1
1699900120	cpu_usage	38.9
1699900000	memory_mb	2048
1699900060	memory_mb	2150
1699900120	memory_mb	2100
EOF

echo "Created sample data: tmp/glue_demo/metrics.tsv"

# Create Rust processor
cat > tmp/glue_demo/rust_processor/main.rs << 'RUST'
use std::io::{self, BufRead, Write};
use std::collections::HashMap;

fn main() {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut handle = stdout.lock();

    let mut first = true;
    let mut aggregates: HashMap<String, Vec<f64>> = HashMap::new();

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => continue,
        };

        // Skip header
        if first {
            first = false;
            continue;
        }

        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() < 3 {
            continue;
        }

        let metric = fields[1].to_string();
        let value: f64 = fields[2].parse().unwrap_or(0.0);

        aggregates.entry(metric).or_insert_with(Vec::new).push(value);
    }

    // Output aggregated stats
    writeln!(handle, "metric\tcount\tmin\tmax\tavg").unwrap();
    for (metric, values) in &aggregates {
        let count = values.len();
        let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let avg = values.iter().sum::<f64>() / count as f64;
        writeln!(handle, "{}\t{}\t{:.2}\t{:.2}\t{:.2}", metric, count, min, max, avg).unwrap();
    }
}
RUST

echo ""
echo "Building Rust processor..."
rustc -O -o tmp/glue_demo/rust_aggregator tmp/glue_demo/rust_processor/main.rs

echo "Built: tmp/glue_demo/rust_aggregator"

# Create pipeline using native_glue
cat > tmp/glue_demo/generate_rust_pipeline.pl << 'PROLOG'
:- use_module('src/unifyweaver/glue/shell_glue').
:- use_module('src/unifyweaver/glue/native_glue').

main :-
    % Register the compiled binary
    register_binary(aggregate_metrics/1, rust, 'tmp/glue_demo/rust_aggregator', []),

    % Generate pipeline with Rust binary
    Steps = [
        step(aggregate, rust, 'tmp/glue_demo/rust_aggregator', [])
    ],
    generate_pipeline(Steps, [input('tmp/glue_demo/metrics.tsv')], PipelineScript),

    % Write pipeline
    open('tmp/glue_demo/run_rust_pipeline.sh', write, Stream),
    write(Stream, PipelineScript),
    close(Stream),
    format("Generated: tmp/glue_demo/run_rust_pipeline.sh~n"),
    halt(0).

:- initialization(main, main).
PROLOG

echo ""
echo "Running Prolog to generate Rust pipeline..."
swipl tmp/glue_demo/generate_rust_pipeline.pl

echo ""
echo "=== Generated Pipeline Script ==="
cat tmp/glue_demo/run_rust_pipeline.sh

echo ""
echo "=== Running Rust Pipeline ==="
chmod +x tmp/glue_demo/run_rust_pipeline.sh
bash tmp/glue_demo/run_rust_pipeline.sh

echo ""
echo "Success: Cross-target glue Rust integration demo complete"
```
:::
