# PowerShell Semantic Target Proposal

## Objective
Achieve **Semantic Capability Parity** with the Python target in the PowerShell backend. This includes:
1.  **Streaming XML Ingestion** (`source(xml)`).
2.  **Vector Embeddings** (via ONNX Runtime).
3.  **Vector Search** (Cosine Similarity).
4.  **Local Storage** (SQLite or efficient JSONL).

## Constraints & Environment
- **Current Env**: Debian PRoot (no `pwsh`).
- **Target Env**: Windows/Linux with PowerShell Core 7+.
- **Strategy**: **Code Generation**. We will implement Prolog predicates that emit PowerShell code. Verification will be done via **Static Analysis Tests** (checking the generated code structure).

## Implementation Plan

### 1. XML Streaming Source (`source(xml)`)
**Approach**: Use .NET `[System.Xml.XmlReader]` for memory-efficient parsing (equivalent to `etree.iterparse`).

**Generated Code Pattern**:
```powershell
function Get-XmlStream {
    param($Path, $Tags)
    $reader = [System.Xml.XmlReader]::Create($Path)
    try {
        while ($reader.Read()) {
            if ($reader.NodeType -eq 'Element' -and $Tags -contains $reader.Name) {
                # Extract content using ReadSubtree or similar
                $doc = [System.Xml.XmlDocument]::new()
                $node = $doc.ReadNode($reader)
                # Flatten to Hashtable
                $obj = @{
                    'tag' = $node.Name
                    'text' = $node.InnerText
                }
                # Attributes
                foreach ($attr in $node.Attributes) {
                    $obj["@$($attr.Name)"] = $attr.Value
                }
                $obj
            }
        }
    } finally {
        $reader.Dispose()
    }
}
```

### 2. Vector Embeddings
**Approach**:
1.  **Preferred**: Load `Microsoft.ML.OnnxRuntime.dll` via `[Reflection.Assembly]::LoadFile`.
2.  **Fallback**: Call out to the `gemini` CLI or a Python script if DLLs are missing (Hybrid mode).

**Generated Code Pattern**:
```powershell
if (Test-Path "bin/Microsoft.ML.OnnxRuntime.dll") {
    Add-Type -Path "bin/Microsoft.ML.OnnxRuntime.dll"
    # ... Instantiate InferenceSession ...
} else {
    Write-Warning "ONNX Runtime not found. Embeddings disabled."
}
```

### 3. Storage & Search
**Approach**:
- **Storage**: SQLite via `System.Data.SQLite.dll` (standard on Windows often, or easily available).
- **Search**:
    - **Native Math**: Pure PowerShell implementation of Cosine Similarity is feasible for small datasets (<10k).
    - **SQL**: If using SQLite, we can try using a math extension or retrieve-and-sort (client-side sort).

**Cosine Similarity (Pure PS)**:
```powershell
function Get-CosineSimilarity ($v1, $v2) {
    $dot = 0.0
    $mag1 = 0.0
    $mag2 = 0.0
    for ($i = 0; $i -lt $v1.Count; $i++) {
        $dot += $v1[$i] * $v2[$i]
        $mag1 += $v1[$i] * $v1[$i]
        $mag2 += $v2[$i] * $v2[$i]
    }
    return $dot / ([Math]::Sqrt($mag1) * [Math]::Sqrt($mag2))
}
```

## Proposed Prolog Modules
1.  `src/unifyweaver/targets/powershell_target.pl`: Main compiler entry point.
2.  `src/unifyweaver/targets/powershell_runtime/xml_reader.pl`: Generates the XML parsing logic.
3.  `src/unifyweaver/targets/powershell_runtime/vector_ops.pl`: Generates vector math.

## Next Steps
1.  Create `tests/test_powershell_codegen.pl` to harness the generator.
2.  Implement `compile_source_xml/3` in `powershell_target.pl`.
3.  Verify the generated output matches the expected .NET patterns.
