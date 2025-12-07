# Chapter 3: PowerShell Semantic Target (XML & Vector)

UnifyWeaver's **PowerShell Semantic Target** bridges the gap between declarative Prolog logic and the .NET ecosystem on Windows (and Linux via PowerShell Core). 

While Python excels at "Scripted AI," PowerShell excels at "Enterprise Integration."

## 3.1 The Objective: Semantic Parity

Our goal is to achieve feature parity with the Python semantic runtime, specifically:
1.  **XML/RDF Ingestion**: Efficient, streaming parsing of large datasets.
2.  **Vector Search**: Embeddings and similarity search.

However, unlike Python (where we use `lxml` and `numpy`), we leverage the **.NET Framework**.

## 3.2 XML Streaming with `.NET`

PowerShell can directly access .NET classes. We use `[System.Xml.XmlReader]` for memory-efficient streaming.

**Prolog Source Definition:**
```prolog
:- source(xml, my_data, [
    input_file('data.xml'),
    target(powershell)
]).
```

**Generated PowerShell (Concept):**
```powershell
function Get-XmlStream {
    param($Path, $Tags)
    $reader = [System.Xml.XmlReader]::Create($Path)
    try {
        while ($reader.Read()) {
            if ($reader.NodeType -eq 'Element' -and $Tags -contains $reader.Name) {
                # ReadSubtree returns a new XmlReader for just this element
                $subReader = $reader.ReadSubtree()
                # Parse into XmlDocument for easy property access
                $doc = [System.Xml.XmlDocument]::new()
                $doc.Load($subReader)
                
                # Flatten to Hashtable (UnifyWeaver Standard Format)
                $obj = @{
                    'tag' = $doc.DocumentElement.Name
                    'text' = $doc.DocumentElement.InnerText
                }
                foreach ($attr in $doc.DocumentElement.Attributes) {
                    $obj["@$($attr.Name)"] = $attr.Value
                }
                
                # Emit to pipeline
                $obj
            }
        }
    } finally {
        $reader.Dispose()
    }
}
```

## 3.3 Vector Search Strategies

For Vector Search, we have two strategies depending on scale:

### Strategy A: Pure PowerShell (Small Scale)
For datasets < 10k items, we can perform Cosine Similarity using pure PowerShell math.

```powershell
function Get-CosineSimilarity ($v1, $v2) {
    $dot = 0.0; $mag1 = 0.0; $mag2 = 0.0
    for ($i = 0; $i -lt $v1.Count; $i++) {
        $dot += $v1[$i] * $v2[$i]
        $mag1 += $v1[$i] * $v1[$i]
        $mag2 += $v2[$i] * $v2[$i]
    }
    return $dot / ([Math]::Sqrt($mag1) * [Math]::Sqrt($mag2))
}
```

### Strategy B: ONNX Runtime (Production)
For production, we load the `Microsoft.ML.OnnxRuntime.dll` assembly.

```powershell
Add-Type -Path "bin/Microsoft.ML.OnnxRuntime.dll"
$session = [Microsoft.ML.OnnxRuntime.InferenceSession]::new("model.onnx")
# ... tensor manipulation ...
```

## 3.4 Architecture of the PowerShell Compiler

The new `src/unifyweaver/targets/powershell_target.pl` orchestrates this:

1.  **`compile_source_xml/3`**: Generates the `XmlReader` logic.
2.  **`compile_vector_ops/3`**: Inlines the vector math helper functions.
3.  **`compile_predicate_to_powershell/3`**: Wraps it all in a PowerShell script (ps1).

This ensures that UnifyWeaver logic can run natively on corporate Windows servers without needing Python installed.

---

## Navigation

**←** [Previous: Chapter 5: Semantic Crawling and Vector Search](05_semantic_crawling) | [📖 Book 3: C# Target](./) | [Next: Book 4: Workflows →](../book-04-workflows/)
