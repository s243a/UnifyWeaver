# agent_refiner.py
import json
from typing import List, Dict, Any, Optional

REFINE_PROMPT = """You are given a query and one or more files (or large chunks).
Propose the most relevant spans with start and end lines, a short label, and a concise rationale.
Output strict JSON as a list of objects: 
[{ "file_path": str, "start_line": int, "end_line": int, "label": str, "rationale": str, "confidence": float }].
Query: {query}
"""

class AgentRefiner:
    def __init__(self, call_long_context=None):
        """
        call_long_context: Optional[Callable[[str], str]] returning raw text (JSON string).
        If not provided, implement your provider call inside refine_files.
        """
        self.call_long_context = call_long_context

    async def refine_files(
        self,
        query: str,
        files: List[str],
        macro_index: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Given top files and optional macro chunks per file, ask a long-context model to propose refined spans.
        """
        if not files:
            return []

        # Construct prompt material — either attach macro chunks or load file contents as needed.
        # For now, only include file paths; integrate your actual content retrieval or macro text here.
        payload = {
            "query": query,
            "files": files,
            "macro_info": {fp: len(macro_index.get(fp, [])) if macro_index else 0 for fp in files},
        }
        prompt = REFINE_PROMPT.format(query=query) + "\nContext:\n" + json.dumps(payload, ensure_ascii=False)

        raw = None
        if self.call_long_context:
            raw = await self.call_long_context(prompt)

        # TODO: Implement provider call here if call_long_context not supplied.
        if raw is None:
            # Safe fallback to empty refinements; Stage 1 results will carry the run.
            return []

        try:
            data = json.loads(raw)
            out: List[Dict[str, Any]] = []
            for rec in data:
                fp = rec.get("file_path")
                sline = int(rec.get("start_line", 0))
                eline = int(rec.get("end_line", 0))
                label = str(rec.get("label", "")).strip()
                rationale = str(rec.get("rationale", "")).strip()
                conf = float(rec.get("confidence", 0.0))
                out.append({
                    "file": fp,
                    "span": f"{sline}-{eline}",
                    "label": label,
                    "rationale": rationale,
                    "score": max(0.0, min(conf, 1.0)),  # map confidence to [0,1] score
                    "content_preview": f"{label}: lines {sline}-{eline}",
                })
            return out
        except Exception:
            # If parsing fails, return no refinements to avoid breaking the pipeline
            return []
