# PR: Add Pearl Model Support to Filing Assistant

## Title
feat(filing): Add support for Pearl model to find similar bookmarks

## Summary
This PR enhances the `bookmark_filing_assistant.py` to support a secondary semantic search model (`--pearl-model`) trained on individual bookmarks (Pearls). This allows the assistant to find existing bookmarks that are semantically similar to the one being filed and provide them as context to the LLM.

This "dual-model" approach combines:
1.  **Structure Awareness**: The primary model finds relevant folders (Trees).
2.  **Content Awareness**: The secondary pearl model finds similar content items (Pearls), enabling "file-like-this" behavior.

## Key Changes

### `scripts/bookmark_filing_assistant.py`
- Added `--pearl-model` argument to CLI.
- Implemented logic to query the pearl model using the `locate_url(...)` template.
- Formats retrieved pearls into a `## Similar Existing Bookmarks` section in the LLM prompt.
- Updated `interactive_mode` to support the new argument.

## Usage example

```bash
python3 scripts/bookmark_filing_assistant.py \
    --bookmark "The Feynman Lectures on Physics" \
    --model models/pearltrees_federated_single.pkl \
    --pearl-model models/pearltrees_full_pearls.pkl \
    --data reports/pearltrees_targets_full_multi_account.jsonl
```

## Co-authored-by
Co-authored-by: Claude <claude@anthropic.com>
Co-authored-by: John William Creighton (s243a) <JohnCreighton_@hotmail.com>
