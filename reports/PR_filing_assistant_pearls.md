# PR: Add Pearl Model Support to Filing Assistant

## Title
feat(filing): Add support for Pearl model to find similar bookmarks

## Summary
Updates the `bookmark_filing_assistant.py` to accept a secondary semantic search model (`--pearl-model`), which is used to find existing bookmarks similar to the one being filed. These similar references are provided to the LLM to improve filing consistency.

## Changes
- **Updated `file_bookmark`**:
  - Accepts `pearl_model_path`.
  - Queries this model using `locate_url(...)` template.
  - Formats results into a "Similar Existing Bookmarks" section in the LLM prompt.
- **Updated CLI**:
  - Added `--pearl-model` argument.
  - Updated `interactive_mode` to pass this model.

## Usage
```bash
python3 scripts/bookmark_filing_assistant.py \
    --bookmark "The Feynman Lectures" \
    --model models/pearltrees_federated_single.pkl \
    --pearl-model models/pearltrees_full_pearls.pkl \
    --data reports/pearltrees_targets_full_multi_account.jsonl
```

## Co-authored-by
Co-authored-by: Claude <claude@anthropic.com>
Co-authored-by: John William Creighton (s243a) <JohnCreighton_@hotmail.com>
