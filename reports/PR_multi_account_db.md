# PR: Multi-Account Database for Pearltrees

## Title
feat(pearltrees): Add multi-account SQLite database with pearl import

## Summary

This PR adds multi-account support to the Pearltrees database schema and enables showing existing folder contents when recommending where to file bookmarks.

## Key Features

### Multi-Account Importer
```python
from importer import PtMultiAccountImporter, extract_account_from_uri

# Auto-extract account from URI
extract_account_from_uri("https://www.pearltrees.com/s243a/item")  # -> "s243a"

# Use database
importer = PtMultiAccountImporter("pearltrees.db")
importer.upsert_object("tree1", "tree", {...}, account="s243a")
pearls = importer.get_pearls_in_tree("tree1", "s243a")
```

### Pearl Import Script
```bash
python3 scripts/import_pearltrees_to_db.py \
  --account s243a data/s243a.rdf \
  --account s243a_groups data/s243a_groups.rdf \
  --output pearltrees.db
```

### Filing Assistant Integration
```bash
python3 scripts/bookmark_filing_assistant.py \
  --bookmark "Machine learning tutorial" \
  --db pearltrees.db
```

The LLM now sees existing folder contents:
```
## Existing Bookmarks in Candidate Folders

**Deep Learning** (#2): "PyTorch tutorial", "Keras getting started"
**Machine Learning** (#3): "Scikit-learn guide", "ML overview"
```

## Schema (v2)

```sql
-- Objects with account
CREATE TABLE objects (
    id TEXT NOT NULL,
    account TEXT NOT NULL DEFAULT 'unknown',
    type TEXT,
    data JSON,
    PRIMARY KEY (id, account)
);

-- Indexes for fast queries
CREATE INDEX idx_objects_account ON objects(account);
CREATE INDEX idx_objects_type ON objects(type);

-- Links with account
CREATE TABLE links (
    source_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    account TEXT NOT NULL DEFAULT 'unknown',
    link_type TEXT DEFAULT 'contains',
    PRIMARY KEY (source_id, target_id, account)
);
```

## Files Changed

| File | Change |
|------|--------|
| `src/.../importer.py` | Rewritten with multi-account support |
| `scripts/import_pearltrees_to_db.py` | New - imports RDF to SQLite |
| `scripts/bookmark_filing_assistant.py` | Added --db flag, shows existing pearls |
| `skills/skill_bookmark_filing.md` | Added DB import section |

## New Methods on PtMultiAccountImporter

| Method | Description |
|--------|-------------|
| `get_pearls_in_tree(tree_id, account)` | Get bookmarks in a tree |
| `get_trees_by_account(account)` | List trees for an account |
| `get_all_accounts()` | List all accounts |
| `search_by_title(query, account)` | Search by title |
| `stats()` | Database statistics |

## Migration

The importer includes automatic schema migration:
- Detects old schema (v1)
- Adds account column if missing
- Creates indexes

## Co-authored-by

Co-authored-by: Claude <claude@anthropic.com>
Co-authored-by: John William Creighton (s243a) <JohnCreighton_@hotmail.com>
