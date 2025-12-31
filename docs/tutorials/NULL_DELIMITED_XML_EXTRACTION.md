# Null-Delimited XML Extraction (Pearltrees example)

Quick steps to extract XML elements as NUL-delimited records and consume them from Prolog or C#.

## Extracting elements with awk

Use the existing selector: `scripts/utils/select_xml_elements.awk`

Examples (case-insensitive post-filter on “physics” to trim output):
```bash
# Extract all <pt:Tree> elements
awk -f scripts/utils/select_xml_elements.awk -v tag="pt:Tree" context/PT/pearltrees_export.rdf > tmp/trees.nulxml

# Extract all pearl-like elements that mention physics
awk -f scripts/utils/select_xml_elements.awk -v tag="pt:.*Pearl" context/PT/pearltrees_export.rdf \
  | LC_ALL=C grep -i "physics" \
  > tmp/pearls_physics.nulxml
```
- Default delimiter is NUL (`\0`) per record.
- Adjust `tag` (regex) as needed, e.g. `rdf:Description`, `pt:Tree`, `pt:.*Pearl`.

## Consuming from Prolog (sketch)
- Open the NUL-delimited file as a stream.
- Read chunks split on NUL; hand each chunk to an XML parser or downstream target.
- Namespace handling: for each fragment, rely on in-fragment `xmlns` declarations (no external lookups).

## Consuming from C# (sketch)
- Read bytes and split on NUL to get each XML fragment.
- For each fragment:
  - Create an `XmlReader` with whitespace/comments ignored.
  - Build an `XmlNamespaceManager` from `xmlns` declarations in the fragment.
  - Match elements/attributes by prefix+local and fully qualified `{ns}local` names; if only a prefix is available, use the local declaration (no external schemas).
- Map to objects or keep as `XElement` for further processing.

## Notes
- Keep extraction filters narrow (e.g. `grep -i "physics"`) to reduce context size for LLM agents.
- If you need a different delimiter, pass `-v delimiter="---"` to the awk script.
# Null-Delimited XML Extraction (Pearltrees example)
**Status:** Draft / proposal (will be updated when canonical documentation lands)

# Null-Delimited XML Extraction (Pearltrees example)
