# XML Selection Pipeline - Generalization Examples

**Purpose:** Demonstrate how the XML selection tools generalize to different data sources beyond Pearltrees.

---

## The Tools

### 1. select_xml_elements.awk
Generic XML element selector - works on **any** XML file.

### 2. filter_by_parent_tree.py
Filters by parent tree ID - **Pearltrees-specific**.

### 3. xml_to_prolog_facts.py
Transforms to facts - currently supports trees and pearls, **extensible** to other types.

---

## Generalization Examples

### Example 1: Product Catalog (e-commerce)

**Input:** Product catalog XML
```xml
<catalog>
  <product id="101">
    <name>Widget A</name>
    <price>29.99</price>
    <category>Tools</category>
  </product>
  <product id="102">
    <name>Widget B</name>
    <price>39.99</price>
    <category>Tools</category>
  </product>
</catalog>
```

**Usage:**
```bash
# Extract all products
awk -f scripts/utils/select_xml_elements.awk \
    -v tag="product" \
    catalog.xml

# Process products (custom transformer needed)
awk -f scripts/utils/select_xml_elements.awk \
    -v tag="product" \
    catalog.xml | \
python3 process_products.py > product_facts.pl
```

**What's reusable:**
- ✅ select_xml_elements.awk (no changes)
- ❌ filter_by_parent_tree.py (Pearltrees-specific)
- ⚠️ xml_to_prolog_facts.py (need new --element-type=product)

---

### Example 2: RSS Feed (news/blog)

**Input:** RSS feed XML
```xml
<rss version="2.0">
  <channel>
    <title>Tech News</title>
    <item>
      <title>Article 1</title>
      <link>http://example.com/1</link>
      <pubDate>Mon, 01 Jan 2025 12:00:00 GMT</pubDate>
    </item>
  </channel>
</rss>
```

**Usage:**
```bash
# Extract all items
awk -f scripts/utils/select_xml_elements.awk \
    -v tag="item" \
    feed.rss

# Extract channel info
awk -f scripts/utils/select_xml_elements.awk \
    -v tag="channel" \
    feed.rss
```

**What's reusable:**
- ✅ select_xml_elements.awk (no changes)
- ❌ filter_by_parent_tree.py (not applicable)
- ⚠️ xml_to_prolog_facts.py (need --element-type=rss_item)

---

### Example 3: SVG Graphics (technical docs)

**Input:** SVG file
```xml
<svg xmlns="http://www.w3.org/2000/svg">
  <circle cx="50" cy="50" r="40" fill="red" />
  <rect x="10" y="10" width="100" height="50" />
  <circle cx="150" cy="50" r="30" fill="blue" />
</svg>
```

**Usage:**
```bash
# Extract all circles
awk -f scripts/utils/select_xml_elements.awk \
    -v tag="circle" \
    diagram.svg

# Extract all rectangles
awk -f scripts/utils/select_xml_elements.awk \
    -v tag="rect" \
    diagram.svg
```

**What's reusable:**
- ✅ select_xml_elements.awk (no changes)
- ❌ filter_by_parent_tree.py (not applicable)
- ⚠️ xml_to_prolog_facts.py (need --element-type=svg_shape)

---

### Example 4: Maven POM (software projects)

**Input:** Maven pom.xml
```xml
<project>
  <dependencies>
    <dependency>
      <groupId>org.example</groupId>
      <artifactId>library-a</artifactId>
      <version>1.0.0</version>
    </dependency>
  </dependencies>
</project>
```

**Usage:**
```bash
# Extract all dependencies
awk -f scripts/utils/select_xml_elements.awk \
    -v tag="dependency" \
    pom.xml
```

**What's reusable:**
- ✅ select_xml_elements.awk (no changes)
- ❌ filter_by_parent_tree.py (not applicable)
- ⚠️ xml_to_prolog_facts.py (need --element-type=maven_dep)

---

## Reusability Matrix

| Component | Pearltrees | Product Catalog | RSS Feed | SVG | Maven POM |
|-----------|------------|-----------------|----------|-----|-----------|
| **select_xml_elements.awk** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **filter_by_parent_tree.py** | ✅ | ❌ | ❌ | ❌ | ❌ |
| **xml_to_prolog_facts.py** | ✅ | ⚠️ | ⚠️ | ⚠️ | ⚠️ |

**Legend:**
- ✅ Works without modification
- ⚠️ Needs extension (new element type)
- ❌ Domain-specific, not applicable

---

## Pattern: Extending xml_to_prolog_facts.py

To support new element types, add a new extraction function:

```python
def extract_product_facts(xml_chunk, debug=False):
    """Extract product facts from XML chunk."""
    # Extract ID
    product_id = extract_attribute(xml_chunk, 'product', 'id')

    # Extract fields
    name = extract_text_content(xml_chunk, 'name')
    price = extract_text_content(xml_chunk, 'price')
    category = extract_text_content(xml_chunk, 'category')

    # Emit fact
    print(f"product({product_id}, '{name}', {price}, '{category}').")

# Add to argument choices
parser.add_argument('--element-type',
                    choices=['tree', 'pearl', 'product', ...],
                    ...)

# Add to main processing
if args.element_type == 'product':
    extract_product_facts(chunk, debug=args.debug)
```

---

## Design Insight: Separation of Concerns

**Why this design is powerful:**

1. **Selection (awk)** - Purely structural, tag-based
   - Works on **any** XML
   - No domain knowledge required
   - Fastest possible implementation

2. **Filtering (Python)** - Domain-specific logic
   - Pearltrees: filter_by_parent_tree.py
   - E-commerce: filter_by_category.py
   - RSS: filter_by_date_range.py
   - Composable in pipelines

3. **Transformation (Python)** - Schema mapping
   - XML → Domain facts
   - Extensible via new element types
   - Reuses extraction utilities

**Result:** Maximum reusability with minimal code duplication.

---

## Future Extensions

### Generic Filters (Beyond Pearltrees)

```bash
# scripts/utils/filter_by_attribute.py
# Filter XML chunks by attribute value

awk -f select_xml_elements.awk -v tag="product" catalog.xml | \
    filter_by_attribute.py --attr=category --value=Tools
```

### Generic Transformers

```bash
# scripts/utils/xml_to_json.py
# Convert XML chunks to JSON

awk -f select_xml_elements.awk -v tag="product" catalog.xml | \
    xml_to_json.py
```

### Integration with Existing Tools

```bash
# Combine with existing XML processing from xml_examples.md

awk -f select_xml_elements.awk -v tag="product" large_catalog.xml | \
    python3 process_with_elementtree.py | \
    prolog
```

---

## Key Takeaway

**The `select_xml_elements.awk` tool is 100% reusable across domains.**

It's a general-purpose XML element extractor that works for:
- Pearltrees RDF
- Product catalogs
- RSS feeds
- SVG graphics
- Configuration files
- Any structured XML

The domain-specific intelligence lives in the **filters** and **transformers**, which are easy to add without modifying the core selector.

This is the **UnifyWeaver way**: composable, reusable, extensible.
