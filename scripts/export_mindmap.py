#!/usr/bin/env python3
"""
Export SimpleMind mind maps (.smmx) to various formats.

Supported formats:
- OPML: Outline format for outline editors, RSS readers
- GraphML: Graph format for yEd, Gephi, Cytoscape
- VUE: Tufts Visual Understanding Environment

Usage:
    python3 scripts/export_mindmap.py input.smmx output.opml
    python3 scripts/export_mindmap.py input.smmx output.graphml
    python3 scripts/export_mindmap.py input.smmx output.vue
"""

import argparse
import zipfile
import xml.etree.ElementTree as ET
from xml.dom import minidom
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple


@dataclass
class Node:
    """A node parsed from SimpleMind XML."""
    id: int
    parent_id: int
    x: float
    y: float
    text: str
    url: str = ""
    children: List['Node'] = field(default_factory=list)


def parse_smmx(smmx_path: Path) -> Tuple[List[Node], Dict[int, Node], str]:
    """Parse a .smmx file and return nodes, lookup dict, and title."""
    with zipfile.ZipFile(smmx_path, 'r') as zf:
        xml_content = zf.read('document/mindmap.xml').decode('utf-8')

    if xml_content.startswith('\ufeff'):
        xml_content = xml_content[1:]

    root = ET.fromstring(xml_content)

    # Get title from meta
    title_elem = root.find('.//meta/title')
    title = title_elem.get('text', 'Mind Map') if title_elem is not None else 'Mind Map'

    topics = root.find('.//topics')
    nodes = []
    node_map = {}

    for topic in topics.findall('topic'):
        node_id = int(topic.get('id', 0))
        parent_id = int(topic.get('parent', -1))
        x = float(topic.get('x', 0))
        y = float(topic.get('y', 0))
        text = topic.get('text', '').replace('\\N', ' ')

        # Parse URL
        url = ""
        link = topic.find('link')
        if link is not None:
            url = link.get('urllink', '')

        node = Node(
            id=node_id,
            parent_id=parent_id,
            x=x,
            y=y,
            text=text,
            url=url
        )
        nodes.append(node)
        node_map[node_id] = node

    # Build parent-child relationships
    for node in nodes:
        if node.parent_id >= 0 and node.parent_id in node_map:
            node_map[node.parent_id].children.append(node)

    return nodes, node_map, title


def escape_xml_attr(text: str) -> str:
    """Escape text for XML attributes."""
    return (text
            .replace('&', '&amp;')
            .replace('<', '&lt;')
            .replace('>', '&gt;')
            .replace('"', '&quot;')
            .replace("'", '&apos;'))


# =============================================================================
# OPML Export
# =============================================================================

def export_opml(nodes: List[Node], node_map: Dict[int, Node], title: str) -> str:
    """
    Export to OPML (Outline Processor Markup Language).

    OPML is a simple hierarchical outline format supported by:
    - Outline editors (OmniOutliner, Dynalist, Workflowy)
    - RSS readers
    - Note-taking apps
    """
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<opml version="2.0">',
        '  <head>',
        f'    <title>{escape_xml_attr(title)}</title>',
        '    <expansionState>0</expansionState>',
        '  </head>',
        '  <body>',
    ]

    def write_outline(node: Node, indent: int = 4):
        """Recursively write outline elements."""
        spaces = ' ' * indent
        attrs = f'text="{escape_xml_attr(node.text)}"'
        if node.url:
            attrs += f' url="{escape_xml_attr(node.url)}"'

        if node.children:
            lines.append(f'{spaces}<outline {attrs}>')
            for child in node.children:
                write_outline(child, indent + 2)
            lines.append(f'{spaces}</outline>')
        else:
            lines.append(f'{spaces}<outline {attrs}/>')

    # Find root node(s)
    roots = [n for n in nodes if n.parent_id == -1]
    for root in roots:
        write_outline(root)

    lines.extend([
        '  </body>',
        '</opml>',
    ])

    return '\n'.join(lines)


# =============================================================================
# GraphML Export
# =============================================================================

def export_graphml(nodes: List[Node], node_map: Dict[int, Node], title: str) -> str:
    """
    Export to GraphML format.

    GraphML is supported by:
    - yEd (free graph editor)
    - Gephi (network analysis)
    - Cytoscape (bioinformatics)
    - NetworkX (Python library)

    Includes node positions for layout preservation.
    """
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<graphml xmlns="http://graphml.graphdrawing.org/xmlns"',
        '         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"',
        '         xmlns:y="http://www.yworks.com/xml/graphml"',
        '         xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns',
        '           http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">',
        '',
        '  <!-- Node attributes -->',
        '  <key id="label" for="node" attr.name="label" attr.type="string"/>',
        '  <key id="url" for="node" attr.name="url" attr.type="string"/>',
        '  <key id="x" for="node" attr.name="x" attr.type="double"/>',
        '  <key id="y" for="node" attr.name="y" attr.type="double"/>',
        '',
        f'  <graph id="{escape_xml_attr(title)}" edgedefault="directed">',
    ]

    # Write nodes
    for node in nodes:
        lines.append(f'    <node id="n{node.id}">')
        lines.append(f'      <data key="label">{escape_xml_attr(node.text)}</data>')
        if node.url:
            lines.append(f'      <data key="url">{escape_xml_attr(node.url)}</data>')
        lines.append(f'      <data key="x">{node.x}</data>')
        lines.append(f'      <data key="y">{node.y}</data>')
        lines.append('    </node>')

    # Write edges
    edge_id = 0
    for node in nodes:
        if node.parent_id >= 0:
            lines.append(f'    <edge id="e{edge_id}" source="n{node.parent_id}" target="n{node.id}"/>')
            edge_id += 1

    lines.extend([
        '  </graph>',
        '</graphml>',
    ])

    return '\n'.join(lines)


# =============================================================================
# VUE Export
# =============================================================================

def export_vue(nodes: List[Node], node_map: Dict[int, Node], title: str) -> str:
    """
    Export to VUE (Visual Understanding Environment) format.

    VUE is a free mind mapping tool from Tufts University.
    Format is XML with specific namespace and structure.
    """
    # VUE uses a specific XML structure
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<LW-MAP xmlns="urn:schemas-tufts-edu:VUE"',
        '        xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"',
        '        xsi:schemaLocation="urn:schemas-tufts-edu:VUE VUE.xsd">',
        f'  <property key="label" value="{escape_xml_attr(title)}"/>',
        '',
    ]

    # Color palette (similar to SimpleMind)
    colors = [
        "#E8E8E8", "#FFD6D6", "#FFE8D6", "#FFFFD6",
        "#D6FFD6", "#D6FFFF", "#D6D6FF", "#FFD6FF"
    ]

    # Write nodes
    for node in nodes:
        color = colors[node.id % len(colors)]

        # Estimate node dimensions based on text
        width = max(80, len(node.text) * 7 + 20)
        height = 30

        lines.append(f'  <node ID="{node.id}"')
        lines.append(f'        label="{escape_xml_attr(node.text)}"')
        lines.append(f'        x="{node.x - width/2:.1f}" y="{node.y - height/2:.1f}"')
        lines.append(f'        width="{width}" height="{height}">')

        # Shape
        lines.append('    <shape xsi:type="roundRect"/>')

        # Fill color
        lines.append(f'    <fillColor>{color}</fillColor>')

        # URL resource
        if node.url:
            lines.append(f'    <resource referenceCreated="true"')
            lines.append(f'              spec="{escape_xml_attr(node.url)}"/>')

        lines.append('  </node>')

    lines.append('')

    # Write links (edges)
    link_id = 1000
    for node in nodes:
        if node.parent_id >= 0:
            lines.append(f'  <link ID="{link_id}"')
            lines.append(f'        ID1="{node.parent_id}" ID2="{node.id}"/>')
            link_id += 1

    lines.append('</LW-MAP>')

    return '\n'.join(lines)


# =============================================================================
# Main
# =============================================================================

EXPORTERS = {
    '.opml': ('OPML', export_opml),
    '.graphml': ('GraphML', export_graphml),
    '.vue': ('VUE', export_vue),
}


def main():
    parser = argparse.ArgumentParser(
        description="Export SimpleMind mind maps to various formats"
    )
    parser.add_argument('input', type=Path, help='Input .smmx file')
    parser.add_argument('output', type=Path, help='Output file (.opml, .graphml, .vue)')
    parser.add_argument('--format', choices=['opml', 'graphml', 'vue'],
                        help='Override output format (default: detect from extension)')

    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        return 1

    # Determine output format
    if args.format:
        ext = '.' + args.format
    else:
        ext = args.output.suffix.lower()

    if ext not in EXPORTERS:
        print(f"Error: Unknown output format: {ext}")
        print(f"Supported formats: {', '.join(EXPORTERS.keys())}")
        return 1

    format_name, exporter = EXPORTERS[ext]

    # Parse input
    print(f"Parsing {args.input}...")
    nodes, node_map, title = parse_smmx(args.input)
    print(f"Found {len(nodes)} nodes")

    # Export
    print(f"Exporting to {format_name}...")
    content = exporter(nodes, node_map, title)

    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content, encoding='utf-8')
    print(f"Written: {args.output}")

    return 0


if __name__ == '__main__':
    exit(main())
