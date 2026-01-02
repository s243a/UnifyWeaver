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
    Export to GraphML format (yEd compatible).

    GraphML is supported by:
    - yEd (free graph editor)
    - Gephi (network analysis)
    - Cytoscape (bioinformatics)
    - NetworkX (Python library)

    Uses yEd-specific extensions for node positions and labels.
    """
    # Color palette
    fill_colors = [
        "#E8E8E8", "#FFD6D6", "#FFE8D6", "#FFFFD6",
        "#D6FFD6", "#D6FFFF", "#D6D6FF", "#FFD6FF"
    ]

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<graphml xmlns="http://graphml.graphdrawing.org/xmlns"',
        '         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"',
        '         xmlns:y="http://www.yworks.com/xml/graphml"',
        '         xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns',
        '           http://www.yworks.com/xml/schema/graphml/1.1/ygraphml.xsd">',
        '',
        '  <!-- yEd node graphics key -->',
        '  <key for="node" id="d0" yfiles.type="nodegraphics"/>',
        '  <key for="edge" id="d1" yfiles.type="edgegraphics"/>',
        '  <!-- Custom attributes -->',
        '  <key id="url" for="node" attr.name="url" attr.type="string"/>',
        '',
        f'  <graph id="{escape_xml_attr(title)}" edgedefault="directed">',
    ]

    # Write nodes with yEd ShapeNode format
    for node in nodes:
        fill = fill_colors[node.id % len(fill_colors)]
        # Estimate node dimensions
        width = max(80, len(node.text) * 7 + 20)
        height = 30

        lines.append(f'    <node id="n{node.id}">')
        if node.url:
            lines.append(f'      <data key="url">{escape_xml_attr(node.url)}</data>')
        lines.append('      <data key="d0">')
        lines.append('        <y:ShapeNode>')
        lines.append(f'          <y:Geometry x="{node.x - width/2:.1f}" y="{node.y - height/2:.1f}" width="{width:.1f}" height="{height:.1f}"/>')
        lines.append(f'          <y:Fill color="{fill}" transparent="false"/>')
        lines.append('          <y:BorderStyle type="line" width="1.0" color="#000000"/>')
        lines.append(f'          <y:NodeLabel>{escape_xml_attr(node.text)}</y:NodeLabel>')
        lines.append('          <y:Shape type="roundrectangle"/>')
        lines.append('        </y:ShapeNode>')
        lines.append('      </data>')
        lines.append('    </node>')

    # Write edges with yEd PolyLineEdge format
    edge_id = 0
    for node in nodes:
        if node.parent_id >= 0:
            lines.append(f'    <edge id="e{edge_id}" source="n{node.parent_id}" target="n{node.id}">')
            lines.append('      <data key="d1">')
            lines.append('        <y:PolyLineEdge>')
            lines.append('          <y:LineStyle type="line" width="1.0" color="#000000"/>')
            lines.append('          <y:Arrows source="none" target="standard"/>')
            lines.append('        </y:PolyLineEdge>')
            lines.append('      </data>')
            lines.append('    </edge>')
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
    Format matches VUE 3.x native .vue files.
    """
    import time
    timestamp = int(time.time() * 1000)

    # Color palette (similar to SimpleMind)
    fill_colors = [
        "#E8E8E8", "#FFD6D6", "#FFE8D6", "#FFFFD6",
        "#D6FFD6", "#D6FFFF", "#D6D6FF", "#FFD6FF"
    ]
    stroke_colors = [
        "#AAAAAA", "#FF9999", "#FFBB99", "#DDDD77",
        "#99DD99", "#99DDDD", "#9999DD", "#DD99DD"
    ]

    lines = [
        f'<!-- Tufts VUE concept-map ({escape_xml_attr(title)}.vue) -->',
        '<!-- Exported from UnifyWeaver -->',
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<LW-MAP xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"',
        f'    xsi:noNamespaceSchemaLocation="none" ID="0" label="{escape_xml_attr(title)}"',
        f'    created="{timestamp}" x="0.0" y="0.0" width="1.4E-45"',
        '    height="1.4E-45" strokeWidth="0.0" autoSized="false">',
        '    <fillColor>#FFFFFF</fillColor>',
        '    <strokeColor>#404040</strokeColor>',
        '    <textColor>#000000</textColor>',
        '    <font>SansSerif-plain-14</font>',
    ]

    # Write nodes as <child> elements
    for node in nodes:
        fill = fill_colors[node.id % len(fill_colors)]
        stroke = stroke_colors[node.id % len(stroke_colors)]

        # Estimate node dimensions based on text
        width = max(80, len(node.text) * 7 + 20)
        height = max(28, 27)

        lines.append(f'    <child ID="{node.id + 100}" label="{escape_xml_attr(node.text)}" layerID="1"')
        lines.append(f'        created="{timestamp}" x="{node.x:.1f}" y="{node.y:.1f}"')
        lines.append(f'        width="{width:.1f}" height="{height:.1f}" strokeWidth="1.0"')
        lines.append('        autoSized="true" xsi:type="node">')

        # URL resource
        if node.url:
            lines.append(f'        <resource referenceCreated="0"')
            lines.append(f'            spec="{escape_xml_attr(node.url)}"')
            lines.append('            type="2" xsi:type="URLResource">')
            lines.append(f'            <property key="URL" value="{escape_xml_attr(node.url)}"/>')
            lines.append('        </resource>')

        lines.append(f'        <fillColor>{fill}</fillColor>')
        lines.append(f'        <strokeColor>{stroke}</strokeColor>')
        lines.append('        <textColor>#000000</textColor>')
        lines.append('        <font>Arial-plain-12</font>')
        lines.append('        <shape arcwidth="20.0" archeight="20.0" xsi:type="roundRect"/>')
        lines.append('    </child>')

    # Write links (edges) as <child> elements with xsi:type="link"
    link_id = 1000
    for node in nodes:
        if node.parent_id >= 0 and node.parent_id in node_map:
            parent = node_map[node.parent_id]
            # Calculate link endpoints
            x1, y1 = parent.x, parent.y
            x2, y2 = node.x, node.y
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            width = abs(x2 - x1)
            height = abs(y2 - y1)

            lines.append(f'    <child ID="{link_id}" layerID="1" created="{timestamp}"')
            lines.append(f'        x="{mid_x:.1f}" y="{mid_y:.1f}"')
            lines.append(f'        width="{max(width, 1):.1f}" height="{max(height, 1):.1f}" strokeWidth="1.0"')
            lines.append('        autoSized="false" controlCount="0" arrowState="2" xsi:type="link">')
            lines.append('        <strokeColor>#404040</strokeColor>')
            lines.append('        <textColor>#404040</textColor>')
            lines.append('        <font>Arial-plain-11</font>')
            lines.append(f'        <point1 x="{x1:.1f}" y="{y1:.1f}"/>')
            lines.append(f'        <point2 x="{x2:.1f}" y="{y2:.1f}"/>')
            lines.append(f'        <ID1 xsi:type="node">{node.parent_id + 100}</ID1>')
            lines.append(f'        <ID2 xsi:type="node">{node.id + 100}</ID2>')
            lines.append('    </child>')
            link_id += 1

    # Add layer definition
    lines.append(f'    <layer ID="1" label="Layer 1" created="{timestamp}" x="0.0"')
    lines.append('        y="0.0" width="1.4E-45" height="1.4E-45" strokeWidth="0.0" autoSized="false"/>')

    # Footer elements
    lines.append('    <userZoom>1.0</userZoom>')
    lines.append('    <userOrigin x="-14.0" y="-14.0"/>')
    lines.append('    <presentationBackground>#202020</presentationBackground>')
    lines.append('    <modelVersion>6</modelVersion>')
    lines.append('</LW-MAP>')

    return '\n'.join(lines)


# =============================================================================
# FreeMind Export
# =============================================================================

def export_freemind(nodes: List[Node], node_map: Dict[int, Node], title: str) -> str:
    """
    Export to FreeMind (.mm) format.

    FreeMind/Freeplane is free, open-source desktop mind mapping software.
    Also compatible with Mind42 import.
    """
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<map version="1.0.1">',
        '<!-- Exported from UnifyWeaver -->',
    ]

    def write_node(node: Node, indent: int = 0):
        """Recursively write FreeMind node elements."""
        spaces = '  ' * indent
        # FreeMind uses POSITION="left" or "right" for first-level children
        position = ''
        if node.parent_id == 0:  # First-level child
            position = ' POSITION="right"' if node.x > node_map[0].x else ' POSITION="left"'

        lines.append(f'{spaces}<node TEXT="{escape_xml_attr(node.text)}"{position}>')

        # Add link if URL present
        if node.url:
            lines.append(f'{spaces}  <attribute NAME="url" VALUE="{escape_xml_attr(node.url)}"/>')
            lines.append(f'{spaces}  <arrowlink DESTINATION="{escape_xml_attr(node.url)}"/>')

        # Recurse for children
        for child in node.children:
            write_node(child, indent + 1)

        lines.append(f'{spaces}</node>')

    # Find root node(s)
    roots = [n for n in nodes if n.parent_id == -1]
    for root in roots:
        write_node(root)

    lines.append('</map>')
    return '\n'.join(lines)


# =============================================================================
# Main
# =============================================================================

EXPORTERS = {
    '.opml': ('OPML', export_opml),
    '.graphml': ('GraphML', export_graphml),
    '.vue': ('VUE', export_vue),
    '.mm': ('FreeMind', export_freemind),
}


def main():
    parser = argparse.ArgumentParser(
        description="Export SimpleMind mind maps to various formats"
    )
    parser.add_argument('input', type=Path, help='Input .smmx file')
    parser.add_argument('output', type=Path, help='Output file (.opml, .graphml, .vue)')
    parser.add_argument('--format', choices=['opml', 'graphml', 'vue', 'mm'],
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
