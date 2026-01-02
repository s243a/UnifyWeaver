#!/usr/bin/env python3
"""
Render SimpleMind mind maps (.smmx) to images (SVG, PNG).

Supports two edge rendering modes:
- Straight lines: Simple, matches crossing detection model
- Cubic Bezier: Matches SimpleMind's curve rendering

Usage:
    python3 scripts/render_mindmap.py input.smmx output.svg
    python3 scripts/render_mindmap.py input.smmx output.png --curves
    python3 scripts/render_mindmap.py input.smmx output.svg --straight --width 2000
"""

import argparse
import math
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple


@dataclass
class Node:
    """A node parsed from SimpleMind XML."""
    id: int
    parent_id: int
    x: float
    y: float
    text: str
    palette: int = 1
    font_scale: float = 1.0
    font_bold: bool = False
    url: str = ""
    borderstyle: str = ""  # SimpleMind borderstyle (sbsHalfRound, sbsEllipse, etc.)
    children: List['Node'] = field(default_factory=list)


# Map SimpleMind borderstyle values to our style names
SIMPLEMIND_BORDERSTYLES = {
    'sbsHalfRound': 'half-round',
    'sbsNone': 'half-round',      # None = default = half-round
    'sbsEllipse': 'ellipse',
    'sbsRectangle': 'rectangle',
    'sbsDiamond': 'diamond',
    '': 'half-round',             # No style = default
}


# SimpleMind palette colors (approximate)
PALETTE_COLORS = {
    1: "#E8E8E8",  # Gray (root)
    2: "#FFD6D6",  # Light red
    3: "#FFE8D6",  # Light orange
    4: "#FFFFD6",  # Light yellow
    5: "#D6FFD6",  # Light green
    6: "#D6FFFF",  # Light cyan
    7: "#D6D6FF",  # Light blue
    8: "#FFD6FF",  # Light purple
}

PALETTE_BORDERS = {
    1: "#AAAAAA",
    2: "#FF9999",
    3: "#FFBB99",
    4: "#DDDD77",
    5: "#99DD99",
    6: "#99DDDD",
    7: "#9999DD",
    8: "#DD99DD",
}

# Node shape styles (matching SimpleMind borderstyle values)
NODE_STYLES = {
    'half-round': 'half-round',   # sbsHalfRound: rx = ry = min(w,h)/6 (default)
    'ellipse': 'ellipse',         # sbsEllipse: full oval rx = w/2, ry = h/2
    'rectangle': 'rectangle',     # sbsRectangle: sharp corners rx = ry = 0
    'diamond': 'diamond',         # sbsDiamond: 45° rotated square
    # Aliases
    'oval': 'ellipse',
    'square': 'rectangle',
}


def parse_smmx(smmx_path: Path) -> Tuple[List[Node], Dict[int, Node]]:
    """Parse a .smmx file and return list of nodes and lookup dict."""
    with zipfile.ZipFile(smmx_path, 'r') as zf:
        xml_content = zf.read('document/mindmap.xml').decode('utf-8')

    # Handle BOM if present
    if xml_content.startswith('\ufeff'):
        xml_content = xml_content[1:]

    root = ET.fromstring(xml_content)
    topics = root.find('.//topics')

    nodes = []
    node_map = {}

    for topic in topics.findall('topic'):
        node_id = int(topic.get('id', 0))
        parent_id = int(topic.get('parent', -1))
        x = float(topic.get('x', 0))
        y = float(topic.get('y', 0))
        text = topic.get('text', '').replace('\\N', '\n')
        palette = int(topic.get('palette', 1))

        # Parse style (font and borderstyle)
        font_scale = 1.0
        font_bold = False
        borderstyle = ""
        style = topic.find('style')
        if style is not None:
            font = style.find('font')
            if font is not None:
                font_scale = float(font.get('scale', 1.0))
                font_bold = font.get('bold', 'False').lower() == 'true'
            borderstyle = style.get('borderstyle', '')

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
            palette=palette,
            font_scale=font_scale,
            font_bold=font_bold,
            url=url,
            borderstyle=borderstyle
        )
        nodes.append(node)
        node_map[node_id] = node

    # Build parent-child relationships
    for node in nodes:
        if node.parent_id >= 0 and node.parent_id in node_map:
            node_map[node.parent_id].children.append(node)

    return nodes, node_map


def calculate_bounds(nodes: List[Node], padding: float = 100) -> Tuple[float, float, float, float]:
    """Calculate bounding box of all nodes with padding."""
    if not nodes:
        return 0, 0, 800, 600

    min_x = min(n.x for n in nodes)
    max_x = max(n.x for n in nodes)
    min_y = min(n.y for n in nodes)
    max_y = max(n.y for n in nodes)

    # Account for node sizes (estimate based on text length and scale)
    for node in nodes:
        lines = node.text.split('\n')
        max_line_len = max(len(line) for line in lines) if lines else 10
        width = max_line_len * 7 * node.font_scale + 20
        height = len(lines) * 16 * node.font_scale + 16

        min_x = min(min_x, node.x - width/2)
        max_x = max(max_x, node.x + width/2)
        min_y = min(min_y, node.y - height/2)
        max_y = max(max_y, node.y + height/2)

    return (min_x - padding, min_y - padding,
            max_x + padding, max_y + padding)


def escape_xml(text: str) -> str:
    """Escape text for XML."""
    return (text
            .replace('&', '&amp;')
            .replace('<', '&lt;')
            .replace('>', '&gt;')
            .replace('"', '&quot;'))


def get_node_dimensions(node: Node) -> Tuple[float, float]:
    """Calculate node width and height based on text."""
    lines = node.text.split('\n')
    max_line_len = max(len(line) for line in lines) if lines else 5

    # Base dimensions
    char_width = 7 * node.font_scale
    line_height = 16 * node.font_scale

    width = max(60, max_line_len * char_width + 24)
    height = max(28, len(lines) * line_height + 12)

    return width, height


def cubic_bezier_control_points(x1: float, y1: float, x2: float, y2: float
                                ) -> Tuple[float, float, float, float]:
    """
    Calculate cubic Bezier control points for SimpleMind-style curves.

    The curve:
    - Starts radially from the parent node center
    - Arrives at child node TANGENT to nearest reference line
      (0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°)
    - Second derivative bends away from these reference lines
    """
    dx = x2 - x1
    dy = y2 - y1
    dist = math.sqrt(dx*dx + dy*dy)

    if dist < 1:
        return x1, y1, x2, y2

    # Angle from parent to child
    angle = math.atan2(dy, dx)

    # Control point distance (fraction of total distance)
    cp_dist = dist * 0.4

    # First control point: radial from parent
    cp1_x = x1 + cp_dist * math.cos(angle)
    cp1_y = y1 + cp_dist * math.sin(angle)

    # Second control point: tangent to nearest reference line at child
    # Reference lines every 45°: 0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°
    # Find nearest reference angle
    ref_angles = [i * math.pi / 4 for i in range(8)]  # 0, 45, 90, ... degrees in radians

    # Normalize angle to [0, 2π)
    norm_angle = angle % (2 * math.pi)

    # Find nearest reference angle
    nearest_ref = min(ref_angles, key=lambda r: min(abs(norm_angle - r),
                                                      abs(norm_angle - r - 2*math.pi),
                                                      abs(norm_angle - r + 2*math.pi)))

    # Control point comes from the direction of the reference angle
    # (curve arrives tangent to this reference line)
    cp2_x = x2 - cp_dist * math.cos(nearest_ref)
    cp2_y = y2 - cp_dist * math.sin(nearest_ref)

    return cp1_x, cp1_y, cp2_x, cp2_y


def get_corner_radii(width: float, height: float, style: str) -> Tuple[float, float]:
    """Calculate corner radii based on node style.

    Returns (rx, ry) for rect elements, or (0, 0) for special shapes like diamond.
    """
    # Resolve aliases
    style = NODE_STYLES.get(style, style)

    if style == 'ellipse':
        return width / 2, height / 2
    elif style == 'half-round':
        # SimpleMind default: ~1/6 of shortest dimension
        r = min(width, height) / 6
        return r, r
    elif style == 'rectangle':
        return 0, 0
    elif style == 'diamond':
        # Diamond uses polygon, not rect - return 0,0 as placeholder
        return 0, 0
    else:
        # Default to half-round
        r = min(width, height) / 6
        return r, r


def render_svg(nodes: List[Node], node_map: Dict[int, Node],
               use_curves: bool = False,
               width: Optional[int] = None,
               height: Optional[int] = None,
               node_style: str = 'half-round') -> str:
    """Render nodes to SVG string.

    Args:
        nodes: List of nodes to render
        node_map: Dict mapping node ID to node
        use_curves: Use cubic Bezier curves instead of straight lines
        width: Output width in pixels
        height: Output height in pixels
        node_style: Node shape style (half-round, oval, rounded, square)
    """

    # Calculate bounds
    min_x, min_y, max_x, max_y = calculate_bounds(nodes)
    content_width = max_x - min_x
    content_height = max_y - min_y

    # Determine output size
    if width and not height:
        scale = width / content_width
        height = int(content_height * scale)
    elif height and not width:
        scale = height / content_height
        width = int(content_width * scale)
    elif not width and not height:
        width = int(content_width)
        height = int(content_height)

    # SVG header
    svg_parts = [
        f'<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" ',
        f'     width="{width}" height="{height}" ',
        f'     viewBox="{min_x} {min_y} {content_width} {content_height}">',
        '',
        '  <!-- Background -->',
        f'  <rect x="{min_x}" y="{min_y}" width="{content_width}" height="{content_height}" fill="white"/>',
        '',
        '  <!-- Edges -->',
        '  <g id="edges" fill="none" stroke-width="2">',
    ]

    # Render edges
    for node in nodes:
        if node.parent_id >= 0 and node.parent_id in node_map:
            parent = node_map[node.parent_id]
            color = PALETTE_BORDERS.get(node.palette, "#999999")

            if use_curves:
                # Cubic Bezier curve
                cp1_x, cp1_y, cp2_x, cp2_y = cubic_bezier_control_points(
                    parent.x, parent.y, node.x, node.y
                )
                svg_parts.append(
                    f'    <path d="M {parent.x} {parent.y} '
                    f'C {cp1_x} {cp1_y}, {cp2_x} {cp2_y}, {node.x} {node.y}" '
                    f'stroke="{color}"/>'
                )
            else:
                # Straight line
                svg_parts.append(
                    f'    <line x1="{parent.x}" y1="{parent.y}" '
                    f'x2="{node.x}" y2="{node.y}" stroke="{color}"/>'
                )

    svg_parts.extend([
        '  </g>',
        '',
        '  <!-- Nodes -->',
        '  <g id="nodes">',
    ])

    # Render nodes (sorted by ID to ensure consistent layering)
    for node in sorted(nodes, key=lambda n: n.id):
        width_n, height_n = get_node_dimensions(node)

        fill = PALETTE_COLORS.get(node.palette, "#E8E8E8")
        stroke = PALETTE_BORDERS.get(node.palette, "#AAAAAA")

        # Determine node style: use node's borderstyle if set, else fallback to argument
        if node.borderstyle and node.borderstyle in SIMPLEMIND_BORDERSTYLES:
            effective_style = SIMPLEMIND_BORDERSTYLES[node.borderstyle]
        else:
            effective_style = NODE_STYLES.get(node_style, node_style)

        svg_parts.append(f'    <g id="node-{node.id}">')

        if effective_style == 'diamond':
            # Diamond: polygon with 4 points
            points = (f"{node.x},{node.y - height_n/2} "  # top
                      f"{node.x + width_n/2},{node.y} "   # right
                      f"{node.x},{node.y + height_n/2} "  # bottom
                      f"{node.x - width_n/2},{node.y}")   # left
            svg_parts.append(
                f'      <polygon points="{points}" '
                f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>'
            )
        else:
            # Rectangle with configurable corner radii
            rx, ry = get_corner_radii(width_n, height_n, effective_style)
            svg_parts.append(
                f'      <rect x="{node.x - width_n/2}" y="{node.y - height_n/2}" '
                f'width="{width_n}" height="{height_n}" rx="{rx}" ry="{ry}" '
                f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>'
            )

        # Node text
        font_size = 12 * node.font_scale
        font_weight = "bold" if node.font_bold else "normal"
        lines = node.text.split('\n')

        # Calculate text starting position (centered)
        line_height = font_size * 1.2
        total_text_height = len(lines) * line_height
        text_y = node.y - total_text_height/2 + font_size * 0.85

        for i, line in enumerate(lines):
            svg_parts.append(
                f'      <text x="{node.x}" y="{text_y + i * line_height}" '
                f'text-anchor="middle" font-family="Arial, sans-serif" '
                f'font-size="{font_size}" font-weight="{font_weight}" '
                f'fill="#333333">{escape_xml(line)}</text>'
            )

        svg_parts.append('    </g>')

    svg_parts.extend([
        '  </g>',
        '</svg>',
    ])

    return '\n'.join(svg_parts)


def svg_to_png(svg_content: str, output_path: Path, scale: float = 1.0):
    """Convert SVG to PNG using available library."""
    try:
        import cairosvg
        cairosvg.svg2png(bytestring=svg_content.encode('utf-8'),
                        write_to=str(output_path),
                        scale=scale)
        return True
    except ImportError:
        pass

    try:
        from PIL import Image
        import io
        # Try svglib + reportlab
        from svglib.svglib import svg2rlg
        from reportlab.graphics import renderPM

        drawing = svg2rlg(io.StringIO(svg_content))
        renderPM.drawToFile(drawing, str(output_path), fmt="PNG")
        return True
    except ImportError:
        pass

    print("Warning: No PNG conversion library available.")
    print("Install cairosvg: pip install cairosvg")
    print("Or svglib: pip install svglib reportlab")
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Render SimpleMind mind maps to images"
    )
    parser.add_argument('input', type=Path, help='Input .smmx file')
    parser.add_argument('output', type=Path, help='Output file (SVG or PNG)')
    parser.add_argument('--curves', action='store_true',
                        help='Use cubic Bezier curves (SimpleMind-style)')
    parser.add_argument('--straight', action='store_true',
                        help='Use straight lines (default)')
    parser.add_argument('--width', type=int, help='Output width in pixels')
    parser.add_argument('--height', type=int, help='Output height in pixels')
    parser.add_argument('--scale', type=float, default=1.0,
                        help='Scale factor for PNG output')
    parser.add_argument('--node-style',
                        choices=['half-round', 'ellipse', 'rectangle', 'diamond', 'oval', 'square'],
                        default='half-round',
                        help='Node shape style: half-round (default), ellipse/oval, rectangle/square, diamond')

    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        return 1

    # Parse input
    print(f"Parsing {args.input}...")
    nodes, node_map = parse_smmx(args.input)
    print(f"Found {len(nodes)} nodes")

    # Render to SVG
    use_curves = args.curves and not args.straight
    print(f"Rendering with {'cubic curves' if use_curves else 'straight lines'}, "
          f"node style: {args.node_style}...")
    svg_content = render_svg(nodes, node_map, use_curves=use_curves,
                             width=args.width, height=args.height,
                             node_style=args.node_style)

    # Write output
    output_ext = args.output.suffix.lower()

    if output_ext == '.svg':
        args.output.write_text(svg_content)
        print(f"Written: {args.output}")
    elif output_ext == '.png':
        # Write SVG first, then convert
        svg_path = args.output.with_suffix('.svg')
        svg_path.write_text(svg_content)

        if svg_to_png(svg_content, args.output, scale=args.scale):
            print(f"Written: {args.output}")
            svg_path.unlink()  # Remove temp SVG
        else:
            print(f"SVG saved as: {svg_path}")
            print("Convert manually or install cairosvg")
    else:
        print(f"Unknown output format: {output_ext}")
        print("Supported: .svg, .png")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
