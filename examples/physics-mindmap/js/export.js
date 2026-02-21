/**
 * Mindmap export functions.
 * Adapted from tools/density_explorer/web/index.html.
 */

function escapeXml(str) {
    return str.replace(/&/g, '&amp;').replace(/</g, '&lt;')
              .replace(/>/g, '&gt;').replace(/"/g, '&quot;')
              .replace(/'/g, '&apos;');
}

function generateGuid() {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, c => {
        const r = Math.random() * 16 | 0;
        return (c === 'x' ? r : (r & 0x3 | 0x8)).toString(16);
    });
}

function titleToWikipediaUrl(title) {
    return 'https://en.wikipedia.org/wiki/' + encodeURIComponent(title.replace(/ /g, '_'));
}

function downloadFile(content, filename, mimeType = 'text/plain') {
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
}

function downloadBlob(blob, filename) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
}

/**
 * Build a nested tree structure from flat nodes/edges.
 */
function buildNestedTree(tree, titles) {
    const childMap = {};
    for (const e of tree.edges) {
        if (!childMap[e.source]) childMap[e.source] = [];
        childMap[e.source].push({ id: e.target, weight: e.weight });
    }
    // Sort children by weight
    for (const k of Object.keys(childMap)) {
        childMap[k].sort((a, b) => a.weight - b.weight);
    }

    function buildNode(id, depth) {
        const children = (childMap[id] || []).map(c => buildNode(c.id, depth + 1));
        return { id, title: titles[id], url: titleToWikipediaUrl(titles[id]), depth, children };
    }

    return buildNode(tree.rootId, 0);
}

// --- Export Formats ---

function exportFreeMind(tree, titles) {
    const nested = buildNestedTree(tree, titles);
    let xml = '<map version="1.0.1">\n';

    function writeNode(node, position) {
        const posAttr = position ? ` POSITION="${position}"` : '';
        const wikiUrl = titleToWikipediaUrl(node.title);
        xml += `<node TEXT="${escapeXml(node.title)}" LINK="${escapeXml(wikiUrl)}"${posAttr}>\n`;
        node.children.forEach((child, i) => {
            const pos = node.depth === 0 ? (i % 2 === 0 ? 'right' : 'left') : '';
            writeNode(child, pos);
        });
        xml += '</node>\n';
    }

    writeNode(nested, '');
    xml += '</map>';

    const rootTitle = titles[tree.rootId].replace(/[^a-zA-Z0-9]/g, '_');
    downloadFile(xml, `${rootTitle}_mindmap.mm`, 'application/xml');
}

function exportOPML(tree, titles) {
    const nested = buildNestedTree(tree, titles);
    let xml = '<?xml version="1.0" encoding="UTF-8"?>\n';
    xml += '<opml version="2.0">\n<head><title>' + escapeXml(titles[tree.rootId]) + '</title></head>\n<body>\n';

    function writeOutline(node) {
        const wikiUrl = titleToWikipediaUrl(node.title);
        if (node.children.length === 0) {
            xml += `<outline text="${escapeXml(node.title)}" type="link" url="${escapeXml(wikiUrl)}" />\n`;
        } else {
            xml += `<outline text="${escapeXml(node.title)}" type="link" url="${escapeXml(wikiUrl)}">\n`;
            node.children.forEach(writeOutline);
            xml += '</outline>\n';
        }
    }

    writeOutline(nested);
    xml += '</body>\n</opml>';

    const rootTitle = titles[tree.rootId].replace(/[^a-zA-Z0-9]/g, '_');
    downloadFile(xml, `${rootTitle}_mindmap.opml`, 'application/xml');
}

function exportMermaid(tree, titles) {
    const nested = buildNestedTree(tree, titles);
    let md = '```mermaid\nflowchart TD\n';

    function sanitizeId(title) {
        return title.replace(/[^a-zA-Z0-9]/g, '_').substring(0, 30);
    }

    function writeNode(node) {
        const id = sanitizeId(node.title);
        // Different shapes by depth
        let shape;
        if (node.depth === 0) shape = `[${escapeXml(node.title)}]`;
        else if (node.depth === 1) shape = `{{${escapeXml(node.title)}}}`;
        else shape = `(${escapeXml(node.title)})`;

        for (const child of node.children) {
            const childId = sanitizeId(child.title);
            let childShape;
            if (child.depth === 1) childShape = `{{${escapeXml(child.title)}}}`;
            else childShape = `(${escapeXml(child.title)})`;
            md += `    ${id}${shape} --> ${childId}${childShape}\n`;
            writeNode(child);
        }
    }

    // Add click links for each node
    function writeLinks(node) {
        const id = sanitizeId(node.title);
        const wikiUrl = titleToWikipediaUrl(node.title);
        md += `    click ${id} "${wikiUrl}" _blank\n`;
        for (const child of node.children) {
            writeLinks(child);
        }
    }

    writeNode(nested);
    writeLinks(nested);
    md += '```\n';

    const rootTitle = titles[tree.rootId].replace(/[^a-zA-Z0-9]/g, '_');
    downloadFile(md, `${rootTitle}_mindmap.md`, 'text/markdown');
}

function exportVUE(tree, titles, positionMap) {
    const nested = buildNestedTree(tree, titles);
    const timestamp = Date.now();

    const fillColors = [
        '#E8E8E8', '#FFD6D6', '#FFE8D6', '#FFFFD6',
        '#D6FFD6', '#D6FFFF', '#D6D6FF', '#FFD6FF'
    ];
    const strokeColors = [
        '#AAAAAA', '#FF9999', '#FFBB99', '#DDDD77',
        '#99DD99', '#99DDDD', '#9999DD', '#DD99DD'
    ];

    const mapLabel = escapeXml(titles[tree.rootId]);
    const filename = titles[tree.rootId].replace(/[^a-zA-Z0-9]/g, '_') + '_mindmap.vue';
    const dateStr = new Date().toString();

    // VUE requires these header comments for proper deserialization
    let xml = `<!-- Tufts VUE 3.3.0 concept-map (${filename}) -->\n`;
    xml += '<!-- Tufts VUE: http://vue.tufts.edu/ -->\n';
    xml += '<!-- Do Not Remove: VUE mapping @version(1.1) jar:file:/tufts/vue/resources/lw_mapping_1_1.xml -->\n';
    xml += `<!-- Do Not Remove: Saved date ${dateStr} -->\n`;
    xml += '<?xml version="1.0" encoding="US-ASCII"?>\n';
    xml += '<LW-MAP xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\n';
    xml += `    xsi:noNamespaceSchemaLocation="none" ID="0" label="${mapLabel}"\n`;
    xml += `    created="${timestamp}" x="0.0" y="0.0" width="1.4E-45"\n`;
    xml += '    height="1.4E-45" strokeWidth="0.0" autoSized="false">\n';
    xml += '    <fillColor>#FFFFFF</fillColor>\n';
    xml += '    <strokeColor>#404040</strokeColor>\n';
    xml += '    <textColor>#000000</textColor>\n';
    xml += '    <font>SansSerif-plain-14</font>\n';

    // Write nodes as <child> with xsi:type="node"
    function writeNode(node) {
        const nodeId = node.id + 100;
        const pos = positionMap[node.id];
        const x = pos[0].toFixed(1);
        const y = pos[1].toFixed(1);
        const w = Math.max(80, node.title.length * 7 + 20);
        const h = 28;
        const fill = fillColors[node.depth % fillColors.length];
        const stroke = strokeColors[node.depth % strokeColors.length];
        const wikiUrl = titleToWikipediaUrl(node.title);

        xml += `    <child ID="${nodeId}" label="${escapeXml(node.title)}" layerID="1"\n`;
        xml += `        created="${timestamp}" x="${x}" y="${y}"\n`;
        xml += `        width="${w.toFixed(1)}" height="${h.toFixed(1)}" strokeWidth="1.0"\n`;
        xml += '        autoSized="true" xsi:type="node">\n';
        xml += `        <resource referenceCreated="0"\n`;
        xml += `            spec="${escapeXml(wikiUrl)}"\n`;
        xml += '            type="2" xsi:type="URLResource">\n';
        xml += `            <property key="URL" value="${escapeXml(wikiUrl)}"/>\n`;
        xml += '        </resource>\n';
        xml += `        <fillColor>${fill}</fillColor>\n`;
        xml += `        <strokeColor>${stroke}</strokeColor>\n`;
        xml += '        <textColor>#000000</textColor>\n';
        xml += '        <font>Arial-plain-12</font>\n';
        xml += '        <shape arcwidth="20.0" archeight="20.0" xsi:type="roundRect"/>\n';
        xml += '    </child>\n';
        node.children.forEach(writeNode);
    }
    writeNode(nested);

    // Write links as <child> with xsi:type="link"
    let linkId = 1000;
    function writeLinks(node) {
        for (const child of node.children) {
            const parentId = node.id + 100;
            const childId = child.id + 100;
            const p1 = positionMap[node.id];
            const p2 = positionMap[child.id];
            const midX = ((p1[0] + p2[0]) / 2).toFixed(1);
            const midY = ((p1[1] + p2[1]) / 2).toFixed(1);
            const w = Math.max(Math.abs(p2[0] - p1[0]), 1).toFixed(1);
            const h = Math.max(Math.abs(p2[1] - p1[1]), 1).toFixed(1);

            xml += `    <child ID="${linkId++}" layerID="1" created="${timestamp}"\n`;
            xml += `        x="${midX}" y="${midY}"\n`;
            xml += `        width="${w}" height="${h}" strokeWidth="1.0"\n`;
            xml += '        autoSized="false" controlCount="0" arrowState="2" xsi:type="link">\n';
            xml += '        <strokeColor>#404040</strokeColor>\n';
            xml += '        <textColor>#404040</textColor>\n';
            xml += '        <font>Arial-plain-11</font>\n';
            xml += `        <point1 x="${p1[0].toFixed(1)}" y="${p1[1].toFixed(1)}"/>\n`;
            xml += `        <point2 x="${p2[0].toFixed(1)}" y="${p2[1].toFixed(1)}"/>\n`;
            xml += `        <ID1 xsi:type="node">${parentId}</ID1>\n`;
            xml += `        <ID2 xsi:type="node">${childId}</ID2>\n`;
            xml += '    </child>\n';
            writeLinks(child);
        }
    }
    writeLinks(nested);

    // Layer definition
    xml += `    <layer ID="1" label="Layer 1" created="${timestamp}" x="0.0"\n`;
    xml += '        y="0.0" width="1.4E-45" height="1.4E-45" strokeWidth="0.0" autoSized="false"/>\n';
    // Footer
    xml += '    <userZoom>1.0</userZoom>\n';
    xml += '    <userOrigin x="-14.0" y="-14.0"/>\n';
    xml += '    <presentationBackground>#202020</presentationBackground>\n';
    xml += '    <modelVersion>6</modelVersion>\n';
    xml += '</LW-MAP>';

    const rootTitle = titles[tree.rootId].replace(/[^a-zA-Z0-9]/g, '_');
    downloadFile(xml, `${rootTitle}_mindmap.vue`, 'application/xml');
}

async function exportSMMX(tree, titles, positionMap) {
    // Load JSZip on demand
    if (typeof JSZip === 'undefined') {
        await new Promise((resolve, reject) => {
            const script = document.createElement('script');
            script.src = 'https://cdnjs.cloudflare.com/ajax/libs/jszip/3.10.1/jszip.min.js';
            script.onload = resolve;
            script.onerror = reject;
            document.head.appendChild(script);
        });
    }

    const nested = buildNestedTree(tree, titles);
    const zip = new JSZip();

    const mapGuid = generateGuid();

    let mindmapXml = '<?xml version="1.0" encoding="utf-8"?>\n';
    mindmapXml += '<simplemind-mindmaps doc-version="3" generator="PhysicsMindmapBuilder">\n';
    mindmapXml += '<mindmap>\n';
    mindmapXml += `<meta><guid>${mapGuid}</guid></meta>\n`;
    mindmapXml += '<topics>\n';

    let topicId = 0;
    const idMap = {};

    function writeTopic(node, parentTopicId) {
        const myId = topicId++;
        idMap[node.id] = myId;
        const guid = generateGuid();
        const pos = positionMap[node.id];
        const x = pos[0].toFixed(1);
        const y = pos[1].toFixed(1);
        const wikiUrl = titleToWikipediaUrl(node.title);

        mindmapXml += `<topic id="${myId}" guid="${guid}" text="${escapeXml(node.title)}" x="${x}" y="${y}"`;
        if (parentTopicId !== null) {
            mindmapXml += ` parent="${parentTopicId}"`;
        }
        if (node.depth === 0) {
            mindmapXml += ' central-topic="true"';
        }
        mindmapXml += '>\n';
        mindmapXml += `  <link urllink="${escapeXml(wikiUrl)}" />\n`;
        mindmapXml += '</topic>\n';

        for (const child of node.children) {
            writeTopic(child, myId);
        }
    }

    writeTopic(nested, null);
    mindmapXml += '</topics>\n';
    mindmapXml += '</mindmap>\n</simplemind-mindmaps>';

    zip.file('document/mindmap.xml', mindmapXml);

    const blob = await zip.generateAsync({ type: 'blob' });
    const rootTitle = titles[tree.rootId].replace(/[^a-zA-Z0-9]/g, '_');
    downloadBlob(blob, `${rootTitle}_mindmap.smmx`);
}

function exportGraphML(tree, titles, positionMap) {
    const nested = buildNestedTree(tree, titles);
    const fillColors = [
        '#E8E8E8', '#FFD6D6', '#FFE8D6', '#FFFFD6',
        '#D6FFD6', '#D6FFFF', '#D6D6FF', '#FFD6FF'
    ];

    let xml = '<?xml version="1.0" encoding="UTF-8"?>\n';
    xml += '<graphml xmlns="http://graphml.graphdrawing.org/xmlns"\n';
    xml += '         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\n';
    xml += '         xmlns:y="http://www.yworks.com/xml/graphml"\n';
    xml += '         xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns\n';
    xml += '           http://www.yworks.com/xml/schema/graphml/1.1/ygraphml.xsd">\n';
    xml += '  <key for="node" id="d0" yfiles.type="nodegraphics"/>\n';
    xml += '  <key for="edge" id="d1" yfiles.type="edgegraphics"/>\n';
    xml += '  <key id="url" for="node" attr.name="url" attr.type="string"/>\n';
    xml += '  <graph id="mindmap" edgedefault="directed">\n';

    let edgeId = 0;

    function writeNode(node) {
        const fill = fillColors[node.depth % fillColors.length];
        const w = Math.max(80, node.title.length * 7 + 20);
        const h = 30;
        const pos = positionMap[node.id];
        const x = pos[0];
        const y = pos[1];
        const wikiUrl = titleToWikipediaUrl(node.title);

        xml += `    <node id="n${node.id}">\n`;
        xml += `      <data key="url">${escapeXml(wikiUrl)}</data>\n`;
        xml += '      <data key="d0">\n';
        xml += '        <y:ShapeNode>\n';
        xml += `          <y:Geometry x="${(x - w/2).toFixed(1)}" y="${(y - h/2).toFixed(1)}" width="${w.toFixed(1)}" height="${h.toFixed(1)}"/>\n`;
        xml += `          <y:Fill color="${fill}" transparent="false"/>\n`;
        xml += '          <y:BorderStyle type="line" width="1.0" color="#000000"/>\n';
        xml += `          <y:NodeLabel>${escapeXml(node.title)}</y:NodeLabel>\n`;
        xml += '          <y:Shape type="roundrectangle"/>\n';
        xml += '        </y:ShapeNode>\n';
        xml += '      </data>\n';
        xml += '    </node>\n';

        node.children.forEach(writeNode);
    }

    function writeEdges(node) {
        for (const child of node.children) {
            xml += `    <edge id="e${edgeId++}" source="n${node.id}" target="n${child.id}">\n`;
            xml += '      <data key="d1">\n';
            xml += '        <y:PolyLineEdge>\n';
            xml += '          <y:LineStyle type="line" width="1.0" color="#000000"/>\n';
            xml += '          <y:Arrows source="none" target="standard"/>\n';
            xml += '        </y:PolyLineEdge>\n';
            xml += '      </data>\n';
            xml += '    </edge>\n';
            writeEdges(child);
        }
    }

    writeNode(nested);
    writeEdges(nested);
    xml += '  </graph>\n</graphml>';

    const rootTitle = titles[tree.rootId].replace(/[^a-zA-Z0-9]/g, '_');
    downloadFile(xml, `${rootTitle}_mindmap.graphml`, 'application/xml');
}

function exportJSON(tree, titles) {
    const nested = buildNestedTree(tree, titles);
    const json = JSON.stringify(nested, null, 2);
    const rootTitle = titles[tree.rootId].replace(/[^a-zA-Z0-9]/g, '_');
    downloadFile(json, `${rootTitle}_tree.json`, 'application/json');
}

// --- Layout Algorithms ---

/**
 * Radial layout: root at center, children in concentric rings.
 * Returns positionMap {nodeId: [x, y]} in pixel coordinates.
 */
function computeRadialLayout(tree, titles) {
    const nested = buildNestedTree(tree, titles);
    const positionMap = {};
    const radiusStep = 300;

    // Count subtree sizes for angular wedge allocation
    function subtreeSize(node) {
        if (node.children.length === 0) return 1;
        let s = 0;
        for (const c of node.children) s += subtreeSize(c);
        return s;
    }

    function layout(node, angleStart, angleEnd, depth) {
        if (depth === 0) {
            positionMap[node.id] = [1000, 1000];
        } else {
            const angle = (angleStart + angleEnd) / 2;
            const r = depth * radiusStep;
            positionMap[node.id] = [
                1000 + r * Math.cos(angle),
                1000 + r * Math.sin(angle)
            ];
        }
        if (node.children.length === 0) return;
        const totalSize = node.children.reduce((s, c) => s + subtreeSize(c), 0);
        let currentAngle = angleStart;
        for (const child of node.children) {
            const wedge = (subtreeSize(child) / totalSize) * (angleEnd - angleStart);
            layout(child, currentAngle, currentAngle + wedge, depth + 1);
            currentAngle += wedge;
        }
    }

    layout(nested, 0, 2 * Math.PI, 0);
    return positionMap;
}

/**
 * Top-down hierarchy: root at top center, levels below.
 * Returns positionMap {nodeId: [x, y]} in pixel coordinates.
 */
function computeTopDownLayout(tree, titles) {
    const nested = buildNestedTree(tree, titles);
    const positionMap = {};
    const levelHeight = 200;

    // Count leaf descendants for horizontal spacing
    function leafCount(node) {
        if (node.children.length === 0) return 1;
        let s = 0;
        for (const c of node.children) s += leafCount(c);
        return s;
    }

    const totalLeaves = leafCount(nested);
    const leafSpacing = 160;
    const totalWidth = totalLeaves * leafSpacing;

    function layout(node, xStart, xEnd, depth) {
        const x = (xStart + xEnd) / 2;
        const y = 100 + depth * levelHeight;
        positionMap[node.id] = [x, y];

        if (node.children.length === 0) return;
        const total = node.children.reduce((s, c) => s + leafCount(c), 0);
        let currentX = xStart;
        for (const child of node.children) {
            const w = (leafCount(child) / total) * (xEnd - xStart);
            layout(child, currentX, currentX + w, depth + 1);
            currentX += w;
        }
    }

    layout(nested, 0, totalWidth, 0);
    return positionMap;
}

/**
 * Semantic layout: uses the 2D embedding projection coordinates.
 * Returns positionMap {nodeId: [x, y]} in pixel coordinates.
 */
function computeSemanticLayout(tree, coords2d) {
    const positionMap = {};
    for (const node of tree.nodes) {
        const id = node.id !== undefined ? node.id : node;
        positionMap[id] = [
            (coords2d[id][0] * 2500) + 1000,
            (coords2d[id][1] * 2500) + 800
        ];
    }
    return positionMap;
}

/** Formats that support spatial layout. */
const SPATIAL_FORMATS = new Set(['vue', 'graphml', 'simplemind']);

/**
 * Main export dispatcher.
 */
async function exportMindmap(format, tree, titles, positionMap) {
    switch (format) {
        case 'freemind': return exportFreeMind(tree, titles);
        case 'simplemind': return await exportSMMX(tree, titles, positionMap);
        case 'opml': return exportOPML(tree, titles);
        case 'mermaid': return exportMermaid(tree, titles);
        case 'vue': return exportVUE(tree, titles, positionMap);
        case 'graphml': return exportGraphML(tree, titles, positionMap);
        case 'json': return exportJSON(tree, titles);
        default: alert('Unknown format: ' + format);
    }
}
