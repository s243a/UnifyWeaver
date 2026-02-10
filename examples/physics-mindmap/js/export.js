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

function exportFreeMind(tree, titles, coords2d) {
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

function exportVUE(tree, titles, coords2d) {
    const nested = buildNestedTree(tree, titles);
    let xml = '<?xml version="1.0" encoding="UTF-8"?>\n';
    xml += '<LW-MAP xmlns="urn:schemas-tufts-edu:LW-MAP" ID="0">\n';

    const nodeGuids = {};
    function assignGuids(node) {
        nodeGuids[node.id] = generateGuid();
        node.children.forEach(assignGuids);
    }
    assignGuids(nested);

    // Write nodes
    function writeNode(node) {
        const guid = nodeGuids[node.id];
        const x = (coords2d[node.id][0] * 200) + 500;
        const y = (coords2d[node.id][1] * 200) + 400;
        const shape = node.depth === 0 ? 'roundRect' : 'ellipse';
        const wikiUrl = titleToWikipediaUrl(node.title);
        xml += `<node ID="${guid}" label="${escapeXml(node.title)}" x="${x.toFixed(1)}" y="${y.toFixed(1)}" shape="${shape}">\n`;
        xml += `  <resource referenceURL="${escapeXml(wikiUrl)}"/>\n`;
        xml += `</node>\n`;
        node.children.forEach(writeNode);
    }
    writeNode(nested);

    // Write links
    function writeLinks(node) {
        for (const child of node.children) {
            xml += `<link ID1="${nodeGuids[node.id]}" ID2="${nodeGuids[child.id]}"/>\n`;
            writeLinks(child);
        }
    }
    writeLinks(nested);

    xml += '</LW-MAP>';

    const rootTitle = titles[tree.rootId].replace(/[^a-zA-Z0-9]/g, '_');
    downloadFile(xml, `${rootTitle}_mindmap.vue`, 'application/xml');
}

async function exportSMMX(tree, titles, coords2d) {
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

    const depthColors = ['palette1', 'palette2', 'palette3', 'palette4', 'palette5', 'palette6', 'palette7', 'palette8'];

    let mindmapXml = '<?xml version="1.0" encoding="utf-8"?>\n';
    mindmapXml += '<simplemind-mindmaps doc-version="3" generator="PhysicsMindmapBuilder">\n';
    mindmapXml += '<mindmap>\n';

    function writeTopic(node) {
        const guid = generateGuid();
        const x = (coords2d[node.id][0] * 150).toFixed(1);
        const y = (coords2d[node.id][1] * 150).toFixed(1);
        const palette = depthColors[node.depth % depthColors.length];
        const wikiUrl = titleToWikipediaUrl(node.title);

        mindmapXml += `<topic guid="${guid}" text="${escapeXml(node.title)}" x="${x}" y="${y}" palette="${palette}"`;
        if (node.depth === 0) {
            mindmapXml += ' central-topic="true" layout="radial"';
        }
        mindmapXml += `>\n`;
        mindmapXml += `  <link url="${escapeXml(wikiUrl)}" />\n`;

        for (const child of node.children) {
            writeTopic(child);
        }
        mindmapXml += '</topic>\n';
    }

    writeTopic(nested);
    mindmapXml += '</mindmap>\n</simplemind-mindmaps>';

    zip.file('document/mindmap.xml', mindmapXml);

    const blob = await zip.generateAsync({ type: 'blob' });
    const rootTitle = titles[tree.rootId].replace(/[^a-zA-Z0-9]/g, '_');
    downloadBlob(blob, `${rootTitle}_mindmap.smmx`);
}

function exportJSON(tree, titles) {
    const nested = buildNestedTree(tree, titles);
    const json = JSON.stringify(nested, null, 2);
    const rootTitle = titles[tree.rootId].replace(/[^a-zA-Z0-9]/g, '_');
    downloadFile(json, `${rootTitle}_tree.json`, 'application/json');
}

/**
 * Main export dispatcher.
 */
async function exportMindmap(format, tree, titles, coords2d) {
    switch (format) {
        case 'freemind': return exportFreeMind(tree, titles, coords2d);
        case 'simplemind': return await exportSMMX(tree, titles, coords2d);
        case 'opml': return exportOPML(tree, titles);
        case 'mermaid': return exportMermaid(tree, titles);
        case 'vue': return exportVUE(tree, titles, coords2d);
        case 'json': return exportJSON(tree, titles);
        default: alert('Unknown format: ' + format);
    }
}
