#!/usr/bin/env python3
"""
Link Pearltrees items to mindmap nodes.

This tool enriches exported .smmx mindmaps by attaching links to Pearltrees
items based on:
1. Exact URL matching - if a node has a URL that matches a PagePearl's external URL
2. Exact title matching to trees/pearls
3. Semantic similarity matching to trees (with configurable threshold)
4. When multiple matches exist, choose the best fit by semantic similarity

Usage:
    python3 scripts/mindmap/link_pearltrees.py \
        --mindmap output/physics_export.smmx \
        --pearls reports/pearltrees_targets_full_pearls.jsonl \
        --trees reports/pearltrees_targets_s243a.jsonl \
        --api-cache .local/data/pearltrees_api/trees \
        --embeddings datasets/pearltrees_combined_2026-01-02_all_fixed_embeddings.npz \
        --output output/physics_linked.smmx \
        --threshold 0.7
"""

import argparse
import json
import zipfile
import tempfile
import shutil
from pathlib import Path
from xml.etree import ElementTree as ET
import numpy as np
from typing import Dict, List, Optional, Tuple
import re


def load_jsonl(path: Path) -> List[dict]:
    """Load items from JSONL file."""
    items = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return items


def load_url_index_from_db(db_path: Path) -> Dict[str, List[dict]]:
    """
    Load URL index from children_index.db SQLite database.

    Returns a dict mapping external URLs to a LIST of PagePearl info dicts,
    since the same URL can exist in multiple Pearltrees locations.
    """
    import sqlite3

    url_index = {}

    if not db_path.exists():
        return url_index

    try:
        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()

        cur.execute('''
            SELECT uri, title, external_url, parent_tree_uri
            FROM children
            WHERE external_url IS NOT NULL AND external_url != ''
        ''')

        for uri, title, external_url, parent_tree_uri in cur.fetchall():
            if external_url not in url_index:
                url_index[external_url] = []
            url_index[external_url].append({
                'external_url': external_url,
                'pearl_uri': uri,
                'title': title or '',
                'parent_tree_uri': parent_tree_uri or ''
            })

        conn.close()
    except Exception as e:
        print(f"Warning: Could not load from database: {e}")

    return url_index


def load_api_cache(cache_dir: Path) -> Dict[str, List[dict]]:
    """
    Load API cache and build URL index.

    Returns a dict mapping external URLs to a LIST of PagePearl info dicts,
    since the same URL can exist in multiple Pearltrees locations.
    """
    url_index = {}

    if not cache_dir.exists():
        return url_index

    for json_file in cache_dir.glob("*.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Handle nested structure: api_response.tree.pearls
            api_resp = data.get('api_response', data)
            tree_data = api_resp.get('tree', api_resp)
            tree_id = data.get('tree_id') or tree_data.get('id')
            tree_uri = tree_data.get('uri', '')
            pearls = tree_data.get('pearls', [])

            for pearl in pearls:
                content_type = pearl.get('contentType')
                if content_type != 1:  # Not a PagePearl
                    continue

                url_obj = pearl.get('url', {})
                if isinstance(url_obj, dict):
                    external_url = url_obj.get('url')
                else:
                    external_url = url_obj

                if not external_url:
                    continue

                pearl_id = pearl.get('pearlId') or pearl.get('id')
                title = pearl.get('title', '')

                # Build Pearltrees URI for this pearl
                # Format: https://www.pearltrees.com/t/.../id{tree_id}#item{pearl_id}
                pearl_uri = f"https://www.pearltrees.com/t/pearl/id{tree_id}#item{pearl_id}" if pearl_id else None

                match_info = {
                    'external_url': external_url,
                    'pearl_uri': pearl_uri,
                    'tree_id': str(tree_id),
                    'pearl_id': pearl_id,
                    'title': title,
                    'parent_tree_uri': tree_uri
                }

                if external_url not in url_index:
                    url_index[external_url] = []
                url_index[external_url].append(match_info)

        except (json.JSONDecodeError, KeyError, TypeError):
            continue

    return url_index


def build_uri_index(pearls: List[dict], trees: List[dict]) -> Dict[str, dict]:
    """
    Build index of URIs to Pearltrees items.

    For PagePearls, the key is the pearl_uri.
    For Trees, the key is the uri.
    """
    index = {}

    # Index trees by URI
    for tree in trees:
        uri = tree.get('uri')
        if uri:
            index[uri] = {
                'type': 'tree',
                'uri': uri,
                'title': tree.get('raw_title', ''),
                'tree_id': tree.get('tree_id'),
                'data': tree
            }

    # Index pearls by pearl_uri
    for pearl in pearls:
        pearl_uri = pearl.get('pearl_uri')
        if pearl_uri:
            index[pearl_uri] = {
                'type': 'pearl',
                'uri': pearl_uri,
                'title': pearl.get('raw_title', ''),
                'parent_tree_uri': pearl.get('parent_tree_uri'),
                'data': pearl
            }

    return index


def build_title_index(items: List[dict]) -> Dict[str, List[dict]]:
    """Build index from normalized titles to items (for fuzzy matching)."""
    index = {}
    for item in items:
        title = item.get('raw_title', '')
        if title:
            # Normalize title
            normalized = normalize_title(title)
            if normalized not in index:
                index[normalized] = []
            index[normalized].append(item)
    return index


def normalize_title(title: str) -> str:
    """Normalize title for comparison."""
    # Remove special characters, lowercase, collapse whitespace
    normalized = re.sub(r'[^\w\s]', '', title.lower())
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    return normalized


def load_embeddings(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load embeddings from NPZ file.

    Returns:
        embeddings: (N, D) array of embeddings
        titles: (N,) array of titles
        uris: (N,) array of URIs
    """
    data = np.load(path, allow_pickle=True)

    # Try different embedding key names
    for key in ['input_nomic', 'embeddings', 'output_nomic']:
        if key in data:
            embeddings = data[key]
            break
    else:
        raise ValueError(f"No embeddings found in {path}. Keys: {list(data.keys())}")

    titles = data.get('titles', np.array([]))
    uris = data.get('uris', np.array([]))

    return embeddings, titles, uris


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def select_best_location(
    query_text: str,
    matches: List[dict],
    embeddings: Optional[np.ndarray],
    embed_uris: Optional[np.ndarray],
    projection_engine: Optional['ProjectionEngine'] = None,
    verbose: bool = False
) -> Tuple[dict, float]:
    """
    Select the best Pearltrees location for a URL when multiple matches exist.

    Uses the projection model to find the best hierarchical fit. The projection
    maps the query into the same space as the target_text (materialized paths +
    structure lists), so comparison finds the best location in the hierarchy.

    Args:
        query_text: The mindmap node text (context for matching)
        matches: List of match dicts with 'pearl_uri' and 'parent_tree_uri'
        embeddings: (N, D) array of target embeddings (projected)
        embed_uris: (N,) array of URIs
        projection_engine: ProjectionEngine for embedding and projecting queries
        verbose: Print debug info

    Returns:
        Tuple of (best_match_dict, score)
    """
    if len(matches) == 1:
        return matches[0], 1.0

    # If no projection engine, return first match
    if projection_engine is None or embeddings is None or embed_uris is None:
        if verbose:
            print(f"    Multiple matches ({len(matches)}), no projection model - using first")
        return matches[0], 0.9

    # Embed and project the query into hierarchical space
    query_proj = projection_engine.embed_and_project(query_text)
    query_proj_norm = query_proj / (np.linalg.norm(query_proj) + 1e-8)

    uri_list = list(embed_uris)

    best_match = matches[0]
    best_score = -1.0

    for match in matches:
        # Try to find embedding for parent tree URI (the folder location)
        parent_uri = match.get('parent_tree_uri', '')

        if parent_uri and parent_uri in uri_list:
            try:
                idx = uri_list.index(parent_uri)
                target_emb = embeddings[idx]
                target_norm = target_emb / (np.linalg.norm(target_emb) + 1e-8)
                score = float(np.dot(query_proj_norm, target_norm))
                if verbose:
                    print(f"    Match: {parent_uri[:60]}... score={score:.3f}")
                if score > best_score:
                    best_score = score
                    best_match = match
            except (ValueError, IndexError):
                pass

        # Also try the pearl URI itself
        pearl_uri = match.get('pearl_uri', '')
        if pearl_uri and pearl_uri in uri_list:
            try:
                idx = uri_list.index(pearl_uri)
                target_emb = embeddings[idx]
                target_norm = target_emb / (np.linalg.norm(target_emb) + 1e-8)
                score = float(np.dot(query_proj_norm, target_norm))
                if score > best_score:
                    best_score = score
                    best_match = match
            except (ValueError, IndexError):
                pass

    if best_score < 0:
        # No embedding match found, use first
        if verbose:
            print(f"    Multiple matches ({len(matches)}), no URI in embeddings - using first")
        return matches[0], 0.9

    if verbose:
        print(f"    Selected best of {len(matches)} matches: score={best_score:.3f}")
    return best_match, best_score


def find_best_match(
    query_title: str,
    uri_index: Dict[str, dict],
    title_index: Dict[str, List[dict]],
    embeddings: Optional[np.ndarray],
    embed_titles: Optional[np.ndarray],
    embed_uris: Optional[np.ndarray],
    threshold: float = 0.7,
    projection_engine: Optional['ProjectionEngine'] = None
) -> Optional[dict]:
    """
    Find the best matching Pearltrees item for a mindmap node.

    Strategy:
    1. Try exact title match
    2. If multiple matches, use projection to find best hierarchical fit
    3. If no exact match but projection available, try semantic search
    """
    normalized_query = normalize_title(query_title)

    # Try exact title match
    if normalized_query in title_index:
        matches = title_index[normalized_query]

        if len(matches) == 1:
            # Single match - return it
            match = matches[0]
            return {
                'type': 'tree' if match.get('type') == 'Tree' else 'pearl',
                'uri': match.get('uri') or match.get('pearl_uri'),
                'title': match.get('raw_title', ''),
                'match_type': 'exact_title',
                'score': 1.0
            }

        # Multiple matches - use projection to find best hierarchical fit
        if embeddings is not None and projection_engine is not None:
            query_proj = projection_engine.embed_and_project(query_title)
            query_norm = query_proj / (np.linalg.norm(query_proj) + 1e-8)
            best_match = None
            best_score = -1

            for match in matches:
                uri = match.get('uri') or match.get('pearl_uri')
                if uri and embed_uris is not None:
                    # Find embedding for this URI
                    uri_list = list(embed_uris)
                    try:
                        idx = uri_list.index(uri)
                        target_emb = embeddings[idx]
                        target_norm = target_emb / (np.linalg.norm(target_emb) + 1e-8)
                        score = float(np.dot(query_norm, target_norm))
                        if score > best_score:
                            best_score = score
                            best_match = match
                    except ValueError:
                        continue

            if best_match and best_score >= threshold:
                return {
                    'type': 'tree' if best_match.get('type') == 'Tree' else 'pearl',
                    'uri': best_match.get('uri') or best_match.get('pearl_uri'),
                    'title': best_match.get('raw_title', ''),
                    'match_type': 'semantic_disambiguate',
                    'score': best_score
                }

        # No semantic disambiguation - return first match
        match = matches[0]
        return {
            'type': 'tree' if match.get('type') == 'Tree' else 'pearl',
            'uri': match.get('uri') or match.get('pearl_uri'),
            'title': match.get('raw_title', ''),
            'match_type': 'first_of_multiple',
            'score': 0.9
        }

    # No exact match - try semantic search using projection
    if embeddings is not None and projection_engine is not None and embed_uris is not None:
        query_proj = projection_engine.embed_and_project(query_title)

        # Compute similarities to all target embeddings (in projected space)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings_norm = embeddings / (norms + 1e-8)
        query_norm = query_proj / (np.linalg.norm(query_proj) + 1e-8)
        similarities = embeddings_norm @ query_norm

        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]

        if best_score >= threshold:
            uri = str(embed_uris[best_idx])
            title = str(embed_titles[best_idx]) if embed_titles is not None and len(embed_titles) > best_idx else ''

            return {
                'type': 'tree',  # Most matches will be trees
                'uri': uri,
                'title': title,
                'match_type': 'semantic_search',
                'score': float(best_score)
            }

    return None


def parse_smmx(path: Path) -> Tuple[ET.Element, dict, bytes]:
    """
    Parse an .smmx file and return the XML tree.

    Returns:
        root: XML root element
        metadata: dict with parsing info
        raw_xml: original XML bytes for reference
    """
    with zipfile.ZipFile(path, 'r') as zf:
        raw_xml = zf.read('document/mindmap.xml')

    xml_content = raw_xml
    # Handle BOM
    if xml_content.startswith(b'\xef\xbb\xbf'):
        xml_content = xml_content[3:]

    root = ET.fromstring(xml_content)

    metadata = {
        'source': str(path),
        'format_version': root.get('doc-version', 'unknown')
    }

    return root, metadata, raw_xml


def write_smmx_from_raw(raw_xml: bytes, link_updates: Dict[str, str], output_path: Path, source_path: Path):
    """
    Write modified XML back to .smmx file using raw XML with regex replacements.

    This preserves the original XML formatting exactly.

    Args:
        raw_xml: Original XML content as bytes
        link_updates: Dict mapping topic id -> new urllink value
        output_path: Where to write the output
        source_path: Original .smmx file (for other files in the archive)
    """
    import re

    # Decode XML
    xml_str = raw_xml.decode('utf-8-sig')  # Handles BOM

    # For each topic that needs updating, replace or add the urllink
    for topic_id, new_url in link_updates.items():
        # Escape special regex chars in URL
        escaped_url = new_url.replace('&', '&amp;').replace("'", '&apos;').replace('"', '&quot;')

        # Pattern to find the topic with this id
        # Match: <topic id="X" ... > ... </topic> or <topic id="X" ... />
        topic_pattern = rf'(<topic\s+id="{topic_id}"[^>]*>)(.*?)(</topic>|/>)'

        def replace_link(match):
            opening = match.group(1)
            content = match.group(2)
            closing = match.group(3)

            if closing == '/>':
                # Self-closing topic - shouldn't happen but handle it
                return opening + content + closing

            # Check if there's already a <link> element
            link_pattern = r'<link\s+([^>]*)/?>'
            link_match = re.search(link_pattern, content)

            if link_match:
                # Update existing link's urllink attribute
                old_link = link_match.group(0)
                link_attrs = link_match.group(1)

                # Check if urllink exists
                if 'urllink=' in link_attrs:
                    # Replace the urllink value
                    new_link_attrs = re.sub(r'urllink="[^"]*"', f'urllink="{escaped_url}"', link_attrs)
                else:
                    # Add urllink attribute
                    new_link_attrs = link_attrs.rstrip() + f' urllink="{escaped_url}"'

                # Preserve self-closing or not
                if old_link.endswith('/>'):
                    new_link = f'<link {new_link_attrs}/>'
                else:
                    new_link = f'<link {new_link_attrs}>'

                new_content = content.replace(old_link, new_link, 1)
            else:
                # No link element - add one after the opening style/layout elements
                # Find a good insertion point (after </style> or </layout> or at start of content)
                insert_point = 0
                for tag in ['</style>', '</layout>']:
                    idx = content.find(tag)
                    if idx != -1:
                        insert_point = max(insert_point, idx + len(tag))

                new_link = f'\n<link urllink="{escaped_url}"/>'
                new_content = content[:insert_point] + new_link + content[insert_point:]

            return opening + new_content + closing

        xml_str = re.sub(topic_pattern, replace_link, xml_str, flags=re.DOTALL)

    # Create temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Extract original .smmx
        with zipfile.ZipFile(source_path, 'r') as zf:
            zf.extractall(tmpdir)

        # Write modified XML
        with open(tmpdir / 'document' / 'mindmap.xml', 'w', encoding='utf-8-sig') as f:
            f.write(xml_str)

        # Create new .smmx
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for file_path in tmpdir.rglob('*'):
                if file_path.is_file():
                    arcname = file_path.relative_to(tmpdir)
                    zf.write(file_path, arcname)


def write_smmx(root: ET.Element, output_path: Path, source_path: Path):
    """
    Write modified XML back to .smmx file.

    Preserves other files from the original .smmx.
    Post-processes output to match SimpleMind's expected XML format.
    """
    # Create temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Extract original .smmx
        with zipfile.ZipFile(source_path, 'r') as zf:
            zf.extractall(tmpdir)

        # Write modified XML
        xml_content = ET.tostring(root, encoding='unicode')

        # Post-process to match SimpleMind's format:
        # 1. Remove space before /> in self-closing tags
        xml_content = re.sub(r'\s+/>', '/>', xml_content)

        # 2. Escape apostrophes in attribute values (SimpleMind uses &apos;)
        # This is tricky - we need to only escape inside attribute values
        def escape_apos_in_attrs(match):
            # match.group(0) is the full attribute like: attr="value's here"
            attr_name = match.group(1)
            attr_value = match.group(2)
            escaped_value = attr_value.replace("'", "&apos;")
            return f'{attr_name}="{escaped_value}"'

        xml_content = re.sub(r'(\w+)="([^"]*)"', escape_apos_in_attrs, xml_content)

        # Add XML declaration and DOCTYPE
        xml_header = '<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE simplemind-mindmaps>\n'
        xml_content = xml_header + xml_content

        with open(tmpdir / 'document' / 'mindmap.xml', 'w', encoding='utf-8') as f:
            f.write('\ufeff' + xml_content)  # Add BOM

        # Create new .smmx
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for file_path in tmpdir.rglob('*'):
                if file_path.is_file():
                    arcname = file_path.relative_to(tmpdir)
                    zf.write(file_path, arcname)


def normalize_url(url: str) -> str:
    """Normalize URL for comparison (strip trailing slashes, lowercase domain)."""
    if not url:
        return ''
    url = url.strip()
    # Remove trailing slash
    if url.endswith('/'):
        url = url[:-1]
    # Lowercase the scheme and domain
    if '://' in url:
        scheme, rest = url.split('://', 1)
        if '/' in rest:
            domain, path = rest.split('/', 1)
            url = f"{scheme.lower()}://{domain.lower()}/{path}"
        else:
            url = f"{scheme.lower()}://{rest.lower()}"
    return url


def link_mindmap_nodes(
    root: ET.Element,
    uri_index: Dict[str, dict],
    title_index: Dict[str, List[dict]],
    url_index: Dict[str, List[dict]],
    embeddings: Optional[np.ndarray],
    embed_titles: Optional[np.ndarray],
    embed_uris: Optional[np.ndarray],
    threshold: float,
    projection_engine: Optional['ProjectionEngine'] = None,
    verbose: bool = False
) -> dict:
    """
    Add Pearltrees links to mindmap nodes.

    Matching priority:
    1. URL match - if node has urllink matching a PagePearl's external URL
       (when multiple matches exist, selects best by semantic similarity)
    2. Title match - exact normalized title match
    3. Semantic match - embedding similarity above threshold

    Returns statistics about the linking process.
    """
    stats = {
        'total_nodes': 0,
        'already_pearltrees': 0,
        'url_matched': 0,
        'title_matched': 0,
        'semantic_matched': 0,
        'no_match': 0,
        'matches': [],
        'link_updates': {},  # topic_id -> new_url for raw XML update
        'new_topics': []  # List of new topic elements to add
    }

    # Build normalized URL index for faster lookup (now stores lists)
    norm_url_index = {}
    for ext_url, match_list in url_index.items():
        norm_url_index[normalize_url(ext_url)] = match_list

    # Find all topic elements
    for mindmap in root.findall('.//mindmap'):
        topics_elem = mindmap.find('topics')
        for topic in mindmap.findall('.//topic'):
            stats['total_nodes'] += 1

            text = topic.get('text', '')
            topic_id = topic.get('id', '')

            # Check existing link
            link_elem = topic.find('link')
            existing_url = link_elem.get('urllink', '') if link_elem is not None else ''

            # Skip if already linked to Pearltrees
            if existing_url and 'pearltrees.com' in existing_url:
                stats['already_pearltrees'] += 1
                continue

            # PRIORITY 1: Check if existing URL matches a PagePearl
            if existing_url and norm_url_index:
                norm_existing = normalize_url(existing_url)
                if norm_existing in norm_url_index:
                    match_list = norm_url_index[norm_existing]
                    clean_text = text.replace('\\N', ' ').strip()

                    # If multiple matches, select the best one using semantic similarity
                    if len(match_list) > 1 and verbose:
                        print(f"  [url] '{clean_text[:30]}...' has {len(match_list)} matches - selecting best")

                    match_info, match_score = select_best_location(
                        clean_text,
                        match_list,
                        embeddings,
                        embed_uris,
                        projection_engine,
                        verbose=verbose
                    )
                    pearl_uri = match_info.get('pearl_uri')

                    if pearl_uri and topics_elem is not None:
                        # Add child node with Pearltrees link (PP = PearlPage)
                        # In SimpleMind, child topics are siblings in <topics>, not nested
                        child_id = f"{topic_id}_pp"
                        child_topic = ET.SubElement(topics_elem, 'topic')
                        child_topic.set('id', child_id)
                        child_topic.set('parent', topic_id)
                        # Get parent position and offset child slightly
                        parent_x = float(topic.get('x', '0'))
                        parent_y = float(topic.get('y', '0'))
                        child_topic.set('x', f"{parent_x + 50:.2f}")
                        child_topic.set('y', f"{parent_y + 30:.2f}")
                        child_topic.set('text', 'PP')  # PearlPage link
                        # Add no-border style for cleaner look
                        child_style = ET.SubElement(child_topic, 'style')
                        child_style.set('borderstyle', 'sbsNone')
                        child_link = ET.SubElement(child_topic, 'link')
                        child_link.set('urllink', pearl_uri)

                        stats['link_updates'][child_id] = {
                            'parent_id': topic_id,
                            'label': 'PP',
                            'url': pearl_uri
                        }

                        match_type = 'url_exact' if len(match_list) == 1 else 'url_best_of_multiple'
                        stats['url_matched'] += 1
                        stats['matches'].append({
                            'node_id': topic_id,
                            'node_text': clean_text,
                            'original_url': existing_url,
                            'match_uri': pearl_uri,
                            'match_title': match_info.get('title', ''),
                            'match_type': match_type,
                            'score': match_score,
                            'num_matches': len(match_list),
                            'child_label': 'PP'
                        })

                        if verbose:
                            print(f"  [{match_type}] '{clean_text[:40]}...' +PP -> {pearl_uri}")
                        continue

            # Skip empty nodes for title/semantic matching
            if not text or not text.strip():
                stats['no_match'] += 1
                continue

            # Clean up text (remove \N newline markers)
            clean_text = text.replace('\\N', ' ').strip()

            # PRIORITY 2 & 3: Title and semantic matching
            match = find_best_match(
                clean_text,
                uri_index,
                title_index,
                embeddings,
                embed_titles,
                embed_uris,
                threshold,
                projection_engine
            )

            if match:
                # Determine label based on match type
                # PT = exact title match to Pearltree (including disambiguated)
                # PT? = semantic/fuzzy match (no exact title match)
                if match['match_type'] in ['exact_title', 'first_of_multiple', 'semantic_disambiguate']:
                    child_label = 'PT'
                    stats['title_matched'] += 1
                else:
                    child_label = 'PT?'
                    stats['semantic_matched'] += 1

                # Add child node with Pearltrees link
                # In SimpleMind, child topics are siblings in <topics>, not nested
                if topics_elem is not None:
                    child_id = f"{topic_id}_pt"
                    child_topic = ET.SubElement(topics_elem, 'topic')
                    child_topic.set('id', child_id)
                    child_topic.set('parent', topic_id)
                    # Get parent position and offset child slightly
                    parent_x = float(topic.get('x', '0'))
                    parent_y = float(topic.get('y', '0'))
                    child_topic.set('x', f"{parent_x + 50:.2f}")
                    child_topic.set('y', f"{parent_y + 30:.2f}")
                    child_topic.set('text', child_label)
                    # Add no-border style for cleaner look
                    child_style = ET.SubElement(child_topic, 'style')
                    child_style.set('borderstyle', 'sbsNone')
                    child_link = ET.SubElement(child_topic, 'link')
                    child_link.set('urllink', match['uri'])

                    stats['link_updates'][child_id] = {
                        'parent_id': topic_id,
                        'label': child_label,
                        'url': match['uri']
                    }

                    stats['matches'].append({
                        'node_id': topic_id,
                        'node_text': clean_text,
                        'match_uri': match['uri'],
                        'match_title': match['title'],
                        'match_type': match['match_type'],
                        'score': match['score'],
                        'child_label': child_label
                    })

                    if verbose:
                        print(f"  [{match['match_type']}] '{clean_text}' +{child_label} -> {match['uri']} (score: {match['score']:.3f})")
            else:
                stats['no_match'] += 1
                if verbose:
                    print(f"  [no match] '{clean_text}'")

    return stats


class ProjectionEngine:
    """
    Engine for projecting query embeddings using the trained model.

    Uses the distilled transformer or federated model to project embeddings
    into the hierarchical space where target_text (materialized paths +
    structure lists) are encoded.
    """

    # Common embedding models and their dimensions
    EMBEDDER_DIMS = {
        "nomic-ai/nomic-embed-text-v1.5": 768,
        "sentence-transformers/all-MiniLM-L6-v2": 384,
        "BAAI/bge-small-en-v1.5": 384,
    }

    def __init__(self, model_path: Path, embedder_name: str = None):
        self.model_path = Path(model_path)
        self._embedder = None
        self._transformer = None
        self._federated = None
        self.embed_dim = None

        # Determine model type by extension
        if model_path.suffix == '.pt':
            self._load_transformer()
        elif model_path.suffix == '.pkl':
            self._load_federated()
        else:
            raise ValueError(f"Unknown model type: {model_path.suffix}")

        # Select embedder based on model dimension if not specified
        if embedder_name:
            self.embedder_name = embedder_name
        else:
            # Auto-select embedder based on embed_dim
            if self.embed_dim == 384:
                self.embedder_name = "sentence-transformers/all-MiniLM-L6-v2"
            else:
                self.embedder_name = "nomic-ai/nomic-embed-text-v1.5"
            print(f"  Auto-selected embedder: {self.embedder_name} (dim={self.embed_dim})")

    def _load_transformer(self):
        """Load the distilled projection transformer."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "unifyweaver" / "targets" / "python_runtime"))
        from projection_transformer import ProjectionTransformer

        print(f"Loading projection transformer: {self.model_path}")
        self._transformer = ProjectionTransformer.load(str(self.model_path), device="cpu")
        info = self._transformer.get_info()
        self.embed_dim = info['embed_dim']
        print(f"  Parameters: {info['total_parameters']:,}, H={info['num_heads']}, L={info['num_layers']}, dim={self.embed_dim}")

    def _load_federated(self):
        """Load the federated Procrustes model."""
        import pickle
        import sys
        from types import ModuleType

        # NumPy 2.0 compatibility shim for pickle files saved with numpy._core
        if not hasattr(np, '_core'):
            np_core = ModuleType('numpy._core')
            np_core.multiarray = np.core.multiarray
            np_core.umath = np.core.umath
            sys.modules['numpy._core'] = np_core
            sys.modules['numpy._core.multiarray'] = np.core.multiarray

        print(f"Loading federated model: {self.model_path}")
        with open(self.model_path, "rb") as f:
            self._federated = pickle.load(f)

        # Determine cluster directory
        if "cluster_dir" in self._federated:
            self.cluster_dir = Path(self._federated["cluster_dir"])
        else:
            self.cluster_dir = self.model_path.with_suffix('')

        # Load routing data
        routing_path = self.cluster_dir / "routing_data.npz"
        if routing_path.exists():
            data = np.load(routing_path)
            self.query_embeddings = data["query_embeddings"]
            self.target_embeddings = data["target_embeddings"]
            keys = data["idx_to_cluster_keys"]
            values = data["idx_to_cluster_values"]
            self.idx_to_cluster = {int(k): str(v) for k, v in zip(keys, values)}
            # Get embedding dimension from query embeddings
            self.embed_dim = self.query_embeddings.shape[1]

        # Load cluster W matrices
        self.clusters = {}
        for cid in self._federated["cluster_ids"]:
            cluster_path = self.cluster_dir / f"{cid}.npz"
            if cluster_path.exists():
                cdata = np.load(cluster_path)
                self.clusters[cid] = {"W": cdata["W_stack"][0]}
                # Get dimension from W matrix if not set
                if self.embed_dim is None:
                    self.embed_dim = cdata["W_stack"][0].shape[0]

        self.temperature = self._federated.get("temperature", 0.1)
        print(f"  Clusters: {len(self.clusters)}, Temperature: {self.temperature}, dim={self.embed_dim}")

    @property
    def embedder(self):
        """Lazy load sentence transformer embedder."""
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            print(f"Loading embedder: {self.embedder_name}")
            self._embedder = SentenceTransformer(self.embedder_name, trust_remote_code=True)
        return self._embedder

    def embed(self, text: str) -> np.ndarray:
        """Embed text using the sentence transformer."""
        return self.embedder.encode([text], show_progress_bar=False)[0].astype(np.float32)

    def project(self, q_emb: np.ndarray) -> np.ndarray:
        """Project query embedding into hierarchical space."""
        if self._transformer is not None:
            return self._transformer.project(q_emb)
        elif self._federated is not None:
            return self._project_federated(q_emb)
        else:
            raise RuntimeError("No model loaded")

    def _project_federated(self, q_emb: np.ndarray, top_k_routing: int = 10) -> np.ndarray:
        """Project using federated model with softmax routing."""
        # Compute similarities to training queries
        sims = q_emb @ self.query_embeddings.T

        # Softmax weights
        sims_shifted = (sims - np.max(sims)) / self.temperature
        weights = np.exp(sims_shifted)
        weights /= weights.sum()

        # Get top-k training queries
        top_indices = np.argsort(weights)[-top_k_routing:]

        # Weighted projection
        proj = np.zeros_like(q_emb)
        for idx in top_indices:
            cid = self.idx_to_cluster.get(int(idx))
            if cid and cid in self.clusters:
                W = self.clusters[cid]["W"]
                proj += weights[idx] * (q_emb @ W)

        return proj

    def embed_and_project(self, text: str) -> np.ndarray:
        """Embed text and project into hierarchical space."""
        q_emb = self.embed(text)
        return self.project(q_emb)


def main():
    parser = argparse.ArgumentParser(
        description="Link Pearltrees items to mindmap nodes"
    )
    parser.add_argument("--mindmap", type=Path, required=True,
                       help="Input .smmx mindmap file")
    parser.add_argument("--pearls", type=Path,
                       help="JSONL file with PagePearl items")
    parser.add_argument("--trees", type=Path,
                       help="JSONL file with Tree items")
    parser.add_argument("--api-cache", type=Path,
                       help="Directory with API cache JSON files (for URL matching)")
    parser.add_argument("--url-db", type=Path,
                       help="SQLite database with external URLs (e.g., children_index.db)")
    parser.add_argument("--embeddings", type=Path,
                       help="NPZ file with embeddings for semantic matching")
    parser.add_argument("--output", type=Path,
                       help="Output .smmx file (default: input with _linked suffix)")
    parser.add_argument("--threshold", type=float, default=0.7,
                       help="Minimum cosine similarity for semantic matching (default: 0.7)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Print detailed matching info")
    parser.add_argument("--dry-run", action="store_true",
                       help="Don't write output, just show what would be linked")
    parser.add_argument("--report", type=Path,
                       help="Write detailed match report to JSON file")
    parser.add_argument("--projection-model", type=Path,
                       help="Projection model (.pt transformer or .pkl federated) for hierarchical matching")

    args = parser.parse_args()

    # Validate input
    if not args.mindmap.exists():
        print(f"Error: Mindmap not found: {args.mindmap}")
        return 1

    # Set default output
    if args.output is None:
        stem = args.mindmap.stem
        args.output = args.mindmap.with_name(f"{stem}_linked.smmx")

    print(f"Loading mindmap: {args.mindmap}")
    root, metadata, raw_xml = parse_smmx(args.mindmap)

    # Load Pearltrees data
    pearls = []
    trees = []

    if args.pearls and args.pearls.exists():
        print(f"Loading pearls: {args.pearls}")
        pearls = load_jsonl(args.pearls)
        print(f"  Loaded {len(pearls)} pearls")

    if args.trees and args.trees.exists():
        print(f"Loading trees: {args.trees}")
        trees = load_jsonl(args.trees)
        print(f"  Loaded {len(trees)} trees")

    # Build indices
    uri_index = build_uri_index(pearls, trees)
    title_index = build_title_index(pearls + trees)
    print(f"Built indices: {len(uri_index)} URIs, {len(title_index)} unique titles")

    # Load URL indices for URL matching
    url_index = {}

    # Load from SQLite database (preferred source - has most data)
    if args.url_db and args.url_db.exists():
        print(f"Loading URL index from database: {args.url_db}")
        url_index = load_url_index_from_db(args.url_db)
        print(f"  Loaded {len(url_index)} external URLs from database")

    # Load from API cache (supplements database)
    if args.api_cache and args.api_cache.exists():
        print(f"Loading API cache: {args.api_cache}")
        api_urls = load_api_cache(args.api_cache)
        # Merge - extend lists for same URL, add new URLs
        added_count = 0
        for url, match_list in api_urls.items():
            if url not in url_index:
                url_index[url] = match_list
                added_count += 1
            else:
                # Extend existing list with new matches
                url_index[url].extend(match_list)
        print(f"  Added {added_count} new URLs from API cache (total: {len(url_index)})")

    # Load embeddings
    embeddings = None
    embed_titles = None
    embed_uris = None
    projection_engine = None

    if args.embeddings and args.embeddings.exists():
        print(f"Loading embeddings: {args.embeddings}")
        embeddings, embed_titles, embed_uris = load_embeddings(args.embeddings)
        print(f"  Loaded {len(embeddings)} embeddings ({embeddings.shape[1]}D)")

    # Load projection model for hierarchical matching
    if args.projection_model and args.projection_model.exists():
        projection_engine = ProjectionEngine(args.projection_model)
    elif args.projection_model:
        print(f"Warning: Projection model not found: {args.projection_model}")

    # Link nodes
    print(f"\nLinking nodes (threshold={args.threshold})...")
    stats = link_mindmap_nodes(
        root,
        uri_index,
        title_index,
        url_index,
        embeddings,
        embed_titles,
        embed_uris,
        args.threshold,
        projection_engine,
        verbose=args.verbose
    )

    # Print summary
    print(f"\n--- Summary ---")
    print(f"Total nodes:       {stats['total_nodes']}")
    print(f"Already Pearltrees:{stats['already_pearltrees']}")
    print(f"URL matched:       {stats['url_matched']}")
    print(f"Title matched:     {stats['title_matched']}")
    print(f"Semantic matched:  {stats['semantic_matched']}")
    print(f"No match:          {stats['no_match']}")

    # Write output
    if not args.dry_run:
        print(f"\nWriting: {args.output}")
        write_smmx(root, args.output, args.mindmap)
    else:
        print("\n[dry-run] No output written")

    # Write report
    if args.report:
        print(f"Writing report: {args.report}")
        with open(args.report, 'w', encoding='utf-8') as f:
            json.dump({
                'source': str(args.mindmap),
                'output': str(args.output),
                'threshold': args.threshold,
                'stats': stats
            }, f, indent=2)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
