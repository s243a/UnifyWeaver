#!/usr/bin/env python3
"""
Batch rename mindmap folders using LLM-generated names.

Renames folders from root to leaves, then updates all cloudmapref links.

Usage:
    # Dry run - show what would be renamed
    python3 scripts/mindmap/batch_rename_folders.py \
        --base-dir output/mindmaps_curated/ \
        --dry-run --verbose

    # Actually rename folders
    python3 scripts/mindmap/batch_rename_folders.py \
        --base-dir output/mindmaps_curated/

    # With descriptions output
    python3 scripts/mindmap/batch_rename_folders.py \
        --base-dir output/mindmaps_curated/ \
        --descriptions output/mindmaps_curated/folder_descriptions.json
"""

import argparse
import json
import os
import re
import shutil
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from xml.etree import ElementTree as ET

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from generate_mindmap import generate_folder_name_llm, sanitize_filename


def get_folder_hierarchy(base_dir: Path) -> List[Tuple[int, Path]]:
    """Get all folders sorted by depth (root first).

    Returns list of (depth, folder_path) tuples.
    """
    folders = []
    for root, dirs, files in os.walk(base_dir):
        # Skip hidden directories
        dirs[:] = [d for d in dirs if not d.startswith('.')]

        root_path = Path(root)
        # Only include folders that contain .smmx files (directly or in subdirs)
        has_smmx = any(f.endswith('.smmx') for f in files)
        has_smmx_subdir = any(
            any(f.endswith('.smmx') for f in subfiles)
            for _, _, subfiles in os.walk(root_path)
        )

        if has_smmx or has_smmx_subdir:
            rel_path = root_path.relative_to(base_dir)
            depth = len(rel_path.parts) if str(rel_path) != '.' else 0
            if depth > 0:  # Skip base dir itself
                folders.append((depth, root_path))

    # Sort by depth (shallowest first)
    folders.sort(key=lambda x: x[0])
    return folders


def get_trees_in_folder(folder: Path, embeddings_data: dict, index: dict) -> List[Dict]:
    """Get tree items for a folder."""
    tree_ids_list = list(embeddings_data['tree_ids'])
    titles_list = list(embeddings_data['titles'])
    id_to_idx = {str(tid): i for i, tid in enumerate(tree_ids_list)}

    tree_items = []
    for tree_id, rel_path in index.items():
        # Check if this tree is in this folder
        if tree_id not in id_to_idx:
            continue
        tree_folder = str(Path(rel_path).parent)
        if tree_folder == str(folder.name) or rel_path.startswith(str(folder.name) + '/'):
            idx = id_to_idx[tree_id]
            tree_items.append({
                "title": str(titles_list[idx]),
                "path": rel_path,
                "tree_id": tree_id
            })

    return tree_items


def generate_folder_name(
    folder: Path,
    base_dir: Path,
    embeddings_data: dict,
    index: dict,
    context_level: str = "jsonl",
    with_descriptions: bool = False,
    parent_name: str = None
) -> Optional[Dict]:
    """Generate a new name for a folder using LLM."""
    # Get relative path for matching
    rel_folder = folder.relative_to(base_dir)

    # Find all trees in this folder (not subfolders)
    tree_ids_list = list(embeddings_data['tree_ids'])
    titles_list = list(embeddings_data['titles'])
    id_to_idx = {str(tid): i for i, tid in enumerate(tree_ids_list)}

    tree_items = []
    for tree_id, rel_path in index.items():
        if tree_id not in id_to_idx:
            continue
        # Check if directly in this folder (not subfolder)
        tree_folder = str(Path(rel_path).parent)
        if tree_folder == str(rel_folder):
            idx = id_to_idx[tree_id]
            tree_items.append({
                "title": str(titles_list[idx]),
                "path": rel_path,
                "tree_id": tree_id
            })

    if not tree_items:
        return None

    # Generate name with parent context
    result = generate_folder_name_llm(
        tree_items,
        context_level=context_level,
        with_descriptions=with_descriptions,
        parent_folder_path=parent_name
    )

    if with_descriptions and result:
        return result
    elif result:
        return {"name": result}
    return None


def update_cloudmaprefs(
    base_dir: Path,
    path_mapping: Dict[str, str],
    dry_run: bool = False,
    verbose: bool = False
) -> int:
    """Update all cloudmapref attributes in all mindmaps.

    Args:
        base_dir: Base directory containing mindmaps
        path_mapping: Dict of old_relative_path -> new_relative_path
        dry_run: If True, don't actually modify files
        verbose: Print detailed info

    Returns:
        Number of files modified
    """
    modified_count = 0

    for smmx_file in base_dir.rglob("*.smmx"):
        try:
            # Read the zip file
            with zipfile.ZipFile(smmx_file, 'r') as zf:
                xml_content = zf.read('document.xml').decode('utf-8')

            original_content = xml_content
            modified = False

            # Find all cloudmapref attributes
            # Pattern: cloudmapref="some/path/file.smmx"
            def replace_cloudmapref(match):
                nonlocal modified
                old_ref = match.group(1)

                # Resolve the reference relative to this file's location
                smmx_dir = smmx_file.parent
                ref_path = (smmx_dir / old_ref).resolve()

                try:
                    old_rel = str(ref_path.relative_to(base_dir))
                except ValueError:
                    return match.group(0)  # Not under base_dir

                # Check if this path needs updating
                for old_path, new_path in path_mapping.items():
                    if old_rel.startswith(old_path):
                        # Replace the path portion
                        new_rel = old_rel.replace(old_path, new_path, 1)
                        # Compute new relative reference from this file
                        new_ref_path = base_dir / new_rel
                        new_ref = os.path.relpath(new_ref_path, smmx_dir)
                        modified = True
                        if verbose:
                            print(f"    {old_ref} -> {new_ref}")
                        return f'cloudmapref="{new_ref}"'

                return match.group(0)

            xml_content = re.sub(
                r'cloudmapref="([^"]+)"',
                replace_cloudmapref,
                xml_content
            )

            if modified and xml_content != original_content:
                modified_count += 1
                if verbose:
                    print(f"  Updating: {smmx_file.relative_to(base_dir)}")

                if not dry_run:
                    # Write back to the zip
                    with zipfile.ZipFile(smmx_file, 'w', zipfile.ZIP_DEFLATED) as zf:
                        zf.writestr('document.xml', xml_content.encode('utf-8'))

        except Exception as e:
            if verbose:
                print(f"  Error processing {smmx_file}: {e}", file=sys.stderr)

    return modified_count


def main():
    parser = argparse.ArgumentParser(
        description="Batch rename mindmap folders using LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--base-dir", "-d",
        type=Path,
        default=Path("output/mindmaps_curated"),
        help="Base directory containing mindmaps"
    )
    parser.add_argument(
        "--embeddings", "-e",
        type=Path,
        default=Path("models/dual_embeddings_combined_2026-01-02_trees_only.npz"),
        help="Path to embeddings NPZ file"
    )
    parser.add_argument(
        "--index", "-i",
        type=Path,
        default=Path("output/mindmaps_curated/index.json"),
        help="Path to mindmap index"
    )
    parser.add_argument(
        "--context-level", "-c",
        choices=["titles", "paths", "jsonl"],
        default="jsonl",
        help="Context level for LLM naming"
    )
    parser.add_argument(
        "--descriptions",
        type=Path,
        help="Output file for folder descriptions JSON"
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Don't actually rename, just show what would happen"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip folders that don't look like auto-generated names (id* pattern)"
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=0,
        help="Limit number of folders to process (0 = all)"
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=0,
        help="Only process folders at this depth (0 = all depths)"
    )

    args = parser.parse_args()

    # Validate paths
    if not args.base_dir.exists():
        print(f"Error: Base directory not found: {args.base_dir}", file=sys.stderr)
        sys.exit(1)

    if not args.embeddings.exists():
        print(f"Error: Embeddings file not found: {args.embeddings}", file=sys.stderr)
        sys.exit(1)

    if not args.index.exists():
        print(f"Error: Index file not found: {args.index}", file=sys.stderr)
        sys.exit(1)

    # Load data
    print("Loading embeddings...")
    import numpy as np
    embeddings_data = dict(np.load(args.embeddings, allow_pickle=True))

    print("Loading index...")
    with open(args.index) as f:
        index_data = json.load(f)
    index = index_data.get('index', index_data)

    # Get folder hierarchy
    print("Scanning folder hierarchy...")
    folders = get_folder_hierarchy(args.base_dir)
    print(f"Found {len(folders)} folders")

    # Filter by depth if specified
    if args.depth > 0:
        folders = [(d, f) for d, f in folders if d == args.depth]
        print(f"Filtered to {len(folders)} folders at depth {args.depth}")

    # Limit if specified
    if args.limit > 0:
        folders = folders[:args.limit]
        print(f"Limited to {len(folders)} folders")

    # Track renames: old_path -> new_path (relative to base_dir)
    path_mapping = {}
    descriptions = {}

    # Process folders from root to leaves
    print("\nGenerating folder names...")
    processed = 0
    for depth, folder in folders:
        rel_path = folder.relative_to(args.base_dir)

        # Skip if not matching id* pattern (already renamed)
        if args.skip_existing:
            if not re.match(r'^id\d+', folder.name) and not any(
                re.match(r'^id\d+', part) for part in rel_path.parts
            ):
                if args.verbose:
                    print(f"  Skipping (already named): {rel_path}")
                continue

        # Get parent name for context
        parent_name = None
        if depth > 1:
            parent_rel = str(rel_path.parent)
            # Use renamed parent if available
            parent_name = path_mapping.get(parent_rel, parent_rel)

        # Generate new name
        processed += 1
        print(f"[{processed}/{len(folders)}] Processing: {rel_path}", end="", flush=True)

        result = generate_folder_name(
            folder,
            args.base_dir,
            embeddings_data,
            index,
            context_level=args.context_level,
            with_descriptions=bool(args.descriptions),
            parent_name=parent_name
        )

        if result is None:
            print(f" -> (no trees)")
            continue

        new_name = result.get("name", folder.name)
        if not new_name:
            print(f" -> (no name generated)")
            continue

        # Sanitize the new name
        new_name = sanitize_filename(new_name, max_length=40)

        # Build new path
        if depth == 1:
            new_rel_path = new_name
        else:
            # Use renamed parent path
            parent_rel = str(rel_path.parent)
            new_parent = path_mapping.get(parent_rel, parent_rel)
            new_rel_path = f"{new_parent}/{new_name}"

        old_rel_path = str(rel_path)

        if old_rel_path != new_rel_path:
            path_mapping[old_rel_path] = new_rel_path
            print(f" -> {new_name}")

            if args.descriptions and 'short' in result:
                descriptions[new_rel_path] = {
                    "old_path": old_rel_path,
                    "name": new_name,
                    "short": result.get("short", ""),
                    "medium": result.get("medium", ""),
                    "long": result.get("long", "")
                }
        else:
            print(f" -> (unchanged)")

    print(f"\nFolders to rename: {len(path_mapping)}")

    if not path_mapping:
        print("No folders to rename.")
        return

    # Phase 1: Rename folders (in reverse depth order to avoid conflicts)
    if not args.dry_run:
        print("\nRenaming folders...")
        # Sort by depth descending (deepest first) to avoid path conflicts
        sorted_renames = sorted(
            path_mapping.items(),
            key=lambda x: x[0].count('/'),
            reverse=True
        )

        for old_rel, new_rel in sorted_renames:
            old_path = args.base_dir / old_rel
            new_path = args.base_dir / new_rel

            if old_path.exists():
                # Ensure parent exists
                new_path.parent.mkdir(parents=True, exist_ok=True)

                if args.verbose:
                    print(f"  Moving: {old_rel} -> {new_rel}")

                shutil.move(str(old_path), str(new_path))

    # Phase 2: Update cloudmapref links
    print("\nUpdating cloudmapref links...")
    modified = update_cloudmaprefs(
        args.base_dir,
        path_mapping,
        dry_run=args.dry_run,
        verbose=args.verbose
    )
    print(f"Modified {modified} mindmap files")

    # Save descriptions
    if args.descriptions and descriptions:
        if not args.dry_run:
            with open(args.descriptions, 'w') as f:
                json.dump(descriptions, f, indent=2)
            print(f"\nSaved descriptions to: {args.descriptions}")
        else:
            print(f"\nWould save {len(descriptions)} descriptions to: {args.descriptions}")

    # Update index
    if not args.dry_run:
        print("\nUpdating index...")
        new_index = {}
        for tree_id, rel_path in index.items():
            new_path = rel_path
            for old_prefix, new_prefix in path_mapping.items():
                if rel_path.startswith(old_prefix + '/') or rel_path.startswith(old_prefix):
                    new_path = rel_path.replace(old_prefix, new_prefix, 1)
                    break
            new_index[tree_id] = new_path

        index_data['index'] = new_index
        with open(args.index, 'w') as f:
            json.dump(index_data, f, indent=2)
        print(f"Updated index: {args.index}")

    print("\nDone!")
    if args.dry_run:
        print("(Dry run - no changes made)")


if __name__ == "__main__":
    main()
