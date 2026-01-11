#!/usr/bin/env python3
"""
Rename mindmap files and update all cloudmapref references.

When a mindmap is renamed, finds all other mindmaps that link to it
and updates their cloudmapref attributes to point to the new filename.

Usage:
    # Rename with explicit new name
    python3 rename_mindmap.py --mindmap id10380971.smmx --new-name Technology_10380971.smmx

    # Auto-generate titled name from root topic
    python3 rename_mindmap.py --mindmap id10380971.smmx --titled

    # Batch rename all mindmaps to titled format
    python3 rename_mindmap.py --batch output/mindmaps_curated/ --titled

    # Dry run to see what would change
    python3 rename_mindmap.py --mindmap id10380971.smmx --titled --dry-run
"""

import argparse
import os
import re
import sys
import zipfile
from pathlib import Path
from typing import List, Optional, Set, Tuple
from xml.etree import ElementTree as ET

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from index_store import create_index_store, IndexStore


def sanitize_filename(title: str, max_length: int = 50) -> str:
    """Sanitize a title for use as a filename.

    Args:
        title: The raw title to sanitize
        max_length: Maximum length for the filename (default: 50)

    Returns:
        A filesystem-safe filename string
    """
    if not title:
        return ""

    # Replace problematic characters with underscores
    # Forbidden in Windows: / \ : * ? " < > |
    # Also replace spaces and other problematic chars
    sanitized = re.sub(r'[/\\:*?"<>|\s]+', '_', title)

    # Remove leading/trailing underscores and dots
    sanitized = sanitized.strip('_.')

    # Collapse multiple underscores
    sanitized = re.sub(r'_+', '_', sanitized)

    # Truncate to max length, but try to break at underscore
    if len(sanitized) > max_length:
        truncated = sanitized[:max_length]
        last_underscore = truncated.rfind('_')
        if last_underscore > max_length // 2:
            sanitized = truncated[:last_underscore]
        else:
            sanitized = truncated

    return sanitized


def extract_tree_id_from_filename(filename: str) -> Optional[str]:
    """Extract tree ID from mindmap filename like 'id75009241.smmx' or 'Title_75009241.smmx'."""
    basename = os.path.basename(filename)
    # Try pattern: id{digits}
    match = re.search(r'id(\d+)', basename)
    if match:
        return match.group(1)
    # Try pattern: _digits.smmx (for titled files)
    match = re.search(r'_(\d+)\.smmx$', basename)
    if match:
        return match.group(1)
    return None


def get_root_topic_title(smmx_path: Path) -> Optional[str]:
    """Extract the root topic title from a mindmap file.

    Args:
        smmx_path: Path to .smmx file

    Returns:
        Root topic title or None if not found
    """
    try:
        with zipfile.ZipFile(smmx_path, 'r') as zf:
            xml_content = zf.read('document/mindmap.xml').decode('utf-8')
    except (zipfile.BadZipFile, KeyError) as e:
        print(f"Error reading {smmx_path}: {e}", file=sys.stderr)
        return None

    try:
        root = ET.fromstring(xml_content)
    except ET.ParseError as e:
        print(f"Error parsing XML in {smmx_path}: {e}", file=sys.stderr)
        return None

    # Find root topic (id="0")
    for topic in root.iter('topic'):
        if topic.get('id') == '0':
            return topic.get('text')

    return None


def generate_titled_filename(smmx_path: Path, max_title_length: int = 50, keep_id: bool = False) -> str:
    """Generate a titled filename from the mindmap's root topic.

    Args:
        smmx_path: Path to .smmx file
        max_title_length: Maximum length for the title part
        keep_id: If True, use 'Title_id12345678.smmx' format; otherwise 'Title_12345678.smmx'

    Returns:
        Filename like 'Title_12345678.smmx' or 'Title_id12345678.smmx'
    """
    tree_id = extract_tree_id_from_filename(str(smmx_path))
    title = get_root_topic_title(smmx_path)

    id_prefix = "id" if keep_id else ""

    if title:
        sanitized = sanitize_filename(title, max_title_length)
        return f"{sanitized}_{id_prefix}{tree_id}.smmx"
    else:
        # Fallback to id-based name
        return f"id{tree_id}.smmx"


def find_referencing_mindmaps(
    target_filename: str,
    search_dir: Path,
    verbose: bool = False
) -> List[Path]:
    """Find all mindmaps with cloudmapref pointing to target filename.

    Args:
        target_filename: The filename being referenced (e.g., 'id10380971.smmx')
        search_dir: Directory to search for mindmaps
        verbose: Print progress

    Returns:
        List of paths to mindmaps that reference the target
    """
    referencing = []
    target_basename = os.path.basename(target_filename)

    for smmx_path in search_dir.rglob('*.smmx'):
        try:
            with zipfile.ZipFile(smmx_path, 'r') as zf:
                xml_content = zf.read('document/mindmap.xml').decode('utf-8')
        except (zipfile.BadZipFile, KeyError):
            continue

        # Quick string check before parsing XML
        if target_basename not in xml_content:
            continue

        try:
            root = ET.fromstring(xml_content)
        except ET.ParseError:
            continue

        # Check for cloudmapref containing target
        for link in root.iter('link'):
            cloudmapref = link.get('cloudmapref', '')
            if target_basename in cloudmapref:
                referencing.append(smmx_path)
                if verbose:
                    print(f"  Found reference in: {smmx_path}")
                break

    return referencing


def update_cloudmapref_in_mindmap(
    smmx_path: Path,
    old_filename: str,
    new_filename: str,
    dry_run: bool = False,
    verbose: bool = False
) -> int:
    """Update cloudmapref attributes in a mindmap.

    Args:
        smmx_path: Path to .smmx file to update
        old_filename: Old filename to replace (basename only)
        new_filename: New filename to use (basename only)
        dry_run: If True, don't save changes
        verbose: Print each update

    Returns:
        Number of links updated
    """
    try:
        with zipfile.ZipFile(smmx_path, 'r') as zf:
            xml_content = zf.read('document/mindmap.xml').decode('utf-8')
    except (zipfile.BadZipFile, KeyError) as e:
        print(f"Error reading {smmx_path}: {e}", file=sys.stderr)
        return 0

    try:
        root = ET.fromstring(xml_content)
    except ET.ParseError as e:
        print(f"Error parsing XML in {smmx_path}: {e}", file=sys.stderr)
        return 0

    old_basename = os.path.basename(old_filename)
    new_basename = os.path.basename(new_filename)
    updates = 0

    for link in root.iter('link'):
        cloudmapref = link.get('cloudmapref', '')
        if old_basename in cloudmapref:
            # Replace the filename in the path
            new_ref = cloudmapref.replace(old_basename, new_basename)
            link.set('cloudmapref', new_ref)
            updates += 1
            if verbose:
                print(f"    {cloudmapref} -> {new_ref}")

    if updates > 0 and not dry_run:
        # Write back to smmx
        xml_output = ET.tostring(root, encoding='unicode')
        xml_output = '<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE simplemind-mindmaps>\n' + xml_output

        with zipfile.ZipFile(smmx_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr('document/mindmap.xml', xml_output.encode('utf-8'))

    return updates


def rename_mindmap(
    old_path: Path,
    new_name: str,
    search_dir: Path = None,
    index_store: IndexStore = None,
    dry_run: bool = False,
    verbose: bool = False
) -> Tuple[bool, int, int]:
    """Rename a mindmap file and update all references to it.

    Args:
        old_path: Current path to the mindmap
        new_name: New filename (just the name, not full path)
        search_dir: Directory to search for referencing mindmaps
        index_store: Optional index store to update
        dry_run: If True, don't make any changes
        verbose: Print detailed progress

    Returns:
        Tuple of (success, files_updated, links_updated)
    """
    old_path = Path(old_path).resolve()
    if not old_path.exists():
        print(f"Error: File not found: {old_path}", file=sys.stderr)
        return (False, 0, 0)

    new_path = old_path.parent / new_name
    old_basename = old_path.name
    new_basename = new_name

    if old_basename == new_basename:
        print(f"Filename unchanged: {old_basename}")
        return (True, 0, 0)

    if new_path.exists() and new_path != old_path:
        print(f"Error: Target file already exists: {new_path}", file=sys.stderr)
        return (False, 0, 0)

    # Determine search directory
    if search_dir is None:
        search_dir = old_path.parent

    print(f"Renaming: {old_basename} -> {new_basename}")

    # Find all mindmaps that reference this file
    if verbose:
        print(f"Searching for references in: {search_dir}")

    referencing = find_referencing_mindmaps(old_basename, search_dir, verbose)
    print(f"Found {len(referencing)} mindmaps with references")

    # Update references in each file
    files_updated = 0
    links_updated = 0

    for ref_path in referencing:
        if ref_path == old_path:
            continue  # Skip self

        if verbose:
            print(f"  Updating: {ref_path.name}")

        count = update_cloudmapref_in_mindmap(
            ref_path, old_basename, new_basename,
            dry_run=dry_run, verbose=verbose
        )

        if count > 0:
            files_updated += 1
            links_updated += count

    # Rename the file
    if not dry_run:
        old_path.rename(new_path)
        print(f"Renamed file: {old_basename} -> {new_basename}")

        # Update index if provided
        if index_store:
            tree_id = extract_tree_id_from_filename(new_basename)
            if tree_id:
                old_rel = os.path.relpath(old_path, index_store.base_dir)
                new_rel = os.path.relpath(new_path, index_store.base_dir)
                index_store.set(tree_id, new_rel)
                if verbose:
                    print(f"Updated index: {tree_id} -> {new_rel}")
    else:
        print(f"[DRY RUN] Would rename: {old_basename} -> {new_basename}")

    return (True, files_updated, links_updated)


def batch_rename_to_titled(
    directory: Path,
    dry_run: bool = False,
    verbose: bool = False,
    max_title_length: int = 50,
    keep_id: bool = False
) -> Tuple[int, int, int]:
    """Batch rename all mindmaps to titled format.

    Args:
        directory: Directory containing mindmaps
        dry_run: If True, don't make changes
        verbose: Print detailed progress
        max_title_length: Maximum length for title part
        keep_id: If True, use 'Title_id12345678.smmx' format

    Returns:
        Tuple of (files_renamed, files_updated, links_updated)
    """
    # First pass: collect all renames needed
    renames = []  # List of (old_path, new_name)

    # Pattern depends on whether we're keeping 'id' prefix
    if keep_id:
        skip_pattern = r'^[^_]+_id\d+\.smmx$'  # Skip Title_id12345678.smmx
    else:
        skip_pattern = r'^[^_]+_\d+\.smmx$'    # Skip Title_12345678.smmx

    for smmx_path in directory.rglob('*.smmx'):
        # Skip already titled files
        basename = smmx_path.name
        if re.match(skip_pattern, basename) and not basename.startswith('id'):
            if verbose:
                print(f"Skipping already titled: {basename}")
            continue

        new_name = generate_titled_filename(smmx_path, max_title_length, keep_id=keep_id)
        if new_name != basename:
            renames.append((smmx_path, new_name))

    print(f"Found {len(renames)} mindmaps to rename")

    if not renames:
        return (0, 0, 0)

    # Build a map of old -> new filenames for reference updates
    rename_map = {old.name: new for old, new in renames}

    files_renamed = 0
    files_updated = 0
    total_links_updated = 0

    # First, update all references (before renaming files)
    print("\nUpdating references...")
    for smmx_path in directory.rglob('*.smmx'):
        updates_in_file = 0

        try:
            with zipfile.ZipFile(smmx_path, 'r') as zf:
                xml_content = zf.read('document/mindmap.xml').decode('utf-8')
        except (zipfile.BadZipFile, KeyError):
            continue

        # Check if any references need updating
        needs_update = False
        for old_name in rename_map:
            if old_name in xml_content:
                needs_update = True
                break

        if not needs_update:
            continue

        try:
            root = ET.fromstring(xml_content)
        except ET.ParseError:
            continue

        for link in root.iter('link'):
            cloudmapref = link.get('cloudmapref', '')
            for old_name, new_name in rename_map.items():
                if old_name in cloudmapref:
                    new_ref = cloudmapref.replace(old_name, new_name)
                    link.set('cloudmapref', new_ref)
                    updates_in_file += 1
                    if verbose:
                        print(f"  {smmx_path.name}: {old_name} -> {new_name}")

        if updates_in_file > 0:
            if not dry_run:
                xml_output = ET.tostring(root, encoding='unicode')
                xml_output = '<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE simplemind-mindmaps>\n' + xml_output

                with zipfile.ZipFile(smmx_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                    zf.writestr('document/mindmap.xml', xml_output.encode('utf-8'))

            files_updated += 1
            total_links_updated += updates_in_file

    # Second, rename all files
    print("\nRenaming files...")
    for old_path, new_name in renames:
        new_path = old_path.parent / new_name

        if new_path.exists() and new_path != old_path:
            print(f"  Skipping (target exists): {old_path.name} -> {new_name}")
            continue

        if not dry_run:
            old_path.rename(new_path)

        files_renamed += 1
        if verbose:
            print(f"  {old_path.name} -> {new_name}")

    return (files_renamed, files_updated, total_links_updated)


def main():
    parser = argparse.ArgumentParser(
        description='Rename mindmap files and update references',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Single file mode
    parser.add_argument('--mindmap', '-m', type=Path,
                        help='Single mindmap file to rename')
    parser.add_argument('--new-name', '-n', type=str,
                        help='New filename (for single file mode)')

    # Batch mode
    parser.add_argument('--batch', '-b', type=Path,
                        help='Directory for batch renaming')

    # Common options
    parser.add_argument('--titled', '-t', action='store_true',
                        help='Generate titled filename from root topic (Title_ID.smmx)')
    parser.add_argument('--id-prefix', dest='id_prefix', action='store_true', default=True,
                        help='Include "id" prefix in filename (Title_id12345678.smmx) [default]')
    parser.add_argument('--no-id-prefix', dest='id_prefix', action='store_false',
                        help='Omit "id" prefix in filename (Title_12345678.smmx)')
    parser.add_argument('--max-title', type=int, default=50,
                        help='Maximum title length (default: 50)')
    parser.add_argument('--index', '-i', type=str,
                        help='Index file to update (JSON, TSV, or SQLite)')
    parser.add_argument('--search-dir', '-s', type=Path,
                        help='Directory to search for references (default: same as mindmap)')
    parser.add_argument('--dry-run', '-d', action='store_true',
                        help='Show what would be done without making changes')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Print detailed progress')

    args = parser.parse_args()

    # Validate arguments
    if args.batch and args.mindmap:
        parser.error("Cannot use --batch and --mindmap together")

    if not args.batch and not args.mindmap:
        parser.error("Must specify either --batch or --mindmap")

    if args.mindmap and not args.titled and not args.new_name:
        parser.error("Must specify --new-name or --titled with --mindmap")

    # Load index if provided
    index_store = None
    if args.index:
        index_store = create_index_store(args.index)
        print(f"Loaded index with {index_store.count()} entries")

    # Batch mode
    if args.batch:
        if not args.batch.is_dir():
            print(f"Error: Not a directory: {args.batch}", file=sys.stderr)
            sys.exit(1)

        renamed, updated, links = batch_rename_to_titled(
            args.batch,
            dry_run=args.dry_run,
            verbose=args.verbose,
            max_title_length=args.max_title,
            keep_id=args.id_prefix
        )

        action = "Would rename" if args.dry_run else "Renamed"
        print(f"\n{action} {renamed} files")
        print(f"Updated {links} links in {updated} files")

    # Single file mode
    else:
        if args.titled:
            new_name = generate_titled_filename(args.mindmap, args.max_title, keep_id=args.id_prefix)
        else:
            new_name = args.new_name

        success, files_updated, links_updated = rename_mindmap(
            args.mindmap,
            new_name,
            search_dir=args.search_dir,
            index_store=index_store,
            dry_run=args.dry_run,
            verbose=args.verbose
        )

        if success:
            action = "Would update" if args.dry_run else "Updated"
            print(f"\n{action} {links_updated} links in {files_updated} files")
        else:
            sys.exit(1)


if __name__ == '__main__':
    main()
