#!/usr/bin/env python3
"""
Add navigation footers to chapter markdown files.

This script:
1. Scans each book directory for numbered chapter files (##_*.md)
2. Extracts chapter titles from the first H1 heading
3. Adds/updates navigation footer with Previous/Next links

Usage:
    python3 add_navigation.py [--dry-run]
"""

import os
import re
import sys
from pathlib import Path

# Book display names (for "Back to Book X" links)
BOOK_NAMES = {
    'book-01-foundations': 'Book 1: Foundations',
    'book-02-bash-target': 'Book 2: Bash Target',
    'book-03-csharp-target': 'Book 3: C# Target',
    'book-04-workflows': 'Book 4: Workflows',
    'book-05-python-target': 'Book 5: Python Target',
    'book-06-go-target': 'Book 6: Go Target',
    'book-07-cross-target-glue': 'Book 7: Cross-Target Glue',
    'book-08-security-firewall': 'Book 8: Security & Firewall',
    'book-09-rust-target': 'Book 9: Rust Target',
    'book-10-sql-target': 'Book 10: SQL Target',
    'book-11-prolog-target': 'Book 11: Prolog Target',
    'book-12-powershell-target': 'Book 12: PowerShell Target',
    'book-13-semantic-search': 'Book 13: Semantic Search',
    'book-awk-target': 'AWK Target (Supplementary)',
}

# Book sequence for "next book" links (in reading order)
BOOK_SEQUENCE = [
    'book-01-foundations',
    'book-02-bash-target',
    'book-03-csharp-target',
    'book-04-workflows',
    'book-05-python-target',
    'book-06-go-target',
    'book-07-cross-target-glue',
    'book-08-security-firewall',
    'book-09-rust-target',
    'book-10-sql-target',
    'book-11-prolog-target',
    'book-12-powershell-target',
    'book-13-semantic-search',
    # book-awk-target is supplementary, not in main sequence
]

def get_next_book(current_book_key):
    """Get the next book in sequence, or None if last."""
    try:
        idx = BOOK_SEQUENCE.index(current_book_key)
        if idx < len(BOOK_SEQUENCE) - 1:
            next_key = BOOK_SEQUENCE[idx + 1]
            return (next_key, BOOK_NAMES.get(next_key, next_key))
    except ValueError:
        pass
    return None

def get_chapter_title(filepath):
    """Extract chapter title from first H1 heading."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        # Look for first # heading (not ##)
        match = re.search(r'^#\s+(.+?)$', content, re.MULTILINE)
        if match:
            return match.group(1).strip()
    except Exception as e:
        print(f"  Warning: Could not read {filepath}: {e}")
    return None

def get_numbered_chapters(book_dir):
    """Get list of numbered chapter files in order."""
    chapters = []
    for f in sorted(os.listdir(book_dir)):
        if re.match(r'^\d{2}_.*\.md$', f) and f != 'README.md':
            chapters.append(f)
    return chapters

def strip_existing_navigation(content):
    """Remove existing navigation footer if present."""
    # Pattern to match navigation section at end of file
    # Matches: ---\n\n## Navigation\n\n...[links]...
    pattern = r'\n---\s*\n+## Navigation\s*\n+.*$'
    content = re.sub(pattern, '', content, flags=re.DOTALL)
    return content.rstrip()

def create_navigation_footer(book_dir, book_name, book_key, chapters, current_idx):
    """Create navigation footer for a chapter."""
    current_file = chapters[current_idx]
    current_base = current_file.replace('.md', '')
    is_last_chapter = current_idx == len(chapters) - 1

    parts = []

    # Previous link
    if current_idx > 0:
        prev_file = chapters[current_idx - 1]
        prev_base = prev_file.replace('.md', '')
        prev_title = get_chapter_title(os.path.join(book_dir, prev_file))
        if prev_title:
            # Shorten title if needed
            short_title = prev_title[:50] + '...' if len(prev_title) > 50 else prev_title
            parts.append(f'**←** [Previous: {short_title}]({prev_base})')
        else:
            parts.append(f'**←** [Previous]({prev_base})')

    # Back to book link (use ./ instead of README for Jekyll compatibility)
    parts.append(f'[📖 {book_name}](./)')

    # Next link - either next chapter or next book
    if current_idx < len(chapters) - 1:
        next_file = chapters[current_idx + 1]
        next_base = next_file.replace('.md', '')
        next_title = get_chapter_title(os.path.join(book_dir, next_file))
        if next_title:
            short_title = next_title[:50] + '...' if len(next_title) > 50 else next_title
            parts.append(f'[Next: {short_title} →]({next_base})')
        else:
            parts.append(f'[Next →]({next_base})')
    else:
        # Last chapter - link to next book or all books
        next_book = get_next_book(book_key)
        if next_book:
            next_book_key, next_book_name = next_book
            parts.append(f'[Next: {next_book_name} →](../{next_book_key}/)')
        else:
            # Last book in sequence - link to all books
            parts.append(f'[📚 All Books →](../)')

    nav_line = ' | '.join(parts)
    return f'\n\n---\n\n## Navigation\n\n{nav_line}\n'

def process_book(book_dir, dry_run=False):
    """Process all chapters in a book directory."""
    book_key = os.path.basename(book_dir)
    book_name = BOOK_NAMES.get(book_key, book_key)

    print(f"\nProcessing {book_name} ({book_dir})")

    chapters = get_numbered_chapters(book_dir)
    if not chapters:
        print(f"  No numbered chapters found")
        return 0

    print(f"  Found {len(chapters)} chapters: {', '.join(chapters)}")

    updated = 0
    for idx, chapter_file in enumerate(chapters):
        filepath = os.path.join(book_dir, chapter_file)

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            print(f"  Error reading {chapter_file}: {e}")
            continue

        # Strip existing navigation
        content = strip_existing_navigation(content)

        # Create new navigation footer
        nav_footer = create_navigation_footer(book_dir, book_name, book_key, chapters, idx)

        # Add navigation
        new_content = content + nav_footer

        if dry_run:
            print(f"  [DRY RUN] Would update {chapter_file}")
        else:
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                print(f"  Updated {chapter_file}")
                updated += 1
            except Exception as e:
                print(f"  Error writing {chapter_file}: {e}")

    return updated

def main():
    dry_run = '--dry-run' in sys.argv

    if dry_run:
        print("=== DRY RUN MODE - No files will be modified ===\n")

    # Find education directory
    script_dir = Path(__file__).parent
    education_dir = script_dir.parent

    if not education_dir.exists():
        print(f"Error: Education directory not found at {education_dir}")
        sys.exit(1)

    print(f"Education directory: {education_dir}")

    total_updated = 0

    # Process each book directory
    for item in sorted(os.listdir(education_dir)):
        item_path = os.path.join(education_dir, item)
        if os.path.isdir(item_path) and item.startswith('book-'):
            updated = process_book(item_path, dry_run)
            total_updated += updated

    print(f"\n{'Would update' if dry_run else 'Updated'} {total_updated} chapter files total")

if __name__ == '__main__':
    main()
