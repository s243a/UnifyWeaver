# GitHub Pages Deployment - UnifyWeaver

**Created:** 2025-11-05
**Purpose:** Static site content for https://s243a.github.io/UnifyWeaver/

---

## Contents

This directory contains a snapshot of UnifyWeaver documentation prepared for GitHub Pages:

```
gh-pages/
├── index.md                          # Landing page
├── project/
│   ├── README.md                     # Main project README
│   └── docs/                         # All documentation from docs/
└── education/
    ├── README.md                     # Education project overview
    ├── book-1-core-bash/             # Complete Book 1
    ├── book-2-csharp-target/         # Complete Book 2
    └── book-misc/                    # NEW: Emerging features
```

---

## Deployment Steps

### Option 1: Manual Git Commands

```bash
# 1. Navigate to main project
cd /path/to/UnifyWeaver

# 2. Create orphan gh-pages branch (one-time)
git checkout --orphan gh-pages
git rm -rf .

# 3. Copy gh-pages content
cp -r context/claude/gh-pages/* .

# 4. Commit and push
git add .
git commit -m "Deploy UnifyWeaver documentation to GitHub Pages

Includes:
- Landing page with project overview
- Project README and docs/
- Education: Books 1, 2, and book-misc
- All markdown converted for Jekyll rendering

Site will be available at: https://s243a.github.io/UnifyWeaver/
"
git push -u origin gh-pages --force

# 5. Return to main branch
git checkout main
```

### Option 2: GitHub Web Interface

1. Go to https://github.com/s243a/UnifyWeaver/settings/pages
2. Under "Source", select branch: `gh-pages` (after pushing)
3. Click "Save"
4. Site will be live at https://s243a.github.io/UnifyWeaver/ in ~1 minute

---

## File Structure

### Landing Page (index.md)
- SEO-friendly title and description
- Quick links to all sections
- Example code snippet
- Use cases and features
- Recent updates section

### Project Section (project/)
- Main README.md (project overview, installation, quick start)
- Complete docs/ folder (all .md files from main repo)
- Links back to GitHub for code examples

### Education Section (education/)
- Overview README
- Book 1: Core Bash Target (14 chapters + appendices)
- Book 2: C# Target (4 chapters)
- Book-Misc: Emerging features (Perl services, Prolog targets)

---

## Link Adjustments

### Internal Links (Within Site)
These work as-is with Jekyll:
- `[Book 1](education/book-1-core-bash/README.md)`
- `[Testing Guide](project/docs/TESTING_GUIDE.md)`

### External Links (Back to Repo)
For code references, use full GitHub URLs:
- Source code: `https://github.com/s243a/UnifyWeaver/tree/main/src`
- Examples: `https://github.com/s243a/UnifyWeaver/tree/main/examples`
- Issues: `https://github.com/s243a/UnifyWeaver/issues`

---

## Jekyll Configuration

GitHub Pages uses Jekyll by default. No additional config needed, but you can optionally add `_config.yml`:

```yaml
# Optional: _config.yml
title: UnifyWeaver
description: Prolog to Multi-Target Code Compiler
theme: jekyll-theme-minimal  # or another GitHub theme
markdown: kramdown
```

---

## Future Updates

### Manual Update Process
1. Update content in main repo
2. Re-copy to `context/claude/gh-pages/`
3. Checkout gh-pages branch
4. Copy updated files
5. Commit and push

### Automated Update (Future)
Add `.github/workflows/pages-deploy.yml`:

```yaml
name: Deploy to GitHub Pages

on:
  push:
    branches: [main]
  workflow_dispatch:

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Prepare gh-pages content
        run: |
          mkdir -p gh-pages/project/docs gh-pages/education
          cp README.md gh-pages/project/
          cp -r docs/* gh-pages/project/docs/
          # ... copy education files ...

      - name: Deploy to gh-pages
        uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./gh-pages
```

---

## Testing Locally

Test the site before deploying:

```bash
# Install Jekyll (if not already installed)
gem install jekyll bundler

# Serve locally
cd context/claude/gh-pages
jekyll serve

# View at: http://localhost:4000
```

---

## Content Checklist

Before deploying, verify:

- [ ] `index.md` has correct project description
- [ ] All internal links work (relative paths)
- [ ] External GitHub links use full URLs
- [ ] Code examples are properly formatted
- [ ] License information is preserved
- [ ] Recent updates section is current
- [ ] All markdown files copied successfully

---

## File Counts

```bash
# Check what was copied
find project -name "*.md" | wc -l        # Project docs
find education -name "*.md" | wc -l      # Education materials
```

---

## Notes

- **SEO:** GitHub Pages sites are indexed by search engines
- **Images:** If markdown references images, copy them too
- **Assets:** Copy any CSS, images, or other assets referenced
- **Size:** Current snapshot is ~2-3MB (all markdown)
- **Update Frequency:** Manual for now, can automate later

---

**Created by:** Claude Code (Sonnet 4.5)
**For:** UnifyWeaver GitHub Pages deployment
