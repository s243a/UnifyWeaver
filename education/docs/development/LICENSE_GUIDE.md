<!--
SPDX-License-Identifier: MIT AND CC-BY-4.0
Copyright (c) 2025 John William Creighton (s243a)

This documentation is dual-licensed under MIT and CC-BY-4.0.
-->

# Education Licensing Guide

## UnifyWeaver Educational Materials Licensing

This document describes the licensing structure for the UnifyWeaver educational materials project.

---

## TL;DR - Quick Checklist

**For Content Contributors:**
- ✅ All new documentation files (`.md`) get SPDX headers with `MIT AND CC-BY-4.0`
- ✅ All new code examples (`.pl`, `.sh`, `.cs`) get SPDX headers with `MIT OR Apache-2.0`
- ✅ Your contribution = agreement to dual license (checked in PR)
- ✅ Keep all license files: `LICENSE-MIT`, `LICENSE-CC-BY-4.0`

**For Educators Using This Content:**
- ✅ Choose either MIT or CC-BY-4.0 for documentation
- ✅ Choose either MIT or Apache-2.0 for code examples
- ✅ Attribution required under both licenses
- ✅ Free to remix, adapt, and share with attribution

---

## 1. Education-Specific Licensing Structure

### Documentation (Chapters, Guides, Tutorials)

**License:** `MIT AND CC-BY-4.0` (dual-licensed)

**What this means:**
- Users can choose **either** MIT **or** CC-BY-4.0
- Perfect for educational content (CC-BY is designed for this)
- Attribution required under both licenses
- Free commercial use allowed

**SPDX Header for `.md` files:**
```markdown
<!--
SPDX-License-Identifier: MIT AND CC-BY-4.0
Copyright (c) 2025 John William Creighton (s243a)

This documentation is dual-licensed under MIT and CC-BY-4.0.
-->
```

**What qualifies as documentation:**
- ✅ Chapters (all `.md` files in book-1-core-bash/, book-2-csharp-target/)
- ✅ README.md files
- ✅ Tutorials and guides
- ✅ Appendices
- ✅ Case studies
- ✅ Video scripts

### Code Examples

**License:** `MIT OR Apache-2.0` (disjunctive dual-license)

**What this means:**
- Users can choose **either** MIT **or** Apache-2.0
- Same as main UnifyWeaver project
- Patent protection available via Apache-2.0
- Simple permissive option via MIT

**SPDX Header for code files (`.pl`, `.sh`, `.cs`, `.py`):**

**Prolog:**
```prolog
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% This file is part of UnifyWeaver.
% Licensed under either MIT or Apache-2.0 at your option.
```

**Bash:**
```bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2025 John William Creighton (s243a)
#
# This file is part of UnifyWeaver.
# Licensed under either MIT or Apache-2.0 at your option.
```

**C#:**
```csharp
// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2025 John William Creighton (s243a)
//
// This file is part of UnifyWeaver.
// Licensed under either MIT or Apache-2.0 at your option.
```

**What qualifies as code examples:**
- ✅ All `.pl` files (Prolog examples)
- ✅ All `.sh` files (generated bash scripts)
- ✅ All `.cs` files (C# examples)
- ✅ All `.py` files (Python examples)
- ✅ Example scripts in `examples/` directories

---

## 2. Why This Dual Approach?

### Documentation: CC-BY-4.0 is Standard for Educational Content

**Advantages:**
- Specifically designed for educational materials
- Recognized standard in academic/educational communities
- Clear attribution requirements
- Allows adaptations and translations

**Plus MIT as alternative:**
- Simpler for software developers familiar with MIT
- Can be incorporated into software documentation easily

### Code Examples: MIT/Apache-2.0 Matches Main Project

**Advantages:**
- Students can copy code into their projects seamlessly
- No licensing friction when using examples
- Patent protection available (Apache-2.0)
- Maximum compatibility with real-world projects

---

## 3. File Header Requirements

### Adding Headers to New Files

**When creating new documentation:**
1. Add the CC-BY-4.0 + MIT header at the very top
2. Include copyright with your name (if you're the author)
3. Brief explanation of dual licensing

**When creating new code examples:**
1. Add the MIT OR Apache-2.0 header
2. Include copyright with your name (if you're the author)
3. Brief explanation of dual licensing

### Files with Multiple Contributors

If multiple people contribute to a file:

```markdown
<!--
SPDX-License-Identifier: MIT AND CC-BY-4.0
Copyright (c) 2025 John William Creighton (s243a)
Copyright (c) 2025 Contributor Name

This documentation is dual-licensed under MIT and CC-BY-4.0.
-->
```

**Guidelines:**
- Original author listed first
- Add contributors making "substantial" contributions
- Substantial = new sections, major rewrites (not just typo fixes)

---

## 4. Contribution Process

### Automatic Agreement

The `CONTRIBUTING.md` states:

> "By submitting a pull request, you agree to license your contribution as follows:
> - **Documentation contributions:** Dual-licensed under MIT and CC-BY-4.0
> - **Code examples:** Dual-licensed under MIT and Apache-2.0"

**This means:**
- PR submission = legal agreement
- No separate paperwork needed
- Just add appropriate SPDX headers

### For Contributors Creating New Files

**New documentation file:**
```markdown
<!--
SPDX-License-Identifier: MIT AND CC-BY-4.0
Copyright (c) 2025 YourName

This documentation is dual-licensed under MIT and CC-BY-4.0.
-->

# Your Chapter Title
```

**New code example:**
```prolog
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 YourName
%
% This example is part of UnifyWeaver educational materials.
% Licensed under either MIT or Apache-2.0 at your option.
```

---

## 5. Using Third-Party Educational Content

### Compatible Licenses for Documentation

You **can** include documentation/content from:

- ✅ CC-BY (any version)
- ✅ CC-BY-SA (with compatibility note)
- ✅ MIT
- ✅ Apache-2.0
- ✅ BSD
- ✅ Public Domain / CC0

### Incompatible Licenses

You **cannot** include content from:

- ❌ CC-BY-NC (Non-Commercial restriction)
- ❌ CC-BY-ND (No Derivatives restriction)
- ❌ Proprietary/All Rights Reserved
- ❌ Content without clear licensing

### How to Attribute Third-Party Content

When incorporating external educational content:

**In the document:**
```markdown
## Section Title

> This section adapted from [Source Name](https://example.com)
> by Original Author, licensed under CC-BY-4.0.
```

**In THIRD_PARTY_CONTENT.md:**
```markdown
# Third-Party Educational Content

## Chapter X: Topic Name
- Original Author: Author Name
- Source: https://example.com/original
- License: CC-BY-4.0
- Modifications: Adapted for UnifyWeaver context, updated examples
```

---

## 6. Repository Requirements

### Required License Files

The education folder must always contain:

1. **`LICENSE-MIT`** - Full MIT license text
2. **`LICENSE-CC-BY-4.0`** - Full CC-BY-4.0 license text
3. **`README.md`** - Must include licensing section
4. **`CONTRIBUTING.md`** - Must state contribution terms

### README.md Licensing Section

Must include:

```markdown
## License

This educational project uses different licenses for different content types:

### Educational Content (Documentation)

All chapters, guides, and documentation files (`.md`) are dual-licensed under:

* **MIT License** ([LICENSE-MIT](LICENSE-MIT))
* **Creative Commons Attribution 4.0 International** ([LICENSE-CC-BY-4.0](LICENSE-CC-BY-4.0))

**SPDX:** `MIT AND CC-BY-4.0`

### Code Examples

All code example files (`.pl`, `.sh`, `.cs`, etc.) are dual-licensed under:

* **MIT License** ([LICENSE-MIT](LICENSE-MIT))
* **Apache License 2.0** (see main project LICENSE-APACHE)

**SPDX:** `MIT OR Apache-2.0`

### Copyright

Copyright (c) 2025 John William Creighton (s243a)
```

---

## 7. Common Scenarios

### Scenario 1: Teacher Wants to Use Chapters in Course

**Question:** Can a teacher use chapters in their university course?

**Answer:** Yes! Under CC-BY-4.0:
- ✅ Can print and distribute to students
- ✅ Can adapt content for their needs
- ✅ Can translate to other languages
- ✅ Can use commercially (paid courses OK)
- ✅ Must provide attribution

**Attribution example:**
> "Based on UnifyWeaver Educational Resources by John William Creighton,
> licensed under CC-BY-4.0. Available at https://github.com/s243a/UnifyWeaver"

### Scenario 2: Student Wants to Use Code in Project

**Question:** Can a student copy code examples into their MIT-licensed project?

**Answer:** Yes! Code examples are dual-licensed MIT OR Apache-2.0:
- Student chooses MIT (since their project is MIT)
- Copy code with attribution
- No licensing conflict

### Scenario 3: Company Wants to Create Internal Training

**Question:** Can a company use the materials for internal employee training?

**Answer:** Yes! Both licenses allow commercial use:
- Company can print, adapt, distribute internally
- Must provide attribution
- Can choose MIT or CC-BY-4.0 (company's choice)

### Scenario 4: Someone Wants to Translate Chapters

**Question:** Can someone translate chapters to another language?

**Answer:** Yes! This is an "adaptation" under CC-BY-4.0:
- ✅ Translation allowed
- ✅ Must credit original author
- ✅ Should state it's a translation
- ✅ Can share the translation

**Proper attribution:**
> "Translated from UnifyWeaver Educational Resources by John William
> Creighton, licensed under CC-BY-4.0"

### Scenario 5: Blog Post Using Examples

**Question:** Can someone write a blog post using code examples and explanations?

**Answer:** Yes!
- Code examples: Copy under MIT or Apache-2.0
- Explanations: Quote with attribution under CC-BY-4.0
- Link back to original material
- Must credit author

---

## 8. Annual Maintenance Checklist

Once per year, verify:

- [ ] `LICENSE-MIT` and `LICENSE-CC-BY-4.0` in education/ root
- [ ] README.md has correct licensing section
- [ ] CONTRIBUTING.md has contribution terms
- [ ] New files from past year have SPDX headers
- [ ] Any third-party content documented in THIRD_PARTY_CONTENT.md

**Time required:** ~10 minutes annually

---

## 9. Getting Help

### Resources

- **Creative Commons:** https://creativecommons.org/licenses/by/4.0/
- **SPDX License List:** https://spdx.org/licenses/
- **MIT License:** https://opensource.org/licenses/MIT
- **Apache-2.0 License:** https://www.apache.org/licenses/LICENSE-2.0

### Questions?

For licensing questions specific to educational use:
- Open an issue at https://github.com/s243a/UnifyWeaver/issues
- Tag with `question` and `licensing`

---

## Summary

Educational materials use **simple, permissive dual licensing**:

✅ **Documentation:** MIT AND CC-BY-4.0
- Standard for educational content
- Attribution required
- Free to use, adapt, share

✅ **Code Examples:** MIT OR Apache-2.0
- Matches main project
- Maximum compatibility
- Students can copy freely

✅ **Low Maintenance:**
- Add SPDX headers to new files
- PR acceptance = agreement
- No CLAs or paperwork

This maximizes educational impact while protecting contributor rights.
