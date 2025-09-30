<!--
SPDX-License-Identifier: MIT OR Apache-2.0
Copyright (c) 2025 John William Creighton (@s243a)
-->
# License Administration Guide

## UnifyWeaver Dual Disjunctive Licensing (MIT OR Apache-2.0)

This document describes the administrative responsibilities for maintaining UnifyWeaver's dual disjunctive licensing scheme.

---

## TL;DR - Quick Checklist

**For Maintainers:**
- ✅ All new core source code files get SPDX headers (e.g. .pl, .py and .sh)
- ✅ Directories that contain files without SPDX headers (e.g. docs, papers) must have a README.md that contains the default licensing for the directory
- ✅ PR contributions must agree to dual licensing (checked in PR template)
- ✅ Keep both `LICENSE-MIT` and `LICENSE-APACHE` in repo root
- ✅ Third-party code: check compatibility, document clearly

**For Contributors:**
- ✅ Add SPDX header to new source code files
- ✅ Your PR submission = agreement to dual license your contribution
- ✅ No additional CLAs or paperwork required

---

## 1. Core Principles

### What "MIT OR Apache-2.0" Means

**Outbound (What Users Get):**
- Users can choose **either** MIT **or** Apache-2.0
- They comply with only one license of their choice
- Forks can pick a single license (See section 6)

**Inbound (What We Require):**
- Contributors must license under **both** MIT **and** Apache-2.0
- This preserves dual-licensing for the main repository
- Ensures all code has patent grants (from Apache-2.0)

### Why This Works
- **Maximum Compatibility**: Users pick the license that fits their needs
- **Patent Protection**: Apache-2.0 provides explicit patent grants
- **Simple Administration**: No CLAs, just PR acceptance = agreement

---

## 2. File Header Requirements

### Source Code Files (Core Files)

**All source code and script files get SPDX headers:**

**Prolog files (`.pl`):**
```prolog
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 s243a

:- encoding(utf8).
```

**Bash scripts (`.sh`):**
```bash
#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2025 s243a
```

**Python files (`.py`):**
```python
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2025 s243a
```

**What qualifies as "source code":**
- ✅ Programming language files (.pl, .py, .sh, .c, .js, etc.)
- ✅ Build scripts (Makefile, etc.)
- ✅ Configuration files that are code-like (.yml, .json for CI/CD)

### User Documentation (Project-Level)

**Project documentation gets SPDX headers:**

```markdown
<!--
SPDX-License-Identifier: MIT OR Apache-2.0
Copyright (c) 2025 s243a
-->

# Document Title
```

**What qualifies:**
- ✅ README.md (root and major directories)
- ✅ ARCHITECTURE.md
- ✅ CONTRIBUTING.md
- ✅ API documentation
- ✅ Tutorials and guides
- ✅ How-to documents

### Special Cases: Academic Papers and Research Documents

**Academic papers and formal research documents do NOT get SPDX headers in the files themselves.**

**Instead, use directory-level licensing:**

```
papers/
├── README.md                    ← SPDX header HERE
├── LICENSE-CC-BY                ← License text
└── state_machines/
    ├── paper.tex                ← NO SPDX header
    ├── paper.bib                ← NO SPDX header
    ├── paper.pdf                ← NO SPDX header
    └── figures/                 ← NO SPDX headers
        └── diagram.png
```

**Example papers/README.md:**
```markdown
<!--
SPDX-License-Identifier: CC-BY-4.0
Copyright (c) 2025 s243a
-->

# UnifyWeaver Research Papers

All papers in this directory are licensed under CC-BY-4.0 
unless otherwise noted.

## Published Papers

- **Parameterized State Machines** (2025) - `state_machines/`
  - Published in: Journal of Logic Programming
  - DOI: 10.xxxx/xxxxx
```

**Why papers are different:**
- Academic papers must follow strict journal formatting
- SPDX headers would violate journal style guidelines  
- Papers remain in journal-submission format
- Licensing is declared at directory level
- No header stripping or modification needed

### Files with Multiple Contributors

As the project grows and others contribute significant changes:

```prolog
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 s243a
% Copyright (c) 2025 Contributor Name
```

**Rules:**
- Original author (you) always listed first
- Add contributors who make "substantial" contributions to a file
- "Substantial" = major refactoring, new functionality, not just typo fixes
- When in doubt, it's fine to add them (being generous is good)

### For New Files

Contributors creating entirely new files should use:

```prolog
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 Contributor Name
```

You (as project maintainer) don't claim copyright on files you didn't write.

---

## 3. Pull Request Process

### Automatic Agreement

The `CONTRIBUTING.md` file states:

> "By submitting a pull request, you agree to license your contribution under both the MIT and Apache-2.0 licenses."

**This means:**
- PR submission = legal agreement
- No separate CLA signing needed
- No paperwork beyond the PR itself

### PR Review Checklist

When reviewing PRs, check:

1. **SPDX Headers Present**
   - [ ] New **source code files** have correct SPDX headers
   - [ ] Copyright line appropriate (contributor's name on new files)
   - [ ] Documentation files in new directories have README.md with licensing

2. **No Licensing Conflicts**
   - [ ] No GPL code (incompatible with Apache-2.0 → MIT dual licensing)
   - [ ] No proprietary code
   - [ ] Third-party code properly documented (see Section 4)

3. **Standard Code Review**
   - [ ] Tests pass
   - [ ] Code quality acceptable
   - [ ] Documentation updated

### Handling Non-Compliance

If a PR is missing SPDX headers:

**Response Template:**
```markdown
Thanks for the contribution! Before we can merge, please add the SPDX 
license header to [filename]:

For Prolog files:
```prolog
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 YourName
```

For bash files:
```bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2025 YourName
```

This confirms you're licensing your contribution under both MIT and 
Apache-2.0 as described in CONTRIBUTING.md.
```

---

## 4. Third-Party Code Integration

### Important Distinction: Including vs. Using

**Including code** (copying into your repository):
- ❌ Cannot include GPL code
- This makes your code a derivative work

**Using/Calling GPL programs** (as external tools):
- ✅ Can call GPL programs like `grep`, `awk`, `sed`
- ✅ Can document that users need GPL tools installed
- These are separate programs, not part of your code

**Importing GPL libraries** (linking/loading):
- ❌ Cannot import GPL Prolog libraries in your compiler
- Dynamic linking creates derivative work under GPL
- Static linking definitely creates derivative work

### Compatible Licenses (Can Include Code)

You **can** include code from these licenses:

- ✅ MIT
- ✅ Apache-2.0
- ✅ BSD (2-clause, 3-clause)
- ✅ ISC
- ✅ CC0 / Public Domain (with care)

### Incompatible Licenses (Cannot Include Code)

You **cannot** include code from:

- ❌ GPL (any version) - copyleft conflicts with disjunctive licensing
- ❌ LGPL - same issue (though some LGPL use might be OK via dynamic linking)
- ❌ AGPL - same issue
- ❌ Proprietary/All Rights Reserved
- ❌ Creative Commons Non-Commercial (CC-BY-NC)

### Real-World Examples for UnifyWeaver

**✅ Allowed:**
```bash
# Generated bash script calling GPL utilities
grep "pattern" file.txt        # grep is GPL - this is fine
awk '{print $1}' data.csv      # awk is GPL - this is fine
```

**✅ Allowed:**
```markdown
# In README.md documentation
## Requirements
- bash 4.0+
- grep (GNU grep recommended)
- awk (GNU awk or compatible)
```

**❌ Not Allowed:**
```prolog
% In your Prolog compiler code
:- use_module(library(gpl_licensed_library)).  % Cannot do this
```

**❌ Not Allowed:**
```prolog
% Copying GPL Prolog code into your repository
% Even with attribution, this makes your code GPL
```

### Why This Matters

**The GPL's copyleft provision** means:
- If you include GPL code, your entire project must be GPL
- This eliminates users' choice between MIT and Apache-2.0
- Your dual licensing becomes impossible

**But calling GPL tools is fine** because:
- They run as separate processes
- Your code doesn't become a derivative work
- It's like using a GPL text editor to write MIT code

### How to Include Third-Party Code

When incorporating external code:

1. **Create a `THIRD_PARTY_LICENSES.md` file**
   ```markdown
   # Third-Party Licenses
   
   ## library_name (version)
   - License: MIT
   - Source: https://github.com/author/library
   - Files: src/unifyweaver/external/library_name.pl
   
   [Include full license text]
   ```

2. **Mark the files clearly**
   ```prolog
   % SPDX-License-Identifier: MIT
   % Copyright (c) 2023 Original Author
   % 
   % This file is from [library_name] (https://...)
   % Included under the MIT license (compatible with UnifyWeaver)
   ```

3. **Keep original copyright notices**
   - Never remove or modify copyright notices from third-party code
   - This is required by both MIT and Apache-2.0

### Public Domain / CC0 Code

If using public domain code:

```prolog
% SPDX-License-Identifier: CC0-1.0 OR MIT OR Apache-2.0
% Originally public domain by Original Author
% Relicensed under MIT OR Apache-2.0 by UnifyWeaver project
```

This makes it clear the code is now part of UnifyWeaver's licensing.

---

## 5. Repository Requirements

### Must-Have Files

These files must always be in the repository root:

1. **`LICENSE-MIT`** - Full MIT license text
2. **`LICENSE-APACHE`** - Full Apache-2.0 license text
3. **`README.md`** - Must include licensing section
4. **`CONTRIBUTING.md`** - Must state contribution licensing terms

### README.md Licensing Section

Must always include:

```markdown
## License

Licensed under either of:

* Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or 
  http://www.apache.org/licenses/LICENSE-2.0)
* MIT license ([LICENSE-MIT](LICENSE-MIT) or 
  http://opensource.org/licenses/MIT)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally 
submitted for inclusion in the work by you, as defined in the Apache-2.0 
license, shall be dual licensed as above, without any additional terms 
or conditions.
```

**Never remove or modify** the contribution paragraph - it's the legal basis for accepting contributions.

---

## 6. Common Scenarios

### Scenario 1: Someone Forks and Picks One License

**Question:** A fork removes `LICENSE-MIT` and only keeps `LICENSE-APACHE`. Is this OK?

**Answer:** Yes! This is the point of disjunctive licensing. Users can:
- Keep both licenses (most common for libraries)
- Pick only MIT (for maximum compatibility)
- Pick only Apache-2.0 (for patent protection)

**Your Action:** Nothing - forks can choose their license strategy.

### Scenario 2: Contributor Wants Different License

**Question:** A contributor says "I want my code under GPL."

**Answer:** Cannot accept. Options:
1. Ask them to reconsider and accept MIT OR Apache-2.0
2. Reject the contribution
3. Don't use their code

**Why:** GPL is copyleft and incompatible with the disjunctive scheme. You can't mix GPL with MIT/Apache-2.0 in a way that preserves the "OR" choice for users.

### Scenario 3: Large Corporate Contribution

**Question:** BigCorp wants to contribute but asks for a CLA.

**Answer:** 
- CLAs are **optional** for you to require
- The CONTRIBUTING.md agreement is sufficient
- If they insist, you can sign their CLA, but you don't need one from them
- Their PR acceptance is their agreement to dual-license

### Scenario 4: AI-Generated Code

**Question:** Someone submits a PR with AI-generated code (GPT, Copilot, etc).

**Answer:** This is a gray area. Be cautious:
1. Ask the contributor to confirm they have rights to submit it
2. Consider if the AI might have reproduced licensed code
3. Check if the code is trivial/common patterns (less concern)
4. When in doubt, reject or ask for human-written alternative

**Conservative approach:** Require contributors to state: "This code was written by me, not generated by AI" for non-trivial contributions.

### Scenario 5: Updating Copyright Years

**Question:** Do I need to update copyright years annually?

**Answer:** Not required but nice to do:
- `Copyright (c) 2025 s243a` - current approach
- `Copyright (c) 2025-2026 s243a` - if you make changes in 2026
- Or leave it as original year - still valid

The copyright year represents first publication, not an expiration date.

### Scenario 6: Academic Papers in Repository

**Question:** How do I license papers I want to publish?

**Answer:** Keep papers out of GitHub until after publication. Then:

1. **After publication:** Add to GitHub with directory-level licensing
2. **Use papers/README.md** to declare license (typically CC-BY-4.0)
3. **Paper files stay clean** - no SPDX headers in .tex/.pdf files
4. **No modification needed** - papers remain in journal format

**Example structure:**
```
papers/
├── README.md              ← License declared here (CC-BY-4.0)
└── state_machines/
    ├── paper.tex          ← Clean, no headers
    └── paper.pdf          ← Clean, no headers
```

---

## 7. Record Keeping (Optional but Recommended)

### Contributors File

Consider maintaining `CONTRIBUTORS.md`:

```markdown
# Contributors

Thank you to everyone who has contributed to UnifyWeaver!

## Core Maintainers
- s243a (@s243a) - Original author and maintainer

## Contributors
- Contributor Name (@username) - Feature XYZ
- Another Person (@handle) - Bug fixes in module ABC

## How to Get Listed
Make a significant contribution! Criteria:
- New features or modules
- Major bug fixes
- Substantial documentation improvements
- Helping with project maintenance
```

**Benefits:**
- Recognizes contributors
- No legal requirement, just good practice
- Makes contributors feel appreciated

### When to Update

Update `CONTRIBUTORS.md` when:
- Accepting a PR with substantial new code
- Someone contributes multiple PRs
- Someone helps with non-code contributions (docs, bug triage)

---

## 8. License Compatibility Reference

### Quick Reference Table

| Their License | Can Include? | Notes |
|--------------|--------------|-------|
| MIT | ✅ Yes | Fully compatible |
| Apache-2.0 | ✅ Yes | Fully compatible |
| BSD-2/3 | ✅ Yes | Fully compatible |
| ISC | ✅ Yes | Very similar to MIT |
| CC0/Public Domain | ✅ Yes | Relicense under MIT OR Apache-2.0 |
| Unlicense | ✅ Yes | Treat like public domain |
| GPLv2 | ❌ No | Copyleft incompatible |
| GPLv3 | ❌ No | Copyleft incompatible |
| LGPL | ❌ No | Copyleft incompatible |
| AGPL | ❌ No | Copyleft incompatible |
| Proprietary | ❌ No | No rights to redistribute |
| "No License" | ❌ No | Assume proprietary |

### When Unsure

If you're unsure about license compatibility:
1. Check [ChooseALicense.com](https://choosealicense.com/appendix/)
2. Ask on [OpenSource.StackExchange.com](https://opensource.stackexchange.com/)
3. When in doubt, **don't include it** or get legal advice

---

## 9. Annual Checklist (Minimal Effort)

Once per year, verify:

- [ ] Both `LICENSE-MIT` and `LICENSE-APACHE` still in repo root
- [ ] README.md still has licensing section
- [ ] CONTRIBUTING.md still has dual-licensing statement
- [ ] New files from the past year have SPDX headers (source code only)
- [ ] Any third-party code is documented in `THIRD_PARTY_LICENSES.md`
- [ ] Directories with non-header files have README.md with licensing

**Time required:** ~15 minutes annually

---

## 10. Getting Help

### Resources

- **SPDX License List:** https://spdx.org/licenses/
- **License Compatibility:** https://www.gnu.org/licenses/license-list.html
- **ChooseALicense.com:** https://choosealicense.com/

### When You Need Legal Advice

This document is **not legal advice**. Consult a lawyer if:
- Large corporation wants to use your code with custom agreement
- Patent concerns arise
- You're considering changing the license
- Complex licensing situation with third-party code
- You receive a cease-and-desist or licensing complaint

---

## Summary

The dual disjunctive licensing is **low maintenance**:

✅ **Easy:**
- Add SPDX headers to new source code files
- Use directory-level README for papers/docs without headers
- Keep both license files in repo
- PR acceptance = contributor agreement

✅ **No Ongoing Work:**
- No CLAs to manage
- No annual renewals
- No registration requirements

✅ **Clear Rules:**
- Accept MIT/Apache-2.0/BSD compatible code
- Reject GPL/proprietary code
- Document third-party code clearly
- Papers use directory-level licensing

This scheme maximizes adoption with minimal administrative burden.