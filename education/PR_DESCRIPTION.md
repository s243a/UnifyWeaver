# PR Title: Add "Go Target" Book to Education Materials

## Description

This PR adds a new educational book, **"Go Target & Cross-Platform Compilation"**, to the `UnifyWeaver_Education` repository. This book documents the Go target features of UnifyWeaver, filling a gap in the existing educational materials.

## Changes

-   **New Directory**: `book-go-target/`
    -   `README.md`: Overview and prerequisites.
    -   `01_introduction.md`: Benefits of the Go target (single binary, cross-platform).
    -   `02_basic_compilation.md`: Compiling facts and rules.
    -   `03_advanced_features.md`: Regex matching, constraints, and aggregations.
    -   `04_json_processing.md`: JSON I/O and schema validation.
-   **Updated Documentation**:
    -   `README.md`: Added a link to the new book in the main index.

## Motivation

The UnifyWeaver project has a mature Go target (v0.5), but the educational materials previously only covered Bash and C#. This addition ensures users can learn how to utilize the Go target for high-performance, cross-platform record processing.

## Verification

-   [x] Verified directory structure.
-   [x] Verified markdown content rendering.
-   [x] Verified links in the main README.
