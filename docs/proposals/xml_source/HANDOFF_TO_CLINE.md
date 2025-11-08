# Handoff to cline.bot - XML Source Test Failure

**Date:** 2025-10-30
**From:** Gemini
**To:** cline.bot

## 1. The Goal

The primary goal is to fix the persistent test failure for the `xml_source.pl` plugin. The specific test that is failing is the `basic_extraction` test in `tests/core/test_xml_source.pl`.

## 2. The Error

The test fails with the following error:

```
read_util:read_stream_to_codes/2: Arguments are not sufficiently instantiated
```

This error occurs when running the plunit test suite for `xml_source.pl`. It seems to be related to stream handling within the test runner, but the root cause is likely a subtle issue in the `xml_source.pl` file itself.

## 3. What I Have Tried

I have tried numerous approaches to debug this issue, including:

*   **Simplifying the test case:** I have simplified the test case to a single call to `xml_source:compile_source/4`, but the error persists.
*   **Checking for syntax errors:** I have checked for and fixed several syntax errors in `xml_source.pl`.
*   **Verifying file updates:** I have confirmed that the test file is being updated correctly.
*   **Creating a new test file:** I created a new test file with a different name to rule out caching issues, but the error remained.
*   **Investigating side effects:** I investigated the `initialization/2` directive and temporarily removed `format/2` calls from the `dynamic_source_compiler.pl` file, but this did not solve the problem.
*   **Different test commands:** I have tried various ways to run the `swipl` command, including using the full path, running from different directories, and using different command-line options.
*   **Error handling:** I have added error handling to the test case to capture stdout and stderr from the executed script, but the error occurs before the script is even executed.

## 4. The Current State

*   The `xml_source.pl` file is located at `src/unifyweaver/sources/xml_source.pl`.
*   The test file is `tests/core/test_xml_source.pl`.
*   A simple test case in `tests/core/test_simple.pl` passes, which suggests that the test environment is likely correct.
*   The current hypothesis is that there is a subtle issue in `xml_source.pl` that is causing the plunit test runner to fail in an unexpected way.

## 5. The Request to `cline.bot`

Your task is to focus solely on fixing the `read_util:read_stream_to_codes/2: Arguments are not sufficiently instantiated` error and getting the `basic_extraction` test in `test_xml_source.pl` to pass.

Please do not add any new features or refactor the code beyond what is necessary to fix the test.

Here are the relevant files:

*   `src/unifyweaver/sources/xml_source.pl`
*   `tests/core/test_xml_source.pl`
*   `tests/test_data/sample.rdf`

Thank you for your help.
