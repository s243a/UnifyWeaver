(ns agent-loop.output-parser-test
  (:require [clojure.test :refer [deftest is testing]]
            [agent-loop.output-parser :as parser]))

(deftest test-is-json-object
  (is (true? (parser/is-json-object "{:key 1}")))
  (is (false? (parser/is-json-object "hello"))))

(deftest test-is-json-content
  (is (true? (parser/is-json-content "{:key 1}")))
  (is (true? (parser/is-json-content "[1, 2, 3]")))
  (is (false? (parser/is-json-content "hello"))))

(deftest test-content-length
  (is (= 5 (parser/content-length "hello")))
  (is (= 0 (parser/content-length ""))))

(deftest test-strip-content
  (is (= "hello" (parser/strip-content "  hello  "))))

(deftest test-is-empty-response
  (is (true? (parser/is-empty-response "")))
  (is (true? (parser/is-empty-response "   ")))
  (is (false? (parser/is-empty-response "hello"))))

(deftest test-content-exceeds-length
  (is (true? (parser/content-exceeds-length "hello world" 5)))
  (is (false? (parser/content-exceeds-length "hi" 5))))

(deftest test-content-is-json
  (is (true? (parser/content-is-json "{key: 1}")))
  (is (true? (parser/content-is-json "[1,2,3]")))
  (is (false? (parser/content-is-json "hello"))))

(deftest test-content-preview
  (is (= "hi" (parser/content-preview "hi" 10)))
  (is (string? (parser/content-preview "hello world this is long" 5))))

(deftest test-is-multiline
  (is (true? (parser/is-multiline "hello world this is long" 5)))
  (is (false? (parser/is-multiline "hi" 10))))

(deftest test-output-has-code-block
  (is (true? (parser/has-code-block "```python\nprint(1)\n```")))
  (is (false? (parser/has-code-block "no code here"))))

(deftest test-output-is-short
  (is (true? (parser/is-short "hi" 10)))
  (is (false? (parser/is-short "hello world foo bar" 5))))

(deftest test-output-is-blank
  (is (true? (parser/is-blank "")))
  (is (true? (parser/is-blank "   ")))
  (is (false? (parser/is-blank "hello"))))

(deftest test-output-has-json-array
  (is (true? (parser/has-json-array "[1,2]")))
  (is (false? (parser/has-json-array "{}"))))

(deftest test-is-markdown
  (is (true? (parser/is-markdown "# heading")))
  (is (false? (parser/is-markdown "plain"))))

(deftest test-exceeds-limit
  (is (true? (parser/exceeds-limit "hello world" 5)))
  (is (false? (parser/exceeds-limit "hi" 10))))
