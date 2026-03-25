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
