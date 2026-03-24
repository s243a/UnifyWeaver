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
