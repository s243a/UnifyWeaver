(ns agent-loop.tools-test
  (:require [clojure.test :refer [deftest is testing]]
            [agent-loop.tools :as tools]))

(deftest test-is-destructive
  (is (tools/is-destructive "bash"))
  (is (tools/is-destructive "write"))
  (is (not (tools/is-destructive "read"))))

(deftest test-is-mcp-tool
  (is (true? (tools/is-mcp-tool "mcp:server/tool")))
  (is (false? (tools/is-mcp-tool "read"))))

(deftest test-is-builtin
  (is (true? (tools/is-builtin "read")))
  (is (true? (tools/is-builtin "bash")))
  (is (false? (tools/is-builtin "mcp__server__tool"))))

(deftest test-needs-path-validation
  (is (true? (tools/needs-path-validation "read")))
  (is (true? (tools/needs-path-validation "write")))
  (is (true? (tools/needs-path-validation "edit")))
  (is (false? (tools/needs-path-validation "bash"))))

(deftest test-tool-category
  (is (string? (tools/tool-category "bash"))))

(deftest test-has-schema
  (is (true? (tools/has-schema {:schema {:type "object"}} "bash")))
  (is (false? (tools/has-schema {:schema {}} "bash"))))

(deftest test-is-safe
  (is (true? (tools/is-safe "read")))
  (is (false? (tools/is-safe "bash"))))

(deftest test-is-mcp-prefixed
  (is (true? (tools/is-mcp-prefixed "mcp:read")))
  (is (false? (tools/is-mcp-prefixed "bash"))))

(deftest test-name-is-valid
  (is (true? (tools/name-is-valid "bash")))
  (is (false? (tools/name-is-valid ""))))

(deftest test-is-readonly
  (is (true? (tools/is-readonly "read")))
  (is (true? (tools/is-readonly "glob")))
  (is (false? (tools/is-readonly "bash"))))

(deftest test-tool-count
  (is (= 0 (tools/tool-count {:schema {}})))
  (is (= 2 (tools/tool-count {:schema {"bash" {} "read" {}}})))
  (is (= 3 (tools/tool-count {:schema {"a" {} "b" {} "c" {}}}))))

(deftest test-requires-confirm
  (is (true? (tools/requires-confirm "bash")))
  (is (true? (tools/requires-confirm "write")))
  (is (false? (tools/requires-confirm "read"))))
