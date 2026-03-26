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

(deftest test-tool-safe-count
  (is (= 0 (tools/safe-count {:args []})))
  (is (= 3 (tools/safe-count {:args ["read" "glob" "grep"]}))))

(deftest test-tool-is-write-op
  (is (true? (tools/is-write-op "write")))
  (is (true? (tools/is-write-op "edit")))
  (is (false? (tools/is-write-op "read"))))

(deftest test-tool-is-bash
  (is (true? (tools/is-bash "bash")))
  (is (false? (tools/is-bash "read"))))

(deftest test-tool-is-edit
  (is (true? (tools/is-edit "edit")))
  (is (false? (tools/is-edit "bash"))))

(deftest test-is-read
  (is (true? (tools/is-read "read")))
  (is (false? (tools/is-read "bash"))))

(deftest test-is-write
  (is (true? (tools/is-write "write")))
  (is (false? (tools/is-write "read"))))
