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
