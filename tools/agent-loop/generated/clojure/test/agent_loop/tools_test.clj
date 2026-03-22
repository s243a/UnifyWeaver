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
