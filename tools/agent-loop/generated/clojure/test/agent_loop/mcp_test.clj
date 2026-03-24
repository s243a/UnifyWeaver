(ns agent-loop.mcp-test
  (:require [clojure.test :refer [deftest is testing]]
            [agent-loop.mcp :as mcp]))

(deftest test-mcp-is-notification
  (is (true? (mcp/mcp-is-notification "notifications/update")))
  (is (false? (mcp/mcp-is-notification "tools/call"))))

(deftest test-mcp-parse-tool-name
  (is (= "read" (mcp/mcp-parse-tool-name "mcp:read")))
  (is (= "read" (mcp/mcp-parse-tool-name "read"))))

(deftest test-mcp-is-connected
  (is (true? (mcp/is-connected {:request-id 5})))
  (is (false? (mcp/is-connected {:request-id 0}))))
