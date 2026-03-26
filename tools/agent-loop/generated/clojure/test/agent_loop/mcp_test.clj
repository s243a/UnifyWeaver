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

(deftest test-mcp-tool-count
  (is (= 0 (mcp/tool-count {:tools []})))
  (is (= 3 (mcp/tool-count {:tools ["a" "b" "c"]}))))

(deftest test-mcp-has-tools
  (is (false? (mcp/has-tools {:tools []})))
  (is (true? (mcp/has-tools {:tools ["read"]}))))

(deftest test-mcp-server-count
  (is (= 0 (mcp/server-count {:servers []})))
  (is (= 2 (mcp/server-count {:servers ["s1" "s2"]}))))

(deftest test-mcp-disconnect-reason
  (is (= "timeout" (mcp/disconnect-reason {:disconnect-reason "timeout"})))
  (is (nil? (mcp/disconnect-reason {:other "field"}))))

(deftest test-mcp-is-tool-call
  (is (true? (mcp/is-tool-call "tools/execute")))
  (is (false? (mcp/is-tool-call "notifications/ready"))))

(deftest test-mcp-server-name
  (is (= "my-server" (mcp/server-name {:name "my-server"}))))

(deftest test-format-error
  (is (= "error 404: not found" (mcp/format-error 404 "not found"))))

(deftest test-mcp-request-count
  (is (= 0 (mcp/request-count {:request-id 0})))
  (is (= 5 (mcp/request-count {:request-id 5})))
  (is (= 42 (mcp/request-count {:request-id 42}))))

(deftest test-has-clients
  (is (false? (mcp/has-clients {:clients {}})))
  (is (true? (mcp/has-clients {:clients {"srv1" {:name "s1"}}}))))
