(ns agent-loop.sessions-test
  (:require [clojure.test :refer [deftest is testing]]
            [agent-loop.sessions :as sessions]))

(deftest test-session-filename
  (is (= "abc.json" (sessions/session-filename "abc"))))

(deftest test-session-is-valid-id
  (is (true? (sessions/session-is-valid-id "abc123")))
  (is (false? (sessions/session-is-valid-id ""))))

(deftest test-session-has-metadata
  (is (true? (sessions/has-metadata {"metadata" {:created "now"}})))
  (is (false? (sessions/has-metadata {"messages" []}))))

(deftest test-session-is-expired
  (is (true? (sessions/is-expired 7200.0 3600.0)))
  (is (false? (sessions/is-expired 1800.0 3600.0))))

(deftest test-session-age
  (is (= 100.0 (sessions/session-age 1000.0 1100.0)))
  (is (= 0.0 (sessions/session-age 500.0 500.0))))

(deftest test-session-count
  (is (= 0 (sessions/session-count [])))
  (is (= 2 (sessions/session-count ["a.json" "b.json" "readme.txt"]))))

(deftest test-session-dir
  (is (string? (sessions/session-dir {:sessions-dir "/tmp/sessions"}))))

(deftest test-session-is-recent
  (is (true? (sessions/is-recent 100.0 105.0 10.0)))
  (is (false? (sessions/is-recent 100.0 200.0 10.0))))

(deftest test-json-path
  (is (= "/tmp/sessions/abc.json" (sessions/json-path "/tmp/sessions" "abc"))))

(deftest test-name-valid
  (is (true? (sessions/name-valid "my-session")))
  (is (false? (sessions/name-valid "")))
  (is (false? (sessions/name-valid "   "))))

(deftest test-session-id-is-long
  (is (true? (sessions/id-is-long "sess-abc123" 5)))
  (is (false? (sessions/id-is-long "ab" 5))))
