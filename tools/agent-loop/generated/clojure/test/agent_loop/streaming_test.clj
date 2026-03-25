(ns agent-loop.streaming-test
  (:require [clojure.test :refer [deftest is]]
            [agent-loop.streaming :as streaming]))

(deftest test-create-counter
  (let [state (streaming/create-streaming-counter)]
    (is (= 0 (:token-count state)))
    (is (= 0 (:char-count state)))
    (is (false? (:show-live state)))))

(deftest test-is-live
  (is (false? (streaming/is-live {:show-live false})))
  (is (true? (streaming/is-live {:show-live true}))))

(deftest test-is-idle
  (is (true? (streaming/is-idle {:token-count 0})))
  (is (false? (streaming/is-idle {:token-count 5}))))

(deftest test-streaming-char-count
  (is (= 0 (streaming/streaming-char-count {:char-count 0})))
  (is (= 42 (streaming/streaming-char-count {:char-count 42}))))

(deftest test-streaming-token-count
  (is (= 0 (streaming/streaming-token-count {:token-count 0})))
  (is (= 10 (streaming/streaming-token-count {:token-count 10}))))

(deftest test-is-active
  (is (true? (streaming/is-active {:token-count 1 :char-count 5})))
  (is (false? (streaming/is-active {:token-count 0 :char-count 0}))))

(deftest test-chunk-is-complete
  (is (streaming/chunk-is-complete "data: hello"))
  (is (not (streaming/chunk-is-complete "partial"))))

(deftest test-avg-token-rate
  (is (= 0.0 (streaming/avg-token-rate {:token-count 100 :elapsed 0})))
  (is (= 50.0 (streaming/avg-token-rate {:token-count 100 :elapsed 2.0}))))

(deftest test-streaming-is-active
  (is (true? (streaming/is-active {:token-count 5 :char-count 0})))
  (is (false? (streaming/is-active {:token-count 0 :char-count 0}))))

(deftest test-chars-per-token
  (is (= 0.0 (streaming/chars-per-token {:char-count 0 :token-count 0})))
  (is (= 4.0 (streaming/chars-per-token {:char-count 40 :token-count 10}))))
