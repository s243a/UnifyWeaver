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
