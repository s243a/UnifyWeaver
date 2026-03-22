(ns agent-loop.retry-test
  (:require [clojure.test :refer [deftest is testing]]
            [agent-loop.retry :as retry]))

(deftest test-is-retryable-status
  (is (some? (retry/is-retryable-status 429)))
  (is (some? (retry/is-retryable-status 503)))
  (is (nil? (retry/is-retryable-status 200)))
  (is (nil? (retry/is-retryable-status 404))))

(deftest test-compute-delay
  (let [delay (retry/compute-delay 1.0 2.0 1 60.0)]
    (is (= 1.0 delay)))
  (let [delay (retry/compute-delay 1.0 2.0 3 60.0)]
    (is (= 4.0 delay))))
