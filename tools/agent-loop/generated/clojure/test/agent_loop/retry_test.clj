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

(deftest test-is-retryable-error
  (is (retry/is-retryable-error 429))
  (is (retry/is-retryable-error 500))
  (is (retry/is-retryable-error 503))
  (is (not (retry/is-retryable-error 200)))
  (is (not (retry/is-retryable-error 404))))

(deftest test-max-retries-reached
  (is (true? (retry/max-retries-reached {:attempt 3 :max-retries 3})))
  (is (false? (retry/max-retries-reached {:attempt 1 :max-retries 3}))))

(deftest test-is-first-attempt
  (is (true? (retry/is-first-attempt 0)))
  (is (false? (retry/is-first-attempt 1))))

(deftest test-attempts-left
  (is (= 2 (retry/attempts-left {:max-retries 3 :attempt 1})))
  (is (= 0 (retry/attempts-left {:max-retries 3 :attempt 3}))))
