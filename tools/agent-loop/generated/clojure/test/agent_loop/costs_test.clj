(ns agent-loop.costs-test
  (:require [clojure.test :refer [deftest is testing]]
            [agent-loop.costs :as costs]))

(deftest test-create-cost-tracker
  (let [state (costs/create-cost-tracker)]
    (is (= 0.0 (:total-cost state)))
    (is (= 0 (:total-input-tokens state)))
    (is (= 0 (:total-output-tokens state)))))

(deftest test-is-over-budget
  (testing "zero budget means unlimited"
    (is (false? (costs/is-over-budget (costs/create-cost-tracker) 0))))
  (testing "under budget"
    (is (false? (costs/is-over-budget {:total-cost 5.0} 10.0))))
  (testing "over budget"
    (is (true? (costs/is-over-budget {:total-cost 15.0} 10.0)))))

(deftest test-budget-remaining
  (testing "unlimited returns -1"
    (is (= -1.0 (costs/budget-remaining (costs/create-cost-tracker) 0))))
  (testing "remaining calculation"
    (is (= 7.0 (costs/budget-remaining {:total-cost 3.0} 10.0)))))

(deftest test-cost-compute
  (let [cost (costs/cost-compute 1000 15.0)]
    (is (< (Math/abs (- cost 0.015)) 0.0001))))

(deftest test-total-tokens
  (is (= 300 (costs/total-tokens {:total-input-tokens 200 :total-output-tokens 100}))))

(deftest test-is-free-model
  (is (true? (costs/is-free-model {:total-cost 0.0})))
  (is (false? (costs/is-free-model {:total-cost 0.5}))))

(deftest test-average-cost-per-message
  (is (= 0.0 (costs/average-cost-per-message {:total-cost 10.0 :message-count 0})))
  (is (= 2.0 (costs/average-cost-per-message {:total-cost 10.0 :message-count 5}))))

(deftest test-is-tracking
  (is (false? (costs/is-tracking {:message-count 0})))
  (is (true? (costs/is-tracking {:message-count 3}))))

(deftest test-has-records
  (is (not (costs/has-records {:records []})))
  (is (costs/has-records {:records [{:tokens 10}]})))

(deftest test-cost-summary-short
  (is (string? (costs/cost-summary-short {:total-cost 1.5 :total-input-tokens 100 :total-output-tokens 50}))))

(deftest test-cost-per-input-token
  (is (= 0.0 (costs/cost-per-input-token {:total-cost 1.0 :total-input-tokens 0})))
  (is (pos? (costs/cost-per-input-token {:total-cost 1.0 :total-input-tokens 1000}))))

(deftest test-reset
  (let [state (costs/reset {:total-cost 5.0 :total-input-tokens 100 :total-output-tokens 50 :message-count 3})]
    (is (= 0.0 (:total-cost state)))
    (is (= 0 (:total-input-tokens state)))))

(deftest test-total-messages
  (is (= 5 (costs/total-messages {:message-count 5})))
  (is (= 0 (costs/total-messages {:message-count 0}))))

(deftest test-cost-exceeds
  (is (true? (costs/cost-exceeds {:total-cost 15.0} 10.0)))
  (is (false? (costs/cost-exceeds {:total-cost 5.0} 10.0))))

(deftest test-input-ratio
  (is (= 0.0 (costs/input-ratio {:total-input-tokens 0 :total-output-tokens 0})))
  (is (= 0.5 (costs/input-ratio {:total-input-tokens 50 :total-output-tokens 50}))))

(deftest test-cost-format-dollars
  (is (= "$1.5" (costs/format-dollars 1.5)))
  (is (= "$0" (costs/format-dollars 0)))
  (is (= "$99.99" (costs/format-dollars 99.99))))

(deftest test-has-usage
  (is (false? (costs/has-usage {:total-input-tokens 0 :total-output-tokens 0})))
  (is (true? (costs/has-usage {:total-input-tokens 100 :total-output-tokens 0}))))
