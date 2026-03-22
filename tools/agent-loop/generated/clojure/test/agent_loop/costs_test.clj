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
