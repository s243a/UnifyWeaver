(ns agent-loop.property-test
  (:require [clojure.test :refer [deftest is testing]]
            [clojure.test.check :as tc]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [clojure.test.check.clojure-test :refer [defspec]]
            [agent-loop.costs :as costs]
            [agent-loop.context :as ctx]
            [agent-loop.retry :as retry]
            [agent-loop.streaming :as streaming]
            [agent-loop.sessions :as sessions]
            [agent-loop.security :as security]
            [agent-loop.tools :as tools]))

(defspec budget-remaining-clamped 50
  (prop/for-all [budget (gen/double* {:min 0.01 :max 1000.0 :NaN? false :infinite? false})
                 cost (gen/double* {:min 0.0 :max 500.0 :NaN? false :infinite? false})]
    (let [state {:total-cost cost}
          remaining (costs/budget-remaining state budget)]
      (and (>= remaining 0.0)
           (<= remaining budget)))))

(defspec cost-compute-non-negative 50
  (prop/for-all [tokens (gen/choose 0 100000)
                 price (gen/double* {:min 0.0 :max 100.0 :NaN? false :infinite? false})]
    (>= (costs/cost-compute tokens price) 0.0)))

(defspec input-output-ratio-sum 50
  (prop/for-all [input (gen/choose 1 10000)
                 output (gen/choose 1 10000)]
    (let [state {:total-input-tokens input :total-output-tokens output}
          ir (costs/input-ratio state)
          or_ (costs/output-ratio state)]
      (< (Math/abs (- (+ ir or_) 1.0)) 0.001))))

(defspec session-age-non-negative 50
  (prop/for-all [created (gen/double* {:min 0.0 :max 1e9 :NaN? false :infinite? false})
                 now (gen/double* {:min 0.0 :max 1e9 :NaN? false :infinite? false})]
    (let [age (sessions/session-age created (max created now))]
      (>= age 0.0))))

(defspec retry-delay-positive 50
  (prop/for-all [base (gen/double* {:min 0.1 :max 10.0 :NaN? false :infinite? false})
                 attempt (gen/choose 0 10)]
    (> (retry/exponential-delay base attempt) 0.0)))

(defspec token-budget-unlimited 50
  (prop/for-all [tokens (gen/choose 0 10000)]
    (= -1 (ctx/token-budget {:max-tokens 0 :token-count tokens}))))

(deftest safe-blocked-disjoint
  (testing "no command is both safe and blocked"
    (doseq [cmd ["ls" "cat file" "rm -rf /" "echo hi" "dd if=x" "grep p" "mkfs.ext4"]]
      (is (not (and (security/is-safe-command cmd) (security/is-blocked-command cmd)))))))

(defspec overflow-count-non-negative 50
  (prop/for-all [count (gen/choose 0 100)
                 limit (gen/choose 0 100)]
    (>= (sessions/overflow-count count limit) 0)))

(defspec budget-over-under-complementary 50
  (prop/for-all [cost (gen/double* {:min 0.0 :max 100.0 :NaN? false :infinite? false})
                 threshold (gen/double* {:min 0.01 :max 100.0 :NaN? false :infinite? false})]
    (let [state {:total-cost cost}]
      (or (costs/is-over-budget state threshold)
          (costs/is-under state threshold)
          (== cost threshold)))))
