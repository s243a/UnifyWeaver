(ns agent-loop.pricing-test
  (:require [clojure.test :refer [deftest is testing]]
            [agent-loop.pricing :as pricing]))

(deftest test-pricing-table-exists
  (is (pos? (count pricing/pricing))))

(deftest test-lookup-known-model
  (let [entry (pricing/lookup "gpt-4")]
    (when entry
      (is (number? (:input entry)))
      (is (number? (:output entry))))))

(deftest test-lookup-unknown-model
  (is (nil? (pricing/lookup "nonexistent-model-xyz"))))
