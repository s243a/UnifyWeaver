(ns agent-loop.tool-cache-test
  (:require [clojure.test :refer [deftest is testing]]
            [agent-loop.tool-cache :as cache]))

(deftest test-create-cache
  (let [state (cache/create-tool-cache)]
    (is (= {} (:cache state)))
    (is (= 100 (:max-size state)))))

(deftest test-cache-clear
  (let [state {:cache {"k1" "v1"} :max-size 100}
        cleared (cache/clear state)]
    (is (= {} (:cache cleared)))))

(deftest test-should-skip
  (let [state {:cache {} :max-size 100 :skip-tools #{"bash" "write"}}]
    (is (true? (cache/should-skip state "bash")))
    (is (false? (cache/should-skip state "read")))))
