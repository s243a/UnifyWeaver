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

(deftest test-cache-len
  (is (= 0 (cache/len {:cache {}})))
  (is (= 2 (cache/len {:cache {"a" 1 "b" 2}}))))

(deftest test-is-empty-cache
  (is (true? (cache/is-empty-cache {:cache {}})))
  (is (false? (cache/is-empty-cache {:cache {"k" "v"}}))))

(deftest test-cache-has-key
  (is (true? (cache/has-key {:cache {"bash_ls" "result"}} "bash_ls")))
  (is (false? (cache/has-key {:cache {}} "bash_ls"))))

(deftest test-make-key
  (let [key (cache/make-key {:cache {}} "bash" {"cmd" "ls -la"})]
    (is (string? key))
    (is (pos? (count key)))))

(deftest test-evict-oldest
  (is (= 0 (cache/evict-oldest {:cache {} :max-size 100})))
  (is (= 1 (cache/evict-oldest {:cache {"a" 1 "b" 2} :max-size 2}))))

(deftest test-cache-hit-rate
  (is (= 0.0 (cache/cache-hit-rate {:hits 0 :total-lookups 0})))
  (is (= 0.5 (cache/cache-hit-rate {:hits 5 :total-lookups 10}))))
