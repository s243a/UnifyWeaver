(ns agent-loop.config-test
  (:require [clojure.test :refer [deftest is testing]]
            [agent-loop.config :as config]))

(deftest test-default-config-exists
  (is (map? config/default-config)))

(deftest test-search-paths-exist
  (is (pos? (count config/config-search-paths))))

(deftest test-api-key-env-vars
  (is (map? config/api-key-env-vars))
  (is (string? (get config/api-key-env-vars :openai)))
  (is (string? (get config/api-key-env-vars :claude))))
