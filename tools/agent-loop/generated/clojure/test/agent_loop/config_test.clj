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

(deftest test-has-key
  (is (true? (config/has-key {:settings {:model "gpt-4"}} :model)))
  (is (false? (config/has-key {:settings {}} :model))))

(deftest test-is-debug
  (is (config/is-debug {:debug "true"}))
  (is (not (config/is-debug {:debug "false"}))))

(deftest test-config-has-field
  (is (true? (config/config-has-field {:model "gpt-4"} :model)))
  (is (false? (config/config-has-field {} :nonexistent))))

(deftest test-config-merge
  (is (= "new-val" (config/merge {:settings {"k" "old"}} "k" "new-val")))
  (is (= "old" (config/merge {:settings {"k" "old"}} "k" ""))))

(deftest test-config-field-count
  (is (= 2 (config/field-count {:settings {:a 1 :b 2}}))))

(deftest test-config-is-empty
  (is (true? (config/is-empty {:settings {}})))
  (is (false? (config/is-empty {:settings {:a 1}}))))

(deftest test-config-is-default-backend
  (is (true? (config/is-default-backend {:settings {}})))
  (is (true? (config/is-default-backend {:settings {"model" "gpt-4"}})))
  (is (false? (config/is-default-backend {:settings {"backend" "openai"}}))))

(deftest test-key-count
  (is (= 0 (config/key-count {:settings {}})))
  (is (= 2 (config/key-count {:settings {:a 1 :b 2}}))))

(deftest test-config-has-backend
  (is (true? (config/has-backend {:settings {"backend" "openai"}})))
  (is (false? (config/has-backend {:settings {"model" "gpt-4"}}))))

(deftest test-config-is-production
  (is (true? (config/is-production {:debug "false"})))
  (is (false? (config/is-production {:debug "true"}))))

(deftest test-config-has-model
  (is (true? (config/has-model {:settings {"model" "gpt-4"}})))
  (is (false? (config/has-model {:settings {}}))))

(deftest test-config-has-api-key
  (is (true? (config/has-api-key {:settings {"api_key" "sk-123"}})))
  (is (false? (config/has-api-key {:settings {}}))))

(deftest test-has-stream
  (is (true? (config/has-stream {:settings {"stream" "true"}})))
  (is (false? (config/has-stream {:settings {}}))))

(deftest test-has-max-tokens
  (is (true? (config/has-max-tokens {:settings {"max_tokens" "100"}})))
  (is (false? (config/has-max-tokens {:settings {}}))))
