(ns agent-loop.context-test
  (:require [clojure.test :refer [deftest is testing]]
            [agent-loop.context :as ctx]))

(deftest test-create-context
  (let [state (ctx/create-context)]
    (is (= [] (:messages state)))
    (is (= "plain" (:format state)))))

(deftest test-context-is-empty
  (is (true? (ctx/is-empty (ctx/create-context))))
  (is (false? (ctx/is-empty {:messages [{:role "user"}]}))))

(deftest test-estimate-tokens
  (let [state {:messages [{:content "a"} {:content "b"} {:content "c"} {:content "d"} {:content "e"}]}]
    (is (>= (ctx/estimate-tokens state) 0))))

(deftest test-word-count
  (is (= 3 (ctx/word-count "hello world foo"))))

(deftest test-message-token-estimate
  (is (pos? (ctx/message-token-estimate "some content here"))))
