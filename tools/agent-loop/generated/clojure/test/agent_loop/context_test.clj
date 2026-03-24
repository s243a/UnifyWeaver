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

(deftest test-context-len
  (is (= 0 (ctx/len {:messages []})))
  (is (= 2 (ctx/len {:messages [{:role "user"} {:role "assistant"}]}))))

(deftest test-context-has-messages
  (is (nil? (ctx/has-messages {:messages []})))
  (is (some? (ctx/has-messages {:messages [{:role "user"}]}))))

(deftest test-context-clear
  (let [state (ctx/clear {:messages [{:role "user"}]})]
    (is (= [] (:messages state)))))

(deftest test-context-message-count
  (is (= 3 (ctx/context-message-count {:messages [{} {} {}]}))))

(deftest test-get-format
  (is (= "json" (ctx/get-format {:format "json"}))))
