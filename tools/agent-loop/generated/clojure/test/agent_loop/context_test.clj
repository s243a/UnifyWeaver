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

(deftest test-token-budget
  (is (= -1 (ctx/token-budget {:max-tokens 0 :token-count 0})))
  (is (= 80 (ctx/token-budget {:max-tokens 100 :token-count 20}))))

(deftest test-context-messages-remaining
  (is (= -1 (ctx/messages-remaining {:max-messages 0 :messages []})))
  (is (= 3 (ctx/messages-remaining {:max-messages 5 :messages [{} {}]}))))

(deftest test-context-word-budget
  (is (= -1 (ctx/word-budget {:max-words 0 :token-count 0})))
  (is (= -1 (ctx/word-budget {:max-words -5 :token-count 10})))
  (is (= 80 (ctx/word-budget {:max-words 100 :token-count 20}))))

(deftest test-is-full
  (is (false? (ctx/is-full {:max-messages 0 :messages []})))
  (is (true? (ctx/is-full {:max-messages 2 :messages [{} {}]}))))

(deftest test-context-char-budget
  (is (= -1 (ctx/char-budget {:max-chars 0 :token-count 0})))
  (is (= 80 (ctx/char-budget {:max-chars 100 :token-count 20}))))

(deftest test-context-has-room
  (is (true? (ctx/has-room {:max-messages 0 :messages []})))
  (is (true? (ctx/has-room {:max-messages 5 :messages [{} {}]})))
  (is (false? (ctx/has-room {:max-messages 2 :messages [{} {}]}))))

(deftest test-context-is-near-full
  (is (false? (ctx/is-near-full {:max-messages 0 :messages []} 80)))
  (is (true? (ctx/is-near-full {:max-messages 10 :messages [{} {} {} {} {} {} {} {} {}]} 80))))

(deftest test-context-usage-pct
  (is (= 0.0 (ctx/usage-pct {:max-messages 0 :messages []})))
  (is (= 50.0 (ctx/usage-pct {:max-messages 10 :messages [{} {} {} {} {}]}))))
