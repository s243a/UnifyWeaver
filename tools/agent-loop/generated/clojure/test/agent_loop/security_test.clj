(ns agent-loop.security-test
  (:require [clojure.test :refer [deftest is testing]]
            [agent-loop.security :as security]))

(deftest test-is-path-safe
  (is (true? (security/is-path-safe "src/main.rs")))
  (is (false? (security/is-path-safe "../../etc/passwd"))))

(deftest test-is-visible-file
  (is (true? (security/is-visible-file "main.rs")))
  (is (false? (security/is-visible-file ".env"))))

(deftest test-profiles-exist
  (is (pos? (count security/profiles))))
