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

(deftest test-blocked-paths-exist
  (is (pos? (count security/blocked-paths))))

(deftest test-path-blocked
  (is (some? (security/path-blocked? "/etc/shadow")))
  (is (not (security/path-blocked? "src/main.rs"))))

(deftest test-is-hidden-path
  (is (true? (security/is-hidden-path ".git")))
  (is (false? (security/is-hidden-path "src"))))
