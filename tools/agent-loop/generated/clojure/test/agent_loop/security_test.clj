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

(deftest test-has-path-traversal
  (is (true? (security/has-path-traversal "..")))
  (is (true? (security/has-path-traversal "../etc")))
  (is (false? (security/has-path-traversal "src/main"))))

(deftest test-is-safe-command
  (is (true? (security/is-safe-command "ls -la")))
  (is (true? (security/is-safe-command "cat file.txt")))
  (is (false? (security/is-safe-command "rm -rf /"))))

(deftest test-is-blocked-command
  (is (true? (security/is-blocked-command "rm -rf /")))
  (is (true? (security/is-blocked-command "dd if=/dev/zero")))
  (is (true? (security/is-blocked-command "mkfs.ext4 /dev/sda")))
  (is (false? (security/is-blocked-command "ls -la"))))

(deftest test-is-writable-path
  (is (true? (security/is-writable-path "/home/user/file.txt")))
  (is (true? (security/is-writable-path "/tmp/scratch")))
  (is (false? (security/is-writable-path "/etc/passwd")))
  (is (false? (security/is-writable-path "/usr/bin/cat")))
  (is (false? (security/is-writable-path "/bin/sh"))))

(deftest test-needs-audit
  (is (true? (security/needs-audit "paranoid")))
  (is (true? (security/needs-audit "guarded")))
  (is (false? (security/needs-audit "open"))))

(deftest test-allows-auto
  (is (true? (security/allows-auto "open")))
  (is (false? (security/allows-auto "paranoid"))))
