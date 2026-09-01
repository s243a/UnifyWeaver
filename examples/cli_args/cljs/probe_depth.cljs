;; SPDX-License-Identifier: MIT OR Apache-2.0
;; Copyright (c) 2026 John William Creighton (s243a)
;;
;; probe_depth.cljs -- how much recursion headroom does the transpiled parser
;; have under nbb?
;;
;;   nbb --classpath . probe_depth.cljs
;;
;; WHY THIS EXISTS. The A3 lowering emits self-calls as DIRECT calls, not
;; `recur`, which is the faithful analogue of the TypeScript lane's
;; `return pred(...)`. Neither JavaScript nor nbb eliminates tail calls, so
;; recursion depth is bounded by input size and the claim "that is fine here"
;; has to be measured rather than asserted.
;;
;; The deepest walks in cli_args are over a TOKEN'S CHARACTERS
;; (long_flag_tail/1, legacy_flag_tail/1, first_char_index/4, drop_brackets/2)
;; and over ARGV (lenient_loop/5, strict_loop/8). The harness's own alphabet
;; tops out at a 26-character token and 7 argv tokens; this probe pushes both
;; far past that and reports where the runtime actually gives out.

(require '[generated.cli-args :as ca])

(defn- try-token [n]
  ;; `--` plus n flag characters: the deepest character walk the parser has.
  (let [tok (str "--a" (apply str (repeat n "b")))]
    (try
      (ca/parse-args-2 ["block" "bob" tok])
      n
      (catch :default _e nil))))

(defn- try-argv [n]
  (try
    (ca/parse-args-2 (vec (cons "block" (repeat n "x"))))
    n
    (catch :default _e nil)))

(defn- deepest [f candidates]
  (last (filter some? (map f candidates))))

(println "corpus/differential bound:  token <= 26 chars, argv <= 7 tokens")
(println "deepest token that parses: "
         (deepest try-token [10 100 1000 2000 3000 4000 6000 8000]))
(println "deepest argv that parses:  "
         (deepest try-argv [10 50 100 200 300 400 500 700 1000]))
