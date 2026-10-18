;; SPDX-License-Identifier: MIT OR Apache-2.0
;; Copyright (c) 2026 John William Creighton (s243a)
;;
;; diff_runner_cljs.cljs -- the TRANSPILED side of the differential harness,
;; under nbb.
;;
;; Byte-for-byte the same protocol as examples/cli_args/diff_runner.mjs (the
;; oracle side), diff_runner.pl (the SWI side) and patternjs's runner: read
;; argv-lines on stdin, one case per line, tokens separated by spaces, skip lines
;; with no tokens, and print one JSON object per line --
;;
;;   {"ok":{"positional":[...],"flags":{...}}}
;;   {"error":"<CliError message>"}
;;   {"crash":"<unexpected error>"}          ;; never expected; a hard failure
;;
;;   nbb --classpath . diff_runner_cljs.cljs < lines.txt > cljs.jsonl
;;
;; ONE process for the whole file. The protocol is designed for that and it is
;; the only workable shape here: nbb's startup dominates its per-line cost, so
;; 5067 processes would take hours where one takes seconds.
;;
;; WHAT IS IN HERE, exhaustively: conversion between UnifyWeaver's term
;; representation and plain JavaScript values, at the module boundary. There is
;; NO parse logic. Every decision about what an argv line means -- the two flag
;; regexes, the strict/lenient split, the schema lookup, the arity check, the
;; exact wording of every error message -- lives in generated/cli_args.cljs,
;; which is compiler output from examples/cli_args/cli_args.pl.
;;
;; The term representation (clojure_target's A3 lowering, G-A3-12/G-A3-13):
;;
;;   Prolog atom / string    a string
;;   true / false            a boolean
;;   list                    a vector or seq
;;   compound f(A1..An)      {:$ "f" :args [...]}
;;
;; so `ok(Positional, Flags)` arrives as
;; `{:$ "ok" :args [(...) [{:$ "-" :args [K V]} ...]]}`.

(ns diff-runner-cljs
  (:require [generated.cli-args :as ca]
            ["fs" :as fs]))

(defn- term? [t f]
  (and (map? t) (= (:$ t) f)))

;; A Prolog list of `Key-Value` pairs becomes a plain JS object, in pair order.
;;
;; A real JS object built by assignment is deliberate rather than a Clojure map:
;; the oracle builds its `flags` exactly this way, so JS object semantics --
;; key-insertion order, and JSON.stringify's treatment of the result -- are
;; reproduced rather than re-implemented. cli_args.pl models the same semantics
;; on the Prolog side (flags_set/4), so the two agree before this is reached.
;; `clj->js` on a Clojure map would lose the insertion order.
(defn- pairs->jsobj [pairs]
  (let [o (js-obj)]
    (doseq [p pairs]
      (when-not (and (term? p "-") (= (count (:args p)) 2))
        (throw (ex-info (str "cljs runner: expected a Key-Value pair, got " (pr-str p)) {})))
      (aset o (nth (:args p) 0) (nth (:args p) 1)))
    o))

(defn- run-line [tokens]
  (try
    (let [result (ca/parse-args-2 (vec tokens))]
      (cond
        (and (term? result "ok") (= (count (:args result)) 2))
        #js {:ok #js {:positional (clj->js (vec (nth (:args result) 0)))
                      :flags      (pairs->jsobj (nth (:args result) 1))}}

        (and (term? result "error") (= (count (:args result)) 1))
        #js {:error (nth (:args result) 0)}

        ;; parse_args/2 is semidet in the compiled calling convention, so it CAN
        ;; answer the failure sentinel. cli_args.pl documents that it never
        ;; fails; if it ever does, that is a compiler or reference bug and must
        ;; be loud rather than silently reported as an error result.
        :else
        #js {:crash (str "parse-args answered neither ok/2 nor error/1: " (pr-str result))}))
    (catch :default e
      #js {:crash (str (or (.-message e) e))})))

(defn- tokenize [line]
  (vec (remove empty? (.split line " "))))

(defn -main []
  (let [input (.readFileSync fs 0 "utf8")
        lines (.split input "\n")
        out   (->> lines
                   (map tokenize)
                   ;; Skip only lines that carry no tokens at all -- the exact
                   ;; rule every other runner applies, so the runners stay
                   ;; line-for-line aligned.
                   (remove empty?)
                   (mapv (fn [tokens] (js/JSON.stringify (run-line tokens)))))]
    (when (seq out)
      (.write (.-stdout js/process) (str (clojure.string/join "\n" out) "\n")))))

(-main)
