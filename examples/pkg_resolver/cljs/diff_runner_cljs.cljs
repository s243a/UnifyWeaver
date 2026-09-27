#!/usr/bin/env nbb
;; SPDX-License-Identifier: MIT OR Apache-2.0
;; Copyright (c) 2026 John William Creighton (s243a)
;;
;; diff_runner_cljs.cljs -- ClojureScript-WAM side of the pkg_resolver
;; differential. Reads the same JSONL as diff_runner.pl (the SWI oracle) and
;; ../wamjs/diff_runner_wamjs.mjs; writes one result object per line.
;;
;;   nbb --classpath examples/pkg_resolver/cljs \
;;       examples/pkg_resolver/cljs/diff_runner_cljs.cljs < cases.jsonl
;;
;; ONE process for the whole file: nbb's startup dominates its per-case cost.
;;
;; WHAT IS IN HERE, exhaustively: stdin in, one JSON line out. Every answer
;; comes from ./resolver.cljs over the compiled resolver.

(ns diff-runner-cljs
  (:require [resolver :as R]
            [clojure.string :as str]
            ["fs" :as fs]))

(defn -main []
  (let [input (.readFileSync fs 0 "utf8")
        lines (remove empty? (.split input "\n"))
        out (mapv (fn [line]
                    (let [row (js->clj (js/JSON.parse line))
                          got (try (R/run-case row)
                                   (catch :default e
                                     {"crash" (str (or (.-stack e) (.-message e) e))}))]
                      (js/JSON.stringify
                        (clj->js (assoc got "id" (get row "id"))))))
                  lines)]
    (when (seq out)
      (.write (.-stdout js/process) (str (str/join "\n" out) "\n")))))

(-main)
