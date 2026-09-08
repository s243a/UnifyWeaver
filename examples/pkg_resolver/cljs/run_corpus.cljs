#!/usr/bin/env nbb
;; SPDX-License-Identifier: MIT OR Apache-2.0
;; Copyright (c) 2026 John William Creighton (s243a)
;;
;; run_corpus.cljs -- drive the 38-scenario contract corpus (dump_corpus.pl's
;; JSONL, SWI expected included) through the ClojureScript-WAM resolver and
;; compare every answer to SWI. The ClojureScript twin of ../wamjs/run_corpus.mjs.
;;
;;   nbb --classpath examples/pkg_resolver/cljs \
;;       examples/pkg_resolver/cljs/run_corpus.cljs <swi.jsonl> <cljs.jsonl>
;;
;; ONE process for the whole corpus: nbb's startup dominates its per-case cost.

(ns run-corpus
  (:require [resolver :as R]
            [clojure.string :as str]
            ["fs" :as fs]))

;; Key-sorted JSON, so a comparison never depends on key order -- the same
;; stableStringify run_corpus.mjs and compare_jsonl.mjs use.
(defn stable-stringify [x]
  (cond
    (map? x) (str "{" (str/join "," (map (fn [k] (str (js/JSON.stringify k) ":"
                                                      (stable-stringify (get x k))))
                                         (sort (keys x)))) "}")
    (or (vector? x) (seq? x)) (str "[" (str/join "," (map stable-stringify x)) "]")
    (nil? x) "null"
    :else (js/JSON.stringify (clj->js x))))

(defn -main [argv]
  (let [[src dest] argv]
    (when (or (nil? src) (nil? dest))
      (.error js/console "usage: run_corpus.cljs <swi.jsonl> <cljs.jsonl>")
      (js/process.exit 2))
    (let [lines (remove empty? (.split (.readFileSync fs src "utf8") "\n"))
          results (atom [])
          diverged (atom 0)]
      (doseq [line lines]
        (let [row (js->clj (js/JSON.parse line))
              got (try (R/run-case row)
                       (catch :default e {"crash" (str (or (.-stack e) (.-message e) e))}))
              exp (get row "expected")]
          (swap! results conj (str "{\"id\":" (js/JSON.stringify (get row "id"))
                                   ",\"got\":" (stable-stringify got) "}"))
          (if (= (stable-stringify got) (stable-stringify exp))
            (println "ok" (get row "id"))
            (do (swap! diverged inc)
                (.error js/console (str "DIVERGE " (get row "id")))
                (.error js/console (str "  expected " (stable-stringify exp)))
                (.error js/console (str "  got      " (stable-stringify got)))))))
      (.writeFileSync fs dest (str (str/join "\n" @results) "\n"))
      (if (zero? @diverged)
        (println (str "corpus-under-nbb: " (count lines) "/" (count lines) " matched SWI"))
        (do (.error js/console (str "corpus-under-nbb: " @diverged " divergences / " (count lines)))
            (js/process.exit 1))))))

(-main (vec (drop 0 *command-line-args*)))
