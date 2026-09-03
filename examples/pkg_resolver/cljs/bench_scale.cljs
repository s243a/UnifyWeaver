#!/usr/bin/env nbb
;; SPDX-License-Identifier: MIT OR Apache-2.0
;; Copyright (c) 2026 John William Creighton (s243a)
;;
;; bench_scale.cljs -- B3: one resolve_layered on the 5k-package scale catalog,
;; with LOAD and RESOLVE timed separately.
;;
;;   nbb --classpath examples/pkg_resolver/cljs \
;;       examples/pkg_resolver/cljs/bench_scale.cljs <scale-dir>
;;
;; "load" is everything before the query can be asked: read the catalog JSON,
;; parse it, and build the WAM term (catalog->term). "resolve" is the query
;; alone, against the already-built term -- the same split store/scale_demo.pl
;; reports as swi_term_load_s / swi_term_resolve_s.

(ns bench-scale
  (:require [resolver :as R]
            ["fs" :as fs]
            ["path" :as path]))

(defn -main [argv]
  (let [dir (first argv)]
    (when (nil? dir)
      (.error js/console "usage: bench_scale.cljs <scale-dir>")
      (js/process.exit 2))
    (let [probe (js->clj (js/JSON.parse
                           (.readFileSync fs (.join path dir "probe.json") "utf8")))
          req (let [a (get probe "args")] (if (vector? a) (first a) a))

          t0 (js/Date.now)
          cat (js->clj (js/JSON.parse
                         (.readFileSync fs (.join path dir "catalog.json") "utf8")))
          term (R/catalog->term cat)
          t1 (js/Date.now)

          r (R/resolve-layered-prepared term [req])
          t2 (js/Date.now)]
      (println (str "cljs_term_load_s " (.toFixed (/ (- t1 t0) 1000) 3)))
      (println (str "cljs_term_resolve_s " (.toFixed (/ (- t2 t1) 1000) 3)))
      (println (str "cljs_term_total_s " (.toFixed (/ (- t2 t0) 1000) 3)))
      (println (str "cljs_term_request " req))
      (println (str "cljs_term_result " (js/JSON.stringify (clj->js r)))))))

(-main (vec *command-line-args*))
