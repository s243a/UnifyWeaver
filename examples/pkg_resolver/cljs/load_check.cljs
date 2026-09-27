;; SPDX-License-Identifier: MIT OR Apache-2.0
;; Copyright (c) 2026 John William Creighton (s243a)
;;
;; load_check.cljs -- the `node --check` analogue for the ClojureScript build:
;; nbb must load the generated namespaces and the edge shim clean, and the
;; instruction table and dispatch map must be non-empty.
;;
;;   nbb --classpath examples/pkg_resolver/cljs \
;;       examples/pkg_resolver/cljs/load_check.cljs

(ns load-check
  (:require [generated.resolver.core :as core]
            [resolver :as R]))

(let [n-instrs (count core/shared-wam-code)
      n-preds  (count core/predicate-dispatch)
      queries  (count R/QUERIES)]
  (when (zero? n-instrs) (throw (ex-info "empty WAM instruction table" {})))
  (when (zero? n-preds) (throw (ex-info "empty predicate dispatch" {})))
  (when (not= 10 queries) (throw (ex-info "expected 10 resolver queries" {:got queries})))
  (println (str "load_check: nbb loaded generated.resolver.core clean -- "
                n-instrs " WAM instructions, " n-preds " predicates, "
                queries " queries wired")))
