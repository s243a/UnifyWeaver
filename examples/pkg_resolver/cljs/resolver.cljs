;; SPDX-License-Identifier: MIT OR Apache-2.0
;; Copyright (c) 2026 John William Creighton (s243a)
;;
;; resolver.cljs -- EDGE of the ClojureScript-WAM compiled uw-resolve P3.
;;
;; The ClojureScript twin of ../wamjs/resolver.mjs, function for function.
;;
;; WHAT IS IN HERE, exhaustively: conversion between JSON catalogs/requests and
;; WAM terms, plus driving the generated predicate wrappers. There is NO
;; resolver logic -- no candidate order, no constraint arithmetic, no layer
;; walk, no freeze rule, no upgrade closure. Those live in
;; generated/resolver/core.cljs, which is compiler output from
;; examples/pkg_resolver/resolver.pl.
;;
;; The WAM term representation (templates/targets/clojure_wam/runtime.clj):
;;
;;   atom          {:tag :atom :id <interned id>}   (a string is interned to one)
;;   integer       a number
;;   compound      {:tag :struct :functor <atom> :args [...]}
;;   variable      {:var <id>}      -- a STRING id here, so it can never collide
;;                                     with the runtime's own integer ids
;;
;; Answers come back by unification into the variable we put in the output
;; argument register, so every query reads its answer out of the SUCCEEDING
;; state (runtime/run-wam-state) rather than from a return value.

(ns resolver
  (:require [generated.resolver.core :as core]
            [generated.resolver.runtime :as rt]
            [clojure.string :as str]))

;; ---------------------------------------------------------------------------
;; JSON -> WAM terms
;; ---------------------------------------------------------------------------

(defn- struct-term [functor args] (rt/structure-term functor (vec args)))

(defn- list-term [items]
  (reduce (fn [tail item] (struct-term "[|]/2" [item tail])) "[]" (reverse items)))

(defn- segs-term [segs]
  (list-term
    (map (fn [seg]
           (let [order (str (or (nth seg 0) ""))
                 num (or (nth seg 1) 0)
                 codes (mapv #(.charCodeAt order %) (range (count order)))]
             (struct-term "s/2" [(list-term codes) num])))
         (or segs []))))

;; P0 v/3 triples [M I P] and P3 {"deb":[Epoch,[[order,num],…],[…]]}.
(defn- ver-term [v]
  (if (and (map? v) (contains? v "deb"))
    (let [d (get v "deb")]
      (struct-term "deb/3" [(or (nth d 0) 0)
                            (segs-term (nth d 1 []))
                            (segs-term (nth d 2 []))]))
    (struct-term "v/3" [(nth v 0) (nth v 1) (nth v 2)])))

(defn- constraint-term [c]
  (cond
    (or (nil? c) (= c "any")) "any"
    (map? c)
    (case (get c "op")
      ("eq" "gte" "lt" "lte" "gt")
      (struct-term (str (get c "op") "/1") [(ver-term (get c "v"))])
      "range" (struct-term "range/2" [(ver-term (get c "lo")) (ver-term (get c "hi"))])
      (throw (ex-info (str "resolver shim: unknown constraint " (pr-str c)) {})))
    :else (throw (ex-info (str "resolver shim: unknown constraint " (pr-str c)) {}))))

(defn- pair-term [name ver] (struct-term "-/2" [name (ver-term ver)]))

;; A base entry is `Name-Ver` (P0) or `base(Name-Ver, Reason)` (P0.5).
(defn- hold-term [row]
  (if (>= (count row) 3)
    (struct-term "base/2" [(pair-term (nth row 0) (nth row 1)) (nth row 2)])
    (pair-term (nth row 0) (nth row 1))))

(defn- layer-term [row]
  (struct-term "layer/2" [(get row "name") (list-term (map hold-term (get row "packages" [])))]))

(defn- alias-term [row] (struct-term "alias/2" [(nth row 0) (nth row 1)]))
(defn- pkg-term [row] (struct-term "package/2" [(nth row 0) (ver-term (nth row 1))]))
(defn- dep-term [row]
  (let [third (nth row 2)
        dep-arg (if (and (map? third) (contains? third "alternatives"))
                  (struct-term "alternatives/1"
                               [(list-term
                                  (map (fn [a]
                                         (struct-term "dep/2"
                                                      [(get a "dep")
                                                       (constraint-term (get a "constraint"))]))
                                       (get third "alternatives" [])))])
                  third)]
    (struct-term "depends/4" [(nth row 0) (ver-term (nth row 1)) dep-arg
                              (constraint-term (nth row 3))])))
(defn- provide-term [row]
  (if (and (>= (count row) 4) (some? (nth row 3)))
    (struct-term "provides/4" [(nth row 0) (ver-term (nth row 1))
                               (nth row 2) (ver-term (nth row 3))])
    (struct-term "provides/3" [(nth row 0) (ver-term (nth row 1)) (nth row 2)])))
(defn- conf-term [row]
  (struct-term "conflicts/3" [(nth row 0) (ver-term (nth row 1)) (nth row 2)]))

(defn- request-term [req]
  (if (and (map? req) (contains? req "req"))
    (struct-term "req/2" [(get req "req") (constraint-term (get req "constraint"))])
    req))

;; catalog/6 when there is nothing P0.5/P3 to carry, catalog/9 for
;; layers/excluded/aliases, catalog/10 when provides are present.
(defn catalog->term [cat]
  (let [c (or cat {})
        core-args [(list-term (map pkg-term (get c "packages" [])))
                   (list-term (map dep-term (get c "depends" [])))
                   (list-term (map conf-term (get c "conflicts" [])))
                   (list-term (map hold-term (get c "base" [])))
                   (list-term (map (fn [p] (pair-term (nth p 0) (nth p 1))) (get c "installed" [])))
                   (list-term (get c "requested" []))]
        layers (get c "layers" [])
        excluded (get c "excluded" [])
        aliases (get c "aliases" [])
        provides (get c "provides" [])]
    (if (and (empty? layers) (empty? excluded) (empty? aliases) (empty? provides))
      (struct-term "catalog/6" core-args)
      (let [nine (conj (vec core-args)
                       (list-term (map layer-term layers))
                       (list-term excluded)
                       (list-term (map alias-term aliases)))]
        (if (empty? provides)
          (struct-term "catalog/9" nine)
          (struct-term "catalog/10" (conj nine (list-term (map provide-term provides)))))))))

;; ---------------------------------------------------------------------------
;; WAM terms -> JS values  (the mirror of resolver.mjs's termToJs)
;; ---------------------------------------------------------------------------

(defn- functor-name [state t]
  (str/replace (rt/deintern-atom (:intern-context state) (:id (:functor t))) #"/\d+$" ""))

(declare term->js)

(defn- list->js [state t]
  (loop [cur t out []]
    (let [cur (rt/deref-value (:bindings state) cur)]
      (cond
        (and (rt/atom-term? cur)
             (= "[]" (rt/deintern-atom (:intern-context state) (:id cur))))
        out

        (and (rt/structure-term? cur) (= "[|]" (functor-name state cur)))
        (recur (nth (:args cur) 1) (conj out (term->js state (nth (:args cur) 0))))

        :else (throw (ex-info "resolver shim: expected a list" {}))))))

(defn- blocked->js [name needs base-has]
  {"name" name "needs" needs "base_has" base-has})

(defn- audit->js [name payload]
  (cond
    (= payload "over_frozen") {"kind" "over_frozen" "name" name}
    (and (vector? payload) (= (first payload) "suggest"))
    {"kind" "suggest" "name" name "reason" (second payload)}
    (and (vector? payload) (= (first payload) "held"))
    {"kind" "held" "name" name "reason" (second payload)}
    :else {"kind" "held" "name" name "reason" payload}))

(defn- term->js [state t0]
  (let [t (rt/deref-value (:bindings state) t0)]
    (cond
      (number? t) t
      (rt/atom-term? t)
      (let [n (rt/deintern-atom (:intern-context state) (:id t))]
        (case n
          "[]" []
          "true" true
          "false" false
          n))

      (rt/structure-term? t)
      (let [n (functor-name state t)]
        (if (= n "[|]")
          (list->js state t)
          (let [args (mapv #(term->js state %) (:args t))]
            (cond
              (and (= n "v") (= 3 (count args))) args
              (and (= n "s") (= 2 (count args)))
              [(if (vector? (nth args 0))
                 (apply str (map char (nth args 0)))
                 "")
               (nth args 1)]
              (and (= n "deb") (= 3 (count args))) {"deb" args}
              (and (= n "-") (= 2 (count args))) args
              (and (= n "blocked") (= 3 (count args)))
              (let [needs (let [x (nth args 1)] (if (and (vector? x) (= "needs" (first x))) (second x) x))
                    third (nth args 2)]
                (if (and (vector? third) (= "providers" (first third)))
                  {"name" (nth args 0) "needs" needs "providers" (second third)}
                  (blocked->js (nth args 0) needs
                               (if (and (vector? third) (= "base_has" (first third))) (second third) third))))
              (and (= n "blocked") (= 1 (count args))
                   (vector? (first args)) (= "alternatives" (ffirst args)))
              {"alternatives" (second (first args))}
              (and (= n "alt") (= 2 (count args))) {"dep" (nth args 0) "reason" (nth args 1)}
              (and (= n "safe") (= 1 (count args)))
              {"verdict" "safe"
               "cost" (let [x (first args)] (if (and (vector? x) (= "cost" (first x))) (second x) x))}
              (and (= n "coordinated") (= 1 (count args))) {"verdict" "coordinated" "set" (first args)}
              (and (= n "unsafe") (= 1 (count args))) {"verdict" "unsafe" "reason" (first args)}
              (and (= n "audit") (= 2 (count args))) (audit->js (nth args 0) (nth args 1))
              (and (= n "ok") (= 1 (count args))) {"__ok_set" (first args)}
              (and (#{"needs" "base_has" "eq" "gte" "lt" "lte" "gt"
                      "cost" "held" "suggest" "providers" "alternatives"} n)
                   (= 1 (count args)))
              [n (first args)]
              (= n "range") {"op" "range" "lo" (nth args 0) "hi" (nth args 1)}
              :else (into [n] args)))))

      (rt/logic-var? t) nil
      :else t)))

(defn- normalize-constraint [c]
  (cond
    (= c "any") "any"
    (and (vector? c) (#{"gte" "eq" "lt" "lte" "gt"} (first c))) {"op" (first c) "v" (second c)}
    (and (map? c) (contains? c "op")) c
    (and (vector? c) (= "range" (first c))) {"op" "range" "lo" (nth c 1) "hi" (nth c 2)}
    :else c))

(defn- normalize-blocked [b]
  (cond
    (and (map? b) (contains? b "alternatives"))
    {"alternatives" (get b "alternatives")}
    (and (map? b) (contains? b "providers"))
    {"name" (get b "name")
     "needs" (normalize-constraint (get b "needs"))
     "providers" (mapv normalize-blocked (or (get b "providers") []))}
    (and (map? b) (contains? b "name"))
    {"base_has" (get b "base_has") "name" (get b "name")
     "needs" (normalize-constraint (get b "needs"))}
    :else b))

(defn- normalize-verdict [v]
  (if (= v "no_candidate") {"verdict" "no_candidate"} v))

(defn- normalize-upgrade [r]
  (cond
    (= r "no_candidate") {"fail" true}
    (and (map? r) (contains? r "__ok_set")) {"ok" (get r "__ok_set")}
    (vector? r) {"ok" r}
    (and (map? r) (contains? r "name") (contains? r "base_has"))
    {"ok" {"blocked" (normalize-blocked r)}}
    :else r))

;; ---------------------------------------------------------------------------
;; Driving the compiled predicates
;; ---------------------------------------------------------------------------

;; A fresh output cell. A STRING var id can never collide with the runtime's
;; own (integer) fresh variables, so the answer is always readable afterwards.
(defn- out-var [] {:var "uw-out"})

(defn- answer [state out]
  (when state (term->js state out)))

;; Each entry names the generated `<pred>-state` wrapper (emitted by
;; wam_clojure_target alongside the boolean `<pred>`), which answers the
;; SUCCEEDING WAM state or nil.
(def QUERIES
  {"resolve"              :resolve
   "resolve_layered"      :resolve-layered
   "explain_blocked"      :explain-blocked
   "layer_closure"        :layer-closure
   "removal_orphans"      :removal-orphans
   "safe_upgrade"         :safe-upgrade
   "upgrade_set"          :upgrade-set
   "freeze_audit"         :freeze-audit
   "dependents"           :dependents
   "dependents_installed" :dependents-installed})

(defn resolve-q [catalog requests]
  (let [o (out-var)
        s (core/resolve-state (catalog->term catalog) (list-term (map request-term requests)) o)]
    (if s {"ok" (answer s o)} {"fail" true})))

(defn resolve-layered [catalog requests]
  (let [o (out-var)
        s (core/resolve-layered-state (catalog->term catalog)
                                      (list-term (map request-term requests)) o)]
    (if s {"ok" (answer s o)} {"fail" true})))

;; The same query on an ALREADY-BUILT catalog term, so a benchmark can time
;; catalog->term (the "load") apart from the search itself.
(defn resolve-layered-prepared [cat-term requests]
  (let [o (out-var)
        s (core/resolve-layered-state cat-term (list-term (map request-term requests)) o)]
    (if s {"ok" (answer s o)} {"fail" true})))

;; B3 census: one resolve_layered (includes G1 index build) plus index_catalog/2
;; alone so query ≈ resolve − build. Bindings do not escape the counted run.
(defn resolve-layered-prepared-counted [cat-term requests]
  (let [o (out-var)
        s (binding [rt/*count-instrs* true]
            (core/resolve-layered-state cat-term (list-term (map request-term requests)) o))]
    {:result (if s {"ok" (answer s o)} {"fail" true})
     :instrs (or (and s (:instr-count s)) 0)}))

(defn index-catalog-counted [cat-term]
  (let [o (out-var)
        s (binding [rt/*count-instrs* true]
            (core/index-catalog-state cat-term o))]
    {:ok (some? s)
     :instrs (or (and s (:instr-count s)) 0)}))

(defn explain-blocked [catalog request]
  (let [o (out-var)
        s (core/explain-blocked-list-state (catalog->term catalog) (request-term request) o)]
    (if s {"ok" (mapv normalize-blocked (or (answer s o) []))} {"fail" true})))

(defn layer-closure [catalog request]
  (let [o (out-var)
        s (core/layer-closure-state (catalog->term catalog) (request-term request) o)]
    (if s {"ok" (answer s o)} {"fail" true})))

(defn removal-orphans [catalog pkg]
  (let [o (out-var)
        s (core/removal-orphans-state (catalog->term catalog) pkg o)]
    (if s {"ok" (answer s o)} {"fail" true})))

(defn safe-upgrade [catalog pkg ver]
  (let [o (out-var)
        s (core/safe-upgrade-state (catalog->term catalog) pkg (ver-term ver) o)]
    (if s {"ok" (normalize-verdict (answer s o))} {"fail" true})))

(defn upgrade-set [catalog pkg ver]
  (let [o (out-var)
        s (core/upgrade-set-result-state (catalog->term catalog) pkg (ver-term ver) o)]
    (if s (normalize-upgrade (answer s o)) {"fail" true})))

(defn freeze-audit [catalog]
  (let [o (out-var)
        s (core/freeze-audit-state (catalog->term catalog) o)]
    (if s {"ok" (or (answer s o) [])} {"fail" true})))

(defn dependents [catalog pkg]
  (let [o (out-var)
        s (core/dependents-state (catalog->term catalog) pkg o)]
    (if s {"ok" (answer s o)} {"fail" true})))

(defn dependents-installed [catalog pkg]
  (let [o (out-var)
        s (core/dependents-installed-state (catalog->term catalog) pkg o)]
    (if s {"ok" (answer s o)} {"fail" true})))

;; ---------------------------------------------------------------------------
;; The corpus / differential entry point (the twin of resolver.mjs's runCase)
;; ---------------------------------------------------------------------------

(defn run-case [row]
  (let [cat (get row "catalog")
        q (get row "query")
        args (get row "args")]
    (case q
      "resolve"              (resolve-q cat args)
      "resolve_layered"      (resolve-layered cat args)
      "explain_blocked"      (explain-blocked cat args)
      "layer_closure"        (layer-closure cat args)
      "removal_orphans"      (removal-orphans cat args)
      "safe_upgrade"         (safe-upgrade cat (nth args 0) (nth args 1))
      "upgrade_set"          (upgrade-set cat (nth args 0) (nth args 1))
      "freeze_audit"         (freeze-audit cat)
      "dependents"           (dependents cat args)
      "dependents_installed" (dependents-installed cat args)
      (throw (ex-info (str "unknown query " q) {})))))
