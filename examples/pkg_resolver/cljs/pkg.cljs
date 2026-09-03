#!/usr/bin/env nbb
;; SPDX-License-Identifier: MIT OR Apache-2.0
;; Copyright (c) 2026 John William Creighton (s243a)
;;
;; pkg.cljs -- the `pkg` CLI in ClojureScript. The twin of ../cli/pkg.mjs:
;; same commands, same documents, same messages, same exit codes.
;;
;;   argv --> transpiled argparser (examples/cli_args/cli_args.pl, the D40
;;            ClojureScript build, driven with cli/generated/pkg_registry.json)
;;        --> dispatch
;;        --> transpiled resolver  (examples/pkg_resolver/resolver.pl, via
;;            ./resolver.cljs over the ClojureScript-WAM build)
;;        --> output
;;
;; WHAT IS IN HERE, exhaustively:
;;   1. conversion   -- version strings <-> [M,I,P] triples, catalog file ->
;;                      JSON data, resolver answers -> the documented JSON doc;
;;   2. dispatch     -- which command calls which resolver query;
;;   3. formatting   -- the human tables and the JSON serialisation;
;;   4. exit codes   -- a table keyed by the doc's `status`.
;;
;; There is NO parse logic (not one flag rule, not one arity check, not one
;; error message -- every usage error you will ever see from `pkg` is a string
;; produced by compiled cli_args) and NO resolve logic (not one candidate
;; ordering, constraint comparison, layer walk, freeze rule or upgrade closure
;; -- those are all compiled resolver.pl).
;;
;; The single deliberate exception is `pkg deps`, which is a *projection* over
;; the catalog's own `depends` rows rather than a resolver query -- exactly as
;; in pkg.mjs, and called out there and here.
;;
;; Documents are built as JS objects with the keys asserted in pkg.mjs's own
;; order, so `--json` output is byte-for-byte what pkg.mjs prints, not merely
;; deeply equal to it.
;;
;;   nbb --classpath examples/cli_args/cljs:examples/pkg_resolver/cljs \
;;       examples/pkg_resolver/cljs/pkg.cljs resolve editor --catalog cat.json

(ns pkg
  (:require [resolver :as R]
            [generated.cli-args :as ca]
            [clojure.string :as str]
            [nbb.core :refer [*file*]]
            ["fs" :as fs]
            ["path" :as path]))

;; nbb binds *file* to the script's own path; the registry sits at a fixed
;; offset from it, exactly as pkg.mjs resolves it from import.meta.url.
(def HERE (.dirname path *file*))

(defn- read-json [p] (js->clj (js/JSON.parse (.readFileSync fs p "utf8"))))

;; The command grammar, rendered from cli/pkg_schema.pl by cli/derive.pl --
;; the SAME file pkg.mjs reads.
(def REGISTRY
  (read-json (.join path HERE ".." "cli" "generated" "pkg_registry.json")))

;; Spellings that name the same query. `layer` is Pkg's `sfs-combine` word.
(def COMMAND-ALIASES {"layer" "install-plan"})

;; status -> process exit code. 0 success, 1 query-false/blocked, 2 usage.
(def EXIT-FOR-STATUS
  {"ok" 0 "clear" 0 "blocked" 1 "fail" 1 "not-frozen" 1})

;; A usage error, carrying compiled cli_args' own message where it has one.
(defn- cli-error [msg] (ex-info msg {::cli true}))
(defn- cli-error? [e] (::cli (ex-data e)))

;; JS truthiness, which is what pkg.mjs's `flags.catalog || env` relies on:
;; the empty string is falsy there and must be here too.
(defn- js-truthy? [x] (and (some? x) (not (false? x)) (not= x "")))

;; ---------------------------------------------------------------------------
;; JS object construction, key order preserved
;; ---------------------------------------------------------------------------

(defn- jso [& kvs]
  (let [o (js-obj)]
    (doseq [[k v] (partition 2 kvs)] (aset o k v))
    o))

(defn- jsarr [coll] (to-array (vec coll)))

;; ---------------------------------------------------------------------------
;; 1. conversion
;; ---------------------------------------------------------------------------

(defn- ver-str [v] (str (nth v 0) "." (nth v 1) "." (nth v 2)))

(defn- parse-ver [s]
  (let [m (re-matches #"(\d+)\.(\d+)\.(\d+)" (str s))]
    (when-not m
      (throw (cli-error (str "bad version: " s " (expected MAJOR.MINOR.PATCH)"))))
    [(js/Number (nth m 1)) (js/Number (nth m 2)) (js/Number (nth m 3))]))

(defn- pair-doc [p] (jso "name" (nth p 0) "version" (ver-str (nth p 1))))

;; The catalog/resolver constraint forms, as one tagged JSON object.
(defn- constraint-doc [c]
  (cond
    (or (nil? c) (= c "any")) (jso "op" "any")
    (= (get c "op") "range")
    (jso "op" "range" "lo" (ver-str (get c "lo")) "hi" (ver-str (get c "hi")))
    (#{"eq" "gte" "lt"} (get c "op"))
    (jso "op" (get c "op") "version" (ver-str (get c "v")))
    :else (throw (ex-info (str "pkg: unhandled constraint " (pr-str c)) {}))))

(defn- load-catalog [p]
  (let [text (try (.readFileSync fs p "utf8")
                  (catch :default e
                    (throw (cli-error (str "cannot read catalog " p ": "
                                           (or (.-code e) (.-message e)))))))]
    (try (js->clj (js/JSON.parse text))
         (catch :default e
           (throw (cli-error (str "catalog " p " is not valid JSON: " (.-message e))))))))

;; ---------------------------------------------------------------------------
;; 2. dispatch -- command -> resolver query
;; ---------------------------------------------------------------------------

(defn- failed? [r] (get r "fail"))

(defn- cmd-resolve [cat names _flags]
  (let [r (R/resolve-q cat names)]
    (if (failed? r)
      (jso "command" "resolve" "status" "fail" "requests" (jsarr names)
           "reason" "no_solution")
      (jso "command" "resolve" "status" "ok" "requests" (jsarr names)
           "selection" (jsarr (map pair-doc (get r "ok")))))))

(defn- cmd-install-plan [cat names _flags]
  (let [lay (R/resolve-layered cat names)]
    (if (failed? lay)
      (jso "command" "install-plan" "status" "fail" "requests" (jsarr names)
           "reason" "no_solution" "manifests" (jsarr []))
      (loop [todo names manifests []]
        (if (empty? todo)
          (jso "command" "install-plan" "status" "ok" "requests" (jsarr names)
               "selection" (jsarr (map pair-doc (get lay "ok")))
               "manifests" (jsarr manifests))
          (let [n (first todo)
                lc (R/layer-closure cat n)]
            (if (failed? lc)
              (jso "command" "install-plan" "status" "fail" "requests" (jsarr names)
                   "reason" (str "no_manifest_for:" n) "manifests" (jsarr []))
              (recur (rest todo)
                     (conj manifests
                           (jso "request" n
                                "order" (jsarr (map pair-doc (get lc "ok")))))))))))))

(defn- cmd-why-blocked [cat [name] _flags]
  (let [b (R/explain-blocked cat name)]
    (if (failed? b)
      (jso "command" "why-blocked" "status" "fail" "request" name
           "reason" "query_failed" "blocked" (jsarr []))
      (let [blocked (mapv (fn [x] (jso "name" (get x "name")
                                       "needs" (constraint-doc (get x "needs"))
                                       "base_has" (ver-str (get x "base_has"))))
                          (get b "ok"))]
        (jso "command" "why-blocked"
             "status" (if (seq blocked) "blocked" "clear")
             "request" name
             "blocked" (jsarr blocked))))))

(defn- cmp-ver [a b]
  (or (first (remove zero? [(- (nth a 0) (nth b 0))
                            (- (nth a 1) (nth b 1))
                            (- (nth a 2) (nth b 2))]))
      0))

(defn- cmp-str [a b] (cond (< a b) -1 (> a b) 1 :else 0))

;; The one catalog projection: `depends` rows for this name, no interpretation.
(defn- cmd-deps [cat [name] _flags]
  (let [rows (->> (get cat "depends" [])
                  (filter #(= (nth % 0) name))
                  (sort (fn [a b]
                          (let [c (cmp-ver (nth a 1) (nth b 1))]
                            (if (zero? c) (cmp-str (nth a 2) (nth b 2)) c))))
                  (mapv (fn [r] (jso "version" (ver-str (nth r 1))
                                     "dep" (nth r 2)
                                     "constraint" (constraint-doc (nth r 3))))))]
    (jso "command" "deps" "status" "ok" "package" name "depends" (jsarr rows))))

(defn- cmd-what-needs [cat [name] flags]
  (let [only (true? (get flags "installed"))
        d (if only (R/dependents-installed cat name) (R/dependents cat name))]
    (if (failed? d)
      (jso "command" "what-needs" "status" "fail" "package" name
           "installed_only" only "reason" "query_failed" "dependents" (jsarr []))
      (jso "command" "what-needs" "status" "ok" "package" name
           "installed_only" only
           "dependents" (jsarr (map pair-doc (get d "ok")))))))

(defn- cmd-orphans [cat [name] _flags]
  (let [o (R/removal-orphans cat name)]
    (if (failed? o)
      (jso "command" "orphans" "status" "fail" "package" name
           "reason" "query_failed" "orphans" (jsarr []))
      (jso "command" "orphans" "status" "ok" "package" name
           "orphans" (jsarr (map pair-doc (get o "ok")))))))

(defn- cmd-why-frozen [cat [name] _flags]
  (let [a (R/freeze-audit cat)]
    (if (failed? a)
      (jso "command" "why-frozen" "status" "fail" "package" name
           "reason" "query_failed" "kind" nil)
      (let [hit (first (filter #(= (get % "name") name) (get a "ok")))]
        (if (nil? hit)
          (jso "command" "why-frozen" "status" "not-frozen" "package" name
               "kind" nil "reason" nil)
          (jso "command" "why-frozen" "status" "ok" "package" name
               "kind" (get hit "kind")
               "reason" (if (contains? hit "reason") (get hit "reason") nil)))))))

(defn- cmd-audit [cat _args _flags]
  (let [a (R/freeze-audit cat)]
    (if (failed? a)
      (jso "command" "audit" "status" "fail" "reason" "query_failed" "audit" (jsarr []))
      (jso "command" "audit" "status" "ok"
           "audit" (jsarr (map (fn [row]
                                 (jso "name" (get row "name")
                                      "kind" (get row "kind")
                                      "reason" (if (contains? row "reason")
                                                 (get row "reason") nil)))
                               (get a "ok")))))))

(defn- cmd-safe-upgrade [cat [name version] _flags]
  (let [ver (parse-ver version)
        s (R/safe-upgrade cat name ver)]
    (if (failed? s)
      (jso "command" "safe-upgrade" "package" name "version" (ver-str ver)
           "status" "fail" "reason" "query_failed" "result" nil)
      (let [v (get s "ok")
            verdict (get v "verdict")]
        (case verdict
          "safe"
          (jso "command" "safe-upgrade" "package" name "version" (ver-str ver)
               "status" "ok" "result" (jso "verdict" "safe" "cost" (get v "cost")))
          "coordinated"
          (jso "command" "safe-upgrade" "package" name "version" (ver-str ver)
               "status" "ok"
               "result" (jso "verdict" "coordinated"
                             "set" (jsarr (map pair-doc (get v "set")))))
          "unsafe"
          (jso "command" "safe-upgrade" "package" name "version" (ver-str ver)
               "status" "fail"
               "result" (jso "verdict" "unsafe" "reason" (get v "reason")))
          "no_candidate"
          (jso "command" "safe-upgrade" "package" name "version" (ver-str ver)
               "status" "fail" "result" (jso "verdict" "no_candidate"))
          (throw (ex-info (str "pkg: unhandled verdict " (pr-str v)) {})))))))

(def DISPATCH
  {"resolve" cmd-resolve
   "install-plan" cmd-install-plan
   "why-blocked" cmd-why-blocked
   "deps" cmd-deps
   "what-needs" cmd-what-needs
   "orphans" cmd-orphans
   "why-frozen" cmd-why-frozen
   "audit" cmd-audit
   "safe-upgrade" cmd-safe-upgrade})

;; ---------------------------------------------------------------------------
;; 3. formatting
;; ---------------------------------------------------------------------------

(defn- rtrim [s] (str/replace s #"\s+$" ""))

(defn- table
  ([rows] (table rows "  "))
  ([rows indent]
   (if (empty? rows)
     ""
     (let [widths (reduce (fn [w r]
                            (reduce (fn [w [i c]]
                                      (assoc w i (max (get w i 0) (count (str c)))))
                                    w (map-indexed vector r)))
                          {} rows)]
       (str/join "\n"
                 (map (fn [r]
                        (rtrim
                          (str indent
                               (str/join "  "
                                         (map-indexed
                                           (fn [i c]
                                             (if (= i (dec (count r)))
                                               (str c)
                                               (let [s (str c)
                                                     pad (- (get widths i 0) (count s))]
                                                 (str s (apply str (repeat (max 0 pad) " "))))))
                                           r)))))
                      rows))))))

(defn constraint-text [c]
  (let [op (aget c "op")]
    (cond
      (= op "any") "*"
      (= op "range") (str ">=" (aget c "lo") ",<" (aget c "hi"))
      :else (str ({"eq" "==" "gte" ">=" "lt" "<"} op) (aget c "version")))))

(defn- plural [n one many] (str n " " (if (= n 1) one many)))

(defn- js-seq [a] (vec (array-seq a)))

;; doc -> the human-readable form. A pure function of the JSON document.
(defn render-human [doc]
  (let [g #(aget doc %)
        L (case (g "command")
            "resolve"
            (if (= (g "status") "fail")
              [(str "no solution for: " (str/join " " (js-seq (g "requests"))))
               "(the catalog has no candidate set that satisfies every request)"]
              [(str "selection for " (str/join " " (js-seq (g "requests")))
                    " — " (plural (alength (g "selection")) "package" "packages"))
               (table (map (fn [p] [(aget p "name") (aget p "version")])
                           (js-seq (g "selection"))))])

            "install-plan"
            (if (= (g "status") "fail")
              [(str "no layered plan for: " (str/join " " (js-seq (g "requests"))))
               (str "(run `pkg why-blocked " (aget (g "requests") 0) "` for the ceiling)")]
              (into [(str "plan for " (str/join " " (js-seq (g "requests")))
                          " — " (plural (alength (g "selection")) "package" "packages")
                          " (frozen base untouched)")
                     (if (zero? (alength (g "selection")))
                       "  nothing to install: every request is already satisfied by a loaded layer"
                       (table (map (fn [p] [(aget p "name") (aget p "version")])
                                   (js-seq (g "selection")))))]
                    (mapcat (fn [m]
                              [(str "install order for " (aget m "request") ":")
                               (if (zero? (alength (aget m "order")))
                                 "  (empty)"
                                 (table (map-indexed
                                          (fn [i p] [(str (inc i) ".") (aget p "name") (aget p "version")])
                                          (js-seq (aget m "order")))))])
                            (js-seq (g "manifests")))))

            "why-blocked"
            (if (= (g "status") "clear")
              [(str (g "request") " is not blocked by the frozen base")]
              [(str (g "request") " is blocked by the frozen base — "
                    (plural (alength (g "blocked")) "ceiling" "ceilings"))
               (table (map (fn [b] [(aget b "name")
                                    (str "needs " (constraint-text (aget b "needs")))
                                    (str "base has " (aget b "base_has"))])
                           (js-seq (g "blocked"))))])

            "deps"
            [(str "direct deps of " (g "package") " — "
                  (plural (alength (g "depends")) "row" "rows") " (catalog projection)")
             (if (pos? (alength (g "depends")))
               (table (map (fn [d] [(aget d "version") (aget d "dep")
                                    (constraint-text (aget d "constraint"))])
                           (js-seq (g "depends"))))
               "")]

            "what-needs"
            [(str (plural (alength (g "dependents")) "package" "packages")
                  " directly need " (g "package")
                  (if (g "installed_only")
                    " (installed or loaded-layer only)"
                    " (all catalog versions)"))
             (if (pos? (alength (g "dependents")))
               (table (map (fn [p] [(aget p "name") (aget p "version")])
                           (js-seq (g "dependents"))))
               "")]

            "orphans"
            (if (zero? (alength (g "orphans")))
              [(str "removing " (g "package") " would orphan nothing")]
              [(str "removing " (g "package") " would orphan "
                    (plural (alength (g "orphans")) "package" "packages"))
               (table (map (fn [p] [(aget p "name") (aget p "version")])
                           (js-seq (g "orphans"))))])

            "why-frozen"
            (cond
              (= (g "status") "not-frozen")
              [(str (g "package") " is not held in the frozen base")]
              (= (g "kind") "held")
              [(str (g "package") " is held in the frozen base: " (g "reason"))]
              (= (g "kind") "suggest")
              [(str (g "package") " is held in the frozen base: blanket")
               (str "  a base package pins it — suggest " (g "reason"))]
              :else
              [(str (g "package") " is held in the frozen base: blanket")
               "  nothing in the base pins it — over-frozen"])

            "audit"
            (let [rows (js-seq (g "audit"))
                  over (count (filter #(= (aget % "kind") "over_frozen") rows))
                  sug (count (filter #(= (aget % "kind") "suggest") rows))]
              [(str "freeze audit — " (plural (count rows) "hold" "holds"))
               (table (map (fn [a] [(aget a "name") (aget a "kind")
                                    (if (nil? (aget a "reason")) "" (aget a "reason"))])
                           rows))
               (str over " over-frozen, " sug " with a suggested reason")])

            "safe-upgrade"
            (let [head (str (g "package") " -> " (g "version"))
                  r (g "result")]
              (cond
                (nil? r) [(str head ": query failed")]
                (= (aget r "verdict") "safe") [(str head ": safe (cost: " (aget r "cost") ")")]
                (= (aget r "verdict") "unsafe") [(str head ": unsafe (" (aget r "reason") ")")]
                (= (aget r "verdict") "no_candidate")
                [(str head ": no candidate")
                 "  (no such catalog version, or the package is not a frozen-base hold)"]
                :else
                [(str head ": coordinated — "
                      (plural (alength (aget r "set")) "package" "packages")
                      " must move together")
                 (table (map (fn [p] [(aget p "name") (aget p "version")])
                             (js-seq (aget r "set"))))]))

            (throw (ex-info (str "pkg: no renderer for " (g "command")) {})))]
    (str (str/join "\n" (remove #(= % "") L)) "\n")))

;; ---------------------------------------------------------------------------
;; 4. the run
;; ---------------------------------------------------------------------------

(def USAGE
  (str/join "\n"
    ["usage: pkg <command> [args] [--catalog <file>] [--json]"
     ""
     "  resolve <name>...          classic closure (may move the frozen base)"
     "  install-plan <name>...     layered closure + install order  (alias: layer)"
     "  why-blocked <name>         the frozen-base ceilings that stop a layered plan"
     "  deps <name>                direct catalog deps of a package"
     "  what-needs <name>          reverse deps          [--installed]"
     "  orphans <name>             what removing it would orphan"
     "  why-frozen <name>          one hold's freeze reason"
     "  audit                      every hold's freeze reason"
     "  safe-upgrade <name> <ver>  verdict + the coordinated set"
     ""
     "  --catalog <file>  catalog JSON (or set PKG_CATALOG)"
     "  --json            machine-readable output"
     ""
     "exit: 0 success · 1 query false / blocked · 2 usage error"]))

;; ---- the registry, as the compiled parser's own term shape ----------------
;; `schema(Options, Positionals)` / `group(Actions)` with `Key-Value` pairs,
;; the CLJS spelling of what cliArgs.mjs builds for the JS lane.
(defn- pair-term [k v] {:$ "-" :args [k v]})

(defn- entry-term [entry]
  (if-let [actions (get entry "actions")]
    {:$ "group" :args [(mapv (fn [k] (pair-term k (entry-term (get actions k))))
                             (keys actions))]}
    {:$ "schema"
     :args [(mapv (fn [k] (pair-term k (str (get (get entry "options" {}) k))))
                  (keys (get entry "options" {})))
            (vec (get entry "positionals" []))]}))

(defn- registry-term [registry]
  (mapv (fn [k] (pair-term k (entry-term (get registry k)))) (keys registry)))

;; ---- the ONLY parse: compiled cli_args/parse_args/3 with our registry -----
(defn- parse-args [argv registry]
  (let [r (ca/parse-args-3 (vec argv) (registry-term registry))]
    (cond
      (and (map? r) (= (:$ r) "ok") (= 2 (count (:args r))))
      {:positional (vec (nth (:args r) 0))
       :flags (reduce (fn [m p]
                        (when-not (and (map? p) (= (:$ p) "-") (= 2 (count (:args p))))
                          (throw (ex-info "pkg: expected a Key-Value pair" {})))
                        (assoc m (nth (:args p) 0) (nth (:args p) 1)))
                      {} (nth (:args r) 1))}

      (and (map? r) (= (:$ r) "error") (= 1 (count (:args r))))
      (throw (cli-error (str (nth (:args r) 0))))

      :else
      (throw (ex-info (str "pkg: parse_args answered neither ok/2 nor error/1: "
                           (pr-str r)) {})))))

(defn run [argv]
  ;; cli_args' own leading-global scan is for ITS two globals (--state/--name);
  ;; an option before the command would silently drop `pkg` onto the legacy
  ;; lenient parser, where nothing is checked. Refuse instead.
  (when (and (seq argv)
             (str/starts-with? (first argv) "--")
             (not= (first argv) "--"))
    (throw (cli-error (str "the command must come first (got " (first argv) ")"))))

  (let [{:keys [positional flags]} (parse-args argv REGISTRY)
        spelled (first positional)]
    (when (nil? spelled) (throw (cli-error "no command given")))
    (let [command (get COMMAND-ALIASES spelled spelled)
          handler (get DISPATCH command)]
      (when (nil? handler) (throw (cli-error (str "unknown command: " spelled))))
      (let [env-cat (aget (.-env js/process) "PKG_CATALOG")
            catalog-path (cond (js-truthy? (get flags "catalog")) (get flags "catalog")
                               (js-truthy? env-cat) env-cat
                               :else nil)]
        (when (nil? catalog-path)
          (throw (cli-error "no catalog: pass --catalog <file> or set PKG_CATALOG")))
        (let [catalog (load-catalog catalog-path)
              doc (handler catalog (vec (rest positional)) flags)]
          {:doc doc :json (true? (get flags "json"))})))))

(defn -main [argv]
  (let [out (try (run argv)
                 (catch :default e
                   (if (cli-error? e)
                     (do (.write (.-stderr js/process)
                                 (str "pkg: " (.-message e) "\n" USAGE "\n"))
                         (set! (.-exitCode js/process) 2)
                         nil)
                     (throw e))))]
    (when out
      (let [doc (:doc out)
            text (if (:json out)
                   (str (js/JSON.stringify doc nil 2) "\n")
                   (render-human doc))
            code (get EXIT-FOR-STATUS (aget doc "status"))]
        (.write (.-stdout js/process) text)
        (set! (.-exitCode js/process) (if (nil? code) 1 code))))))

(-main (vec *command-line-args*))
