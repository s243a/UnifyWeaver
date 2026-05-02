(ns generated.lmdb.clojure-probe
  (:require [clojure.string :as str])
  (:gen-class)
  (:import [generated.lmdb LmdbArtifactReader]))

(defn -main
  [& args]
  (when (not= 1 (count args))
    (binding [*out* *err*]
      (println "usage: generated.lmdb.clojure-probe <artifact-dir>"))
    (System/exit 1))
  (let [artifact-dir (first args)
        reader (LmdbArtifactReader/open (java.nio.file.Path/of artifact-dir (make-array String 0)))
        lookup-rows (map (fn [row] (str (.key row) "\t" (.value row)))
                         (.lookupArg1 reader "a"))
        scan-rows (set (map (fn [row] (str (.key row) "\t" (.value row)))
                            (.scan reader)))]
    (when (not= ["a\t1" "a\t2"] (vec lookup-rows))
      (throw (ex-info "unexpected lookup rows" {:rows (vec lookup-rows)})))
    (when-not (contains? scan-rows "a\t1")
      (throw (ex-info "scan missing a\t1" {:rows scan-rows})))
    (when-not (contains? scan-rows "a\t2")
      (throw (ex-info "scan missing a\t2" {:rows scan-rows})))
    (when-not (contains? scan-rows "b\t3")
      (throw (ex-info "scan missing b\t3" {:rows scan-rows})))
    (println
      (str "clojure_lmdb_jni_probe_ok lookup="
           (str/join "," lookup-rows)
           " scan_count=" (count scan-rows)
           " db=" (.dbName (.manifest reader))))))
