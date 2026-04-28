# Scala LMDB JNI Probe

This is a standalone JVM-side probe for consuming the LMDB relation-artifact contract through JNI.

It is intentionally isolated from the current Haskell and Elixir target paths, but the shape is now closer to a reusable JVM seam than to a one-off Scala demo. It now validates both Scala and Clojure consumption against the same Java/JNI reader API.

## What it does

- builds a tiny LMDB relation artifact from `testdata/edge.tsv`
- parses `manifest.json` in Java
- exposes a small reusable JVM-side API:
  - `LmdbArtifactManifest`
  - `LmdbArtifactReader`
  - `LmdbArtifactStore`
  - `LmdbRow`
- uses JNI-backed C code to:
  - open the named LMDB DB from the manifest
  - perform `arg1` lookup
  - perform full scan
- asserts the returned rows match the fixture
- proves the same JVM-side seam is consumable from both Scala and Clojure

## Files

- `src/main/java/generated/lmdb/LmdbArtifactJNI.java`
- `src/main/java/generated/lmdb/LmdbArtifactManifest.java`
- `src/main/java/generated/lmdb/LmdbArtifactReader.java`
- `src/main/java/generated/lmdb/LmdbArtifactStore.java`
- `src/main/java/generated/lmdb/LmdbRow.java`
- `src/main/scala/generated/lmdb/LmdbArtifactProbe.scala`
- `src/main/clojure/generated/lmdb/clojure_probe.clj`
- `native/lmdb_artifact_jni.c`
- `build.sh`

## Run

```sh
sh examples/scala_lmdb_jni_probe/build.sh
```

Expected success output:

```text
scala_lmdb_jni_probe_ok lookup=a	1,a	2 scan_count=3 db=edge/2
clojure_lmdb_jni_probe_ok lookup=a	1,a	2 scan_count=3 db=edge/2
```

## Notes

- this uses only local toolchain pieces already present in Termux:
  - `cargo`
  - `java`
  - `javac`
  - `scalac`
  - `gcc`
  - installed `liblmdb`
- manifest parsing and reader ownership are kept in Java so JVM languages like Clojure or a future Scala hybrid WAM can reuse the same seam
- the reader now reuses one native LMDB store per JVM thread via a
  thread-local seam instead of reopening LMDB on every lookup
- the reader also exposes an optional `openMemoized(...)` constructor
  that adds a thread-local `arg1` memoization layer above the native
  store seam
- the reader also exposes `openSharedCached(...)` and `openTwoLevel(...)`
  for narrow shared-cache experiments above the same seam
- LMDB access itself is done through JNI against the native `liblmdb`
- this is still a probe, not yet a generated target integration
