# Scala LMDB JNI Probe

This is a standalone JVM-side probe for consuming the LMDB relation-artifact contract through JNI.

It is intentionally isolated from the current Haskell and Elixir target paths.

## What it does

- builds a tiny LMDB relation artifact from `testdata/edge.tsv`
- parses `manifest.json` in Scala
- uses JNI-backed C code to:
  - open the named LMDB DB from the manifest
  - perform `arg1` lookup
  - perform full scan
- asserts the returned rows match the fixture

## Files

- `src/main/scala/generated/lmdb/LmdbArtifactProbe.scala`
- `src/main/java/generated/lmdb/LmdbArtifactJNI.java`
- `native/lmdb_artifact_jni.c`
- `build.sh`

## Run

```sh
sh examples/scala_lmdb_jni_probe/build.sh
```

Expected success output:

```text
scala_lmdb_jni_probe_ok lookup=a	1,a	2 scan_count=3 db=edge/2
```

## Notes

- this uses only local toolchain pieces already present in Termux:
  - `cargo`
  - `java`
  - `javac`
  - `scalac`
  - `gcc`
  - installed `liblmdb`
- manifest parsing is kept in Scala to avoid adding a JSON dependency or JNI-side JSON parser
- LMDB access itself is done through JNI against the native `liblmdb`
- this is a probe, not yet a generated Scala target integration
