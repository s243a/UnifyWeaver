#!/usr/bin/env sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname "$0")/../.." && pwd)
PROBE_DIR="$ROOT_DIR/examples/scala_lmdb_jni_probe"
OUT_DIR="$PROBE_DIR/out"
CLASSES_DIR="$OUT_DIR/classes"
LIB_DIR="$OUT_DIR/lib"
ARTIFACT_DIR="$OUT_DIR/artifact"
CLOJURE_SRC_DIR="$PROBE_DIR/src/main/clojure"

JAVA_BIN=${JAVA_BIN:-$(command -v java)}
JAVAC_BIN=${JAVAC_BIN:-$(command -v javac)}
SCALAC_BIN=${SCALAC_BIN:-$(command -v scalac)}
GCC_BIN=${GCC_BIN:-$(command -v gcc)}
CARGO_BIN=${CARGO_BIN:-$(command -v cargo)}

JAVAC_REAL=$(readlink -f "$JAVAC_BIN")
JAVA_HOME_DIR=$(CDPATH= cd -- "$(dirname "$JAVAC_REAL")/.." && pwd)
JNI_INCLUDE="$JAVA_HOME_DIR/include"
JNI_PLATFORM_INCLUDE="$JNI_INCLUDE/linux"
SCALA_LIB_JAR=${SCALA_LIB_JAR:-/data/data/com.termux/files/usr/opt/scala/maven2/org/scala-lang/scala-library/3.8.3/scala-library-3.8.3.jar}
CLOJURE_JAR=${CLOJURE_JAR:-/data/data/com.termux/files/usr/var/lib/proot-distro/installed-rootfs/debian/usr/share/java/clojure-1.11.1.jar}
CLOJURE_SPEC_JAR=${CLOJURE_SPEC_JAR:-/data/data/com.termux/files/usr/var/lib/proot-distro/installed-rootfs/debian/usr/share/java/spec.alpha-0.3.218.jar}
CLOJURE_CORE_SPECS_JAR=${CLOJURE_CORE_SPECS_JAR:-/data/data/com.termux/files/usr/var/lib/proot-distro/installed-rootfs/debian/usr/share/java/core.specs.alpha-0.2.62.jar}

mkdir -p "$CLASSES_DIR" "$LIB_DIR" "$ARTIFACT_DIR"

"$CARGO_BIN" run --manifest-path "$ROOT_DIR/examples/lmdb_relation_artifact/Cargo.toml" -- \
  build edge/2 "$PROBE_DIR/testdata/edge.tsv" "$ARTIFACT_DIR" --dupsort

"$JAVAC_BIN" -d "$CLASSES_DIR" -h "$OUT_DIR" "$PROBE_DIR"/src/main/java/generated/lmdb/*.java
cat "$PROBE_DIR/native/lmdb_artifact_jni.c" | "$GCC_BIN" -x c -shared -fPIC \
  -include stddef.h \
  -I"$JNI_INCLUDE" -I"$JNI_PLATFORM_INCLUDE" \
  -I/data/data/com.termux/files/usr/include \
  -L/data/data/com.termux/files/usr/lib \
  -o "$LIB_DIR/liblmdb_artifact_jni.so" \
  - -llmdb

"$SCALAC_BIN" -d "$CLASSES_DIR" -classpath "$CLASSES_DIR" \
  "$PROBE_DIR/src/main/scala/generated/lmdb/LmdbArtifactProbe.scala"

"$JAVA_BIN" -Djava.library.path="$LIB_DIR" -cp "$CLASSES_DIR:$SCALA_LIB_JAR" generated.lmdb.LmdbArtifactProbe "$ARTIFACT_DIR"

exec "$JAVA_BIN" -Djava.library.path="$LIB_DIR" -cp "$CLASSES_DIR:$SCALA_LIB_JAR:$CLOJURE_JAR:$CLOJURE_SPEC_JAR:$CLOJURE_CORE_SPECS_JAR:$CLOJURE_SRC_DIR" clojure.main -m generated.lmdb.clojure-probe "$ARTIFACT_DIR"
