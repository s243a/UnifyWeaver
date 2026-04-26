package generated.lmdb

object LmdbArtifactProbe {
  def main(args: Array[String]): Unit = {
    if (args.length != 1) {
      System.err.println("usage: LmdbArtifactProbe <artifact-dir>")
      sys.exit(1)
    }

    val artifactDir = args(0)
    val reader = LmdbArtifactReader.open(java.nio.file.Path.of(artifactDir))

    val lookupRows = reader.lookupArg1("a").iterator.map(row => s"${row.key}\t${row.value}").toVector
    val scanRows = reader.scan().iterator.map(row => s"${row.key}\t${row.value}").toVector

    require(lookupRows == Vector("a\t1", "a\t2"), s"unexpected lookup rows: $lookupRows")
    require(scanRows.contains("a\t1"), s"scan missing a\\t1: $scanRows")
    require(scanRows.contains("a\t2"), s"scan missing a\\t2: $scanRows")
    require(scanRows.contains("b\t3"), s"scan missing b\\t3: $scanRows")

    println(s"scala_lmdb_jni_probe_ok lookup=${lookupRows.mkString(",")} scan_count=${scanRows.size} db=${reader.manifest.dbName}")
  }
}
