package generated.lmdb

import scala.io.Source
import scala.util.matching.Regex

object LmdbArtifactProbe {
  final case class Manifest(dbName: String, dupsort: Boolean)

  private val dbNamePattern: Regex = """"db_name"\s*:\s*"([^"]+)"""".r
  private val dupsortPattern: Regex = """"dupsort"\s*:\s*(true|false)""".r

  def main(args: Array[String]): Unit = {
    if (args.length != 1) {
      System.err.println("usage: LmdbArtifactProbe <artifact-dir>")
      sys.exit(1)
    }

    val artifactDir = args(0)
    val manifest = readManifest(s"$artifactDir/manifest.json")

    val lookupRows = splitRows(LmdbArtifactJNI.lookupRaw(artifactDir, manifest.dbName, "a", manifest.dupsort))
    val scanRows = splitRows(LmdbArtifactJNI.scanRaw(artifactDir, manifest.dbName))

    require(lookupRows == Vector("a\t1", "a\t2"), s"unexpected lookup rows: $lookupRows")
    require(scanRows.contains("a\t1"), s"scan missing a\\t1: $scanRows")
    require(scanRows.contains("a\t2"), s"scan missing a\\t2: $scanRows")
    require(scanRows.contains("b\t3"), s"scan missing b\\t3: $scanRows")

    println(s"scala_lmdb_jni_probe_ok lookup=${lookupRows.mkString(",")} scan_count=${scanRows.size} db=${manifest.dbName}")
  }

  private def readManifest(path: String): Manifest = {
    val content = Source.fromFile(path, "UTF-8").mkString
    val dbName = dbNamePattern.findFirstMatchIn(content).map(_.group(1)).getOrElse("edge/2")
    val dupsort = dupsortPattern.findFirstMatchIn(content).exists(_.group(1) == "true")
    Manifest(dbName, dupsort)
  }

  private def splitRows(raw: String): Vector[String] = {
    if (raw == null || raw.isEmpty) Vector.empty
    else raw.split("\n").iterator.filter(_.nonEmpty).toVector
  }
}
