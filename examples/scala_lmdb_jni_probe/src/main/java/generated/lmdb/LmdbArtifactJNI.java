package generated.lmdb;

public final class LmdbArtifactJNI {
    static {
        System.loadLibrary("lmdb_artifact_jni");
    }

    private LmdbArtifactJNI() {}

    public static native LmdbRow[] lookupRows(String artifactDir, String dbName, String key, boolean dupsort);

    public static native LmdbRow[] scanRows(String artifactDir, String dbName);
}
