package generated.lmdb;

public final class LmdbArtifactJNI {
    static {
        System.loadLibrary("lmdb_artifact_jni");
    }

    private LmdbArtifactJNI() {}

    public static native String lookupRaw(String artifactDir, String dbName, String key, boolean dupsort);

    public static native String scanRaw(String artifactDir, String dbName);
}
