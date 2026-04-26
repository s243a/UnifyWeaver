package generated.lmdb;

import java.nio.file.Path;

final class LmdbArtifactStore implements AutoCloseable {
    private final long handle;
    private final boolean dupsort;

    private LmdbArtifactStore(long handle, boolean dupsort) {
        this.handle = handle;
        this.dupsort = dupsort;
    }

    static LmdbArtifactStore open(Path artifactDir, LmdbArtifactManifest manifest) {
        long handle = LmdbArtifactJNI.openStore(
            artifactDir.toString(),
            manifest.dbName(),
            manifest.dupsort()
        );
        if (handle == 0L) {
            throw new IllegalStateException("failed to open LMDB artifact store");
        }
        return new LmdbArtifactStore(handle, manifest.dupsort());
    }

    LmdbRow[] lookupArg1(String key) {
        return LmdbArtifactJNI.lookupRowsForHandle(handle, key, dupsort);
    }

    LmdbRow[] scan() {
        return LmdbArtifactJNI.scanRowsForHandle(handle);
    }

    @Override
    public void close() {
        LmdbArtifactJNI.closeStore(handle);
    }
}
