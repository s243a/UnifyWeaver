package generated.lmdb;

import java.io.IOException;
import java.nio.file.Path;

public final class LmdbArtifactReader {
    private final Path artifactDir;
    private final LmdbArtifactManifest manifest;

    public LmdbArtifactReader(Path artifactDir, LmdbArtifactManifest manifest) {
        this.artifactDir = artifactDir;
        this.manifest = manifest;
    }

    public static LmdbArtifactReader open(Path artifactDir) throws IOException {
        return new LmdbArtifactReader(
            artifactDir,
            LmdbArtifactManifest.read(artifactDir.resolve("manifest.json"))
        );
    }

    public LmdbRow[] lookupArg1(String key) {
        return LmdbArtifactJNI.lookupRows(
            artifactDir.toString(),
            manifest.dbName(),
            key,
            manifest.dupsort()
        );
    }

    public LmdbRow[] scan() {
        return LmdbArtifactJNI.scanRows(
            artifactDir.toString(),
            manifest.dbName()
        );
    }

    public LmdbArtifactManifest manifest() {
        return manifest;
    }
}
