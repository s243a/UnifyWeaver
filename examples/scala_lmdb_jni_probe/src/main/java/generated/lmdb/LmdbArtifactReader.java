package generated.lmdb;

import java.io.IOException;
import java.nio.file.Path;

public final class LmdbArtifactReader {
    private final LmdbArtifactManifest manifest;
    private final ThreadLocal<LmdbArtifactStore> threadLocalStore;
    private final LmdbLookupCache lookupCache;

    public LmdbArtifactReader(Path artifactDir, LmdbArtifactManifest manifest) {
        this(artifactDir, manifest, LmdbLookupCache.Policy.NONE);
    }

    private LmdbArtifactReader(Path artifactDir, LmdbArtifactManifest manifest, LmdbLookupCache.Policy cachePolicy) {
        this.manifest = manifest;
        this.threadLocalStore = ThreadLocal.withInitial(() -> LmdbArtifactStore.open(artifactDir, manifest));
        this.lookupCache = new LmdbLookupCache(
            cachePolicy,
            artifactDir.toAbsolutePath().normalize() + "::"
        );
    }

    public static LmdbArtifactReader open(Path artifactDir) throws IOException {
        return new LmdbArtifactReader(
            artifactDir,
            LmdbArtifactManifest.read(artifactDir.resolve("manifest.json"))
        );
    }

    public static LmdbArtifactReader openMemoized(Path artifactDir) throws IOException {
        return new LmdbArtifactReader(
            artifactDir,
            LmdbArtifactManifest.read(artifactDir.resolve("manifest.json")),
            LmdbLookupCache.Policy.MEMOIZE
        );
    }

    public static LmdbArtifactReader openSharedCached(Path artifactDir) throws IOException {
        return new LmdbArtifactReader(
            artifactDir,
            LmdbArtifactManifest.read(artifactDir.resolve("manifest.json")),
            LmdbLookupCache.Policy.SHARED
        );
    }

    public static LmdbArtifactReader openTwoLevel(Path artifactDir) throws IOException {
        return new LmdbArtifactReader(
            artifactDir,
            LmdbArtifactManifest.read(artifactDir.resolve("manifest.json")),
            LmdbLookupCache.Policy.TWO_LEVEL
        );
    }

    public LmdbRow[] lookupArg1(String key) {
        return lookupCache.lookupArg1(
            key,
            () -> threadLocalStore.get().lookupArg1(key)
        );
    }

    public LmdbRow[] scan() {
        return threadLocalStore.get().scan();
    }

    public LmdbArtifactManifest manifest() {
        return manifest;
    }

    public LmdbCacheStats cacheStats() {
        return lookupCache.snapshot();
    }

    public void resetCurrentThreadStats() {
        lookupCache.resetCurrentThreadStats();
    }

    public void closeCurrentThread() {
        LmdbArtifactStore store = threadLocalStore.get();
        try {
            store.close();
        } finally {
            lookupCache.resetCurrentThreadStats();
            threadLocalStore.remove();
        }
    }
}
