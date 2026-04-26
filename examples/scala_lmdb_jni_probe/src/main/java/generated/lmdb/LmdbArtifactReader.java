package generated.lmdb;

import java.io.IOException;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;

public final class LmdbArtifactReader {
    private final LmdbArtifactManifest manifest;
    private final ThreadLocal<LmdbArtifactStore> threadLocalStore;
    private final boolean memoizeArg1;
    private final ThreadLocal<Map<String, LmdbRow[]>> threadLocalArg1Cache;

    public LmdbArtifactReader(Path artifactDir, LmdbArtifactManifest manifest) {
        this(artifactDir, manifest, false);
    }

    public LmdbArtifactReader(Path artifactDir, LmdbArtifactManifest manifest, boolean memoizeArg1) {
        this.manifest = manifest;
        this.memoizeArg1 = memoizeArg1;
        this.threadLocalStore = ThreadLocal.withInitial(() -> LmdbArtifactStore.open(artifactDir, manifest));
        this.threadLocalArg1Cache = memoizeArg1
            ? ThreadLocal.withInitial(HashMap::new)
            : null;
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
            true
        );
    }

    public LmdbRow[] lookupArg1(String key) {
        if (!memoizeArg1) {
            return threadLocalStore.get().lookupArg1(key);
        }
        Map<String, LmdbRow[]> cache = threadLocalArg1Cache.get();
        LmdbRow[] cached = cache.get(key);
        if (cached != null) {
            return cached;
        }
        LmdbRow[] rows = threadLocalStore.get().lookupArg1(key);
        cache.put(key, rows);
        return rows;
    }

    public LmdbRow[] scan() {
        return threadLocalStore.get().scan();
    }

    public LmdbArtifactManifest manifest() {
        return manifest;
    }

    public void closeCurrentThread() {
        LmdbArtifactStore store = threadLocalStore.get();
        try {
            store.close();
        } finally {
            if (threadLocalArg1Cache != null) {
                threadLocalArg1Cache.remove();
            }
            threadLocalStore.remove();
        }
    }
}
