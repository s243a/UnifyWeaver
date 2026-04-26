package generated.lmdb;

import java.io.IOException;
import java.nio.file.Path;
import java.util.concurrent.ConcurrentHashMap;
import java.util.HashMap;
import java.util.Map;

public final class LmdbArtifactReader {
    private enum CachePolicy {
        NONE,
        MEMOIZE,
        SHARED,
        TWO_LEVEL
    }

    private static final ConcurrentHashMap<String, LmdbRow[]> SHARED_ARG1_CACHE = new ConcurrentHashMap<>();

    private final LmdbArtifactManifest manifest;
    private final ThreadLocal<LmdbArtifactStore> threadLocalStore;
    private final CachePolicy cachePolicy;
    private final String sharedCachePrefix;
    private final ThreadLocal<Map<String, LmdbRow[]>> threadLocalArg1Cache;

    public LmdbArtifactReader(Path artifactDir, LmdbArtifactManifest manifest) {
        this(artifactDir, manifest, CachePolicy.NONE);
    }

    private LmdbArtifactReader(Path artifactDir, LmdbArtifactManifest manifest, CachePolicy cachePolicy) {
        this.manifest = manifest;
        this.cachePolicy = cachePolicy;
        this.sharedCachePrefix = artifactDir.toAbsolutePath().normalize() + "::";
        this.threadLocalStore = ThreadLocal.withInitial(() -> LmdbArtifactStore.open(artifactDir, manifest));
        this.threadLocalArg1Cache = usesThreadLocalCache(cachePolicy)
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
            CachePolicy.MEMOIZE
        );
    }

    public static LmdbArtifactReader openSharedCached(Path artifactDir) throws IOException {
        return new LmdbArtifactReader(
            artifactDir,
            LmdbArtifactManifest.read(artifactDir.resolve("manifest.json")),
            CachePolicy.SHARED
        );
    }

    public static LmdbArtifactReader openTwoLevel(Path artifactDir) throws IOException {
        return new LmdbArtifactReader(
            artifactDir,
            LmdbArtifactManifest.read(artifactDir.resolve("manifest.json")),
            CachePolicy.TWO_LEVEL
        );
    }

    public LmdbRow[] lookupArg1(String key) {
        return switch (cachePolicy) {
            case NONE -> threadLocalStore.get().lookupArg1(key);
            case MEMOIZE -> lookupMemoized(key);
            case SHARED -> lookupShared(key);
            case TWO_LEVEL -> lookupTwoLevel(key);
        };
    }

    private LmdbRow[] lookupMemoized(String key) {
        Map<String, LmdbRow[]> cache = threadLocalArg1Cache.get();
        LmdbRow[] cached = cache.get(key);
        if (cached != null) {
            return cached;
        }
        LmdbRow[] rows = threadLocalStore.get().lookupArg1(key);
        cache.put(key, rows);
        return rows;
    }

    private LmdbRow[] lookupShared(String key) {
        return SHARED_ARG1_CACHE.computeIfAbsent(
            sharedCacheKey(key),
            ignored -> threadLocalStore.get().lookupArg1(key)
        );
    }

    private LmdbRow[] lookupTwoLevel(String key) {
        Map<String, LmdbRow[]> cache = threadLocalArg1Cache.get();
        LmdbRow[] cached = cache.get(key);
        if (cached != null) {
            return cached;
        }
        LmdbRow[] rows = SHARED_ARG1_CACHE.computeIfAbsent(
            sharedCacheKey(key),
            ignored -> threadLocalStore.get().lookupArg1(key)
        );
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

    private static boolean usesThreadLocalCache(CachePolicy cachePolicy) {
        return cachePolicy == CachePolicy.MEMOIZE || cachePolicy == CachePolicy.TWO_LEVEL;
    }

    private String sharedCacheKey(String key) {
        return sharedCachePrefix + key;
    }
}
