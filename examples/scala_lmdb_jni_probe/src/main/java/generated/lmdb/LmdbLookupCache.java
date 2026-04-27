package generated.lmdb;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.LongAdder;
import java.util.function.Supplier;

final class LmdbLookupCache {
    enum Policy {
        NONE,
        MEMOIZE,
        SHARED,
        TWO_LEVEL
    }

    private static final ConcurrentHashMap<String, LmdbRow[]> SHARED_ARG1_CACHE = new ConcurrentHashMap<>();

    private final Policy policy;
    private final String sharedCachePrefix;
    private final ThreadLocal<Map<String, LmdbRow[]>> threadLocalArg1Cache;
    private final LongAdder localHits;
    private final LongAdder sharedHits;
    private final LongAdder misses;

    LmdbLookupCache(Policy policy, String sharedCachePrefix) {
        this.policy = policy;
        this.sharedCachePrefix = sharedCachePrefix;
        this.threadLocalArg1Cache = usesThreadLocalCache(policy)
            ? ThreadLocal.withInitial(HashMap::new)
            : null;
        this.localHits = new LongAdder();
        this.sharedHits = new LongAdder();
        this.misses = new LongAdder();
    }

    LmdbRow[] lookupArg1(String key, Supplier<LmdbRow[]> storeLookup) {
        return switch (policy) {
            case NONE -> lookupDirect(storeLookup);
            case MEMOIZE -> lookupMemoized(key, storeLookup);
            case SHARED -> lookupShared(key, storeLookup);
            case TWO_LEVEL -> lookupTwoLevel(key, storeLookup);
        };
    }

    LmdbCacheStats snapshot() {
        return new LmdbCacheStats(
            policy.name().toLowerCase(),
            localHits.sum(),
            sharedHits.sum(),
            misses.sum()
        );
    }

    void resetCurrentThreadStats() {
        if (threadLocalArg1Cache != null) {
            threadLocalArg1Cache.remove();
        }
        localHits.reset();
        sharedHits.reset();
        misses.reset();
    }

    private LmdbRow[] lookupDirect(Supplier<LmdbRow[]> storeLookup) {
        misses.increment();
        return storeLookup.get();
    }

    private LmdbRow[] lookupMemoized(String key, Supplier<LmdbRow[]> storeLookup) {
        Map<String, LmdbRow[]> cache = threadLocalArg1Cache.get();
        LmdbRow[] cached = cache.get(key);
        if (cached != null) {
            localHits.increment();
            return cached;
        }
        misses.increment();
        LmdbRow[] rows = storeLookup.get();
        cache.put(key, rows);
        return rows;
    }

    private LmdbRow[] lookupShared(String key, Supplier<LmdbRow[]> storeLookup) {
        String sharedKey = sharedCacheKey(key);
        LmdbRow[] cached = SHARED_ARG1_CACHE.get(sharedKey);
        if (cached != null) {
            sharedHits.increment();
            return cached;
        }
        misses.increment();
        LmdbRow[] rows = SHARED_ARG1_CACHE.computeIfAbsent(sharedKey, ignored -> storeLookup.get());
        return rows;
    }

    private LmdbRow[] lookupTwoLevel(String key, Supplier<LmdbRow[]> storeLookup) {
        Map<String, LmdbRow[]> localCache = threadLocalArg1Cache.get();
        LmdbRow[] localRows = localCache.get(key);
        if (localRows != null) {
            localHits.increment();
            return localRows;
        }

        String sharedKey = sharedCacheKey(key);
        LmdbRow[] sharedRows = SHARED_ARG1_CACHE.get(sharedKey);
        if (sharedRows != null) {
            sharedHits.increment();
            localCache.put(key, sharedRows);
            return sharedRows;
        }

        misses.increment();
        LmdbRow[] rows = SHARED_ARG1_CACHE.computeIfAbsent(sharedKey, ignored -> storeLookup.get());
        localCache.put(key, rows);
        return rows;
    }

    private static boolean usesThreadLocalCache(Policy policy) {
        return policy == Policy.MEMOIZE || policy == Policy.TWO_LEVEL;
    }

    private String sharedCacheKey(String key) {
        return sharedCachePrefix + key;
    }
}
