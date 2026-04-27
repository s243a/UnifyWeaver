package generated.lmdb;

public record LmdbCacheStats(
    String policy,
    long localHits,
    long sharedHits,
    long misses
) {}
