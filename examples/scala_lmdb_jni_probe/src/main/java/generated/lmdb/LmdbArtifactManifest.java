package generated.lmdb;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public record LmdbArtifactManifest(String dbName, boolean dupsort) {
    private static final Pattern DB_NAME_PATTERN = Pattern.compile("\"db_name\"\\s*:\\s*\"([^\"]+)\"");
    private static final Pattern DUPSORT_PATTERN = Pattern.compile("\"dupsort\"\\s*:\\s*(true|false)");

    public static LmdbArtifactManifest read(Path manifestPath) throws IOException {
        String content = Files.readString(manifestPath, StandardCharsets.UTF_8);
        String dbName = matchOrDefault(DB_NAME_PATTERN, content, "edge/2");
        boolean dupsort = "true".equals(matchOrDefault(DUPSORT_PATTERN, content, "false"));
        return new LmdbArtifactManifest(dbName, dupsort);
    }

    private static String matchOrDefault(Pattern pattern, String input, String fallback) {
        Matcher matcher = pattern.matcher(input);
        return matcher.find() ? matcher.group(1) : fallback;
    }
}
