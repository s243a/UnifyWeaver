#!/bin/sh
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 John William Creighton (s243a)
#
# plawk benchmark harness. Generates text and binary workloads, checks
# that plawk and the system awk produce IDENTICAL output on each text
# job (a correctness gate before any timing), then reports best-of-3
# wall times.
#
#   sh examples/plawk/bench/bench.sh            # N=2000000 records
#   N=100000 sh examples/plawk/bench/bench.sh   # smaller run
#
# Workloads:
#   W1 filter-count (text):   $1 == "ERROR" { e++ } { t++ }
#   W2 aggregate (text):      $3 > 500 { s += $3 ; n++ }
#   W3 aggregate (binary):    same job over i64+f64 records (plawk
#                             binary reader vs the system awk parsing
#                             the text encoding of the SAME data)
#   W4 group-by (text):       counts[$1]++ + for-in report
set -eu

N="${N:-2000000}"
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)
PLAWK="$ROOT/examples/plawk/bin/plawk"
AWK="${AWK:-awk}"
WORK="${BENCH_DIR:-${TMPDIR:-/tmp}/plawk_bench_$$}"
mkdir -p "$WORK"
trap 'rm -rf "$WORK"' EXIT

echo "# plawk bench: N=$N records, awk=$($AWK -W version 2>&1 | head -1 || true)"
echo "# workdir: $WORK"

# --- data ------------------------------------------------------------------
python3 - "$N" "$WORK" <<'PYEOF'
import random, struct, sys
n, work = int(sys.argv[1]), sys.argv[2]
random.seed(42)
levels = ["INFO", "ERROR", "WARN", "DEBUG"]
with open(f"{work}/data.txt", "w") as t, \
     open(f"{work}/nums.txt", "w") as x, \
     open(f"{work}/data.bin", "wb") as b:
    tl, xl = [], []
    for i in range(n):
        lvl = levels[random.randrange(4)]
        num = random.randrange(1000)
        val = num / 8.0
        tl.append(f"{lvl} disk {num}\n")
        xl.append(f"{num} {val}\n")
        b.write(struct.pack("<qd", num, val))
        if len(tl) >= 65536:
            t.write("".join(tl)); x.write("".join(xl)); tl, xl = [], []
    t.write("".join(tl)); x.write("".join(xl))
PYEOF
ls -l "$WORK"/data.txt "$WORK"/data.bin | awk '{print "#", $NF, $5, "bytes"}'

# --- programs ----------------------------------------------------------------
cat > "$WORK/w1.plawk" <<'EOF'
$1 == "ERROR" { e++ }
{ t++ }
END { print e, t }
EOF
W1_AWK='$1 == "ERROR" { e++ } { t++ } END { print e, t }'

cat > "$WORK/w2.plawk" <<'EOF'
$3 > 500 { s += $3 ; n++ }
END { print n, s }
EOF
W2_AWK='$3 > 500 { s += $3 ; n++ } END { print n, s }'

cat > "$WORK/w3.plawk" <<'EOF'
BEGIN { BINFMT = "i64 f64" }
$1 > 500 { n++ }
END { print n }
EOF
W3_AWK='$1 > 500 { n++ } END { print n }'

cat > "$WORK/w4.plawk" <<'EOF'
{ counts[$1]++ }
END { for (k in counts) print k, counts[k] }
EOF
W4_AWK='{ counts[$1]++ } END { for (k in counts) print k, counts[k] }'

# --- build -------------------------------------------------------------------
for w in w1 w2 w3 w4; do
    swipl "$PLAWK" build "$WORK/$w.plawk" -o "$WORK/${w}_bin" 2>/dev/null
done

# --- correctness gate ----------------------------------------------------------
sorted() { sort "$@"; }
"$WORK/w1_bin" "$WORK/data.txt" > "$WORK/w1.plawk.out"
$AWK "$W1_AWK" "$WORK/data.txt" > "$WORK/w1.awk.out"
cmp -s "$WORK/w1.plawk.out" "$WORK/w1.awk.out" || { echo "W1 OUTPUT MISMATCH"; exit 1; }
"$WORK/w2_bin" "$WORK/data.txt" > "$WORK/w2.plawk.out"
$AWK "$W2_AWK" "$WORK/data.txt" > "$WORK/w2.awk.out"
cmp -s "$WORK/w2.plawk.out" "$WORK/w2.awk.out" || { echo "W2 OUTPUT MISMATCH"; exit 1; }
"$WORK/w3_bin" "$WORK/data.bin" > "$WORK/w3.plawk.out"
$AWK "$W3_AWK" "$WORK/nums.txt" > "$WORK/w3.awk.out"
cmp -s "$WORK/w3.plawk.out" "$WORK/w3.awk.out" || { echo "W3 OUTPUT MISMATCH"; exit 1; }
"$WORK/w4_bin" "$WORK/data.txt" | sorted > "$WORK/w4.plawk.out"
$AWK "$W4_AWK" "$WORK/data.txt" | sorted > "$WORK/w4.awk.out"
cmp -s "$WORK/w4.plawk.out" "$WORK/w4.awk.out" || { echo "W4 OUTPUT MISMATCH"; exit 1; }
echo "# correctness gate: all outputs identical"

# --- timing --------------------------------------------------------------------
best_of_3() {
    best=""
    for _i in 1 2 3; do
        start=$(date +%s%N)
        "$@" > /dev/null
        end=$(date +%s%N)
        ms=$(( (end - start) / 1000000 ))
        if [ -z "$best" ] || [ "$ms" -lt "$best" ]; then best=$ms; fi
    done
    echo "$best"
}

report() { printf "%-28s %8s ms %8s ms   %sx\n" "$1" "$2" "$3" \
    "$(python3 -c "print(f'{$3/$2:.2f}')" 2>/dev/null || echo '?')"; }

printf "%-28s %11s %11s   %s\n" "workload" "plawk" "$AWK" "awk/plawk"
P=$(best_of_3 "$WORK/w1_bin" "$WORK/data.txt")
A=$(best_of_3 $AWK "$W1_AWK" "$WORK/data.txt")
report "W1 filter-count (text)" "$P" "$A"
P=$(best_of_3 "$WORK/w2_bin" "$WORK/data.txt")
A=$(best_of_3 $AWK "$W2_AWK" "$WORK/data.txt")
report "W2 aggregate (text)" "$P" "$A"
P=$(best_of_3 "$WORK/w3_bin" "$WORK/data.bin")
A=$(best_of_3 $AWK "$W3_AWK" "$WORK/nums.txt")
report "W3 aggregate (binary)" "$P" "$A"
P=$(best_of_3 "$WORK/w4_bin" "$WORK/data.txt")
A=$(best_of_3 $AWK "$W4_AWK" "$WORK/data.txt")
report "W4 group-by (text)" "$P" "$A"
