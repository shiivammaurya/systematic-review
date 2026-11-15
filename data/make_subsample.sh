#!/usr/bin/env bash
# Build the small CICIoT2023 subsample used in the experiments.
# For every attack/benign class folder in the official CICIoT2023 "CSV" directory, take the first CSV file
# (alphabetical order) and keep its header plus the first 5,000 rows. The class name is the folder name.
#
# Usage:  bash data/make_subsample.sh /path/to/CICIoT2023/CSV
# Output: data/ciciot_small/<class>.csv  and  data/ciciot_small.zip
set -euo pipefail
SRC="${1:?Usage: bash data/make_subsample.sh /path/to/CICIoT2023/CSV}"
OUT="$(cd "$(dirname "$0")" && pwd)/ciciot_small"
ROWS=5001   # header + 5,000 rows
mkdir -p "$OUT"
cd "$SRC"
for d in */; do
  n="${d%/}"
  f=$(find "$d" -name "*.csv" | sort | head -1)
  [ -n "$f" ] && head -n "$ROWS" "$f" > "$OUT/$n.csv"
done
cd "$OUT" && wc -l ./*.csv | tail -1
cd "$(dirname "$OUT")" && zip -rq ciciot_small.zip ciciot_small && ls -lh ciciot_small.zip
echo "Done: $(ls "$OUT" | wc -l) class files."
