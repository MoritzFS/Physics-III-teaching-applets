#!/bin/bash
# Saves a screenshot of the app, e.g. for slides or to check the drawing.
#   ./shot.sh out.png levels 0          bench (levels, gas, atoms, classical, quantum) and example number (0 = first)
#   SECS=10 ./shot.sh out.png gas 1     ... after 10 s instead of 4
# The window pops up for a few seconds. In a Linux container without a display:
#   xvfb-run -a -s "-screen 0 1600x1100x24" ./shot.sh out.png quantum 1
cd "$(dirname "$0")"
OUT="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
STAT_BENCH="${2:-levels}" STAT_EXAMPLE="${3:-0}" STAT_SHOT="$OUT" STAT_SHOT_SECS="${SECS:-4}" ./target/release/stat_phys_bench
