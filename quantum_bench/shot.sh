#!/bin/bash
# Saves a screenshot of Quantum Bench (or of one tab's solo binary), e.g. for
# slides or to check a scene. The window pops up for a moment.
#
#   ./shot.sh out.png                          full app, first tab, preset 0, t = 0
#   ./shot.sh out.png wave1d 0 3.5             tab wave1d, preset 0, at t = 3.5
#   ./shot.sh out.png wave1d 0 4 "view_both;measure_x@2"
#                                              actions (Tab::action); "@t" runs one at time t
#   BIN=spin_solo ./shot.sh out.png            a solo binary instead of the full app
#   PRESENT=1 ./shot.sh …                      presentation mode
#   LIGHT=1 ./shot.sh …                        light theme
#   SEED=7 SIZE=1600x1000 WAIT=1 ./shot.sh …   RNG seed, window size, extra wait (s)
#
# Builds in release mode first (quick when nothing changed).
set -e
cd "$(dirname "$0")"
[[ -z "$1" ]] && { sed -n '2,15p' "$0"; exit 1; }
BIN="${BIN:-quantum_bench}"
case "$BIN" in
    quantum_bench) PKG=quantum_bench ;;
    *_solo) PKG="qb_${BIN%_solo}" ;;
    *) echo "unknown BIN=$BIN"; exit 1 ;;
esac
cargo build --release -q -p "$PKG" --bin "$BIN"
OUT="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
export QB_SHOT="$OUT"
[[ -n "$2" ]] && export QB_TAB="$2"
[[ -n "$3" ]] && export QB_PRESET="$3"
[[ -n "$4" ]] && export QB_T="$4"
[[ -n "$5" ]] && export QB_ACTION="$5"
[[ -n "$PRESENT" ]] && export QB_PRESENT="$PRESENT"
[[ -n "$LIGHT" ]] && export QB_LIGHT="$LIGHT"
[[ -n "$SEED" ]] && export QB_SEED="$SEED"
[[ -n "$SIZE" ]] && export QB_SIZE="$SIZE"
[[ -n "$WAIT" ]] && export QB_WAIT="$WAIT"
./target.nosync/release/"$BIN"
