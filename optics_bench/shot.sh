#!/bin/bash
# Saves a screenshot of the app, e.g. for slides.
#   ./shot.sh out.png 4                 ray-optics example number 4 (0 = first in the menu)
#   ./shot.sh out.png f2                Fourier-optics (4f) example number 2
#   ./shot.sh out.png my_config.json    a saved configuration
#   ./shot.sh out.png 4 3d              ... with the 3D bench view
# The window pops up for a few seconds while the image converges.
cd "$(dirname "$0")"
OUT="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
if [[ "$2" == *.json ]]; then
    export OPTICS_LOAD="$(cd "$(dirname "$2")" && pwd)/$(basename "$2")"
elif [[ "$2" == f* ]]; then
    export OPTICS_FOURIER="${2#f}"
else
    export OPTICS_PRESET="${2:-0}"
fi
[[ "$3" == "3d" ]] && export OPTICS_BENCH3D=1
OPTICS_SHOT="$OUT" OPTICS_SHOT_SECS="${SECS:-5}" ./target/release/optics_bench
