#!/usr/bin/env bash
# Fetch the public-domain test films used in the README from the Internet Archive.
set -euo pipefail
dest="${1:-footage}"
mkdir -p "$dest"
curl -L -o "$dest/momotaro_umiwashi_1943.mp4" \
  "https://archive.org/download/momotaro-no-umiwashi-1943/Momotaro%20no%20umiwashi%20%281943%29.mp4"
curl -L -o "$dest/sora_no_momotaro_1931.mp4" \
  "https://archive.org/download/sora-no-momotaro-1931/Sora%20no%20Momotar%C3%B4%20%281931%29.mp4"
echo "done -> $dest"
echo "try: anime2stereo $dest/momotaro_umiwashi_1943.mp4 -o out --start 270 --duration 10"
