#!/bin/bash
# Chunked upload to a Colab VM with an integrity check (single large uploads get cut off by proxies).
# usage: upload.sh SESSION LOCAL REMOTE
SESSION=$1; LOCAL=$2; REMOTE=$3
C="colab --auth oauth2"
T=$(mktemp -d); split -b 20M -d "$LOCAL" "$T/part."
sha=$(sha256sum "$LOCAL" | cut -c1-64); dir=$(dirname "$REMOTE")
echo "import os; os.makedirs('$dir', exist_ok=True)" | $C exec -s $SESSION --timeout 60 >/dev/null 2>&1
for p in "$T"/part.*; do
  for try in 1 2 3; do
    $C upload -s $SESSION "$p" "$REMOTE.$(basename $p)" 2>&1 | grep -q Uploaded && break
    sleep 5
  done
done
printf '%s\n' "import glob, hashlib, os" "ps = sorted(glob.glob('$REMOTE.part.*'))" \
  "open('$REMOTE', 'wb').write(b''.join(open(p, 'rb').read() for p in ps)); [os.remove(p) for p in ps]" \
  "print(hashlib.sha256(open('$REMOTE', 'rb').read()).hexdigest())" \
  | $C exec -s $SESSION --timeout 120 2>&1 | tail -1 | grep -q "$sha" && echo "OK $REMOTE" || echo "HASH MISMATCH $REMOTE"
rm -rf "$T"
