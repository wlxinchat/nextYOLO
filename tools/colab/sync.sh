#!/bin/bash
# Back up a Colab run every 3 min into $COLAB_SYNC_DIR/RUN: logs/history always; last.pt whenever a new (mid-epoch or
# epoch-end) checkpoint was logged; best.pt + summary at the end. Exits 2 when the VM is gone.
# usage: sync.sh SESSION RUN [MINUTES]
SESSION=$1; RUN=$2; MIN=${3:-110}
C="colab --auth oauth2"
L=${COLAB_SYNC_DIR:-$HOME/colab_sync}/$RUN; mkdir -p $L
end=$(( $(date +%s) + MIN*60 )); last_n=""; stale=0; last_sz=""
while [ $(date +%s) -lt $end ]; do
  for f in log.txt history.json summary.json; do
    $C download -s $SESSION /content/runs/$RUN/$f $L/$f.new >/dev/null 2>&1 && mv $L/$f.new $L/$f
  done
  n=$(grep -cE "checkpoint at epoch|epoch [0-9]+ done" $L/log.txt 2>/dev/null)
  if [ -n "$n" ] && [ "$n" != "0" ] && [ "$n" != "$last_n" ]; then
    sleep 20  # let the atomic save finish
    if $C download -s $SESSION /content/runs/$RUN/last.pt $L/last.pt.tmp >/dev/null 2>&1; then
      mv $L/last.pt.tmp $L/last.pt; last_n=$n
      echo "$(date +%T) saved last.pt ($(grep -E 'checkpoint at|epoch [0-9]+ done' $L/log.txt | tail -1 | cut -c1-60))"
    fi
  fi
  if [ -s $L/summary.json ]; then
    $C download -s $SESSION /content/runs/$RUN/best.pt $L/best.pt >/dev/null 2>&1
    echo "finished $(date +%T)"; break
  fi
  sz=$(stat -c %s $L/log.txt 2>/dev/null || echo 0)
  if [ "$sz" = "$last_sz" ]; then stale=$((stale + 1)); else stale=0; fi; last_sz=$sz
  if [ $stale -ge 4 ] && ! $C sessions 2>/dev/null | grep -q "$SESSION"; then
    echo "VM LOST $(date +%T): $RUN log unchanged and session $SESSION is gone (resume from last.pt)"; exit 2
  fi
  sleep 180
done
grep -E "eval ep|final" $L/log.txt | tail -6
