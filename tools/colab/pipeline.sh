#!/bin/bash
# Run a list of Colab GPU jobs to completion across free-tier VM reclaims and across invocations of this script.
# Remembers the VM (current_session) and which job it launched there (<run>/running_on), so a later invocation
# attaches to a job still running instead of re-uploading / re-launching it. On a new VM it uploads colab_job.py, the
# teacher (if the job needs it) and the local last.pt, then launches; syncs logs/checkpoints every 3 min.
# A teacher given as distill.teacher=/content/FILE is uploaded from $WEIGHTS_DIR/FILE.
# usage: pipeline.sh JOBS_FILE [BUDGET_MIN]     JOBS_FILE lines: NAME|colab_job.py args (no --name); see jobs_example.txt
# env: COLAB_SYNC_DIR (default ~/colab_sync) holds per-run backups + state; WEIGHTS_DIR (default ~/weights)
# exit: 0 all done, 1 budget exhausted (resumable: run again), 3 a job failed on a live VM (needs a look)
S=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$S/../.." && pwd)
D=${COLAB_SYNC_DIR:-$HOME/colab_sync}; W=${WEIGHTS_DIR:-$HOME/weights}; mkdir -p $D
export COLAB_SYNC_DIR=$D
JOBS=$1; end=$(( $(date +%s) + ${2:-110} * 60 ))
C="colab --auth oauth2"; SES=$(cat $D/current_session 2>/dev/null)
left() { echo $(( (end - $(date +%s)) / 60 )); }
alive() { [ -n "$SES" ] && $C sessions < /dev/null 2>/dev/null | grep -q "$SES"; }
lsize() { stat -c %s $1/log.txt 2>/dev/null || echo 0; }

get_vm() {
  while [ $(left) -gt 25 ]; do
    SES=gp$(date +%H%M)
    out=$(timeout 300 $C new -s $SES --gpu T4 < /dev/null 2>&1 | tail -1)
    if echo "$out" | grep -q READY; then echo "$(date +%T) VM $SES ready"; echo $SES > $D/current_session; return 0; fi
    echo "$(date +%T) no GPU: ${out: -30}"; SES=""; sleep 900
  done
  return 1
}

while IFS='|' read -r NAME ARGS <&3; do
  [ -z "$NAME" ] && continue
  L=$D/$NAME; mkdir -p $L
  while [ ! -s $L/summary.json ]; do
    epid=""
    if alive && [ "$(cat $L/running_on 2>/dev/null)" = "$SES" ]; then
      echo "$(date +%T) attaching to $NAME running on $SES"
    else
      if ! alive; then
        get_vm || { echo "budget exhausted before $NAME finished"; exit 1; }
        $C upload -s $SES $REPO/tools/colab_job.py /content/colab_job.py < /dev/null > /dev/null 2>&1
        [ -s $L/last.pt ] && $S/upload.sh $SES $L/last.pt /content/runs/$NAME/last.pt < /dev/null
      fi
      teacher=$(grep -oE 'distill\.teacher=/content/[^ ]+' <<< "$ARGS" | cut -d/ -f3)
      if [ -n "$teacher" ] && [ "$(cat $D/teacher_on 2>/dev/null)" != "$SES:$teacher" ]; then
        $S/upload.sh $SES $W/$teacher /content/$teacher < /dev/null && echo "$SES:$teacher" > $D/teacher_on
      fi
      argv=$(python3 -c 'import sys, shlex; print(repr(["colab_job.py", "--name", sys.argv[1]] + shlex.split(sys.argv[2])))' "$NAME" "$ARGS")
      printf '%s\n' "import sys, runpy" "sys.argv = $argv" "_ = runpy.run_path('/content/colab_job.py', run_name='__main__')" > $L/job.py
      echo "$(date +%T) start $NAME on $SES ($(left) min left)"
      $C exec -s $SES --timeout 9000 < $L/job.py >> $L/exec.log 2>&1 &
      epid=$!; echo $SES > $L/running_on
      sleep 60
    fi
    $S/sync.sh $SES $NAME $(( $(left) - 3 )) < /dev/null &
    spid=$!; sz0=$(lsize $L); t_change=$(date +%s)
    while kill -0 $spid 2>/dev/null; do
      sz=$(lsize $L); [ "$sz" != "$sz0" ] && { sz0=$sz; t_change=$(date +%s); }
      # the job stopped writing its log on a live VM for 20 min: crashed or interrupted
      if [ $(( $(date +%s) - t_change )) -gt 1200 ] && alive; then
        kill $spid 2>/dev/null; rm -f $L/running_on
        if [ -n "$epid" ] && ! kill -0 $epid 2>/dev/null; then
          echo "$(date +%T) JOB FAILED: $NAME stopped on a live VM; exec.log tail:"; tail -5 $L/exec.log | cut -c1-200
          exit 3
        fi
        echo "$(date +%T) $NAME log stale on live VM; relaunching (resumes from its last.pt)"; break
      fi
      sleep 30
    done
    wait $spid 2>/dev/null; rc=$?
    [ -s $L/summary.json ] && { echo "$(date +%T) $NAME finished"; rm -f $L/running_on; break; }
    [ $rc -eq 2 ] && { SES=""; rm -f $D/current_session $L/running_on; echo "$(date +%T) VM lost during $NAME; will resume"; continue; }
    [ $(left) -le 25 ] && { echo "budget exhausted during $NAME (resumable; job keeps running on $SES)"; exit 1; }
  done
done 3< "$JOBS"
echo "all jobs done $(date +%T)"
