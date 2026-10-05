#!/bin/bash
# Unattended: wait for the self-play collection (PID $1) to exit, then align, cut clips,
# train and write the report. Logs in training/runs/. Run from training/:
#   nohup ./run_full_pipeline.sh <collection_pid> > runs/pipeline.log 2>&1 &
set -e
cd "$(dirname "$0")"
if [ -n "$1" ]; then
    while kill -0 "$1" 2>/dev/null; do sleep 60; done
fi
echo "$(date) collection done, starting pipeline"
.venv/bin/python selfplay_align.py > runs/align.log 2>&1
.venv/bin/python selfplay_clips.py > runs/clips_full.log 2>&1
# A GPU fault kills the process; resume from the last epoch (up to 3 tries).
for try in 1 2 3; do
    .venv/bin/python train_plays.py --epochs 12 --resume >> runs/train_plays.log 2>&1 && break
    echo "$(date) training failed (try $try), resuming"
    sleep 60
done
.venv/bin/python plays_report.py > runs/report.log 2>&1
echo "$(date) pipeline done: see runs/plays/report.md"
