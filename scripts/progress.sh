#!/bin/bash
# One-shot visual progress for the Step 7 run. Usage:  bash scripts/progress.sh
cd "$(dirname "$0")/.."
LOG=.cache/step7_run.log
LIST=.cache/videos150.txt

# done = how many of the selected videos already have a per-video feature file
TOTAL=$(tr ',' '\n' < "$LIST" | grep -c .)
DONE=0
for v in $(tr ',' ' ' < "$LIST"); do
  [ -f "features/frames_siglip_32f/${v%.mp4}.npy" ] && DONE=$((DONE+1))
done
PCT=$(( DONE * 100 / TOTAL ))
FILL=$(( PCT * 40 / 100 ))
BAR="$(printf "%${FILL}s" | tr ' ' '#')$(printf "%$((40-FILL))s" | tr ' ' '.')"

echo "+------------------------------------------------+"
echo "  STEP 7 : SigLIP frame extraction (32 frames)"
echo "  [$BAR]"
echo "  $DONE/$TOTAL videos done  ($PCT%)"
pgrep -f eda_frame_features.py >/dev/null && echo "  status: RUNNING" || echo "  status: NOT RUNNING"
echo "+-- last 3 videos -------------------------------+"
grep -oE "\[[0-9:]{8}\] \[[0-9]+/[0-9]+\] [^ ]+ +[^|]*" "$LOG" | tail -3 | sed 's/^/  /'
echo "+------------------------------------------------+"
