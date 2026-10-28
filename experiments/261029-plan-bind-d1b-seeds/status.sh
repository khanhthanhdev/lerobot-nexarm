#!/usr/bin/env bash
# Prints where a D3 run stands: GPUs, finished stages, each trained policy's checkpoints, eval cells done,
# and the last line of every worker's log. Run it in a second terminal; `watch -n 30 ...` keeps it fresh.
#   experiments/261029-plan-bind-d1b-seeds/status.sh [full|smoke]
EXP_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT="$EXP_DIR/outputs/${1:-full}"
[ -d "$OUT" ] || { echo "no run yet at $OUT"; exit 1; }

echo "== $(date '+%F %T')  $OUT"
echo "-- GPUs (index, util %, memory used/total MiB)"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo "nvidia-smi not found"

echo "-- finished stages"
stages=$(ls -A "$OUT" | sed -n 's/^\.done_//p' | tr '\n' ' ')
echo "${stages:-(none yet)}"

echo "-- training (latest checkpoint step per policy and seed)"
for dir in "$OUT"/train_*; do
  [ -d "$dir/checkpoints" ] || continue
  last=$(ls -1 "$dir/checkpoints" 2>/dev/null | grep -E '^[0-9]+$' | sort | tail -n1)
  printf '%-28s step %s\n' "$(basename "$dir")" "${last:-none}"
done

done_cells=$(find "$OUT/eval" -name eval_info.json 2>/dev/null | wc -l)
echo "-- eval cells finished: $done_cells"
ls -1t "$OUT/eval"/*/eval_info.json 2>/dev/null | head -n3 | while read -r f; do
  dirname "$f" | xargs basename
done | sed 's/^/   latest: /'

echo "-- worker logs (last line)"
for log in "$OUT"/logs/worker_*.log; do
  [ -f "$log" ] || continue
  printf '%s: ' "$(basename "$log" .log)"
  tail -c 600 "$log" | tr '\r' '\n' | grep -v '^$' | tail -n1 | cut -c1-200
done
