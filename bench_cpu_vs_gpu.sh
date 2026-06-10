#!/usr/bin/env bash
set -euo pipefail

CORES=$(nproc)
END=${1:-100}
OUTDIR=.benchmark
mkdir -p "$OUTDIR" bench/cpu bench/gpu

./bench_cpu write bench_initial 1000000021ff12ff 1000000012ff21ff 2>/dev/null

echo ""
echo "=== CPU (${CORES} cores, sum 16..${END}) ==="
rm -f "$OUTDIR"/* bench/*.gv
time ./bench_cpu -c=${CORES} -i=bench_initial -b=${OUTDIR}/ -e=${END} generate 2>&1
cp bench/*.gv bench/cpu/gen.gv

echo ""
echo "=== GPU (sum 16..${END}) ==="
rm -f "$OUTDIR"/* bench/*.gv
time ./bench_gpu -c=${CORES} -i=bench_initial -b=${OUTDIR}/ -e=${END} generate 2>&1
cp bench/*.gv bench/gpu/gen.gv

echo ""
echo "=== GEN_MOVE timing per layer (ms) ==="
printf "%-10s %10s %10s %10s\n" "layer" "cpu_ms" "gpu_ms" "speedup"
join \
  <(grep 't(ms)' bench/cpu/gen.gv | grep 'GEN_MOVE[0-9]' \
      | sed 's/.*GEN_MOVE\([0-9]*\) \[label.*t(ms): \([0-9]*\).*/\1 \2/' | sort -n) \
  <(grep 't(ms)' bench/gpu/gen.gv | grep 'GEN_MOVE[0-9]' \
      | sed 's/.*GEN_MOVE\([0-9]*\) \[label.*t(ms): \([0-9]*\).*/\1 \2/' | sort -n) \
| awk 'BEGIN{sum_cpu=0; sum_gpu=0}
       { cpu=$2; gpu=$3;
         sp = (gpu>0) ? cpu/gpu : 0;
         printf "%-10s %10s %10s %10.2fx\n", $1, cpu, gpu, sp;
         sum_cpu+=cpu; sum_gpu+=gpu }
       END{ sp = (sum_gpu>0) ? sum_cpu/sum_gpu : 0;
            printf "%-10s %10s %10s %10.2fx\n", "TOTAL", sum_cpu, sum_gpu, sp }'
