#!/bin/bash
# Runs the experiments of Section V in sequence; finished jobs are skipped on a re-run.
cd "$(dirname "$0")/.."
mkdir -p results
for e in X1 X3 X4 X5 X8; do
  python3 -m experiments.run --exp $e --jobs ${JOBS:-4} 2>&1 | tee -a results/$e.log
done
python3 -m experiments.x6_scalability 2>&1 | tee results/X6.log
python3 -m experiments.x7_variability 2>&1 | tee results/X7.log
python3 -m experiments.plots
