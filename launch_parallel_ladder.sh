#!/usr/bin/env bash
# launch_parallel_ladder.sh
# Runs CFCA-I Ablation Ladder across 5 seeds in parallel across GPU 3, GPU 1, and GPU 0.

PYTHON=/DATA/raghvendra3088/MFND_env/bin/python
SEEDS="42 123 129 2024 3407"

mkdir -p logs predictions

echo "=========================================================="
echo "Launching Fast CFCA-I Ablation Ladder on Free GPUs (3, 1, 0)"
echo "Started at: $(date)"
echo "=========================================================="

# Worker 1 on GPU 3 (M5 Full CFCA-I -> M4 +L_FG)
(
    echo "--- [GPU 3] Launching M5 (Full CFCA-I) ---"
    $PYTHON -u train_cfcai.py --model M5 --gpu 3 --seeds $SEEDS --max_epochs 30 > logs/ladder_M5.log 2>&1
    echo "--- [GPU 3] M5 Finished! Launching M4 (+L_FG) ---"
    $PYTHON -u train_cfcai.py --model M4 --gpu 3 --seeds $SEEDS --max_epochs 30 > logs/ladder_M4.log 2>&1
    echo "--- [GPU 3] M4 Finished! ---"
) &
PID_GPU3=$!

# Worker 2 on GPU 1 (M5a Clean CFCA-I No Meta -> M6 Bilinear Control)
(
    echo "--- [GPU 1] Launching M5a (Clean CFCA-I No Meta) ---"
    $PYTHON -u train_cfcai.py --model M5a --gpu 1 --seeds $SEEDS --max_epochs 30 > logs/ladder_M5a.log 2>&1
    echo "--- [GPU 1] M5a Finished! Launching M6 (Bilinear Control) ---"
    $PYTHON -u train_cfcai.py --model M6 --gpu 1 --seeds $SEEDS --max_epochs 30 > logs/ladder_M6.log 2>&1
    echo "--- [GPU 1] M6 Finished! ---"
) &
PID_GPU1=$!

# Worker 3 on GPU 0 (M2 CA Uniform -> M3 CA+Importance)
(
    echo "--- [GPU 0] Launching M2 (CA Uniform) ---"
    $PYTHON -u train_cfcai.py --model M2 --gpu 0 --seeds $SEEDS --max_epochs 30 > logs/ladder_M2.log 2>&1
    echo "--- [GPU 0] M2 Finished! Launching M3 (CA+Importance) ---"
    $PYTHON -u train_cfcai.py --model M3 --gpu 0 --seeds $SEEDS --max_epochs 30 > logs/ladder_M3.log 2>&1
    echo "--- [GPU 0] M3 Finished! ---"
) &
PID_GPU0=$!

echo "Processes launched in parallel:"
echo "  GPU 3 (M5, M4)  PID: $PID_GPU3"
echo "  GPU 1 (M5a, M6) PID: $PID_GPU1"
echo "  GPU 0 (M2, M3)  PID: $PID_GPU0"

# Wait for all workers to finish
wait $PID_GPU3 $PID_GPU1 $PID_GPU0

echo "=========================================================="
echo "All Parallel Ablation Runs Completed at $(date)!"
echo "Compiling statistical evaluation and McNemar tests..."
echo "=========================================================="

$PYTHON eval_ablation_ladder.py > logs/ladder_final_results.log 2>&1
cat logs/ladder_final_results.log
