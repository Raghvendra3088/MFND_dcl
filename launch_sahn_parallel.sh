#!/bin/bash
# ==============================================================================
# launch_sahn_parallel.sh
# Multi-GPU Parallel Execution Engine for SAHN-DCL
# Distributes experiments across free GPUs (GPU 2 and GPU 3) simultaneously.
# ==============================================================================

set -e

PY="/DATA/raghvendra3088/MFND_env/bin/python"
WORKDIR="/DATA/raghvendra3088/MFND_project/files"
LOGDIR="$WORKDIR/logs"
mkdir -p "$LOGDIR"
mkdir -p "$WORKDIR/predictions"
mkdir -p "$WORKDIR/checkpoints"
cd "$WORKDIR"

echo "=============================================================================="
echo "LAUNCHING SAHN-DCL EXPERIMENTS IN PARALLEL ACROSS FREE GPUs (GPU 2 & GPU 3)"
echo "Time: $(date)"
echo "=============================================================================="

# Stream 1: Assigned to GPU 2
run_stream_gpu2() {
    echo "[GPU 2 Stream] Starting E0 (Baseline MFND-DCL)..."
    $PY train_sahn.py --exp_name E0 --gpu 2 > "$LOGDIR/sahn_E0.log" 2>&1
    echo "[GPU 2 Stream] Finished E0."

    echo "[GPU 2 Stream] Starting E1 (Masking Only)..."
    $PY train_sahn.py --exp_name E1 --gpu 2 > "$LOGDIR/sahn_E1.log" 2>&1
    echo "[GPU 2 Stream] Finished E1."

    echo "[GPU 2 Stream] Starting E2 (SAHN Sample Loss Only)..."
    $PY train_sahn.py --exp_name E2 --gpu 2 --use_sahn_sample --alpha_sample 1.0 > "$LOGDIR/sahn_E2.log" 2>&1
    echo "[GPU 2 Stream] Finished E2."

    echo "[GPU 2 Stream] Starting E6 (Fine-Grained CFCA-I + SAHN-DCL)..."
    $PY train_sahn.py --exp_name E6 --gpu 2 --is_token_arch --use_sahn_sample --use_sahn_modal --use_ambiguity --alpha_sample 1.0 --alpha_modal 0.8 --beta 0.5 > "$LOGDIR/sahn_E6.log" 2>&1
    echo "[GPU 2 Stream] Finished E6."
}

# Stream 2: Assigned to GPU 3
run_stream_gpu3() {
    echo "[GPU 3 Stream] Starting E0a (Clean Baseline without Meta)..."
    $PY train_sahn.py --exp_name E0a --gpu 3 --drop_temporal > "$LOGDIR/sahn_E0a.log" 2>&1
    echo "[GPU 3 Stream] Finished E0a."

    echo "[GPU 3 Stream] Starting E3 (SAHN Modality Loss Only)..."
    $PY train_sahn.py --exp_name E3 --gpu 3 --use_sahn_modal --alpha_modal 0.8 > "$LOGDIR/sahn_E3.log" 2>&1
    echo "[GPU 3 Stream] Finished E3."

    echo "[GPU 3 Stream] Starting E4 (Combined SAHN-DCL: Sample + Modal)..."
    $PY train_sahn.py --exp_name E4 --gpu 3 --use_sahn_sample --use_sahn_modal --alpha_sample 1.0 --alpha_modal 0.8 > "$LOGDIR/sahn_E4.log" 2>&1
    echo "[GPU 3 Stream] Finished E4."

    echo "[GPU 3 Stream] Starting E5 (Full SAHN-DCL with Ambiguity Proxy)..."
    $PY train_sahn.py --exp_name E5 --gpu 3 --use_sahn_sample --use_sahn_modal --use_ambiguity --alpha_sample 1.0 --alpha_modal 0.8 --beta 0.5 > "$LOGDIR/sahn_E5.log" 2>&1
    echo "[GPU 3 Stream] Finished E5."
}

# Launch streams in parallel
run_stream_gpu2 &
PID_GPU2=$!

run_stream_gpu3 &
PID_GPU3=$!

echo "Stream 1 PID on GPU 2: $PID_GPU2"
echo "Stream 2 PID on GPU 3: $PID_GPU3"

# Wait for both streams to complete
wait $PID_GPU2
echo "Stream 1 (GPU 2) successfully completed!"

wait $PID_GPU3
echo "Stream 2 (GPU 3) successfully completed!"

echo "=============================================================================="
echo "ALL PARALLEL EXPERIMENTS FINISHED! RUNNING STATISTICAL EVALUATION..."
echo "=============================================================================="
$PY eval_sahn.py
echo "=============================================================================="
echo "SAHN-DCL PIPELINE COMPLETE AT $(date)"
echo "=============================================================================="
