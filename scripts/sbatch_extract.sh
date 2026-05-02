#!/bin/bash
# ============================================================
# sbatch_extract.sh — overnight feature extraction
# PlantCLEF 2026, JPEG 4:2:2 q85, scales [6,5,4,3,2,1]
#
# Submit:  sbatch scripts/sbatch_extract.sh
# Monitor: tail -f logs/extract_<jobid>.log
# ============================================================

#SBATCH --job-name=plantclef_extract
#SBATCH --partition=gpu-preempt          # change to gpu if you want non-preemptable
#SBATCH --gres=gpu:v100:1               # V100-16GB; change to a100 if available
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=logs/extract_%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=calpeeppuff@gmail.com

# ── Environment ──────────────────────────────────────────────
set -euo pipefail
PROJECT=/path/to/PlantCLEF2026        # ← update this to your scratch3 path
cd "$PROJECT"

mkdir -p logs

echo "==== Job started: $(date) ===="
echo "Node: $SLURM_NODELIST"
echo "GPU:  $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

# Activate conda environment (update env name if different)
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate orchard

echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__, torch.cuda.is_available())')"

# ── Dry-run first: check image count and pending work ────────
echo ""
echo "==== Dry-run ===="
python scripts/extract_overnight.py --dry-run

# ── Full extraction ───────────────────────────────────────────
echo ""
echo "==== Extraction ===="
python scripts/extract_overnight.py

echo ""
echo "==== Job finished: $(date) ===="
