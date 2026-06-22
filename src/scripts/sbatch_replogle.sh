#!/bin/bash
#SBATCH --job-name=pvae-repl
#SBATCH --partition=guest_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm-pvae-repl-%j.out
set -e
cd /work/sextonlab/bfentaw/perturbvae
module purge || true; module load cuda/13.0 || module load cuda || true
source /work/sextonlab/bfentaw/Diffusion_project/.venv/bin/activate
H=data/replogle_k562_essential/perturb_processed.h5ad
echo "=== BASELINES (replogle) ==="
python3 src/scripts/run_baselines.py --h5ad $H --out results/baselines_replogle.json
echo "=== CVAE (replogle) ==="
python3 src/scripts/train.py --h5ad $H --out results/cvae_replogle_delta.json --epochs 40
echo "ALL_DONE_REPLOGLE"
