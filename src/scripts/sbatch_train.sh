#!/bin/bash
#SBATCH --job-name=pv-cvae
#SBATCH --account=sextonlab
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu_h200
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:40:00
#SBATCH --output=logs/slurm-pvcvae-%j.out
#SBATCH --error=logs/slurm-pvcvae-%j.err
set -euo pipefail
module purge || true; module load cuda/13.0 || module load cuda || true
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source /work/sextonlab/bfentaw/Diffusion_project/.venv/bin/activate
export TMPDIR=/work/sextonlab/bfentaw/tmp OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
mkdir -p logs results
python3 src/scripts/train.py --h5ad data/norman/perturb_processed.h5ad --out results/cvae_norman_delta.json --epochs 40
