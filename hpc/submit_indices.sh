#!/bin/bash

#SBATCH --job-name=UAV_Zugspitze_snow_topo_analysis
#SBATCH --exclude=hpdar01c03s04
#SBATCH --account=pn39sa-c
#SBATCH --partition=hpda2_compute
#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --export=NONE
#SBATCH --get-user-env
#SBATCH --cpus-per-task=25
#SBATCH --mem=300GB
#SBATCH --output=/dss/dsstbyfs02/pn49ci/pn49ci-dss-0026/di54xen/experiments/UAV_Zugspitze/slurm_logs/%j_out_log.out
#SBATCH --error=/dss/dsstbyfs02/pn49ci/pn49ci-dss-0026/di54xen/experiments/UAV_Zugspitze/slurm_logs/%j_err_log.err
# optional:
#SBATCH --mail-type=END
#SBATCH --mail-user=kjelldre.wundram@gmail.com

# Setup
logs="$WORK/logs"
module load slurm_setup
module load python/3.10
module load uv
cd /dss/dsshome1/02/di54xen/projects/zugspitze/UAV_Zugspitze
source /dss/dsshome1/02/di54xen/projects/zugspitze/UAV_Zugspitze/.venv/bin/activate


# Run training
echo "[SLURM] Running topo_indices with UV..."
PYTHONUNBUFFERED=1 uv run topo_indices
echo "[SLURM] UV done."
# Organize logs into dated subdirectories
LOG_BASE="/dss/dsstbyfs02/pn49ci/pn49ci-dss-0026/di54xen/experiments/UAV_Zugspitze/slurm_logs"
LOG_DATE=$(date +%Y-%m-%d)
LOG_DIR="${LOG_BASE}/${LOG_DATE}"
mkdir -p "${LOG_DIR}"

# Move SLURM output and error logs to dated directory
if [ -f "${LOG_BASE}/${SLURM_JOB_ID}_out_log.out" ]; then
  mv "${LOG_BASE}/${SLURM_JOB_ID}_out_log.out" "${LOG_DIR}/"
fi
if [ -f "${LOG_BASE}/${SLURM_JOB_ID}_err_log.err" ]; then
  mv "${LOG_BASE}/${SLURM_JOB_ID}_err_log.err" "${LOG_DIR}/"
fi
