#!/bin/bash

# Script para lanzar el experimento en SLURM
# Uso: ./run.sh [MODEL_NAME] [GPU_TYPE]
# Ejemplos:
#   ./run.sh "gpt-oss:20b" "L40S:1"
#   ./run.sh "gpt-oss:20b" "H100:1"
#   ./run.sh (usa valores por defecto: gpt-oss:20b, L40S:1)

set -e

# Parámetros
MODEL_NAME="${1:-gpt-oss:20b}"     # Modelo por defecto
GPU_TYPE="${2:-L40S:1}"            # GPU por defecto
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Limpiar nombres para carpeta (reemplazar : con -)
MODEL_CLEAN="${MODEL_NAME//:/}"
GPU_CLEAN="${GPU_TYPE//:/}"

echo "================================="
echo "Lanzando experimento en SLURM"
echo "Modelo: $MODEL_NAME"
echo "GPU: $GPU_TYPE"
echo "================================="
echo ""

# Lanzar sbatch con el modelo y GPU como parámetros
JOB_ID=$(sbatch --parsable --gpus="$GPU_TYPE" "$SCRIPT_DIR/slurm_run_experiment.sh" "$MODEL_NAME" "$GPU_TYPE")

PROJECT_DIR="$HOME/llm-commit-classification"
JOB_DIR="$PROJECT_DIR/experiments/logs/$JOB_ID"
MODEL_CLEAN="${MODEL_NAME//:/}"
GPU_CLEAN="${GPU_TYPE//:/}"
EXPERIMENT_FILE="$JOB_DIR/${MODEL_CLEAN}_${GPU_CLEAN}.experiment"
# Crear archivo de identificación del experimento
mkdir -p "$JOB_DIR"
touch "$EXPERIMENT_FILE"

echo "Job enviado exitosamente"
echo "Job ID: $JOB_ID"
echo ""
echo "Para ver el estado del job:"
echo "  squeue -j $JOB_ID"
echo ""
echo "Para ver los logs:"
echo "  tail -f experiments/logs/${JOB_ID}/job_${JOB_ID}_out.log"
echo ""
echo "Archivo de identificación del experimento:"
echo "  experiments/logs/${JOB_ID}/${MODEL_CLEAN}_${GPU_CLEAN}.experiment"
echo "====================================="
