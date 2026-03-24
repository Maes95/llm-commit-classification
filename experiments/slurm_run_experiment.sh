#!/bin/bash

#SBATCH --job-name=llm-annotation-exp
#SBATCH --gpus=L40S:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=experiments/logs/%j/job_%j_out.log
#SBATCH --error=experiments/logs/%j/job_%j_err.log

# Script para ejecutar el experimento completo con Ollama
# Uso: sbatch slurm_run_experiment.sh [MODEL_NAME] [GPU_TYPE]
# Ejemplo: sbatch slurm_run_experiment.sh "gpt-oss:20b" "H100:1"

set -e  # Exit on any error

PROJECT_DIR="$HOME/llm-commit-classification"
MODEL_NAME="${1:-gpt-oss:20b}"  # Parámetro 1 o valor por defecto
GPU_TYPE="${2:-L40S:1}"          # Parámetro 2 o valor por defecto

# Generar puerto único por job (basado en JOB_ID)
# Rango: 11434-12433 (1000 puertos disponibles)
OLLAMA_PORT=$((11434 + (SLURM_JOB_ID % 1000)))
OLLAMA_HOST="127.0.0.1:$OLLAMA_PORT"

# Limpiar nombres para archivo de identificación (reemplazar : con -)
MODEL_CLEAN="${MODEL_NAME//:/}"
GPU_CLEAN="${GPU_TYPE//:/}"
SLURM_LOG_DIR="$PROJECT_DIR/experiments/logs/$SLURM_JOB_ID"

echo "========================================"
echo "LLM Commit Classification Experiment"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU configurada: $GPU_TYPE"
echo "GPU asignada: $SLURM_GPUS"
echo "Nodos: $SLURM_NODELIST"
echo "Project Dir: $PROJECT_DIR"
echo ""

cd "$PROJECT_DIR"
source .venv/bin/activate

# Crear directorio de logs si no existe
mkdir -p "$SLURM_LOG_DIR"

# Verificar si el puerto está disponible, si no esperar un poco
echo "Verificando disponibilidad del puerto $OLLAMA_PORT..."
max_wait=10
wait_count=0
while lsof -i :$OLLAMA_PORT >/dev/null 2>&1 && [ $wait_count -lt $max_wait ]; do
    echo "Puerto $OLLAMA_PORT en uso, esperando..."
    sleep 1
    wait_count=$((wait_count + 1))
done

# Start Ollama in background
echo "[2/4] Iniciando Ollama en puerto $OLLAMA_PORT..."
export OLLAMA_HOST="$OLLAMA_HOST"
export OLLAMA_BASE_URL="http://localhost:$OLLAMA_PORT"
ollama serve > "$SLURM_LOG_DIR/ollama.log" 2>&1 &
OLLAMA_PID=$!
echo "Ollama PID: $OLLAMA_PID (Puerto: $OLLAMA_PORT)"

# Wait for Ollama to start
max_retries=30
retry_count=0
echo "Esperando a que Ollama esté listo..."
while [ $retry_count -lt $max_retries ]; do
    if curl -s http://$OLLAMA_HOST/api/tags > /dev/null 2>&1; then
        echo "✓ Ollama está operativo"
        break
    fi
    echo "Intento $((retry_count + 1))/$max_retries..."
    sleep 2
    retry_count=$((retry_count + 1))
done

if [ $retry_count -eq $max_retries ]; then
    echo "✗ Error: Ollama no respondió en tiempo límite"
    kill $OLLAMA_PID || true
    exit 1
fi

# Pull model if not already present
echo "[3/4] Verificando/descargando modelo: $MODEL_NAME"
echo "GPU: $GPU_TYPE"
if ! ollama list | grep -q "$MODEL_NAME"; then
    echo "Descargando modelo $MODEL_NAME..."
    ollama pull $MODEL_NAME
else
    echo "✓ Modelo $MODEL_NAME ya está presente"
fi

# Run the experiment
echo "[4/4] Ejecutando experimento..."
echo "Modelo a usar: ollama/$MODEL_NAME"
echo "========================================="

if bash experiments/runAll.sh "ollama/$MODEL_NAME"; then
    echo "========================================="
    echo "✓ Experimento completado exitosamente"
    echo "========================================="
    EXIT_CODE=0
else
    echo "========================================="
    echo "✗ El experimento falló"
    echo "========================================="
    EXIT_CODE=1
fi

# Cleanup
echo "Limpiando procesos..."
kill $OLLAMA_PID || true
sleep 2

echo "Logs del experimento: $SLURM_LOG_DIR/"
echo "Ollama corrió en puerto: $OLLAMA_PORT"
exit $EXIT_CODE
