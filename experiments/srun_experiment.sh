#!/bin/bash

# Script interactivo con srun para ejecutar el experimento
# Uso: ./srun_experiment.sh

set -e

PROJECT_DIR="$HOME/llm-commit-classification"
OLLAMA_HOST="127.0.0.1:1995"
OLLAMA_PORT="1995"
MODEL_NAME="gpt-oss:20b"

# Crear directorio de logs con timestamp
TIMESTAMP=$(date +%s)
LOG_DIR="$PROJECT_DIR/experiments/logs/interactive_${TIMESTAMP}"
EXPERIMENT_FILE="$LOG_DIR/interactive_${MODEL_NAME//:/}.experiment"

echo "========================================"
echo "LLM Commit Classification Experiment"
echo "con srun (Interactivo)"
echo "========================================"

# Crear directorio de logs
mkdir -p "$LOG_DIR"

# Crear archivo de identificación del experimento
touch "$EXPERIMENT_FILE"

cd "$PROJECT_DIR"
source .venv/bin/activate

# Kill any existing ollama processes
echo "[1/4] Limpiando procesos anteriores..."
pkill -f "ollama serve" || true
sleep 2

# Start Ollama in background
echo "[2/4] Iniciando Ollama..."
export OLLAMA_HOST="127.0.0.1:$OLLAMA_PORT"

# Use srun to allocate GPU and start ollama
srun --pty --gpus=L40S:1 ollama serve > "$LOG_DIR/ollama.log" 2>&1 &
OLLAMA_PID=$!
echo "Ollama PID: $OLLAMA_PID"

# Wait for Ollama to be ready
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
    echo "✗ Error: Ollama no respondió"
    kill $OLLAMA_PID || true
    exit 1
fi

# Pull model
echo "[3/4] Verificando/descargando modelo: $MODEL_NAME"
if ! ollama list | grep -q "$MODEL_NAME"; then
    echo "Descargando modelo $MODEL_NAME..."
    ollama pull $MODEL_NAME
else
    echo "✓ Modelo $MODEL_NAME ya está presente"
fi

# Run experiment
echo "[4/4] Ejecutando experimento..."
echo "Modelo a usar: ollama/$MODEL_NAME"
echo "====================================="

if bash experiments/runAll.sh "ollama/$MODEL_NAME"; then
    echo "====================================="
    echo "✓ Experimento completado"
    echo "====================================="
    EXIT_CODE=0
else
    echo "✗ El experimento falló"
    EXIT_CODE=1
fi

# Cleanup
echo "Limpiando procesos..."
kill $OLLAMA_PID || true
sleep 2

echo "Logs del experimento: $LOG_DIR/"
echo "Archivo de identificación: $EXPERIMENT_FILE"
exit $EXIT_CODE
