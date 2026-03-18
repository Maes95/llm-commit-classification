#!/bin/bash

#!/bin/bash

# Script to run annotation validation tests with different context mode combinations
# Executes sequentially for the same model:
# r1: "message"
# r2: "single-label"
# r3: "single-label+few-shot"
# r4: "diff+single-label"
# r5: "diff+single-label+few-shot"
#
# Uso: ./runAll.sh [MODEL_NAME]
# Ejemplo: ./runAll.sh "ollama/gpt-oss:20b"

MODEL="${1:-ollama/gpt-oss:20b}"  # Parámetro 1 o valor por defecto
INPUT_FILE="data/50-random-commits-validation-with-diff.jsonl"
WORKERS=10

source .venv/bin/activate

echo "=========================================="
echo "Starting annotation validation runs"
echo "Model: $MODEL"
echo "Input: $INPUT_FILE"
echo "Workers: $WORKERS"
echo "=========================================="

# Tuple-like mapping via associative array (run_id -> context_mode)
declare -A RUN_MODE=(
  [r1]="message"
  [r2]="single-label"
  [r3]="single-label+few-shot"
  [r4]="diff+single-label"
  [r5]="diff+single-label+few-shot"
)

# Explicit order because associative arrays are unordered
ORDER=(r1 r2 r3 r4 r5)
TOTAL_RUNS=${#ORDER[@]}

for i in "${!ORDER[@]}"; do
  run_id="${ORDER[$i]}"
  context_mode="${RUN_MODE[$run_id]}"
  run_number=$((i + 1))

  echo ""
  echo "[$run_number/$TOTAL_RUNS] Running with context-mode: $context_mode"
  echo "Output directory: output/$run_id/"

  python annotate_validation_set.py \
    --context-mode "$context_mode" \
    --model "$MODEL" \
    --workers $WORKERS \
    --input "$INPUT_FILE" \
    --output "output/$run_id/" \
    --no-colors

  if [ $? -ne 0 ]; then
    echo "Run $run_id failed!"
    exit 1
  fi
done

echo ""
echo "=========================================="
echo "All annotation runs completed successfully!"
echo "=========================================="
