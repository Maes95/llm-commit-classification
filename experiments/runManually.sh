# python annotate_simple.py data/sample-commits/724-7392d87a9febb5f46f28d4704eb5636c5e22cdeb.json\
#  --model "ollama/gpt-oss:20b" \
#  --context-mode diff
# Custom model and input file

MODEL="ollama/gpt-oss:120b"
CONTEXT_MODE="single-label+few-shot+diff"
python annotate_validation_set.py \
--context-mode "$CONTEXT_MODE" \
--model "$MODEL" \
--workers 10 \
--input data/50-random-commits-validation-with-diff.jsonl \
--output output/r5/