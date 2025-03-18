#!/bin/bash

# Default file path
DEFAULT_FP="/gscratch/zlab/jyyh/cse503/human-eval/results/meta-llama/Llama-3.2-1B-Instruct/cad_samples_10_10.jsonl"

FP=${1:-$DEFAULT_FP}
echo "Using sample file: ${FP}"
python -m human_eval.evaluate_functional_correctness --sample_file "${FP}" --is_mbpp True 
exit()
# --score_samples 