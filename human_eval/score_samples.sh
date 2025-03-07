#FP="/gscratch/zlab/jyyh/cse503/human-eval/results/meta-llama/Llama-3.2-1B-Instruct/cad_samples_5_pp.jsonl"
FP="/cad/cse503/human-eval/results/meta-llama/Llama-3.2-1B-Instruct/cad_samples_10_mbpp_pp.jsonl"
[[ "$FP" == *"pp"* ]] || { echo "Assertion failed: 'pp' not in FP"; exit 1; }
python3 human_eval/evaluate_functional_correctness.py --sample_file ${FP} 