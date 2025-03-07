#FP="/gscratch/zlab/jyyh/cse503/human-eval/results/meta-llama/Llama-3.2-1B-Instruct/cad_samples_5_pp.jsonl"
FP=" /gscratch/zlab/jyyh/cse503/human-eval/results/bigcode/starcoderbase-1b/cad_samples_5_pp.jsonl"
[[ "$FP" == *"pp"* ]] || { echo "Assertion failed: 'pp' not in FP"; exit 1; }
evaluate_functional_correctness --sample_file ${FP} 