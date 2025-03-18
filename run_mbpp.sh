mode="cad"
model="meta-llama/Llama-3.2-1B-Instruct"
max_samples=5


FP="/gscratch/zlab/jyyh/cse503/human-eval/mbpp_results/${model}/${mode}_samples_10_${max_samples}.jsonl"
python -m human-eval.generate_mbpp_samples --mode "$mode" --model_name_or_path "$model" --max_samples ${max_samples} --save_path ${FP}
echo ${FP}
bash human-eval/score_mbpp_samples.sh ${FP}

exit 0