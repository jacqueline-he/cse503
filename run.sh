mode="cad"
model="meta-llama/Llama-3.2-1B-Instruct"

for alpha in 0.7; do 
    FP="/gscratch/zlab/jyyh/cse503/human-eval/results/${model}/${mode}_samples_10_alpha_${alpha}.jsonl"
    python -m human-eval.generate_samples --mode "$mode" --model_name_or_path "$model" --save_path ${FP}
    echo ${FP}
    bash human-eval/score_samples.sh ${FP}
done 
exit 0


# for model in "google/gemma-2-2b-it" "jinaai/starcoder-1b-textbook"; do #  "bigcode/starcoderbase-1b" "google/gemma-2-2b-it" 
#         # Assign each process to a different GPU
#         python -m human-eval.generate_samples --mode "regular" --model_name_or_path "$model"
# done
