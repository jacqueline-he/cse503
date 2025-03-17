import jsonlines
import pdb
import numpy as np

def split(s, mode):
    if mode == 'cad':
        return s.split("Context: ", 1)
    else:
        return s.split("\n\n", 1)

def load_jsonlines(file):
    with jsonlines.open(file, 'r') as jsonl_f:
        lst = [obj for obj in jsonl_f]
    return lst

def save_file_jsonl(data, fp):
    with jsonlines.open(fp, mode='w') as writer:
        writer.write_all(data)


fp = f"/gscratch/zlab/jyyh/cse503/human-eval/results/meta-llama/Llama-3.2-1B-Instruct/cad_samples_10_10.jsonl"
mode = "regular" if "regular" in fp else "cad" if "cad" in fp else "rag"
data = load_jsonlines(fp)

for d in data:
    prompt, completion = split(d['completion'], mode)
    d['completion'] = completion


tts = [sample['throughput_tokens_per_sec'] for sample in data]
mos = [sample['memory_overhead_MB'] for sample in data]
print(f"Average Throughput: {np.mean(tts):.2f} tokens/sec")
print(f"Average Memory Overhead: {np.mean(mos):.2f} MB")

pdb.set_trace()
new_fp = fp.replace(".jsonl", "_pp.jsonl")
save_file_jsonl(data, new_fp)
print(f"Saved to {new_fp}")