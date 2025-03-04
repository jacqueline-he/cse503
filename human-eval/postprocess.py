import jsonlines
import pdb

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


fp = f"/gscratch/zlab/jyyh/cse503/human-eval/results/bigcode/starcoderbase-1b/regular_samples_5.jsonl"
mode = "regular" if "regular" in fp else "cad" if "cad" in fp else "rag"
data = load_jsonlines(fp)

for d in data:
    prompt, completion = split(d['completion'], mode)
    d['completion'] = completion
new_fp = fp.replace(".jsonl", "_pp.jsonl")
save_file_jsonl(data, new_fp)
print(f"Saved to {new_fp}")