import jsonlines
import pdb 

def load_jsonlines(file):
    with jsonlines.open(file, 'r') as jsonl_f:
        lst = [obj for obj in jsonl_f]
    return lst

def save_file_jsonl(data, fp):
    with jsonlines.open(fp, mode='w') as writer:
        writer.write_all(data)

fp = "/gscratch/zlab/jyyh/cse503/retrieval/out/humaneval_0.jsonl"
data = load_jsonlines(fp)

for d in data:
    pdb.set_trace()