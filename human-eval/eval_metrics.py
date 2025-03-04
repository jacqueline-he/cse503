import jsonlines
import pdb
from itertools import groupby

def group_by_key(lst, key_func):
    """
    Groups a list into a list of lists where each sublist has the same key.
    
    Args:
        lst (list): The input list.
        key_func (function): A function that extracts the key for grouping.

    Returns:
        list: A list of lists, where each inner list has the same key.
    """
    lst.sort(key=key_func)  # Sort to ensure grouping works correctly
    return [list(group) for _, group in groupby(lst, key=key_func)]

def load_jsonlines(file):
    with jsonlines.open(file, 'r') as jsonl_f:
        lst = [obj for obj in jsonl_f]
    return lst

def save_file_jsonl(data, fp):
    with jsonlines.open(fp, mode='w') as writer:
        writer.write_all(data)

fp = "/gscratch/zlab/jyyh/cse503/human-eval/results/cad_samples_10_pp.jsonl_results.jsonl"
data = load_jsonlines(fp)
key_func = lambda x: x['task_id']  # Group by the first element (fruit name)

grouped_data = group_by_key(data, key_func)
print(len(grouped_data))
for d in grouped_data:
    pass

pdb.set_trace()
