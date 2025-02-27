import json 
from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("jacquelinehe/mbpp_processed")

rows = ds['train'].iter(batch_size=1)

"""
{'task_id': ['mbpp_11'], 'code': ['def remove_Occ(s,ch): \r\n    for i in range(len(s)): \r\n        if (s[i] == ch): \r\n            s = s[0 : i] + s[i + 1:] \r\n            break\r\n    for i in range(len(s) - 1,-1,-1):  \r\n        if (s[i] == ch): \r\n            s = s[0 : i] + s[i + 1:] \r\n            break\r\n    return s '], 'test_list': [['assert remove_Occ("hello","l") == "heo"', 'assert remove_Occ("abcda","a") == "bcd"', 'assert remove_Occ("PHP","P") == "H"']], 'context': ['Write a python function to remove first and last occurrence of a given character from the string.'], 'function_name': ['def remove_Occ(s,ch):'], 'gold_generation': ['for i in range(len(s)): \r\n        if (s[i] == ch): \r\n            s = s[0 : i] + s[i + 1:] \r\n            break\r\n    for i in range(len(s) - 1,-1,-1):  \r\n        if (s[i] == ch): \r\n            s = s[0 : i] + s[i + 1:] \r\n            break\r\n    return s']}
"""

"""
{
    "input_index": 0, // instances that decode together should have the same input_index
    "assigned_model": "huggyllama/llama-7b", // same model for all instances in context-aware decoding, but can use different models here, e.g., DExperts, contrastive decoding, proxy tuning, etc.
    "assigned_process": 0, // which GPU should take this instance
    "context_string": "The fourth season of Chicago Fire , an American drama television series with executive producer Dick Wolf , and producers Derek Haas , Michael Brandt , and Matt Olmstead , was ordered on February 5 , 2015 , by NBC , and premiered on October 13 , 2015 and concluded on May 17 , 2016 . The season contained 1078 episodes . How many episodes are in chicago fire season 4 ?", // the context-aware input
    "assigned_weight": 2, // weight for current instance/process (1+alpha, weights should add up to 1 by default, but can also incorporate sampling temperature if needed)
    "filter_p": 1.0, // optional filtering for low-probablity tokens, disabled by default
}
{
    "input_index": 0, // instances that decode together should have the same input_index
    "assigned_model": "huggyllama/llama-7b", // same model for all instances in context-aware decoding, but can use different models here, e.g., DExperts, contrastive decoding, proxy tuning, etc.
    "assigned_process": 1, // which GPU should take this instance
    "context_string": "How many episodes are in chicago fire season 4 ?", // the context-unaware input
    "assigned_weight": -1, // weight for current instance/process (-alpha, weights should add up to 1 by default, but can also incorporate sampling temperature if needed)
}
...
"""
assigned_model = "meta-llama/Llama-3.2-1B-Instruct"
assigned_weight_one_plus_alpha = 2
assigned_weight_minus_alpha = -1 
filter_p = 1.0 

json_l = []

for i, row in enumerate(rows):
    # print(row)
    
    code = row['code'][0]
    context = row["context"][0]
    function_name = row["function_name"][0]
    gold = row["gold_generation"][0]

    without_context = {
        "input_index": i,
        "assigned_model": assigned_model,
        "assigned_process": 0,
        "context_string": "Output only code for the following question: %s" % context,
        "assigned_weight": assigned_weight_one_plus_alpha,
        "filter_p": 1.0
    }

    with_context = {
        "input_index": i,
        "assigned_model": assigned_model,
        "assigned_process": 0,
        "context_string": "Output only code for the following question: %s Here is is the method signature=%s" % (context, function_name),
        "assigned_weight": assigned_weight_one_plus_alpha,
        "filter_p": 1.0
    }

    json_l.append(without_context)
    json_l.append(with_context)
    # print(without_context)
    # print(with_context)
    # input()

    if i > 10:
        break

with open('mbpp_cad.jsonl', 'w') as outfile:
    for entry in json_l:
        json.dump(entry, outfile)
        outfile.write('\n')

