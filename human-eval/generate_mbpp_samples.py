import pdb
import os
from human_eval.data import write_jsonl, read_problems
from context_aware_decoding.simple_cad import standard_decoding, context_aware_sampling, context_aware_sampling_fast
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import torch
import re 
from tqdm import tqdm
import time 
import argparse 
import jsonlines 
from itertools import islice
import numpy as np
import sys 
from datasets import load_dataset
# NOTE: Don't change the \n\n!!! They are intentional and critical for postprocessing completions
FORMAT_STR="Your task is to accurately complete a Python programming problem, given a function description. Return only code for the completed function, with no additional explanation, print statements, or unit tests. {prompt}\n\n{func_name}"
CAD_FORMAT_STR="Your task is to accurately complete a Python programming problem, given a function description. Return only code for the completed function, with no additional explanation, print statements, or unit tests. {prompt}\n\n"
RAG_FORMAT_STR="Your task is to accurately complete a Python programming problem, given a function description. Return only code for the completed function, with no additional explanation, print statements, or unit tests. Here is some context that may be helpful: {rag_doc}\n{prompt}\n\n{func_name}"
COT_FORMAT_STR="Your task is to accurately complete a Python programming problem, given a function description. Before completing the function, think step by step about the solution. First, outline the approach based on the function signature and docstring. Break down the problem, identify edge cases, and describe the logic needed. Then, implement the function accordingly. Return only code for the completed function, with no additional explanation, print statements, or unit tests.\n{prompt}\n\n{func_name}"

global st_model # sentence xfmr model used for SC baseline

def split(s, mode):
    if mode == 'cad':
        # if "context" in s.lower():
        #     return s.split("Context: ", 1)
        # else:
        #     return s
        completion = s.split("\n\n", 1)
        return completion
    else:
        s = re.sub(r'```+', '', s).strip() # remove backticks
        completion = s.split("\n\n", 1)
        # completion = re.sub(r'```python', '', completion)  # Remove starting triple-backticks
        # completion = re.sub(r'```+', '', completion)  # Remove excessive closing backticks
        
        # # Trim leading/trailing whitespace
        # completion = completion.strip()
        return completion


def find_most_consistent_output(outputs):
    embeddings = np.array(st_model.encode(outputs))

    # Compute pairwise cosine similarities
    similarity_matrix = np.dot(embeddings, embeddings.T)

    # Find the output with the highest total similarity
    most_consistent_index = np.argmax(similarity_matrix.sum(axis=1))
    return outputs[most_consistent_index]

def load_jsonlines(file):
    with jsonlines.open(file, 'r') as jsonl_f:
        lst = [obj for obj in jsonl_f]
    return lst

def save_file_jsonl(data, fp):
    with jsonlines.open(fp, mode='w') as writer:
        writer.write_all(data)

def extract_function_info(code: str) -> tuple[str, str]:
    match = re.split(r'("""|\'\'\')(.*)', code, maxsplit=1, flags=re.DOTALL)

    if len(match) > 2:
        question = match[0].strip()
        context = match[1] + match[2]  # Concatenating the matched delimiter and the docstring
    else:
        question = code.strip()  # If no triple quotes found, return everything as question
        context = ""  # No context found

    return question, context

def get_function_name(code):
    match = re.search(r'^\s*def\s+[a-zA-Z_]\w*\s*\(.*?\):', code, re.MULTILINE)
    return match.group(0) if match else None  # Return the signature if found

def generate_one_completion(prompt, func_name, model, tokenizer, mode="regular", max_len=256, benchmark=False, rag_ctx=None):
    torch.cuda.synchronize()
    mem_before = torch.cuda.memory_allocated(model.device)
    start_time = time.time()

    if mode == "regular":
        formatted_prompt = FORMAT_STR.format(prompt=prompt, func_name=func_name)
        prompt_input_ids = tokenizer(formatted_prompt, return_tensors="pt").input_ids.to(model.device)
        completion_ids, output = standard_decoding(prompt_input_ids, model, tokenizer, max_length=max_len)
        # pdb.set_trace()
    elif mode == "cad":
        formatted_context = CAD_FORMAT_STR.format(prompt=prompt)
        question = func_name
        context_input = tokenizer(formatted_context, return_tensors="pt").input_ids.to(model.device)
        question_input = tokenizer(question, return_tensors="pt").input_ids.to(model.device)
        input_ids = torch.cat([context_input, question_input], dim=-1)
        completion_ids, output = context_aware_sampling_fast(input_ids, context_input, model, tokenizer, alpha=0.5, max_length=max_len, temperature=0.3)
        #print(f"CAD OUTPUT:\n\n{output}\n\n")
    elif mode == "rag":
        formatted_prompt = RAG_FORMAT_STR.format(prompt=prompt, rag_doc=rag_ctx, func_name=func_name)
        prompt_input_ids = tokenizer(formatted_prompt, return_tensors="pt").input_ids.to(model.device)
        completion_ids, output = standard_decoding(prompt_input_ids, model, tokenizer, max_length=max_len)
        #print(output)
    elif mode == 'cot':
        formatted_prompt = COT_FORMAT_STR.format(prompt=prompt, func_name=func_name)
        prompt_input_ids = tokenizer(formatted_prompt, return_tensors="pt").input_ids.to(model.device)
        completion_ids, output = standard_decoding(prompt_input_ids, model, tokenizer, max_length=max_len)
        #print(output)
    elif mode == 'sc':
        sc_num_samples = 5 
        formatted_prompt = FORMAT_STR.format(prompt=prompt, func_name=func_name)
        prompt_input_ids = tokenizer(formatted_prompt, return_tensors="pt").input_ids.to(model.device)
        sampled_outputs = []
        for _ in range(sc_num_samples):
            completion_ids, sc_output = standard_decoding(
                prompt_input_ids, model, tokenizer, max_length=max_len, temperature=0.8  # Sampling for diversity
            )
            sampled_outputs.append(sc_output)
        output = find_most_consistent_output(sampled_outputs)
    else:
        raise ValueError(f"Invalid mode: {mode}")
    ans = split(output.strip(), args.mode)[1]
    print(ans)
    torch.cuda.synchronize()
    end_time = time.time()
    # Measure VRAM after inference
    mem_after = torch.cuda.memory_allocated(model.device)
    total_time = end_time - start_time
    
    num_tokens = completion_ids.numel()  # Count tokens in output
    throughput = num_tokens / total_time if total_time > 0 else 0
    memory_overhead = (mem_after - mem_before) / (1024 ** 2)  # Convert bytes to MB
    return {
        "completion": ans,
        "throughput_tokens_per_sec": throughput,
        "memory_overhead_MB": memory_overhead,
        "total_time_sec": total_time,
        "num_tokens_generated": num_tokens
    }
    
###############################################################################
# from cse503 dir, run "python -m human-eval.generate_mbpp_samples"
# make sure to set mode accordingly


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default="rag", help='Mode for generating samples', choices=['cad', 'regular', 'rag', 'cot', 'sc'])
    parser.add_argument('--model_name_or_path', type=str, default="meta-llama/Llama-3.2-1B-Instruct", help='Model name or path')
    parser.add_argument('--benchmark', action='store_true', help='Benchmark efficiency')
    parser.add_argument('--rag_path', type=str, default="/gscratch/zlab/jyyh/cse503/retrieval/out/mbpp_0.jsonl")
    parser.add_argument('--max_len', type=int, default=256, help='Maximum length of generated completions')
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--n', type=int, default=10, help='Number of samples per task')
    parser.add_argument('--alpha', type=int, default=0.1, help='Alpha value for context-aware decoding')
    parser.add_argument('--save_path', type=str, default=None, help='Path to save generated samples')
    args = parser.parse_args()

    problems = load_dataset('google-research-datasets/mbpp')['validation'] # 90 samples
    if args.max_samples is not None:
        print(f"Sampling {args.max_samples} samples")
        problems = problems.select(range(args.max_samples)).to_list()
    # problems = read_problems()


    model_name = args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id  # Ensure correct padding behavior
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, trust_remote_code=True, device_map="auto")
    # quantization_config = BitsAndBytesConfig(load_in_8bit=True)  # Enable 8-bit inference
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_name, 
    #     quantization_config=quantization_config
    # )
    model.eval()

    if args.mode == 'sc':
        st_model = SentenceTransformer("all-mpnet-base-v2")

    if args.mode == "rag":
        args.max_len=2048 # longer window size
        rag_docs = load_jsonlines(args.rag_path)
        ctxs = {
            f"{i}": doc['ctxs'][0]['text'] if 'ctxs' in doc and doc['ctxs'] else None
            for i, doc in enumerate(rag_docs)
        }
    else:
        ctxs = None 

    num_samples_per_task = args.n
    # samples = [
    #     dict(task_id=task_id, completion=generate_one_completion(problems[task_id]["prompt"], model, tokenizer, args.mode, max_len=args.max_len, benchmark=args.benchmark, rag_ctx=ctxs[task_id] if args.mode == "rag" else None))
    #     for task_id in tqdm(problems, desc="Processing Tasks")
    #     for _ in range(num_samples_per_task) 
    # ]
    # for problem in problems:
    #     try:
    #         print(get_function_name(problem['code']))
    #     except:
    #         pdb.set_trace()
    #         print(problem['code'])

    samples = []
    for i, problem in tqdm(enumerate(problems), desc=f"Processing Tasks for {args.mode}"):
        task_id = problem['task_id']
        for _ in range(num_samples_per_task):
            func_name = get_function_name(problem['code'])
            result = generate_one_completion(
                problem['text'],
                func_name,
                model,
                tokenizer,
                args.mode,
                max_len=args.max_len,
                benchmark=args.benchmark,
                rag_ctx=ctxs[str(i)] if args.mode == "rag" else None
            )
            
            # Store the result as a dictionary with task_id
            sample_entry = {"task_id": task_id, **result} # starts at 511
            samples.append(sample_entry)

    # for task_id in tqdm(problems, desc=f"Processing Tasks for {args.mode}"):
    #     for _ in range(num_samples_per_task):
    #         result = generate_one_completion(
    #             problems[task_id]["prompt"],
    #             model,
    #             tokenizer,
    #             args.mode,
    #             max_len=args.max_len,
    #             rag_ctx=ctxs[task_id] if args.mode == "rag" else None
    #         )
            
    #         # Store the result as a dictionary with task_id
    #         sample_entry = {"task_id": task_id, **result}
    #         samples.append(sample_entry)
            
    tts = [sample['throughput_tokens_per_sec'] for sample in samples]
    mos = [sample['memory_overhead_MB'] for sample in samples]

    print(f"Average Throughput: {np.mean(tts):.2f} tokens/sec")
    print(f"Average Memory Overhead: {np.mean(mos):.2f} MB")

    out_dir=f"human-eval/mbpp_results/{model_name}"
    os.makedirs(out_dir, exist_ok=True) 
    if args.save_path is None:
        if args.max_samples is not None:
            fp = f"{out_dir}/{args.mode}_samples_{num_samples_per_task}_{args.max_samples}.jsonl"
        else:
            fp = f"{out_dir}/{args.mode}_samples_{num_samples_per_task}.jsonl"
    else:
        fp = args.save_path
        
    write_jsonl(fp, samples)
    print(f"Saved samples to {fp}")