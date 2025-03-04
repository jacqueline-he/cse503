import pdb
import os
from human_eval.data import write_jsonl, read_problems
from context_aware_decoding.simple_cad import standard_decoding, context_aware_sampling, context_aware_sampling_fast
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList, LogitsProcessor
from torch.nn import functional as F
import torch
import re 
from tqdm import tqdm
import time 
import argparse 
import jsonlines 
from itertools import islice
import numpy as np

device = "cuda" # "cpu"
FORMAT_STR="Your task is to complete a Python programming problem, given its function signature and docstring. Return only code for the completed function, with no additional explanation, print statements, or unit tests.\n\n {prompt}"
CAD_FORMAT_STR="Your task is to complete a Python programming problem, given its function signature and docstring. Return only code for the completed function, with no additional explanation, print statements, or unit tests.\n\n Context: {context}\n"
RAG_FORMAT_STR="Your task is to complete a Python programming problem, given its function signature and docstring. Return only code for the completed function, with no additional explanation, print statements, or unit tests. Here is some context that may be helpful: {rag_doc}\n\n {prompt}"

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

def generate_one_completion(prompt, model, tokenizer, mode="regular", max_len=256, benchmark=False, rag_ctx=None):
    if benchmark:
        torch.cuda.synchronize()
        mem_before = torch.cuda.memory_allocated(device)
        start_time = time.time()

    if mode == "regular":
        formatted_prompt = FORMAT_STR.format(prompt=prompt)
        prompt_input_ids = tokenizer(formatted_prompt, return_tensors="pt").input_ids.to(device)
        completion_ids, output = standard_decoding(prompt_input_ids, model, tokenizer, max_length=max_len)
        print(output)
    elif mode == "cad":
        question, context = extract_function_info(prompt)
        formatted_context = CAD_FORMAT_STR.format(context=context)
        context_input = tokenizer(formatted_context, return_tensors="pt").input_ids.to(device)
        question_input = tokenizer(question, return_tensors="pt").input_ids.to(device)
        input_ids = torch.cat([context_input, question_input], dim=-1)
        
        completion_ids, output = context_aware_sampling_fast(input_ids, context_input, model, tokenizer, alpha=0.5, max_length=max_len, temperature=0.8)
        print(f"CAD OUTPUT:\n\n{output}\n\n")
    elif mode == "rag":
        formatted_prompt = RAG_FORMAT_STR.format(prompt=prompt, rag_doc=rag_ctx)
        prompt_input_ids = tokenizer(formatted_prompt, return_tensors="pt").input_ids.to(device)
        completion_ids, output = standard_decoding(prompt_input_ids, model, tokenizer, max_length=max_len)
        print(output)
    else:
        raise ValueError(f"Invalid mode: {mode}")
    ans = output

    if benchmark:
        torch.cuda.synchronize()
        end_time = time.time()
        # Measure VRAM after inference
        mem_after = torch.cuda.memory_allocated(device)
        total_time = end_time - start_time
        
        num_tokens = completion_ids.numel()  # Count tokens in output
        print(num_tokens)
        throughput = num_tokens / total_time if total_time > 0 else 0
        memory_overhead = (mem_after - mem_before) / (1024 ** 2)  # Convert bytes to MB
        print(throughput)
        print(memory_overhead)
        return {
            "completion": ans,
            "throughput_tokens_per_sec": throughput,
            "memory_overhead_MB": memory_overhead,
            "total_time_sec": total_time,
            "num_tokens_generated": num_tokens
        }
    else:
        return ans
###############################################################################
# from cse503 dir, run "python -m human-eval.generate_samples"
# make sure to set mode accordingly


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default="rag", help='Mode for generating samples', choices=['cad', 'regular', 'rag'])
    parser.add_argument('--model_name_or_path', type=str, default="meta-llama/Llama-3.2-1B-Instruct", help='Model name or path')
    parser.add_argument('--benchmark', action='store_true', help='Benchmark efficiency')
    parser.add_argument('--rag_path', type=str, default="/gscratch/zlab/jyyh/cse503/retrieval/out/humaneval_0.jsonl")
    parser.add_argument('--max_len', type=int, default=512, help='Maximum length of generated completions')
    parser.add_argument('--n', type=int, default=10, help='Number of samples per task')
    args = parser.parse_args()
    problems = read_problems()


    model_name = args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id  # Ensure correct padding behavior
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    # quantization_config = BitsAndBytesConfig(load_in_8bit=True)  # Enable 8-bit inference
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_name, 
    #     quantization_config=quantization_config
    # )
    model.eval()
    model.to(device)

    
    if args.benchmark:
        problems = dict(islice(problems.items(), 10))
        args.n = 1 # Only generate one sample per task for benchmarking
    if args.mode == "rag":
        args.max_len=2048 # longer window size
        rag_docs = load_jsonlines(args.rag_path)
        ctxs = {
            f"HumanEval/{i}": doc['ctxs'][0]['text'] if 'ctxs' in doc and doc['ctxs'] else None
            for i, doc in enumerate(rag_docs)
        }
    else:
        ctxs = None 
    num_samples_per_task = args.n
    samples = [
        dict(task_id=task_id, completion=generate_one_completion(problems[task_id]["prompt"], model, tokenizer, args.mode, max_len=args.max_len, benchmark=args.benchmark, rag_ctx=ctxs[task_id] if args.mode == "rag" else None))
        for task_id in tqdm(problems, desc="Processing Tasks")
        for _ in range(num_samples_per_task) 
    ]
    tts = [sample['completion']['throughput_tokens_per_sec'] for sample in samples]
    mos = [sample['completion']['memory_overhead_MB'] for sample in samples]
    print(np.mean(tts))
    print(np.mean(mos))
    pdb.set_trace()
    out_dir=f"human-eval/results/{model_name}"
    os.makedirs(out_dir, exist_ok=True) 
    write_jsonl(f"{out_dir}/{args.mode}_samples_{num_samples_per_task}.jsonl", samples)