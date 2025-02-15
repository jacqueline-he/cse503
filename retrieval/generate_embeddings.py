# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
import argparse
import pickle as pkl

import numpy as np
import torch
from tqdm import tqdm
import time
import jsonlines

import src.slurm

import src.utils

import src.data
import src.normalize_text
from FlagEmbedding import FlagModel
import os 
from datasets import load_dataset



# Disable tokenizers parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Limit the number of threads
os.environ["OMP_NUM_THREADS"] = "1"

def load_jsonlines(file):
    with jsonlines.open(file, 'r') as jsonl_f:
        return [obj for obj in jsonl_f]

def split_data_into_chunks(text, n):
    text = text.split()
    chunks = [' '.join(text[i:i+n]) for i in range(0,len(text),n)]
    return chunks

def chunkify_code_samples(samples):
    all_chunks = []
    for sample in samples:
        doc_content, doc_id = sample['doc_content'], sample['doc_id']
        chunks = split_data_into_chunks(doc_content, 500) # 500 words separated by whitespace; BGE's max tokens is 512
        for chunk in chunks:
            all_chunks.append(f"Function: {doc_id}\nSnippet: {chunk}")
    for i, chunk in enumerate(all_chunks):
        all_chunks[i] = {'id': i, 'text': chunk}
    return all_chunks

def embed_passages(args, passages, model):
    total = 0
    allids, allembeddings = [], []
    batch_ids, batch_text = [], []

    with torch.no_grad():
        for k, p in tqdm(enumerate(passages)):
            batch_ids.append(p['id'])
            if args.no_title or not 'title' in p:
                text = p['text']
            else:
                text = p['title'] + ' ' + p['text']
            if args.lowercase:
                text = text.lower()
            if args.normalize_text:
                text = src.normalize_text.normalize(text)
            
            batch_text.append(text)
            

            if len(batch_text) == args.per_gpu_batch_size or k == len(passages) - 1:
                try:
                    embeddings = model.encode(batch_text, batch_size=16)
                except Exception as e:
                    print(f"Error at batch {k}: {e}")
                    raise
                
                total += len(batch_ids)
                allids.extend(batch_ids)
                allembeddings.append(embeddings)

                batch_text = []
                batch_ids = []
                if k % 100000 == 0 and k > 0:
                    print('Encoded passages %d', total)

    #allembeddings = torch.cat(allembeddings, dim=0).numpy()
    allembeddings  = np.concatenate(allembeddings, axis=0)
    return allids, allembeddings


def main(args):
    model = FlagModel('/mmfs1/gscratch/zlab/jyyh/sft/retrieval/bge-large-en-v1.5',
                  query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
                  use_fp16=True)
    
    start_time = time.time()

    ds = load_dataset("code-rag-bench/library-documentation")
    samples = ds['train'] # 34K samples of the form (doc_content, doc_id)
    chunked_samples = chunkify_code_samples(samples)
    print(f"Total number of chunks: {len(chunked_samples)}")
    passages = chunked_samples
            
    
    # Split passages into shards
    shard_size = len(passages) // args.num_shards
    start_idx = args.shard_id * shard_size
    end_idx = start_idx + shard_size
 
    if args.shard_id == args.num_shards-1:
        end_idx = len(passages)
  
    
    passages = passages[start_idx:end_idx]
    print(f"Sample passage: {passages[0:5]}")
    print(f'Embedding generation for {len(passages)} passages from idx {start_idx} to {end_idx}.')

    # this is all the passages from the shard
    allids, allembeddings = embed_passages(args, passages, model)

    # Save embeddings to file 
    output_dir = args.output_dir
    save_file = os.path.join(output_dir, args.prefix + f'_{args.shard_id:02d}')
    os.makedirs(output_dir, exist_ok=True)
    print(f'Saving {len(allids)} passage embeddings to {save_file}.')
    with open(save_file, mode='wb') as f:
        pkl.dump((allids, allembeddings), f)
    end_time = time.time()
    print(f'Total passages processed {len(allids)} in {end_time - start_time} seconds. Written to {save_file}.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # paths
    parser.add_argument("--passages_embeddings", type=str, default=None, help="Glob path to encoded passages")
    parser.add_argument('--output_dir', type=str, required=True, help='dir path to save embeddings')
    parser.add_argument('--prefix', type=str, default='passages', help='prefix path to save embeddings')
    parser.add_argument('--shard_id', type=int, default=0, help="Id of the current shard")
    parser.add_argument('--num_shards', type=int, default=1, help="Total number of shards")
    parser.add_argument('--per_gpu_batch_size', type=int, default=256, help="Batch size for the passage encoder forward pass")
    parser.add_argument('--passage_maxlength', type=int, default=512, help="Maximum number of tokens in a passage")
    parser.add_argument('--no_fp16', action='store_true', help="inference in fp32")
    parser.add_argument('--no_title', action='store_true', help="title not added to the passage body")
    parser.add_argument('--lowercase', action='store_true', help="lowercase text before encoding")
    parser.add_argument('--normalize_text', action='store_true', help="lowercase text before encoding")
    parser.add_argument("--emb_suffix", type=str, default=None, help="Glob path to encoded passages")

    args = parser.parse_args()
    print('-' * 50, 'Retrieval args:\n', args)

    print(f'Initializing distributed mode...')
    src.slurm.init_distributed_mode(args)

    main(args)

