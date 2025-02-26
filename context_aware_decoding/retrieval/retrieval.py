# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import argparse
import csv
import json
import pickle
import time
import glob
import jsonlines

import numpy as np
import torch
from FlagEmbedding import FlagModel
import src.index

import src.utils
import src.slurm
import src.data
from src.evaluation import calculate_matches
import src.normalize_text
from generate_embeddings import chunkify_code_samples
from datasets import load_dataset 


os.environ["TOKENIZERS_PARALLELISM"] = "true"

def load_jsonlines(file):
    with jsonlines.open(file, 'r') as jsonl_f:
        lst = [obj for obj in jsonl_f]
    return lst

def embed_queries(args, queries, model):
    embeddings, batch_question = [], []
    with torch.no_grad():

        for k, q in enumerate(queries):
            if args.lowercase:
                q = q.lower()
            if args.normalize_text:
                q = src.normalize_text.normalize(q)
            batch_question.append(q)

            if len(batch_question) == args.per_gpu_batch_size or k == len(queries) - 1:
                try:
                    output = model.encode(batch_question, batch_size=16)
                except Exception as e:
                    print(f"Error at batch {k}: {e}")
                    raise
                if isinstance(output, np.ndarray):
                    output = torch.from_numpy(output)
                embeddings.append(output)

                batch_question = []
   
    embeddings = torch.cat(embeddings, dim=0)
    print(f"Questions embeddings shape: {embeddings.size()}")

    return embeddings.numpy()


def index_encoded_data(index, embedding_file, indexing_batch_size):
    allids = []
    allembeddings = np.array([])
    embedding_files = [embedding_file] # hacky
    print(embedding_files)
    for i, file_path in enumerate(embedding_files):
        if "faiss" not in file_path:
            print(f"Loading file {file_path}")
            with open(file_path, "rb") as fin:
                ids, embeddings = pickle.load(fin)

            allembeddings = np.vstack((allembeddings, embeddings)) if allembeddings.size else embeddings
            allids.extend(ids)
            
            while allembeddings.shape[0] > indexing_batch_size:
                allembeddings, allids = add_embeddings(index, allembeddings, allids, indexing_batch_size)

    while allembeddings.shape[0] > 0:
        allembeddings, allids = add_embeddings(index, allembeddings, allids, indexing_batch_size)
    print("Data indexing completed.")


def add_embeddings(index, embeddings, ids, indexing_batch_size):
    end_idx = min(indexing_batch_size, embeddings.shape[0])
    ids_toadd = ids[:end_idx]
    embeddings_toadd = embeddings[:end_idx]
    ids = ids[end_idx:]
    embeddings = embeddings[end_idx:]
    index.index_data(ids_toadd, embeddings_toadd)
    return embeddings, ids


def validate(data, workers_num):
    match_stats = calculate_matches(data, workers_num)
    top_k_hits = match_stats.top_k_hits

    print("Validation results: top k documents hits %s", top_k_hits)
    top_k_hits = [v / len(data) for v in top_k_hits]
    message = ""
    for k in [5, 10, 20, 100]:
        if k <= len(top_k_hits):
            message += f"R@{k}: {top_k_hits[k-1]} "
    print(message)
    return match_stats.questions_doc_hits

def add_passages(data, passages, top_passages_and_scores):
    print(len(passages))
    assert len(data) == len(top_passages_and_scores), f"mismatch between {len(data)} and {len(top_passages_and_scores)}"
    for i, d in enumerate(data):
        results_and_scores = top_passages_and_scores[i]
        docs = []
        for doc_id in results_and_scores[0]:
            docs.append(passages[doc_id])
        scores = [str(score) for score in results_and_scores[1]]
        ctxs_num = len(docs)
        d["ctxs"] = [
            {
                "id": results_and_scores[0][c],
                "text": docs[c],
                "score": scores[c],
            }
            for c in range(ctxs_num)
        ]


def add_hasanswer(data, hasanswer):
    # add hasanswer to data
    for i, ex in enumerate(data):
        for k, d in enumerate(ex["ctxs"]):
            d["hasanswer"] = hasanswer[i][k]


def load_data(data_path):
    if data_path.endswith(".json"):
        with open(data_path, "r") as fin:
            data = json.load(fin)
    elif data_path.endswith(".jsonl"):
        data = []
        with open(data_path, "r") as fin:
            for _, example in enumerate(fin):
                example = json.loads(example)
                data.append(example)
    return data


def main(args):

    model = FlagModel('/mmfs1/gscratch/zlab/jyyh/sft/retrieval/bge-large-en-v1.5',
                  query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
                  use_fp16=True)
    print(f'successfully loaded model')

    index = src.index.Indexer(args.projection_size, args.n_subquantizers, args.n_bits)
    print(args.passages_embeddings)
    assert os.path.exists(args.passages_embeddings), "Check that passages embeddings dir. is correct"
    # index all passages
    if "*" not in args.passages_embeddings:
        args.passages_embeddings += "/*"
    input_paths = glob.glob(args.passages_embeddings)
    input_paths = sorted(input_paths)
    embeddings_dir = os.path.dirname(input_paths[0])
    index_path = os.path.join(embeddings_dir, "index.faiss")

    if args.save_or_load_index and os.path.exists(index_path):
        index.deserialize_from(embeddings_dir)
    else:
        input_path = input_paths[args.shard]
        print(f"Indexing passages from files {input_path}")
        start_time_indexing = time.time()
        index_encoded_data(index, input_path, args.indexing_batch_size)
        print(f"Indexing time: {time.time()-start_time_indexing:.1f} s.")
        if args.save_or_load_index:
            index.serialize(embeddings_dir)
    
    
    ds = load_dataset("code-rag-bench/library-documentation")
    samples = ds['train'] # 34K samples of the form (doc_content, doc_id)
    passages = chunkify_code_samples(samples)
    print(f"Total number of chunks: {len(passages)}")
    assert len(passages) > 0

    passage_id_map = {}
    for i, d in enumerate(passages):
        passage_id_map[str(i)] = d
    assert len(passages) == len(passage_id_map)


    assert os.path.exists(args.data)

    data = load_jsonlines(args.data)
    filename = os.path.basename(args.data).replace(".jsonl", f"_{args.shard}.jsonl")
    output_path = os.path.join(args.output_dir, filename)
    if not os.path.exists(output_path):
        try:
            queries = [ex["query"] for ex in data]
        except:
            queries = [ex["question"] for ex in data]
        questions_embedding = embed_queries(args, queries, model)

        # get top k results
        start_time_retrieval = time.time()
        top_ids_and_scores = index.search_knn(questions_embedding, args.n_docs)
        print(f"Search time: {time.time()-start_time_retrieval:.1f} s.")
        add_passages(data, passage_id_map, top_ids_and_scores)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if not output_path.endswith(".jsonl"):
            output_path = output_path.replace(".json", ".jsonl")
        save_file_jsonl(data, output_path)

        print(f"Saved results to {output_path}")
    else:
        print(f"File {output_path} already exists.")

def save_file_jsonl(data, fp):
    with jsonlines.open(fp, mode='w') as writer:
        writer.write_all(data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        required=True,
        type=str,
        default=None,
        help=".jsonl file containing question and answers, similar format to reader data",
    )

    parser.add_argument("--passages_embeddings", type=str, default=None, help="Glob path to encoded passages")
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Results are written to outputdir with data suffix"
    )
    parser.add_argument("--n_docs", type=int, default=100, help="Number of documents to retrieve per questions")
    parser.add_argument(
        "--validation_workers", type=int, default=32, help="Number of parallel processes to validate results"
    )
    parser.add_argument("--per_gpu_batch_size", type=int, default=64, help="Batch size for question encoding")
    parser.add_argument(
        "--save_or_load_index", action="store_true", help="If enabled, save index and load index if it exists"
    )
    parser.add_argument("--no_fp16", action="store_true", help="inference in fp32")
    parser.add_argument("--question_maxlength", type=int, default=512, help="Maximum number of tokens in a question")
    parser.add_argument(
        "--indexing_batch_size", type=int, default=1000000, help="Batch size of the number of passages indexed"
    )
    parser.add_argument("--projection_size", type=int, default=1024)
    parser.add_argument(
        "--n_subquantizers",
        type=int,
        default=0,
        help="Number of subquantizer used for vector quantization, if 0 flat index is used",
    )
    parser.add_argument("--n_bits", type=int, default=8, help="Number of bits per subquantizer")
    parser.add_argument("--lang", nargs="+")
    parser.add_argument("--lowercase", action="store_true", help="lowercase text before encoding")
    parser.add_argument("--normalize_text", action="store_true", help="normalize text")
    parser.add_argument("--shard", type=int, help="psg. embed. shard to consider")


    args = parser.parse_args()
    src.slurm.init_distributed_mode(args)
    main(args)
