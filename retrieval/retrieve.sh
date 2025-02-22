
EMB_DIR=/gscratch/scrubbed/jyyh/code-doc-embs
DATASET="mbpp" # or "mbpp"
OUT_DIR=out

 python retrieval.py \
  --passages_embeddings $EMB_DIR \
  --dataset $DATASET \
  --output_dir $OUT_DIR \
  --n_docs 5 \
  --projection_size 1024 \
  --shard 0
