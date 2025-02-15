
EMB_DIR=/gscratch/scrubbed/jyyh/code-doc-embs
DATA=test.jsonl
OUT_DIR=out

 python retrieval.py \
  --passages_embeddings $EMB_DIR \
  --data $DATA  \
  --output_dir $OUT_DIR \
  --n_docs 5 \
  --projection_size 1024 \
  --shard 0
