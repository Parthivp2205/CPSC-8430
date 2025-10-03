#!/bin/bash
# Usage: ./hw2_seq2seq.sh <data_dir> <output_file>

DATA_DIR=$1
OUTPUT_FILE=$2

python3 model_seq2seq.py \
  --data_dir "$DATA_DIR" \
  --output "$OUTPUT_FILE" \
  --weights your_seq2seq_model/model_weights.h5 \
  --meta your_seq2seq_model/model_meta.json
