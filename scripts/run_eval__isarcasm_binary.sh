#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

# Path to trained model
model_path=$1

if [ -z "$model_path" ]; then
    echo "Usage: ./run_eval__isarcasm_binary.sh <path_to_model.pt>"
    echo "Example: ./run_eval__isarcasm_binary.sh runs/isarcasm/dragon_binary__*/model.pt"
    exit 1
fi

if [ ! -f "$model_path" ]; then
    echo "Error: Model file not found: $model_path"
    exit 1
fi

dataset="isarcasm"
encoder='roberta-large'
k=5
gnndim=200
max_seq_len=128
max_node_num=200
encoder_layer=-1

ent_emb=data/cpnet/tzw.ent.npy
kg=cpnet
kg_vocab_path=data/cpnet/concept.txt

# Binary classifier mode
use_binary_classifier=true

echo "***** DRAGON Binary Classification Evaluation *****"
echo "Model: $model_path"
echo "Dataset: $dataset"
echo "Binary classifier: $use_binary_classifier"
echo "***************************************************"

python3 -u dragon.py \
    --dataset $dataset \
    --encoder $encoder \
    -k $k \
    --gnn_dim $gnndim \
    --encoder_layer=${encoder_layer} \
    -sl ${max_seq_len} \
    --max_node_num ${max_node_num} \
    --load_model_path $model_path \
    --use_binary_classifier ${use_binary_classifier} \
    --mode eval \
    --ent_emb_paths ${ent_emb} \
    --kg $kg \
    --kg_vocab_path $kg_vocab_path \
    --data_dir data

echo ""
echo "Evaluation completed!"
