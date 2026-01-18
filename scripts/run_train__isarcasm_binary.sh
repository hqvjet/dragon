#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=true
export OMP_NUM_THREADS=4
dt=`date '+%Y%m%d_%H%M%S'`


dataset="isarcasm"
shift
encoder='roberta-large'
args=$@


# Binary Classification Fine-tuning (SIMPLER & FASTER)
elr="2e-5"           # Encoder LR (pre-trained RoBERTa)
dlr="1e-3"           # Decoder (GNN + Classifier head) LR
bs=32                # Batch size
mbs=2                # Mini-batch size (gradient accumulation)
unfreeze_epoch=0     # Unfreeze all from start
k=5                  # 5 GNN layers
residual_ie=2
gnndim=200


encoder_layer=-1
max_node_num=200
seed=42              # Fixed seed for reproducibility
lr_schedule=warmup_linear
warmup_steps=150

n_epochs=10          # Binary classification converges faster
max_epochs_before_stop=5
ie_dim=400


max_seq_len=128      # Sarcasm text usually short
ent_emb=data/cpnet/tzw.ent.npy
kg=cpnet
kg_vocab_path=data/cpnet/concept.txt
inhouse=false


# Info exchange for capturing incongruity (critical for sarcasm!)
info_exchange=true
ie_layer_num=1
resume_checkpoint=None
resume_id=None
sep_ie_layers=false
random_ent_emb=false

# Load pre-trained DRAGON
load_model_path=models/general_model.pt

echo "***** DRAGON Binary Classification Fine-tuning *****"
echo "Mode: BINARY CLASSIFICATION (Simple & Fast)"
echo "Pre-trained model: $load_model_path"
echo "dataset: $dataset"
echo "encoder: $encoder"
echo "batch_size: $bs (mini_batch: $mbs)"
echo "learning_rate: encoder=$elr decoder=$dlr"
echo "gnn: dim=$gnndim layers=$k"
echo "ie_dim: $ie_dim"
echo "seed: $seed"
echo "epochs: $n_epochs"
echo "****************************************************"

save_dir_pref='runs'
mkdir -p $save_dir_pref
mkdir -p logs

run_name=dragon_binary__${dataset}_elr${elr}_dlr${dlr}_b${bs}_e${n_epochs}_sd${seed}__${dt}
log=logs/train__${run_name}.log.txt

###### Binary Classification Fine-tuning ######
echo "Starting binary classification training at $(date)"

python3 -u dragon_binary.py \
    --dataset $dataset \
    --encoder $encoder -k $k --gnn_dim $gnndim -elr $elr -dlr $dlr -bs $bs --seed $seed -mbs ${mbs} --unfreeze_epoch ${unfreeze_epoch} --encoder_layer=${encoder_layer} -sl ${max_seq_len} --max_node_num ${max_node_num} \
    --n_epochs $n_epochs --max_epochs_before_stop ${max_epochs_before_stop} --use_wandb true \
    --save_dir ${save_dir_pref}/${dataset}/${run_name} --save_model 1 \
    --run_name ${run_name} \
    --load_model_path $load_model_path \
    --residual_ie $residual_ie \
    --ie_dim ${ie_dim} --info_exchange ${info_exchange} --ie_layer_num ${ie_layer_num} --resume_checkpoint ${resume_checkpoint} --resume_id ${resume_id} --sep_ie_layers ${sep_ie_layers} --random_ent_emb ${random_ent_emb} --ent_emb_paths ${ent_emb//,/ } --lr_schedule ${lr_schedule} --warmup_steps $warmup_steps -ih ${inhouse} --kg $kg --kg_vocab_path $kg_vocab_path \
    --data_dir data \
> ${log}

echo "Training completed at $(date)"
echo "Log: $log"
echo "Model: ${save_dir_pref}/${dataset}/${run_name}/model.pt"

# Show quick stats
echo ""
echo "=== Quick Stats ==="
tail -20 ${log} | grep -E "(Epoch|accuracy|loss)" || echo "Check log file for details"
