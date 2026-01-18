#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=true
export OMP_NUM_THREADS=4
dt=`date '+%Y%m%d_%H%M%S'`


dataset="isarcasm"
shift
encoder='roberta-large'
args=$@


# Fine-tuning hyperparameters (optimized for sarcasm detection)
elr="2e-5"           # Slightly higher LR for encoder fine-tuning
dlr="1e-3"           # Decoder (GNN) learning rate
bs=32                # Smaller batch size for stability
mbs=2                # Mini-batch size
unfreeze_epoch=0     # Unfreeze from start (fine-tuning, not feature extraction)
k=5                  # num of gnn layers
residual_ie=2
gnndim=200


encoder_layer=-1
max_node_num=200
seed=42              # Fixed seed for reproducibility
lr_schedule=warmup_linear
warmup_steps=150     # More warmup steps for stable fine-tuning

n_epochs=10          # Fewer epochs (fine-tuning converges faster)
max_epochs_before_stop=5  # Early stopping
ie_dim=400


max_seq_len=128      # Sarcasm text is usually shorter
ent_emb=data/cpnet/tzw.ent.npy
kg=cpnet
kg_vocab_path=data/cpnet/concept.txt
inhouse=false


# CRITICAL: Keep info_exchange for capturing incongruity
info_exchange=true
ie_layer_num=1
resume_checkpoint=None
resume_id=None
sep_ie_layers=false
random_ent_emb=false  # Use pre-trained embeddings!

fp16=true
upcast=true

# MUST load pre-trained model for fine-tuning
load_model_path=models/general_model.pt

echo "***** DRAGON Fine-tuning for Sarcasm Detection *****"
echo "Mode: FINE-TUNING (NOT training from scratch)"
echo "Pre-trained model: $load_model_path"
echo "dataset: $dataset"
echo "enc_name: $encoder"
echo "batch_size: $bs mini_batch_size: $mbs"
echo "learning_rate: elr $elr dlr $dlr (fine-tuning rates)"
echo "gnn: dim $gnndim layer $k"
echo "ie_dim: ${ie_dim}, info_exchange: ${ie_exchange}"
echo "seed: $seed (deterministic mode enabled)"
echo "unfreeze_epoch: $unfreeze_epoch (immediate fine-tuning)"
echo "******************************************************"

save_dir_pref='runs'
mkdir -p $save_dir_pref
mkdir -p logs

run_name=dragon_finetune__${dataset}_elr${elr}_dlr${dlr}_W${warmup_steps}_b${bs}_e${n_epochs}_sd${seed}__${dt}
log=logs/train__${run_name}.log.txt

###### Fine-tuning (NOT training from scratch) ######
echo "Starting fine-tuning at $(date)"
echo "Loading pre-trained weights from: $load_model_path"

python3 -u dragon.py \
    --dataset $dataset \
    --encoder $encoder -k $k --gnn_dim $gnndim -elr $elr -dlr $dlr -bs $bs --seed $seed -mbs ${mbs} --unfreeze_epoch ${unfreeze_epoch} --encoder_layer=${encoder_layer} -sl ${max_seq_len} --max_node_num ${max_node_num} \
    --n_epochs $n_epochs --max_epochs_before_stop ${max_epochs_before_stop} --fp16 $fp16 --upcast $upcast --use_wandb true \
    --save_dir ${save_dir_pref}/${dataset}/${run_name} --save_model 1 \
    --run_name ${run_name} \
    --load_model_path $load_model_path \
    --residual_ie $residual_ie \
    --ie_dim ${ie_dim} --info_exchange ${info_exchange} --ie_layer_num ${ie_layer_num} --resume_checkpoint ${resume_checkpoint} --resume_id ${resume_id} --sep_ie_layers ${sep_ie_layers} --random_ent_emb ${random_ent_emb} --ent_emb_paths ${ent_emb//,/ } --lr_schedule ${lr_schedule} --warmup_steps $warmup_steps -ih ${inhouse} --kg $kg --kg_vocab_path $kg_vocab_path \
    --data_dir data \
> ${log}

echo "Fine-tuning completed at $(date)"
echo "Log saved to: $log"
echo "Model saved to: ${save_dir_pref}/${dataset}/${run_name}/"
