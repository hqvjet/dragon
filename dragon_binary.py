"""
Binary Classification Training Script for DRAGON
Simplified training loop for Sarcasm Detection and other binary tasks.
"""

import argparse
import logging
import random
import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️  wandb not available, logging disabled")

from modeling.modeling_dragon_binary import DRAGONBinaryClassifier, create_optimizer_grouped_parameters
from utils import data_utils, utils, parser_utils

logger = logging.getLogger(__name__)


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_data(args, devices, kg):
    """Load dataset for binary classification."""
    set_seed(args.seed)
    
    dataset = data_utils.DRAGON_DataLoader(
        args, 
        args.train_statements, args.train_adj,
        args.dev_statements, args.dev_adj,
        args.test_statements, args.test_adj,
        batch_size=args.batch_size, 
        eval_batch_size=args.eval_batch_size,
        device=devices,
        model_name=args.encoder,
        max_node_num=args.max_node_num, 
        max_seq_length=args.max_seq_len,
        is_inhouse=args.inhouse, 
        inhouse_train_qids_path=args.inhouse_train_qids,
        subsample=args.subsample, 
        n_train=args.n_train, 
        debug=args.debug, 
        cxt_node_connects_all=args.cxt_node_connects_all, 
        kg=kg
    )
    
    return dataset


def construct_model(args, kg, dataset):
    """Construct DRAGON binary classifier."""
    # Load concept embeddings
    cp_emb = [np.load(path) for path in args.ent_emb_paths]
    cp_emb = np.concatenate(cp_emb, 1)
    cp_emb = torch.tensor(cp_emb, dtype=torch.float)
    
    concept_num, concept_in_dim = cp_emb.size(0), cp_emb.size(1)
    print(f'| num_concepts: {concept_num} |')
    
    if args.random_ent_emb:
        cp_emb = None
        freeze_ent_emb = False
        concept_in_dim = args.gnn_dim
    else:
        freeze_ent_emb = args.freeze_ent_emb
    
    # Set node and edge types based on KG
    if kg == "cpnet":
        n_ntype = 4
        n_etype = 38
    elif kg == "ddb":
        n_ntype = 4
        n_etype = 34
    elif kg == "umls":
        n_ntype = 4
        n_etype = dataset.final_num_relation * 2
    else:
        raise ValueError(f"Invalid KG: {kg}")
    
    if args.cxt_node_connects_all:
        n_etype += 2
    
    print(f'n_ntype={n_ntype}, n_etype={n_etype}')
    
    # Construct binary classifier
    model = DRAGONBinaryClassifier(
        args=args,
        k=args.k,
        n_ntype=n_ntype,
        n_etype=n_etype,
        sent_dim=dataset.sent_dim if hasattr(dataset, 'sent_dim') else 1024,
        n_concept=concept_num,
        concept_dim=args.gnn_dim,
        concept_in_dim=concept_in_dim,
        hidden_size=1024,  # RoBERTa-large
        pretrained_concept_emb=cp_emb,
        freeze_ent_emb=freeze_ent_emb,
        init_range=args.init_range,
        dropout=args.dropoutf
    )
    
    return model


def evaluate(args, model, eval_loader, device):
    """Evaluate model on dev/test set."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating"):
            qids, labels, *input_data = batch
            labels = labels.to(device)
            
            # Forward pass
            logits = model(*input_data)
            
            # Calculate loss
            loss = F.cross_entropy(logits, labels)
            total_loss += loss.item() * labels.size(0)
            
            # Calculate accuracy
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / total
    accuracy = correct / total
    
    return avg_loss, accuracy, all_preds, all_labels


def train(args):
    """Main training loop."""
    print("=" * 60)
    print("🐉 DRAGON Binary Classification Training")
    print("=" * 60)
    print(f"Args: {json.dumps(vars(args), indent=2)}")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    n_gpu = torch.cuda.device_count()
    print(f"Device: {device}, n_gpu: {n_gpu}")
    
    # Setup wandb
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.init(project="dragon-binary", name=args.run_name, config=vars(args))
    
    # Load data
    print("\n📦 Loading dataset...")
    dataset = load_data(args, device, args.kg)
    train_loader = dataset.train()
    dev_loader = dataset.dev()
    test_loader = dataset.test() if dataset.test_size() > 0 else None
    
    print(f"Train: {dataset.train_size()} | Dev: {dataset.dev_size()} | Test: {dataset.test_size()}")
    
    # Construct model
    print("\n🏗️  Building model...")
    model = construct_model(args, args.kg, dataset)
    
    # Load pre-trained weights
    if args.load_model_path:
        print(f"\n🔥 Loading pre-trained DRAGON from {args.load_model_path}")
        model.load_pretrained_dragon(args.load_model_path)
    
    model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,} | Trainable: {trainable_params:,}")
    
    # Setup optimizer
    print("\n⚙️  Setting up optimizer...")
    # Create Args class for optimizer function
    class OptimizerArgs:
        encoder_lr = args.elr
        decoder_lr = args.dlr
        weight_decay = args.weight_decay
    
    param_groups = create_optimizer_grouped_parameters(model, OptimizerArgs())
    optimizer = AdamW(param_groups, eps=args.adam_epsilon)
    
    # Calculate total steps
    num_training_steps = len(train_loader) * args.n_epochs
    num_warmup_steps = args.warmup_steps
    
    # Setup scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    print(f"Total steps: {num_training_steps} | Warmup: {num_warmup_steps}")
    
    # Training loop
    print("\n🚀 Starting training...")
    best_dev_acc = 0
    best_epoch = 0
    global_step = 0
    
    for epoch in range(args.n_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{args.n_epochs}")
        print(f"{'='*60}")
        
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            qids, labels, *input_data = batch
            labels = labels.to(device)
            
            # Forward pass
            logits = model(*input_data)
            
            # Calculate loss
            loss = F.cross_entropy(logits, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            
            optimizer.step()
            scheduler.step()
            
            # Stats
            train_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=-1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            global_step += 1
            
            # Log
            if args.use_wandb and WANDB_AVAILABLE and global_step % 10 == 0:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/lr": scheduler.get_last_lr()[0],
                    "global_step": global_step
                })
            
            if args.debug:
                break
        
        # Training stats
        train_loss /= train_total
        train_acc = train_correct / train_total
        print(f"\n📊 Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        
        # Evaluation
        print("\n📊 Evaluating on dev set...")
        dev_loss, dev_acc, dev_preds, dev_labels = evaluate(args, model, dev_loader, device)
        print(f"Dev Loss: {dev_loss:.4f} | Dev Acc: {dev_acc:.4f}")
        
        # Log to wandb
        if args.use_wandb and WANDB_AVAILABLE:
            wandb.log({
                "epoch": epoch + 1,
                "train/loss_epoch": train_loss,
                "train/acc_epoch": train_acc,
                "dev/loss": dev_loss,
                "dev/acc": dev_acc
            })
        
        # Save best model
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            best_epoch = epoch + 1
            
            if args.save_model:
                save_path = os.path.join(args.save_dir, "model.pt")
                os.makedirs(args.save_dir, exist_ok=True)
                torch.save(model.state_dict(), save_path)
                print(f"✅ Saved best model to {save_path}")
        
        print(f"\n🏆 Best Dev Acc: {best_dev_acc:.4f} (Epoch {best_epoch})")
        
        # Early stopping
        if epoch + 1 - best_epoch >= args.max_epochs_before_stop:
            print(f"\n⚠️  Early stopping at epoch {epoch + 1}")
            break
    
    # Test evaluation
    if test_loader:
        print("\n" + "="*60)
        print("📊 Final evaluation on test set...")
        print("="*60)
        
        # Load best model
        if args.save_model:
            model.load_state_dict(torch.load(os.path.join(args.save_dir, "model.pt")))
        
        test_loss, test_acc, test_preds, test_labels = evaluate(args, model, test_loader, device)
        print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
        
        if args.use_wandb and WANDB_AVAILABLE:
            wandb.log({"test/loss": test_loss, "test/acc": test_acc})
    
    print("\n" + "="*60)
    print("🎉 Training completed!")
    print("="*60)
    
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.finish()


def main():
    """Entry point."""
    parser = parser_utils.get_parser()
    args, _ = parser.parse_known_args()
    
    # Set defaults for binary classification
    args.end_task = True
    args.mlm_task = False
    args.link_task = False
    
    # Create save directory
    if not args.debug and args.save_model:
        os.makedirs(args.save_dir, exist_ok=True)
    
    # Train
    train(args)


if __name__ == "__main__":
    main()
