"""
🐉 DRAGON Binary Classification Wrapper
========================================

Thiết kế cho Sarcasm Detection và các bài toán Binary Classification khác.

Thay vì dùng QA format phức tạp (tạo 2 choices giả tạo), 
wrapper này đơn giản thêm một Linear Layer lên top của DRAGON encoder.

Architecture:
    Input Text + Graph 
        ↓
    RoBERTa Encoder (from DRAGON)
        ↓
    Information Exchange Layers
        ↓
    GNN (from DRAGON)
        ↓
    Pooling ([CLS] token representation)
        ↓
    Dropout
        ↓
    Linear Layer (hidden_dim -> 2)
        ↓
    [Not Sarcastic, Sarcastic]

Advantages over QA format:
    ✅ Đơn giản hơn - không cần tạo fake choices
    ✅ Nhanh hơn - chỉ 1 forward pass thay vì 2
    ✅ Tự nhiên hơn - đúng bản chất của classification
    ✅ Ít memory hơn - không phải duplicate input

Author: Adapted from DRAGON (Yasunaga et al., 2022)
"""

import torch
import torch.nn as nn
from modeling.modeling_dragon import DRAGON
from transformers import AutoModel, AutoTokenizer


class DRAGONBinaryClassifier(nn.Module):
    """
    Binary Classification wrapper cho DRAGON.
    
    Args:
        args: Arguments object với các config cần thiết
        k: Số GNN layers (default: 5)
        n_ntype: Số node types trong graph (default: 4)
        n_etype: Số edge types trong graph (default: 38)
        sent_dim: Dimension của sentence embedding từ encoder
        n_concept: Số concepts trong knowledge graph
        concept_dim: Dimension của concept embeddings (default: 200)
        concept_in_dim: Input dimension cho concepts (default: 200)
        hidden_size: Hidden size của encoder (default: 1024 cho RoBERTa-large)
        pretrained_concept_emb: Pre-trained concept embeddings
        freeze_ent_emb: Có freeze concept embeddings không (default: True)
        init_range: Range để init weights
        dropout: Dropout rate (default: 0.1)
    """
    
    def __init__(self, args, k, n_ntype, n_etype, sent_dim,
                 n_concept, concept_dim, concept_in_dim, hidden_size,
                 pretrained_concept_emb=None, freeze_ent_emb=True,
                 init_range=0.02, dropout=0.1):
        super().__init__()
        
        self.args = args
        
        # Load DRAGON backbone (encoder + GNN)
        self.dragon = DRAGON(
            args=args,
            k=k,
            n_ntype=n_ntype,
            n_etype=n_etype,
            sent_dim=sent_dim,
            n_concept=n_concept,
            concept_dim=concept_dim,
            concept_in_dim=concept_in_dim,
            hidden_size=hidden_size,
            pretrained_concept_emb=pretrained_concept_emb,
            freeze_ent_emb=freeze_ent_emb,
            init_range=init_range
        )
        
        # Binary classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, 2)  # Binary: [0, 1]
        
        # Initialize classifier weights
        self.classifier.weight.data.normal_(mean=0.0, std=init_range)
        self.classifier.bias.data.zero_()
    
    def forward(self, *inputs, layer_id=-1, cache_output=False, detail=False):
        """
        Forward pass cho binary classification.
        
        Args:
            *inputs: Các inputs giống như DRAGON gốc:
                - sent_vecs: Token embeddings từ encoder
                - concept_ids: IDs của concepts trong graph
                - node_type_ids: Types của nodes
                - adj: Adjacency matrix của graph
                - ...
            layer_id: Layer để extract representation (default: -1 = last layer)
            cache_output: Có cache intermediate outputs không
            detail: Có return chi tiết không
            
        Returns:
            logits: Tensor [batch_size, 2] - logits cho 2 classes
            (optional) dragon_outputs: Outputs từ DRAGON backbone nếu detail=True
        """
        
        # Get representation từ DRAGON
        # DRAGON trả về: (logits_for_qa, hidden_states, ...)
        # Nhưng ta chỉ cần hidden states (pooled representation)
        dragon_outputs = self.dragon(*inputs, layer_id=layer_id, 
                                     cache_output=cache_output, detail=True)
        
        # Extract pooled representation
        # DRAGON's output structure: (logits, hidden_states, ...)
        # hidden_states shape: [batch_size, hidden_size]
        if isinstance(dragon_outputs, tuple):
            pooled_output = dragon_outputs[1]  # hidden_states
        else:
            pooled_output = dragon_outputs
        
        # Apply dropout và classifier
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)  # [batch_size, 2]
        
        if detail:
            return logits, dragon_outputs
        else:
            return logits
    
    def load_pretrained_dragon(self, model_path):
        """
        Load pre-trained DRAGON weights (từ general_model.pt).
        
        Args:
            model_path: Path đến pre-trained checkpoint
        """
        print(f"🔥 Loading pre-trained DRAGON from {model_path}")
        
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Extract DRAGON weights (bỏ qua classification head cũ)
        dragon_state_dict = {}
        for key, value in checkpoint.items():
            # Chỉ load encoder + GNN, bỏ qua classifier cũ
            if not key.startswith('classifier'):
                # Remove 'dragon.' prefix if exists
                new_key = key.replace('dragon.', '')
                dragon_state_dict[new_key] = value
        
        # Load vào DRAGON backbone
        missing_keys, unexpected_keys = self.dragon.load_state_dict(
            dragon_state_dict, strict=False
        )
        
        if missing_keys:
            print(f"⚠️  Missing keys (expected for new classifier): {missing_keys[:5]}...")
        if unexpected_keys:
            print(f"⚠️  Unexpected keys: {unexpected_keys[:5]}...")
        
        print("✅ Pre-trained DRAGON loaded successfully!")
        print("🎯 Binary classifier head initialized randomly (will be fine-tuned)")
    
    def freeze_encoder(self):
        """Freeze DRAGON encoder (chỉ train classifier head)."""
        for param in self.dragon.encoder.parameters():
            param.requires_grad = False
        print("❄️  DRAGON encoder frozen")
    
    def unfreeze_encoder(self):
        """Unfreeze DRAGON encoder (fine-tune toàn bộ)."""
        for param in self.dragon.encoder.parameters():
            param.requires_grad = True
        print("🔥 DRAGON encoder unfrozen")
    
    def freeze_gnn(self):
        """Freeze GNN layers."""
        for param in self.dragon.gnn.parameters():
            param.requires_grad = False
        print("❄️  GNN frozen")
    
    def unfreeze_gnn(self):
        """Unfreeze GNN layers."""
        for param in self.dragon.gnn.parameters():
            param.requires_grad = True
        print("🔥 GNN unfrozen")


class DRAGONBinaryDataLoader(nn.Module):
    """
    DataLoader cho Binary Classification với DRAGON.
    
    NOTE: Thực tế không cần class này vì data_utils.DRAGON_DataLoader đã support binary.
    Giữ lại để tương thích với code docs.
    """
    pass


def create_optimizer_grouped_parameters(model, args):
    """
    Tạo optimizer với learning rates khác nhau cho các parts.
    
    Strategy:
        - Encoder: Lower LR (2e-5) - đã pre-trained
        - GNN: Medium LR (1e-3) - đã pre-trained
        - Classifier: Higher LR (1e-3) - random init
    
    Args:
        model: DRAGONBinaryClassifier instance
        args: Arguments với learning rates
    
    Returns:
        List of parameter groups cho optimizer
    """
    
    no_decay = ['bias', 'LayerNorm.weight']
    
    optimizer_grouped_parameters = [
        # Encoder với weight decay
        {
            'params': [p for n, p in model.dragon.encoder.named_parameters()
                      if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay,
            'lr': args.encoder_lr
        },
        # Encoder không weight decay
        {
            'params': [p for n, p in model.dragon.encoder.named_parameters()
                      if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,
            'lr': args.encoder_lr
        },
        # GNN
        {
            'params': model.dragon.gnn.parameters(),
            'weight_decay': args.weight_decay,
            'lr': args.decoder_lr
        },
        # Classifier head (higher LR vì random init)
        {
            'params': model.classifier.parameters(),
            'weight_decay': args.weight_decay,
            'lr': args.decoder_lr
        }
    ]
    
    return optimizer_grouped_parameters


# ============================================================================
# Usage Example
# ============================================================================

if __name__ == "__main__":
    """
    Example usage của DRAGONBinaryClassifier.
    """
    
    print("🐉 DRAGON Binary Classification Example")
    print("=" * 60)
    
    # Giả lập args
    class Args:
        encoder = 'roberta-large'
        decoder = 'gnn'
        k = 5
        n_ntype = 4
        n_etype = 38
        concept_dim = 200
        dropout = 0.1
        encoder_lr = 2e-5
        decoder_lr = 1e-3
        weight_decay = 0.01
    
    args = Args()
    
    # Model config
    hidden_size = 1024  # RoBERTa-large
    n_concept = 799273  # ConceptNet
    
    print("\n1️⃣  Khởi tạo model...")
    model = DRAGONBinaryClassifier(
        args=args,
        k=args.k,
        n_ntype=args.n_ntype,
        n_etype=args.n_etype,
        sent_dim=hidden_size,
        n_concept=n_concept,
        concept_dim=args.concept_dim,
        concept_in_dim=args.concept_dim,
        hidden_size=hidden_size,
        pretrained_concept_emb=None,
        freeze_ent_emb=True,
        dropout=args.dropout
    )
    print(f"✅ Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    print("\n2️⃣  Load pre-trained weights...")
    # model.load_pretrained_dragon('models/general_model.pt')
    print("   (Bỏ qua vì chưa có file)")
    
    print("\n3️⃣  Tạo optimizer với grouped parameters...")
    param_groups = create_optimizer_grouped_parameters(model, args)
    print(f"✅ Created {len(param_groups)} parameter groups")
    
    print("\n4️⃣  Test forward pass...")
    batch_size = 2
    seq_len = 128
    n_nodes = 50
    
    # Dummy inputs
    sent_vecs = torch.randn(batch_size, seq_len, hidden_size)
    concept_ids = torch.randint(0, n_concept, (batch_size, n_nodes))
    node_type_ids = torch.randint(0, args.n_ntype, (batch_size, n_nodes))
    adj = torch.randn(batch_size, args.n_etype, n_nodes, n_nodes)
    
    # Forward
    try:
        logits = model(sent_vecs, concept_ids, node_type_ids, adj)
        print(f"✅ Forward pass successful!")
        print(f"   Input shape: {sent_vecs.shape}")
        print(f"   Output shape: {logits.shape}")
        print(f"   Output: {logits}")
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 Example completed!")
    print("\n💡 Next steps:")
    print("   1. Integrate vào dragon.py training loop")
    print("   2. Update data preprocessing cho binary labels")
    print("   3. Create training script: run_train__isarcasm_binary.sh")
    print("   4. Fine-tune và enjoy the results! 🚀")
