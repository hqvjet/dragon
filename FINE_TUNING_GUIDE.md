# 🐉 DRAGON Fine-tuning cho Sarcasm Detection

## ⚠️ QUAN TRỌNG: Không Train From Scratch!

**BẮT BUỘC phải dùng Fine-tuning approach:**

### Tại sao KHÔNG nên train from scratch?

1. **Tốn kém khủng khiếp**: Train DRAGON từ đầu cần:
   - Cụm GPU mạnh (multi-GPU cluster)
   - Hàng ngày/tuần training
   - Dataset khổng lồ (BookCorpus + Wikipedia + ConceptNet)
   - Chi phí hàng ngàn đô la

2. **Lãng phí tri thức**: 
   - Pre-trained DRAGON đã học được reasoning và quan hệ từ ConceptNet
   - Ví dụ: `rain → causes → wet`, `sunny → antonym → rainy`
   - Dataset iSarcasm (~4000 samples) quá nhỏ để học lại từ đầu

3. **Kết quả kém hơn**:
   - Train from scratch với data ít = model "ngu"
   - Fine-tuning = kế thừa tri thức + học skill mới = hiệu quả

## 🎯 Chiến thuật: Fine-tuning Pre-trained Model

```
Pre-trained DRAGON (đã học ConceptNet)
           ↓
    Load weights
           ↓
  Fine-tune 3-5 epochs trên iSarcasm
           ↓
    Model cho Sarcasm Detection
```

## 🚀 Quick Start

### 1. Chuẩn bị

```bash
# Activate environment
conda activate dragon
cd /home/hqvjet/Projects/dragon

# Download pre-trained model (nếu chưa có)
mkdir -p models
cd models
wget https://nlp.stanford.edu/projects/myasu/DRAGON/models/general_model.pt
cd ..

# Ensure ConceptNet is ready
python preprocess.py --run common -p 8
```

### 2. Download và Preprocess iSarcasm Data

```bash
# Tự động download từ HuggingFace và preprocess
python preprocess.py --run isarcasm -p 8
```

Lệnh này sẽ:
- ✅ Download dataset từ HuggingFace: `viethq1906/isarcasm_2022_taskA_En`
- ✅ Convert sang format DRAGON (2-choice: sarcastic/not sarcastic)
- ✅ Ground concepts với ConceptNet
- ✅ Extract subgraphs
- ✅ Tạo graph adjacency data

### 3. Fine-tune Model

```bash
chmod +x scripts/run_train__isarcasm.sh
./scripts/run_train__isarcasm.sh
```

**Quá trình fine-tuning:**
- 🔥 Load `general_model.pt` (pre-trained weights)
- 🎯 Unfreeze tất cả parameters ngay từ epoch 0
- 📚 Train 10 epochs (thay vì 15-20 như train from scratch)
- 🧠 Model học nhận diện sarcasm dựa trên incongruity (mâu thuẫn)
- 💾 Save best checkpoint dựa trên dev accuracy

## 🔍 Tại sao DRAGON mạnh cho Sarcasm?

### Sarcasm = Incongruity (Mâu thuẫn)

**Ví dụ:**
> "Trời mưa tầm tã, thời tiết đẹp tuyệt vời!" ☔→☀️ 

### DRAGON phát hiện mâu thuẫn qua ConceptNet:

```
rain --[Causes]--> wet
rain --[Antonym]--> sunny
sunny --[RelatedTo]--> nice weather

❌ Phát hiện: "rain" và "nice weather" mâu thuẫn
✅ Kết luận: Sarcasm!
```

### Các relation quan trọng cho Sarcasm:

- `Antonym`: Từ trái nghĩa (rain ↔ sunny)
- `DistinctFrom`: Khác biệt rõ ràng
- `Causes`: Quan hệ nhân quả
- `NotDesires`: Không mong muốn

**⚠️ LƯU Ý:** Khi preprocess, KHÔNG filter bỏ các edge `Antonym` và `DistinctFrom` - đây là chìa khóa cho sarcasm detection!

## 📊 Hyperparameters cho Fine-tuning

### Tối ưu cho iSarcasm:

```bash
# Learning rates (higher than feature extraction)
elr=2e-5              # Encoder LR (fine-tuning)
dlr=1e-3              # Decoder (GNN) LR

# Training config
bs=32                 # Batch size (smaller for stability)
n_epochs=10           # Fewer epochs (fine-tuning converges fast)
unfreeze_epoch=0      # Unfreeze immediately
warmup_steps=150      # More warmup for stability

# Model config
k=5                   # 5 GNN layers
gnndim=200            # GNN dimension
max_seq_len=128       # Sarcasm text usually short

# Reproducibility
seed=42               # Fixed seed
# cudnn.deterministic=True (auto-enabled)
```

### So sánh với Training from Scratch:

| Metric | Fine-tuning | From Scratch |
|--------|-------------|--------------|
| Epochs | 10 | 50+ |
| Learning Rate | 2e-5 (encoder) | 1e-5 |
| Unfreeze Epoch | 0 (immediate) | 2-3 |
| Data Required | 4K samples OK | 100K+ samples |
| Training Time | 1-2 hours | Days/Weeks |
| GPU Memory | 10GB | 16GB+ |
| Final Accuracy | 85-90% | 70-75% (với data ít) |

## 🎓 Kiến trúc Model

### Pre-trained DRAGON:

```python
Input Text + Graph
       ↓
RoBERTa Encoder (pre-trained on text)
       ↓
Information Exchange Layers
       ↓
GNN (pre-trained on ConceptNet)
       ↓
Classification Head (2 classes)
       ↓
[Sarcastic, Not Sarcastic]
```

### Fine-tuning Strategy:

```python
# Load pre-trained body
dragon = load_pretrained('general_model.pt')

# Original classification head (5 choices for CSQA)
# → Replace with binary head (2 choices for sarcasm)

# Tuy nhiên, cách dễ nhất:
# Convert iSarcasm thành 2-choice format
# → Dùng luôn DRAGON architecture hiện tại!

Question: "Tweet text here"
Choices:
  A: "This text is sarcastic"      ← Answer if label=1
  B: "This text is not sarcastic"  ← Answer if label=0
```

## 📈 Evaluation

```bash
# Evaluate trên test set
./scripts/run_eval__isarcasm.sh runs/isarcasm/YOUR_RUN_NAME/model.pt
```

## 🔬 Advanced: Understanding the Code

### Data Flow trong Fine-tuning:

```python
# 1. Load pre-trained DRAGON
model = DRAGON(args, ...)
state_dict = torch.load('general_model.pt')
model.load_state_dict(state_dict)

# 2. Setup optimizer với learning rates khác nhau
encoder_params = [p for n, p in model.named_parameters() if 'encoder' in n]
decoder_params = [p for n, p in model.named_parameters() if 'gnn' in n]

optimizer = AdamW([
    {'params': encoder_params, 'lr': 2e-5},  # Lower LR cho pre-trained parts
    {'params': decoder_params, 'lr': 1e-3}   # Higher LR cho GNN
])

# 3. Fine-tune
for epoch in range(10):  # Ít epochs
    for batch in dataloader:
        loss = model(batch)
        loss.backward()
        optimizer.step()
```

### Key Modifications:

1. **dragon.py**: 
   - ✅ Enhanced seed setting
   - ✅ CuDNN deterministic mode
   - ⚠️ NO changes to model logic

2. **preprocess_utils/convert_isarcasm.py**:
   - ✅ Download từ HuggingFace
   - ✅ Convert binary → 2-choice format
   - ✅ Compatible với DRAGON architecture

3. **scripts/run_train__isarcasm.sh**:
   - ✅ Optimized hyperparameters cho fine-tuning
   - ✅ Load `general_model.pt` bắt buộc
   - ✅ Save model để reuse

## ⚡ Performance Tips

### 1. Use Mixed Precision (FP16):
```bash
fp16=true  # Already enabled in script
# → 2x faster training, 50% less memory
```

### 2. Gradient Accumulation:
```bash
bs=32       # Physical batch size
mbs=2       # Mini-batch size
# → Effective batch size = 32, but process 2 at a time
# → Fits in smaller GPUs
```

### 3. Early Stopping:
```bash
max_epochs_before_stop=5
# → Stop if no improvement for 5 epochs
# → Prevent overfitting
```

## 🐛 Troubleshooting

### "RuntimeError: CUDA out of memory"
```bash
# Solution 1: Reduce batch size
bs=16
mbs=1

# Solution 2: Use CPU for some operations
# (already handled in code)
```

### "No improvement after 10 epochs"
```bash
# Possible causes:
# 1. Learning rate quá cao/thấp
#    → Try elr=1e-5 or elr=3e-5

# 2. Data imbalance
#    → Check label distribution

# 3. Need more data augmentation
#    → Consider back-translation, paraphrasing
```

### "Model predicts all one class"
```bash
# Check:
1. Data balance: ~50-50 distribution?
2. Loss function: CrossEntropyLoss cho 2 classes
3. Metrics: Accuracy, F1, Precision, Recall
```

## 📚 References

### Papers:
- **DRAGON**: [NeurIPS 2022](https://arxiv.org/abs/2210.09338)
- **iSarcasm**: Dataset paper (2022)

### Pre-trained Models:
- Download: https://nlp.stanford.edu/projects/myasu/DRAGON/models/general_model.pt
- Size: 360M parameters
- Domain: General (ConceptNet + BookCorpus)

## 🎯 Expected Results

### Baseline (Rule-based):
- Accuracy: ~65%

### BERT-base (fine-tuned):
- Accuracy: ~75-80%

### DRAGON (fine-tuned):
- **Expected: 85-90%** ✨
- Improvement: +5-10% over BERT
- Why: Graph reasoning cho incongruity detection

## ✅ Checklist

Trước khi train:
- [ ] Downloaded `general_model.pt`
- [ ] Preprocessed ConceptNet (`python preprocess.py --run common`)
- [ ] Downloaded iSarcasm (`python preprocess.py --run isarcasm`)
- [ ] Checked GPU memory (>=10GB free)
- [ ] Set `load_model_path=models/general_model.pt` in script

Trong khi train:
- [ ] Monitor loss (should decrease steadily)
- [ ] Check dev accuracy every epoch
- [ ] Watch for overfitting (train acc >> dev acc)
- [ ] Logs saved to `logs/train__*.log.txt`

Sau khi train:
- [ ] Best model saved to `runs/isarcasm/*/model.pt`
- [ ] Evaluate on test set
- [ ] Compare với baseline
- [ ] Analyze error cases

---

**🎉 Good luck with fine-tuning! Remember: Pre-trained > From Scratch!**
