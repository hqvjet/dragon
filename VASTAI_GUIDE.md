# 🚀 DRAGON Training on Vast.ai - Complete Guide

## 📋 Prerequisites

1. **Vast.ai Account**: https://vast.ai/
2. **Local Machine**: Linux/Mac với rsync và ssh
3. **Budget**: ~$0.5-1/hour cho RTX 3090/4090

## 🎯 Step-by-Step Guide

### 1️⃣ Rent GPU trên Vast.ai

#### Recommendations:
```
GPU: RTX 3090 (24GB) hoặc RTX 4090 (24GB)
VRAM: ≥ 16GB (tối thiểu), 24GB (recommended)
Storage: ≥ 50GB
Connection: ≥ 100 Mbps
Price: $0.3-0.7/hour
```

#### Search Filters:
- GPU RAM: ≥ 16GB
- Storage: ≥ 50GB  
- CUDA: ≥ 11.8
- Sort by: $/hour (lowest first)

#### Start Instance:
1. Click "RENT" trên instance bạn chọn
2. Select: **PyTorch** image (hoặc Ubuntu 22.04)
3. Click "Create & Start"
4. Wait ~1-2 minutes for instance to start

### 2️⃣ Connect to Instance

Sau khi instance start, lấy SSH info:

```bash
# Vast.ai sẽ show command như:
ssh -p 12345 root@123.45.67.89

# Test connection:
ssh -p PORT root@IP_ADDRESS
```

### 3️⃣ Upload Project to Vast.ai

Từ **máy local** (đang ở trong `/home/viethq/Projects/dragon`):

```bash
# Tạo tar file để upload nhanh hơn
cd /home/viethq/Projects
tar -czf dragon.tar.gz dragon/ --exclude='dragon/venv' --exclude='dragon/__pycache__' --exclude='dragon/.git'

# Upload tar file
scp -P PORT dragon.tar.gz root@IP_ADDRESS:/root/

# SSH vào và extract
ssh -p PORT root@IP_ADDRESS
cd /root
tar -xzf dragon.tar.gz
cd dragon
```

**HOẶC** dùng rsync (nhanh hơn, sync incremental):

```bash
# Từ máy local
rsync -avz -e "ssh -p PORT" \
  --exclude 'venv' \
  --exclude '__pycache__' \
  --exclude '.git' \
  --exclude 'data/cpnet/conceptnet-assertions*' \
  /home/viethq/Projects/dragon/ \
  root@IP_ADDRESS:/root/dragon/
```

### 4️⃣ Run Setup Script

SSH vào Vast.ai instance:

```bash
ssh -p PORT root@IP_ADDRESS

# Navigate to project
cd /root/dragon

# Make setup script executable
chmod +x setup_vastai.sh

# Run setup (takes ~10-15 minutes)
./setup_vastai.sh
```

**Setup script sẽ:**
- ✅ Install Python 3.10 + dependencies
- ✅ Create virtual environment
- ✅ Install PyTorch + CUDA 11.8
- ✅ Install PyTorch Geometric
- ✅ Download pre-trained DRAGON model (1.4GB)
- ✅ Download ConceptNet (449MB)
- ✅ Install spaCy model

### 5️⃣ Run Training (Auto)

**Option A: Fully Automated** (Recommended)

```bash
chmod +x run_vastai.sh
./run_vastai.sh
```

Này sẽ tự động:
1. Preprocess ConceptNet
2. Preprocess iSarcasm  
3. Start training
4. Save best model

**Option B: Manual Steps**

```bash
# Activate environment
source venv/bin/activate

# Preprocess ConceptNet (~5 minutes)
python3 preprocess.py --run common -p 8

# Preprocess iSarcasm (~2 minutes)
python3 preprocess.py --run isarcasm -p 8

# Start training (~1-2 hours)
chmod +x scripts/run_train__isarcasm_binary.sh
./scripts/run_train__isarcasm_binary.sh
```

### 6️⃣ Monitor Training

Mở terminal mới (keep SSH connection alive):

```bash
# Terminal 1: Monitor log
ssh -p PORT root@IP_ADDRESS
cd /root/dragon
tail -f logs/train__dragon_binary*.log.txt

# Terminal 2: Monitor GPU
ssh -p PORT root@IP_ADDRESS
watch -n 1 nvidia-smi
```

### 7️⃣ Download Trained Model

Sau khi training xong:

```bash
# Từ máy local
scp -P PORT root@IP_ADDRESS:/root/dragon/runs/isarcasm/dragon_binary__*/model.pt ./trained_model.pt

# Or download entire runs folder
rsync -avz -e "ssh -p PORT" \
  root@IP_ADDRESS:/root/dragon/runs/ \
  ./vast_runs/
```

### 8️⃣ Stop Instance (IMPORTANT!)

⚠️ **ĐỪNG QUÊN DESTROY INSTANCE** sau khi xong để không bị charge tiếp!

```bash
# On Vast.ai website:
1. Go to "Instances"
2. Click "Destroy" button
3. Confirm destruction

# Or use CLI:
vastai destroy instance INSTANCE_ID
```

## 🔧 Troubleshooting

### Out of Memory (OOM)

```bash
# Edit scripts/run_train__isarcasm_binary.sh
bs=16                # Reduce from 32
mbs=2                # Keep at 2
max_node_num=150     # Reduce from 200
```

### Connection Lost

```bash
# Use tmux/screen để training không bị dừng khi disconnect
ssh -p PORT root@IP_ADDRESS
tmux new -s dragon

# Run training inside tmux
./run_vastai.sh

# Detach: Ctrl+B then D
# Reattach: tmux attach -t dragon
```

### Slow Upload

```bash
# Compress files before upload
tar -czf dragon_minimal.tar.gz \
  dragon_binary.py \
  preprocess.py \
  modeling/ \
  scripts/ \
  utils/ \
  preprocess_utils/

# Upload compressed file
scp -P PORT dragon_minimal.tar.gz root@IP:/root/
ssh -p PORT root@IP "cd /root && tar -xzf dragon_minimal.tar.gz"
```

## 💰 Cost Estimation

| GPU | Price/hour | Training Time | Total Cost |
|-----|------------|---------------|------------|
| RTX 3090 (24GB) | $0.40 | ~1.5 hours | **~$0.60** |
| RTX 4090 (24GB) | $0.60 | ~1 hour | **~$0.60** |
| A100 (40GB) | $1.50 | ~45 minutes | **~$1.10** |

**Lưu ý:** Giá có thể thay đổi. Check real-time pricing trên Vast.ai.

## 📊 Expected Results

```
Epoch 1/10: Train Acc=0.72, Dev Acc=0.75
Epoch 2/10: Train Acc=0.81, Dev Acc=0.82
Epoch 3/10: Train Acc=0.86, Dev Acc=0.85
...
Epoch 7/10: Train Acc=0.92, Dev Acc=0.88 ← Best
Epoch 8/10: Train Acc=0.93, Dev Acc=0.87
Early stopping!

Test Accuracy: 0.87-0.89
```

## 🎯 Quick Commands Cheatsheet

```bash
# Setup
./setup_vastai.sh

# Train (auto)
./run_vastai.sh

# Train (manual)
source venv/bin/activate
python3 preprocess.py --run common -p 8
python3 preprocess.py --run isarcasm -p 8
./scripts/run_train__isarcasm_binary.sh

# Monitor
tail -f logs/train__*.log.txt
watch nvidia-smi

# Download model
scp -P PORT root@IP:/root/dragon/runs/isarcasm/*/model.pt ./

# Use tmux
tmux new -s dragon     # Create session
Ctrl+B then D          # Detach
tmux attach -t dragon  # Reattach
```

## 🆘 Need Help?

1. Check logs: `cat logs/train__*.log.txt`
2. Check GPU: `nvidia-smi`
3. Check disk: `df -h`
4. Check memory: `free -h`

## ✅ Checklist

Trước khi training:
- [ ] Đã rent GPU trên Vast.ai (≥16GB VRAM)
- [ ] Đã upload code lên instance
- [ ] Đã chạy `setup_vastai.sh` thành công
- [ ] Đã test SSH connection
- [ ] Đã setup tmux/screen để avoid disconnect

Sau khi training:
- [ ] Đã download model về local
- [ ] Đã download logs về local
- [ ] **ĐÃ DESTROY INSTANCE** ⚠️

---

**Good luck! 🚀**
