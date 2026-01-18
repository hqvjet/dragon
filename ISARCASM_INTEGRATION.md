# iSarcasm Dataset Integration for DRAGON

## Tổng quan

Đã tích hợp dataset iSarcasm (sarcasm detection) vào DRAGON model. Dataset được load từ HuggingFace: `viethq1906/isarcasm_2022_taskA_En`

## Kiến trúc và Data Flow

### 1. Kiến trúc Model DRAGON

```
Input Text → Tokenization → BERT/RoBERTa Encoder
                                    ↓
                          Information Exchange
                                    ↓
                                  GNN ← Knowledge Graph (ConceptNet)
                                    ↓
                          Information Exchange
                                    ↓
                            Classification Head → Output
```

**Các thành phần chính:**
- **Language Model**: BERT/RoBERTa để encode text
- **Knowledge Graph**: ConceptNet để cung cấp thông tin thế giới
- **GNN**: Graph Neural Network để reasoning trên KG
- **Information Exchange**: Bidirectional flow giữa LM và GNN

### 2. Data Flow

```
Raw Data (HuggingFace) 
    ↓
Convert to JSONL format (preprocess_utils/convert_isarcasm.py)
    ↓
Statement files (data/isarcasm/*.jsonl)
    ↓
Grounding concepts (link text to ConceptNet entities)
    ↓
Grounded files (data/isarcasm/grounded/*.grounded.jsonl)
    ↓
Extract subgraphs from ConceptNet
    ↓
Graph adjacency files (data/isarcasm/graph/*.graph.adj.pk)
    ↓
Training with DRAGON model
    ↓
Prediction
```

### 3. Format dữ liệu

**Input format (sau khi convert):**
```json
{
  "id": "isarcasm_train_0",
  "question": {
    "stem": "This is the text to classify",
    "choices": [
      {"label": "A", "text": "This text is sarcastic"},
      {"label": "B", "text": "This text is not sarcastic"}
    ]
  },
  "answerKey": "A"
}
```

Dataset gốc là binary classification, được convert sang 2-choice format để phù hợp với DRAGON.

## Setup và Sử dụng

### 1. Cài đặt dependencies

Đảm bảo đã cài đặt các dependencies cần thiết:
```bash
conda activate dragon
pip install datasets  # For HuggingFace datasets
```

### 2. Download và preprocess data

**Bước 1: Download và convert từ HuggingFace**
```bash
# Tự động download từ HuggingFace và convert sang format DRAGON
python preprocess.py --run isarcasm -p 8
```

Lệnh này sẽ:
1. Download dataset từ HuggingFace
2. Convert sang format JSONL của DRAGON
3. Ground concepts với ConceptNet
4. Extract subgraphs
5. Tạo graph adjacency data

**Hoặc chạy từng bước riêng lẻ:**

```bash
# Bước 1: Download và convert
python -c "from preprocess_utils.convert_isarcasm import convert_isarcasm_to_dragon; convert_isarcasm_to_dragon()"

# Bước 2: Ground concepts (cần có ConceptNet)
python preprocess.py --run common -p 8  # Setup ConceptNet nếu chưa có
# Sau đó ground cho iSarcasm
```

### 3. Training

**Sử dụng script có sẵn:**
```bash
chmod +x scripts/run_train__isarcasm.sh
./scripts/run_train__isarcasm.sh
```

**Hoặc chạy trực tiếp với custom parameters:**
```bash
export CUDA_VISIBLE_DEVICES=0

python3 dragon.py \
    --dataset isarcasm \
    --encoder roberta-large \
    --seed 42 \
    -k 5 \
    --gnn_dim 200 \
    -elr 1e-5 \
    -dlr 1e-3 \
    -bs 64 \
    --n_epochs 15 \
    --load_model_path models/general_model.pt \
    --kg cpnet \
    --kg_vocab_path data/cpnet/concept.txt \
    --ent_emb_paths data/cpnet/tzw.ent.npy \
    --data_dir data
```

### 4. Evaluation

```bash
chmod +x scripts/run_eval__isarcasm.sh
./scripts/run_eval__isarcasm.sh /path/to/model.pt
```

## Reproducibility

### Seed Support

Code đã được cải thiện để đảm bảo reproducibility:

**Seeds được set ở:**
1. Python random: `random.seed(seed)`
2. NumPy: `np.random.seed(seed)`
3. PyTorch CPU: `torch.manual_seed(seed)`
4. PyTorch CUDA: `torch.cuda.manual_seed_all(seed)`
5. CuDNN deterministic: `torch.backends.cudnn.deterministic = True`
6. CuDNN benchmark: `torch.backends.cudnn.benchmark = False`

**Sử dụng:**
```bash
# Chỉ định seed qua command line
python dragon.py --seed 42 ...

# Hoặc trong script
seed=42
```

**Lưu ý về reproducibility:**
- Với cùng seed, kết quả sẽ giống nhau khi chạy trên cùng hardware
- Multi-GPU: Mỗi GPU process sử dụng seed khác nhau để đảm bảo diversity
- Deterministic mode có thể làm chậm training (~10-20%)
- Để tắt deterministic mode (nhanh hơn nhưng không reproducible), sửa trong `dragon.py`:
  ```python
  torch.backends.cudnn.deterministic = False
  torch.backends.cudnn.benchmark = True
  ```

### Kiểm tra reproducibility

```bash
# Chạy 2 lần với cùng seed
python dragon.py --seed 42 --debug ...
python dragon.py --seed 42 --debug ...

# So sánh log outputs - nên giống nhau
```

## Cấu trúc Files

```
dragon/
├── preprocess_utils/
│   └── convert_isarcasm.py          # Converter cho iSarcasm
├── scripts/
│   ├── run_train__isarcasm.sh       # Training script
│   └── run_eval__isarcasm.sh        # Evaluation script
├── data/
│   └── isarcasm/
│       ├── train.jsonl              # Raw converted data
│       ├── dev.jsonl
│       ├── test.jsonl
│       ├── statement/               # Same as above (DRAGON format)
│       ├── grounded/                # Grounded concepts
│       └── graph/                   # Graph adjacency data
├── dragon.py                         # Main training script (enhanced seed support)
└── preprocess.py                     # Data preprocessing (added isarcasm support)
```

## Hyperparameters cho iSarcasm

Default settings trong `scripts/run_train__isarcasm.sh`:

```bash
encoder='roberta-large'
elr="1e-5"           # Encoder learning rate
dlr="1e-3"           # Decoder (GNN) learning rate
bs=64                # Batch size
mbs=2                # Mini-batch size
unfreeze_epoch=2     # Epoch để unfreeze encoder
k=5                  # Số GNN layers
gnndim=200           # GNN dimension
seed=42              # Random seed
n_epochs=15          # Số epochs
max_seq_len=128      # Max sequence length
```

## Modifications Made

### 1. Files Added:
- `preprocess_utils/convert_isarcasm.py`: Converter từ HuggingFace format
- `scripts/run_train__isarcasm.sh`: Training script
- `scripts/run_eval__isarcasm.sh`: Evaluation script
- `ISARCASM_INTEGRATION.md`: Documentation này

### 2. Files Modified:
- `preprocess.py`: 
  - Added import for `convert_isarcasm`
  - Added `isarcasm` paths in `input_paths` and `output_paths`
  - Added `isarcasm` routine in preprocessing pipeline
  
- `dragon.py`:
  - Enhanced seed setting với `torch.cuda.manual_seed_all()`
  - Added cudnn deterministic settings cho reproducibility
  - Không thay đổi logic xử lý dữ liệu hoặc model

### 3. No Changes to Model Logic:
- Không thay đổi architecture của DRAGON model
- Không thay đổi training/evaluation logic
- Chỉ thêm data adapter và reproducibility improvements

## Troubleshooting

### Issue: "No module named 'datasets'"
```bash
pip install datasets
```

### Issue: ConceptNet not found
```bash
# Download và preprocess ConceptNet trước
./download_raw_data.sh
python preprocess.py --run common -p 8
```

### Issue: CUDA out of memory
```bash
# Giảm batch size trong script
bs=32  # hoặc nhỏ hơn
mbs=1
```

### Issue: Slow training with deterministic mode
```bash
# Trade-off giữa speed và reproducibility
# Sửa trong dragon.py:
torch.backends.cudnn.deterministic = False  # Nhanh hơn
torch.backends.cudnn.benchmark = True       # Tối ưu cho hardware
```

## Performance Notes

- Training time: ~2-3 hours trên 1 GPU V100 (tùy thuộc vào dataset size)
- Memory: ~10-12GB GPU memory với batch_size=64
- Deterministic mode có thể chậm hơn ~10-20% so với non-deterministic

## Citation

Nếu sử dụng code này, vui lòng cite:

```bibtex
@inproceedings{dragon2022,
  title={DRAGON: Deep Bidirectional Language-Knowledge Graph Pretraining},
  author={Yasunaga, Michihiro and Bosselut, Antoine and Ren, Hongyu and Zhang, Xikun and Manning, Christopher D and Liang, Percy and Leskovec, Jure},
  booktitle={NeurIPS},
  year={2022}
}

@dataset{isarcasm2022,
  title={iSarcasm: A Dataset of Intended Sarcasm},
  author={..},
  year={2022}
}
```
