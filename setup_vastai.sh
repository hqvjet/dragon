#!/bin/bash
set -e  # Exit on error

echo "=================================================="
echo "🚀 DRAGON Setup Script for Vast.ai"
echo "=================================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_step() {
    echo -e "${GREEN}[STEP]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Step 0: Check GPU
print_step "Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo ""
else
    print_warning "nvidia-smi not found. Continuing anyway..."
fi

# Step 1: Update system
print_step "Updating system packages..."
sudo apt-get update -qq
sudo apt-get install -y wget curl git build-essential

# Step 2: Setup Python environment
print_step "Setting up Python 3.10..."
if ! command -v python3.10 &> /dev/null; then
    sudo apt-get install -y python3.10 python3.10-venv python3.10-dev
fi

# Step 3: Setup project directory
print_step "Setting up project directory..."
WORK_DIR="$HOME/dragon"
mkdir -p $WORK_DIR
cd $WORK_DIR

# Step 4: Clone or sync project (if not exists)
if [ ! -d "dragon" ]; then
    print_step "Cloning DRAGON repository..."
    # Option 1: If you have a git repo
    # git clone https://github.com/your-username/dragon.git
    # cd dragon
    
    # Option 2: Create structure manually
    mkdir -p dragon
    cd dragon
    mkdir -p data models logs runs
    mkdir -p modeling preprocess_utils scripts utils
    
    print_warning "Project structure created. You'll need to upload your code."
    print_warning "Use: rsync -avz -e ssh /local/path/dragon/ vastai:/root/dragon/dragon/"
else
    print_step "Project directory exists. Syncing..."
    cd dragon
fi

# Step 5: Create virtual environment
print_step "Creating Python virtual environment..."
python3.10 -m venv venv
source venv/bin/activate

# Step 6: Upgrade pip
print_step "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Step 7: Install PyTorch (CUDA 11.8)
print_step "Installing PyTorch with CUDA 11.8..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Step 8: Install PyTorch Geometric
print_step "Installing PyTorch Geometric..."
pip install torch-geometric
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

# Step 9: Install Transformers and other dependencies
print_step "Installing Transformers and dependencies..."
pip install transformers datasets
pip install spacy nltk tqdm numpy scipy scikit-learn
pip install pandas matplotlib seaborn

# Step 10: Download spaCy model
print_step "Downloading spaCy English model..."
python -m spacy download en_core_web_sm

# Step 11: Install optional dependencies
print_step "Installing optional dependencies..."
pip install wandb huggingface_hub

# Step 12: Download pre-trained DRAGON model
print_step "Downloading pre-trained DRAGON model..."
mkdir -p models
cd models
if [ ! -f "general_model.pt" ]; then
    wget -c https://nlp.stanford.edu/projects/myasu/DRAGON/models/general_model.pt
    echo "Model size: $(du -h general_model.pt | cut -f1)"
else
    print_warning "general_model.pt already exists. Skipping download."
fi
cd ..

# Step 13: Download ConceptNet
print_step "Downloading ConceptNet assertions..."
mkdir -p data/cpnet
cd data/cpnet
if [ ! -f "conceptnet-assertions-5.6.0.csv" ]; then
    if [ ! -f "conceptnet-assertions-5.6.0.csv.gz" ]; then
        wget -c https://s3.amazonaws.com/conceptnet/downloads/2018/edges/conceptnet-assertions-5.6.0.csv.gz
    fi
    print_step "Extracting ConceptNet..."
    gunzip -k conceptnet-assertions-5.6.0.csv.gz
    echo "ConceptNet size: $(du -h conceptnet-assertions-5.6.0.csv | cut -f1)"
else
    print_warning "ConceptNet already exists. Skipping download."
fi
cd ../..

# Step 14: Verify installation
print_step "Verifying installation..."
python -c "
import torch
import transformers
import torch_geometric
print(f'✅ PyTorch: {torch.__version__}')
print(f'✅ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✅ CUDA version: {torch.version.cuda}')
    print(f'✅ GPU: {torch.cuda.get_device_name(0)}')
    print(f'✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
print(f'✅ Transformers: {transformers.__version__}')
print(f'✅ PyTorch Geometric: {torch_geometric.__version__}')
"

# Step 15: Check if code files exist
print_step "Checking project files..."
REQUIRED_FILES=(
    "dragon_binary.py"
    "preprocess.py"
    "modeling/modeling_dragon_binary.py"
    "modeling/modeling_dragon.py"
    "scripts/run_train__isarcasm_binary.sh"
)

MISSING_FILES=()
for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        MISSING_FILES+=("$file")
    fi
done

if [ ${#MISSING_FILES[@]} -gt 0 ]; then
    print_error "Missing files detected!"
    echo "You need to upload these files:"
    for file in "${MISSING_FILES[@]}"; do
        echo "  - $file"
    done
    echo ""
    echo "From your local machine, run:"
    echo "  rsync -avz -e 'ssh -p PORT' /path/to/dragon/ root@VAST_IP:/root/dragon/dragon/"
    echo ""
else
    print_step "All required files present!"
fi

echo ""
echo "=================================================="
echo "✅ Setup Complete!"
echo "=================================================="
echo ""
echo "Next steps:"
echo ""
echo "1. If files are missing, upload them:"
echo "   rsync -avz -e 'ssh -p PORT' /local/dragon/ root@VAST_IP:/root/dragon/dragon/"
echo ""
echo "2. Preprocess ConceptNet (takes ~5 minutes):"
echo "   cd ~/dragon/dragon"
echo "   source venv/bin/activate"
echo "   python3 preprocess.py --run common -p 8"
echo ""
echo "3. Preprocess iSarcasm dataset (takes ~2 minutes):"
echo "   python3 preprocess.py --run isarcasm -p 8"
echo ""
echo "4. Start training:"
echo "   chmod +x scripts/run_train__isarcasm_binary.sh"
echo "   ./scripts/run_train__isarcasm_binary.sh"
echo ""
echo "5. Monitor training:"
echo "   tail -f logs/train__*.log.txt"
echo ""
echo "=================================================="
