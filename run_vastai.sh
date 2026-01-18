#!/bin/bash
# Quick setup + training script - runs everything automatically
set -e

echo "🚀 DRAGON Auto-Training on Vast.ai"
echo "===================================="
echo ""

# Activate venv
source ~/dragon/dragon/venv/bin/activate

# Navigate to project
cd ~/dragon/dragon

# Check GPU
echo "📊 GPU Info:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

# Preprocess ConceptNet if not done
if [ ! -f "data/cpnet/concept.txt" ]; then
    echo "📦 Preprocessing ConceptNet..."
    python3 preprocess.py --run common -p 8
else
    echo "✅ ConceptNet already preprocessed"
fi

# Preprocess iSarcasm if not done
if [ ! -f "data/isarcasm/graph/train.graph.adj.pk" ]; then
    echo "📦 Preprocessing iSarcasm..."
    python3 preprocess.py --run isarcasm -p 8
else
    echo "✅ iSarcasm already preprocessed"
fi

# Make script executable
chmod +x scripts/run_train__isarcasm_binary.sh

# Start training
echo ""
echo "🔥 Starting training..."
echo "===================================="
./scripts/run_train__isarcasm_binary.sh

echo ""
echo "🎉 Training completed!"
echo ""
echo "📁 Model saved to: runs/isarcasm/dragon_binary__*/model.pt"
echo "📄 Log file: logs/train__*.log.txt"
