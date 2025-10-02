#!/bin/bash
#
# Download Pretrained VitFly Models
#
# This script helps download pretrained models from the original VitFly project
# Source: https://upenn.app.box.com/v/ViT-quad-datashare
#

set -e

echo "======================================"
echo "VitFly Pretrained Models Download"
echo "======================================"
echo ""

# Define paths
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="${PROJECT_ROOT}/models"
PRETRAINED_DIR="${MODELS_DIR}/pretrained"

echo "Project root: ${PROJECT_ROOT}"
echo "Models directory: ${MODELS_DIR}"
echo ""

# Create directories
mkdir -p "${PRETRAINED_DIR}"

echo "Available pretrained models from VitFly (ICRA 2025):"
echo ""
echo "  1. ViT+LSTM (Best performance) - vitlstm_best.pth"
echo "  2. ViT - vit_best.pth"
echo "  3. ConvNet - convnet_best.pth"
echo "  4. LSTMNet - lstmnet_best.pth"
echo "  5. UNet - unet_best.pth"
echo ""
echo "======================================"
echo "DOWNLOAD INSTRUCTIONS:"
echo "======================================"
echo ""
echo "1. Visit: https://upenn.app.box.com/v/ViT-quad-datashare"
echo "2. Download: pretrained_models.tar (50MB)"
echo "3. Save to: /tmp/pretrained_models.tar"
echo "4. Run this script again to extract"
echo ""

# Check if tar file exists
if [ -f "/tmp/pretrained_models.tar" ]; then
    echo "Found pretrained_models.tar in /tmp/"
    echo "Extracting models..."

    tar -xvf /tmp/pretrained_models.tar -C "${MODELS_DIR}"

    echo ""
    echo "✓ Models extracted successfully!"
    echo ""
    echo "Available models:"
    find "${MODELS_DIR}" -name "*.pth" -o -name "*.pt"
    echo ""
    echo "======================================"
    echo "NEXT STEPS:"
    echo "======================================"
    echo ""
    echo "To use a pretrained model in simulation:"
    echo ""
    echo "  python scripts/simulate.py \\"
    echo "    --config config/simulation_config.yaml \\"
    echo "    --mode model \\"
    echo "    --model-path models/vitlstm_best.pth \\"
    echo "    --model-type ViTLSTM"
    echo ""
else
    echo "pretrained_models.tar not found in /tmp/"
    echo ""
    echo "Please download it first from:"
    echo "  https://upenn.app.box.com/v/ViT-quad-datashare"
    echo ""
    echo "Then save it to /tmp/pretrained_models.tar and run this script again."
    echo ""
    exit 1
fi
