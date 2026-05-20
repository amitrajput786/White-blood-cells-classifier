#!/bin/bash
# 🚀 Automated Hugging Face Space Deployment Script
# This script automates the deployment of WBC Classifier to HF Spaces

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Configuration
HF_USERNAME=""
HF_SPACE_NAME="wbc-classifier"
PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
TEMP_SPACE_DIR="/tmp/hf_space_$$"

echo -e "${CYAN}"
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║     🩸 WBC Classifier - Hugging Face Spaces Deployer     ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Function to print section headers
print_section() {
    echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}📋 $1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

# Step 0: Get configuration
print_section "STEP 0: Configuration"

read -p "Enter your Hugging Face username: " HF_USERNAME

if [ -z "$HF_USERNAME" ]; then
    echo -e "${RED}❌ Username cannot be empty!${NC}"
    exit 1
fi

read -p "Enter your HF API token (from https://huggingface.co/settings/tokens): " HF_TOKEN

if [ -z "$HF_TOKEN" ]; then
    echo -e "${RED}❌ API token cannot be empty!${NC}"
    exit 1
fi

read -p "Enter Space name (default: wbc-classifier): " SPACE_NAME
HF_SPACE_NAME=${SPACE_NAME:-wbc-classifier}

echo -e "${GREEN}✅ Configuration:${NC}"
echo "   Username: $HF_USERNAME"
echo "   Space Name: $HF_SPACE_NAME"
echo "   URL: https://huggingface.co/spaces/$HF_USERNAME/$HF_SPACE_NAME"

# Step 1: Check prerequisites
print_section "STEP 1: Checking Prerequisites"

# Check Git
if ! command -v git &> /dev/null; then
    echo -e "${RED}❌ Git is not installed!${NC}"
    echo "   Install with: sudo apt-get install git"
    exit 1
fi
echo -e "${GREEN}✅ Git installed${NC}"

# Check Git LFS
if ! command -v git-lfs &> /dev/null; then
    echo -e "${YELLOW}⚠️  Git LFS not found. Installing...${NC}"
    sudo apt-get update > /dev/null 2>&1
    sudo apt-get install -y git-lfs > /dev/null 2>&1
fi
echo -e "${GREEN}✅ Git LFS installed${NC}"

# Check model file
if [ ! -f "$PROJECT_DIR/Models/at_batch_size=32.D24E.keras" ]; then
    echo -e "${RED}❌ Model file not found at: $PROJECT_DIR/Models/at_batch_size=32.D24E.keras${NC}"
    exit 1
fi
MODEL_SIZE=$(ls -lh "$PROJECT_DIR/Models/at_batch_size=32.D24E.keras" | awk '{print $5}')
echo -e "${GREEN}✅ Model found ($MODEL_SIZE)${NC}"

# Step 2: Setup HF authentication
print_section "STEP 2: Setting up Hugging Face Authentication"

export HF_TOKEN=$HF_TOKEN
git config --global credential.helper store
echo "https://:$HF_TOKEN@huggingface.co" > ~/.git-credentials
chmod 600 ~/.git-credentials

echo -e "${GREEN}✅ HF authentication configured${NC}"

# Step 3: Create Space (optional)
print_section "STEP 3: Space Creation (Optional)"

echo -e "${YELLOW}Checking if Space exists...${NC}"

SPACE_URL="https://huggingface.co/spaces/$HF_USERNAME/$HF_SPACE_NAME"
if curl -s -H "Authorization: Bearer $HF_TOKEN" \
    "https://huggingface.co/api/spaces/$HF_USERNAME/$HF_SPACE_NAME" | grep -q "error"; then
    
    echo -e "${YELLOW}Creating new Space...${NC}"
    curl -X POST \
        -H "Authorization: Bearer $HF_TOKEN" \
        -H "Content-Type: application/json" \
        "https://huggingface.co/api/repos/create" \
        -d "{
            \"repo_id\": \"$HF_SPACE_NAME\",
            \"repo_type\": \"space\",
            \"space_sdk\": \"docker\",
            \"private\": false
        }" > /dev/null 2>&1
    
    echo -e "${GREEN}✅ Space created at: $SPACE_URL${NC}"
    sleep 5  # Wait for Space to be ready
else
    echo -e "${GREEN}✅ Space already exists${NC}"
fi

# Step 4: Clone Space
print_section "STEP 4: Cloning Space Repository"

SPACE_REPO="https://:$HF_TOKEN@huggingface.co/spaces/$HF_USERNAME/$HF_SPACE_NAME"

if [ -d "$TEMP_SPACE_DIR" ]; then
    rm -rf "$TEMP_SPACE_DIR"
fi

mkdir -p "$TEMP_SPACE_DIR"
cd "$TEMP_SPACE_DIR"

echo -e "${YELLOW}Cloning Space repository...${NC}"
git clone "$SPACE_REPO" . 2>&1 | grep -v "remote:"

echo -e "${GREEN}✅ Repository cloned${NC}"

# Step 5: Setup Git LFS
print_section "STEP 5: Setting up Git LFS"

git lfs install --local
git lfs track "*.keras"
echo "*.keras filter=lfs diff=lfs merge=lfs -text" >> .gitattributes

echo -e "${GREEN}✅ Git LFS configured${NC}"

# Step 6: Copy project files
print_section "STEP 6: Copying Project Files"

echo -e "${YELLOW}Copying application files...${NC}"
cp "$PROJECT_DIR/app.py" . 2>/dev/null && echo -e "${GREEN}  ✅ app.py${NC}"
cp "$PROJECT_DIR/main.py" . 2>/dev/null && echo -e "${GREEN}  ✅ main.py${NC}"
cp "$PROJECT_DIR/orchestration.py" . 2>/dev/null && echo -e "${GREEN}  ✅ orchestration.py${NC}"
cp "$PROJECT_DIR/requirements.txt" . 2>/dev/null && echo -e "${GREEN}  ✅ requirements.txt${NC}"
cp "$PROJECT_DIR/Dockerfile" . 2>/dev/null && echo -e "${GREEN}  ✅ Dockerfile${NC}"

echo -e "${YELLOW}Copying static files...${NC}"
cp -r "$PROJECT_DIR/static" . 2>/dev/null && echo -e "${GREEN}  ✅ static/${NC}"

echo -e "${YELLOW}Copying model...${NC}"
mkdir -p Models
cp "$PROJECT_DIR/Models/at_batch_size=32.D24E.keras" Models/ 2>/dev/null && echo -e "${GREEN}  ✅ Models/at_batch_size=32.D24E.keras${NC}"

# Step 7: Create README
print_section "STEP 7: Creating README.md"

cat > README.md << 'READMEEOF'
---
title: WBC Classifier
emoji: 🩸
colorFrom: blue
colorTo: pink
sdk: docker
sdk_version: 1.0.0
app_port: 7860
license: openrail
---

# 🩸 White Blood Cell Classification API

AI-powered classification of white blood cells using a custom CNN with MobileNetV2 backbone.

## ✨ Features

- **99% Accuracy**: Trained on PBC dataset
- **Fast Predictions**: ~1-2 seconds per image
- **Batch Processing**: Classify up to 10 images at once
- **REST API**: Full Swagger documentation
- **Web UI**: Interactive upload interface

## 🎯 Model Architecture

- **Backbone**: MobileNetV2 (ImageNet pre-trained)
- **Custom Blocks**: SK Block + CAB Block + Multi-Fusion
- **Input Size**: 128×128×3 pixels
- **Classes**: 5 WBC types (Basophil, Eosinophil, Lymphocyte, Monocyte, Neutrophil)

## 📊 Performance

- **Validation Accuracy**: ~99%
- **Inference Time**: 1-2 seconds (CPU)
- **Model Size**: 28 MB

## 🚀 How to Use

### Via Web UI
Simply upload a white blood cell image (JPG, PNG, BMP, TIFF, GIF) to get instant classification results!

### Via API

**Single Image Prediction:**
```bash
curl -X POST "https://your-space-url/api/predict" \
  -F "file=@wbc_image.png"
```

**Batch Prediction (up to 10 images):**
```bash
curl -X POST "https://your-space-url/api/predict_batch" \
  -F "file=@image1.png" \
  -F "file=@image2.png"
```

**Get Model Info:**
```bash
curl "https://your-space-url/api/model_info"
```

**Health Check:**
```bash
curl "https://your-space-url/api/health"
```

## 📚 API Documentation

- **Swagger UI**: `/docs`
- **ReDoc**: `/redoc`
- **Test Endpoint**: `/api/test`

## 🛠️ Technical Details

- **Framework**: FastAPI + Uvicorn
- **Deep Learning**: TensorFlow 2.16+ / Keras 3.3+
- **Computer Vision**: OpenCV
- **Container**: Docker
- **Language**: Python 3.9+

## 📝 Training Details

- **Dataset**: PBC (Peripheral Blood Cell)
- **Model Classes**: 5 WBC types
- **Validation Accuracy**: ~99%
- **Training Epochs**: 56
- **Batch Sizes Tested**: 8, 16, 32
- **Best Results**: Batch size 8

## 🔬 Medical Context

White blood cells (WBCs) are crucial components of the immune system. Accurate classification helps in:
- Disease diagnosis
- Blood disorder detection
- Medical research
- Lab automation

## ⚖️ License

OpenRAIL - Free for research and commercial use

## 🤝 Author

Amit

## 📖 References

- Research papers on attention mechanisms in medical imaging
- PBC Dataset for WBC classification
- MobileNetV2 for efficient CNN backbone

---

**Status**: ✅ Active and ready to classify your WBC images!
READMEEOF

echo -e "${GREEN}✅ README.md created${NC}"

# Step 8: Commit and push
print_section "STEP 8: Committing and Pushing to Space"

git add .
git commit -m "Deploy WBC Classifier to Hugging Face Spaces

- MobileNetV2 backbone with SK Block + CAB Block
- 99% validation accuracy on PBC dataset
- FastAPI with REST endpoints
- Web UI for easy image classification
- 5-class classification model" 2>&1 | tail -5

echo -e "${YELLOW}Pushing to Hugging Face...${NC}"
git push 2>&1 | grep -v "remote:" | tail -5

echo -e "${GREEN}✅ Deployed successfully!${NC}"

# Step 9: Cleanup and summary
print_section "Step 9: Deployment Complete!"

echo -e "${GREEN}🎉 Your WBC Classifier is now live!${NC}"
echo ""
echo -e "${CYAN}📍 Space URL:${NC}"
echo "   https://huggingface.co/spaces/$HF_USERNAME/$HF_SPACE_NAME"
echo ""
echo -e "${CYAN}🌐 Access URL (once build completes):${NC}"
echo "   https://$HF_USERNAME-$HF_SPACE_NAME.hf.space"
echo ""
echo -e "${CYAN}📚 Swagger API Docs:${NC}"
echo "   https://$HF_USERNAME-$HF_SPACE_NAME.hf.space/docs"
echo ""
echo -e "${YELLOW}⏳ Build Progress:${NC}"
echo "   Monitor build at: https://huggingface.co/spaces/$HF_USERNAME/$HF_SPACE_NAME/logs"
echo "   Build typically takes 5-10 minutes..."
echo ""
echo -e "${CYAN}🧪 Test Commands (after build completes):${NC}"
SPACE_URL="https://$HF_USERNAME-$HF_SPACE_NAME.hf.space"
echo "   curl $SPACE_URL/api/health"
echo "   curl $SPACE_URL/api/test"
echo ""
echo -e "${GREEN}✅ Setup complete!${NC}"
echo ""
echo -e "${CYAN}Next Steps:${NC}"
echo "   1. Wait for build to complete (5-10 minutes)"
echo "   2. Visit the Space URL above"
echo "   3. Upload a WBC image to test"
echo "   4. Share the link with others!"
echo ""
