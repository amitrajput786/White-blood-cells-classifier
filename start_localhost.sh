#!/bin/bash
# WBC Classification - Localhost Quick Start Script
# This script prepares and runs the FastAPI server for local testing

set -e  # Exit on error

echo "=========================================="
echo "🩸 WBC Classification - Localhost Setup"
echo "=========================================="
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo -e "${BLUE}📂 Working Directory:${NC} $(pwd)"
echo ""

# 1. Check Python version
echo -e "${BLUE}🔍 Checking Python environment...${NC}"
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "   Python Version: $PYTHON_VERSION"
echo ""

# 2. Check if model exists
echo -e "${BLUE}🔍 Verifying model file...${NC}"
if [ -f "Models/at_batch_size=32.D24E.keras" ]; then
    SIZE=$(ls -lh Models/at_batch_size=32.D24E.keras | awk '{print $5}')
    echo -e "   ${GREEN}✅ Model found:${NC} $SIZE"
else
    echo -e "   ${RED}❌ ERROR: Model not found at Models/at_batch_size=32.D24E.keras${NC}"
    exit 1
fi
echo ""

# 3. Check required directories
echo -e "${BLUE}🔍 Checking required directories...${NC}"
mkdir -p static uploads
echo -e "   ${GREEN}✅${NC} static/ directory ready"
echo -e "   ${GREEN}✅${NC} uploads/ directory ready"
echo ""

# 4. Install/Update dependencies
echo -e "${BLUE}📦 Installing Python dependencies...${NC}"
echo "   This may take a few minutes on first run..."
pip install -r requirements.txt --upgrade > /dev/null 2>&1 || {
    echo -e "   ${RED}❌ Failed to install dependencies${NC}"
    exit 1
}
echo -e "   ${GREEN}✅ Dependencies installed${NC}"
echo ""

# 5. Verify imports
echo -e "${BLUE}🔍 Verifying dependencies...${NC}"
python3 << EOF
try:
    import fastapi
    import tensorflow
    import keras
    import cv2
    from PIL import Image
    print("   ✅ All dependencies verified")
except ImportError as e:
    print(f"   ❌ Import error: {e}")
    exit(1)
EOF
echo ""

# 6. Display startup information
echo -e "${GREEN}=========================================="
echo "✅ Setup Complete! Ready to Start Server"
echo "==========================================${NC}"
echo ""
echo -e "${BLUE}To start the FastAPI server, run:${NC}"
echo ""
echo "  ${YELLOW}uvicorn main:app --host 0.0.0.0 --port 8000 --reload${NC}"
echo ""
echo -e "${BLUE}Or use this docker command:${NC}"
echo ""
echo "  ${YELLOW}docker build -t wbc-classifier:latest .${NC}"
echo "  ${YELLOW}docker run -p 8000:7860 -v \$(pwd)/Models:/app/Models:ro wbc-classifier:latest${NC}"
echo ""
echo -e "${BLUE}Once running, access:${NC}"
echo ""
echo "  🌐 Web UI:          ${GREEN}http://localhost:8000${NC}"
echo "  📚 API Docs:        ${GREEN}http://localhost:8000/docs${NC}"
echo "  ❤️  Health Check:    ${GREEN}http://localhost:8000/api/health${NC}"
echo ""
echo -e "${BLUE}Model Info:${NC}"
echo "  📦 File: Models/at_batch_size=32.D24E.keras ($SIZE)"
echo "  🎯 Classes: Basophil, Eosinophil, Lymphocyte, Monocyte, Neutrophil"
echo "  📐 Input Size: 128×128×3 pixels"
echo "  🎯 Accuracy: ~99% (on PBC dataset)"
echo ""
