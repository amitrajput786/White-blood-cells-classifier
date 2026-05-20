# 🚀 HUGGING FACE HUB DEPLOYMENT GUIDE
**Model:** White Blood Cell (WBC) Classification  
**Target:** Hugging Face Spaces  
**Date:** March 5, 2026

---

## 📋 **QUICK OVERVIEW**

Hugging Face Spaces provides **free cloud hosting** for your FastAPI application. Your WBC classifier will be accessible to anyone with a public link.

### **Deployment Options on Hugging Face:**

| Option | Cost | Resources | Best For |
|--------|------|-----------|----------|
| **Spaces (Free Tier)** | Free | Shared CPU, 2GB RAM, 50GB storage | Development, demos |
| **Spaces (Premium)** | $7/month | Dedicated GPU/CPU | Production workloads |
| **Model Hub** | Free | Model hosting only | Sharing trained weights |
| **Model + Space** | Free/Paid | Combined approach | Full application hosting |

---

## 🔑 **STEP 1: Create Hugging Face Account & Get API Token**

### 1.1 Create Account
```bash
# Go to: https://huggingface.co/join
# Fill in: Username, Email, Password
# Verify email
```

### 1.2 Generate API Token
```
1. Login to https://huggingface.co
2. Click Profile Icon (top-right) → Settings
3. Go to "Access Tokens"
4. Click "New Token" → Give it a name
5. Set role to "write" (needed for uploading)
6. Copy the token (looks like: hf_xxxxxxxxxxxxx)
7. Keep it safe! ⚠️
```

### 1.3 Authenticate Locally (Optional for CLI)
```bash
pip install huggingface-hub

huggingface-cli login
# Paste your token when prompted
# Or set environment variable:
export HF_TOKEN="hf_xxxxxxxxxxxxx"
```

---

## 🎯 **STEP 2: Create a Hugging Face Space**

### 2.1 Create New Space via Web UI
```
1. Go to: https://huggingface.co/spaces
2. Click "Create New Space"
3. Fill in details:
   - Owner: your-username
   - Space name: "wbc-classifier" (or your choice)
   - License: OpenRAIL (or pick one)
   - Space SDK: Docker ✅ (important!)
   - Visibility: Public ✅
   - README template: Keep default
4. Click "Create Space"
5. You'll get a Space URL like: 
   https://huggingface.co/spaces/your-username/wbc-classifier
```

### 2.2 Create Space via CLI (Alternative)
```bash
huggingface-cli repo create \
  --repo-type space \
  --space-sdk docker \
  wbc-classifier

# URL will be: https://huggingface.co/spaces/your-username/wbc-classifier
```

---

## 📦 **STEP 3: Prepare Project for Hugging Face**

### 3.1 Project Structure for Spaces

Your Space needs this structure:

```
wbc-classifier/
├── Dockerfile  ✅ Already have
├── app.py  ✅ Already have
├── main.py  ✅ Already have
├── orchestration.py  ✅ Already have
├── requirements.txt  ✅ Updated
├── Models/
│   └── at_batch_size=32.D24E.keras  ⚠️ NEEDS HANDLING
├── static/
│   └── index.html  ✅ Already have
├── uploads/  ✅ Auto-created
└── README.md  ⚠️ Create/update
```

### 3.2 Update requirements.txt for Spaces

The requirements.txt you updated is perfect. However, for Spaces you can make it stricter:

```txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
python-multipart==0.0.20
tensorflow==2.16.1
keras==3.3.3
opencv-python==4.9.0.80
matplotlib==3.9.0
pandas==2.2.0
huggingface-hub==0.21.0
Pillow==10.1.0
```

---

## 🎨 **STEP 4: Create/Update README.md for Space**

Create a professional README that Hugging Face will display:

```markdown
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
- **Classes**: 5 WBC types
  - Basophil
  - Eosinophil
  - Lymphocyte
  - Monocyte
  - Neutrophil

## 📊 Performance

- **Validation Accuracy**: ~99%
- **Training Dataset**: PBC (Peripheral Blood Cell)
- **Model Size**: 28 MB
- **Inference Time**: 1-2 seconds (CPU)

## 🚀 How to Use

### Via Web UI
1. Go to the app interface
2. Upload a white blood cell image (JPG, PNG, BMP, TIFF, GIF)
3. Get instant classification results

### Via API

#### Single Image Prediction
```bash
curl -X POST "https://your-space-url/api/predict" \
  -F "file=@wbc_image.png"
```

#### Batch Prediction
```bash
curl -X POST "https://your-space-url/api/predict_batch" \
  -F "file=@image1.png" \
  -F "file=@image2.png"
```

#### Get Model Info
```bash
curl "https://your-space-url/api/model_info"
```

#### Health Check
```bash
curl "https://your-space-url/api/health"
```

## 📚 API Documentation

- **Swagger UI**: `/docs`
- **ReDoc**: `/redoc`
- **Test Endpoint**: `/api/test`

## 🔧 Supported Formats

- **Images**: JPG, JPEG, PNG, BMP, TIFF, GIF
- **Max Size**: 10 MB per image
- **Batch Limit**: 10 images per request

## 📈 Model Results

- **Validation Accuracy**: 99%
- **ROC AUC**: High across all classes
- **Confusion Matrix**: Excellent class separation

## 🛠️ Technical Details

- **Framework**: FastAPI
- **Deep Learning**: TensorFlow 2.16+
- **Computer Vision**: OpenCV
- **Server**: Uvicorn
- **Container**: Docker

## 📝 License

OpenRAIL License - Free for research and commercial use

## 🤝 Author

Amit

## 🙏 Acknowledgments

- Hugging Face for Spaces platform
- Dataset: PBC (Peripheral Blood Cell)
- Research papers on attention mechanisms in medical imaging
```

---

## ⚠️ **STEP 5: Handle the Large Model File**

### **CRITICAL: Model File Size**

Your `at_batch_size=32.D24E.keras` (28 MB) is **LARGER than Hugging Face's git LFS limits for free tier**.

### **Solution Options:**

#### **Option A: Use Git LFS (Recommended)**

```bash
# 1. Install Git LFS
sudo apt-get install git-lfs  # Linux
# or
brew install git-lfs  # macOS

# 2. Initialize Git LFS (in your Space folder)
git lfs install

# 3. Track .keras files
git lfs track "*.keras"

# 4. Add & commit your large files
git add Models/at_batch_size=32.D24E.keras
git commit -m "Add trained WBC model"
git push

# Works! Hugging Face supports LFS free tier up to 10GB
```

#### **Option B: Download from Hugging Face Hub at Runtime**

Modify `main.py` to download the model when Space starts:

```python
# In main.py, modify get_model_path():

def get_model_path():
    """Get model path dynamically"""
    possible_paths = [
        "./Models/at_batch_size=32.D24E.keras",
        "Models/at_batch_size=32.D24E.keras",
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            logger.info(f"Found model at: {path}")
            return path
    
    # Download from Hugging Face Hub instead
    logger.warning("Model not found, downloading from Hub...")
    try:
        from huggingface_hub import hf_hub_download
        model_path = hf_hub_download(
            repo_id="your-username/wbc-classifier",
            filename="model.keras",
            repo_type="dataset"
        )
        logger.info(f"Downloaded model to: {model_path}")
        return model_path
    except Exception as e:
        logger.error(f"Failed to download: {e}")
        raise
```

#### **Option C: Split Model + Upload to Dataset**

```bash
# 1. Create a separate Dataset repo
# https://huggingface.co/datasets

# 2. Upload model there
# 3. Space downloads it on startup (Option B approach)

# 4. This keeps your Space code clean
```

### **Recommendation: Use Option A (Git LFS)**
It's simplest and Hugging Face fully supports it free.

---

## 🌐 **STEP 6: Clone and Upload to Space**

### **6.1 Clone the Space**

```bash
# Clone your new Space repo
git clone https://huggingface.co/spaces/your-username/wbc-classifier
cd wbc-classifier
```

### **6.2 Copy Your Files**

```bash
# From your local project, copy these files to the cloned Space folder:

# Copy main application files
cp /home/amit/White_blood_application/app.py .
cp /home/amit/White_blood_application/main.py .
cp /home/amit/White_blood_application/orchestration.py .
cp /home/amit/White_blood_application/requirements.txt .
cp /home/amit/White_blood_application/Dockerfile .

# Copy static files
cp -r /home/amit/White_blood_application/static .

# Copy model directory with Git LFS
mkdir -p Models
cp /home/amit/White_blood_application/Models/at_batch_size=32.D24E.keras Models/
```

### **6.3 Update requirements.txt for Spaces**

The one you have should work, but Spaces recommend simpler syntax:

```bash
# Edit requirements.txt in your Space folder
# Make sure it matches your local version
```

### **6.4 Create Dockerfile**

You already have one! Ensure it's in the Space root:

```bash
cp /home/amit/White_blood_application/Dockerfile .
```

**Update Dockerfile for Spaces:**

```dockerfile
# Use Python 3.9 slim image as base
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Create necessary directories
RUN mkdir -p Models static uploads

# Set environment variables for Hugging Face Spaces
ENV PYTHONPATH=/app
ENV PORT=7860

# Expose the port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=5 \
    CMD curl -f http://localhost:7860/api/health || exit 1

# Command to run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
```

**Key Points:**
- `PORT=7860` - Hugging Face Spaces uses port 7860
- `--host 0.0.0.0` - Listen on all interfaces
- Entry point must be `app:app` (from `app.py`)

---

## 📤 **STEP 7: Push to Hugging Face Space**

### **7.1 Set up Git LFS**

```bash
# Inside your cloned Space directory:
cd wbc-classifier

# Install Git LFS
git lfs install

# Track large files
git lfs track "*.keras"
echo "*.keras filter=lfs diff=lfs merge=lfs -text" >> .gitattributes
```

### **7.2 Commit and Push**

```bash
# Add all files
git add .

# Commit
git commit -m "Initial WBC classifier deployment

- MobileNetV2 backbone with SK + CAB blocks
- 99% validation accuracy
- FastAPI with web UI
- 5-class classification (Basophil, Eosinophil, Lymphocyte, Monocyte, Neutrophil)"

# Push to Hugging Face
git push

# This will trigger automatic Docker build on Hugging Face
# Go to your Space settings to monitor build progress
```

### **7.3 Monitor Build**

```
1. Go to: https://huggingface.co/spaces/your-username/wbc-classifier
2. Check the "Logs" tab
3. Wait for green checkmark ✅
4. Build typically takes 5-10 minutes
```

---

## 🧪 **STEP 8: Test Your Deployed Space**

Once build completes:

```bash
# Your Space URL:
# https://huggingface.co/spaces/your-username/wbc-classifier

# Test endpoints:
SPACE_URL="https://your-username-wbc-classifier.hf.space"

# 1. Health check
curl "$SPACE_URL/api/health"

# 2. API test
curl "$SPACE_URL/api/test"

# 3. Model info
curl "$SPACE_URL/api/model_info"

# 4. Single image prediction
curl -X POST "$SPACE_URL/api/predict" \
  -F "file=@your_image.png"

# 5. Open web UI
# https://your-username-wbc-classifier.hf.space
```

---

## ⚙️ **STEP 9: Configuration & Secrets (If Needed)**

### **Add Environment Variables to Space**

If you need secrets (API keys, credentials):

```
1. Go to Space settings
2. Scroll to "Repository secrets"
3. Add: Name="HF_TOKEN", Value="your-token"
4. Space automatically loads as env variables
```

Your code can access:
```python
import os
api_key = os.getenv("HF_TOKEN")
```

---

## 🔄 **STEP 10: Updates & Maintenance**

### **Update Your Space Code**

```bash
cd wbc-classifier

# Make changes locally
# Edit main.py, requirements.txt, etc.

# Commit and push
git add .
git commit -m "Update: improved error handling"
git push

# Hugging Face auto-rebuilds in ~2-5 minutes
```

### **Update Model File**

```bash
# If you trained a new model:
cp new_model.keras Models/

git add Models/at_batch_size=32.D24E.keras
git commit -m "Update: use new trained model (v2)"
git push
```

---

## 📊 **STEP 11: Monitor & Analytics**

Hugging Face Spaces provides:

- **Traffic Analytics**: View in Space settings
- **Build Logs**: See deployment status
- **Persistent Storage**: Upload directory auto-saved
- **CPU/Memory Usage**: Monitor in logs

---

## 🎯 **FULL DEPLOYMENT CHECKLIST**

- [ ] Create Hugging Face account
- [ ] Generate API token
- [ ] Create new Space (Docker SDK)
- [ ] Create/update README.md
- [ ] Install Git LFS locally
- [ ] Clone Space repository
- [ ] Copy project files
- [ ] Update Dockerfile for Spaces
- [ ] Review requirements.txt
- [ ] Track .keras file with Git LFS
- [ ] Commit changes
- [ ] Push to Space
- [ ] Monitor build (5-10 min)
- [ ] Test all API endpoints
- [ ] Share Space link publicly
- [ ] Monitor for errors

---

## 💻 **COMPLETE DEPLOYMENT COMMANDS (All at Once)**

```bash
# 1. Install Git LFS
sudo apt-get install git-lfs

# 2. Clone Space
git clone https://huggingface.co/spaces/your-username/wbc-classifier
cd wbc-classifier

# 3. Setup LFS
git lfs install
git lfs track "*.keras"
echo "*.keras filter=lfs diff=lfs merge=lfs -text" >> .gitattributes

# 4. Copy files
cp /home/amit/White_blood_application/{app.py,main.py,orchestration.py,requirements.txt,Dockerfile} .
cp -r /home/amit/White_blood_application/static .
mkdir -p Models
cp /home/amit/White_blood_application/Models/at_batch_size=32.D24E.keras Models/

# 5. Create README (see template above)
cat > README.md << 'EOF'
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
...
EOF

# 6. Commit
git add .
git commit -m "Deploy WBC classifier to Hugging Face Spaces"

# 7. Push
git push

# 8. Wait and check: https://huggingface.co/spaces/your-username/wbc-classifier
```

---

## 🚀 **FINAL RESULT**

After deployment:

✅ Your model runs on **free Hugging Face infrastructure**  
✅ Accessible via **public URL** (shareable link)  
✅ **Web UI** for easy image uploads  
✅ **REST API** for programmatic access  
✅ **Swagger docs** built-in  
✅ **Auto-scaling** - handles traffic spikes  
✅ **Free tier** works for demos & development  

---

## 🆘 **TROUBLESHOOTING**

### **Build Fails**
- Check Dockerfile syntax
- Ensure all files copied correctly
- Check requirements.txt for typos
- View logs in Space settings

### **Model Not Found**
- Verify Git LFS is installed: `git lfs --version`
- Ensure .keras file is tracked: `git lfs ls-files`
- Push with: `git lfs push --all`

### **Out of Memory**
- Upgrade to Spaces Pro ($7/month)
- Or optimize model (quantization)
- Or limit batch size

### **Slow Response**
- Normal on free tier (shared resources)
- Upgrade to GPU for faster inference
- Or Spaces Pro for better CPU

---

## 📚 **USEFUL LINKS**

- **Hugging Face Spaces:** https://huggingface.co/spaces
- **Spaces Documentation:** https://huggingface.co/docs/hub/spaces-overview
- **Spaces Docker Guide:** https://huggingface.co/docs/hub/spaces-sdks-docker
- **Git LFS Guide:** https://huggingface.co/docs/hub/security-git-ssh

---

**Ready to deploy? Start with Step 1! 🚀**
