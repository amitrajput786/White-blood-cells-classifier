# ⚡ HUGGING FACE DEPLOYMENT - QUICK REFERENCE

## 🚀 **TL;DR - 5 Minute Deployment**

```bash
# 1. Create Space at: https://huggingface.co/spaces
#    Settings: Docker SDK, Public visibility

# 2. Clone Space
git clone https://huggingface.co/spaces/USERNAME/wbc-classifier
cd wbc-classifier

# 3. Setup Git LFS
git lfs install --local
git lfs track "*.keras"

# 4. Copy your project
cp /path/to/your/project/{app.py,main.py,orchestration.py,requirements.txt,Dockerfile} .
cp -r /path/to/your/project/static .
mkdir -p Models && cp /path/to/your/project/Models/*.keras Models/

# 5. Commit & Push
git add .
git commit -m "Deploy WBC Classifier"
git push

# 6. Wait for build ✅ (5-10 minutes)
# 7. Share your Space link! 🎉
```

---

## 📋 **Pre-Deployment Checklist**

- [ ] Hugging Face account created
- [ ] API token generated (from settings)
- [ ] Git LFS installed (`git lfs --version`)
- [ ] Space created (Docker SDK)
- [ ] Model file exists (28 MB)
- [ ] All code files ready
- [ ] Dockerfile is correct
- [ ] requirements.txt updated

---

## 🎯 **Key Configuration Points**

### **Dockerfile - CRITICAL for Spaces**

```dockerfile
# MUST have these:
FROM python:3.9-slim
WORKDIR /app
ENV PORT=7860
EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
```

### **README.md - MUST have**

```markdown
---
title: WBC Classifier
emoji: 🩸
colorFrom: blue
colorTo: pink
sdk: docker
app_port: 7860
---
```

### **requirements.txt - Latest versions**

```txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
tensorflow==2.16.1
keras==3.3.3
opencv-python==4.9.0.80
# ... etc
```

---

## 🔗 **Important URLs**

| Task | URL |
|------|-----|
| Create Space | https://huggingface.co/spaces |
| Generate Token | https://huggingface.co/settings/tokens |
| Your Spaces | https://huggingface.co/user/spaces |
| Space Settings | https://huggingface.co/spaces/USERNAME/SPACE_NAME/settings |
| View Logs | https://huggingface.co/spaces/USERNAME/SPACE_NAME/logs |

---

## 📊 **File Structure for Space**

```
wbc-classifier/
├── app.py                    ← Entry point
├── main.py                   ← FastAPI routes
├── orchestration.py          ← Model pipeline
├── requirements.txt          ← Dependencies
├── Dockerfile                ← Build config
├── README.md                 ← Space metadata
├── Models/
│   └── at_batch_size=32.D24E.keras  ← Large file (Git LFS)
├── static/
│   └── index.html
└── uploads/                  ← Auto-created
```

---

## ⚙️ **Environment Variables (if needed)**

Set in Space → Settings → Repository secrets

```
HF_TOKEN = your-hf-token
LOG_LEVEL = INFO
```

Access in code:
```python
import os
token = os.getenv("HF_TOKEN")
```

---

## 🧪 **Test Commands**

After deployment:

```bash
SPACE_URL="https://username-wbc-classifier.hf.space"

# Test API
curl $SPACE_URL/api/test

# Health check
curl $SPACE_URL/api/health

# Single prediction
curl -X POST "$SPACE_URL/api/predict" \
  -F "file=@image.png"

# Batch prediction
curl -X POST "$SPACE_URL/api/predict_batch" \
  -F "file=@img1.png" \
  -F "file=@img2.png"

# Model info
curl $SPACE_URL/api/model_info
```

---

## ⚠️ **Common Issues**

| Issue | Solution |
|-------|----------|
| **Build fails** | Check Dockerfile syntax, ensure all files copied |
| **Git LFS not working** | Install: `sudo apt-get install git-lfs` |
| **Model not found** | Ensure `.keras` file tracked with Git LFS |
| **Port conflict** | Must use port 7860 for Spaces |
| **Out of memory** | Upgrade to Spaces Pro ($7/month) |
| **Slow response** | Normal on free tier, upgrade for GPU |

---

## 🔐 **Security Best Practices**

1. **Never commit API tokens** - Use repository secrets
2. **Keep Space public** - But validate inputs
3. **Set file size limits** - Max 10MB per image
4. **Rate limiting** - Consider adding for production
5. **Validate uploads** - Check file extensions & format

---

## 📈 **After Deployment**

### **Monitor Performance**
- View logs: Space → Settings → Logs tab
- Check traffic: Space → Settings → Analytics
- Monitor uptime: 99.9% on Hugging Face

### **Update Model**
```bash
# Update locally
cp new_model.keras Models/

# Push update
git add Models/
git commit -m "Update: new trained model"
git push
```

### **Scale Up**
- **Free tier**: Shared CPU, 2GB RAM
- **Spaces Pro**: Dedicated resources ($7/month)
- **Spaces ZeroGPU**: Free GPU access (limited)
- **API Endpoints**: Faster inference with API plan

---

## 🔄 **Automated Deployment Script**

```bash
chmod +x deploy_to_huggingface.sh
./deploy_to_huggingface.sh

# Prompts for:
# - HF username
# - API token
# - Space name

# Then fully automates deployment!
```

---

## 📞 **Getting Help**

- **HF Spaces Docs**: https://huggingface.co/docs/hub/spaces
- **Docker Guide**: https://huggingface.co/docs/hub/spaces-sdks-docker
- **Git LFS Help**: https://huggingface.co/docs/hub/security-git-ssh
- **Community**: https://discuss.huggingface.co/c/spaces/

---

## 🎯 **Success Indicators**

✅ Space shows green checkmark  
✅ Web UI loads at main URL  
✅ `/docs` shows Swagger API  
✅ `/api/health` returns status  
✅ Can upload and classify images  
✅ Logs show no errors  

---

## 💡 **Pro Tips**

1. **Use `.gitignore`** to exclude unnecessary files:
   ```
   __pycache__/
   *.pyc
   .DS_Store
   uploads/*
   ```

2. **Optimize model loading** for faster startup:
   ```python
   # Load once at startup, reuse for all requests
   @app.on_event("startup")
   async def load_model():
       global pipeline
       pipeline = WBCClassificationPipeline(MODEL_PATH)
   ```

3. **Cache predictions** if same image uploaded multiple times

4. **Monitor logs frequently** during first week

5. **Share Space URL** to get community feedback

---

## 🚀 **You're Ready!**

Your WBC Classifier will be live on:
```
https://huggingface.co/spaces/your-username/wbc-classifier
```

Public, free, and shareable! 🎉

---

**Last updated:** March 5, 2026
**Status:** Ready for deployment
**Support:** Full documentation in HUGGINGFACE_DEPLOYMENT_GUIDE.md
