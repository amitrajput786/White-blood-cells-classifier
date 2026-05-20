#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastAPI Web Application for WBC Classification
This is the main FastAPI application file that handles all API endpoints
and serves the frontend HTML file.

@author: amit
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import logging
from typing import List
import traceback
from pathlib import Path
import subprocess
import sys
import os
os.environ["HF_HOME"] = "/data/hf_cache"
from huggingface_hub import hf_hub_download

# Import your orchestration pipeline
from orchestration import WBCClassificationPipeline

app = FastAPI(
    title="WBC Classification API",
    description="AI-Powered White Blood Cell Classification System",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration - DYNAMIC PATHS FOR HUGGING FACE SPACES
def get_model_path():
    """Get model path dynamically based on environment"""
    possible_paths = [
        # Try different possible locations
        "./Models/at_batch_size=32.D24E.keras",
        "at_batch_size=32.D24E.keras",
        "Models/at_batch_size=32.D24E.keras",
        os.path.join(os.getcwd(), "at_batch_size=32.D24E.keras"),
        os.path.join(os.getcwd(), "Models", "at_batch_size=32.D24E.keras"),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            logger.info(f"Found model at: {path}")
            return path
    
    # If not found, try to download from Hugging Face Hub
    logger.warning("Model not found locally, trying to download from Hugging Face Hub")
    try:
        model_path = hf_hub_download(
            repo_id="adffedccasfe/WBC",
            filename="Models/at_batch_size=32.D24E.keras",
            repo_type="space"
        )
        logger.info(f"Downloaded model to: {model_path}")
        return model_path
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        return possible_paths[0]  # Return default for error handling

MODEL_PATH = get_model_path()
STATIC_DIR = "static"
UPLOADS_DIR = "uploads"
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif"}

# Create directories if they don't exist
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)

# Global pipeline instance
pipeline = None

# Mount static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")



# Startup event - Initialize the model
@app.on_event("startup")
async def startup_event():
    global pipeline
    global MODEL_PATH  # <-- Move this to the top before any use of MODEL_PATH

    logger.info("🚀 Starting WBC Classification API...")
    logger.info(f"📂 Current working directory: {os.getcwd()}")
    logger.info(f"🎯 Model path: {MODEL_PATH}")
    logger.info(f"📁 Files in current directory: {os.listdir('.')}")
    
    # List files in Models directory if it exists
    if os.path.exists("Models"):
        logger.info(f"📁 Files in Models directory: {os.listdir('Models')}")
    
    try:
        # Check if model file exists
        if not os.path.exists(MODEL_PATH):
            logger.error(f"❌ Model file not found at: {MODEL_PATH}")
            
            # Try to find any .keras files
            for root, dirs, files in os.walk("."):
                for file in files:
                    if file.endswith(".keras"):
                        logger.info(f"🔍 Found keras file: {os.path.join(root, file)}")
            
            # Try to download from Hugging Face Hub
            logger.info("🔄 Attempting to download model from Hugging Face Hub...")
            try:
                downloaded_path = hf_hub_download(
                    repo_id="adffedccasfe/WBC",
                    filename="Models/at_batch_size=32.D24E.keras",
                    repo_type="space"
                )
                logger.info(f"✅ Downloaded model to: {downloaded_path}")
                MODEL_PATH = downloaded_path  # Now this works, since global is at the top
            except Exception as download_error:
                logger.error(f"❌ Failed to download model: {download_error}")
                # Continue without model for now
                return
        
        # Initialize pipeline
        logger.info("🔧 Initializing pipeline...")
        pipeline = WBCClassificationPipeline(MODEL_PATH)
        
        # Load model
        logger.info("🔄 Loading model...")
        if pipeline.load_model():
            logger.info("✅ Model loaded successfully!")
            logger.info(f"📊 Available classes: {pipeline.class_names}")
        else:
            logger.error("❌ Failed to load model")
            pipeline = None
            
    except Exception as e:
        logger.error(f"❌ Startup failed: {str(e)}")
        logger.error(traceback.format_exc())
        pipeline = None
        # Don't exit - let the API start even without model


# Add a simple startup check endpoint
@app.get("/api/startup_check")
async def startup_check():
    """Check if the app has started properly"""
    return {
        "status": "running",
        "message": "API has started",
        "model_loaded": pipeline is not None and pipeline.is_loaded if pipeline else False
    }

# Root endpoint - Serve the main HTML page
@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main HTML page"""
    try:
        # Try multiple possible locations for the HTML file
        possible_paths = [
            os.path.join(STATIC_DIR, "index.html"),
            "index.html",
            os.path.join(".", "index.html")
        ]
        
        html_content = None
        for html_path in possible_paths:
            if os.path.exists(html_path):
                with open(html_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                logger.info(f"Serving HTML from: {html_path}")
                break
        
        if html_content:
            return HTMLResponse(content=html_content)
        else:
            # Return a basic HTML page if index.html is not found
            return HTMLResponse(
                content="""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>WBC Classification API</title>
                    <style>
                        body { font-family: Arial, sans-serif; margin: 40px; }
                        .container { max-width: 800px; margin: 0 auto; }
                        .upload-area { border: 2px dashed #ccc; padding: 40px; text-align: center; margin: 20px 0; }
                        .btn { background: #007bff; color: white; padding: 10px 20px; border: none; cursor: pointer; }
                        .results { margin-top: 20px; padding: 20px; background: #f8f9fa; }
                        .status { padding: 10px; margin: 10px 0; border-radius: 5px; }
                        .status.success { background: #d4edda; color: #155724; }
                        .status.warning { background: #fff3cd; color: #856404; }
                        .status.error { background: #f8d7da; color: #721c24; }
                    </style>
                </head>
                <body>
                    <div class="container">
                        <h1>🩸 WBC Classification API</h1>
                        <p>Upload white blood cell images for classification</p>
                        
                        <div id="status" class="status"></div>
                        
                        <div class="upload-area">
                            <input type="file" id="fileInput" accept="image/*" style="display: none;">
                            <button class="btn" onclick="document.getElementById('fileInput').click()">
                                Select Image
                            </button>
                            <p>Supported formats: JPG, PNG, BMP, TIFF, GIF</p>
                        </div>
                        
                        <div id="results" class="results" style="display: none;"></div>
                        
                        <div>
                            <h3>API Endpoints:</h3>
                            <ul>
                                <li><a href="/docs">API Documentation</a></li>
                                <li><a href="/api/health">Health Check</a></li>
                                <li><a href="/api/startup_check">Startup Check</a></li>
                                <li><a href="/api/model_info">Model Information</a></li>
                            </ul>
                        </div>
                    </div>
                    
                    <script>
                        // Check startup status
                        fetch('/api/startup_check')
                            .then(response => response.json())
                            .then(data => {
                                const statusDiv = document.getElementById('status');
                                if (data.model_loaded) {
                                    statusDiv.textContent = 'Model loaded successfully! Ready to classify images.';
                                    statusDiv.className = 'status success';
                                } else {
                                    statusDiv.textContent = 'Warning: Model not loaded. Please check logs.';
                                    statusDiv.className = 'status warning';
                                }
                            })
                            .catch(error => {
                                const statusDiv = document.getElementById('status');
                                statusDiv.textContent = 'Error checking startup status.';
                                statusDiv.className = 'status error';
                            });
                    
                        document.getElementById('fileInput').addEventListener('change', function(event) {
                            const file = event.target.files[0];
                            if (file) {
                                const formData = new FormData();
                                formData.append('file', file);
                                
                                fetch('/api/predict', {
                                    method: 'POST',
                                    body: formData
                                })
                                .then(response => response.json())
                                .then(data => {
                                    const resultsDiv = document.getElementById('results');
                                    if (data.success) {
                                        resultsDiv.innerHTML = `
                                            <h3>Classification Results:</h3>
                                            <p><strong>Predicted Class:</strong> ${data.results.predicted_class}</p>
                                            <p><strong>Confidence:</strong> ${(data.results.confidence * 100).toFixed(2)}%</p>
                                            <h4>All Probabilities:</h4>
                                            <ul>
                                                ${Object.entries(data.results.all_probabilities).map(([className, prob]) => 
                                                    `<li>${className}: ${(prob * 100).toFixed(2)}%</li>`
                                                ).join('')}
                                            </ul>
                                        `;
                                    } else {
                                        resultsDiv.innerHTML = `<p style="color: red;">Error: ${data.detail || 'Unknown error'}</p>`;
                                    }
                                    resultsDiv.style.display = 'block';
                                })
                                .catch(error => {
                                    const resultsDiv = document.getElementById('results');
                                    resultsDiv.innerHTML = `<p style="color: red;">Error: ${error.message}</p>`;
                                    resultsDiv.style.display = 'block';
                                });
                            }
                        });
                    </script>
                </body>
                </html>
                """,
                status_code=200
            )
            
    except Exception as e:
        logger.error(f"Error serving root page: {str(e)}")
        return HTMLResponse(
            content="<h1>Error loading page</h1>",
            status_code=500
        )

# Health check endpoint
@app.get("/api/health")
async def health_check():
    """Check if the API and model are working properly"""
    try:
        model_loaded = pipeline is not None and pipeline.is_loaded
        model_info = pipeline.get_model_info() if model_loaded else {}
        
        return {
            "status": "healthy" if model_loaded else "starting",
            "model_loaded": model_loaded,
            "model_info": model_info,
            "api_version": "1.0.0",
            "model_path": MODEL_PATH,
            "model_exists": os.path.exists(MODEL_PATH)
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "error",
            "model_loaded": False,
            "error": str(e)
        }

# Rest of your endpoints remain the same...
# [Include all your existing endpoints: validate_uploaded_file, predict_single_image, predict_batch_images, get_model_info, test_endpoint]

# Utility function to validate uploaded files
def validate_uploaded_file(file: UploadFile) -> tuple[bool, str]:
    """Validate uploaded file format and size"""
    
    # Check file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        return False, f"Invalid file type. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
    
    return True, "Valid file"

# Single image prediction endpoint
@app.post("/api/predict")
async def predict_single_image(file: UploadFile = File(...)):
    """
    Predict WBC class for a single uploaded image
    
    Args:
        file: Uploaded image file
        
    Returns:
        JSON response with prediction results
    """
    try:
        # Check if model is loaded
        if not pipeline or not pipeline.is_loaded:
            raise HTTPException(
                status_code=503, 
                detail="Model not loaded. Please check server logs."
            )
        
        # Validate file
        is_valid, error_msg = validate_uploaded_file(file)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg)
        
        # Read file content
        file_content = await file.read()
        
        # Validate image format
        is_valid_image, validation_error = pipeline.validate_image_format(file_content)
        if not is_valid_image:
            raise HTTPException(status_code=400, detail=validation_error)
        
        # Make prediction
        logger.info(f"Processing image: {file.filename}")
        prediction_result = pipeline.predict_single_image(file_content)
        
        return {
            "success": True,
            "filename": file.filename,
            "results": prediction_result,
            "message": "Prediction completed successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error for {file.filename}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

# Batch prediction endpoint
@app.post("/api/predict_batch")
async def predict_batch_images(images: List[UploadFile] = File(...)):
    """
    Predict WBC classes for multiple uploaded images
    """
    try:
        # Check if model is loaded
        if not pipeline or not pipeline.is_loaded:
            raise HTTPException(
                status_code=503,
                detail="Model not loaded. Please check server logs."
            )
        
        # Check limits
        if not images:
            raise HTTPException(status_code=400, detail="No images uploaded")
        
        if len(images) > 10:
            raise HTTPException(
                status_code=400,
                detail="Too many files. Maximum 10 images allowed per batch."
            )
        
        logger.info(f"Processing batch of {len(images)} images")
        
        # Process each image
        batch_results = []
        
        for idx, file in enumerate(images):
            try:
                # Validate file
                is_valid, error_msg = validate_uploaded_file(file)
                if not is_valid:
                    batch_results.append({
                        "filename": file.filename,
                        "success": False,
                        "error": error_msg
                    })
                    continue
                
                # Read file content
                file_content = await file.read()
                
                # Validate image format
                is_valid_image, validation_error = pipeline.validate_image_format(file_content)
                if not is_valid_image:
                    batch_results.append({
                        "filename": file.filename,
                        "success": False,
                        "error": validation_error
                    })
                    continue
                
                # Make prediction
                logger.info(f"Processing image {idx + 1}/{len(images)}: {file.filename}")
                prediction_result = pipeline.predict_single_image(file_content)
                
                batch_results.append({
                    "filename": file.filename,
                    "success": True,
                    "results": prediction_result
                })
                
            except Exception as e:
                logger.error(f"Error processing {file.filename}: {str(e)}")
                batch_results.append({
                    "filename": file.filename,
                    "success": False,
                    "error": str(e)
                })
        
        # Calculate success rate
        successful_predictions = sum(1 for result in batch_results if result["success"])
        success_rate = successful_predictions / len(batch_results) * 100
        
        return {
            "success": True,
            "total_images": len(images),
            "successful_predictions": successful_predictions,
            "success_rate": round(success_rate, 1),
            "batch_results": batch_results,
            "message": f"Batch processing completed. {successful_predictions}/{len(images)} images processed successfully."
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

# Model information endpoint
@app.get("/api/model_info")
async def get_model_info():
    """Get information about the loaded model"""
    try:
        if not pipeline:
            raise HTTPException(status_code=503, detail="Model not initialized")
        
        model_info = pipeline.get_model_info()
        return {
            "success": True,
            "model_info": model_info
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving model information: {str(e)}"
        )

# Test endpoint
@app.get("/api/test")
async def test_endpoint():
    """Simple test endpoint"""
    return {
        "message": "WBC Classification API is working!",
        "status": "ok",
        "model_loaded": pipeline is not None and pipeline.is_loaded if pipeline else False,
        "model_path": MODEL_PATH,
        "model_exists": os.path.exists(MODEL_PATH),
        "endpoints": [
            "/ - Main page",
            "/api/health - Health check",
            "/api/startup_check - Startup check",
            "/api/predict - Single image prediction",
            "/api/predict_batch - Batch image prediction",
            "/api/model_info - Model information",
            "/docs - API documentation"
        ]
    }