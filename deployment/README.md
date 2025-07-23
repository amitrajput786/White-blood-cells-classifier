---
title: WBC Classification
emoji: 🩸
colorFrom: red
colorTo: blue
sdk: docker
pinned: false
license: mit
app_port: 7860
---

# WBC Classification System

An AI-powered White Blood Cell (WBC) classification system that can identify different types of white blood cells from microscopic images.

## Features

- **Real-time Classification**: Upload images and get instant predictions
- **Batch Processing**: Process multiple images simultaneously
- **High Accuracy**: Deep learning model trained on medical imagery
- **User-friendly Interface**: Simple web interface for easy interaction

## Supported Cell Types

- Basophil
- Eosinophil
- Lymphocyte
- Monocyte
- Neutrophil

## Usage

1. **Single Image**: Upload a single microscopic image for classification
2. **Batch Processing**: Upload multiple images for simultaneous processing
3. **View Results**: Get detailed predictions with confidence scores

## API Endpoints

- `GET /` - Main web interface
- `POST /api/predict` - Single image prediction
- `POST /api/predict_batch` - Batch image prediction
- `GET /api/health` - Health check
- `GET /api/model_info` - Model information

## Technical Details

- **Framework**: FastAPI with TensorFlow
- **Model**: Custom CNN architecture
- **Input**: 128x128 RGB images
- **Output**: Classification with confidence scores

## Development

Built for medical image analysis and research purposes. The model is trained on high-quality microscopic images of white blood cells.

## License

MIT License - See LICENSE file for details.