#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 12:55:13 2025

@author: amit
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Orchestration Pipeline for WBC Classification
This file handles model loading, image preprocessing, and prediction orchestration

@author: amit
"""

import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
import io
import logging
from typing import List, Dict, Tuple, Optional
import os
import glob
from pathlib import Path

from tensorflow.keras.models import load_model
import tensorflow as tf
import pandas as pd

# Define the custom functions
def split_attention(x):
    return tf.split(x, num_or_size_splits=2, axis=1)

def split_attention_output_shape(input_shape):
    filters = input_shape[-1] // 2
    return [input_shape[:-1] + (filters,), input_shape[:-1] + (filters,)]

# Define custom objects
custom_objects = {
    'split_attention': split_attention,
    'split_attention_output_shape': split_attention_output_shape
}

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WBCClassificationPipeline:
    """
    Complete pipeline for WBC classification including model loading,
    image preprocessing, and prediction generation
    """
    
    def __init__(self, model_path: str):
        """
        Initialize the pipeline with model path
        
        Args:
            model_path (str): Path to the trained .keras model file
        """
        self.model_path = model_path
        self.model = None
        self.class_names = ['Basophil', 'Eosinophil', 'Lymphocyte', 'Monocyte', 'Neutrophil']
        self.input_shape = (128, 128, 3)  # Updated to correct input shape (was 112, now 128)
        self.is_loaded = False
        
    def load_model(self) -> bool:
        """
        Load the trained model with custom objects
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            logger.info(f"Loading model from: {self.model_path}")
            
            # Define custom functions for your model
            def split_attention(x):
                return tf.split(x, num_or_size_splits=2, axis=1)
            
            def split_attention_output_shape(input_shape):
                filters = input_shape[-1] // 2
                return [input_shape[:-1] + (filters,), input_shape[:-1] + (filters,)]
            
            # Define custom objects dictionary
            custom_objects = {
                'split_attention': split_attention,
                'split_attention_output_shape': split_attention_output_shape
            }
            
            # Load the model with custom objects
            self.model = load_model(self.model_path, custom_objects=custom_objects)
            self.is_loaded = True
            
            logger.info("Model loaded successfully!")
            logger.info(f"Model input shape: {self.model.input_shape}")
            logger.info(f"Model output shape: {self.model.output_shape}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            self.is_loaded = False
            return False
    
    def preprocess_image(self, image_data: bytes) -> Optional[np.ndarray]:
        """
        Preprocess uploaded image for model prediction
        
        Args:
            image_data (bytes): Raw image data from file upload
            
        Returns:
            np.ndarray: Preprocessed image array ready for prediction
        """
        try:
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_data))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Convert PIL to numpy array
            image_array = np.array(image)
            
            # Resize image to model input shape (128x128) - UPDATED FROM 112x112
            resized_image = cv2.resize(image_array, (self.input_shape[0], self.input_shape[1]))
            
            # Normalize pixel values to [0, 1]
            normalized_image = resized_image.astype(np.float32) / 255.0
            
            # Add batch dimension
            preprocessed_image = np.expand_dims(normalized_image, axis=0)
            
            logger.info(f"Image preprocessed successfully. Shape: {preprocessed_image.shape}")
            return preprocessed_image
            
        except Exception as e:
            logger.error(f"Failed to preprocess image: {str(e)}")
            return None
    
    def preprocess_image_from_path(self, image_path: str) -> Optional[np.ndarray]:
        """
        Preprocess image from file path (for local testing)
        
        Args:
            image_path (str): Path to image file
            
        Returns:
            np.ndarray: Preprocessed image array ready for prediction
        """
        try:
            # Read image file as bytes
            with open(image_path, 'rb') as f:
                image_data = f.read()
            
            return self.preprocess_image(image_data)
            
        except Exception as e:
            logger.error(f"Failed to preprocess image from path {image_path}: {str(e)}")
            return None
    
    def predict_single_image(self, image_data: bytes) -> Dict:
        """
        Predict WBC class for a single image
        
        Args:
            image_data (bytes): Raw image data
            
        Returns:
            Dict: Prediction results with class, confidence, and all probabilities
        """
        try:
            if not self.is_loaded:
                raise Exception("Model not loaded. Call load_model() first.")
            
            # Preprocess the image
            preprocessed_image = self.preprocess_image(image_data)
            if preprocessed_image is None:
                raise Exception("Failed to preprocess image")
            
            # Make prediction
            predictions = self.model.predict(preprocessed_image, verbose=0)
            
            # Get prediction probabilities
            probabilities = predictions[0]
            
            # Get predicted class index and confidence
            predicted_class_index = np.argmax(probabilities)
            confidence = float(probabilities[predicted_class_index])
            predicted_class = self.class_names[predicted_class_index]
            
            # Create probability dictionary for all classes
            all_probabilities = {
                class_name: float(prob) 
                for class_name, prob in zip(self.class_names, probabilities)
            }
            
            # Sort probabilities in descending order
            sorted_probabilities = dict(
                sorted(all_probabilities.items(), key=lambda x: x[1], reverse=True)
            )
            
            result = {
                'predicted_class': predicted_class,
                'confidence': confidence,
                'all_probabilities': sorted_probabilities,
                'predicted_class_index': int(predicted_class_index)
            }
            
            logger.info(f"Prediction successful: {predicted_class} ({confidence:.3f})")
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise e
    
    def predict_single_image_from_path(self, image_path: str) -> Dict:
        """
        Predict WBC class for a single image from file path (for local testing)
        
        Args:
            image_path (str): Path to image file
            
        Returns:
            Dict: Prediction results with class, confidence, and all probabilities
        """
        try:
            # Read image file as bytes
            with open(image_path, 'rb') as f:
                image_data = f.read()
            
            return self.predict_single_image(image_data)
            
        except Exception as e:
            logger.error(f"Failed to predict image from path {image_path}: {str(e)}")
            raise e
    
    def predict_batch_images(self, images_data: List[Tuple[str, bytes]]) -> List[Dict]:
        """
        Predict WBC classes for multiple images
        
        Args:
            images_data (List[Tuple[str, bytes]]): List of tuples containing 
                                                   (filename, image_data)
        
        Returns:
            List[Dict]: List of prediction results for each image
        """
        results = []
        
        for filename, image_data in images_data:
            try:
                logger.info(f"Processing image: {filename}")
                
                # Get prediction for single image
                prediction_result = self.predict_single_image(image_data)
                
                # Add filename to result
                result = {
                    'filename': filename,
                    'success': True,
                    'results': prediction_result
                }
                
            except Exception as e:
                logger.error(f"Failed to process {filename}: {str(e)}")
                result = {
                    'filename': filename,
                    'success': False,
                    'error': str(e)
                }
            
            results.append(result)
        
        logger.info(f"Batch processing completed. Processed {len(results)} images.")
        return results
    
    def predict_batch_from_folder(self, folder_path: str, image_extensions: List[str] = None) -> List[Dict]:
        """
        Predict WBC classes for all images in a folder (for local testing)
        
        Args:
            folder_path (str): Path to folder containing images
            image_extensions (List[str]): List of image extensions to process
        
        Returns:
            List[Dict]: List of prediction results for each image
        """
        if image_extensions is None:
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        
        # Get all image files from folder
        image_files = []
        for ext in image_extensions:
            pattern = os.path.join(folder_path, ext)
            image_files.extend(glob.glob(pattern, recursive=False))
            # Also check uppercase extensions
            pattern = os.path.join(folder_path, ext.upper())
            image_files.extend(glob.glob(pattern, recursive=False))
        
        if not image_files:
            logger.warning(f"No image files found in {folder_path}")
            return []
        
        logger.info(f"Found {len(image_files)} images in {folder_path}")
        
        # Process each image
        results = []
        for image_path in image_files:
            try:
                filename = os.path.basename(image_path)
                logger.info(f"Processing image: {filename}")
                
                # Get prediction for single image
                prediction_result = self.predict_single_image_from_path(image_path)
                
                # Add filename and path to result
                result = {
                    'filename': filename,
                    'filepath': image_path,
                    'success': True,
                    'results': prediction_result
                }
                
            except Exception as e:
                logger.error(f"Failed to process {image_path}: {str(e)}")
                result = {
                    'filename': os.path.basename(image_path),
                    'filepath': image_path,
                    'success': False,
                    'error': str(e)
                }
            
            results.append(result)
        
        logger.info(f"Batch processing completed. Processed {len(results)} images.")
        return results
    
    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model
        
        Returns:
            Dict: Model information including classes, input shape, etc.
        """
        return {
            'is_loaded': self.is_loaded,
            'model_path': self.model_path,
            'class_names': self.class_names,
            'num_classes': len(self.class_names),
            'input_shape': self.input_shape
        }
    
    def validate_image_format(self, image_data: bytes) -> Tuple[bool, str]:
        """
        Validate if the uploaded file is a valid image
        
        Args:
            image_data (bytes): Raw image data
            
        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        try:
            image = Image.open(io.BytesIO(image_data))
            
            # Check if it's a valid image format
            valid_formats = ['JPEG', 'PNG', 'BMP', 'TIFF', 'GIF']
            if image.format not in valid_formats:
                return False, f"Unsupported image format: {image.format}"
            
            # Check image dimensions
            width, height = image.size
            if width < 32 or height < 32:
                return False, "Image too small. Minimum size is 32x32 pixels"
            
            if width > 4096 or height > 4096:
                return False, "Image too large. Maximum size is 4096x4096 pixels"
            
            return True, "Valid image"
            
        except Exception as e:
            return False, f"Invalid image file: {str(e)}"


# Global pipeline instance
pipeline = None

def initialize_pipeline(model_path: str) -> bool:
    """
    Initialize the global pipeline instance
    
    Args:
        model_path (str): Path to the model file
        
    Returns:
        bool: True if initialization successful
    """
    global pipeline
    try:
        pipeline = WBCClassificationPipeline(model_path)
        success = pipeline.load_model()
        return success
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {str(e)}")
        return False

def get_pipeline() -> Optional[WBCClassificationPipeline]:
    """
    Get the global pipeline instance
    
    Returns:
        WBCClassificationPipeline: The pipeline instance or None if not initialized
    """
    global pipeline
    return pipeline

def load_configuration():
    """
    Load configuration settings for the pipeline
    You can modify this function to load from config files, environment variables, etc.
    
    Returns:
        Dict: Configuration dictionary
    """
    config = {
        'model_path': 'path/to/your/model.keras',  # Update this path
        'test_image_path': None,  # Single image for testing
        'test_folder_path': None,  # Folder containing test images
        'image_extensions': ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff'],
        'log_level': 'INFO'
    }
    
    # You can override these with environment variables or config files
    # Example:
    # config['model_path'] = os.getenv('WBC_MODEL_PATH', config['model_path'])
    
    return config

def test_single_image(pipeline_instance: WBCClassificationPipeline, image_path: str):
    """
    Test pipeline with a single image
    
    Args:
        pipeline_instance: Initialized pipeline
        image_path (str): Path to test image
    """
    print(f"\n{'='*50}")
    print("TESTING SINGLE IMAGE")
    print(f"{'='*50}")
    
    try:
        result = pipeline_instance.predict_single_image_from_path(image_path)
        
        print(f"Image: {os.path.basename(image_path)}")
        print(f"Predicted Class: {result['predicted_class']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print("\nAll Probabilities:")
        for class_name, prob in result['all_probabilities'].items():
            print(f"  {class_name}: {prob:.4f}")
            
    except Exception as e:
        print(f"Error testing single image: {str(e)}")

def test_batch_images(pipeline_instance: WBCClassificationPipeline, folder_path: str):
    """
    Test pipeline with batch of images from folder
    
    Args:
        pipeline_instance: Initialized pipeline
        folder_path (str): Path to folder containing test images
    """
    print(f"\n{'='*50}")
    print("TESTING BATCH IMAGES")
    print(f"{'='*50}")
    
    try:
        results = pipeline_instance.predict_batch_from_folder(folder_path)
        
        print(f"Processed {len(results)} images from {folder_path}")
        print("\nResults Summary:")
        print("-" * 80)
        
        successful_predictions = 0
        for result in results:
            if result['success']:
                successful_predictions += 1
                pred_result = result['results']
                print(f"{result['filename']:<30} | {pred_result['predicted_class']:<12} | {pred_result['confidence']:.4f}")
            else:
                print(f"{result['filename']:<30} | ERROR: {result['error']}")
        
        print(f"\nSuccess Rate: {successful_predictions}/{len(results)} ({100*successful_predictions/len(results):.1f}%)")
        
    except Exception as e:
        print(f"Error testing batch images: {str(e)}")

def print_model_info(pipeline_instance: WBCClassificationPipeline):
    """
    Print model information
    
    Args:
        pipeline_instance: Initialized pipeline
    """
    print(f"\n{'='*50}")
    print("MODEL INFORMATION")
    print(f"{'='*50}")
    
    info = pipeline_instance.get_model_info()
    print(f"Model Loaded: {info['is_loaded']}")
    print(f"Model Path: {info['model_path']}")
    print(f"Input Shape: {info['input_shape']}")
    print(f"Number of Classes: {info['num_classes']}")
    print(f"Class Names: {', '.join(info['class_names'])}")

def main():
    """
    Main function for testing the pipeline
    """
    print("WBC Classification Pipeline - Testing Mode")
    print("=" * 60)
    
    # Load configuration
    config = load_configuration()
    
    # Update these paths according to your setup
    MODEL_PATH = "Models/at_batch_size=32.D24E.keras"  # Update this path
    TEST_IMAGE_PATH = "/home/amit/White_blood_application/uploads"  # Update this path  
    TEST_FOLDER_PATH = "/home/amit/White_blood_application/uploads"  # Update this path
    
    # You can override the paths from config
    if 'model_path' in config and config['model_path'] != 'path/to/your/model.keras':
        MODEL_PATH = config['model_path']
    
    # Check if model file exists
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file not found at {MODEL_PATH}")
        print("Please update the MODEL_PATH variable with the correct path to your model.")
        return
    
    # Initialize pipeline
    print(f"Initializing pipeline with model: {MODEL_PATH}")
    pipeline_instance = WBCClassificationPipeline(MODEL_PATH)
    
    # Load model
    if not pipeline_instance.load_model():
        print("Failed to load model. Exiting.")
        return
    
    # Print model information
    print_model_info(pipeline_instance)
    
    # Test single image if path provided and exists
    if TEST_IMAGE_PATH and os.path.exists(TEST_IMAGE_PATH):
        test_single_image(pipeline_instance, TEST_IMAGE_PATH)
    else:
        print(f"\nSkipping single image test (file not found: {TEST_IMAGE_PATH})")
    
    # Test batch images if folder exists
    if TEST_FOLDER_PATH and os.path.exists(TEST_FOLDER_PATH):
        test_batch_images(pipeline_instance, TEST_FOLDER_PATH)
    else:
        print(f"\nSkipping batch test (folder not found: {TEST_FOLDER_PATH})")
    
    print(f"\n{'='*60}")
    print("Testing completed!")
    print("Update the file paths above to test with your actual data.")
    print(f"{'='*60}")

