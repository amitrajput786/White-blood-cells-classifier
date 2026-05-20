#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 12 15:39:32 2025

@author: amit
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 12 15:10:26 2025

@author: amit
"""

#!/usr/bin/env python3
"""
🧪 AI-Powered Clothing Listing Orchestrator
======================================================================
This script orchestrates multiple AI models to automatically generate
clothing listings from photos, following the implementation priority order.
"""





import os
import time
import json
from typing import Dict, List, Tuple, Optional
from PIL import Image, ImageEnhance
import torch
from transformers import (
    AutoImageProcessor, AutoModelForImageClassification,
    CLIPProcessor, CLIPModel, CLIPTokenizer,
    DetrImageProcessor, DetrForObjectDetection,
    TrOCRProcessor, VisionEncoderDecoderModel,
    pipeline
)
import numpy as np
import warnings

warnings.filterwarnings('ignore')

# Configuration
MODEL_PATHS = {
    "fashion_classification": "/home/amit/odoo_hackathon/odoo_models/models--jolual2747--vit-clothes-classification",
    "clip_model": "/home/amit/odoo_hackathon/odoo_models/models--openai--clip-vit-base-patch32",
    "object_detection": "/home/amit/odoo_hackathon/odoo_models/models--facebook--detr-resnet-50",
    "ocr_base": "/home/amit/odoo_hackathon/odoo_models/models--microsoft--trocr-base-printed"
}

# Fallback model names if local paths fail
FALLBACK_MODEL_NAMES = {
    "fashion_classification": "jolual2747/vit-clothes-classification",
    "clip_model": "openai/clip-vit-base-patch32",
    "object_detection": "facebook/detr-resnet-50",
    "ocr_base": "microsoft/trocr-base-printed"
}

IMAGE_PATH = "/home/amit/odoo_hackathon/image/WhatsApp Image 2025-07-12 at 14.29.52 (1).jpeg"

# Label sets for different classification tasks
CLOTHING_LABELS = [
    'casual t-shirt', 'formal shirt', 'polo shirt', 'tank top',
    'long sleeve shirt', 'short sleeve shirt', 'graphic tee',
    'plain shirt', 'button-up shirt', 'henley shirt', 'blouse',
    'dress shirt', 'flannel shirt', 'jersey', 'sweater',
    'hoodie', 'cardigan', 'jacket', 'blazer', 'coat',
    'dress', 'skirt', 'pants', 'jeans', 'shorts',
    'trousers', 'leggings', 'joggers', 'sweatpants'
]

CONDITION_LABELS = [
    'new with tags', 'like new', 'excellent condition',
    'good condition', 'fair condition', 'worn condition'
]

COLOR_LABELS = [
    'black', 'white', 'gray', 'blue', 'red', 'green',
    'yellow', 'orange', 'purple', 'pink', 'brown',
    'navy', 'beige', 'cream', 'maroon', 'teal'
]

STYLE_LABELS = [
    'casual', 'formal', 'business', 'sporty', 'vintage',
    'modern', 'classic', 'trendy', 'elegant', 'comfortable'
]

class ClothingListingOrchestrator:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🚀 Initializing Clothing Listing Orchestrator")
        print(f"📱 Device: {self.device}")
        print("=" * 70)
        
        # Initialize all models
        self.models = {}
        self.processors = {}
        self.tokenizers = {}
        self.object_detection_pipeline = None
        self._load_all_models()
        
    def _load_all_models(self):
        """Load all AI models following priority order"""
        
        # Step 1: Load Object Detection Model
        print("🔧 Phase 1: Loading Object Detection Model...")
        start_time = time.time()
        
        try:
            # First try with pipeline approach
            self.object_detection_pipeline = pipeline(
                "object-detection", 
                model=FALLBACK_MODEL_NAMES['object_detection'],
                device=0 if torch.cuda.is_available() else -1
            )
            print("   ✅ Using pipeline approach for Object Detection")
        except Exception as e:
            print(f"   ⚠️  Pipeline approach failed: {e}")
            try:
                # Fall back to direct model loading
                self.processors['object_detection'] = DetrImageProcessor.from_pretrained(
                    FALLBACK_MODEL_NAMES['object_detection']
                )
                self.models['object_detection'] = DetrForObjectDetection.from_pretrained(
                    FALLBACK_MODEL_NAMES['object_detection']
                ).to(self.device)
                self.object_detection_pipeline = None
                print("   ✅ Using direct model loading with model name")
            except Exception as e2:
                print(f"   ❌ Both approaches failed: {e2}")
                raise e2
        
        print(f"   ✅ Object Detection loaded in {time.time() - start_time:.2f}s")
        
        # Step 2: Load Fashion Classification Model
        print("🔧 Phase 2: Loading Fashion Classification Model...")
        start_time = time.time()
        try:
            # Try local path first
            self.processors['fashion'] = AutoImageProcessor.from_pretrained(
                MODEL_PATHS['fashion_classification']
            )
            self.models['fashion'] = AutoModelForImageClassification.from_pretrained(
                MODEL_PATHS['fashion_classification']
            ).to(self.device)
            print("   ✅ Using local path for Fashion Classification")
        except Exception as e:
            print(f"   ⚠️  Local path failed: {e}")
            print("   ⚠️  Trying fallback model name...")
            try:
                self.processors['fashion'] = AutoImageProcessor.from_pretrained(
                    FALLBACK_MODEL_NAMES['fashion_classification']
                )
                self.models['fashion'] = AutoModelForImageClassification.from_pretrained(
                    FALLBACK_MODEL_NAMES['fashion_classification']
                ).to(self.device)
                print("   ✅ Using fallback model name for Fashion Classification")
            except Exception as e2:
                print(f"   ❌ Both approaches failed: {e2}")
                raise e2
        print(f"   ✅ Fashion Classification loaded in {time.time() - start_time:.2f}s")
        
        # Step 3: Load CLIP Model
        print("🔧 Phase 3: Loading CLIP Model...")
        start_time = time.time()
        try:
            # Try local path first
            self.tokenizers['clip'] = CLIPTokenizer.from_pretrained(MODEL_PATHS['clip_model'])
            self.processors['clip'] = CLIPProcessor.from_pretrained(MODEL_PATHS['clip_model'])
            self.models['clip'] = CLIPModel.from_pretrained(MODEL_PATHS['clip_model']).to(self.device)
            print("   ✅ Using local path for CLIP Model")
        except Exception as e:
            print(f"   ⚠️  Local path failed: {e}")
            print("   ⚠️  Trying fallback model name...")
            try:
                self.tokenizers['clip'] = CLIPTokenizer.from_pretrained(FALLBACK_MODEL_NAMES['clip_model'])
                self.processors['clip'] = CLIPProcessor.from_pretrained(FALLBACK_MODEL_NAMES['clip_model'])
                self.models['clip'] = CLIPModel.from_pretrained(FALLBACK_MODEL_NAMES['clip_model']).to(self.device)
                print("   ✅ Using fallback model name for CLIP Model")
            except Exception as e2:
                print(f"   ❌ Both approaches failed: {e2}")
                raise e2
        print(f"   ✅ CLIP Model loaded in {time.time() - start_time:.2f}s")
        
        # Step 4: Load TrOCR Model
        print("🔧 Phase 4: Loading TrOCR Model...")
        start_time = time.time()
        try:
            # Try local path first
            self.processors['ocr'] = TrOCRProcessor.from_pretrained(MODEL_PATHS['ocr_base'])
            self.models['ocr'] = VisionEncoderDecoderModel.from_pretrained(
                MODEL_PATHS['ocr_base']
            ).to(self.device)
            print("   ✅ Using local path for TrOCR Model")
        except Exception as e:
            print(f"   ⚠️  Local path failed: {e}")
            print("   ⚠️  Trying fallback model name...")
            try:
                self.processors['ocr'] = TrOCRProcessor.from_pretrained(FALLBACK_MODEL_NAMES['ocr_base'])
                self.models['ocr'] = VisionEncoderDecoderModel.from_pretrained(
                    FALLBACK_MODEL_NAMES['ocr_base']
                ).to(self.device)
                print("   ✅ Using fallback model name for TrOCR Model")
            except Exception as e2:
                print(f"   ❌ Both approaches failed: {e2}")
                raise e2
        print(f"   ✅ TrOCR Model loaded in {time.time() - start_time:.2f}s")
        
        print("=" * 70)
    
    def crop_clothing_from_image(self, image: Image.Image) -> Tuple[Image.Image, Dict]:
        """Step 1: Detect and crop clothing area from image"""
        print("🔍 Step 1: Object Detection & Cropping...")
        start_time = time.time()
        
        detection_info = {
            'detections': [],
            'best_person_box': None,
            'cropped_image': image  # Default to original if no person found
        }
        
        if self.object_detection_pipeline is not None:
            # Use pipeline approach
            try:
                results = self.object_detection_pipeline(image)
                
                # Find the best person detection
                best_person_score = 0
                best_person_box = None
                
                for result in results:
                    if result['score'] > 0.7:  # Confidence threshold
                        detection_info['detections'].append({
                            'label': result['label'],
                            'score': result['score'],
                            'box': [result['box']['xmin'], result['box']['ymin'], 
                                   result['box']['xmax'], result['box']['ymax']]
                        })
                        
                        if result['label'] == 'person' and result['score'] > best_person_score:
                            best_person_score = result['score']
                            best_person_box = [result['box']['xmin'], result['box']['ymin'], 
                                             result['box']['xmax'], result['box']['ymax']]
                
            except Exception as e:
                print(f"   ⚠️  Pipeline detection failed: {e}")
                # Fall back to original image
                pass
        else:
            # Use direct model approach
            try:
                inputs = self.processors['object_detection'](images=image, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.models['object_detection'](**inputs)
                
                # Process results
                target_sizes = torch.tensor([image.size[::-1]])
                results = self.processors['object_detection'].post_process_object_detection(
                    outputs, target_sizes=target_sizes, threshold=0.7
                )[0]
                
                # Find the best person detection
                best_person_score = 0
                best_person_box = None
                
                for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                    if score > 0.7:  # Confidence threshold
                        label_name = self.models['object_detection'].config.id2label[label.item()]
                        box_coords = box.cpu().numpy().astype(int)
                        
                        detection_info['detections'].append({
                            'label': label_name,
                            'score': score.item(),
                            'box': box_coords.tolist()
                        })
                        
                        if label_name == 'person' and score > best_person_score:
                            best_person_score = score.item()
                            best_person_box = box_coords
                            
            except Exception as e:
                print(f"   ⚠️  Direct model detection failed: {e}")
                # Fall back to original image
                pass
        
        # Crop the image if person detected
        if best_person_box is not None:
            x1, y1, x2, y2 = best_person_box
            # Add some padding
            padding = 20
            x1 = max(0, int(x1) - padding)
            y1 = max(0, int(y1) - padding)
            x2 = min(image.width, int(x2) + padding)
            y2 = min(image.height, int(y2) + padding)
            
            cropped_image = image.crop((x1, y1, x2, y2))
            detection_info['cropped_image'] = cropped_image
            detection_info['best_person_box'] = [x1, y1, x2, y2]
        
        print(f"   ✅ Object detection completed in {time.time() - start_time:.3f}s")
        print(f"   📊 Found {len(detection_info['detections'])} objects")
        if best_person_box is not None:
            print(f"   🎯 Best person detection: {best_person_score:.3f} confidence")
        
        return detection_info['cropped_image'], detection_info
    
    def classify_clothing_category(self, image: Image.Image) -> Dict:
        """Step 2: Primary clothing classification"""
        print("🔍 Step 2: Primary Clothing Classification...")
        start_time = time.time()
        
        # Preprocess image
        inputs = self.processors['fashion'](images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = self.models['fashion'](**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Get top predictions
        top_predictions = torch.topk(predictions, 5)
        
        results = []
        for score, idx in zip(top_predictions.values[0], top_predictions.indices[0]):
            label = self.models['fashion'].config.id2label[idx.item()]
            results.append({
                'label': label,
                'score': score.item(),
                'confidence': score.item()
            })
        
        classification_info = {
            'primary_category': results[0]['label'],
            'confidence': results[0]['score'],
            'all_predictions': results
        }
        
        print(f"   ✅ Fashion classification completed in {time.time() - start_time:.3f}s")
        print(f"   🎯 Primary category: {results[0]['label']} ({results[0]['score']:.3f})")
        
        return classification_info
    
    def refine_clothing_type(self, image: Image.Image, primary_category: str) -> Dict:
        """Step 3: CLIP-based clothing type refinement"""
        print("🔍 Step 3: CLIP Clothing Type Refinement...")
        start_time = time.time()
        
        # Filter clothing labels based on primary category
        relevant_labels = self._get_relevant_labels(primary_category, CLOTHING_LABELS)
        
        # Run CLIP classification
        inputs = self.processors['clip'](
            text=relevant_labels,
            images=image,
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.models['clip'](**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
        
        # Get top predictions
        top_probs, top_indices = torch.topk(probs, min(5, len(relevant_labels)))
        
        results = []
        for prob, idx in zip(top_probs[0], top_indices[0]):
            results.append({
                'label': relevant_labels[idx.item()],
                'score': prob.item(),
                'confidence': prob.item()
            })
        
        refinement_info = {
            'refined_type': results[0]['label'],
            'confidence': results[0]['score'],
            'all_predictions': results
        }
        
        print(f"   ✅ CLIP refinement completed in {time.time() - start_time:.3f}s")
        print(f"   🎯 Refined type: {results[0]['label']} ({results[0]['score']:.3f})")
        
        return refinement_info
    
    def assess_condition(self, image: Image.Image) -> Dict:
        """Step 4: Assess clothing condition using CLIP"""
        print("🔍 Step 4: Condition Assessment...")
        start_time = time.time()
        
        # Run CLIP classification for condition
        inputs = self.processors['clip'](
            text=CONDITION_LABELS,
            images=image,
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.models['clip'](**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
        
        # Get top predictions
        top_probs, top_indices = torch.topk(probs, len(CONDITION_LABELS))
        
        results = []
        for prob, idx in zip(top_probs[0], top_indices[0]):
            results.append({
                'label': CONDITION_LABELS[idx.item()],
                'score': prob.item(),
                'confidence': prob.item()
            })
        
        condition_info = {
            'condition': results[0]['label'],
            'confidence': results[0]['score'],
            'all_predictions': results
        }
        
        print(f"   ✅ Condition assessment completed in {time.time() - start_time:.3f}s")
        print(f"   🎯 Condition: {results[0]['label']} ({results[0]['score']:.3f})")
        
        return condition_info
    
    def detect_color_and_style(self, image: Image.Image) -> Dict:
        """Step 5: Detect color and style using CLIP"""
        print("🔍 Step 5: Color & Style Detection...")
        start_time = time.time()
        
        # Detect color
        color_inputs = self.processors['clip'](
            text=COLOR_LABELS,
            images=image,
            return_tensors="pt",
            padding=True
        )
        color_inputs = {k: v.to(self.device) for k, v in color_inputs.items()}
        
        with torch.no_grad():
            color_outputs = self.models['clip'](**color_inputs)
            color_probs = color_outputs.logits_per_image.softmax(dim=1)
        
        # Get top color
        top_color_prob, top_color_idx = torch.topk(color_probs, 1)
        detected_color = COLOR_LABELS[top_color_idx[0].item()]
        color_confidence = top_color_prob[0].item()
        
        # Detect style
        style_inputs = self.processors['clip'](
            text=STYLE_LABELS,
            images=image,
            return_tensors="pt",
            padding=True
        )
        style_inputs = {k: v.to(self.device) for k, v in style_inputs.items()}
        
        with torch.no_grad():
            style_outputs = self.models['clip'](**style_inputs)
            style_probs = style_outputs.logits_per_image.softmax(dim=1)
        
        # Get top style
        top_style_prob, top_style_idx = torch.topk(style_probs, 1)
        detected_style = STYLE_LABELS[top_style_idx[0].item()]
        style_confidence = top_style_prob[0].item()
        
        attributes_info = {
            'color': detected_color,
            'color_confidence': color_confidence,
            'style': detected_style,
            'style_confidence': style_confidence
        }
        
        print(f"   ✅ Color & style detection completed in {time.time() - start_time:.3f}s")
        print(f"   🎨 Color: {detected_color} ({color_confidence:.3f})")
        print(f"   ✨ Style: {detected_style} ({style_confidence:.3f})")
        
        return attributes_info
    
    def extract_text_info(self, image: Image.Image) -> Dict:
        """Step 6: Extract text information using TrOCR"""
        print("🔍 Step 6: Text Extraction...")
        start_time = time.time()
        
        # Enhance image for better text recognition
        enhanced_image = self._enhance_image_for_ocr(image)
        
        # Run TrOCR
        pixel_values = self.processors['ocr'](
            images=enhanced_image, 
            return_tensors="pt"
        ).pixel_values.to(self.device)
        
        with torch.no_grad():
            generated_ids = self.models['ocr'].generate(pixel_values)
        
        generated_text = self.processors['ocr'].batch_decode(
            generated_ids, 
            skip_special_tokens=True
        )[0]
        
        # Extract useful information from text
        text_info = self._parse_extracted_text(generated_text)
        
        print(f"   ✅ Text extraction completed in {time.time() - start_time:.3f}s")
        print(f"   📝 Extracted text: '{generated_text}'")
        if text_info['brand']:
            print(f"   🏷️  Potential brand: {text_info['brand']}")
        
        return text_info
    
    def generate_listing(self, image_path: str) -> Dict:
        """Main orchestration function - generate complete listing"""
        print("🚀 STARTING CLOTHING LISTING GENERATION")
        print("=" * 70)
        
        overall_start_time = time.time()
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        print(f"📸 Loaded image: {os.path.basename(image_path)}")
        print(f"📐 Image size: {image.size}")
        print("=" * 70)
        
        # Step 1: Object Detection & Cropping
        cropped_image, detection_info = self.crop_clothing_from_image(image)
        
        # Step 2: Primary Classification
        classification_info = self.classify_clothing_category(cropped_image)
        
        # Step 3: CLIP Refinement
        refinement_info = self.refine_clothing_type(
            cropped_image, 
            classification_info['primary_category']
        )
        
        # Step 4: Condition Assessment
        condition_info = self.assess_condition(cropped_image)
        
        # Step 5: Color & Style Detection
        attributes_info = self.detect_color_and_style(cropped_image)
        
        # Step 6: Text Extraction
        text_info = self.extract_text_info(cropped_image)
        
        # Generate final listing
        listing = self._compile_final_listing(
            classification_info, refinement_info, condition_info,
            attributes_info, text_info, detection_info
        )
        
        total_time = time.time() - overall_start_time
        print("=" * 70)
        print("🎉 CLOTHING LISTING GENERATION COMPLETED!")
        print(f"⏱️  Total processing time: {total_time:.2f}s")
        print("=" * 70)
        
        return listing
    
    def _get_relevant_labels(self, primary_category: str, all_labels: List[str]) -> List[str]:
        """Filter labels based on primary category"""
        primary_lower = primary_category.lower()
        
        if 't-shirt' in primary_lower or 'tshirt' in primary_lower:
            return [label for label in all_labels if 'shirt' in label or 'tee' in label]
        elif 'dress' in primary_lower:
            return [label for label in all_labels if 'dress' in label or 'gown' in label]
        elif 'shirt' in primary_lower:
            return [label for label in all_labels if 'shirt' in label]
        elif 'top' in primary_lower:
            return [label for label in all_labels if 'top' in label or 'shirt' in label or 'blouse' in label]
        else:
            return all_labels[:15]  # Return subset for efficiency
    
    def _enhance_image_for_ocr(self, image: Image.Image) -> Image.Image:
        """Enhance image for better OCR results"""
        # Convert to grayscale and enhance contrast
        
        # Convert to grayscale
        gray_image = image.convert('L')
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(gray_image)
        enhanced_image = enhancer.enhance(2.0)
        
        # Convert back to RGB
        return enhanced_image.convert('RGB')
    
    def _parse_extracted_text(self, text: str) -> Dict:
        """Parse extracted text for useful information"""
        # Common clothing brands (you can expand this list)
        common_brands = [
            'nike', 'adidas', 'puma', 'under armour', 'calvin klein',
            'tommy hilfiger', 'polo', 'ralph lauren', 'gap', 'zara',
            'h&m', 'uniqlo', 'forever 21', 'american eagle', 'hollister'
        ]
        
        # Common sizes
        size_patterns = ['XS', 'S', 'M', 'L', 'XL', 'XXL', '2XL', '3XL']
        
        text_lower = text.lower()
        
        # Look for brand
        detected_brand = None
        for brand in common_brands:
            if brand in text_lower:
                detected_brand = brand.title()
                break
        
        # Look for size
        detected_size = None
        for size in size_patterns:
            if size in text.upper():
                detected_size = size
                break
        
        return {
            'raw_text': text,
            'brand': detected_brand,
            'size': detected_size,
            'has_text': len(text.strip()) > 0
        }
    
    def _compile_final_listing(self, classification_info: Dict, refinement_info: Dict,
                             condition_info: Dict, attributes_info: Dict,
                             text_info: Dict, detection_info: Dict) -> Dict:
        """Compile all information into final listing"""
        
        # Calculate overall confidence
        confidences = [
            classification_info['confidence'],
            refinement_info['confidence'],
            condition_info['confidence'],
            attributes_info['color_confidence'],
            attributes_info['style_confidence']
        ]
        overall_confidence = sum(confidences) / len(confidences)
        
        # Generate title
        title_parts = []
        if text_info['brand']:
            title_parts.append(text_info['brand'])
        
        title_parts.append(refinement_info['refined_type'].title())
        
        if attributes_info['color'] != 'black':  # Don't add black as it's common
            title_parts.append(f"- {attributes_info['color'].title()}")
        
        if condition_info['condition'] != 'good condition':
            title_parts.append(f"- {condition_info['condition'].title()}")
        
        suggested_title = ' '.join(title_parts)
        
        # Generate description
        description = f"Comfortable {attributes_info['style']} {refinement_info['refined_type']} "
        description += f"in {condition_info['condition']}. "
        description += f"Perfect for {attributes_info['style']} wear."
        
        # Compile final listing
        final_listing = {
            'listing_info': {
                'title': suggested_title,
                'description': description,
                'category': classification_info['primary_category'],
                'subcategory': refinement_info['refined_type'],
                'condition': condition_info['condition'],
                'color': attributes_info['color'],
                'style': attributes_info['style'],
                'brand': text_info['brand'],
                'size': text_info['size'],
                'overall_confidence': overall_confidence
            },
            'model_results': {
                'object_detection': detection_info,
                'classification': classification_info,
                'refinement': refinement_info,
                'condition': condition_info,
                'attributes': attributes_info,
                'text_extraction': text_info
            },
            'metadata': {
                'models_used': list(MODEL_PATHS.keys()),
                'processing_pipeline': [
                    'object_detection', 'classification', 'refinement',
                    'condition_assessment', 'attribute_detection', 'text_extraction'
                ]
            }
        }
        
        return final_listing
    
    def print_final_results(self, listing: Dict):
        """Print beautifully formatted final results"""
        print("📋 FINAL CLOTHING LISTING")
        print("=" * 70)
        
        listing_info = listing['listing_info']
        
        print(f"🏷️  Title: {listing_info['title']}")
        print(f"📝 Description: {listing_info['description']}")
        print(f"📂 Category: {listing_info['category']}")
        print(f"🔍 Subcategory: {listing_info['subcategory']}")
        print(f"⭐ Condition: {listing_info['condition']}")
        print(f"🎨 Color: {listing_info['color']}")
        print(f"✨ Style: {listing_info['style']}")
        if listing_info['brand']:
            print(f"🏷️  Brand: {listing_info['brand']}")
        if listing_info['size']:
            print(f"📏 Size: {listing_info['size']}")
        print(f"🎯 Overall Confidence: {listing_info['overall_confidence']:.3f}")
        
        print("\n📊 MODEL PERFORMANCE SUMMARY")
        print("-" * 70)
        model_results = listing['model_results']
        
        print(f"🔍 Object Detection: {len(model_results['object_detection']['detections'])} objects found")
        print(f"📂 Classification: {model_results['classification']['primary_category']} "
              f"({model_results['classification']['confidence']:.3f})")
        print(f"🔍 Refinement: {model_results['refinement']['refined_type']} "
              f"({model_results['refinement']['confidence']:.3f})")
        print(f"⭐ Condition: {model_results['condition']['condition']} "
              f"({model_results['condition']['confidence']:.3f})")
        print(f"🎨 Color: {model_results['attributes']['color']} "
              f"({model_results['attributes']['color_confidence']:.3f})")
        print(f"✨ Style: {model_results['attributes']['style']} "
              f"({model_results['attributes']['style_confidence']:.3f})")
        print(f"📝 Text: '{model_results['text_extraction']['raw_text']}'")
        
        print("=" * 70) 
              
              
              
              
              
              
           
def main():
    """Main function to run the clothing listing orchestrator"""
    print("🧪 CLOTHING LISTING AI ORCHESTRATOR")
    print("=" * 70)
    
    # Initialize orchestrator
    orchestrator = ClothingListingOrchestrator()
    
    # Generate listing
    listing = orchestrator.generate_listing(IMAGE_PATH)
    
    # Print results
    orchestrator.print_final_results(listing)
    
    # Save results to JSON
    output_file = "clothing_listing_results.json"
    with open(output_file, 'w') as f:
        json.dump(listing, f, indent=2)
    
    print(f"💾 Results saved to: {output_file}")
    print("🎉 ORCHESTRATION COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    main()