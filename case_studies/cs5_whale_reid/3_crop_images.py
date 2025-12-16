#!/usr/bin/env python3
"""
Step 3: Crop images using Grounding DINO

Uses Grounding DINO to detect whale tails in images and crop to the detected region.
Processes both iNaturalist and HappyWhale images.
"""

import os
import sys
import argparse
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from torchvision.ops import box_convert

# Add Grounding DINO to path
GROUNDING_DINO_PATH = os.path.join(os.path.dirname(__file__), "GroundingDINO")
if os.path.exists(GROUNDING_DINO_PATH):
    sys.path.insert(0, GROUNDING_DINO_PATH)

try:
    from groundingdino.util.inference import load_model, predict, load_image
except ImportError:
    print("Warning: Grounding DINO not found. Please install following README instructions.")
    print("You can still run other steps of the pipeline.")
    exit(1)

def crop_image_with_box(image_path, box):
    """
    Crop an image using the provided bounding box.
    
    Args:
        image_path: Path to the image
        box: Bounding box in cxcywh format (normalized)
        
    Returns:
        PIL Image: Cropped image
    """
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    
    # Convert box to pixel coordinates
    box = box * torch.Tensor([w, h, w, h])
    box = box_convert(boxes=box, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    
    # Crop the image
    cropped = image.crop(box)
    return cropped

def process_image(image_path, model, prompt, box_threshold, text_threshold, device):
    """
    Process a single image: detect whale tail and crop.
    
    Args:
        image_path: Path to input image
        model: Grounding DINO model
        prompt: Detection prompt (e.g., "whale tail")
        box_threshold: Box confidence threshold
        text_threshold: Text confidence threshold
        device: torch device
        
    Returns:
        PIL Image: Cropped image (or full image if no detection)
    """
    try:
        image_source, image = load_image(image_path)
        
        boxes, logits, phrases = predict(
            model=model,
            image=image,
            caption=prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=device,
        )
        
        # Use the first detected box, or full image if no detection
        if len(boxes) > 0:
            cropped_image = crop_image_with_box(image_path, boxes[0])
        else:
            # No detection - use full image
            cropped_image = Image.open(image_path).convert("RGB")
        
        return cropped_image
    
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        # Return full image on error
        try:
            return Image.open(image_path).convert("RGB")
        except:
            return None

def process_directory(input_dir, output_dir, model, prompt, box_threshold, text_threshold, device, skip_existing=True):
    """
    Process all images in a directory.
    
    Args:
        input_dir: Input directory with images
        output_dir: Output directory for cropped images
        model: Grounding DINO model
        prompt: Detection prompt
        box_threshold: Box confidence threshold
        text_threshold: Text confidence threshold
        device: torch device
        skip_existing: Skip already processed images
        
    Returns:
        tuple: (processed_count, skipped_count, failed_count)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    image_extensions = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
    image_files = [f for f in os.listdir(input_dir) if f.endswith(image_extensions)]
    
    processed = 0
    skipped = 0
    failed = 0
    
    for img_file in tqdm(image_files, desc=f"Processing {os.path.basename(input_dir)}"):
        input_path = os.path.join(input_dir, img_file)
        output_path = os.path.join(output_dir, img_file)
        
        # Skip if already processed
        if skip_existing and os.path.exists(output_path):
            skipped += 1
            continue
        
        # Process the image
        cropped_image = process_image(
            input_path, model, prompt, box_threshold, text_threshold, device
        )
        
        if cropped_image is not None:
            cropped_image.save(output_path)
            processed += 1
        else:
            failed += 1
    
    return processed, skipped, failed

def main():
    parser = argparse.ArgumentParser(description="Crop whale images using Grounding DINO")
    parser.add_argument("--inat-dir", type=str, default="inat_images",
                        help="Directory with iNaturalist images")
    parser.add_argument("--hw-dir", type=str, default="happywhale_images",
                        help="Directory with HappyWhale images")
    parser.add_argument("--output", type=str, default="cropped_images",
                        help="Output directory for cropped images")
    parser.add_argument("--config", type=str, 
                        default="GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
                        help="Path to Grounding DINO config")
    parser.add_argument("--weights", type=str,
                        default="GroundingDINO/weights/groundingdino_swint_ogc.pth",
                        help="Path to Grounding DINO weights")
    parser.add_argument("--prompt", type=str, default="whale tail",
                        help="Detection prompt")
    parser.add_argument("--box-threshold", type=float, default=0.35,
                        help="Box confidence threshold")
    parser.add_argument("--text-threshold", type=float, default=0.25,
                        help="Text confidence threshold")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda or cpu)")
    parser.add_argument("--skip-existing", action="store_true", default=True,
                        help="Skip already cropped images")
    
    args = parser.parse_args()
    
    # Check if input directories exist
    if not os.path.exists(args.inat_dir):
        print(f"Warning: iNaturalist directory not found: {args.inat_dir}")
        print("Run step 1 first to download iNaturalist images.")
    
    if not os.path.exists(args.hw_dir):
        print(f"Warning: HappyWhale directory not found: {args.hw_dir}")
        print("Run step 2 first to download HappyWhale images.")
    
    # Create output directories
    inat_output = os.path.join(args.output, "inat")
    hw_output = os.path.join(args.output, "happywhale")
    os.makedirs(inat_output, exist_ok=True)
    os.makedirs(hw_output, exist_ok=True)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load Grounding DINO model
    print("\nLoading Grounding DINO model...")
    try:
        model = load_model(args.config, args.weights).to(device)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure Grounding DINO is installed and weights are downloaded.")
        return
    
    # Process iNaturalist images
    if os.path.exists(args.inat_dir):
        print(f"\n{'='*60}")
        print("Processing iNaturalist images...")
        print(f"{'='*60}")
        inat_processed, inat_skipped, inat_failed = process_directory(
            args.inat_dir, inat_output, model, args.prompt,
            args.box_threshold, args.text_threshold, device, args.skip_existing
        )
        print(f"Processed: {inat_processed}, Skipped: {inat_skipped}, Failed: {inat_failed}")
    
    # Process HappyWhale images
    if os.path.exists(args.hw_dir):
        print(f"\n{'='*60}")
        print("Processing HappyWhale images...")
        print(f"{'='*60}")
        hw_processed, hw_skipped, hw_failed = process_directory(
            args.hw_dir, hw_output, model, args.prompt,
            args.box_threshold, args.text_threshold, device, args.skip_existing
        )
        print(f"Processed: {hw_processed}, Skipped: {hw_skipped}, Failed: {hw_failed}")
    
    print(f"\n{'='*60}")
    print("Cropping complete!")
    print(f"{'='*60}")
    print(f"iNaturalist cropped images: {inat_output}")
    print(f"HappyWhale cropped images: {hw_output}")

if __name__ == "__main__":
    main()
