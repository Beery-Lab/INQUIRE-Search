#!/usr/bin/env python3
"""
Step 1: Download iNaturalist images from CSV

Downloads whale images from iNaturalist using URLs provided in a CSV file.
Skips images that have already been downloaded.
"""

import os
import csv
import requests
import time
import argparse
from pathlib import Path
from tqdm import tqdm

def download_image(photo_id, url, save_dir, delay=0.5):
    """
    Download an image from a URL and save it to the specified directory.
    
    Args:
        photo_id: Unique identifier for the photo
        url: Download URL for the image
        save_dir: Directory to save the image
        delay: Delay in seconds after download
        
    Returns:
        bool: True if download successful, False otherwise
    """
    file_path = os.path.join(save_dir, f"{photo_id}.jpg")
    
    # Skip if already exists
    if os.path.exists(file_path):
        return True
    
    # Try with different extensions
    extensions_to_try = ['jpg', 'jpeg']
    
    for ext_index, ext in enumerate(extensions_to_try):
        try:
            # Modify URL if trying alternate extension
            current_url = url
            if ext_index > 0:
                current_url = url.replace('large.jpg', f'large.{ext}').replace('medium.jpg', f'medium.{ext}')
            
            response = requests.get(current_url, stream=True, timeout=30)
            response.raise_for_status()
            
            # Save the image
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Add delay to avoid overwhelming server
            time.sleep(delay)
            return True
            
        except Exception as e:
            if ext_index < len(extensions_to_try) - 1:
                continue
            else:
                print(f"Error downloading {photo_id}: {e}")
                return False
    
    return False

def main():
    parser = argparse.ArgumentParser(description="Download iNaturalist whale images from CSV")
    parser.add_argument("--csv", type=str, default="whale_results_with_url_filtered.csv",
                        help="Path to CSV file with photo_id and download_url columns")
    parser.add_argument("--output", type=str, default="inat_images",
                        help="Output directory for downloaded images")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="Delay in seconds between downloads")
    parser.add_argument("--photo-id-col", type=str, default="photo_id",
                        help="Name of photo ID column in CSV")
    parser.add_argument("--url-col", type=str, default="download_url",
                        help="Name of download URL column in CSV")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    print(f"Created/verified output directory: {args.output}")
    
    # Read CSV and download images
    if not os.path.exists(args.csv):
        print(f"Error: CSV file not found: {args.csv}")
        return
    
    total_images = 0
    downloaded_images = 0
    skipped_images = 0
    failed_images = 0
    
    with open(args.csv, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        total_images = len(rows)
        
        print(f"Found {total_images} images in CSV")
        print("Starting download...")
        
        for row in tqdm(rows, desc="Downloading images"):
            photo_id = row[args.photo_id_col]
            download_url = row[args.url_col]
            
            # Check if already exists
            file_path = os.path.join(args.output, f"{photo_id}.jpg")
            if os.path.exists(file_path):
                skipped_images += 1
                continue
            
            # Download the image
            if download_image(photo_id, download_url, args.output, args.delay):
                downloaded_images += 1
            else:
                failed_images += 1
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Download Summary:")
    print(f"{'='*60}")
    print(f"Total images in CSV:     {total_images}")
    print(f"Images downloaded:       {downloaded_images}")
    print(f"Images skipped (exist):  {skipped_images}")
    print(f"Images failed:           {failed_images}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
