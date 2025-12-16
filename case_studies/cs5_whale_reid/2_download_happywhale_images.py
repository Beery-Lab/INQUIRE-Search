#!/usr/bin/env python3
"""
Step 2: Download HappyWhale dataset images

Downloads whale images from the HappyWhale dataset using the wildlife-datasets library.
Only downloads a subset of images if a limit is specified.
"""

import os
import argparse
import shutil
from pathlib import Path
from tqdm import tqdm

try:
    from wildlife_datasets import datasets, loader
except ImportError:
    print("Error: wildlife-datasets not installed. Install with: pip install wildlife-datasets")
    exit(1)

def main():
    parser = argparse.ArgumentParser(description="Download HappyWhale dataset images")
    parser.add_argument("--output", type=str, default="happywhale_images",
                        help="Output directory for downloaded images")
    parser.add_argument("--dataset", type=str, default="HumpbackWhaleID",
                        choices=["HumpbackWhaleID", "HappyWhale"],
                        help="Dataset to download from wildlife-datasets")
    parser.add_argument("--wildlife-datasets-path", type=str, default="wildlife_datasets_data",
                        help="Path where wildlife-datasets will store data")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip downloading and use existing data at wildlife-datasets-path")
    parser.add_argument("--limit", type=int, default=None,
                        help="Maximum number of images to download (None = all)")
    parser.add_argument("--species-filter", type=str, default=None,
                        help="Filter by species name (e.g., 'humpback_whale')")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(args.wildlife_datasets_path, exist_ok=True)
    
    print(f"Output directory: {args.output}")
    print(f"Wildlife datasets path: {args.wildlife_datasets_path}")
    
    # Load dataset
    print(f"\nLoading {args.dataset} dataset...")
    print("This will download the dataset if it's not already cached...")
    print("(This may take a while on first run)")
    
    if args.dataset == "HumpbackWhaleID":
        dataset_cls = datasets.humpback_whale_id.HumpbackWhaleID
    else:
        dataset_cls = datasets.happy_whale.HappyWhale
    
    try:
        # Download the dataset if not already present and not skipped
        if not args.skip_download:
            print("Downloading dataset (if not already cached)...")
            print("Note: This requires Kaggle API credentials.")
            print("If you already have the data, use --skip-download flag")
            try:
                dataset_cls.download(root=args.wildlife_datasets_path)
            except Exception as download_error:
                print(f"\nDownload failed: {download_error}")
                print("\nTo download this dataset, you need:")
                print("1. Kaggle API credentials (kaggle.json)")
                print("2. Accept the dataset terms on Kaggle website")
                print("\nAlternatively, use an existing wildlife-datasets directory:")
                print(f"   python3 2_download_happywhale_images.py --wildlife-datasets-path /path/to/existing/data --skip-download")
                return
        else:
            print("Skipping download, using existing data...")
        
        # Load the dataset
        print("Loading dataset...")
        wildlife_dataset = dataset_cls(root=args.wildlife_datasets_path)
        df = wildlife_dataset.df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("\nMake sure wildlife-datasets is properly installed and you have the data.")
        print("If you have the data elsewhere, specify the path with --wildlife-datasets-path")
        return
    
    print(f"Dataset loaded: {len(df)} total images")
    
    # Filter by species if requested
    if args.species_filter and 'species' in df.columns:
        df = df[df['species'] == args.species_filter]
        print(f"Filtered to {len(df)} images for species: {args.species_filter}")
    
    # Limit number of images if requested
    if args.limit and args.limit < len(df):
        df = df.head(args.limit)
        print(f"Limited to {args.limit} images")
    
    # Copy images to output directory
    print(f"\nCopying images to {args.output}...")
    
    copied = 0
    skipped = 0
    failed = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Copying images"):
        src_path = row['path']
        
        # Create unique filename using image_id
        image_id = row['image_id']
        dst_filename = f"{image_id}.jpg"
        dst_path = os.path.join(args.output, dst_filename)
        
        # Skip if already exists
        if os.path.exists(dst_path):
            skipped += 1
            continue
        
        # Copy the image
        try:
            if os.path.exists(src_path):
                shutil.copy2(src_path, dst_path)
                copied += 1
            else:
                print(f"Source file not found: {src_path}")
                failed += 1
        except Exception as e:
            print(f"Error copying {src_path}: {e}")
            failed += 1
    
    # Save metadata CSV
    metadata_path = os.path.join(args.output, "metadata.csv")
    df.to_csv(metadata_path, index=False)
    print(f"\nSaved metadata to: {metadata_path}")
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Download Summary:")
    print(f"{'='*60}")
    print(f"Total images in dataset: {len(df)}")
    print(f"Images copied:           {copied}")
    print(f"Images skipped (exist):  {skipped}")
    print(f"Images failed:           {failed}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
