#!/usr/bin/env python3
"""
Step 4: Generate embeddings and find matches

Uses the MiewID model to:
1. Generate embeddings for all cropped images (iNaturalist and HappyWhale)
2. Compute distance matrix
3. For each iNaturalist query, find top-K closest HappyWhale images
4. Save results to CSV
"""

import os
import argparse
import torch
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

try:
    from transformers import AutoModel
except ImportError:
    print("Error: transformers not installed. Install with: pip install transformers")
    exit(1)

def load_model(model_name="conservationxlabs/miewid-msv2", device="cuda"):
    """
    Load the MiewID model from HuggingFace.
    
    Args:
        model_name: HuggingFace model identifier
        device: torch device
        
    Returns:
        tuple: (model, transforms, image_size)
    """
    print(f"Loading model: {model_name}")
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    model.to(device)
    model.eval()
    
    # Define transforms for MiewID
    img_transforms = transforms.Compose([
        transforms.Resize((440, 440)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    return model, img_transforms, (440, 440)

def load_image(image_path, transform):
    """
    Load and transform an image.
    
    Args:
        image_path: Path to image
        transform: torchvision transform
        
    Returns:
        torch.Tensor: Transformed image
    """
    try:
        img = Image.open(image_path).convert("RGB")
        return transform(img)
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        return None

def generate_embeddings(image_dir, model, transform, device, batch_size=16):
    """
    Generate embeddings for all images in a directory.
    
    Args:
        image_dir: Directory containing images
        model: MiewID model
        transform: Image transforms
        device: torch device
        batch_size: Batch size for processing
        
    Returns:
        tuple: (embeddings tensor, list of filenames)
    """
    # Get all image files
    image_extensions = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(image_extensions)])
    
    if len(image_files) == 0:
        print(f"Warning: No images found in {image_dir}")
        return torch.empty(0), []
    
    print(f"Found {len(image_files)} images in {image_dir}")
    
    embeddings_list = []
    valid_files = []
    
    # Process in batches
    for i in tqdm(range(0, len(image_files), batch_size), desc="Generating embeddings"):
        batch_files = image_files[i:i+batch_size]
        batch_images = []
        batch_valid_files = []
        
        # Load batch
        for img_file in batch_files:
            img_path = os.path.join(image_dir, img_file)
            img_tensor = load_image(img_path, transform)
            if img_tensor is not None:
                batch_images.append(img_tensor)
                batch_valid_files.append(img_file)
        
        if len(batch_images) == 0:
            continue
        
        # Stack and move to device
        batch_tensor = torch.stack(batch_images).to(device)
        
        # Generate embeddings
        with torch.no_grad():
            batch_embeddings = model(batch_tensor)
        
        embeddings_list.append(batch_embeddings.cpu())
        valid_files.extend(batch_valid_files)
    
    if len(embeddings_list) == 0:
        return torch.empty(0), []
    
    # Concatenate all embeddings
    all_embeddings = torch.cat(embeddings_list, dim=0)
    
    return all_embeddings, valid_files

def compute_distance_matrix(embeddings1, embeddings2=None):
    """
    Compute pairwise distance matrix between embeddings.
    
    Args:
        embeddings1: First set of embeddings
        embeddings2: Second set of embeddings (if None, compute within embeddings1)
        
    Returns:
        torch.Tensor: Distance matrix
    """
    if embeddings2 is None:
        embeddings2 = embeddings1
    
    # Normalize embeddings
    embeddings1 = torch.nn.functional.normalize(embeddings1, p=2, dim=1)
    embeddings2 = torch.nn.functional.normalize(embeddings2, p=2, dim=1)
    
    # Compute cosine distance (1 - cosine similarity)
    similarity = torch.mm(embeddings1, embeddings2.t())
    distance = 1 - similarity
    
    return distance

def find_top_matches(query_embeddings, gallery_embeddings, top_k=3):
    """
    Find top-K closest gallery images for each query.
    
    Args:
        query_embeddings: Query embeddings
        gallery_embeddings: Gallery embeddings
        top_k: Number of top matches to retrieve
        
    Returns:
        tuple: (indices, distances) for top matches
    """
    # Compute distance matrix
    dist_matrix = compute_distance_matrix(query_embeddings, gallery_embeddings)
    
    # Find top-K smallest distances
    distances, indices = torch.topk(dist_matrix, k=top_k, dim=1, largest=False)
    
    return indices, distances

def save_results_to_csv(query_files, gallery_files, indices, distances, 
                        inat_csv=None, hw_metadata=None, output_csv="matches.csv"):
    """
    Save matching results to CSV.
    
    Args:
        query_files: List of query image filenames
        gallery_files: List of gallery image filenames
        indices: Top-K indices for each query
        distances: Top-K distances for each query
        inat_csv: Path to original iNaturalist CSV (optional)
        hw_metadata: Path to HappyWhale metadata CSV (optional)
        output_csv: Output CSV path
    """
    # Load metadata if available
    inat_df = None
    if inat_csv and os.path.exists(inat_csv):
        inat_df = pd.read_csv(inat_csv)
    
    hw_df = None
    if hw_metadata and os.path.exists(hw_metadata):
        hw_df = pd.read_csv(hw_metadata)
    
    rows = []
    
    for q_idx, q_file in enumerate(query_files):
        # Extract photo ID from filename
        q_annot = os.path.splitext(q_file)[0]
        
        # Get query metadata
        q_label = None
        q_url = None
        if inat_df is not None:
            q_row = inat_df[inat_df['photo_id'].astype(str) == str(q_annot)]
            if len(q_row) > 0:
                q_row = q_row.iloc[0]
                q_label = q_row.get('individual_id', None)
                q_url = q_row.get('inat_url', None)
                if pd.isna(q_url) and 'photo_id' in q_row:
                    q_url = f"https://www.inaturalist.org/photos/{q_row['photo_id']}"
        
        # Get top-K matches
        for rank in range(indices.shape[1]):
            n_idx = indices[q_idx, rank].item()
            n_file = gallery_files[n_idx]
            n_annot = os.path.splitext(n_file)[0]
            distance = distances[q_idx, rank].item()
            
            # Get neighbor metadata
            n_label = None
            n_url = n_file  # Default to filename
            if hw_df is not None:
                n_row = hw_df[hw_df['image_id'].astype(str) == str(n_annot)]
                if len(n_row) > 0:
                    n_row = n_row.iloc[0]
                    n_label = n_row.get('identity', None)
                    if pd.isna(n_label):
                        n_label = n_row.get('individual_id', None)
            
            rows.append({
                "query_annot": q_annot,
                "query_label": q_label,
                "query_url": q_url,
                "rank": rank + 1,
                "neighbor_annot": n_annot,
                "neighbor_label": n_label,
                "neighbor_url": n_url,
                "distance": distance,
            })
    
    # Create DataFrame and save
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"\nSaved {len(df)} matches to {output_csv}")
    
    return df

def main():
    parser = argparse.ArgumentParser(description="Generate embeddings and find matches")
    parser.add_argument("--inat-dir", type=str, default="cropped_images/inat",
                        help="Directory with cropped iNaturalist images")
    parser.add_argument("--hw-dir", type=str, default="cropped_images/happywhale",
                        help="Directory with cropped HappyWhale images")
    parser.add_argument("--output", type=str, default="matches.csv",
                        help="Output CSV file for matches")
    parser.add_argument("--embeddings-dir", type=str, default="embeddings",
                        help="Directory to save embeddings")
    parser.add_argument("--model", type=str, default="conservationxlabs/miewid-msv2",
                        help="MiewID model name from HuggingFace")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for embedding generation")
    parser.add_argument("--top-k", type=int, default=3,
                        help="Number of top matches to retrieve")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda or cpu)")
    parser.add_argument("--inat-csv", type=str, default="whale_results_with_url_filtered.csv",
                        help="Original iNaturalist CSV with metadata")
    parser.add_argument("--hw-metadata", type=str, default="happywhale_images/metadata.csv",
                        help="HappyWhale metadata CSV")
    
    args = parser.parse_args()
    
    # Check input directories
    if not os.path.exists(args.inat_dir):
        print(f"Error: iNaturalist directory not found: {args.inat_dir}")
        print("Run step 3 first to crop images.")
        return
    
    if not os.path.exists(args.hw_dir):
        print(f"Error: HappyWhale directory not found: {args.hw_dir}")
        print("Run step 3 first to crop images.")
        return
    
    # Create output directory
    os.makedirs(args.embeddings_dir, exist_ok=True)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model, transform, img_size = load_model(args.model, device)
    
    # Generate embeddings for iNaturalist images
    print(f"\n{'='*60}")
    print("Generating iNaturalist embeddings...")
    print(f"{'='*60}")
    inat_embeddings, inat_files = generate_embeddings(
        args.inat_dir, model, transform, device, args.batch_size
    )
    print(f"Generated {len(inat_embeddings)} iNaturalist embeddings")
    
    # Generate embeddings for HappyWhale images
    print(f"\n{'='*60}")
    print("Generating HappyWhale embeddings...")
    print(f"{'='*60}")
    hw_embeddings, hw_files = generate_embeddings(
        args.hw_dir, model, transform, device, args.batch_size
    )
    print(f"Generated {len(hw_embeddings)} HappyWhale embeddings")
    
    if len(inat_embeddings) == 0 or len(hw_embeddings) == 0:
        print("Error: No embeddings generated. Check your input directories.")
        return
    
    # Save embeddings
    embeddings_path = os.path.join(args.embeddings_dir, "embeddings.pt")
    torch.save({
        'inat_embeddings': inat_embeddings,
        'hw_embeddings': hw_embeddings,
        'inat_files': inat_files,
        'hw_files': hw_files,
    }, embeddings_path)
    print(f"\nSaved embeddings to: {embeddings_path}")
    
    # Find top matches
    print(f"\n{'='*60}")
    print(f"Finding top-{args.top_k} matches for each query...")
    print(f"{'='*60}")
    indices, distances = find_top_matches(inat_embeddings, hw_embeddings, args.top_k)
    
    # Compute full distance matrix for future use
    print("Computing full distance matrix...")
    all_embeddings = torch.cat([inat_embeddings, hw_embeddings], dim=0)
    distmat = compute_distance_matrix(all_embeddings, all_embeddings)
    
    distmat_path = os.path.join(args.embeddings_dir, "distmat.pt")
    torch.save(distmat, distmat_path)
    print(f"Saved distance matrix to: {distmat_path}")
    
    # Save results to CSV
    print(f"\n{'='*60}")
    print("Saving results...")
    print(f"{'='*60}")
    df = save_results_to_csv(
        inat_files, hw_files, indices, distances,
        args.inat_csv, args.hw_metadata, args.output
    )
    
    # Print summary
    print(f"\n{'='*60}")
    print("Summary:")
    print(f"{'='*60}")
    print(f"Total iNaturalist queries:  {len(inat_files)}")
    print(f"Total HappyWhale gallery:   {len(hw_files)}")
    print(f"Total matches found:        {len(df)}")
    print(f"Results saved to:           {args.output}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
