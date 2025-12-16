#!/usr/bin/env python3
"""
Master script to run the entire whale re-identification pipeline.

This script runs all four steps in sequence:
1. Download iNaturalist images
2. Download HappyWhale images
3. Crop images using Grounding DINO
4. Generate embeddings and find matches

You can also run individual steps by passing --step argument.
"""

import os
import sys
import argparse
import subprocess

def run_command(cmd, step_name):
    """
    Run a command and handle errors.
    
    Args:
        cmd: Command to run (list of strings)
        step_name: Name of the step (for logging)
        
    Returns:
        bool: True if successful, False otherwise
    """
    print(f"\n{'='*70}")
    print(f"Running: {step_name}")
    print(f"{'='*70}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*70}\n")
    
    try:
        result = subprocess.run(cmd, check=True)
        print(f"\n✓ {step_name} completed successfully\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {step_name} failed with error code {e.returncode}\n")
        return False
    except FileNotFoundError:
        print(f"\n✗ Script not found for {step_name}\n")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run whale re-identification pipeline")
    parser.add_argument("--step", type=str, choices=["1", "2", "3", "4", "all"],
                        default="all",
                        help="Which step to run (1-4 or 'all')")
    parser.add_argument("--inat-csv", type=str, default="whale_results_with_url_filtered.csv",
                        help="Path to iNaturalist CSV")
    parser.add_argument("--inat-dir", type=str, default="inat_images",
                        help="Directory for iNaturalist images")
    parser.add_argument("--hw-dir", type=str, default="happywhale_images",
                        help="Directory for HappyWhale images")
    parser.add_argument("--cropped-dir", type=str, default="cropped_images",
                        help="Directory for cropped images")
    parser.add_argument("--output", type=str, default="matches.csv",
                        help="Output CSV for matches")
    parser.add_argument("--hw-limit", type=int, default=None,
                        help="Limit HappyWhale images (for testing)")
    parser.add_argument("--top-k", type=int, default=3,
                        help="Number of top matches to retrieve")
    parser.add_argument("--skip-existing", action="store_true", default=True,
                        help="Skip already processed files")
    
    args = parser.parse_args()
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define steps
    steps = {
        "1": {
            "name": "Step 1: Download iNaturalist images",
            "script": os.path.join(script_dir, "1_download_inat_images.py"),
            "cmd": lambda: ["python", os.path.join(script_dir, "1_download_inat_images.py"),
                          "--csv", args.inat_csv,
                          "--output", args.inat_dir]
        },
        "2": {
            "name": "Step 2: Download HappyWhale images",
            "script": os.path.join(script_dir, "2_download_happywhale_images.py"),
            "cmd": lambda: ["python", os.path.join(script_dir, "2_download_happywhale_images.py"),
                          "--output", args.hw_dir] + 
                          (["--limit", str(args.hw_limit)] if args.hw_limit else [])
        },
        "3": {
            "name": "Step 3: Crop images with Grounding DINO",
            "script": os.path.join(script_dir, "3_crop_images.py"),
            "cmd": lambda: ["python", os.path.join(script_dir, "3_crop_images.py"),
                          "--inat-dir", args.inat_dir,
                          "--hw-dir", args.hw_dir,
                          "--output", args.cropped_dir]
        },
        "4": {
            "name": "Step 4: Generate embeddings and find matches",
            "script": os.path.join(script_dir, "4_embed_and_match.py"),
            "cmd": lambda: ["python", os.path.join(script_dir, "4_embed_and_match.py"),
                          "--inat-dir", os.path.join(args.cropped_dir, "inat"),
                          "--hw-dir", os.path.join(args.cropped_dir, "happywhale"),
                          "--output", args.output,
                          "--inat-csv", args.inat_csv,
                          "--hw-metadata", os.path.join(args.hw_dir, "metadata.csv"),
                          "--top-k", str(args.top_k)]
        }
    }
    
    # Determine which steps to run
    if args.step == "all":
        steps_to_run = ["1", "2", "3", "4"]
    else:
        steps_to_run = [args.step]
    
    # Print pipeline summary
    print("="*70)
    print("WHALE RE-IDENTIFICATION PIPELINE")
    print("="*70)
    print(f"Running steps: {', '.join(steps_to_run)}")
    print(f"iNaturalist CSV:     {args.inat_csv}")
    print(f"iNaturalist images:  {args.inat_dir}")
    print(f"HappyWhale images:   {args.hw_dir}")
    print(f"Cropped images:      {args.cropped_dir}")
    print(f"Output matches:      {args.output}")
    if args.hw_limit:
        print(f"HappyWhale limit:    {args.hw_limit}")
    print(f"Top K matches:       {args.top_k}")
    print("="*70)
    
    # Run steps
    success_count = 0
    fail_count = 0
    
    for step_num in steps_to_run:
        step = steps[step_num]
        
        # Check if script exists
        if not os.path.exists(step["script"]):
            print(f"\n✗ Script not found: {step['script']}")
            fail_count += 1
            continue
        
        # Run the step
        if run_command(step["cmd"](), step["name"]):
            success_count += 1
        else:
            fail_count += 1
            
            # Ask if user wants to continue
            if step_num != steps_to_run[-1]:
                response = input("\nStep failed. Continue to next step? (y/n): ")
                if response.lower() != 'y':
                    print("Pipeline stopped by user.")
                    break
    
    # Print final summary
    print("\n" + "="*70)
    print("PIPELINE SUMMARY")
    print("="*70)
    print(f"Steps completed successfully: {success_count}")
    print(f"Steps failed:                 {fail_count}")
    
    if fail_count == 0:
        print("\n✓ Pipeline completed successfully!")
        print(f"\nResults saved to: {args.output}")
    else:
        print("\n✗ Pipeline completed with errors.")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
