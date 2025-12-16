"""
Configuration file for the whale re-identification pipeline.
Edit these settings to customize the pipeline behavior.
"""

import os

# ============================================================
# PATHS
# ============================================================

# Input CSV with iNaturalist query images
INAT_CSV = "whale_results_with_url_filtered.csv"

# Output directories
INAT_IMAGES_DIR = "inat_images"
HAPPYWHALE_IMAGES_DIR = "happywhale_images"
CROPPED_IMAGES_DIR = "cropped_images"
EMBEDDINGS_DIR = "embeddings"
OUTPUT_CSV = "matches.csv"

# Grounding DINO paths (adjust based on your installation)
GROUNDING_DINO_CONFIG = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_WEIGHTS = "weights/groundingdino_swint_ogc.pth"

# ============================================================
# GROUNDING DINO SETTINGS
# ============================================================

# Detection prompt for whale tails
DINO_PROMPT = "whale tail"

# Detection thresholds
DINO_BOX_THRESHOLD = 0.35
DINO_TEXT_THRESHOLD = 0.25

# ============================================================
# EMBEDDING SETTINGS
# ============================================================

# MiewID model (from HuggingFace)
MIEWID_MODEL = "conservationxlabs/miewid-msv2"

# Image size for embeddings
EMBEDDING_IMAGE_SIZE = (440, 440)

# Batch size for embedding generation
EMBEDDING_BATCH_SIZE = 16

# ============================================================
# MATCHING SETTINGS
# ============================================================

# Number of top matches to retrieve for each query
TOP_K = 3

# ============================================================
# DATASET SETTINGS
# ============================================================

# HappyWhale dataset name (from wildlife-datasets)
HAPPYWHALE_DATASET = "HumpbackWhaleID"

# Maximum number of HappyWhale images to download (None = all)
# Set to a smaller number for testing (e.g., 100)
HAPPYWHALE_LIMIT = None

# Wildlife datasets path (will be created if doesn't exist)
WILDLIFE_DATASETS_PATH = "wildlife_datasets_data"

# ============================================================
# METADATA COLUMNS
# ============================================================

# Column names in input CSV
PHOTO_ID_COL = "photo_id"
DOWNLOAD_URL_COL = "download_url"
SPECIES_COL = "species"
INAT_URL_COL = "inat_url"

# Label/identity column for evaluation
LABEL_COL = "individual_id"

# ============================================================
# PROCESSING SETTINGS
# ============================================================

# Number of worker threads for data loading
NUM_WORKERS = 4

# Device for computation ('cuda' or 'cpu')
DEVICE = "cuda"

# Skip already processed files
SKIP_EXISTING = True

# Download delay (seconds) to avoid overwhelming servers
DOWNLOAD_DELAY = 0.5
