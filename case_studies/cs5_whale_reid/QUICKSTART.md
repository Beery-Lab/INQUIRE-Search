# Setup

```bash
# 1. Create and activate environment
conda create -n whale_reid python=3.10
conda activate whale_reid

# 2. Install PyTorch (adjust for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install Grounding DINO
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO && pip install -e . && cd ..

# 5. Download model weights
mkdir -p weights
wget -O weights/groundingdino_swint_ogc.pth \
  https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```

## Run the Pipeline

### Option 1: Full Pipeline
```bash
# Run all steps (downloads images, crops, embeds, and matches)
python run_pipeline.py
```

### Option 2: Test with Small Dataset
```bash
# Test with only 100 HappyWhale images
python run_pipeline.py --hw-limit 100
```

### Option 3: Run Individual Steps
```bash
# Step 1: Download iNaturalist images
python 1_download_inat_images.py

# Step 2: Download HappyWhale images
python 2_download_happywhale_images.py --limit 1000

# Step 3: Crop images
python 3_crop_images.py

# Step 4: Generate embeddings and match
python 4_embed_and_match.py
```

## Files for Analysis

We include pre-computed embeddings in `inat_happywhale_embeddings_combined` (you may need to download the linked files in `download_embeddings.md`) and intermediate output files as .csvs.

- **`analyze_results.py`** - Provides comprehensive analysis of matching results including accuracy metrics, distance distributions, per-rank statistics, and example match visualizations.

- **`plot_top_matches.py`** - Creates visual plots showing query images alongside their top matching results from the database.

- **`plot_visual_for_paper.py`** - Generates publication-quality visualizations of matching results for research papers.

- **`filter_multi_query_neighbors.py`** - Filters matching results when multiple query images map to the same individuals.

- **`list_identified_inat_whales.py`** - Creates a list of iNaturalist whale observations that have been identified.

- **`process_labels.py`** - Processes and standardizes label data from various sources.

- **`look_at_data.py`** - Quick data exploration script for examining dataset contents and statistics.

- **`analysis.ipynb`** - Interactive Jupyter notebook for exploring results, visualizing matches, and computing metrics.