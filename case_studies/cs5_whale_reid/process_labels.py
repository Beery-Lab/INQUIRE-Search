from wildlife_datasets import datasets
import os
import torch
import sys
import cv2
from PIL import Image, ImageOps
import numpy as np
from tqdm import tqdm
import pandas as pd

sys.path.append("../..")
sys.path.append("../../wbia-plugin-miew-id")
sys.path.append('../../wildlife-embeddings/')
from loaders import whale_inquire_data, wildlife_datasets_data, CLUSTER_WD_PATH, multispecies_model_and_transforms
from xai_dataset import XAIDataset

output_dir = "matches"
os.makedirs(output_dir, exist_ok=True)

whale_inquire_df = whale_inquire_data("whale_results_with_url_filtered.csv")
mod = "whales_bounded/"
whale_inquire_df["image"] = whale_inquire_df["image"].str.replace("inquire/whales/images/", mod)
inquire_annots = list(whale_inquire_df["annot"])

dataset_name = datasets.humpback_whale_id.HumpbackWhaleID
whale_wd_df = wildlife_datasets_data(dataset_name, cluster=True)
whale_wd_df["image"] = os.path.join(CLUSTER_WD_PATH, dataset_name.__name__) + "/" + whale_wd_df["image"]
wd_annots = list(whale_wd_df["annot"])
wd_names = set(whale_wd_df["name"])
print("IS IT??", "unknown" in wd_names)

preprocess_dir = "inat_happywhale_embeddings_combined"

distmat = torch.load(os.path.join(preprocess_dir, 'distmat.pt'), weights_only=False)
embedding_dict = torch.load(os.path.join(preprocess_dir, 'embeddings.pt'), weights_only=False)
annots = list(embedding_dict["annots"])
names = list(embedding_dict["labels"])

manual_labels = pd.read_csv("manual_labels.csv")
print(len(manual_labels))

n_matches = 0
n_wd_matches = 0
wd_match_names = {}

for annot_0 in tqdm(whale_inquire_df["annot"], total=len(whale_inquire_df)):
    idx_0 = annots.index(annot_0)
    scores, indices = torch.topk(distmat[idx_0], 3, largest=False)
    img_stack = []

    row = manual_labels[manual_labels["inquire_annot"] == annot_0].iloc[0]
    matches = (row["First"], row["Second"], row["Third"])
    matches = [match=="y" for match in matches]

    is_match = False
    is_wd_match = False

    for score, idx_1, match in zip(scores, indices, matches):
        annot_1 = annots[idx_1]
        name_1 = names[idx_1]
        if match:
            is_match = True
            if annot_1 in wd_annots:
                is_wd_match = True
                wd_match_names[name_1] = wd_match_names.get(name_1, 0) + 1
    
    if is_match:
        n_matches += 1
    if is_wd_match:
        n_wd_matches += 1
    
print("Total with Matches: ", n_matches)
print("Total with WD Matches: ", n_wd_matches)
print(wd_match_names)
print(len(wd_match_names))
#print("Total Matched WD Individuals", len(wd_match_names))



