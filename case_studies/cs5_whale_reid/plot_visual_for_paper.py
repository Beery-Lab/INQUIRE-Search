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

output_dir = "for_paper"
os.makedirs(output_dir, exist_ok=True)

multispecies_model, multispecies_img_size, multispecies_img_transforms = multispecies_model_and_transforms(cluster=True)

whale_inquire_df = whale_inquire_data("whale_results_with_url_filtered.csv")
mod = "whales_bounded/"
whale_inquire_df["image"] = whale_inquire_df["image"].str.replace("inquire/whales/images/", mod)
inquire_annots = list(whale_inquire_df["annot"])
inquire_dataset = XAIDataset(whale_inquire_df, multispecies_img_size, multispecies_img_transforms)

dataset_name = datasets.humpback_whale_id.HumpbackWhaleID
whale_wd_df = wildlife_datasets_data(dataset_name, cluster=True)
whale_wd_df["image"] = os.path.join(CLUSTER_WD_PATH, dataset_name.__name__) + "/" + whale_wd_df["image"]
wd_annots = list(whale_wd_df["annot"])
wd_dataset = XAIDataset(whale_wd_df, multispecies_img_size, multispecies_img_transforms)

preprocess_dir = "inat_happywhale_embeddings_combined"

distmat = torch.load(os.path.join(preprocess_dir, 'distmat.pt'), weights_only=False)
embedding_dict = torch.load(os.path.join(preprocess_dir, 'embeddings.pt'), weights_only=False)
annots = list(embedding_dict["annots"])

def pad_to_size(img, target_width, target_height, pad_value=0):
    h, w = img.shape[:2]
    scale = min(target_width / w, target_height / h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    pad_top = (target_height - new_h) // 2
    pad_bottom = target_height - new_h - pad_top
    pad_left = (target_width - new_w) // 2
    pad_right = target_width - new_w - pad_left

    padding = ((pad_top, pad_bottom), (pad_left, pad_right)) if img.ndim == 2 else \
              ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0))
    
    return np.pad(resized, padding, mode='constant', constant_values=pad_value)

def load_image(annot, rectangle=False):
    if annot in inquire_annots:
        img, _, _ = inquire_dataset.get_image_pretransform(annot, resize=False)
        if rectangle:
            color = (0, 0, 255)
            thickness = 2
            img = cv2.rectangle(img,
                                (0,0),
                                (img.shape[1] - 1, img.shape[0] - 1),
                                color,
                                thickness)
    
    elif annot in wd_annots:
        img, _, _ = wd_dataset.get_image_pretransform(annot, resize=False)
        if rectangle:
            color = (255, 0, 0)
            thickness = 2
            img = cv2.rectangle(img,
                                (0,0),
                                (img.shape[1] - 1, img.shape[0] - 1),
                                color,
                                thickness)
    if not rectangle:
        color = (0, 0, 0)
        thickness = 2
        img = cv2.rectangle(img,
                            (0,0),
                            (img.shape[1] - 1, img.shape[0] - 1),
                            color,
                            thickness)
        
    else:
        raise Exception("annot not found?")

    return img

annot_0 = 26077575

idx_0 = annots.index(annot_0)
scores, indices = torch.topk(distmat[idx_0], 3, largest=False)
img_stack = []

for score, idx_1 in zip(scores, indices):
    annot_1 = annots[idx_1]
    img_0 = load_image(annot_0)
    img_1 = load_image(annot_1)

    print(img_0.shape, img_1.shape)
    print(type(img_0))

    img_1 = cv2.resize(img_1, tuple(reversed(img_0.shape[:2])))

    print(img_0.shape, img_1.shape)

    img_stack.append(cv2.hconcat((img_0, img_1)))

    Image.fromarray(img_0).save(os.path.join(output_dir, f"plain_{annot_0}.jpg"))
    Image.fromarray(img_1).save(os.path.join(output_dir, f"plain_{annot_1}.jpg"))

img_stack = cv2.vconcat(img_stack)
Image.fromarray(img_stack).save(os.path.join(output_dir, f"{annot_0}.jpg"))



