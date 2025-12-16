import sys
import os
import torch
from wildlife_datasets import datasets
import numpy as np

sys.path.append("../..")
sys.path.append("../../wbia-plugin-miew-id")
sys.path.append('../../wildlife-embeddings/')
from loaders import whale_inquire_data, wildlife_datasets_data

# load the INQUIRE data as a pandas df
whale_inquire_df = whale_inquire_data("whale_results_with_url_filtered.csv",
                                      image_dir="whales_bounded")

# load the HappyWhale data as a pandas df
dataset_name = datasets.humpback_whale_id.HumpbackWhaleID
whale_wd_df = wildlife_datasets_data(dataset_name, cluster=True)

# load the distance matrix as a 2D torch tensor
preprocess_dir = "inat_happywhale_embeddings_combined"
distmat = torch.load(os.path.join(preprocess_dir, 'distmat.pt'), weights_only=False)

# load embedding_dict to get the annot list that matches against the distance matrix
embedding_dict = torch.load(os.path.join(preprocess_dir, 'embeddings.pt'), weights_only=False)
annots = list(embedding_dict["annots"])
labels = list(embedding_dict["labels"])

# modify the distance matrix to ignore INQUIRE <=> INQUIRE matches
inquire_indices = [annots.index(a) for a in whale_inquire_df["annot"]]
for i in inquire_indices:
    for j in inquire_indices:
        distmat[i][j] = np.inf

# look at top matches for a given INQUIRE annot
inquire_annot = 2562748
inquire_idx = annots.index(inquire_annot)
_, happywhale_indices = torch.topk(distmat[inquire_idx], 3, largest=False)
happywhale_annots = [annots[i] for i in happywhale_indices]
print(happywhale_annots)
