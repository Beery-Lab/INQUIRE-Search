# INQUIRE-Search: A Framework for Interactive Discovery in Large-Scale Biodiversity Databases
#### Edward Vendrow*, Julia Chae*, Rupa Kurinchi-Vendhan*, Isaac Eckert, Jazlynn Hall, Marta Jarzyna, Reymond Miyajima, Ruth Oliver, Laura Pollock, Lauren Schrack, Scott Yanco, Oisin Mac Aodha, Sara Beery

Community science platforms such as iNaturalist contain hundreds of millions of biodiversity images that capture ecological context on behaviors, interactions, phenology, and habitat. Yet ecological workflows to surface these data rely on metadata filtering or manual inspection, leaving this "secondary information" largely inaccessible at scale. We introduce INQUIRE-Search, an open-source system that uses natural language to enable scientists to rapidly search within an ecological image database for specific concepts, verify and export relevant observations, and use these outputs for downstream scientific analysis. Compared to traditional methods, INQUIRE-Search takes a fraction of the time, opening up new possibilities for scientific questions that can be explored. Through five case studies, we demonstrate the range of applications INQUIRE-Search can support, from seasonal variation in behavior across species to forest regrowth after wildfires. These examples demonstrate a new paradigm for interactive, efficient, and scalable scientific discovery that can begin to unlock previously inaccessible scientific value in large-scale biodiversity datasets. Finally, we highlight how AI-enabled discovery tools for science requires reframing aspects of the scientific process, including experiment design, data collection, survey effort, and uncertainty analysis.

## Case Studies

To access the retrieval data and analysis scripts for our case studies, see `case_studies`.


## Run our demo
We've set up a mini-demo over a smaller subset of iNaturalist from an existing publication presenting [INat2024](https://huggingface.co/datasets/evendrow/INQUIRE-Rerank). This version runs over a smaller set of data and only inludes the month filter as a proof of concept. We provide instructions for how to launch the demo locally below.

### Setting up the Conda Environment
If you do not have anaconda locally, install [conda](https://www.anaconda.com/docs/getting-started/miniconda/install#macos-2) 

### Setting up the Conda Environment
In your terminal, run the following commands:
```
conda create --name nat_img_ret python=3.8
conda activate nat_img_ret
```

### Installing Dependencies
Install the required packages with:
```
pip install -r requirements.txt
```

### Installing Node Dependencies for Frontend Server
This is a React webapp using the Next.js framework. This webapp was developed on node version `18.17.0`.

If you haven't already installed node, please do so from their website. It can be useful to use a node package manager such as `nvm`. Instructions on installing `nvm` are [here](https://github.com/nvm-sh/nvm?tab=readme-ov-file#installing-and-updating)

Once installed, move to the app folder

```bash
cd app
``` 

If using `nvm`, use the right node version:

```bash
nvm install 18.17.0
nvm use 18.17.0
```

Next, install dependencies:

```bash
npm install
```

### Launch Backend
Now from the main directory, launch the backend. To do this, run
```bash
python3 app.py
```
and in a new terminal,
```bash
cd app
npm run dev
```
for the frontend. This is a front-end server which assumes that the backend (flask) server is already running on port `5001`. Make sure that this backend server is already running.

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result. The page auto-updates as you edit the file.

## Want to run INQUIRE-Search on your own data?
To spin up an instance of INQUIRE-Search on your own data, you'll need to (1) replace our demo embeddings with your own and (2) recompute the search index. We include sample

### Generating Embeddings
The model choice and the path to the embeddings and the output .json to store the labels are defined within app.py.

```
import glob
import os
import json
from PIL import Image
from tqdm import tqdm
import numpy as np

from transformers import AutoProcessor, AutoModel
import torch
import torch.nn.functional as F


images = sorted(glob.glob(os.path.join(data_path, "images")))

model = AutoModel.from_pretrained("google/siglip-so400m-patch14-384").to("cuda")
processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")

emb_mapping = {}
embs = []
for idx, img in tqdm(enumerate(images), total=len(images)):
    image = Image.open(img).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to("cuda")

    with torch.no_grad():
        # 4. Get the image features specifically
        image_embeddings = model.get_image_features(**inputs)
        image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)
    emb_mapping[img] = idx
    embs.append(image_embeddings.cpu().numpy())

embs = np.vstack(embs)

np.save(os.path.join(data_path, "embeddings", "siglip_so400m_patch14_384_inat24_rerank_image_embeddings.npy"), embs)

with open(os.path.join(data_path, "mappings", "siglip_so400m_patch14_384_inat24_rerank_img_mapping.json"), "w") as f:
    json.dump(emb_mapping, f)
```

### Generating Metadata Files
If you have any metadata associated with your images, convert them to an memory-mapped (mmap) file for faster filtering and querying.

### Generating an Index
We use the [AutoFaiss](https://github.com/criteo/autofaiss) index for similarity search. To generate an kNN search index over your own data, run the following
```
from autofaiss import build_index

index, index_infos = build_index(
    embeddings=embs,
    index_path="knn.index",
    index_infos_path="index_infos.json",
    metric_type="ip",  
    save_on_disk=True
)
```

### A Note on Advanced Filters

The repository currently uses a small set of precomputed "filter" files (simple NumPy arrays and inverted indexes) to support species, month, and geospatial filtering on top of a FAISS index. These filter metadata are available in `static/mappings/.`

This is referenced in the following files:

- `run_search_index.py` sets `self.filters_dir` (default above) and loads the `.npy` files into memory to support the `EnhancedSearchIndex` filtered search paths.
- `services/search_service.py` contains helper methods that parse UI-provided filters (species, bounding box, months) and, when enabled, pass them to the search backends.

If you want to customize or disable filters, check these files: `config.py` (DATA_PATH and optional FILTERS_ENABLED flag), the `static/mappings/` directory (where files like `image_id_to_month.npy` or `month_to_image_rows.npz` live), `run_search_index.py` (filter loader and metadata->Faiss translation), and `services/search_service.py` (UI filter parsing). To disable filters quickly, rename or remove `static/mappings/` or set a `FILTERS_ENABLED = False` flag in `config.py` and restart the server.

### You're ready to search!
Follow the instructions above to launch your modified demo. 
