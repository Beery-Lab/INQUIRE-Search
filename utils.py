import os
from pathlib import Path
import glob
import faiss
import numpy as np
import pandas as pd
import json
import torch

# From clip-retrieval
# https://github.com/rom1504/clip-retrieval/blob/main/clip_retrieval/clip_back.py#L521
class ParquetMetadataProvider:
    """The parquet metadata provider provides metadata from contiguous ids using parquet"""

    def __init__(self, parquet_folder):
        data_dir = Path(parquet_folder)
        self.metadata_df = pd.concat(
            pd.read_parquet(parquet_file) for parquet_file in sorted(data_dir.glob("*.parquet"))
        )
    def get(self, ids, cols=None):
        if cols is None:
            cols = self.metadata_df.columns.tolist()
        else:
            cols = list(set(self.metadata_df.columns.tolist()) & set(cols))

        return [self.metadata_df[i : (i + 1)][cols].to_dict(orient="records")[0] for i in ids]

    def __len__(self):
        return len(self.metadata_df)
    
class MemoryMappedMetadataProvider:
    def __init__(self, metadata_dir):
        """Initialize metadata provider."""
        self.metadata_dir = Path(metadata_dir)
        # We'll attempt to detect the correct dtype for the memmap file.
        # Historically different preprocessing scripts wrote different structured dtypes
        # (for example a simple [('photo_id','S64'),('license','S64')] or a larger
        # record with numeric fields). Try a few common candidates and pick the one
        # that successfully memory-maps the file.
        self.dtype = None
        
        # Cache for opened memory maps
        self.mmap_cache = {}
        
        # Find all chunks and their sizes
        self.chunk_files = sorted(self.metadata_dir.glob("metadata_*.mmap"))
        self.chunks = [int(f.stem.split('_')[1]) for f in self.chunk_files]
        
        if not self.chunks:
            raise ValueError(f"No metadata files found in {self.metadata_dir}")
        # Detect dtype using the first chunk file
        first_chunk_path = self.chunk_files[0]
        # Candidate dtypes (order matters: prefer the richer dtype first)
        candidate_dtypes = [
            np.dtype([
                ('photo_id', np.int64),
                ('photo_uuid', 'S36'),
                ('observer_id', np.int64),
                ('license', 'S64'),
                ('width', np.int32),
                ('height', np.int32)
            ]),
            # older/simpler preprocessing wrote two string fields (S64 each)
            np.dtype([('photo_id', 'S64'), ('license', 'S64')]),
        ]

        detect_success = False
        detection_errors = []
        for cand in candidate_dtypes:
            try:
                # try to create a memmap with this dtype
                m = np.memmap(first_chunk_path, dtype=cand, mode='r')
                # accessing len(m) will raise if sizes don't align
                _ = len(m)
                # success
                self.dtype = cand
                detect_success = True
                break
            except Exception as e:
                detection_errors.append((cand, str(e)))

        if not detect_success:
            # helpful diagnostic before raising
            size = first_chunk_path.stat().st_size
            rem_table = []
            for cand in candidate_dtypes:
                rem_table.append((cand, size % cand.itemsize))
            msg_lines = [f"Failed to detect memmap dtype for {first_chunk_path}",
                         f"file size: {size} bytes"]
            for cand, rem in rem_table:
                msg_lines.append(f" candidate dtype itemsize={cand.itemsize} remainder={rem}")
            for cand, err in detection_errors:
                msg_lines.append(f" tried {cand} -> {err}")
            raise ValueError("\n".join(msg_lines))

        # Get size of each chunk
        self.chunk_sizes = {}
        self.total_entries = 0
        for chunk_idx, chunk_file in zip(self.chunks, self.chunk_files):
            mmap = self._get_mmap(chunk_idx)
            self.chunk_sizes[chunk_idx] = len(mmap)
            self.total_entries += len(mmap)
    
    def _get_mmap(self, chunk_idx):
        """Get or create memory map for given chunk index"""
        if chunk_idx not in self.mmap_cache:
            chunk_path = self.metadata_dir / f"metadata_{chunk_idx}.mmap"
            self.mmap_cache[chunk_idx] = np.memmap(chunk_path, dtype=self.dtype, mode='r')
        return self.mmap_cache[chunk_idx]
    
    def get(self, idx):
        """Get metadata for given global index"""
        if not 0 <= idx < self.total_entries:
            raise IndexError(f"Index {idx} out of bounds")
            
        # Find which chunk contains this index
        current_count = 0
        for chunk_idx in self.chunks:
            chunk_size = self.chunk_sizes[chunk_idx]
            if current_count + chunk_size > idx:
                local_idx = idx - current_count
                mmap = self._get_mmap(chunk_idx)
                return mmap[local_idx]
            current_count += chunk_size
    
    
class EmbeddingProvider:

    def __init__(self, embedding_folder):
        data_dir = Path(embedding_folder)
        # Accept either a directory containing .npy shard files or a single
        # .npy file path. This makes the loader robust to Config pointing to
        # either a folder or a single file.
        if data_dir.is_file() and data_dir.suffix == '.npy':
            emb_files = [data_dir]
        else:
            emb_files = sorted(data_dir.glob("*.npy"))

        # Load each .npy (memory-mapped) file
        self.embs = []
        for f in emb_files:
            try:
                arr = np.load(f, mmap_mode='r')
            except Exception:
                # fall back to non-mmap load if mmap failed for some reason
                arr = np.load(f, allow_pickle=False)
            self.embs.append(arr)
        self.size = sum([len(emb) for emb in self.embs])

        print("Loaded memory-mapped embeddings with total:", self.size)

    def get_one(self, idx):
        if idx < 0:
            raise ValueError("Index must be nonnegative")
        for emb in self.embs:
            if idx >= len(emb):
                idx -= len(emb)
            else:
                return emb[idx]
        raise ValueError(f"index not found: {idx}")

    def get(self, idxs):
        results = []
        for idx in idxs:
            results.append(self.get_one(idx))
        return np.asarray(results)


class NaiveKNNIndex:
    def __init__(self, data_path, model_name, tiny=False, use_gpu=False):
        emb_files_paths = sorted(glob.glob(os.path.join(data_path, 'embs', f'{model_name}/img_emb/img_emb_*.npy')))
        
        self.device = 'cpu'
        if use_gpu:
            if torch.cuda.is_available():
                self.device == 'cuda'
            elif torch.backends.mps.is_available():
                self.device = 'mps'
            print("[knn] Using device for Naive KNN Index: ", self.device)

        print('[knn] Initializing index with', len(emb_files_paths), '.npy files')
        self.all_embs = []
        for emb_file in emb_files_paths:
            embs = torch.from_numpy(np.load(emb_file, allow_pickle=True)).to(torch.float16).to(self.device)
            if tiny:
                embs = embs[:10000]
                self.all_embs.append(embs)
                break
            self.all_embs.append(embs)
        print('[knn] ... Done loading')

    def search(self, query, k):
        query = torch.from_numpy(query.squeeze()).to(torch.float16).to(self.device)
        assert len(query.shape) == 1, "Embedding should be 1-dimensional"

        all_scores = []
        for embs in self.all_embs:
            all_scores.append(embs @ query)
        
        # scores = torch.from_numpy(np.concatenate(all_scores))
        print('[knn]', all_scores[0].shape)
        scores = torch.cat(all_scores).cpu()
        assert len(scores.shape) == 1, "???"

        indices = torch.flip(scores.argsort(), dims=(0,))[:k]
        distances = scores[indices]
        return [distances], [indices]


def load_inat_metadata(inat_metadata_path):
    # Use HDf5 index, which is much faster to load.
    if os.path.exists(inat_metadata_path+".parquet"):
        print("Loading metadata file from parquet")
        inat_metadata = pd.read_parquet(inat_metadata_path + ".parquet")
    elif os.path.exists(inat_metadata_path + ".csv"):
        print("No parquet metadata found. Will create one from CSV")
        inat_metadata = pd.read_csv(inat_metadata_path + ".csv")
        inat_metadata.to_parquet(inat_metadata_path + ".parquet")
    else:
        raise ValueError("No metadata file found at " + inat_metadata_path + ".csv")

    return inat_metadata

def load_species_to_index_map(data_path, metadata_name):
    label_to_index_map = {}
    label_to_index_path = os.path.join(data_path, metadata_name+"--label_to_index_map.json")
    print(label_to_index_path)
    if not os.path.exists(label_to_index_path):
        print("WARNING: No label-to-index map found for this dataset! Search by class will not work.\n"
            "Please run `python scripts/create_class_to_index_map.py` to generate it.")
        label_to_index_map = {}
    else:
        with open(label_to_index_path, 'r') as f:
            label_to_index_map = json.load(f)
    return label_to_index_map

def load_species_to_common_name_map(data_path="data/"):
    with open(os.path.join(data_path, "map_species_to_common_name.json"), "r") as file:
        data = json.load(file)
    return data