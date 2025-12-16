import numpy as np
import faiss
from flask import Flask, request, jsonify
import os
from config import Config

class EnhancedSearchIndex:
    """
    Manages geospatial + other filtering on top of a Faiss index.
    """

    def __init__(self, faiss_index_path, emb_to_inat_index=None):
        """
        :param faiss_index_path: Path to a Faiss index file.
        :param inat_metadata: A DataFrame with columns for lat, lng, and so on.
        :param emb_to_inat_index: A NumPy array mapping Faiss vector IDs -> row indices in inat_metadata 
                                  (or the reverse mapping, depending on how you set up Faiss).
        """

        # Load the FAISS index
        import time
        t_load_start = time.time()
        print(f"[EnhancedSearchIndex.__init__] Loading FAISS index from: {faiss_index_path}", flush=True)
        
        try:
            # Try loading without mmap first to see if that's the issue
            print(f"[EnhancedSearchIndex.__init__] Attempting regular load (not mmap)...", flush=True)
            self.index = faiss.read_index(faiss_index_path)
            print(f"[EnhancedSearchIndex.__init__] Regular load successful", flush=True)
        except Exception as e:
            print(f"[EnhancedSearchIndex.__init__] Regular load failed: {e}, trying mmap...", flush=True)
            try:
                self.index = faiss.read_index(faiss_index_path, faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY)
                print(f"[EnhancedSearchIndex.__init__] Mmap load successful", flush=True)
            except Exception as e2:
                print(f"[EnhancedSearchIndex.__init__] Both load methods failed: {e2}", flush=True)
                raise
        
        t_load_elapsed = time.time() - t_load_start
        print(f"[EnhancedSearchIndex.__init__] FAISS index loaded in {t_load_elapsed:.3f}s", flush=True)
        print(f"[EnhancedSearchIndex.__init__] Index type: {type(self.index).__name__}", flush=True)
        print(f"[EnhancedSearchIndex.__init__] Index ntotal: {self.index.ntotal}", flush=True)
        print(f"[EnhancedSearchIndex.__init__] Index dimension: {self.index.d}", flush=True)
        
        # Check if index is trained
        if hasattr(self.index, 'is_trained'):
            print(f"[EnhancedSearchIndex.__init__] Index is_trained: {self.index.is_trained}", flush=True)
            if not self.index.is_trained:
                print(f"[EnhancedSearchIndex.__init__] WARNING: Index is not trained!", flush=True)
        
        import sys
        sys.stdout.flush()
        
        # Optimize FAISS index for search speed
        try:
            # If it's an IVF index, set nprobe for faster search
            if hasattr(self.index, 'nprobe'):
                old_nprobe = self.index.nprobe
                self.index.nprobe = 1  # Start with fastest setting
                print(f"[EnhancedSearchIndex.__init__] Set nprobe from {old_nprobe} to {self.index.nprobe} for faster search", flush=True)
            
            # If it's an HNSW index, set efSearch
            if hasattr(self.index, 'hnsw') and hasattr(self.index.hnsw, 'efSearch'):
                old_ef = self.index.hnsw.efSearch
                self.index.hnsw.efSearch = 8  # Start with fastest setting
                print(f"[EnhancedSearchIndex.__init__] Set HNSW efSearch from {old_ef} to {self.index.hnsw.efSearch} for faster search", flush=True)
        except Exception as e:
            print(f"[EnhancedSearchIndex.__init__] Could not optimize index parameters: {e}", flush=True)
        
        sys.stdout.flush()
        
        # If Faiss index IDs match the row indices in inat_metadata exactly,
        # then we can just do bounding-box filtering with row indices.
        # Otherwise, you'll need to map between "Faiss ID" <-> "DataFrame row index".
        # For example, if I is a result from self.index.search, then inat row = emb_to_inat_index[I].
        self.emb_to_inat_index = emb_to_inat_index

        # Determine filters directory strictly from explicit config/env paths.
        # Priority:
        # 1) FILTERS_DIR env var
        # 2) Config.FILTERS_DIR attribute
        # 3) dirname of any explicit Config.*_PATH attributes (if they exist and its dir exists)
        # 4) local fallback 'static/rerank_prepare'
        env_filters = os.environ.get('FILTERS_DIR')
        if env_filters and os.path.exists(env_filters):
            self.filters_dir = env_filters
        elif hasattr(Config, 'FILTERS_DIR') and Config.FILTERS_DIR and os.path.exists(Config.FILTERS_DIR):
            self.filters_dir = Config.FILTERS_DIR
        else:
            # Look for explicit configured file paths in Config and use their directory
            candidate_dirs = []
            for attr in ('IMAGE_ID_TO_TAXON_ID_PATH', 'MONTH_INDEX_PATH', 'PHOTO_ID_TO_EXT_MAP_PATH',
                         'INDEX_PATH', 'METADATA_PATH', 'EMBEDDINGS_PATH'):
                if hasattr(Config, attr):
                    val = getattr(Config, attr)
                    if val:
                        d = os.path.dirname(val) if os.path.isabs(val) or os.path.sep in val else os.path.dirname(os.path.join(os.path.dirname(__file__), val))
                        if d and os.path.exists(d):
                            candidate_dirs.append(d)

            if candidate_dirs:
                # Prefer the first configured dir
                self.filters_dir = candidate_dirs[0]
            else:
                # Local repo fallback
                local_fallback = os.path.join(os.path.dirname(__file__), 'static', 'rerank_prepare')
                if os.path.exists('static/rerank_prepare'):
                    self.filters_dir = 'static/rerank_prepare'
                else:
                    self.filters_dir = local_fallback
        print(f"Using filters directory: {self.filters_dir}")

        
        self.geo_index = None
        self._build_geo_index()
        self._build_species_index()
        self._build_month_index()
        # Try to load a mapping that maps Faiss vector positions -> metadata row indices
        # This is used when Faiss IDs do not equal the metadata row indices.
        self.faiss_to_metadata = None
        self.metadata_to_faiss = None
        try:
            map_path = getattr(Config, 'PHOTO_ID_TO_EXT_MAP_PATH', None)
            if map_path and os.path.exists(map_path):
                arr = np.load(map_path, allow_pickle=True)
                # Expect an array of ints (metadata row indices) indexed by Faiss idx
                try:
                    # Be tolerant: mapping file may be an int array or an object array
                    # Try direct integer conversion first
                    try:
                        arr_int = np.asarray(arr).astype(np.int64)
                    except Exception:
                        # Fallback: try to parse each element to int (handles tuples like (url, id))
                        parsed = []
                        for x in arr.tolist() if hasattr(arr, 'tolist') else list(arr):
                            val = None
                            try:
                                if isinstance(x, (list, tuple)) and len(x) > 1:
                                    cand = x[1]
                                else:
                                    cand = x
                                val = int(cand)
                            except Exception:
                                val = None
                            parsed.append(val)
                        if all(v is not None for v in parsed):
                            arr_int = np.asarray(parsed, dtype=np.int64)
                        else:
                            raise ValueError("mapping elements not convertible to int")
                    self.faiss_to_metadata = arr_int
                    # Build reverse mapping (metadata row -> faiss id)
                    # If the mapping is a permutation, we can do argsort; else build dict
                    if arr_int.size > 0:
                        # If arr_int is a permutation of 0..N-1, argsort works
                        if set(arr_int.tolist()) == set(range(arr_int.size)):
                            inv = np.argsort(arr_int)
                            self.metadata_to_faiss = inv
                        else:
                            # fallback dict
                            self.metadata_to_faiss = {int(m): int(f) for f, m in enumerate(arr_int.tolist())}
                    print(f"Loaded faiss->metadata mapping from {map_path}")
                except Exception:
                    print(f"Loaded mapping from {map_path} but could not interpret as integer array; ignoring mapping.")
        except Exception:
            pass


    def _build_species_index(self):
        taxon_path = os.path.join(self.filters_dir, 'map_taxon_id_to_image_ids.npy')
        descendants_path = os.path.join(self.filters_dir, 'taxon_descendants.npy')
        if os.path.exists(taxon_path) and os.path.exists(descendants_path):
            self.taxon_index = np.load(taxon_path, allow_pickle=True).item()
            self.taxon_descendants = np.load(descendants_path, allow_pickle=True).item()
            print("Loaded species inverted index.")
        else:
            self.taxon_index = None
            self.taxon_descendants = None
            print(f"Species index files not found in {self.filters_dir}; species filtering disabled.")

    def _build_month_index(self):
        month_path = os.path.join(self.filters_dir, 'image_id_to_month.npy')
        if os.path.exists(month_path):
            self.month_index = np.load(month_path)
            print("Loaded month index.")
        else:
            self.month_index = None
            print(f"Month index not found in {self.filters_dir}; month filtering disabled.")


    ##############################
    # Functions for array geo index
    def _encode_lat_lon(self, lat, lon):
        # Shift latitude by +90° and scale
        lat_enc = np.uint16(np.round((lat + 90.0) * 100))
        # Shift longitude by +180° and scale
        lon_enc = np.uint16(np.round((lon + 180.0) * 100))
        return (lat_enc, lon_enc)

    def _decode_lat_lon(self, lat_enc, lon_enc):
        lat = (lat_enc / 100.0) - 90.0
        lon = (lon_enc / 100.0) - 180.0
        return (lat, lon)

    def _build_geo_index(self):
        geo_path = os.path.join(self.filters_dir, 'image_id_to_latlong_int.npy')
        if os.path.exists(geo_path):
            self.geo_index = np.load(geo_path)
            print("Loaded geo index.")
        else:
            self.geo_index = None
            print(f"Geo index not found in {self.filters_dir}; geo filtering disabled.")
    ##############################

    def _get_ids_in_bounding_box(self, min_lat, max_lat, min_lng, max_lng):
        """
        Return the 'row indices' (which must match the Faiss ID or be mappable to the Faiss ID)
        for images whose lat/lng is within the bounding box.
        """
        if self.geo_index is None:
            raise ValueError("Geo index not built. Call _build_geo_index() first.")

        print('[geo] Filtering by bounding box:', min_lat, max_lat, min_lng, max_lng)
        min_lat_enc, min_lon_enc = self._encode_lat_lon(min_lat, min_lng)
        max_lat_enc, max_lon_enc = self._encode_lat_lon(max_lat, max_lng)
        
        # Optimized: use numpy's efficient boolean indexing
        # Pre-extract columns to avoid repeated indexing
        lats = self.geo_index[:, 0]
        lons = self.geo_index[:, 1]
        
        # Combine masks efficiently
        mask = ((lats >= min_lat_enc) & (lats <= max_lat_enc) & 
                (lons >= min_lon_enc) & (lons <= max_lon_enc))
        subset_ids = np.flatnonzero(mask)  # More efficient than np.where()[0]

        print(f"[geo] Found {len(subset_ids)} images in bounding box")
        return subset_ids

    def _is_hnsw_index(self):
        """Return True if the loaded Faiss index is an HNSW-style index."""
        try:
            cls_name = type(self.index).__name__
            return 'IndexHNSW' in cls_name or 'HNSW' in cls_name
        except Exception:
            return False

    def _metadata_to_faiss_ids(self, metadata_indices):
        """Convert metadata row indices -> Faiss vector IDs if mapping available.

        :param metadata_indices: 1D array-like of metadata row indices
        :returns: NumPy array of Faiss IDs (int64)
        """
        meta_arr = np.asarray(metadata_indices).astype(np.int64)
        if self.metadata_to_faiss is None:
            # No mapping available; assume metadata indices == Faiss IDs
            return meta_arr

        # metadata_to_faiss may be an ndarray (argsort result) or a dict
        if isinstance(self.metadata_to_faiss, np.ndarray):
            # ensure indices are within range
            mask = (meta_arr >= 0) & (meta_arr < self.metadata_to_faiss.shape[0])
            faiss_ids = self.metadata_to_faiss[meta_arr[mask]]
            return np.asarray(faiss_ids, dtype=np.int64)
        else:
            # dict
            faiss_list = []
            for m in meta_arr.tolist():
                f = self.metadata_to_faiss.get(int(m), None)
                if f is not None:
                    faiss_list.append(int(f))
            return np.asarray(faiss_list, dtype=np.int64)

    def _search_and_filter_post(self, query_vec, k, valid_ids, nprobe=10, multiplier=10):
        """
        Fallback search: For HNSW indices that don't support ID selectors,
        we use different strategies based on the size of the valid subset.
        """
        import time
        import sys
        t_start = time.time()
        print(f"[_search_and_filter_post] Starting with {len(valid_ids)} valid IDs, k={k}", flush=True)
        sys.stdout.flush()
        
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)

        num_valid = len(valid_ids)
        
        # Set OpenMP threads to 1 for speed
        try:
            old_threads = faiss.omp_get_max_threads()
            faiss.omp_set_num_threads(1)
            print(f"[_search_and_filter_post] Set OMP threads from {old_threads} to 1", flush=True)
        except Exception as e:
            print(f"[_search_and_filter_post] Could not set OMP threads: {e}", flush=True)
        
        # If the valid subset is very large (>50% of index), just do normal search
        # and post-filter - it's faster than reconstructing so many vectors
        selectivity = num_valid / max(self.index.ntotal, 1)
        
        # Be even more aggressive with large subsets - use simple search for >30% or >3000 images
        if selectivity > 0.3 or num_valid > 3000:
            # Large subset: use simple post-filter with minimal multiplier
            print(f"[fallback] Large subset ({num_valid} ids, {selectivity:.1%}), using minimal post-filter search", flush=True)
            # For very large subsets, use an even smaller multiplier
            if num_valid > 10000:
                k_try = min(int(k * 1.2), 300)  # Ultra-conservative for huge subsets
            else:
                k_try = min(int(k * 1.5), 500)  # Conservative multiplier
            
            print(f"[fallback] Searching for k_try={k_try} results...", flush=True)
            sys.stdout.flush()
            t_search_start = time.time()
            D_all, I_all = self.index.search(query_vec, k_try)
            t_search_elapsed = time.time() - t_search_start
            print(f"[fallback] Search took {t_search_elapsed:.3f}s", flush=True)
            
            # Convert to set for fast lookup - use frozenset for large arrays
            print(f"[fallback] Converting {num_valid} IDs to set...", flush=True)
            t_set_start = time.time()
            valid_set = set(valid_ids.tolist()) if num_valid < 100000 else set(int(x) for x in valid_ids)
            t_set_elapsed = time.time() - t_set_start
            print(f"[fallback] Set creation took {t_set_elapsed:.3f}s", flush=True)
            
            print(f"[fallback] Filtering results...", flush=True)
            t_filter_start = time.time()
            filtered_I = []
            filtered_D = []
            for dist_row, idx_row in zip(D_all, I_all):
                row_I = []
                row_D = []
                for d, idx in zip(dist_row, idx_row):
                    if int(idx) in valid_set:
                        row_I.append(int(idx))
                        row_D.append(float(d))
                        if len(row_I) >= k:
                            break
                while len(row_I) < k:
                    row_I.append(-1)
                    row_D.append(float('inf'))
                filtered_I.append(np.array(row_I, dtype=np.int64))
                filtered_D.append(np.array(row_D, dtype=np.float32))
            
            t_filter_elapsed = time.time() - t_filter_start
            print(f"[fallback] Filtering took {t_filter_elapsed:.3f}s", flush=True)
            t_total = time.time() - t_start
            print(f"[fallback] Total post-filter time: {t_total:.3f}s", flush=True)
            sys.stdout.flush()
            
            return np.vstack(filtered_D), np.vstack(filtered_I)
        
        # Small subset: build temporary index for exact search
        print(f"[fallback] Small subset ({num_valid} ids, {selectivity:.1%}), building temporary index")
        
        valid_ids_list = valid_ids.tolist()
        
        # Try to reconstruct vectors from the index
        try:
            # Get dimension from query
            d = query_vec.shape[1] if query_vec.ndim > 1 else query_vec.shape[0]
            
            # Build a flat index with just the filtered vectors
            subset_vectors = np.zeros((len(valid_ids_list), d), dtype=np.float32)
            for i, vid in enumerate(valid_ids_list):
                try:
                    subset_vectors[i] = self.index.reconstruct(int(vid))
                except Exception:
                    # If reconstruct fails, this approach won't work
                    print("[fallback] Cannot reconstruct vectors, falling back to brute force post-filter")
                    # Fall back to the old approach but with minimal over-fetch
                    k_try = min(k * 3, len(valid_ids_list), 1000)
                    D_all, I_all = self.index.search(query_vec, k_try)
                    valid_set = set(valid_ids_list)
                    
                    filtered_I = []
                    filtered_D = []
                    for dist_row, idx_row in zip(D_all, I_all):
                        row_I = []
                        row_D = []
                        for d, idx in zip(dist_row, idx_row):
                            if int(idx) in valid_set:
                                row_I.append(int(idx))
                                row_D.append(float(d))
                                if len(row_I) >= k:
                                    break
                        while len(row_I) < k:
                            row_I.append(-1)
                            row_D.append(float('inf'))
                        filtered_I.append(np.array(row_I, dtype=np.int64))
                        filtered_D.append(np.array(row_D, dtype=np.float32))
                    
                    return np.vstack(filtered_D), np.vstack(filtered_I)
            
            # Create temporary flat index for exact search on subset
            temp_index = faiss.IndexFlatIP(d)  # Use inner product (same as your main index likely uses)
            temp_index.add(subset_vectors)
            
            # Search in the subset
            D_sub, I_sub = temp_index.search(query_vec, min(k, len(valid_ids_list)))
            
            # Map back to original IDs
            result_I = []
            result_D = []
            for dist_row, idx_row in zip(D_sub, I_sub):
                row_I = []
                row_D = []
                for d, idx in zip(dist_row, idx_row):
                    if idx >= 0 and idx < len(valid_ids_list):
                        row_I.append(valid_ids_list[idx])
                        row_D.append(float(d))
                while len(row_I) < k:
                    row_I.append(-1)
                    row_D.append(float('inf'))
                result_I.append(np.array(row_I[:k], dtype=np.int64))
                result_D.append(np.array(row_D[:k], dtype=np.float32))
            
            print(f"[fallback] Completed exact search on {len(valid_ids_list)} vectors")
            return np.vstack(result_D), np.vstack(result_I)
            
        except Exception as e:
            print(f"[fallback] Error in optimized search: {e}, falling back to simple post-filter")
            # Final fallback - minimal fetch and filter
            k_try = min(k * 3, 1000)
            D_all, I_all = self.index.search(query_vec, k_try)

        valid_set = set(valid_ids.tolist())

        # Collect first k matches that are in valid_set
        filtered_I = []
        filtered_D = []
        for dist_row, idx_row in zip(D_all, I_all):
            row_I = []
            row_D = []
            for d, idx in zip(dist_row, idx_row):
                if int(idx) in valid_set:
                    row_I.append(int(idx))
                    row_D.append(float(d))
                    if len(row_I) >= k:
                        break
            # pad if not enough results
            while len(row_I) < k:
                row_I.append(-1)
                row_D.append(float('inf'))
            filtered_I.append(np.array(row_I, dtype=np.int64))
            filtered_D.append(np.array(row_D, dtype=np.float32))

        # Return stacked arrays shaped (1, k)
        return np.vstack(filtered_D), np.vstack(filtered_I)

    def search_with_geo_filter(self, query_vec, k, min_lat, max_lat, min_lng, max_lng):
        """
        Perform KNN search restricted to images in the bounding box.
        
        :param query_vec: A NumPy array of shape (1, d) or (d,) containing the query embedding.
        :param k: Number of nearest neighbors.
        :param min_lat, max_lat, min_lng, max_lng: bounding box coordinates.
        :return: (D, I) where D are distances, I are Faiss IDs.
        """
        import time
        import sys
        t_start = time.time()
        print(f"[search_with_geo_filter] Starting geo-filtered search with k={k}", flush=True)
        sys.stdout.flush()
        
        # 1. Find row indices in bounding box
        subset_ids = self._get_ids_in_bounding_box(min_lat, max_lat, min_lng, max_lng)
        if len(subset_ids) == 0:
            # No images in that bounding box
            print("[search_with_geo_filter] No images in bounding box", flush=True)
            return np.array([[]]), np.array([[]])
        
        print(f"[search_with_geo_filter] Searching with {len(subset_ids)} images in bounding box.", flush=True)
        # 2. Convert metadata row indices -> Faiss IDs (if mapping available), then build selector
        faiss_ids = self._metadata_to_faiss_ids(subset_ids)

        # For IVF-style indexes we can attach the selector via SearchParametersIVF.
        # HNSW-style indexes (IndexHNSW) don't accept SearchParametersIVF and will
        # raise the 'params type invalid' error. Detect and fallback to a
        # post-filter approach for those indexes.
        if self._is_hnsw_index():
            print("Index appears to be HNSW-style; using post-filter fallback for geo selection.", flush=True)
            sys.stdout.flush()
            return self._search_and_filter_post(query_vec, k, faiss_ids, nprobe=1)

        # Set OpenMP threads to 1 for speed
        try:
            old_threads = faiss.omp_get_max_threads()
            faiss.omp_set_num_threads(1)
            print(f"[search_with_geo_filter] Set OMP threads from {old_threads} to 1", flush=True)
        except Exception as e:
            print(f"[search_with_geo_filter] Could not set OMP threads: {e}", flush=True)

        sel = faiss.IDSelectorBatch(faiss_ids.astype(np.int64))  # must be int64 for Faiss
        params = faiss.SearchParametersIVF(sel=sel, nprobe=1)

        # 4. Run the search restricted to that subset
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)

        t_search_start = time.time()
        print(f"[search_with_geo_filter] Calling index.search with nprobe=1...", flush=True)
        sys.stdout.flush()
        D, I = self.index.search(query_vec, k, params=params)
        t_search_elapsed = time.time() - t_search_start

        print(f"[search_with_geo_filter] Search took {t_search_elapsed:.3f}s, found {len(I[0])} results.", flush=True)
        t_total = time.time() - t_start
        print(f"[search_with_geo_filter] Total time: {t_total:.3f}s", flush=True)
        sys.stdout.flush()

        return D, I
    

    def _get_ids_in_species(self, taxon_id):
        """
        Return the 'row indices' (which must match the Faiss ID or be mappable to the Faiss ID)
        for images with the given species.
        """
        import time
        t_start = time.time()
        
        # Collect all arrays first, then concatenate once (much faster than extend in loop)
        arrays_to_concat = [np.asarray(self.taxon_index[taxon_id])]
        
        for descendant_taxon_id in self.taxon_descendants[taxon_id]:
            arrays_to_concat.append(np.asarray(self.taxon_index[descendant_taxon_id]))
        
        # Single concatenate is much faster than repeated list extends
        subset_ids = np.concatenate(arrays_to_concat) if len(arrays_to_concat) > 1 else arrays_to_concat[0]
        
        t_elapsed = time.time() - t_start
        print(f"[species] Found {len(subset_ids)} images with taxon_id {taxon_id} in {t_elapsed:.3f}s", flush=True)
        return subset_ids
    
    def search_with_species_filter(self, query_vec, k, taxon_id, nprobe=1):
        """
        Perform KNN search restricted to images with the given species.
        
        :param query_vec: A NumPy array of shape (1, d) or (d,) containing the query embedding.
        :param k: Number of nearest neighbors.
        :return: (D, I) where D are distances, I are Faiss IDs.
        """
        import time
        import sys
        t_start = time.time()
        print(f"[search_with_species_filter] Starting search for taxon_id={taxon_id}, k={k}", flush=True)
        sys.stdout.flush()
        
        # 1. Find row indices in species
        t_species_start = time.time()
        subset_ids = self._get_ids_in_species(taxon_id)
        t_species_elapsed = time.time() - t_species_start
        print(f"[search_with_species_filter] Got species IDs in {t_species_elapsed:.3f}s", flush=True)
        
        if len(subset_ids) == 0:
            print("[search_with_species_filter] No images found for taxon_id", flush=True)
            return [np.array([])], [np.array([])]
        
        # 2. Convert to FAISS IDs
        t_map_start = time.time()
        faiss_ids = self._metadata_to_faiss_ids(subset_ids)
        t_map_elapsed = time.time() - t_map_start
        print(f"[search_with_species_filter] Mapped to {len(faiss_ids)} FAISS IDs in {t_map_elapsed:.3f}s", flush=True)

        if self._is_hnsw_index():
            print("Index appears to be HNSW-style; using post-filter fallback for species selection.", flush=True)
            sys.stdout.flush()
            return self._search_and_filter_post(query_vec, k, faiss_ids, nprobe=nprobe)

        # Set OpenMP threads to 1 for speed
        try:
            old_threads = faiss.omp_get_max_threads()
            faiss.omp_set_num_threads(1)
            print(f"[search_with_species_filter] Set OMP threads from {old_threads} to 1", flush=True)
        except Exception as e:
            print(f"[search_with_species_filter] Could not set OMP threads: {e}", flush=True)
        
        # Build selector
        t_selector_start = time.time()
        sel = faiss.IDSelectorBatch(faiss_ids.astype(np.int64))
        params = faiss.SearchParametersIVF(sel=sel, nprobe=nprobe)
        t_selector_elapsed = time.time() - t_selector_start
        print(f"[search_with_species_filter] Built selector in {t_selector_elapsed:.3f}s", flush=True)

        t_search_start = time.time()
        print(f"[search_with_species_filter] Calling index.search with nprobe={nprobe}...", flush=True)
        sys.stdout.flush()
        D, I = self.index.search(query_vec, k, params=params)
        t_search_elapsed = time.time() - t_search_start
        
        print(f"[search_with_species_filter] FAISS search took {t_search_elapsed:.3f}s, found {len(I[0])} results.", flush=True)
        t_total = time.time() - t_start
        print(f"[search_with_species_filter] Total time: {t_total:.3f}s", flush=True)
        sys.stdout.flush()
        return D, I
    

    def search_with_filters(self, query_vec, k, geo_filter=None, taxon_id=None, months=None, nprobe=1):
        """
        Perform KNN search restricted to images in the bounding box.
        
        :param query_vec: A NumPy array of shape (1, d) or (d,) containing the query embedding.
        :param k: Number of nearest neighbors.
        :param geo_filter (optional): bounding box coordinate tuple (min_lat, max_lat, min_lng, max_lng)
        :param taxon_id (optional): species ID
        :param months (optional): list of months (numbered 1-12) to filter by
        :param nprobe: number of IVF probes (i.e. search strength)
        :return: (D, I) where D are distances, I are Faiss IDs.
        """
        import time
        import sys
        t_start = time.time()
        print(f"[search_with_filters] Starting filtered search with k={k}, nprobe={nprobe}", flush=True)
        sys.stdout.flush()

        subset_ids = None

        if geo_filter is not None:
            min_lat, max_lat, min_lng, max_lng = geo_filter
            subset_ids = self._get_ids_in_bounding_box(min_lat, max_lat, min_lng, max_lng)
            print("[filter] Have ", len(subset_ids), "images in filter after geo filter.", flush=True)
            
        if taxon_id is not None: # combine geo and taxon_id if needed
            taxon_subset_ids = self._get_ids_in_species(taxon_id)
            if subset_ids is None:
                subset_ids = taxon_subset_ids
            else:
                subset_ids = np.intersect1d(subset_ids, taxon_subset_ids)
            print("[filter] Have ", len(subset_ids), "images in filter after taxon filter.", flush=True)

        # Apply month filtering. Previously this only ran when subset_ids was
        # already populated (from geo or taxon filters). That made "month-only"
        # filters no-ops. Support both cases:
        #  - If subset_ids is None, build it from the month_index (month-only filter)
        #  - If subset_ids exists, intersect it with the months
        if months is not None:
            if self.month_index is None:
                print("Month filter provided but month index not available; ignoring months.", flush=True)
            else:
                if subset_ids is None:
                    # Build subset of all image ids matching the requested months
                    month_subset = np.where(np.isin(self.month_index, months))[0]
                    subset_ids = month_subset
                    print("[filter] Have ", len(subset_ids), "images in filter after month-only filter.", flush=True)
                else:
                    # Intersect the existing subset with the months
                    # Keep the previous safety check for extremely large subsets
                    if len(subset_ids) < 500 * 1000:
                        subset_ids = subset_ids[np.isin(self.month_index[subset_ids], months)]
                        print("[filter] Have ", len(subset_ids), "images in filter after month filter.", flush=True)
                    else:
                        # If the existing subset is huge, intersection may be expensive
                        # but still perform it; warn for visibility.
                        subset_ids = subset_ids[np.isin(self.month_index[subset_ids], months)]
                        print("[filter] (large) Have ", len(subset_ids), "images in filter after month filter.", flush=True)

        #     month_subset_ids = self._get_ids_in_months(months)
        #     if subset_ids is None:
        #         subset_ids = month_subset_ids
        #     else:
        #         subset_ids = np.intersect1d(subset_ids, month_subset_ids)

        if subset_ids is not None:
            if len(subset_ids) == 0:
                print("No images match the filters.", flush=True)
                return [np.array([])], [np.array([])]
            print(f"[search_with_filters] Searching with {len(subset_ids)} images that match filters.", flush=True)
            
            # 2. Convert metadata indices -> Faiss IDs and build an IDSelectorBatch
            faiss_ids = self._metadata_to_faiss_ids(subset_ids)

            if self._is_hnsw_index():
                print("Index appears to be HNSW-style; using post-filter fallback for combined filters.", flush=True)
                sys.stdout.flush()
                return self._search_and_filter_post(query_vec, k, faiss_ids, nprobe=nprobe)

            # Set OpenMP threads to 1 for speed
            try:
                old_threads = faiss.omp_get_max_threads()
                faiss.omp_set_num_threads(1)
                print(f"[search_with_filters] Set OMP threads from {old_threads} to 1", flush=True)
            except Exception as e:
                print(f"[search_with_filters] Could not set OMP threads: {e}", flush=True)

            sel = faiss.IDSelectorBatch(faiss_ids.astype(np.int64))
            params = faiss.SearchParametersIVF(sel=sel, nprobe=nprobe)

            t_search_start = time.time()
            print(f"[search_with_filters] Calling index.search with nprobe={nprobe}...", flush=True)
            sys.stdout.flush()
            result = self.index.search(query_vec, k, params=params)
            t_search_elapsed = time.time() - t_search_start
            
            print(f"[search_with_filters] Search took {t_search_elapsed:.3f}s", flush=True)
            t_total = time.time() - t_start
            print(f"[search_with_filters] Total time: {t_total:.3f}s", flush=True)
            sys.stdout.flush()
            return result
        else:
            print("[search_with_filters] No filters applied, using unfiltered search", flush=True)
            sys.stdout.flush()
            return self.search(query_vec, k)

    
    def search(self, query_vec, k):
        """
        Perform KNN search without any geospatial filtering.
        
        :param query_vec: A NumPy array of shape (1, d) or (d,) containing the query embedding.
        :param k: Number of nearest neighbors.
        :return: (D, I) where D are distances, I are Faiss IDs.
        """
        import time
        import sys
        t_start = time.time()
        print(f"[EnhancedSearchIndex.search] Starting unfiltered search for k={k} neighbors", flush=True)
        sys.stdout.flush()
        
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)
        
        print(f"[EnhancedSearchIndex.search] Query shape: {query_vec.shape}", flush=True)
        print(f"[EnhancedSearchIndex.search] Index type: {type(self.index).__name__}", flush=True)
        print(f"[EnhancedSearchIndex.search] Index ntotal: {self.index.ntotal}", flush=True)
        sys.stdout.flush()
        
        # Check if this is an IVF index that needs training
        if hasattr(self.index, 'nprobe'):
            print(f"[EnhancedSearchIndex.search] IVF index detected, current nprobe: {self.index.nprobe}", flush=True)
            # Use very aggressive nprobe for speed
            self.index.nprobe = 1
            print(f"[EnhancedSearchIndex.search] Set nprobe to 1 for maximum speed", flush=True)
        
        if hasattr(self.index, 'hnsw'):
            print(f"[EnhancedSearchIndex.search] HNSW index detected", flush=True)
            if hasattr(self.index.hnsw, 'efSearch'):
                print(f"[EnhancedSearchIndex.search] Current efSearch: {self.index.hnsw.efSearch}", flush=True)
                self.index.hnsw.efSearch = 8  # Very aggressive for speed
                print(f"[EnhancedSearchIndex.search] Set efSearch to 8 for maximum speed", flush=True)
        
        sys.stdout.flush()
        print(f"[EnhancedSearchIndex.search] About to call faiss index.search...", flush=True)
        sys.stdout.flush()
        
        # Set OpenMP threads to 1 to avoid thread pool hangs
        try:
            old_threads = faiss.omp_get_max_threads()
            faiss.omp_set_num_threads(1)
            print(f"[EnhancedSearchIndex.search] Set OMP threads from {old_threads} to 1", flush=True)
            sys.stdout.flush()
        except Exception as e:
            print(f"[EnhancedSearchIndex.search] Could not set OMP threads: {e}", flush=True)
        
        t_faiss_start = time.time()
        try:
            print(f"[EnhancedSearchIndex.search] Calling self.index.search(query_vec, {k})...", flush=True)
            sys.stdout.flush()
            D, I = self.index.search(query_vec, k)
            t_faiss_elapsed = time.time() - t_faiss_start
            print(f"[EnhancedSearchIndex.search] FAISS search completed in {t_faiss_elapsed:.3f}s", flush=True)
        except Exception as e:
            print(f"[EnhancedSearchIndex.search] FAISS search failed: {e}", flush=True)
            raise
        
        print(f"[EnhancedSearchIndex.search] Search completed, found {len(I[0])} results", flush=True)
        t_total = time.time() - t_start
        print(f"[EnhancedSearchIndex.search] Total time: {t_total:.3f}s", flush=True)
        sys.stdout.flush()
        return D, I


# Example usage
# def test_search_index():
#     import os

#     data_path = "/data/vision/beery/natural_retrieval/inaturalist"
#     index_name = "siglip-so400m-14-384--290M-lg" 
#     index_path = os.path.join(data_path, "index", index_name, "knn.index")

#     index = EnhancedSearchIndex(index_path, emb_to_inat_index=None)

#     # Example query without geo
#     print("Query without geo")
#     query = np.random.randn(1152).astype(np.float32)
#     k = 10
#     D, I = index.search(query, k)
#     print(D, I)

#     # Example query with geo
#     print("Query with geo")

#     from all_clip import load_clip
#     import torch
#     device = 'cpu'
#     model_name = 'open_clip:ViT-SO400M-14-SigLIP-384/webli' 
#     model, preprocess, tokenizer = load_clip(model_name, device='cpu', use_jit=False)
#     model.eval()

#     from utils import MemoryMappedMetadataProvider
#     metadata_name = "siglip-so400m-14-384"
#     meta = MemoryMappedMetadataProvider(os.path.join(data_path, "embs", metadata_name, "metadata"))

#     def get_text_embeddings(query_text):
#         text_inputs = torch.cat([tokenizer(query_text)]).to(device)
#         with torch.no_grad():
#             text_features = model.encode_text(text_inputs)
#             text_features /= text_features.norm(dim=-1, keepdim=True)
            
#         return text_features
    
#     query = get_text_embeddings('cat').cpu().numpy()

#     # min_lat, max_lat = 30, 35
#     # min_lng, max_lng = -82, -80
#     min_lat, max_lat = 42.31, 42.38
#     min_lng, max_lng = -71.14, -71.01
#     D, I = index.search_with_geo_filter(query, k, min_lat, max_lat, min_lng, max_lng)
#     print("final results:")
#     print('IDs:', I[0])
#     photo_ids = [meta.get(photo_index)[0] if photo_index != -1 else -1 for photo_index in I[0] ]
#     print('Photo IDs:', photo_ids)
#     print('Distances:', D[0])



def run_search_service():

    search_service = Flask(__name__)
    enhanced_index = None

    @search_service.route('/search', methods=['POST'])
    def search_endpoint():
        data = request.get_json()
        query_vec = np.array(data["query_vec"], dtype=np.float32)
        k = data["k"]
        taxon_id = data.get("taxon_id", None)
        geo_filter = data.get("geo_filter", None)
        month_filter = data.get("month_filter", None)
        D, I = enhanced_index.search_with_filters(query_vec, k, geo_filter=geo_filter,
                                                  taxon_id=taxon_id, months=month_filter)
        return jsonify({"distances": D.tolist(), "indices": I.tolist()})
    
    # Initialize enhanced_index here:
    data_path = "/data/vision/beery/natural_retrieval/inaturalist"
    index_name = "siglip-so400m-14-384--290M-lg"
    index_path = os.path.join(data_path, "index", index_name, "knn.index")
    enhanced_index = EnhancedSearchIndex(index_path, emb_to_inat_index=None)

    from waitress import serve
    print("Running on port 5003...")
    serve(search_service, host="0.0.0.0", port=5002)
    # search_service.run(host='0.0.0.0', port=5003)
    

if __name__ == "__main__":
    run_search_service()