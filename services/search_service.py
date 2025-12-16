import numpy as np
import requests
import os
from config import Config
from utils import MemoryMappedMetadataProvider, EmbeddingProvider
import pandas as pd

class SearchService:
    def __init__(self):
        self.embeddings = EmbeddingProvider(Config.EMBEDDINGS_PATH)
        self.meta = MemoryMappedMetadataProvider(Config.METADATA_PATH)
        # Load photo id -> extension mapping. Prefer JSON if available because
        # some environments produce pickled .npy files that may not unpickle
        # cleanly across numpy versions (ModuleNotFoundError: numpy._core).
        self.photo_id_to_ext_map = {}
        try:
            # if a .json sibling exists, prefer it
            json_path = None
            if Config.PHOTO_ID_TO_EXT_MAP_PATH.endswith('.npy'):
                json_candidate = Config.PHOTO_ID_TO_EXT_MAP_PATH[:-4] + '.json'
                if os.path.exists(json_candidate):
                    json_path = json_candidate
            if json_path and os.path.exists(json_path):
                import json as _json
                with open(json_path, 'r') as _f:
                    self.photo_id_to_ext_map = _json.load(_f)
            else:
                # fallback to np.load but catch pickle/import errors and give a
                # clear message
                try:
                    self.photo_id_to_ext_map = np.load(Config.PHOTO_ID_TO_EXT_MAP_PATH, allow_pickle=True).item()
                except Exception as e:
                    raise RuntimeError(f"Failed to load PHOTO_ID_TO_EXT_MAP_PATH ({Config.PHOTO_ID_TO_EXT_MAP_PATH}): {e}\n" \
                                       f"If you have a JSON mapping available next to the .npy (same name + .json), it will be used automatically.")
        except Exception as e:
            raise
        
        self.image_id_to_taxon_id = None

        # Load species names from metadata CSV for populating search results
        self.photoid_to_filename = {}
        self.filename_to_species = {}
        try:
            import csv
            csv_path = os.path.join(getattr(Config, 'DATA_PATH', 'static'), 'metadata', 'metadata.csv')
            print(f"[SearchService.__init__] Attempting to load species names from: {csv_path}", flush=True)
            if os.path.exists(csv_path):
                print(f"[SearchService.__init__] CSV file exists, opening...", flush=True)
                with open(csv_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    # Check the header
                    fieldnames = reader.fieldnames
                    print(f"[SearchService.__init__] CSV columns: {fieldnames[:5]}...", flush=True)
                    required_cols = ['inat24_species_name', 'file_name', 'inat24_image_id']
                    missing = [c for c in required_cols if c not in fieldnames]
                    if missing:
                        print(f"[SearchService.__init__] WARNING: Missing columns: {missing}", flush=True)
                    
                    row_count = 0
                    for row in reader:
                        file_name = row.get('file_name', '')
                        species_name = row.get('inat24_species_name', '')
                        photo_id = row.get('inat24_image_id', '')
                        
                        if file_name and species_name:
                            self.filename_to_species[file_name] = species_name
                        if photo_id and file_name:
                            # Store as int and string for flexible lookup
                            try:
                                self.photoid_to_filename[int(photo_id)] = file_name
                                self.photoid_to_filename[str(photo_id)] = file_name
                            except ValueError:
                                self.photoid_to_filename[photo_id] = file_name
                        
                        row_count += 1
                        if row_count % 5000 == 0:
                            print(f"[SearchService.__init__] Processed {row_count} rows...", flush=True)
                    
                print(f"[SearchService.__init__] Loaded {len(self.filename_to_species)} species names and {len(self.photoid_to_filename)} photo_id mappings from {row_count} total rows", flush=True)
            else:
                print(f"[SearchService.__init__] ERROR: CSV file not found at {csv_path}", flush=True)
        except Exception as e:
            print(f"[SearchService.__init__] ERROR loading species names from CSV: {e}", flush=True)
            import traceback
            traceback.print_exc()
            self.photoid_to_filename = {}
            self.filename_to_species = {}

        # Load geo index: prefer explicit config path, otherwise look for
        # the common mappings file under DATA_PATH/mappings/image_id_to_latlong_int.npy
        self.geo_index = None
        try:
            if hasattr(Config, 'GEO_INDEX_PATH') and Config.GEO_INDEX_PATH and os.path.exists(Config.GEO_INDEX_PATH):
                self.geo_index = np.load(Config.GEO_INDEX_PATH, mmap_mode='r')
            else:
                candidate = os.path.join(getattr(Config, 'DATA_PATH', 'static'), 'mappings', 'image_id_to_latlong_int.npy')
                if os.path.exists(candidate):
                    self.geo_index = np.load(candidate, mmap_mode='r')
        except Exception as e:
            print(f"Error loading geo index: {e}")
        self.month_index = np.load(Config.MONTH_INDEX_PATH, mmap_mode='r')

        self.taxa_df = None
        self.species_name_to_taxon_id = {}
        try:
            species_map_path = os.path.join(getattr(Config, 'DATA_PATH', 'static'), 'mappings', 'species_name_to_taxon_id.json')
            if os.path.exists(species_map_path):
                import json as _json
                with open(species_map_path, 'r') as _f:
                    self.species_name_to_taxon_id = _json.load(_f)
        except Exception as e:
            print(f"Error loading species name map: {e}")
        
    def _load_taxa(self):
        try:
            self.taxa_df = pd.read_csv(Config.TAXA_DF_PATH)
            self.taxa_df.set_index('taxon_id', inplace=True)
        except Exception as e:
            print(f"Error loading taxa file: {e}")
            self.taxa_df = pd.DataFrame()
    
    def search(self, question_embedding, k, filters=None):
        """Perform search with optional filters"""
        import time
        t_start = time.time()
        print(f"[SearchService.search] START - k={k}, embedding shape={question_embedding.shape}")

        if filters is None:
            filters = {}

        try:
            print(f"[SearchService] raw filters: {filters}")
        except Exception:
            pass

        # Species filter: Extract species name for post-filtering
        t_filter_start = time.time()
        species_filter_name = None
        taxon_id = None
        try:
            species_filter_name = filters.get('species') or filters.get('species_name')
            if species_filter_name:
                print(f"[SearchService] Species filter will be applied post-search: '{species_filter_name}'", flush=True)
                # Don't use taxon_id filtering in index due to data issue
                taxon_id = None
        except Exception as e:
            print(f"[SearchService] ERROR in species filter processing: {e}", flush=True)
            import traceback
            traceback.print_exc()
            species_filter_name = None
            taxon_id = None

        # Extract and process geo filter
        geo_filter = self._process_geo_filter(filters)
        
        # Extract and process month filter
        month_filter = self._process_month_filter(filters)
        
        t_filter_elapsed = time.time() - t_filter_start
        print(f"[SearchService] Filter processing took {t_filter_elapsed:.3f}s")
        
        # Debug logging
        try:
            print(f"[SearchService] processed taxon_id: {taxon_id}")
            print(f"[SearchService] processed geo_filter: {geo_filter}")
            print(f"[SearchService] processed month_filter: {month_filter}")
        except Exception:
            pass

        # Perform search (with species post-filtering if needed)
        t_search_start = time.time()
        try:
            # If species filter is active, we need to retrieve more results
            # and filter them afterwards (due to incorrect taxon mapping indices)
            search_k = k
            if species_filter_name:
                # Retrieve 10x results to ensure we get enough after filtering
                search_k = min(k * 10, 10000)
                print(f"[SearchService] Retrieving {search_k} results for post-filtering to {k}", flush=True)
            
            if Config.USE_SEARCH_SERVICE:
                print("[SearchService] Calling external search service...")
                distances, indices = self._call_external_search(
                    question_embedding, search_k, taxon_id, geo_filter, month_filter
                )
            else:
                print("[SearchService] Calling local search...")
                distances, indices = self._call_local_search(
                    question_embedding, search_k, taxon_id, geo_filter, month_filter
                )
        except Exception as e:
            print(f"[SearchService] ERROR in search execution: {e}", flush=True)
            import traceback
            traceback.print_exc()
            raise
        t_search_elapsed = time.time() - t_search_start
        print(f"[SearchService] Core search took {t_search_elapsed:.3f}s")

        # Process results
        t_process_start = time.time()
        print("[SearchService] Processing search results...")
        results = self._process_search_results(distances, indices)
        
        # Apply species post-filter if needed
        if species_filter_name and results['species']:
            print(f"[SearchService] Applying species post-filter for '{species_filter_name}'", flush=True)
            results = self._apply_species_post_filter(results, species_filter_name, k)
            print(f"[SearchService] After filtering: {len(results['id'])} results", flush=True)
        
        t_process_elapsed = time.time() - t_process_start
        print(f"[SearchService] Result processing took {t_process_elapsed:.3f}s")
        
        t_total = time.time() - t_start
        print(f"[SearchService.search] COMPLETE - Total time: {t_total:.3f}s")
        return results
    
    def _apply_species_post_filter(self, results, target_species, k):
        """Filter search results to only include target species"""
        # Find indices where species matches
        matching_indices = []
        for i, species in enumerate(results['species']):
            if species == target_species:
                matching_indices.append(i)
                if len(matching_indices) >= k:
                    break
        
        if not matching_indices:
            print(f"[SearchService] WARNING: No results found for species '{target_species}'", flush=True)
            # Return empty results
            return {
                'id': [],
                'img_url': [],
                'file_name': [],
                'scores': [],
                'species': [],
                'latitudes': [],
                'longitudes': [],
                'months': [],
            }
        
        # Filter all result arrays to only include matching indices
        filtered_results = {}
        for key in results.keys():
            if isinstance(results[key], list):
                filtered_results[key] = [results[key][i] for i in matching_indices]
            else:
                filtered_results[key] = results[key]
        
        return filtered_results
        
    def _process_taxon_filter(self, species_filter):
        """Process the species filter to get taxon ID"""
        if not species_filter:
            return None

        # If species_filter is numeric, return it directly
        try:
            tid = int(species_filter)
            return tid
        except Exception:
            pass

        # 1) try taxa_df if available
        try:
            if self.taxa_df is not None and not self.taxa_df.empty:
                taxon_matches = self.taxa_df.loc[self.taxa_df['name'] == species_filter]
                if len(taxon_matches) > 0:
                    return int(taxon_matches.index.values[0])
        except Exception:
            pass

        # 2) try species_name_to_taxon_id map generated from metadata
        try:
            if species_filter in self.species_name_to_taxon_id:
                return int(self.species_name_to_taxon_id[species_filter])
            # case-insensitive match
            low = species_filter.lower()
            for name, tid in self.species_name_to_taxon_id.items():
                if name.lower() == low:
                    return int(tid)
        except Exception:
            pass

        print(f'No taxon id found for species: {species_filter}')
        return None
        
    def _process_geo_filter(self, filters):
        """Process geographic filters"""
        if not filters.get('use_geo_filters', False):
            return None
            
        try:
            min_lat = float(filters['latitudeMin'])
            max_lat = float(filters['latitudeMax'])
            min_lng = float(filters['longitudeMin'])
            max_lng = float(filters['longitudeMax'])
            return (min_lat, max_lat, min_lng, max_lng)
        except Exception as e:
            print(f'Error parsing geo filters: {e}')
            return None
            
    def _process_month_filter(self, filters):
        """Process month filters"""
        if not filters.get('months', False):
            return None
            
        months = filters.get('months')
        if not months or len(months) == 0:
            return None
            
        return [int(month) for month in months]
    
    def _decode_lat_lon(self, lat_enc, lon_enc):
        lat = (lat_enc / 100.0) - 90.0
        lon = (lon_enc / 100.0) - 180.0
        return (lat, lon)
        
    def _call_external_search(self, query_vec, k, taxon_id, geo_filter, month_filter):
        """Call external search service"""
        payload = {
            "query_vec": query_vec.tolist(),
            "k": k,
            "taxon_id": taxon_id,
            "geo_filter": geo_filter,
            "month_filter": month_filter
        }
        try:
            # Short timeout so frontend doesn't hang if external service is down
            resp = requests.post("http://localhost:5002/search", json=payload, timeout=5)
        except Exception as e:
            print(f"Error calling search service (request failed): {e}")
            # Fallback to local index search
            try:
                return self._call_local_search(query_vec, k, taxon_id, geo_filter, month_filter)
            except Exception as le:
                print(f"Local fallback also failed: {le}")
                return np.array([[0]]), np.array([[-1]])

        # If we get a response, ensure it's valid JSON
        if resp.status_code != 200:
            print(f"Search service returned status {resp.status_code}: {resp.text}")
            try:
                return self._call_local_search(query_vec, k, taxon_id, geo_filter, month_filter)
            except Exception as le:
                print(f"Local fallback also failed: {le}")
                return np.array([[0]]), np.array([[-1]])

        try:
            data = resp.json()
        except ValueError:
            # resp.text may be empty or not JSON; include it in logs for debugging
            print(f"Error calling search service: response not JSON. Body:\n{resp.text}")
            try:
                return self._call_local_search(query_vec, k, taxon_id, geo_filter, month_filter)
            except Exception as le:
                print(f"Local fallback also failed: {le}")
                return np.array([[0]]), np.array([[-1]])

        # Successful parse
        return np.array(data.get("distances", [[0]])), np.array(data.get("indices", [[-1]]))
            
    def _call_local_search(self, query_vec, k, taxon_id, geo_filter, month_filter):
        """Use local search index"""
        import time
        t_start = time.time()
        print("[SearchService._call_local_search] START", flush=True)
        
        # Note: This would require implementing EnhancedSearchIndex class
        from run_search_index import EnhancedSearchIndex
        
        # Initialize index if needed
        if not hasattr(self, 'index') or self.index is None:
            t_init_start = time.time()
            print("[SearchService._call_local_search] Index not initialized, creating EnhancedSearchIndex...", flush=True)
            print(f"[SearchService._call_local_search] Index path: {Config.INDEX_PATH}", flush=True)
            self.index = EnhancedSearchIndex(Config.INDEX_PATH)
            t_init_elapsed = time.time() - t_init_start
            print(f"[SearchService._call_local_search] Index initialization took {t_init_elapsed:.3f}s", flush=True)
            
            # Warm up the index with a dummy search
            try:
                print("[SearchService._call_local_search] Warming up index with dummy search...", flush=True)
                import sys
                sys.stdout.flush()
                dummy_vec = np.zeros((1, query_vec.shape[1] if query_vec.ndim > 1 else len(query_vec)), dtype=np.float32)
                t_warmup_start = time.time()
                _, _ = self.index.search(dummy_vec, min(k, 10))
                t_warmup_elapsed = time.time() - t_warmup_start
                print(f"[SearchService._call_local_search] Warmup search took {t_warmup_elapsed:.3f}s", flush=True)
                sys.stdout.flush()
            except Exception as e:
                print(f"[SearchService._call_local_search] Warmup failed (non-critical): {e}", flush=True)
        else:
            print("[SearchService._call_local_search] Using cached index", flush=True)
        
        t_query_start = time.time()
        print(f"[SearchService._call_local_search] Calling index.search_with_filters(k={k}, taxon_id={taxon_id}, geo_filter={geo_filter}, months={month_filter})...", flush=True)
        import sys
        sys.stdout.flush()
        result = self.index.search_with_filters(
            query_vec, k=k, 
            taxon_id=taxon_id, 
            geo_filter=geo_filter, 
            months=month_filter
        )
        t_query_elapsed = time.time() - t_query_start
        print(f"[SearchService._call_local_search] Index query took {t_query_elapsed:.3f}s", flush=True)
        
        t_total = time.time() - t_start
        print(f"[SearchService._call_local_search] COMPLETE - Total time: {t_total:.3f}s", flush=True)
        sys.stdout.flush()
        return result
        
    def _process_search_results(self, distances, indices):
        """Process search results into response format"""
        scores = distances[0]
        image_ids = [i for i in indices[0] if i != -1]
        
        # Get photo IDs from metadata
        # `MemoryMappedMetadataProvider.get` returns a numpy.record-like item
        # where the first field is the photo_id. That value can be bytes or a
        # numpy integer depending on how the memmap was written. Normalize it
        # to a Python str/int and defensively look up the extension in the
        # mapping using several common key formats. If nothing matches, fall
        # back to 'jpg'.
        photo_ids = []
        for image_id in image_ids:
            rec = self.meta.get(image_id)
            pid = rec[0]
            # normalize bytes -> str, numpy ints -> int
            try:
                import numpy as _np
                if isinstance(pid, (_np.bytes_, bytes)):
                    pid = pid.decode('utf-8')
                elif isinstance(pid, _np.integer):
                    pid = int(pid)
            except Exception:
                # if numpy isn't available here for some reason, try basic
                # conversions
                if isinstance(pid, bytes):
                    pid = pid.decode('utf-8')
            photo_ids.append(pid)

        # Resolve extensions robustly: mapping keys are often strings in JSON
        # files, so try several variants before defaulting to 'jpg'.
        extensions = []
        for pid in photo_ids:
            ext = None
            # try as-is, then str(pid), then int(pid) string
            candidates = [pid, str(pid)]
            try:
                if isinstance(pid, int):
                    candidates.append(str(int(pid)))
            except Exception:
                pass

            for c in candidates:
                if c in self.photo_id_to_ext_map:
                    ext = self.photo_id_to_ext_map[c]
                    break

            if ext is None:
                # last-ditch: try integer key lookup if mapping uses ints
                try:
                    ext = self.photo_id_to_ext_map[int(pid)]
                except Exception:
                    ext = 'jpg'
            extensions.append(ext)
        
        # Generate URLs
        # Check if photo_ids are already URLs or need to be converted
        img_urls = []
        for photo_id, ext in zip(photo_ids, extensions):
            # If photo_id looks like a URL, use it directly
            if isinstance(photo_id, str) and (photo_id.startswith('http://') or photo_id.startswith('https://')):
                img_urls.append(photo_id)
            else:
                # Otherwise, construct iNaturalist URL
                img_urls.append(self.to_url(photo_id, ext))
        
        # Species names: Use file_name -> species_name lookup
        # Note: photo_ids here are actually file paths from the memmap, not numeric IDs
        species_names = []
        file_names = []
        print(f"[SearchService] Extracting species names for {len(image_ids)} images", flush=True)
        print(f"[SearchService] First 5 photo_ids (file paths): {photo_ids[:5]}", flush=True)
        print(f"[SearchService] Total species mappings: {len(self.filename_to_species)}", flush=True)
        
        for idx, image_id in enumerate(image_ids):
            # Get the full metadata record from memmap
            rec = self.meta.get(image_id)
            # The first field is photo_id, which is actually the file path
            file_path = rec[0]
            # Normalize bytes -> str
            if isinstance(file_path, bytes):
                file_path = file_path.decode('utf-8')
            
            file_names.append(file_path)
            
            # Look up species_name from file_name
            species_name = self.filename_to_species.get(file_path, '')
            species_names.append(species_name)
            
            if idx < 3:
                print(f"[SearchService] image_id={image_id}, file_path='{file_path}', species_name='{species_name}'", flush=True)
        
        non_empty = sum(1 for s in species_names if s)
        print(f"[SearchService] Extracted {len(species_names)} species names, {non_empty} non-empty", flush=True)
        print(f"[SearchService] First 3 species: {species_names[:3]}", flush=True)

        # add geo coordinates if exists
        if self.geo_index is not None:
            if len(image_ids) == 0:
                latitudes = []
                longitudes = []
            else:
                lat_lon = self.geo_index[image_ids] # (N, 2) numpy array
                lat_enc = lat_lon[:, 0]
                lon_enc = lat_lon[:, 1]
                lat_lon = [self._decode_lat_lon(lat, lon) for lat, lon in zip(lat_enc, lon_enc)]
                latitudes, longitudes = list(zip(*lat_lon))
        else:
            latitudes = [None] * len(photo_ids)
            longitudes = [None] * len(photo_ids)

        if self.month_index is not None:
            months = self.month_index[image_ids]
            months = [int(month) for month in months]
        else:
            months = [None] * len(photo_ids)

        # Convert photo_ids to appropriate format
        # Note: photo_ids are file paths, not numeric IDs in this dataset
        converted_ids = []
        for pid in photo_ids:
            try:
                converted_ids.append(int(pid))
            except (ValueError, TypeError):
                # If it's a string path or can't be converted, keep as string
                converted_ids.append(str(pid))

        return {
            'id': converted_ids,
            'img_url': img_urls,
            'file_name': file_names,
            'scores': scores.tolist(),
            'species': species_names,
            'latitudes': latitudes,
            'longitudes': longitudes,
            'months': months,
        }

    def to_url(self, photo_id, extension="jpg"):
        """Convert photo_id to URL"""
        if isinstance(photo_id, str) and (photo_id.startswith('http://') or photo_id.startswith('https://')):
            return photo_id
        if isinstance(photo_id, str) and '/' in photo_id:
            base_url = "https://inquire-search.s3.us-east-2.amazonaws.com/rerank_arxiv/"
            return base_url + photo_id
        return f"https://inaturalist-open-data.s3.amazonaws.com/photos/{photo_id}/medium.{extension}"
        
    def get_autocomplete_suggestions(self, query, max_matches=15):
        """Get autocomplete suggestions for species names"""
        if not query:
            return []
        
        query = query.lower()
        matches = []
        
        # Use species_name_to_taxon_id map for autocomplete if taxa_df not available
        if self.taxa_df is None or self.taxa_df.empty:
            if not self.species_name_to_taxon_id:
                print("No species data available for autocomplete suggestions.")
                return []
            
            # Search through species names
            for species_name in self.species_name_to_taxon_id.keys():
                if query in species_name.lower():
                    matches.append({
                        'name': species_name,
                        'common_name': None,
                        'rank': 'species'
                    })
                    if len(matches) >= max_matches:
                        break
            return matches
        
        # Original taxa_df based autocomplete
        for i, search_string in enumerate(self.taxa_df["search_string"].values):
            if query in search_string:
                matches.append(self.taxa_df.iloc[i])
                if len(matches) >= max_matches:
                    break
                    
        if not matches:
            return []
            
        return pd.DataFrame(matches)[['name', 'common_name', 'rank']].to_dict(orient="records")