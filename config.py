import os
from dotenv import load_dotenv

load_dotenv()  # Load variables from .env file


class Config:
    # Flask config
    # The secret key is used to securely store session data (i.e. the user ID)
    SECRET_KEY = os.environ.get('FLASK_SECRET_KEY')
    ENV = os.environ.get('FLASK_ENV', 'development')

    # Application config
    DATA_PATH = "static" # this is pointing to the cluster path
    LOGIN_REQUIRED = os.environ.get('LOGIN_REQUIRED', 'False').lower() == 'true'
    USE_SEARCH_SERVICE = False

    # Model config
    MODEL_NAME = 'open_clip:ViT-SO400M-14-SigLIP-384/webli'
    METADATA_NAME = "siglip-so400m-14-384"
    INDEX_NAME = "siglip-so400m-14-384--290M-lg"

    # File paths
    INDEX_PATH = os.path.join(DATA_PATH, "index", "knn.index")
    EMBEDDINGS_PATH = os.path.join(DATA_PATH, "embeddings", "embeddings_0.npy")
    METADATA_PATH = os.path.join(DATA_PATH, "metadata")

    PHOTO_ID_TO_EXT_MAP_PATH = os.path.join(DATA_PATH, "mappings", "siglip_so400m_patch14_384_inat24_rerank_img_mapping.npy") # os.path.join(DATA_PATH, "notebooks/photo_id_to_ext_siglip_so400m.npy")
    IMAGE_ID_TO_TAXON_ID_PATH = os.path.join(DATA_PATH, "mappings", "map_taxon_id_to_image_ids.npy")

    # Image metadata
    GEO_INDEX_PATH = os.path.join(DATA_PATH, 'mappings', 'image_id_to_latlong_int.npy')
    MONTH_INDEX_PATH = os.path.join(DATA_PATH, "mappings", "image_id_to_month.npy")

    # Filters directory (for species, geo, month filters)
    FILTERS_DIR = os.path.join(DATA_PATH, "mappings")

    # Taxa config
    TAXA_FILE = os.path.join(DATA_PATH, "mappings", "species_name_to_taxon_id.npy")