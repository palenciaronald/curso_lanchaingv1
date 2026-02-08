from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CHROMA_DB_PATH = str(BASE_DIR / "vectorstore_chroma")

VECTORSTORE_TYPE = "local"

EMBEDDING_MODEL = "text-embedding-3-small"
QUERY_MODEL = "gpt-4o-mini"
GENERATION_MODEL = "gpt-4o"

SEARCH_TYPE = "mmr"
MMR_DIVERSITY_LAMBDA = 0.7
MMR_FETCH_K = 20
SEARCH_K = 2

ENABLE_HYBRID_SEARCH = True
SIMILARITY_THRESHOLD = 0.70
