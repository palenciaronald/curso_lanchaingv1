# Configuración de modelos
EMBEDDING_MODEL = "text-embedding-3-small"
QUERY_MODEL = "gpt-4o-mini"
GENERATION_MODEL = "gpt-4o"


# configuracion de vector store

CHROMA_DB_PATH = "C:\\Users\\Usuario\\OneDrive - mathlorean\\Documentos\\tesis_laureada\\Cursos_agentes_udemy\\curso_agentes_lanchaing\\Tema3\\vectorstore_chroma"


# Configuración del retriever
SEARCH_TYPE = "mmr"
MMR_DIVERSITY_LAMBDA = 0.7
MMR_FETCH_K = 20
SEARCH_K = 2

# Configuracion alternativa para retriever hibrido
ENABLE_HYBRID_SEARCH = True
SIMILARITY_THRESHOLD = 0.70