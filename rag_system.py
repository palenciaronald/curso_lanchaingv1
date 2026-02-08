from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_classic.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_classic.retrievers import EnsembleRetriever
import streamlit as st
import os

from config import *
from prompts import *

# --- OPENAI KEY: Cloud (st.secrets) o local (env var) ---
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Falta OPENAI_API_KEY. Configúrala en Streamlit Cloud > Secrets o como variable de entorno.")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


@st.cache_resource
def initialize_rag_system():
    # Vector Store (local, viene en el repo)
    vectorestore = Chroma(
        embedding_function=OpenAIEmbeddings(model=EMBEDDING_MODEL),
        persist_directory=CHROMA_DB_PATH
    )

    # Modelos
    llm_queries = ChatOpenAI(model=QUERY_MODEL, temperature=0)
    llm_generation = ChatOpenAI(model=GENERATION_MODEL, temperature=0)

    base_retriever = vectorestore.as_retriever(
        search_type=SEARCH_TYPE,
        search_kwargs={
            "k": SEARCH_K,
            "lambda_mult": MMR_DIVERSITY_LAMBDA,
            "fetch_k": MMR_FETCH_K
        }
    )

    similarity_retriever = vectorestore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": SEARCH_K}
    )

    multi_query_prompt = PromptTemplate.from_template(MULTI_QUERY_PROMPT)

    mmr_multi_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm_queries,
        prompt=multi_query_prompt
    )

    if ENABLE_HYBRID_SEARCH:
        final_retriever = EnsembleRetriever(
            retrievers=[mmr_multi_retriever, similarity_retriever],
            weights=[0.7, 0.3],
            similarity_threshold=SIMILARITY_THRESHOLD
        )
    else:
        final_retriever = mmr_multi_retriever

    prompt = PromptTemplate.from_template(RAG_TEMPLATE)

    def format_docs(docs):
        formatted = []
        for i, doc in enumerate(docs, 1):
            header = f"[Fragmento {i}]"
            if doc.metadata:
                source = os.path.basename(doc.metadata.get("source", "No especificada"))
                page = doc.metadata.get("page")
                header += f" - Fuente: {source}"
                if page is not None:
                    header += f" - Pagina: {page}"

            formatted.append(f"{header}\n{doc.page_content.strip()}")
        return "\n\n".join(formatted)

    rag_chain = (
        {
            "context": final_retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm_generation
        | StrOutputParser()
    )

    # OJO: retorna el mismo retriever que usa el chain (para que docs y respuesta coincidan)
    return rag_chain, final_retriever


def query_rag(question):
    try:
        rag_chain, retriever = initialize_rag_system()
        response = rag_chain.invoke(question)
        docs = retriever.invoke(question)

        docs_info = []
        for i, doc in enumerate(docs[:SEARCH_K], 1):
            docs_info.append({
                "fragmento": i,
                "contenido": doc.page_content[:1000] + "..." if len(doc.page_content) > 1000 else doc.page_content,
                "fuente": os.path.basename(doc.metadata.get("source", "No especificada")),
                "pagina": doc.metadata.get("page", "No especificada")
            })

        return response, docs_info

    except Exception as e:
        return f"Error al procesar la consulta: {str(e)}", []


def get_retriever_info():
    return {
        "tipo": f"{SEARCH_TYPE.upper()} + MultiQuery" + (" + Hybrid" if ENABLE_HYBRID_SEARCH else ""),
        "documentos": SEARCH_K,
        "diversidad": MMR_DIVERSITY_LAMBDA,
        "candidatos": MMR_FETCH_K,
        "umbral": SIMILARITY_THRESHOLD if ENABLE_HYBRID_SEARCH else "N/A"
    }
