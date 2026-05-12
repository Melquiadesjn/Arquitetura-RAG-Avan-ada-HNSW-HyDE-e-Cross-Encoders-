"""
Laboratório 09 — RAG Avançada: HNSW + HyDE + Cross-Encoder

Pipeline de Retrieval-Augmented Generation de nível de produção aplicado
a manuais médicos. Demonstra as três camadas do funil de recuperação:
  1. Indexação vetorial via HNSW (ChromaDB)
  2. Query Transformation via HyDE (OpenAI LLM)
  3. Recuperação rápida via Bi-Encoder (Top-10)
  4. Re-ranking fino via Cross-Encoder (Top-3)
"""

from __future__ import annotations

import logging
import os
import textwrap
from dataclasses import dataclass

import chromadb
import numpy as np
from openai import OpenAI
from sentence_transformers import CrossEncoder

from data.medical_corpus import MEDICAL_DOCUMENTS

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
# basicConfig é chamado apenas em main() para não poluir o logger raiz
# quando este módulo for importado por outro código.
logger = logging.getLogger("rag_pipeline")

# ---------------------------------------------------------------------------
# Constantes configuráveis
# ---------------------------------------------------------------------------
# Deixe esta constante vazia. Defina OPENAI_API_KEY no arquivo .env
# (copie .env.example para .env e preencha sua chave).
# A chave NUNCA deve ser escrita diretamente neste arquivo.
OPENAI_API_KEY = ""

COLLECTION_NAME = "medical_manuals"
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Hiperparâmetros HNSW — ver análise no README
HNSW_M = 16                 # nº de arestas bidirecionais por nó
HNSW_EF_CONSTRUCTION = 200  # tamanho da fila dinâmica na construção

TOP_K_RETRIEVE = 10  # "funil largo" do Bi-Encoder
TOP_K_RERANK = 3     # documentos finais após Cross-Encoder

_SEP_SINGLE = "─" * 70
_SEP_DOUBLE = "═" * 70


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------
@dataclass
class RetrievedDocument:
    """Representa um fragmento recuperado com seus metadados e scores."""

    doc_id: str
    title: str
    text: str
    cosine_score: float
    rerank_score: float | None = None


# ---------------------------------------------------------------------------
# Passo 1: Construção e Indexação do Grafo HNSW
# ---------------------------------------------------------------------------
class HNSWIndex:
    """Gerencia a criação e consulta do índice vetorial HNSW via ChromaDB."""

    def __init__(self, client: OpenAI) -> None:
        self._openai = client
        # EphemeralClient: índice em memória, sem persistência em disco.
        # Para produção, use chromadb.PersistentClient(path="./chroma_db").
        self._chroma = chromadb.EphemeralClient()
        self._collection = self._chroma.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={
                "hnsw:space": "cosine",
                "hnsw:M": HNSW_M,
                "hnsw:ef_construction": HNSW_EF_CONSTRUCTION,
            },
        )

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """Converte uma lista de textos em vetores densos via OpenAI."""
        response = self._openai.embeddings.create(
            model=EMBEDDING_MODEL,
            input=texts,
        )
        return [item.embedding for item in response.data]

    def build(self, documents: list[dict[str, str]]) -> None:
        """
        Vetoriza e indexa todos os documentos no grafo HNSW.

        Operação idempotente: se a coleção já contiver documentos suficientes,
        a indexação é ignorada para evitar duplicatas.
        """
        if not documents:
            raise ValueError("A lista de documentos não pode ser vazia.")

        current_count = self._collection.count()
        if current_count >= len(documents):
            logger.info(
                "Índice HNSW já contém %d documentos — ignorando reinserção.",
                current_count,
            )
            return

        logger.info(
            "Indexando %d documentos com HNSW (M=%d, ef_construction=%d)...",
            len(documents),
            HNSW_M,
            HNSW_EF_CONSTRUCTION,
        )

        ids, texts, metadatas = zip(
            *((doc["id"], doc["text"], {"title": doc["title"]}) for doc in documents)
        )
        embeddings = self._embed(list(texts))

        self._collection.add(
            ids=list(ids),
            embeddings=embeddings,
            documents=list(texts),
            metadatas=list(metadatas),
        )
        logger.info("Indexação concluída: %d vetores inseridos.", len(documents))

    def query(
        self,
        query_embedding: list[float],
        n_results: int = TOP_K_RETRIEVE,
    ) -> list[RetrievedDocument]:
        """
        Executa busca ANN (Approximate Nearest Neighbors) no índice HNSW.

        Args:
            query_embedding: Vetor da query (ou do documento hipotético HyDE).
            n_results: Quantidade de documentos mais próximos a retornar.

        Returns:
            Lista de RetrievedDocument ordenada por similaridade de cosseno
            decrescente.
        """
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        retrieved: list[RetrievedDocument] = []
        for i, doc_id in enumerate(results["ids"][0]):
            # ChromaDB com hnsw:space=cosine usa: distance = 1 - cosine_sim,
            # com valores em [0, 2]. Inversão correta: cosine_sim = 1 - distance.
            distance: float = results["distances"][0][i]
            cosine_similarity = round(1.0 - distance, 4)

            retrieved.append(
                RetrievedDocument(
                    doc_id=doc_id,
                    title=results["metadatas"][0][i]["title"],
                    text=results["documents"][0][i],
                    cosine_score=cosine_similarity,
                )
            )
        return retrieved

    def embed_single(self, text: str) -> list[float]:
        """Vetoriza um único texto e retorna seu vetor denso."""
        return self._embed([text])[0]
