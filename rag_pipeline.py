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


# ---------------------------------------------------------------------------
# Passo 2: Query Transformation — HyDE
# ---------------------------------------------------------------------------
class HyDETransformer:
    """
    Hypothetical Document Embeddings (HyDE).

    Em vez de vetorizar a query coloquial do usuário diretamente, pedimos
    ao LLM que "alucine" um documento técnico que responderia à pergunta.
    O vetor desse documento hipotético serve de âncora geométrica no espaço
    dos manuais médicos — eliminando a lacuna semântica entre linguagem
    coloquial e jargão clínico.
    """

    _SYSTEM_PROMPT = (
        "Você é um médico especialista redator de manuais clínicos. "
        "Dado um sintoma ou queixa relatada por um paciente em linguagem coloquial, "
        "escreva um parágrafo técnico de 80 a 120 palavras — como se fosse um "
        "fragmento de um manual médico — descrevendo a condição correspondente "
        "usando terminologia clínica precisa (CID-10, biomarcadores, escalas "
        "diagnósticas, protocolos). NÃO faça diagnóstico definitivo. "
        "Responda APENAS com o parágrafo, sem introdução ou conclusão."
    )

    def __init__(self, client: OpenAI) -> None:
        self._openai = client

    def generate_hypothetical_document(self, user_query: str) -> str:
        """
        Gera um documento técnico hipotético a partir da query do usuário.

        Args:
            user_query: Pergunta ou sintoma em linguagem natural/coloquial.

        Returns:
            Texto técnico gerado pelo LLM (documento hipotético).
        """
        logger.info("HyDE — gerando documento hipotético para: '%s'", user_query)

        response = self._openai.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": self._SYSTEM_PROMPT},
                {"role": "user", "content": user_query},
            ],
            temperature=0.3,
            max_tokens=300,
        )
        content = response.choices[0].message.content
        if content is None:
            raise RuntimeError(
                "O LLM não retornou conteúdo (finish_reason pode ser 'content_filter')."
            )
        hypothetical_doc = content.strip()
        if not hypothetical_doc:
            raise RuntimeError("O LLM retornou conteúdo vazio após strip().")
        logger.info("Documento hipotético gerado (%d chars).", len(hypothetical_doc))
        return hypothetical_doc


# ---------------------------------------------------------------------------
# Passo 4: Re-ranking fino via Cross-Encoder
# ---------------------------------------------------------------------------
class CrossEncoderReranker:
    """
    Re-ranker baseado em Cross-Encoder (modelo de atenção bilateral).

    Diferente do Bi-Encoder (que gera embeddings independentes), o
    Cross-Encoder processa query + documento juntos ([CLS] Query [SEP] Doc),
    permitindo atenção cruzada e scores de relevância muito mais precisos,
    ao custo de latência maior — por isso aplicado apenas nos Top-10.
    """

    def __init__(self, model_name: str = CROSS_ENCODER_MODEL) -> None:
        logger.info("Carregando Cross-Encoder: %s ...", model_name)
        self._model = CrossEncoder(model_name)
        logger.info("Cross-Encoder carregado com sucesso.")

    def rerank(
        self,
        query: str,
        documents: list[RetrievedDocument],
        top_k: int = TOP_K_RERANK,
    ) -> list[RetrievedDocument]:
        """
        Pontua e re-ordena documentos usando atenção profunda bilateral.

        Args:
            query: Query ORIGINAL do usuário (não o documento hipotético HyDE).
            documents: Lista de RetrievedDocument vindos do Bi-Encoder.
                Nota: os objetos são mutados in-place com o campo rerank_score.
            top_k: Número de documentos a retornar após re-ranking.

        Returns:
            Top-k documentos re-ordenados pelo score do Cross-Encoder.
        """
        if not documents:
            return []

        pairs = [(query, doc.text) for doc in documents]
        # predict() retorna ndarray quando len(pairs) > 1 e float escalar
        # quando len == 1; np.atleast_1d normaliza ambos os casos.
        raw_scores: list[float] = np.atleast_1d(self._model.predict(pairs)).tolist()

        for doc, score in zip(documents, raw_scores):
            doc.rerank_score = round(score, 4)

        # Todos os rerank_score foram preenchidos acima — o cast é seguro.
        reranked = sorted(
            documents,
            key=lambda d: d.rerank_score,  # type: ignore[return-value]
            reverse=True,
        )
        return reranked[:top_k]


# ---------------------------------------------------------------------------
# Pipeline orquestrador
# ---------------------------------------------------------------------------
class AdvancedRAGPipeline:
    """
    Orquestra os quatro passos do pipeline RAG avançado:
      1. Indexação HNSW
      2. HyDE (query transformation)
      3. Bi-Encoder retrieval (Top-10)
      4. Cross-Encoder re-ranking (Top-3)
    """

    def __init__(self, api_key: str | None) -> None:
        if not api_key or not api_key.strip():
            raise ValueError(
                "OPENAI_API_KEY não encontrada. "
                "Copie .env.example para .env e preencha sua chave de API."
            )
        validated_key: str = api_key  # narrowing explícito: str | None → str após guard
        self._openai_client = OpenAI(api_key=validated_key)
        self._index = HNSWIndex(client=self._openai_client)
        self._hyde = HyDETransformer(client=self._openai_client)
        self._reranker = CrossEncoderReranker()

    # ------------------------------------------------------------------
    # Passo 1 — Indexação
    # ------------------------------------------------------------------
    def build_index(self, documents: list[dict[str, str]]) -> None:
        """Constrói o índice HNSW com os documentos do corpus."""
        self._index.build(documents)

    # ------------------------------------------------------------------
    # Passos 2-4 — Execução completa do pipeline de recuperação
    # ------------------------------------------------------------------
    def run(self, user_query: str) -> list[RetrievedDocument]:
        """
        Executa o pipeline completo para uma query do usuário.

        Args:
            user_query: Pergunta em linguagem natural/coloquial (não vazia).

        Returns:
            Top-3 documentos mais relevantes após re-ranking pelo
            Cross-Encoder, prontos para injeção no contexto do LLM gerador.
        """
        if not user_query or not user_query.strip():
            raise ValueError("user_query não pode ser vazia.")

        # ── Passo 2: HyDE ────────────────────────────────────────────────
        print(f"\n{_SEP_SINGLE}")
        print("PASSO 2 — HyDE: Transformação da Query")
        print(_SEP_SINGLE)
        hypothetical_doc = self._hyde.generate_hypothetical_document(user_query)
        print(f"Query original  : {user_query}")
        print("\nDocumento hipotético gerado pelo LLM:\n")
        print(textwrap.fill(hypothetical_doc, width=70))

        # ── Passo 2b: Vetorizar documento hipotético ─────────────────────
        hyde_embedding = self._index.embed_single(hypothetical_doc)

        # ── Passo 3: Bi-Encoder retrieval via HNSW ───────────────────────
        print(f"\n{_SEP_SINGLE}")
        print(f"PASSO 3 — Bi-Encoder: Top-{TOP_K_RETRIEVE} via HNSW")
        print(_SEP_SINGLE)
        top_candidates = self._index.query(
            query_embedding=hyde_embedding,
            n_results=TOP_K_RETRIEVE,
        )

        if not top_candidates:
            logger.warning("Nenhum documento recuperado para a query: '%s'", user_query)
            return []

        print(f"\n{'#':>3}  {'Score Cosseno':>14}  Título")
        print(_SEP_SINGLE)
        for rank, doc in enumerate(top_candidates, start=1):
            print(f"{rank:>3}  {doc.cosine_score:>14.4f}  {doc.title}")

        # ── Passo 4: Cross-Encoder re-ranking ────────────────────────────
        print(f"\n{_SEP_SINGLE}")
        print(f"PASSO 4 — Cross-Encoder: Top-{TOP_K_RERANK} após re-ranking")
        print(_SEP_SINGLE)
        top_results = self._reranker.rerank(
            query=user_query,
            documents=top_candidates,
            top_k=TOP_K_RERANK,
        )

        print(f"\n{'#':>3}  {'Score CE':>10}  {'Score Cos':>10}  Título")
        print(_SEP_SINGLE)
        for rank, doc in enumerate(top_results, start=1):
            if doc.rerank_score is None:
                raise RuntimeError(
                    f"rerank_score não preenchido para: {doc.doc_id}"
                )
            print(
                f"{rank:>3}  {doc.rerank_score:>10.4f}  "
                f"{doc.cosine_score:>10.4f}  {doc.title}"
            )

        print(f"\n{_SEP_SINGLE}")
        print("DOCUMENTOS FINAIS — Prontos para injeção no contexto do LLM")
        print(_SEP_SINGLE)
        for rank, doc in enumerate(top_results, start=1):
            # rerank_score já verificado no loop acima — seguro acessar.
            score = doc.rerank_score
            assert score is not None  # satisfaz o type checker
            print(f"\n[{rank}] {doc.title}")
            print(f"    Score Cross-Encoder : {score:.4f}")
            print(f"    Score Cosseno       : {doc.cosine_score:.4f}")
            print("    Trecho              :")
            print(textwrap.fill(doc.text, width=66, initial_indent="    "))

        return top_results


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
def main() -> None:
    """Demonstra o pipeline com duas queries coloquiais de exemplo."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    # Env var tem precedência; fallback para a constante do módulo.
    # `or` garante que string vazia no ambiente não mascare a constante.
    api_key = os.getenv("OPENAI_API_KEY") or OPENAI_API_KEY

    pipeline = AdvancedRAGPipeline(api_key=api_key)

    # ── Passo 1: Indexação ────────────────────────────────────────────────
    print(f"\n{_SEP_DOUBLE}")
    print("PASSO 1 — Construção e Indexação do Grafo HNSW")
    print(_SEP_DOUBLE)
    pipeline.build_index(MEDICAL_DOCUMENTS)
    print(f"Total de documentos indexados: {len(MEDICAL_DOCUMENTS)}")

    # Query 1 — exemplifica o problema semântico descrito no enunciado
    pipeline.run(
        "dor de cabeça latejante, luz incomodando e enjoo quando me mexo"
    )

    # Query 2 — demonstra versatilidade do pipeline
    pipeline.run(
        "meu coração dispara do nada e eu desmaio, médico falou algo de arritmia"
    )


if __name__ == "__main__":
    main()
