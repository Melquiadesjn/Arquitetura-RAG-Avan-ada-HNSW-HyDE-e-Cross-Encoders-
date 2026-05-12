# Laboratório 09 — RAG Avançada: HNSW · HyDE · Cross-Encoders

> **Disciplina:** Inteligência Artificial Aplicada  
> **Entrega:** via GitHub com tag `v1.0`

---

## Sumário

1. [Objetivo](#1-objetivo)
2. [Arquitetura do Pipeline](#2-arquitetura-do-pipeline)
3. [Análise Técnica — HNSW vs. KNN Exato](#3-análise-técnica--hnsw-vs-knn-exato)
4. [Estrutura do Projeto](#4-estrutura-do-projeto)
5. [Como Executar](#5-como-executar)
6. [Exemplos de Saída](#6-exemplos-de-saída)
7. [Decisões de Design](#7-decisões-de-design)
8. [Declaração de Uso de IA](#8-declaração-de-uso-de-ia)

---

## 1. Objetivo

Implementar um pipeline de **Retrieval-Augmented Generation (RAG) de nível de produção** que resolve o problema de lacuna semântica entre linguagem coloquial de pacientes e jargão técnico de manuais médicos.

---

## 2. Arquitetura do Pipeline

```
Query coloquial do usuário
         │
         ▼
┌─────────────────────────┐
│  PASSO 2 — HyDE         │  LLM "alucina" documento técnico hipotético
│  (Query Transformation)  │  que serve de âncora no espaço vetorial dos manuais
└──────────┬──────────────┘
           │  vetor do documento hipotético
           ▼
┌─────────────────────────┐
│  PASSO 3 — Bi-Encoder   │  ANN search no grafo HNSW → Top-10 candidatos
│  (HNSW ANN Retrieval)   │  (busca em O(log n), sub-linear em memória)
└──────────┬──────────────┘
           │  10 documentos candidatos
           ▼
┌─────────────────────────┐
│  PASSO 4 — Cross-Encoder│  Re-ranking com atenção bilateral
│  (Re-ranking Fino)      │  [CLS] Query [SEP] Doc → score de relevância preciso
└──────────┬──────────────┘
           │  Top-3 documentos finais
           ▼
  Injeção no contexto do LLM gerador (RAG clássico)
```

**Passo 1 (Indexação HNSW)** precede todos e é executado uma única vez.

---

## 3. Análise Técnica — HNSW vs. KNN Exato

### 3.1 O que é o HNSW?

**Hierarchical Navigable Small World (HNSW)** é um algoritmo de índice vetorial baseado em grafos hierárquicos. Ele organiza os vetores em múltiplas camadas (layers), onde camadas superiores contêm poucos nós com arestas "longas" (saltos de alta cobertura) e camadas inferiores contêm todos os nós com arestas "curtas" (vizinhança local refinada). A busca começa no topo e "desce" greedy até a camada base.

### 3.2 Hiperparâmetros e seu impacto no consumo de RAM

| Hiperparâmetro | Papel | Efeito no consumo de RAM |
|---|---|---|
| **`M`** | Número máximo de arestas bidirecionais por nó em cada camada | RAM ∝ `M × n_vetores × dim × sizeof(float32)`. Dobrando M de 16 → 32, o grafo consome ~2× mais memória de adjacência |
| **`ef_construction`** | Tamanho da fila de candidatos durante a inserção | Afeta apenas o tempo de construção; **não** aumenta a memória do grafo persistido, mas sim o uso de RAM temporário durante o build |

**Fórmula de memória aproximada do grafo HNSW:**

```
RAM ≈ n_vetores × (dim × 4 bytes) + n_vetores × M × 2 × 4 bytes
           └── vetores em si ──┘    └──── arestas do grafo ────┘
```

Para 1 milhão de vetores de dimensão 1536 (OpenAI text-embedding-3-small) com M = 16:

```
Vetores : 1.000.000 × 1536 × 4 B  ≈  5,86 GB
Arestas : 1.000.000 × 16  × 2 × 4 B  ≈  0,12 GB
Total   ≈  6 GB
```

### 3.3 Comparação com KNN Exato (Flat Index)

| Critério | KNN Exato (Flat) | HNSW |
|---|---|---|
| **Complexidade de busca** | O(n × d) — linear | O(log n × M × d) — sub-linear |
| **Uso de RAM** | Apenas os vetores: `n × d × 4 bytes` | Vetores + grafo: cresce com `M` |
| **Recall** | 100% (exato) | 95–99% (aproximado, configurável via `ef_search`) |
| **Latência (1M vetores)** | ~segundos | ~milissegundos |
| **Custo de construção** | Nenhum (apenas cópia) | Alto (controlado por `ef_construction`) |
| **Indicado para** | Datasets pequenos (< 100 k) | Datasets grandes em produção |

### 3.4 Conclusão prática

Em servidores com corpora de milhões de documentos médicos, o **KNN exato é inviável**: uma busca em 10 milhões de vetores de 1536 dimensões exigiria ~60 GB de varredura sequencial a cada query. O HNSW com `M=16` e `ef_construction=200` oferece latência de busca ~1.000× menor com recall > 97%, ao custo de ~15% de RAM adicional para armazenar o grafo. Aumentar `M` melhora o recall mas encarece linearmente o índice — o ponto ótimo para produção médica é `M ∈ [12, 24]`.

---

## 4. Estrutura do Projeto

```
lab09_rag_avancado/
├── rag_pipeline.py        # Pipeline principal (4 passos)
├── data/
│   ├── __init__.py
│   └── medical_corpus.py  # 25 fragmentos de manuais médicos simulados
├── requirements.txt       # Dependências
├── .env.example           # Template de variáveis de ambiente
├── .gitignore
└── README.md              # Este arquivo
```

---

## 5. Como Executar

### 5.1 Pré-requisitos

- Python 3.11+
- Chave de API da OpenAI

### 5.2 Instalação

```bash
# Clonar o repositório
git clone <URL_DO_REPOSITÓRIO>
cd lab09_rag_avancado

# Criar e ativar ambiente virtual
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows

# Instalar dependências
pip install -r requirements.txt
```

### 5.3 Configuração

```bash
# Copiar o template de variáveis de ambiente
cp .env.example .env

# Editar .env e inserir sua chave
# OPENAI_API_KEY=sk-...
```

### 5.4 Execução

```bash
python rag_pipeline.py
```

A saída no console mostrará, de forma estruturada:
- **Passo 1**: confirmação de indexação dos 25 documentos no HNSW
- **Passo 2**: o documento hipotético gerado pelo HyDE
- **Passo 3**: tabela com os Top-10 documentos (score de cosseno)
- **Passo 4**: tabela com os Top-3 re-rankeados (score Cross-Encoder) e trechos

---

## 6. Exemplos de Saída

### Query 1

```
"dor de cabeça latejante, luz incomodando e enjoo quando me mexo"
```

**Passo 2 — HyDE transforma para:**

> "Cefaleia pulsátil de alta intensidade associada à fotofobia e náuseas
> constitui quadro compatível com enxaqueca (migrânea), CID-10 G43..."

**Passo 3 — Top-10 Bi-Encoder (exemplo):**

```
 #   Score Cosseno  Título
──────────────────────────────────────────────────────────────────────
 1          0.9312  Cefaleia Pulsátil e Fotofobia — Diagnóstico Diferencial
 2          0.8741  Crise Hipertensiva — Emergência vs. Urgência Hipertensiva
 3          0.8523  Avaliação Neurológica da Perda de Consciência Transitória
...
```

**Passo 4 — Top-3 Cross-Encoder:**

```
 #    Score CE  Score Cos  Título
──────────────────────────────────────────────────────────────────────
 1      8.7241     0.9312  Cefaleia Pulsátil e Fotofobia — Diagnóstico Diferencial
 2      3.1205     0.8523  Avaliação Neurológica da Perda de Consciência Transitória
 3      1.8932     0.8741  Crise Hipertensiva — Emergência vs. Urgência Hipertensiva
```

> Note: o Cross-Encoder reordenou os resultados 2 e 3 do Bi-Encoder — demonstrando o valor do re-ranking fino.

---

## 7. Decisões de Design

| Decisão | Alternativa considerada | Justificativa |
|---|---|---|
| ChromaDB com `hnsw:space=cosine` | FAISS IndexHNSWFlat | ChromaDB expõe os hiperparâmetros HNSW diretamente nos metadados da coleção e tem API de alto nível mais legível para fins pedagógicos |
| `text-embedding-3-small` | BERT via HuggingFace | Qualidade superior de embedding multilingual sem necessidade de GPU local |
| `gpt-4o-mini` para HyDE | `gpt-3.5-turbo` | Melhor compreensão de terminologia médica a custo baixo |
| Query original no Cross-Encoder | Documento hipotético HyDE | O Cross-Encoder avalia relevância semântica real entre a intenção do usuário e o documento — usar HyDE aqui adicionaria ruído |
| `temperature=0.3` no HyDE | `temperature=0.0` | Leve criatividade mantém riqueza lexical sem perder coerência clínica |
| `cosine_sim = 1 - distance` | `1 - distance/2` | Fórmula correta verificada empiricamente: ChromaDB retorna `distance = 1 - cosine_sim`, portanto a inversão é direta |

---

## 8. Validação Técnica

### 8.1 Fórmula de similaridade de cosseno — verificação empírica

A conversão de distância ChromaDB para similaridade de cosseno foi verificada
com vetores de referência de similaridade conhecida:

```python
# Vetores com similaridades de cosseno exatas:
# [1,0] vs [1,0]  → cosine_sim =  1.0 (idênticos)
# [1,0] vs [0,1]  → cosine_sim =  0.0 (ortogonais)
# [1,0] vs [-1,0] → cosine_sim = -1.0 (opostos)

# ChromaDB hnsw:space=cosine retorna:
# identical : distance=0.000000  → 1 - d = 1.0  ✓
# orthogonal: distance=1.000000  → 1 - d = 0.0  ✓
# opposite  : distance=2.000000  → 1 - d = -1.0 ✓

# Fórmula INCORRETA (1 - d/2) daria: 1.0, 0.5, 0.0 — errado para ortogonal/oposto
cosine_similarity = round(1.0 - distance, 4)  # fórmula correta
```

### 8.2 Checklist de qualidade do código

- [x] Sintaxe Python válida (`ast.parse`)
- [x] Todos os métodos com type hints completos
- [x] Nenhuma linha acima de 100 caracteres
- [x] Nenhum `bare except`
- [x] Nenhuma chave de API exposta no código-fonte
- [x] `logging.basicConfig` apenas em `main()` (não polui módulos importadores)
- [x] Guards para: `api_key` vazia, `user_query` vazia, `documents` vazia,
  `content` None do LLM, `hypothetical_doc` vazio, `top_candidates` vazio
- [x] `np.atleast_1d` para normalizar `predict()` escalar vs ndarray
- [x] Imports de terceiros em ordem alfabética (PEP 8)
- [x] `chromadb.EphemeralClient()` — API atual (não depreciada)

---

## 9. Declaração de Uso de IA

> **"Partes deste laboratório foram geradas/complementadas com IA, revisadas e validadas por Dimmy"**


Especificamente:
- Os **25 fragmentos do corpus médico** foram gerados com auxílio de IA generativa e revisados para garantir terminologia clínica correta (CID-10, biomarcadores, protocolos como ATLS, ARDSNet, Sepsis-3).
- O **código** foi estruturado, revisado e validado manualmente, com compreensão integral de cada componente arquitetural (HNSW, HyDE, Bi-Encoder vs. Cross-Encoder).
- A **análise de hiperparâmetros HNSW** (Seção 3) foi elaborada com base em literatura técnica revisada criticamente.
