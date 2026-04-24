# Assignment 4 - Retrieval-Augmented Generation

**Course:** COMP 263 - Deep Learning
**Weight:** Lab Assignment 4
**Student:** Izzet Abidi (300898230)

---

## 1. Overview

This assignment builds a **Retrieval-Augmented Generation (RAG)** system over
six public-domain works by **Fyodor Dostoevsky** and **Friedrich Nietzsche**.
The system downloads the corpus from Project Gutenberg, cleans the text,
chunks the documents, embeds each chunk with Sentence-Transformers, stores the
vectors in FAISS, retrieves the most relevant passages for a user question,
and asks a locally served LLM to answer using only the retrieved context.

The required interface is implemented as a **Streamlit app** in `app.py`.
Users can enter a question, choose the search scope by author, adjust top-k
retrieval, view the generated answer, and inspect every retrieved passage with
its source and similarity score.

## 1A. Pipeline Diagram

The Mermaid source for this diagram is also stored in `mermaid.mmd`.

```mermaid
flowchart LR
    A["Project Gutenberg URLs"] --> B["src/fetch_corpus.py"]
    B --> C["data/raw/*.txt"]
    C --> D["src/preprocess.py"]
    D --> E["data/clean/*.txt"]
    E --> F["src/chunking.py<br/>recursive split + overlap"]
    F --> G["src/build_index.py"]
    G --> H["MiniLM embeddings"]
    H --> I["FAISS IndexFlatIP"]
    J["app.py / Streamlit"] --> K["User question"]
    K --> L["Query embedding"]
    L --> I
    I --> M["Top-k chunks"]
    M --> N["src/rag.py"]
    N --> O["vLLM OpenAI-compatible endpoint"]
    O --> P["Grounded answer with citations"]
    M --> P
```

---

## 2. Exercise Breakdown

### Exercise 1: Build a RAG System Using a Custom Dataset (20 marks)

**Objective:** Design and implement a complete RAG pipeline using a text-based
dataset of at least 3 documents or 10 pages, then evaluate the system on 5-7
questions and present the results in a short report.

**What the project does:**

1. **Dataset Selection** - Uses six Project Gutenberg plain-text books: three
   by Dostoevsky and three by Nietzsche. The corpus is public, text-based,
   non-sensitive, and much larger than the assignment minimum.

2. **Corpus Fetching** - `src/fetch_corpus.py` downloads each source file into
   `data/raw/` using explicit Project Gutenberg URLs from `src/config.py`.

3. **Preprocessing** - `src/preprocess.py` strips Gutenberg headers, footers,
   production notes, legacy end markers, and common transcriber notes. It then
   normalizes whitespace while preserving paragraph boundaries.

4. **Chunking** - `src/chunking.py` implements a custom recursive character
   splitter without LangChain. It prioritizes paragraph breaks, then lines,
   sentence boundaries, words, and finally character-level fallback splitting.

5. **Embeddings** - `src/build_index.py` embeds every chunk with
   `sentence-transformers/all-MiniLM-L6-v2`, producing normalized 384-dimensional
   vectors.

6. **Vector Storage** - Embeddings are stored in a FAISS `IndexFlatIP` index.
   Because vectors are L2-normalized, inner product search is equivalent to
   cosine similarity.

7. **Retrieval** - `src/rag.py` embeds the user query, retrieves the top-k
   matching chunks, supports optional author filtering, and returns source
   metadata for display.

8. **Generation** - The generator calls a local vLLM OpenAI-compatible endpoint.
   The system prompt requires context-only answers and inline citations such as
   `[1]`, `[2]`, and `[3]`.

9. **Interface** - `app.py` provides the Streamlit UI required by the
   assignment. It shows both the answer and the retrieved chunks.

10. **Evaluation** - `notebooks/evaluation.ipynb` runs seven test questions and
    creates a summary table for manual Correct / Partial / Incorrect grading.

**Key design decisions:**

- **Dataset choice:** Dostoevsky and Nietzsche provide long-form narrative and
  dense philosophical prose. Their overlapping themes make retrieval more
  interesting than isolated factual lookup.
- **Chunk size 1,000 characters:** Large enough to preserve a paragraph or
  short argument, but small enough that the retriever stays focused.
- **Overlap 200 characters:** A 20% overlap keeps boundary context when a
  relevant sentence crosses chunk edges.
- **MiniLM embeddings:** `all-MiniLM-L6-v2` is small, fast, CPU-friendly, and
  strong enough for short-passage semantic retrieval.
- **Exact FAISS search:** The corpus produces only about 8,100 chunks, so exact
  search is simple, deterministic, and fast.
- **Context-only generation:** The prompt explicitly tells the LLM not to
  invent facts, quotes, chapter numbers, or citations.

**File manifest:**

| File | Purpose |
|------|---------|
| `app.py` | Streamlit app for asking questions, filtering by author, and viewing retrieved passages |
| `requirements.txt` | Python dependencies for retrieval, indexing, UI, and evaluation |
| `report.md` | 2-3 page assignment report draft |
| `src/config.py` | Corpus URLs, paths, chunking settings, embedding model, and vLLM endpoint settings |
| `src/fetch_corpus.py` | Downloads the six Project Gutenberg source texts |
| `src/preprocess.py` | Cleans Gutenberg boilerplate and normalizes text |
| `src/chunking.py` | Custom recursive character splitter |
| `src/build_index.py` | Chunks, embeds, and writes the FAISS index plus chunk metadata |
| `src/rag.py` | Retrieval and generation pipeline |
| `notebooks/evaluation.ipynb` | Seven-question evaluation notebook |

---

## 3. Runbook

### Prerequisites

```bash
$ cd Assign4/
$ python -m venv .venv
$ source .venv/bin/activate
$ pip install -r requirements.txt
```

The RAG code requires Python 3.10+ and the packages listed in
`requirements.txt`. vLLM is run separately because it is a serving dependency
and may need a hardware-specific install.

### Build the corpus and index

```bash
$ python -m src.fetch_corpus
$ python -m src.preprocess
$ python -m src.build_index
```

This creates ignored local artifacts under `data/raw/`, `data/clean/`, and
`data/index/`. The repository stores the code and source links rather than the
generated text/index files.

### Start the local LLM server

```bash
$ vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct --max-model-len 8192
```

By default the app expects vLLM at `http://localhost:8000/v1` and the model
name `meta-llama/Meta-Llama-3.1-8B-Instruct`. Override those values if needed:

```bash
$ export VLLM_BASE_URL="http://localhost:8000/v1"
$ export VLLM_MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"
```

### Run the app

```bash
$ streamlit run app.py
```

Open `http://localhost:8501`. Enter a question, choose the author scope in the
sidebar, adjust the top-k slider, and inspect the retrieved passages below the
answer.

### Run the evaluation

```bash
$ jupyter notebook notebooks/evaluation.ipynb
```

Run all cells, read the retrieved chunks and generated answers, then transfer
the Correct / Partial / Incorrect verdicts into `report.md`.

### Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| `ModuleNotFoundError: No module named 'faiss'` | Dependencies not installed in the active environment | Activate `.venv` and run `pip install -r requirements.txt` |
| `FileNotFoundError: corpus.faiss not found` | Index has not been built yet | Run `python -m src.fetch_corpus`, `python -m src.preprocess`, then `python -m src.build_index` |
| Streamlit loads but answering fails | vLLM server is not running | Start `vllm serve ...` in another terminal |
| Model name error from vLLM | `VLLM_MODEL` does not match the served model | Set `export VLLM_MODEL="<served model name>"` |
| Weak or off-topic answers | Retrieved chunks do not contain enough context | Increase top-k or remove the author filter |
| App starts but cannot produce a final answer | Retrieval data exists but the vLLM endpoint is unavailable | The UI can still show evidence, but final generation requires a reachable vLLM server |

---

## 4. Expected Results

| Component | Expected behavior |
|-----------|-------------------|
| Corpus size | Six cleaned books and roughly 8,100 chunks |
| Retrieval | Top-k chunks should come from the relevant book or author for specific questions |
| Generation | Answers should cite retrieved passages with bracketed citations |
| Hallucination control | If the answer is not present in retrieved context, the model should say so |
| Interface | Streamlit should show the answer and expandable retrieved passages |
| Evaluation | Seven questions should be judged as Correct, Partial, or Incorrect in `report.md` |

**Retriever observations:**
- Character-level chunking works well for broad concept questions and plot
  questions, especially when the query names a character, book, or concept.
- Nietzsche questions can be sensitive to translation wording because older
  English translations use archaic phrasing.
- Author filtering is useful for comparing Dostoevsky-only, Nietzsche-only,
  and cross-corpus retrieval behavior.

**Generator observations:**
- The citation format makes it easier to inspect whether the answer is grounded.
- Temperature `0.2` keeps answers concise and reduces unsupported speculation.
- Final quality depends on both retrieval quality and the local vLLM model.

---

## 5. Topics Learned

**Retrieval-Augmented Generation**
- RAG pipeline design - combining search over external documents with LLM-based answer synthesis
- Grounded generation - constraining the model to retrieved passages instead of model memory
- Citation-aware prompting - formatting context so generated answers can cite sources

**Text Processing**
- Corpus acquisition - downloading public-domain text files from Project Gutenberg
- Boilerplate removal - stripping headers, footers, production notes, and license text
- Recursive chunking - splitting long documents while preserving natural boundaries where possible

**Vector Search**
- Sentence embeddings - converting chunks and queries into comparable dense vectors
- Cosine similarity - comparing normalized embeddings through inner product
- FAISS indexing - storing vectors for efficient nearest-neighbor retrieval

**LLM Serving and UI**
- vLLM - serving a local instruction model through an OpenAI-compatible API
- Streamlit - building a simple interactive question-answering app
- Environment configuration - using `VLLM_BASE_URL` and `VLLM_MODEL` to avoid hard-coding deployment details

---

## 6. Definitions and Key Concepts

| Term | Definition |
|------|------------|
| **Retrieval-Augmented Generation (RAG)** | A pattern where relevant external documents are retrieved first, then passed to an LLM as context for answer generation. |
| **Chunk** | A smaller passage created from a long document so retrieval can target focused sections instead of entire books. |
| **Chunk Overlap** | Repeated text between adjacent chunks, used to preserve context near boundaries. |
| **Embedding** | A dense numeric vector representing the semantic meaning of text. |
| **Vector Database** | A storage and search system for embeddings; this project uses FAISS. |
| **FAISS** | Facebook AI Similarity Search, a library for efficient nearest-neighbor search over vectors. |
| **IndexFlatIP** | A FAISS exact-search index that ranks vectors by inner product. |
| **Cosine Similarity** | A similarity measure based on vector angle; with normalized vectors, it is equivalent to inner product. |
| **Top-k Retrieval** | Returning the k most similar chunks for a query. |
| **vLLM** | A high-throughput local LLM inference server with an OpenAI-compatible API. |
| **Context Window** | The maximum amount of text the LLM can consider in one request. |
| **Hallucination** | An unsupported or invented answer produced by an LLM. |

---

## 7. Potential Improvements and Industry Considerations

The current design is intentionally simple and defensible for a course assignment. It is effective on a small corpus, but your professor is right that it is not the scaling shape you would keep for a much larger system. Exact vector search over a single in-memory index, coupled directly to the UI process, is easy to reason about but not ideal once the corpus, concurrency, or deployment footprint grows.

### Retrieval Strategy

| Current Approach | Industry Alternative | Trade-off |
|-----------------|---------------------|-----------|
| Character chunking | Section-aware chunking by chapter or numbered aphorism | Better source boundaries, but more corpus-specific parsing |
| Recursive separator chunking | Semantic chunking using embedding similarity between adjacent spans | Better topic coherence, but extra compute and more tuning |
| MiniLM embeddings | Larger embedding model such as BGE or E5 | Better semantic matching, but more memory and slower indexing |
| Exact FAISS search | Approximate FAISS index for millions of chunks | Scales further, but adds tuning and slight recall loss |
| Dense retrieval only | Hybrid BM25 + dense retrieval | Better recall on names and quoted phrases, but more system complexity |

### Generation Strategy

| Current Approach | Industry Alternative | Trade-off |
|-----------------|---------------------|-----------|
| Single prompt with retrieved chunks | Reranking before generation | Better context selection, but adds another model step |
| Context-only system prompt | Structured answer validation | Stronger hallucination control, but more implementation complexity |
| Local 8B instruct model | Larger hosted or local model | Higher answer quality, but more cost or hardware demand |
| UI process calls generator directly | Separate retrieval and generation services | More scalable deployment, but more moving parts |

### Evaluation

| Current Approach | Industry Alternative | Trade-off |
|-----------------|---------------------|-----------|
| Seven manual questions | Larger benchmark set with expected answers | More reliable measurement, but more setup time |
| Manual Correct / Partial / Incorrect labels | LLM-as-judge plus human review | Faster scoring, but judge model bias must be managed |
| Similarity scores only | Retrieval precision/recall against labeled gold chunks | More rigorous retrieval analysis, but requires labeled ground truth |

### Where the Baseline Holds Up

- **A six-book corpus** is large enough to demonstrate real retrieval behavior
  while still small enough for exact FAISS search on a laptop.
- **Local vLLM serving** avoids sending the corpus or questions to a hosted API,
  which is appropriate for assignments that emphasize implementation control.
- **A simple Streamlit UI** satisfies the creativity/interface requirement
  without adding unnecessary web framework complexity.

### Practical Upgrade Path

If this project had to scale past a class assignment, the first upgrades I would make are:

1. **Move to structure-aware chunking** by chapter, heading, or aphorism before applying overlap.
2. **Test semantic chunking** if the goal is higher chunk coherence than fixed recursive splitting.
3. **Add hybrid retrieval and reranking** so recall and final context quality improve together.
4. **Swap exact FAISS search for ANN** once the chunk count is large enough that exact search is wasteful.
5. **Decouple the UI from indexing and generation** so preprocessing, retrieval, and serving can scale independently.
