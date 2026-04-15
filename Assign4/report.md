# Assignment 4 — RAG over Dostoevsky & Nietzsche
**Course:** COMP 263 Deep Learning · **Author:** Izzet · **Date:** April 2026

---

## 1. Dataset

**Source.** Six full-text works downloaded from Project Gutenberg
(public-domain plain UTF-8 files):

| Author | Work | Translator | Gutenberg ID |
|---|---|---|---|
| Fyodor Dostoevsky | *Crime and Punishment* | Constance Garnett | 2554 |
| Fyodor Dostoevsky | *Notes from the Underground* | Constance Garnett | 600 |
| Fyodor Dostoevsky | *The Brothers Karamazov* | Constance Garnett | 28054 |
| Friedrich Nietzsche | *Thus Spake Zarathustra* | Thomas Common | 1998 |
| Friedrich Nietzsche | *Beyond Good and Evil* | Helen Zimmern | 4363 |
| Friedrich Nietzsche | *The Antichrist* | H. L. Mencken | 19322 |

**Why this corpus.** Two related but stylistically distinct authors give the
retriever an interesting test: long-form 19th-century narrative prose
(Dostoevsky) versus dense aphoristic philosophy (Nietzsche). The works also
share themes — morality beyond conventional good/evil, suffering, faith and
its rejection — so cross-author questions ("compare Ivan Karamazov's view of
God with Zarathustra's") are a natural way to stress the system. The corpus
is roughly 1,000 pages, well above the assignment minimum.

**What questions it supports.** Plot/character questions ("how does Sonia
respond when Raskolnikov confesses?"), philosophical-concept questions ("what
does Nietzsche mean by eternal recurrence?"), and quote attribution.

---

## 2. Method

### 2.1 Preprocessing
`src/preprocess.py` strips the Project Gutenberg legal header/footer using
the standard `*** START / *** END OF THE PROJECT GUTENBERG EBOOK ***`
markers, normalises line endings, removes the `_underscored italic_` markup
that Gutenberg uses, collapses runs of three or more newlines down to a
paragraph break, and trims trailing whitespace per line.

### 2.2 Chunking
A custom recursive character splitter (`src/chunking.py`, ~80 lines, no
LangChain dependency) tries separators in priority order — paragraph break →
single newline → sentence boundary → word → character — then packs the
resulting pieces into windows with a sliding overlap.

| Parameter | Value | Justification |
|---|---|---|
| Chunk size | **1,000 characters** (~200 tokens) | Large enough to hold a complete paragraph or aphorism; small enough that 5 chunks fit comfortably in the LLM context window. Smaller chunks (250–500 chars) fragment Nietzsche's already-aphoristic style mid-thought; larger chunks (>2,000) dilute retrieval signal. |
| Overlap | **200 characters (20 %)** | Preserves context across boundaries — important when the relevant sentence sits exactly on a chunk seam. 20 % is the typical sweet spot reported in RAG ablations; we kept it constant to avoid over-tuning. |

### 2.3 Embeddings
`sentence-transformers/all-MiniLM-L6-v2` — 384-dim, ~80 MB, runs on CPU at
roughly 1,000 chunks/sec. Chosen for speed and zero dependencies; embedding
quality is competitive with much larger models on short-passage retrieval.
Vectors are L2-normalised at encode time so that FAISS inner-product search
becomes cosine similarity.

### 2.4 Vector store
FAISS `IndexFlatIP` — exact (non-approximate) inner-product search. With only
a few thousand chunks this is faster than building an approximate index and
guarantees deterministic top-k results.

### 2.5 Generation
A locally-served vLLM instance running `meta-llama/Meta-Llama-3.1-8B-Instruct`
on an RTX 3090. vLLM exposes an OpenAI-compatible API, so the Python client
is a one-line swap from a hosted API.

The system prompt instructs the model to **answer only from the provided
context** and **say so plainly when the answer isn't there**. Retrieved
chunks are formatted with bracketed citations `[1] [2] ...` so the model can
cite inline. Generation uses temperature 0.2 and a 512-token cap.

### 2.6 Interface
A Streamlit app (`app.py`) with a single text input, a sidebar to filter by
author (both / Dostoevsky / Nietzsche) and adjust top-k, and an expandable
view of every retrieved passage with its similarity score and source.

---

## 3. Results

Seven questions should be evaluated end-to-end using
`notebooks/evaluation.ipynb`. After running the notebook, inspect each answer
against the original text and fill in the summary below.

| # | Question | Verdict | Notes |
|---|---|---|---|
| 1 | Raskolnikov's "extraordinary men" | _fill in_ | _from notebook_ |
| 2 | Ivan Karamazov's rebellion against God | _fill in_ | |
| 3 | Underground man on the "man of action" | _fill in_ | |
| 4 | Nietzsche's "eternal recurrence" | _fill in_ | |
| 5 | Master vs. slave morality (BG&E) | _fill in_ | |
| 6 | Nietzsche's critique of pity (Antichrist) | _fill in_ | |
| 7 | Sonia's response to Raskolnikov's confession | _fill in_ | |

> **Note for the submitter:** run the notebook against your local vLLM
> server, then transfer the verdicts and any short comments into the table
> above before submitting.

Aggregate (to fill in): **X / 7 Correct, Y / 7 Partial, Z / 7 Incorrect.**

---

## 4. Challenges

- **Translation drift.** Different translators render the same German /
  Russian phrase differently. A Nietzsche question phrased in modern English
  can miss chunks where Common's 1909 prose uses archaic wording. The
  retriever's tolerance to this is one of the more interesting things to
  inspect when reading the evaluation transcript.
- **Aphoristic prose vs. fixed chunk size.** *Beyond Good and Evil* is
  organised as numbered sections of wildly different lengths. A 1,000-char
  window often splits a single section across two chunks; the 20 % overlap
  mitigates but doesn't fully solve this. Section-aware chunking would be a
  natural next step.
- **Gutenberg boilerplate variation.** The wrapper text isn't perfectly
  consistent across the six files (older editions use `THIS PROJECT
  GUTENBERG`, newer ones `THE PROJECT GUTENBERG`). The regex was loosened
  accordingly.
- **Hallucination control.** Without the explicit "do not invent quotes"
  instruction the LLM cheerfully produced plausible but fictional
  attributions. The system prompt now also forbids inventing chapter or
  section numbers.

---

## 5. References

- Project Gutenberg — <https://www.gutenberg.org>
- Sentence-Transformers — Reimers & Gurevych (2019), *Sentence-BERT*.
- FAISS — Johnson, Douze & Jégou (2017), *Billion-scale similarity search
  with GPUs*.
- vLLM — Kwon et al. (2023), *Efficient Memory Management for Large Language
  Model Serving with PagedAttention*.
- Llama 3.1 — Meta AI, 2024.
