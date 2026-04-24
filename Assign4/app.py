"""Streamlit UI for the Dostoevsky + Nietzsche RAG system.

Run with:
    streamlit run app.py
"""
from __future__ import annotations

from collections import Counter
from html import escape
from statistics import mean

import streamlit as st

from src.config import TOP_K, VLLM_BASE_URL, VLLM_MODEL
from src.rag import RAG, RetrievedChunk


st.set_page_config(
    page_title="Literary RAG Lab",
    page_icon="RAG",
    layout="wide",
)


EXAMPLE_QUESTIONS = [
    "What does Raskolnikov mean by extraordinary men?",
    "How does Sonia respond when Raskolnikov confesses?",
    "What is Ivan Karamazov's rebellion against God?",
    "What does Nietzsche say about master and slave morality?",
    "What is Nietzsche's critique of pity in The Antichrist?",
    "Compare Dostoevsky and Nietzsche on suffering.",
]


AUTHOR_SCOPES = {
    "Both authors": None,
    "Dostoevsky only": ["Fyodor Dostoevsky"],
    "Nietzsche only": ["Friedrich Nietzsche"],
}


CUSTOM_CSS = """
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1180px;
    }
    .rag-title {
        font-size: 2.4rem;
        line-height: 1.1;
        font-weight: 750;
        margin-bottom: 0.35rem;
    }
    .rag-subtitle {
        color: #4b5563;
        font-size: 1.02rem;
        margin-bottom: 1.15rem;
    }
    .status-strip {
        border: 1px solid #d6d3d1;
        border-left: 5px solid #0f766e;
        border-radius: 8px;
        padding: 0.9rem 1rem;
        background: #fafaf9;
        margin-bottom: 1rem;
    }
    .source-card {
        border: 1px solid #d6d3d1;
        border-radius: 8px;
        padding: 0.9rem 1rem;
        margin-bottom: 0.85rem;
        background: #ffffff;
    }
    .source-meta {
        color: #57534e;
        font-size: 0.9rem;
        margin-bottom: 0.45rem;
    }
    .score-pill {
        display: inline-block;
        border: 1px solid #99f6e4;
        border-radius: 8px;
        padding: 0.1rem 0.45rem;
        background: #f0fdfa;
        color: #115e59;
        font-size: 0.82rem;
        font-weight: 650;
    }
    .small-muted {
        color: #78716c;
        font-size: 0.88rem;
    }
</style>
"""


@st.cache_resource(show_spinner="Loading FAISS index and embedding model...")
def get_rag() -> RAG:
    return RAG()


def corpus_summary(rag: RAG) -> dict:
    title_counts = Counter(chunk["title"] for chunk in rag.chunks)
    author_counts = Counter(chunk["author"] for chunk in rag.chunks)
    avg_chars = round(mean(len(chunk["text"]) for chunk in rag.chunks))
    return {
        "chunks": len(rag.chunks),
        "books": len(title_counts),
        "authors": len(author_counts),
        "avg_chars": avg_chars,
        "title_counts": title_counts,
        "author_counts": author_counts,
    }


def set_question(question: str) -> None:
    st.session_state.question = question


def format_scope(authors: list[str] | None) -> str:
    if not authors:
        return "Full corpus"
    if authors == ["Fyodor Dostoevsky"]:
        return "Dostoevsky"
    return "Nietzsche"


def retrieval_strength(chunks: list[RetrievedChunk]) -> tuple[str, float]:
    if not chunks:
        return "No retrieval", 0.0
    top_score = max(chunks[0].score, 0.0)
    if top_score >= 0.68:
        return "Strong match", top_score
    if top_score >= 0.58:
        return "Usable match", top_score
    return "Weak match", top_score


def render_source_card(chunk: dict, show_full_text: bool) -> None:
    text = chunk["text"] if show_full_text else chunk["text"][:650].rstrip()
    if not show_full_text and len(chunk["text"]) > len(text):
        text += "..."
    title = escape(chunk["title"])
    author = escape(chunk["author"])
    body = escape(text)

    st.markdown(
        f"""
        <div class="source-card">
            <div>
                <strong>[{chunk["rank"]}] {title}</strong>
                <span class="score-pill">score {chunk["score"]:.3f}</span>
            </div>
            <div class="source-meta">
                {author} | chunk {chunk["chunk_index"]}
            </div>
            <div>{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def generate_result(
    rag: RAG,
    question: str,
    top_k: int,
    authors: list[str] | None,
) -> dict:
    chunks = rag.retrieve(question, k=top_k, authors=authors)
    try:
        answer = rag.generate(question, chunks)
        error = None
    except Exception as exc:
        answer = (
            "The retriever found evidence, but the LLM endpoint did not return "
            "an answer. The retrieved passages are still shown below."
        )
        error = f"{type(exc).__name__}: {exc}"
    return {
        "question": question,
        "answer": answer,
        "chunks": [chunk.to_dict() for chunk in chunks],
        "error": error,
    }


def render_examples() -> None:
    st.markdown("**Try one of these**")
    rows = [EXAMPLE_QUESTIONS[:3], EXAMPLE_QUESTIONS[3:]]
    for row in rows:
        columns = st.columns(3)
        for column, question in zip(columns, row):
            column.button(
                question,
                key=f"example-{question}",
                use_container_width=True,
                on_click=set_question,
                args=(question,),
            )


def main() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    rag = get_rag()
    summary = corpus_summary(rag)

    with st.sidebar:
        st.header("RAG Controls")
        scope_label = st.radio("Search scope", list(AUTHOR_SCOPES), index=0)
        author_filter = AUTHOR_SCOPES[scope_label]
        top_k = st.slider("Retrieved passages", 2, 10, TOP_K)
        show_full_text = st.toggle("Show full source text", value=False)

        st.divider()
        st.caption("Runtime")
        st.code(f"Model: {VLLM_MODEL}", language="text")
        st.code(f"Endpoint: {VLLM_BASE_URL}", language="text")
        st.caption("Embeddings: all-MiniLM-L6-v2")
        st.caption("Vector store: FAISS IndexFlatIP")

    st.markdown(
        """
        <div class="rag-title">Literary RAG Lab</div>
        <div class="rag-subtitle">
            Ask a grounded question across Dostoevsky and Nietzsche, then inspect
            exactly which passages shaped the answer.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="status-strip">
            <strong>{summary["books"]} books</strong> indexed into
            <strong>{summary["chunks"]:,} passages</strong>.
            Current scope: <strong>{format_scope(author_filter)}</strong>.
            Answers are generated through <strong>{VLLM_MODEL}</strong>.
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_cols = st.columns(4)
    metric_cols[0].metric("Books", summary["books"])
    metric_cols[1].metric("Indexed passages", f"{summary['chunks']:,}")
    metric_cols[2].metric("Avg. passage size", f"{summary['avg_chars']} chars")
    metric_cols[3].metric("Top-k", top_k)

    render_examples()

    if "question" not in st.session_state:
        st.session_state.question = ""

    with st.form("question-form", clear_on_submit=False):
        question = st.text_area(
            "Question",
            key="question",
            height=105,
            placeholder="Ask about a theme, character, argument, or comparison.",
        )
        submitted = st.form_submit_button(
            "Run grounded answer",
            type="primary",
            use_container_width=True,
        )

    if not submitted and not st.session_state.get("last_result"):
        st.info("Choose an example or write a question to start.")
        return

    if submitted:
        clean_question = question.strip()
        if not clean_question:
            st.warning("Enter a question first.")
            return
        with st.spinner("Retrieving passages and asking the local vLLM model..."):
            st.session_state.last_result = generate_result(
                rag,
                clean_question,
                top_k,
                author_filter,
            )
            st.session_state.last_scope = format_scope(author_filter)

    result = st.session_state.last_result
    chunks = result["chunks"]
    strength_label, top_score = retrieval_strength(
        [
            RetrievedChunk(
                rank=chunk["rank"],
                score=chunk["score"],
                text=chunk["text"],
                author=chunk["author"],
                title=chunk["title"],
                chunk_index=chunk["chunk_index"],
            )
            for chunk in chunks
        ]
    )

    tabs = st.tabs(["Answer", "Evidence", "Retrieval Diagnostics"])

    with tabs[0]:
        st.chat_message("user").write(result["question"])
        st.chat_message("assistant").write(result["answer"])
        if result["error"]:
            st.error(result["error"])

    with tabs[1]:
        if not chunks:
            st.warning("No passages matched the current search scope.")
        for chunk in chunks:
            render_source_card(chunk, show_full_text)

    with tabs[2]:
        diag_cols = st.columns(3)
        diag_cols[0].metric("Top score", f"{top_score:.3f}")
        diag_cols[1].metric("Retrieval strength", strength_label)
        diag_cols[2].metric("Scope used", st.session_state.get("last_scope", ""))

        st.progress(min(max(top_score, 0.0), 1.0))

        st.markdown("**Source distribution**")
        title_counts = Counter(chunk["title"] for chunk in chunks)
        if title_counts:
            st.bar_chart(title_counts)
        else:
            st.caption("No retrieved sources to chart.")

        with st.expander("Exact retrieved chunk metadata"):
            st.json(
                [
                    {
                        "rank": chunk["rank"],
                        "score": round(chunk["score"], 4),
                        "author": chunk["author"],
                        "title": chunk["title"],
                        "chunk_index": chunk["chunk_index"],
                    }
                    for chunk in chunks
                ]
            )


if __name__ == "__main__":
    main()
