"""Streamlit UI for the Dostoevsky + Nietzsche RAG system.

Run with:
    streamlit run app.py
"""
from __future__ import annotations

import streamlit as st

from src.config import TOP_K, VLLM_MODEL
from src.rag import RAG


st.set_page_config(
    page_title="Dostoevsky x Nietzsche RAG",
    page_icon="📚",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading FAISS index and embedding model...")
def get_rag() -> RAG:
    return RAG()


def main() -> None:
    st.title("Dostoevsky × Nietzsche — RAG Assistant")
    st.caption(
        "Ask questions about *Crime and Punishment*, *Notes from the Underground*, "
        "*The Brothers Karamazov*, *Thus Spake Zarathustra*, *Beyond Good and Evil*, "
        "and *The Antichrist*. Answers are grounded in retrieved passages only."
    )

    with st.sidebar:
        st.header("Settings")
        author_choice = st.radio(
            "Search scope",
            ["Both authors", "Dostoevsky only", "Nietzsche only"],
            index=0,
        )
        top_k = st.slider("Passages to retrieve (k)", 1, 10, TOP_K)
        st.markdown("---")
        st.markdown(f"**LLM:** `{VLLM_MODEL}`")
        st.markdown("**Embedder:** `MiniLM-L6-v2` (384-dim)")
        st.markdown("**Vector DB:** FAISS (cosine / inner product)")

    author_filter: list[str] | None
    if author_choice == "Dostoevsky only":
        author_filter = ["Fyodor Dostoevsky"]
    elif author_choice == "Nietzsche only":
        author_filter = ["Friedrich Nietzsche"]
    else:
        author_filter = None

    rag = get_rag()

    question = st.text_input(
        "Question",
        placeholder=(
            "e.g. What does Raskolnikov mean by 'extraordinary men'?"
        ),
    )

    if not question:
        st.info("Type a question above and press Enter.")
        return

    with st.spinner("Retrieving passages and generating answer..."):
        result = rag.answer(question, k=top_k, authors=author_filter)

    st.subheader("Answer")
    st.write(result["answer"])

    st.subheader("Retrieved passages")
    if not result["chunks"]:
        st.warning("No passages matched the current author filter.")
    for chunk in result["chunks"]:
        with st.expander(
            f"[{chunk['rank']}] {chunk['author']} — {chunk['title']} "
            f"(chunk {chunk['chunk_index']}, score {chunk['score']:.3f})"
        ):
            st.write(chunk["text"])


if __name__ == "__main__":
    main()
