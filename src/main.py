from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.ingest import ingest
from src.query import query_rag_pipeline


def run_cli() -> None:
    print("Document Q&A Bot")
    print("Type 'ingest' to index documents, 'exit' to quit, or ask a question.")

    while True:
        user_input = input("\n> ").strip()

        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit"}:
            break

        if user_input.lower() == "ingest":
            count = ingest(reset=True)
            print(f"Indexed {count} chunks.")
            continue

        try:
            result = query_rag_pipeline(user_input)
        except Exception as exc:
            print(f"Error: {exc}")
            continue

        print(f"\n{result['answer']}")

        if result["citations"]:
            print("\nCitations:")
            for citation in result["citations"]:
                print(f"- {citation}")


def run_streamlit() -> None:
    import streamlit as st

    st.set_page_config(page_title="Document Q&A Bot", layout="wide")
    st.title("Document Q&A Bot")

    with st.sidebar:
        st.subheader("Index")
        if st.button("Rebuild document index", type="primary"):
            with st.spinner("Reading documents and creating embeddings..."):
                count = ingest(reset=True)
            st.success(f"Indexed {count} chunks.")

    question = st.text_input("Ask a question from your indexed documents")

    if question:
        with st.spinner("Searching documents and generating an answer..."):
            try:
                result = query_rag_pipeline(question)
            except Exception as exc:
                st.error(str(exc))
                return

        st.markdown("### Answer")
        st.write(result["answer"])

        if result["citations"]:
            st.markdown("### Citations")
            for citation in result["citations"]:
                st.write(f"- {citation}")

        with st.expander("Retrieved context"):
            for context in result["raw_context"]:
                st.write(context)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the document Q&A bot.")
    parser.add_argument("--cli", action="store_true", help="Force CLI mode.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.cli:
        run_cli()
        return

    try:
        import streamlit.runtime.scriptrunner as scriptrunner

        if scriptrunner.get_script_run_ctx() is not None:
            run_streamlit()
            return
    except Exception:
        pass

    run_cli()


if __name__ == "__main__":
    main()
