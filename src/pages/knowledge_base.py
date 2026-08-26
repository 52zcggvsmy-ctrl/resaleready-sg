"""Administrative/demo page for session-scoped knowledge-base uploads."""

from __future__ import annotations

import logging
from typing import Any

import streamlit as st

from src.app_config import secret_or_environment
from src.rag.embeddings import OpenAIEmbeddingProvider
from src.rag.ingestion import load_document_manifest
from src.rag.runtime import SOURCE_MANIFEST, ensure_vector_store
from src.rag.uploads import (
    MAX_UPLOAD_BYTES,
    SESSION_UPLOAD_RESULTS_KEY,
    SESSION_UPLOAD_STORE_KEY,
    DuplicateUploadError,
    UploadedVectorStore,
    UploadValidationError,
    chunk_uploaded_document,
    validate_upload,
)

LOGGER = logging.getLogger(__name__)


def _get_store() -> UploadedVectorStore:
    store = st.session_state.get(SESSION_UPLOAD_STORE_KEY)
    if not isinstance(store, UploadedVectorStore):
        store = UploadedVectorStore()
        st.session_state[SESSION_UPLOAD_STORE_KEY] = store
    return store


def _show_results(results: list[dict[str, Any]]) -> None:
    if not results:
        return
    st.subheader("Processing results")
    st.dataframe(
        results,
        column_config={
            "Filename": st.column_config.TextColumn(width="medium"),
            "Status": st.column_config.TextColumn(width="small"),
            "Chunks": st.column_config.NumberColumn(format="%d"),
            "Details": st.column_config.TextColumn(width="large"),
        },
        hide_index=True,
        width="stretch",
    )


def render() -> None:
    st.title("📄 Knowledge Base")
    st.write(
        "Upload only PDF or UTF-8 TXT files to bring in other external or own sources "
        "of info for use in Ask ResaleReady. Uploads will be available only in this "
        "browser session and are not written to the repository."
    )
    st.info(
        "The curated official HDB documents remain the primary knowledge base. "
        "Uploaded content is unverified reference material and is never treated as "
        "instructions or official HDB policy."
    )

    store = _get_store()
    curated_document_count = len(load_document_manifest(SOURCE_MANIFEST))
    metric_one, metric_two, metric_three = st.columns(3)
    metric_one.metric("Curated HDB documents", curated_document_count)
    metric_two.metric("Session uploads", store.document_count)
    metric_three.metric("Uploaded chunks", store.chunk_count)

    if store.filenames:
        with st.expander("Active session uploads"):
            for filename in store.filenames:
                st.write(f"- {filename}")

    max_mb = MAX_UPLOAD_BYTES // (1024 * 1024)
    files = st.file_uploader(
        "Choose documents",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        help=f"PDF and UTF-8 TXT only. Maximum {max_mb} MB per file.",
    )

    api_key = secret_or_environment("OPENAI_API_KEY")
    if not api_key:
        st.warning(
            "Document processing is unavailable until `OPENAI_API_KEY` is configured "
            "in Streamlit Secrets or the local environment."
        )

    process_clicked = st.button(
        "Process and add to knowledge base",
        type="primary",
        disabled=not files or not api_key,
        width="stretch",
    )
    if process_clicked:
        results: list[dict[str, Any]] = []
        with st.spinner("Validating and processing uploaded documents..."):
            try:
                ensure_vector_store(api_key=api_key)
                embedding_provider = OpenAIEmbeddingProvider(api_key=api_key)
            except Exception:
                LOGGER.exception("Unable to initialise the curated knowledge base.")
                st.error(
                    "The active knowledge base could not be initialised. Check the "
                    "OpenAI configuration and Streamlit logs, then try again."
                )
                return

            for uploaded_file in files:
                filename = uploaded_file.name
                try:
                    validated = validate_upload(filename, uploaded_file.getvalue())
                    chunks = chunk_uploaded_document(validated)
                    chunk_count = store.add_document(
                        validated,
                        chunks,
                        embedding_provider,
                    )
                    results.append(
                        {
                            "Filename": validated.filename,
                            "Status": "Success",
                            "Chunks": chunk_count,
                            "Details": "Added to this session's active knowledge base.",
                        }
                    )
                except DuplicateUploadError as error:
                    results.append(
                        {
                            "Filename": filename,
                            "Status": "Already active",
                            "Chunks": 0,
                            "Details": str(error),
                        }
                    )
                except UploadValidationError as error:
                    results.append(
                        {
                            "Filename": filename,
                            "Status": "Rejected",
                            "Chunks": 0,
                            "Details": str(error),
                        }
                    )
                except Exception:
                    LOGGER.exception("Failed to process uploaded file %s.", filename)
                    results.append(
                        {
                            "Filename": filename,
                            "Status": "Failed",
                            "Chunks": 0,
                            "Details": "The document could not be processed or embedded.",
                        }
                    )

        st.session_state[SESSION_UPLOAD_RESULTS_KEY] = results
        st.rerun()

    _show_results(st.session_state.get(SESSION_UPLOAD_RESULTS_KEY, []))

    st.caption(
        "Demo limitation: uploaded documents are not shared with other users or "
        "retained after logout, session expiry, or app restart."
    )
