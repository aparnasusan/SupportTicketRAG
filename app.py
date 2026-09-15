"""Streamlit interface for the Support Ticket RAG Assistant."""

from __future__ import annotations

import streamlit as st

from support_ticket_rag.config import get_settings
from support_ticket_rag.errors import ServiceError
from support_ticket_rag.rag import generate_resolution
from support_ticket_rag.validation import (
    MAX_ISSUE_LENGTH,
    MAX_TOP_K,
    MIN_TOP_K,
    InputValidationError,
)

st.set_page_config(page_title="Support Ticket RAG Assistant", page_icon="🛠️")

st.title("🛠️ Support Ticket RAG Assistant")
st.caption("Retrieve historical support tickets and generate an evidence-grounded resolution.")

with st.sidebar:
    st.header("Retrieval settings")
    top_k = st.slider(
        "Historical tickets to retrieve", min_value=MIN_TOP_K, max_value=MAX_TOP_K, value=3
    )
    model = st.text_input(
        "Local Ollama model",
        value=get_settings().ollama_model,
        help="The model must already be available in Ollama on this computer.",
    )

issue = st.text_area(
    "Describe a technical support issue",
    placeholder="Example: Customer cannot log in after resetting their password.",
    height=120,
    max_chars=MAX_ISSUE_LENGTH,
)

if st.button("Find resolution", type="primary", disabled=not issue.strip()):
    with st.spinner("Searching historical tickets and generating a grounded response..."):
        try:
            answer, tickets = generate_resolution(issue.strip(), top_k=top_k, model=model.strip())
        except (ServiceError, InputValidationError) as error:
            st.error(str(error))
        else:
            st.subheader("Evidence-grounded suggested resolution")
            st.markdown(answer)

            with st.expander("Retrieved historical tickets", expanded=True):
                for rank, ticket in enumerate(tickets, start=1):
                    st.markdown(
                        f"**{rank}. {ticket.ticket_id} — {ticket.product}**  "
                        f"Similarity: `{ticket.similarity_score:.3f}`"
                    )
                    st.write(f"**Issue:** {ticket.issue}")
                    st.write(f"**Resolution:** {ticket.resolution}")
                    if rank < len(tickets):
                        st.divider()
else:
    st.info("Enter a support issue, then select **Find resolution**.")
