"""Streamlit interface for the Support Ticket RAG Assistant."""

from __future__ import annotations

import streamlit as st

from config import get_settings
from errors import ServiceError
from rag import generate_resolution

st.set_page_config(page_title="Support Ticket RAG Assistant", page_icon="🛠️")

st.title("🛠️ Support Ticket RAG Assistant")
st.caption("Retrieve historical support tickets and generate an evidence-grounded resolution.")

with st.sidebar:
    st.header("Retrieval settings")
    top_k = st.slider("Historical tickets to retrieve", min_value=1, max_value=5, value=3)
    model = st.text_input(
        "Local Ollama model",
        value=get_settings().ollama_model,
        help="The model must already be available in Ollama on this computer.",
    )

issue = st.text_area(
    "Describe a technical support issue",
    placeholder="Example: Customer cannot log in after resetting their password.",
    height=120,
)

if st.button("Find resolution", type="primary", disabled=not issue.strip()):
    with st.spinner("Searching historical tickets and generating a grounded response..."):
        try:
            answer, tickets = generate_resolution(issue.strip(), top_k=top_k, model=model.strip())
        except ServiceError as error:
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
