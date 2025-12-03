import os
from dotenv import load_dotenv
import streamlit as st
from utils import (
    extracted_text_from_pdf,
    summarize_text,
    get_summary_styles,
    build_vector_store,
    answer_question
)

# Load environment variables
load_dotenv()

# Streamlit page config
st.set_page_config(page_title="Aurallis", layout="wide")
st.title("📄 Aurallis : Advanced PDF Intelligence")
st.write("Upload your PDF, choose summary style, view key points, and chat with your document.")

# PDF uploader
uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

# Cache the vector store for performance
@st.cache_resource
def cached_vector_store(text):
    return build_vector_store(text)

if uploaded_file:
    st.info("Extracting text from PDF...")
    raw_text = extracted_text_from_pdf(uploaded_file)

    if raw_text.strip() != "":
        st.success("Text extracted successfully.")

        # Summary options in sidebar
        st.sidebar.header("Summary Options")
        summary_type = st.sidebar.selectbox(
            "Choose summary format:",
            ["Short", "Detailed", "Bulleted"]
        )

        # Generate summary
        st.markdown("### 📘 Summary")
        with st.spinner("Generating summary..."):
            summary = summarize_text(raw_text, summary_type)
        st.write(summary)

        # Display Key Points
        st.markdown("### ⭐ Key Points (Auto-Highlighted)")
        key_points_html = get_summary_styles(raw_text)  # Make sure it returns proper <ul><li> HTML
        st.markdown(key_points_html, unsafe_allow_html=True)

        # Q&A Section
        st.markdown("### 💬 Ask Your PDF Anything")
        vector_store = cached_vector_store(raw_text)

        user_q = st.text_input("Ask a question about the PDF:")
        if user_q:
            with st.spinner("Thinking..."):
                answer = answer_question(vector_store, user_q)
            st.markdown(f"**Answer:** {answer}")

    else:
        st.warning("No extractable text found in this PDF.")
