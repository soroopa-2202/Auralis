# 📄 Aurallis – Generative AI Powered PDF Assistant

Aurallis is an advanced AI application that analyzes PDF documents using
Retrieval-Augmented Generation (RAG). It can extract text, generate summaries,
highlight key insights, and support conversational Q&A with any uploaded file.

---

## ✨ Features
| Feature | Description |
|--------|-------------|
| PDF Text Extraction | Reads and processes content automatically |
| Multi-Style Summaries | Short, detailed, or bullet-format summaries |
| Key Point Highlighting | Extracts 5–7 main ideas with clean HTML formatting |
| Chat with Your PDF | Ask questions and get context-aware responses |
| Built with RAG | Uses FAISS + embeddings for accurate answers |

---

## 🧠 Tech Stack
- Python, Streamlit
- Groq LLM + LangChain
- HuggingFace embeddings
- FAISS Vector Search

---

## 🚀 How to Run Locally

```bash
git clone https://github.com/<your-username>/Aurallis-GenAI-PDF-Assistant.git
cd Aurallis-GenAI-PDF-Assistant
pip install -r requirements.txt
cp .env.example .env   # Add your Groq API key
streamlit run app.py
