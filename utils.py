import os
from dotenv import load_dotenv
from pypdf import PdfReader

# LangChain imports
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableMap, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


load_dotenv()


# ------------------------------------------
# PDF Extraction
# ------------------------------------------
def extracted_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""

    for page in reader.pages:
        text += page.extract_text() or ""

    return text


# ------------------------------------------
# Text Summary Generation
# ------------------------------------------
def summarize_text(text, summary_type="Short"):
    styles = {
        "Short": "Give a concise 3–4 sentence summary.",
        "Detailed": "Give a detailed paragraph-level summary.",
        "Bulleted": "Summarize the text into clear bullet points."
    }

    llm = ChatGroq(
        model="groq/compound",
        groq_api_key=os.getenv("GROQ_API_KEY")
    )

    prompt = f"{styles[summary_type]}\n\nText:\n{text}\n\nSummary:"
    response = llm.invoke(prompt)

    return response.content


# ------------------------------------------
# Key Point Extraction (HTML Formatting)
# ------------------------------------------
def get_summary_styles(text):
    llm = ChatGroq(
        model="groq/compound",
        groq_api_key=os.getenv("GROQ_API_KEY")
    )

    prompt = (
        "Extract the 5–7 most important key points from the text. "
        #"Return them as HTML <li> elements with <b>bold</b> highlights.\n\n"
        f"Text:\n{text}"
    )

    response = llm.invoke(prompt)
    points = response.content.split("\n")

    html_list = "<ul>" + "".join(
        f"<li><b>{p.strip()}</b></li>" for p in points if p.strip()
    ) + "</ul>"

    return html_list


# ------------------------------------------
# Build FAISS Vector Store
# ------------------------------------------
def build_vector_store(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_text(text)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore


# ------------------------------------------
# Question Answering with RAG (Correct LCEL)
# ------------------------------------------
def answer_question(vector_store, question):
    import inspect, sys
    print(">>> USING UTILS FILE:", __file__)
    print(">>> LOADED answer_question():")
    print(inspect.getsource(sys.modules[__name__]))

    retriever = vector_store.as_retriever()

    llm = ChatGroq(
        model="groq/compound",
        groq_api_key=os.getenv("GROQ_API_KEY")
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Use the context to answer the question."),
        ("human", "Context:\n{context}\n\nQuestion: {question}")
    ])

    chain = (
        RunnableMap({
            "context": retriever,             # retriever(question)
            "question": RunnablePassthrough() # passes the question
        })
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain.invoke(question)
