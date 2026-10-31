
import os
import streamlit as st
import numpy as np
import faiss

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from google import genai

# Load API key
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

# Page configuration
st.set_page_config(page_title="DocuMind RAG", page_icon="📚")
st.title("DocuMind - PDF Question Answering")
st.write("Upload a PDF and ask questions about its content.")

# Load embedding model
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_embedding_model()

# Split text into chunks
def split_text(text, chunk_size=800, overlap=150):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks

# Extract text from PDF
def extract_text(pdf):
    reader = PdfReader(pdf)
    text = ""

    for page in reader.pages:
        text += page.extract_text() or ""

    return text

# Generate embeddings and create FAISS index
def create_vector_database(chunks):
    embeddings = model.encode(
        chunks,
        normalize_embeddings=True
    )

    embeddings = np.array(embeddings, dtype="float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return index

# Retrieve relevant chunks
def retrieve_chunks(question, index, chunks):
    query_embedding = model.encode(
        [question],
        normalize_embeddings=True
    )

    query_embedding = np.array(
        query_embedding, dtype="float32"
    )

    scores, indices = index.search(query_embedding, k=min(3, len(chunks)))

    results = [chunks[i] for i in indices[0] if i >= 0]

    return results

# Generate answer using Gemini
def generate_answer(question, context):
    client = genai.Client(api_key=API_KEY)

    prompt = f"""
    Answer the question using ONLY the context below.

    If the answer is not present in the context,
    say that the document does not contain the answer.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt
    )

    return response.text

# PDF upload
uploaded_file = st.file_uploader(
    "Upload your PDF document",
    type=["pdf"]
)

if uploaded_file:
    if st.button("Process PDF"):
        with st.spinner("Reading and processing PDF..."):
            text = extract_text(uploaded_file)

            if not text.strip():
                st.error("No readable text found in this PDF.")
                st.stop()

            chunks = split_text(text)
            index = create_vector_database(chunks)

            st.session_state.chunks = chunks
            st.session_state.index = index

            st.success(f"PDF processed! Created {len(chunks)} chunks.")

# Ask questions
if "index" in st.session_state:
    question = st.text_input("Ask a question about your PDF")

    if st.button("Get Answer"):
        if not question.strip():
            st.warning("Please enter a question.")
        elif not API_KEY:
            st.error("Gemini API key missing. Check your .env file.")
        else:
            with st.spinner("Finding answer..."):
                relevant_chunks = retrieve_chunks(
                    question,
                    st.session_state.index,
                    st.session_state.chunks
                )

                context = "\n\n".join(relevant_chunks)

                try:
                    answer = generate_answer(question, context)

                    st.subheader("Answer")
                    st.write(answer)

                    with st.expander("View retrieved source chunks"):
                        for i, chunk in enumerate(relevant_chunks):
                            st.write(f"Source {i + 1}")
                            st.write(chunk)

                except Exception as e:
                    st.error(f"Gemini API error: {e}")
