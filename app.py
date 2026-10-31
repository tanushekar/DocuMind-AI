
import os
import math
from io import BytesIO

import streamlit as st
import numpy as np
import faiss

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from google import genai


# ---------------- CONFIGURATION ----------------

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-3.5-flash-lite"

st.set_page_config(
    page_title="DocuMind AI",
    page_icon="D",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ---------------- CUSTOM DESIGN ----------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: #0b0d17;
    color: #f1f2fa;
}

[data-testid="stSidebar"] {
    background: #101321;
    border-right: 1px solid #252941;
}

[data-testid="stSidebar"] > div {
    padding-top: 2rem;
}

.hero {
    padding: 30px;
    border-radius: 20px;
    background: linear-gradient(120deg, #191b38, #28204c, #171a32);
    border: 1px solid #38345f;
    margin-bottom: 25px;
}

.hero h1 {
    font-size: 38px;
    font-weight: 800;
    color: #ffffff;
    margin-bottom: 8px;
}

.hero p {
    color: #b7b9d4;
    font-size: 15px;
}

.tag {
    display: inline-block;
    background: #30285c;
    color: #c8baff;
    border: 1px solid #57478e;
    border-radius: 20px;
    padding: 5px 12px;
    font-size: 12px;
    margin-bottom: 14px;
}

.section-title {
    font-size: 19px;
    font-weight: 700;
    margin: 18px 0 12px 0;
    color: #f4f3ff;
}

.metric-card {
    background: #15192a;
    border: 1px solid #2b3049;
    border-radius: 14px;
    padding: 18px;
    min-height: 100px;
}

.metric-label {
    color: #9298b6;
    font-size: 12px;
    margin-bottom: 8px;
}

.metric-value {
    color: #ffffff;
    font-size: 25px;
    font-weight: 700;
}

.upload-card {
    background: #15192a;
    border: 1px dashed #6352a5;
    border-radius: 16px;
    padding: 22px;
    margin: 12px 0;
}

.stButton > button {
    background: linear-gradient(100deg, #7655e8, #9b65ee);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 700;
    min-height: 43px;
    transition: 0.2s;
}

.stButton > button:hover {
    background: linear-gradient(100deg, #8869f5, #ad78f5);
    color: white;
    border: none;
}

.stTextInput input {
    background: #171b2c;
    color: white;
    border: 1px solid #383d5b;
    border-radius: 12px;
}

[data-testid="stChatMessage"] {
    background: #15192a;
    border: 1px solid #292e48;
    border-radius: 14px;
    padding: 15px;
    margin-bottom: 12px;
}

.stExpander {
    background: #141829;
    border: 1px solid #303550;
    border-radius: 12px;
}

hr {
    border-color: #292e46;
}

footer {
    visibility: hidden;
}

#MainMenu {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


# ---------------- SESSION STATE ----------------

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "index" not in st.session_state:
    st.session_state.index = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "document_name" not in st.session_state:
    st.session_state.document_name = None

if "processed_file" not in st.session_state:
    st.session_state.processed_file = None


# ---------------- LOAD EMBEDDING MODEL ----------------

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


# ---------------- PDF EXTRACTION ----------------

def extract_text(pdf_bytes):
    reader = PdfReader(BytesIO(pdf_bytes))
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append((page_number, text))

    return pages


# ---------------- TEXT CHUNKING ----------------

def split_text(pages, chunk_size=800, overlap=150):
    chunks = []

    for page_number, text in pages:
        start = 0

        while start < len(text):
            chunk = text[start:start + chunk_size]

            if chunk.strip():
                chunks.append({
                    "text": chunk,
                    "page": page_number
                })

            start += chunk_size - overlap

    return chunks


# ---------------- VECTOR DATABASE ----------------

def create_vector_database(chunks, model):
    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    embeddings = np.array(embeddings, dtype="float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return index


# ---------------- RETRIEVAL ----------------

def retrieve_chunks(question, index, chunks, model):
    query_embedding = model.encode(
        [question],
        normalize_embeddings=True
    )

    query_embedding = np.array(
        query_embedding,
        dtype="float32"
    )

    scores, indices = index.search(
        query_embedding,
        k=min(3, len(chunks))
    )

    results = []

    for i, score in zip(indices[0], scores[0]):
        if i >= 0:
            result = chunks[i].copy()
            result["score"] = float(score)
            results.append(result)

    return results


# ---------------- GEMINI GENERATION ----------------

def generate_answer(question, context):
    if not API_KEY:
        raise ValueError("Gemini API key is missing. Check your .env file.")

    client = genai.Client(api_key=API_KEY)

    prompt = f"""
You are DocuMind, a helpful document assistant.

Answer the user's question using ONLY the supplied document context.

Instructions:
- Explain concepts in simple and clear language.
- Use headings and bullet points where useful.
- Do not invent information.
- If the answer is unavailable in the context, say so.
- Keep the answer relevant to the question.

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
"""

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )

    return response.text


# ---------------- SIDEBAR ----------------

with st.sidebar:

    st.markdown("""
    <h2 style="color:#b9a5ff; margin-bottom:0;">
        DOCUMIND
    </h2>
    <p style="color:#858ba9; font-size:12px; letter-spacing:2px;">
        AI DOCUMENT INTELLIGENCE
    </p>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown("### Your workspace")
    st.caption("Upload a document to start asking questions.")

    uploaded_file = st.file_uploader(
        "Choose a PDF document",
        type=["pdf"]
    )

    if uploaded_file:
        file_id = (
            uploaded_file.name,
            uploaded_file.size
        )

        if file_id != st.session_state.processed_file:
            st.session_state.chunks = []
            st.session_state.index = None
            st.session_state.messages = []
            st.session_state.document_name = None
            st.session_state.processed_file = None

        if st.button("Process document", use_container_width=True):

            with st.spinner("Preparing your document..."):

                try:
                    model = load_embedding_model()

                    pdf_bytes = uploaded_file.getvalue()
                    pages = extract_text(pdf_bytes)

                    if not pages:
                        st.error("No readable text found in this PDF.")
                    else:
                        chunks = split_text(pages)

                        if not chunks:
                            st.error("Could not create text chunks.")
                        else:
                            index = create_vector_database(
                                chunks, model
                            )

                            st.session_state.chunks = chunks
                            st.session_state.index = index
                            st.session_state.document_name = uploaded_file.name
                            st.session_state.processed_file = file_id
                            st.session_state.messages = []

                            st.success("Document processed successfully!")

                except Exception as e:
                    st.error(f"Processing failed: {e}")

    st.divider()

    if st.session_state.index is not None:

        st.markdown("### Current document")

        st.markdown(
            f"**{st.session_state.document_name}**"
        )

        st.caption(
            f"{len(st.session_state.chunks)} text chunks indexed"
        )

        st.success("Ready for questions")

        if st.button("Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    else:
        st.info("No document processed yet.")

    st.divider()

    st.markdown("### How it works")

    st.markdown("""
    <p style="color:#aeb3cd; font-size:13px; line-height:2;">
    01 &nbsp; Document extraction<br>
    02 &nbsp; Text chunking<br>
    03 &nbsp; Embedding generation<br>
    04 &nbsp; FAISS retrieval<br>
    05 &nbsp; Gemini response
    </p>
    """, unsafe_allow_html=True)

    st.caption("Powered by Python · FAISS · Gemini")


# ---------------- MAIN DASHBOARD ----------------

st.markdown("""
<div class="hero">
    <div class="tag">RETRIEVAL AUGMENTED GENERATION</div>
    <h1>Your documents.<br>Understood by AI.</h1>
    <p>
        Upload your PDFs, ask questions, and get intelligent
        answers grounded in your document content.
    </p>
</div>
""", unsafe_allow_html=True)


# ---------------- METRICS ----------------

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">DOCUMENT STATUS</div>
        <div class="metric-value">
            {"Ready" if st.session_state.index is not None else "Waiting"}
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">TEXT CHUNKS</div>
        <div class="metric-value">
            {len(st.session_state.chunks)}
        </div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">QUESTIONS ASKED</div>
        <div class="metric-value">
            {len([m for m in st.session_state.messages if m["role"] == "user"])}
        </div>
    </div>
    """, unsafe_allow_html=True)


st.markdown('<div class="section-title">Ask your document</div>',
            unsafe_allow_html=True)


# ---------------- CHAT HISTORY ----------------

if st.session_state.index is None:

    st.info(
        "Upload a PDF from the sidebar and click Process document "
        "to begin."
    )

    st.markdown("""
    <div class="upload-card">
        <h3 style="color:#c9baff;">Start exploring your documents</h3>
        <p style="color:#9da3c1;">
        Your AI assistant will retrieve relevant information
        and generate answers based on your uploaded PDF.
        </p>
    </div>
    """, unsafe_allow_html=True)

else:

    if not st.session_state.messages:
        st.markdown("""
        <div class="upload-card">
            <h3 style="color:#c9baff;">What would you like to know?</h3>
            <p style="color:#9da3c1;">
            Ask for summaries, explanations, definitions,
            comparisons, or important points from your document.
            </p>
        </div>
        """, unsafe_allow_html=True)

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])

            if message["role"] == "assistant" and message.get("sources"):

                with st.expander("View retrieved sources"):

                    for i, source in enumerate(message["sources"], 1):

                        st.markdown(
                            f"**Source {i} · Page {source['page']}**"
                        )

                        st.caption(
                            f"Similarity score: {source['score']:.3f}"
                        )

                        st.write(source["text"])

                        st.divider()


    # ---------------- NEW QUESTION ----------------

    question = st.chat_input(
        "Ask anything about your document..."
    )

    if question:

        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):

            try:
                model = load_embedding_model()

                with st.spinner("Searching your document..."):

                    sources = retrieve_chunks(
                        question,
                        st.session_state.index,
                        st.session_state.chunks,
                        model
                    )

                    context = "\n\n".join(
                        source["text"] for source in sources
                    )

                with st.spinner("Generating your answer..."):

                    answer = generate_answer(
                        question,
                        context
                    )

                st.markdown(answer)

                with st.expander("View retrieved sources"):

                    for i, source in enumerate(sources, 1):

                        st.markdown(
                            f"**Source {i} · Page {source['page']}**"
                        )

                        st.caption(
                            f"Similarity score: {source['score']:.3f}"
                        )

                        st.write(source["text"])

                        st.divider()

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })

            except Exception as e:
                st.error(f"Could not generate answer: {e}")

        st.rerun()


# ---------------- FOOTER ----------------

st.divider()

st.markdown("""
<div style="text-align:center; padding:12px; color:#737a99;
font-size:12px;">
    DOCUMIND AI &nbsp; · &nbsp; Intelligent Document Question Answering
    <br><br>
    Built with Streamlit, Sentence Transformers, FAISS and Gemini
</div>
""", unsafe_allow_html=True)
