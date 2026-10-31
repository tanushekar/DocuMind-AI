
# DocuMind AI – Intelligent PDF Question Answering using RAG

DocuMind AI is an AI-powered document question-answering application that allows users to upload PDF documents and ask questions about their contents. It uses Retrieval Augmented Generation (RAG) to retrieve relevant information and generate context-aware answers using Google Gemini.

## Features

- PDF document upload and text extraction
- Text chunking for efficient document processing
- Semantic embeddings using Sentence Transformers
- Vector similarity search using FAISS
- Context-aware answer generation using Google Gemini
- Source chunk references with page numbers
- Interactive dashboard built with Streamlit

## Tech Stack

| Technology | Purpose |
|---|---|
| Python | Backend development |
| Streamlit | User interface |
| PyPDF | PDF text extraction |
| Sentence Transformers | Text embeddings |
| FAISS | Vector similarity search |
| Google Gemini API | Answer generation |
| NumPy | Numerical operations |

## RAG Architecture

1. Document Upload – User uploads a PDF.
2. Text Extraction – Extract text from PDF pages.
3. Chunking – Split extracted text into smaller chunks.
4. Embedding Generation – Convert chunks into vector representations.
5. Vector Storage – Store embeddings in a FAISS index.
6. Retrieval – Retrieve the most relevant chunks for a user query.
7. Generation – Pass retrieved context to Gemini to generate an answer.

## Project Structure

```text
DocuMind-AI/
│
├── app.py
├── requirements.txt
├── .env
├── .gitignore
└── README.md
```

Note: The `.env` file is excluded from version control and must be created locally.

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/DocuMind-AI.git
cd DocuMind-AI
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Gemini API key

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_api_key_here
```

Obtain an API key from Google AI Studio:
https://aistudio.google.com/apikey

### 5. Run the application

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal.

## How It Works

DocuMind converts PDF content into numerical embeddings and stores them in a FAISS vector index. When a user submits a question, the application generates a query embedding, retrieves the three most relevant text chunks, and supplies them as context to Google Gemini.

This helps generate answers grounded in the uploaded document rather than relying only on the model's general knowledge.

## Future Improvements

- Support multiple PDF documents
- Persistent vector storage
- Conversation memory
- Improved retrieval using reranking
- Source citations and page-level references
- Deployment to a cloud platform

## Author

Tanu Shekar
