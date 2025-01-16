# RAG
This project is a Retrieval-Augmented Generation (RAG) Backend built using FastAPI. It provides APIs for training document embeddings and performing inference, enabling advanced question-answering systems powered by vector search and pre-trained language models.

# Technologies Used:
**FastAPI:**
- A modern web framework for building APIs with Python, ensuring high performance and easy scalability.

**LangChain:**
- Used for document loading, chunking, and embeddings.
- Manages the integration between the document processing pipeline and vector database.

**FAISS:**
- Facebook AI Similarity Search, used for efficient vector storage and similarity search.

**HuggingFaceEmbeddings:**
- Generates embeddings for document chunks to store in the vector database.

**Pydantic:**
- Used for input validation and data modeling.
