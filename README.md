# RAG
This project is a Retrieval-Augmented Generation (RAG) Backend built using FastAPI. It provides APIs for training document embeddings and performing inference, enabling advanced question-answering systems powered by vector search and pre-trained language models.

# How to Use:

Follow the steps below to set up and run the project.

## **Step1**: Install Dependencies

    pip install -r requirements.txt
    
## **Step2**: Start the application using Uvicorn:

    uvicorn main:app

## **Step3**: Access the API Documentation
- Open your browser and navigate to: 
    
    http://127.0.0.1:8000/docs

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
