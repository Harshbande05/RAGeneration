from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredExcelLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
import os
import shutil

datapath = "././data"
embed_path = "././embeddings"

l_map = {
    "txt": TextLoader,
    "pdf": PyPDFLoader,
    "csv": CSVLoader,
    "xlsx": UnstructuredExcelLoader,
}

def train_document():
    try:

        if os.path.isdir(embed_path):
            shutil.rmtree(embed_path, ignore_errors=True)

        os.makedirs(embed_path, exist_ok=True)

        all_docs = []

        for ext, loader_cls in l_map.items():
            loader = DirectoryLoader(datapath, glob=f"**/*.{ext}", loader_cls=loader_cls)
            docs = loader.load()
            all_docs.extend(docs)

        if not all_docs:
            return {"Mssg": "No supported files found for training"}

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
        all_splits = text_splitter.split_documents(all_docs)


        embeddings = HuggingFaceEmbeddings()
        db = FAISS.from_documents(all_splits, embeddings)

        db.save_local(embed_path)

        return {"Mssg": "Training Successful"}

    except Exception as e:
        return {"Mssg": f"An error occurred during training: {str(e)}"}