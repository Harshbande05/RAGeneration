import os
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama


embed_path = "././embeddings"

combine_prompt_template = """Given the extracted content and the question, create a final answer.
If the answer is not contained in the context, say "answer not available in context." 

Context: 
{context}

Question: 
{question}

Answer:
"""
combine_prompt = PromptTemplate(
    template=combine_prompt_template, input_variables=["context", "question"]
)

LLM = ChatOllama(
    model="Llama3.2:3b",
    temperature=0,
)

chain = load_qa_chain(LLM, chain_type="stuff", prompt=combine_prompt)

embeddings = HuggingFaceEmbeddings()

docsearch = FAISS.load_local(embed_path, embeddings, allow_dangerous_deserialization=True)


def response(question):
    try:

        docs = docsearch.similarity_search(question)

        stuff_answer = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
        # document_names = [doc.metadata.get("source", "Unknown Source").split("\\")[-1].split(".")[0] for doc in docs]
        result = stuff_answer.get("output_text", "Answer not available.")
        
        # print(f"question: {question}")
        # print(f"docs: {document_names}")

        final_result = {
            'reply': result,
        }
        
        return final_result
    
    except Exception as e:
        return {"Message": str(e)}