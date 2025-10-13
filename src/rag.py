import os
# Workaround for macOS OpenMP duplicate runtime error with faiss/libomp
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import faiss
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OpenAIEmbeddings
import numpy as np
from langchain_community.chat_models import ChatOpenAI
from typing import List
from dotenv import load_dotenv
import warnings
from langchain_core._api.deprecation import LangChainDeprecationWarning
import logging

warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)

load_dotenv()  # Load environment variables from .env file

logging.basicConfig(level=logging.INFO)

class RAGConfig:
    def __init__(self, embedding_model: str = "text-embedding-3-small", faiss_index_path: str = "faiss_index.index"):
        self.embedding_model = embedding_model
        self.faiss_index_path = faiss_index_path
        self.documents_folder = "documents/"
        self.documents = self.read_documents()
    
    def read_documents(self) -> List[str]:
        """
        Read the folder with documents and add thems to documents. Uses PyPDFLoader from Lanchain Community.
        """
        documents = []
        for filename in os.listdir(self.documents_folder):
            if filename.endswith(".pdf"):
                loader = PyPDFLoader(os.path.join(self.documents_folder, filename))
                docs = loader.load()
                documents.extend(docs)
        return documents
    
    def chunk_documents(self, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """
        Chunk documents into smaller pieces for better retrieval.
        """
        chunked_docs = []
        for doc in self.documents:
            text = doc.page_content
            for i in range(0, len(text), chunk_size - overlap):
                chunked_docs.append(text[i:i + chunk_size])
        return chunked_docs

    def generate_embeddings(self, texts: List[str]):
        """
        Generate embeddings for the chunked documents
        """
        embedding_model = OpenAIEmbeddings(model=self.embedding_model)
        embeddings = embedding_model.embed_documents(texts)
        return embeddings
            
    def save_vectors(self, vectors):
        """
        Save the embeddings to a FAISS index
        """
        dimension = len(vectors[0])
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(vectors).astype('float32'))
        faiss.write_index(index, self.faiss_index_path)

    def load_index(self):
        """
        Load the FAISS index from disk
        """
        if os.path.exists(self.faiss_index_path):
            index = faiss.read_index(self.faiss_index_path)
            return index
        else:
            return None

    def create_index(self):
        """
        Create or load the FAISS index
        """
        index = self.load_index()
        if index is None:
            logging.info("FAISS index not found. Creating a new one...")
            chunked_docs = self.chunk_documents()
            embeddings = self.generate_embeddings(chunked_docs)
            self.save_vectors(embeddings)
            index = self.load_index()
        else:
            logging.info("FAISS index loaded from disk.")
        return index


        
        

class LLMConfig:
    def __init__(self, model_name: str = "gpt-4.1-nano", temperature: float = 0.7):
        self.model_name = model_name
        self.temperature = temperature
        self.llm = ChatOpenAI(model=self.model_name, temperature=self.temperature)

    def answer_question(self, question:str, context: str) -> str:
        """
        Use the LLM to answer a question given some context
        """
        prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
        # Use LangChain's standard invoke API; returns AIMessage
        response = self.llm.invoke(prompt)
        return response.content

class RAG:
    def __init__(self, rag_config: RAGConfig, llm_config: LLMConfig):
        self.rag_config = rag_config
        self.llm_config = llm_config
        self.index = self.rag_config.create_index()

    def retrieve_documents(self, query: str, top_k: int = 5) -> List[str]:
        """
        Retrieve relevant documents from the FAISS index
        """
        embedding_model = OpenAIEmbeddings(model=self.rag_config.embedding_model)
        query_embedding = embedding_model.embed_query(query)
        D, I = self.index.search(np.array([query_embedding]).astype('float32'), top_k)
        retrieved_docs = [self.rag_config.documents[i].page_content for i in I[0] if i < len(self.rag_config.documents)]
        return retrieved_docs

    def generate_response(self, query: str, documents: List[str]) -> str:
        """
        Generate a response using the LLM with the retrieved documents as context
        """
        context = "\n\n".join(documents)
        response = self.llm_config.answer_question(query, context)
        return response

    def proccess_query(self, query: str) -> str:
        """
        Process a user query end-to-end
        """
        relevant_docs = self.retrieve_documents(query)
        response = self.generate_response(query, relevant_docs)
        return response






