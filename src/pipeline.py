from rag import RAG, RAGConfig, LLMConfig


class Pipeline:
    def __init__(self, rag: RAG):
        self.rag = rag

    def process_query(self, query: str) -> str:
        # Step 1: Retrieve relevant documents using RAG
        relevant_docs = self.rag.retrieve_documents(query)

        # Step 2: Generate a response using the LLM with the retrieved documents as context
        response = self.rag.generate_response(query, relevant_docs)

        return response

if __name__ == "__main__":
    rag_config = RAGConfig()
    llm_config = LLMConfig()
    rag = RAG(rag_config, llm_config)
    pipeline = Pipeline(rag)

    user_query = "Detalla los estudios de Cristian, y las empresas en las que ha trabajado"
    answer = pipeline.process_query(user_query)
    print(f"Answer: {answer}")