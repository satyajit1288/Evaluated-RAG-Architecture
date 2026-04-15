from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from evaluate import evaluate_faithfulness, evaluate_answer_relevance

def setup_rag_pipeline():
    print("Loading existing database...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vectorstore = Chroma(
        persist_directory="./chroma_db", 
        embedding_function=embeddings
    )
    
    #hybrid retriever
    
    # vector Retriever (Dense)
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

    # keyword Retriever (BM25)

    # get raw documents from Chroma -> build the keyword index
    print("Building Keyword Index (BM25)...")
    all_docs = vectorstore.get()

    # create BM25 index using text of your chunks
    bm25_retriever = BM25Retriever.from_texts(all_docs['documents'])
    bm25_retriever.k = 10 

    # Combine them into an Ensemble (Hybrid Search)
    # weights=[0.7, 0.3] --> 70% Vector, --> 30% Keyword
    ensemble_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[0.7, 0.3]
    )


    # ading re-ranker (2 - stage pipeline)

    print("Loading Re-ranker...")
    cross_encoder_model = HuggingFaceCrossEncoder(
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    compressor = CrossEncoderReranker(model=cross_encoder_model, top_n=5)

    # wrap hybrid retriever with the re-ranker
    final_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=ensemble_retriever
    )

    llm = ChatOllama(model="mistral", temperature=0.2)
    
    return final_retriever, llm

def ask_question(query, retriever, llm):
    print("\n[hybrid Search + re-ranking ]")
    retrieved_docs = retriever.invoke(query) 
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    prompt = f"""
    You are a research assistant.
    Use the following context to answer the question.
    If the answer is not present, say you don't know.

    Context:
    {context}

    Question:
    {query}
    """

    response = llm.invoke(prompt)
    return response.content, context

if __name__ == "__main__":
    retriever, llm = setup_rag_pipeline()
    
    while True:
        user_query = input("Ask a question: ")
        if user_query.lower() == 'exit':
            break

        # get (answer, context) from Mistral
        answer, context = ask_question(user_query, retriever, llm)

        print("\n--- ANSWER ---\n")
        print(answer)
        print("\n----------------\n")

        # evaluate the answer
        print("[Evaluating output for hallucinations...]")
        score = evaluate_faithfulness(user_query, context, answer)
        
        # output
        print(f"System Faithfulness Score: {score}/5")

        print("[Evaluating output for Answer Relevance...]")
        relevance_score = evaluate_answer_relevance(user_query, answer)
        print(f"Answer Relevance Score: {relevance_score}/5")