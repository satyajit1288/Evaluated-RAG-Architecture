from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate

def evaluate_faithfulness(query, context, generated_answer):
    evaluator_llm = ChatOllama(model="mistral", temperature=0.0)

    eval_prompt = PromptTemplate(
        input_variables=["query", "context", "answer"],
        template="""
        You are an impartial judge evaluating a RAG system.
        Evaluate the FAITHFULNESS of the generated answer.
        
        Faithfulness means: Does the answer only contain facts that exist within the Context? 
        If the answer contains information not found in the Context, it is a hallucination.

        Query: {query}
        Context: {context}
        Generated Answer: {answer}

        Score the faithfulness from 1 to 5.

        Provide only the integer score.
        """
    )

    formatted_prompt = eval_prompt.format(
        query=query, context=context, answer=generated_answer
    )

    score_response = evaluator_llm.invoke(formatted_prompt)
    return score_response.content.strip()

def evaluate_answer_relevance(query, answer):
    evaluator_llm = ChatOllama(model="mistral", temperature=0.0)

    prompt = f"""
    Evaluate how well the answer addresses the query.

    Query: {query}
    Answer: {answer}

    Score from 1 to 5.
    Only return number.
    """

    response = evaluator_llm.invoke(prompt)
    return response.content.strip()