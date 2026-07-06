import os
from typing import Any, Dict

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import OllamaEmbeddings

load_dotenv()

#initialize embeddings (same as ingestion.py)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

#initialize vector store
vectorstore = PineconeVectorStore(
    index_name=os.getenv("INDEX_NAME_DOC"),
    embedding=embeddings,
)

#initialize chat model
chat_model = init_chat_model(
    model="llama3.2:latest",
    model_provider="ollama",
    temperature=0.0,
)


@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve relevant documentation to help answer the user queries about Langchain."""

    # retrieve top 4 most similar documents
    retrieved_docs = vectorstore.as_retriever(search_kwargs={"k": 4}).invoke(query)

    #serialize documents for the model
    serialized = "\n\n".join(
        (f"Source: {doc.metadata.get('source','Unknown')}\n\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )

    # return both serialized content and raw documents
    return serialized, retrieved_docs

def run_llm(query: str)-> Dict[str, Any]:
    """
    Run the RAG pipeline to answer a query using retrieved documentation.

    Args:
        query: The user's question

    Returns:
        Dictionary containing:
        - answer: The generated answer
        - context: List of retrieved documents
    """
    # create the agent with retrieval tool
    system_prompt = (
        "You are a helpful AI assistant that answer questions about Langchain documentation."
        "You have access to a tool that retrieves relevant documentation. "
        "Use the tool to find the relevant information before answering the question."
        "Always cite the sources you use in your answers."
        "If you cannot find the answer in the retrieved documentation, say 'I don't know.'"
    )

    agent = create_agent(chat_model, tools=[retrieve_context], system_prompt=system_prompt)

    # build messages list
    messages = [{"role": "user", "content": query}]

    #invoke the agent
    response = agent.invoke({"messages": messages})

    # extract the answer from the last AI message
    answer = response["messages"][-1].content

    # extract context documents from ToolMessage with artifacts
    context_docs = []
    for message in response["messages"]:
        #check if this is toolMessage with artifacts
        if isinstance(message, ToolMessage) and message.artifact:
            #the artifacts should contain the list of document objects
            if isinstance(message.artifact, list):
                context_docs.extend(message.artifact)

    return{
        "answer": answer,
        "context": context_docs,
    }

if __name__ == "__main__":
    query = "What are deep agents?"
    result = run_llm(query)
    print("User Query:", query)
    print("Answer:", result["answer"])
    print("Context Documents:", len(result["context"]))
    print("-" * 50)





