import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.tools import retriever
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
import ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter

load_dotenv()

print("initializing Components...")
MODEL = "qwen3:1.7b"
embeddings =  OllamaEmbeddings(model="qwen3-embedding:8b")
llm = init_chat_model(f"ollama:{MODEL}", temperature=0)
vectorstore = PineconeVectorStore(index_name=os.environ["INDEX_NAME"], embedding = embeddings)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
prompt_template = ChatPromptTemplate.from_template(
    """ Answer the questions based only on the following context:
{context}

Question: {question}

Provide a detailed answer:"""
)

def format_docs(docs):
    """Format retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

def retrieval_chain_without_lcel(query: str):

    docs = retriever.invoke(query)
    context = format_docs(docs)
    messages = prompt_template.format_messages(context=context, question=query)
    response = llm.invoke(messages)
    return response.content

def create_retrieval_chain_with_lcel():

    retrieval_chain = (RunnablePassthrough.assign(context=itemgetter("question") | retriever | format_docs) |
    prompt_template | llm | StrOutputParser())
    return retrieval_chain

if __name__ == "__main__":
    print("retrieving...")
    query = "what is pinecone is machine learning"
    # print("\n" + "=" * 70)
    # print("Implementation 0: Raw invocation of LLM (NO RAG)")
    # print("=" * 70)
    # result_raw = llm.invoke([HumanMessage(content=query)])
    # print("\nAnswer:")
    # print(result_raw.content)

    # print("\n" + "=" * 70)
    # print("Implementation 1: RAG without LCEL")
    # print("=" * 70)
    # result_without_lcel = retrieval_chain_without_lcel(query)
    # print("\nAnswer:")
    # print(result_without_lcel)

    print("\n" + "=" * 70)
    print("Implementation 2: RAG with LCEL")
    print("=" * 70)
    chain_with_lcel = create_retrieval_chain_with_lcel()
    result_with_lcel = chain_with_lcel.invoke({"question": query})
    print("\nAnswer:")
    print(result_with_lcel)
    