from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from huggingface_hub import InferenceClient
import os

# Load environment variables
load_dotenv()
hf_token = os.getenv("huf")

# Connect to your document database
persistent_directory = "db/chroma_db"

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
)

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)

# Set up HuggingFace client
client = InferenceClient(
    provider="novita",
    api_key=hf_token,
)

MODEL = "Qwen/Qwen2.5-72B-Instruct"

# Store conversation history as list of dicts (role/content format)
chat_history = []


def invoke_model(messages: list) -> str:
    """Call HuggingFace model and return response text."""
    result = client.chat_completion(
        model=MODEL,
        messages=messages,
        max_tokens=512,
    )
    return result.choices[0].message.content.strip()


def ask_question(user_question):
    print(f"\n--- You asked: {user_question} ---")

    # Step 1: Rewrite question using conversation history
    if chat_history:
        messages = [
            {"role": "system", "content": "Given the chat history, rewrite the new question to be standalone and searchable. Just return the rewritten question."},
        ] + chat_history + [
            {"role": "user", "content": f"New question: {user_question}"}
        ]

        search_question = invoke_model(messages)
        print(f"Searching for: {search_question}")
    else:
        search_question = user_question

    # Step 2: Find relevant documents
    retriever = db.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(search_question)

    print(f"Found {len(docs)} relevant documents:")
    for i, doc in enumerate(docs, 1):
        lines = doc.page_content.split('\n')[:2]
        preview = '\n'.join(lines)
        print(f"  Doc {i}: {preview}...")

    # Step 3: Build final prompt
    combined_input = f"""Based on the following documents, please answer this question: {user_question}

Documents:
{chr(10).join([f"- {doc.page_content}" for doc in docs])}

Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
"""

    # Step 4: Get the answer (include chat history for context)
    messages = [
        {"role": "system", "content": "You are a helpful assistant that answers questions based on provided documents and conversation history."},
    ] + chat_history + [
        {"role": "user", "content": combined_input}
    ]

    answer = invoke_model(messages)

    # Step 5: Save conversation history
    chat_history.append({"role": "user", "content": user_question})
    chat_history.append({"role": "assistant", "content": answer})

    print(f"Answer: {answer}")
    return answer


def start_chat():
    print("Ask me questions! Type 'quit' to exit.")

    while True:
        question = input("\nYour question: ")

        if question.lower() == 'quit':
            print("Goodbye!")
            break

        ask_question(question)


if __name__ == "__main__":
    start_chat()