# SQLite Fix for ChromaDB
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
import hashlib
import chromadb
from chromadb.utils import embedding_functions
from bs4 import BeautifulSoup
from openai import OpenAI
from anthropic import Anthropic
from mistralai import Mistral

# Streamlit Page Config
st.set_page_config(page_title="Homework 5", layout="wide")
st.title("HW5: Syracuse Student Organizations RAG Chatbot - Code Enhanced by Functions")

st.sidebar.title("Settings")
model_option = st.sidebar.selectbox("Choose a model:", [
    "gpt-5-nano (OpenAI)",
    "Claude Sonnet (Anthropic)",
    "Mistral Small (Mistral)"
])

max_memory = 5  # Fixed conversation buffer size

# API Keys
openai_key = st.secrets.get("OpenAI_API_KEY")
anthropic_key = st.secrets.get("Anthropic_API_KEY")
mistral_key = st.secrets.get("Mistral_API_KEY")

# Model selection
provider, model_id, client = None, None, None
if model_option.startswith("gpt"):
    provider, model_id, client = "openai", "gpt-4", OpenAI(api_key=openai_key)
elif model_option.startswith("Claude"):
    provider, model_id, client = "anthropic", "claude-sonnet-4-20250514", Anthropic(api_key=anthropic_key)
elif model_option.startswith("Mistral"):
    provider, model_id, client = "mistral", "mistral-small-latest", Mistral(api_key=mistral_key)

# Embedding Helped Function
def get_embedding(client, provider, text):
    if provider == "openai":
        return client.embeddings.create(input=text, model="text-embedding-3-small").data[0].embedding
    elif provider == "mistral":
        try:
            return client.embeddings.create(text, model="text-embedding-3-small").data[0].embedding
        except TypeError:
            try:
                return client.embeddings.create(model="text-embedding-3-small", texts=[text]).data[0].embedding
            except Exception as e:
                if openai_key:
                    openai_client = OpenAI(api_key=openai_key)
                    return openai_client.embeddings.create(input=text, model="text-embedding-3-small").data[0].embedding
                else:
                    raise RuntimeError(f"Mistral embedding failed and no fallback: {e}")
    elif provider == "anthropic":
        if openai_key:
            openai_client = OpenAI(api_key=openai_key)
            return openai_client.embeddings.create(input=text, model="text-embedding-3-small").data[0].embedding
        else:
            raise RuntimeError("Anthropic does not support embeddings. Provide OpenAI_API_KEY for fallback.")
    else:
        raise RuntimeError(f"Unknown provider {provider}")

# New Function for HW5: Returns relevant info from ChromaDB Query as a String
def get_relevant_club_info(query):
    """
    Takes a user query, returns relevant text chunks from ChromaDB as a string.
    """
    embedding = get_embedding(client, provider, query)
    results = collection.query(query_embeddings=[embedding], n_results=3)
    context_chunks = results["documents"][0]
    return "\n\n---\n\n".join(context_chunks)

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./chromadb_store")
collection_name = "HTML_RAG_Collection"
collection = None

# Chunking HTML
def chunk_html_text(text, chunk_size=1000):
    import re
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    current = ""
    for sentence in sentences:
        if len(current) + len(sentence) < chunk_size:
            current += sentence + " "
        else:
            chunks.append(current.strip())
            current = sentence + " "
    if current:
        chunks.append(current.strip())
    return chunks[:2]  # Keep only first two chunks

# OpenAI Embedding Function
openai_embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
    api_key=openai_key,
    model_name="text-embedding-3-small"
)

# Build Vector DB
def build_vector_db():
    global collection
    if collection_name in [c.name for c in chroma_client.list_collections()]:
        st.info("Vector DB already exists. Loading existing DB...")
        collection = chroma_client.get_collection(name=collection_name, embedding_function=openai_embedding_fn)
        return

    st.info("Creating vector DB from HTML files...")
    collection = chroma_client.create_collection(name=collection_name, embedding_function=openai_embedding_fn)

    html_dir = "./hw4files"
    for filename in os.listdir(html_dir):
        if not filename.endswith(".html"):
            continue
        filepath = os.path.join(html_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f.read(), "html.parser")
            raw_text = soup.get_text(separator=" ", strip=True)
            chunks = chunk_html_text(raw_text)

            for i, chunk in enumerate(chunks):
                doc_id = hashlib.md5((filename + str(i)).encode()).hexdigest()
                collection.add(
                    documents=[chunk],
                    metadatas=[{"source": filename, "chunk": i}],
                    ids=[doc_id]
                )
    st.success("Vector DB created!")

# Check to ensure DB is built
if 'vector_db_built' not in st.session_state:
    build_vector_db()
    st.session_state.vector_db_built = True
else:
    collection = chroma_client.get_collection(name=collection_name, embedding_function=openai_embedding_fn)

# Chat Memory
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def get_memory(history, limit=5):
    return history[-limit:]

# Chat Input
user_query = st.chat_input("Ask something based on the HTML documents...")

# Display chat history
for msg in st.session_state.chat_history:
    st.chat_message(msg["role"]).markdown(msg["content"])

# Handle new user input
if user_query:
    st.chat_message("user").markdown(user_query)
    st.session_state.chat_history.append({"role": "user", "content": user_query})

    try:
        # 🔧 Use helper function to get context
        relevant_club_info = get_relevant_club_info(user_query)

        # 🔧 Refactored prompt
        prompt = f"""You are a helpful assistant. Use the following information extracted from Syracuse student organization webpages to answer the user's question.

If the information is not sufficient to answer, respond with: "I'm not sure based on the documents."

Relevant Club Info:
{relevant_club_info}

User Question:
{user_query}
"""

        # LLM Calls
        if provider == "openai":
            completion = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            answer = completion.choices[0].message.content

        elif provider == "anthropic":
            completion = client.messages.create(
                model=model_id,
                max_tokens=800,
                messages=[{"role": "user", "content": prompt}]
            )
            answer = completion.content[0].text

        elif provider == "mistral":
            completion = client.chat.complete(
                model=model_id,
                messages=[{"role": "user", "content": prompt}]
            )
            answer = completion.choices[0].message.content

        # Show assistant message
        st.chat_message("assistant").markdown(answer)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})

        # Maintain memory window
        st.session_state.chat_history = get_memory(st.session_state.chat_history, max_memory * 2)

    except Exception as e:
        st.error(f"Error communicating with model: {e}")
