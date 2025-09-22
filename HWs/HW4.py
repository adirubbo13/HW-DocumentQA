# SQLite Fix for ChromaDB 
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st #Now imports can be brought in traditionally
import os
import hashlib
import chromadb
from chromadb.utils import embedding_functions
from bs4 import BeautifulSoup
from openai import OpenAI
from anthropic import Anthropic
from mistralai import Mistral

# Streamlit Page Config
st.set_page_config(page_title="Homework 4", layout="wide")
st.title("HW4: Syracuse Student Organizations RAG Chatbot")

st.sidebar.title("Settings")
model_option = st.sidebar.selectbox("Choose a model:", [
    "gpt-5-nano (OpenAI)",
    "Claude Sonnet (Anthropic)",
    "Mistral Small (Mistral)"
])

max_memory = 5 # Fixed conversation buffer size 

# API Keys 
openai_key = st.secrets.get("OpenAI_API_KEY")
anthropic_key = st.secrets.get("Anthropic_API_KEY")
mistral_key = st.secrets.get("Mistral_API_KEY")

# Select LLM Client for Chatting
provider, model_id, client = None, None, None
if model_option.startswith("gpt"):
    provider, model_id, client = "openai", "gpt-4", OpenAI(api_key=openai_key)
elif model_option.startswith("Claude"):
    provider, model_id, client = "anthropic", "claude-sonnet-4-20250514", Anthropic(api_key=anthropic_key)
elif model_option.startswith("Mistral"):
    provider, model_id, client = "mistral", "mistral-small-latest", Mistral(api_key=mistral_key)

# Initialize ChromaDB 
chroma_client = chromadb.PersistentClient(path="./chromadb_store")
collection_name = "HTML_RAG_Collection"
collection = None

# Function for chunking html pages
def chunk_html_text(text, chunk_size=1000):
    """
    Chunking strategy: split by sentences and group into ~1000-character blocks (chunk_size).
    Reason: Preserves semantics better than fixed-size slicing.
    Only returns the first two chunks per document (per instructions).
    """
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

# OpenAI Embedding Functions for ChromaDB
openai_embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
    api_key=openai_key,
    model_name="text-embedding-3-small"
)

# Function to Build VectorDB
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

# Check to see if ChromaDB is built, if not use the above fubnction
if 'vector_db_built' not in st.session_state:
    build_vector_db()
    st.session_state.vector_db_built = True
else:
    collection = chroma_client.get_collection(name=collection_name, embedding_function=openai_embedding_fn)

# Chat History / Memory Check
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def get_memory(history, limit=5):
    return history[-limit:]

# Embedding Helper Function
def get_embedding(client, provider, text):
    if provider == "openai":
        return client.embeddings.create(input=text, model="text-embedding-3-small").data[0].embedding

    elif provider == "mistral":
        # Try two possible call signatures for embeddings
        try:
            return client.embeddings.create(text, model="text-embedding-3-small").data[0].embedding
        except TypeError:
            try:
                return client.embeddings.create(model="text-embedding-3-small", texts=[text]).data[0].embedding
            except Exception as e:
                # If embedding call fails, fallback to OpenAI if possible (should not happen now)
                if openai_key:
                    openai_client = OpenAI(api_key=openai_key)
                    return openai_client.embeddings.create(input=text, model="text-embedding-3-small").data[0].embedding
                else:
                    raise RuntimeError(f"Mistral embedding failed and no OpenAI key for fallback: {e}")

    elif provider == "anthropic":
        # Anthropic currently has no embeddings API, fallback to OpenAI if available
        if openai_key:
            openai_client = OpenAI(api_key=openai_key)
            return openai_client.embeddings.create(input=text, model="text-embedding-3-small").data[0].embedding
        else:
            raise RuntimeError("Anthropic client does not support embeddings. Provide OpenAI_API_KEY for embeddings fallback.")

    else:
        raise RuntimeError(f"Unknown provider {provider} for embeddings")
    
# NOTE: all these embedding helpings for each model fall back to OpenAI, I thought thise made sense because openAI did the initial embeddings

user_query = st.chat_input("Ask something based on the HTML documents...") #User input for chat

# Displaying the existing chat
for msg in st.session_state.chat_history:
    st.chat_message(msg["role"]).markdown(msg["content"])

# Handling new user queryies
if user_query:
    st.chat_message("user").markdown(user_query)
    st.session_state.chat_history.append({"role": "user", "content": user_query})

    # Embed query & retrieve context using new helper
    try:
        embedding = get_embedding(client, provider, user_query)
        results = collection.query(query_embeddings=[embedding], n_results=3)
        context_chunks = results["documents"][0]
        context = "\n\n---\n\n".join(context_chunks)

        # Build prompt
        prompt = f"""You are a helpful assistant. Use the context below to answer the user's question.
If the context is not sufficient, say "I'm not sure based on the documents."

Context:
{context}

Question: {user_query}

Answer:"""

        # --- Send prompt to selected LLM ---
        if provider == "openai":
            completion = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "system", "content": "You are a helpful assistant."},
                          {"role": "user", "content": prompt}]
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

        # Display assistant response
        st.chat_message("assistant").markdown(answer)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})

        # Apply fixed memory buffer of 5 Q&A pairs (10 messages total)
        st.session_state.chat_history = get_memory(st.session_state.chat_history, max_memory * 2)

    except Exception as e:
        st.error(f" Error communicating with model: {e}")
