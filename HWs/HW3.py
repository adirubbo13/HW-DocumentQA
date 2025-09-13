import streamlit as st
import requests
from bs4 import BeautifulSoup
from openai import OpenAI, AuthenticationError, APIConnectionError
from anthropic import Anthropic
from mistralai import Mistral

# Sidebar settings
st.sidebar.title("Settings")

url_mode = st.sidebar.radio("Number of URLs to Analyze:", ["1 URL", "2 URLs"])
model_option = st.sidebar.selectbox(
    "Choose LLM:",
    [
        "gpt-5-nano (OpenAI)",
        "gpt-4o (OpenAI)",
        "Claude Sonnet 4 (Anthropic)",
        "Claude Opus 4 (Anthropic)",
        "Mistral Large (Mistral)",
        "Mistral Small (Mistral)"
    ],
)
memory_option = st.sidebar.radio(
    "Conversation Memory Strategy:",
    [
        "Buffer of 6 questions",
        "Conversation Summary",
        "Buffer of 2000 tokens"
    ]
)
output_language = st.sidebar.selectbox("Response Language:", ["English", "Italian", "Japanese"])

# API keys
openai_key = st.secrets.get("OpenAI_API_KEY")
anthropic_key = st.secrets.get("Anthropic_API_KEY")
mistral_key = st.secrets.get("Mistral_API_KEY")

if model_option.startswith("gpt") and not openai_key:
    st.stop()
elif model_option.startswith("Claude") and not anthropic_key:
    st.stop()
elif model_option.startswith("Mistral") and not mistral_key:
    st.stop()

# Model setup
client = None
model_id = None
provider = None

if model_option.startswith("gpt"):
    client = OpenAI(api_key=openai_key)
    model_id = "gpt-4o" if "4o" in model_option else "gpt-5-nano"
    provider = "openai"
elif model_option.startswith("Claude"):
    client = Anthropic(api_key=anthropic_key)
    model_id = "claude-sonnet-4-20250514" if "Sonnet" in model_option else "claude-opus-4-20250514"
    provider = "anthropic"
elif model_option.startswith("Mistral"):
    client = Mistral(api_key=mistral_key)
    model_id = "mistral-large-latest" if "Large" in model_option else "mistral-small-latest"
    provider = "mistral"

st.title("HW3: Streaming Chatbot that Discusses a URL")

url_1 = st.text_input("Enter URL 1:")
url_2 = st.text_input("Enter URL 2 (optional):", disabled=(url_mode == "1 URL"))

# Fetch and clean webpage text
def fetch_url_text(url):
    try:
        resp = requests.get(url)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.content, "html.parser")
        return soup.get_text()
    except Exception as e:
        return f"[Error fetching {url}]: {e}"

# Combine webpage content
combined_text = ""
if url_1:
    combined_text += f"\n--- WEBPAGE 1 CONTENT ({url_1}) ---\n" + fetch_url_text(url_1)
if url_mode == "2 URLs" and url_2:
    combined_text += f"\n\n--- WEBPAGE 2 CONTENT ({url_2}) ---\n" + fetch_url_text(url_2)

if combined_text:
    with st.expander("Analyzed Webpage Content (For Reference)"):
        st.text_area("Fetched Text", value=combined_text, height=300)

# Updated, clearer system prompt
base_system_prompt = (
    f"You are a helpful assistant. Use {output_language} as the language to answer the user's questions.\n\n"
    "The user will ask questions based on one or two webpages.\n"
    "Below, you will find the raw extracted text content from those webpages. "
    "This text may be messy, unstructured, or incomplete.\n"
    "Interpret the content as if it came from online articles or blog posts, and explain it clearly — as if you're speaking to a 15-year-old.\n"
    "End each answer with: 'Do you want more info?'"
)

if combined_text:
    system_prompt_content = base_system_prompt + f"\n\n--- BEGIN WEBPAGE CONTENT ---\n{combined_text}\n--- END WEBPAGE CONTENT ---"
else:
    system_prompt_content = base_system_prompt

system_prompt = {"role": "system", "content": system_prompt_content}

# Memory strategies
def get_memory_messages(memory_type, messages, max_tokens=2000):
    if memory_type == "Buffer of 6 questions":
        intro = [messages[0]] if messages and messages[0]["role"] == "system" else []
        buffer = []
        count = 0
        for m in reversed(messages[1:]):
            buffer.insert(0, m)
            if m["role"] == "user":
                count += 1
                if count >= 6:
                    break
        return intro + buffer

    elif memory_type == "Buffer of 2000 tokens":
        from tiktoken import get_encoding
        enc = get_encoding("cl100k_base")
        token_count = lambda msg: len(enc.encode(msg["content"]))
        history = []
        total = 0
        for msg in reversed(messages):
            total += token_count(msg)
            if total > max_tokens:
                break
            history.insert(0, msg)
        return history

    elif memory_type == "Conversation Summary":
        summary = "The user is asking questions about the contents of the provided webpages."
        summary_msg = {"role": "assistant", "content": f"Conversation Summary: {summary}"}
        return [system_prompt, summary_msg] + messages[-4:]

    return messages

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [system_prompt]
    st.session_state.chat_history.append({"role": "assistant", "content": "Hi there! What would you like to talk about?"})

# Display chat history
for msg in st.session_state.chat_history[1:]:  # Skip system prompt
    st.chat_message(msg["role"]).write(msg["content"])

# Handle user input
if user_input := st.chat_input("Say something..."):
    st.chat_message("user").markdown(user_input)
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    messages_to_send = get_memory_messages(memory_option, st.session_state.chat_history)

    try:
        if provider == "openai":
            stream = client.chat.completions.create(
                model=model_id,
                messages=messages_to_send,
                stream=True
            )
            with st.chat_message("assistant"):
                response_container = st.empty()
                full_response = ""
                for chunk in stream:
                    content = chunk.choices[0].delta.content or ""
                    full_response += content
                    response_container.markdown(full_response + "▌")
                response_container.markdown(full_response)
            st.session_state.chat_history.append({"role": "assistant", "content": full_response})

        elif provider == "anthropic":
            history_messages = [
                {"role": "user", "content": m["content"]} if m["role"] == "user"
                else {"role": "assistant", "content": m["content"]}
                for m in st.session_state.chat_history[1:]
            ]
            response = client.messages.create(
                model=model_id,
                system=system_prompt_content,
                max_tokens=1000,
                messages=history_messages
            )
            reply = response.content[0].text if isinstance(response.content, list) else response.content
            st.chat_message("assistant").write(reply)
            st.session_state.chat_history.append({"role": "assistant", "content": reply})

        elif provider == "mistral":
            response = client.chat.complete(
                model=model_id,
                messages=messages_to_send,
                stream=False
            )
            reply = response.choices[0].message.content
            st.chat_message("assistant").write(reply)
            st.session_state.chat_history.append({"role": "assistant", "content": reply})

    except Exception as e:
        st.error(f"Error communicating with model: {e}")
