import streamlit as st
import requests
from bs4 import BeautifulSoup

from openai import OpenAI, AuthenticationError, APIConnectionError
from anthropic import Anthropic
from mistralai import Mistral  

# Function to fetch and extract text from a URL
def read_url_content(url):
    try:
        resp = requests.get(url)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.content, "html.parser")
        return soup.get_text()
    except Exception as e:
        st.error(f"Error reading URL: {e}")
        return None

st.title("HW2: URL Summarizer Across Multiple LLMs")
st.write("Enter a URL and ask a question—the selected AI model will respond!")

# Sidebar options
summary_format = st.sidebar.radio(
    "Choose Summary Format:",
    ["Summarize in 100 words", "Summarize in 2 paragraphs", "Summarize in 5 bullet points"]
)

model_option = st.sidebar.selectbox(
    "Choose Model:",
    [
        "gpt-5-nano (OpenAI)",
        "gpt-4o (OpenAI)",
        "Claude Sonnet 4 (Anthropic)",
        "Claude Opus 4 (Anthropic)",
        "Mistral Large (Mistral)",
        "Mistral Small (Mistral)"
    ],
)

# Load API keys
openai_key = st.secrets.get("OpenAI_API_KEY")
anthropic_key = st.secrets.get("Anthropic_API_KEY")
mistral_key = st.secrets.get("Mistral_API_KEY")

# Notify if any key is missing
if model_option.startswith("gpt") and not openai_key:
    st.info("Add your OpenAI API key in secrets to continue.", icon="🗝️")
elif model_option.startswith("Claude") and not anthropic_key:
    st.info("Add your Anthropic API key in secrets to continue.", icon="🗝️")
elif model_option.startswith("Mistral") and not mistral_key:
    st.info("Add your Mistral API key in secrets to continue.", icon="🗝️")

# Main inputs
url_input = st.text_input("Enter a URL to summarize:")
question = st.text_area("Ask a question about the page:", disabled=not url_input)
output_language = st.selectbox("Choose Output Language:", ["English", "Italian", "Japanese"], disabled=not url_input)
submit_clicked = st.button("Submit", disabled=not (url_input and question))

#Clicking button to interact with model APIS
if submit_clicked:
    document = read_url_content(url_input)
    if not document:
        st.error("Failed to fetch URL content.")
    else:
        # Model instructions instructions
        format_instr = {
            "Summarize in 100 words": "Please summarize the document in approximately 100 words.",
            "Summarize in 2 paragraphs": "Please summarize the document in 2 well‑connected paragraphs.",
            "Summarize in 5 bullet points": "Please summarize the document using 5 bullet points."
        }[summary_format]
        lang_instr = f"Please write the response in {output_language}."
        full_prompt = (
            f"Here's the content from a webpage:\n{document}\n\n---\n\n"
            f"{question}\n\n{format_instr}\n{lang_instr}"
        )
        try: #simple way to have the models switch I think
            if model_option.startswith("gpt"):
                client = OpenAI(api_key=openai_key)
                m = "gpt-4o" if "4o" in model_option else "gpt-5-nano"
                stream = client.chat.completions.create(model=m, messages=[{"role": "user", "content": full_prompt}], stream=True)
                st.write_stream(stream)

            elif model_option.startswith("Claude"):
                client = Anthropic(api_key=anthropic_key)
                anth_model = "claude-sonnet-4-20250514" if "Sonnet" in model_option else "claude-opus-4-20250514"
                resp = client.messages.create(model=anth_model, messages=[{"role": "user", "content": full_prompt}], max_tokens=1000)
                # Access content as attribute
                st.write(resp.content)

            elif model_option.startswith("Mistral"):
                client = Mistral(api_key=mistral_key)
                m_model = "mistral-large-latest" if "Large" in model_option else "mistral-small-latest"
                resp = client.chat.complete(model=m_model, messages=[{"role": "user", "content": full_prompt}], stream=False)
                st.write(resp.choices[0].message.content)

        except AuthenticationError:
            st.error("Invalid OpenAI API key. Please check and try again.")
        except APIConnectionError:
            st.error("Network error: Unable to reach OpenAI servers.")
        except Exception as e:
            st.error(f"An error occurred: {e}")
