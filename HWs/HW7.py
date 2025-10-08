import streamlit as st
import pandas as pd
from openai import OpenAI
from anthropic import Anthropic

# Page Titling
st.set_page_config(page_title="HW7: Legal NewsBot", layout="wide")
st.title("HW7: Legal NewsBot: News from a CSV file")

# API Keys
openai_key = st.secrets.get("OpenAI_API_KEY")
anthropic_key = st.secrets.get("Anthropic_API_KEY")

# Model Selector from Sidebar
st.sidebar.header("Settings")
model_option = st.sidebar.selectbox("Choose LLM:", [
    "gpt-5-nano (OpenAI)",
    "gpt-4o (OpenAI)",
    "Claude Sonnet 4 (Anthropic)",
    "Claude Opus 4 (Anthropic)"
])

# Function to load the csv into a dataframe with pandas
@st.cache_data #keeps the csv contents in cache
def load_news_csv(path):
    df = pd.read_csv(path)
    df["Document"] = df["Document"].fillna("")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df
df = load_news_csv("/workspaces/HW-DocumentQA/hw7files/Example_news_info_for_testing.csv")

# Scoring function, using key legal keywords
def get_top_ranked_news(df, top_k=5):
    legal_keywords = [
        "lawsuit", "SEC", "CFPB", "FTC", "compliance", "fraud",
        "investigation", "regulator", "DOJ", "probe", "audit", "legal"
    ]
    scores = df["Document"].str.lower().apply(lambda text: sum(kw in text for kw in legal_keywords))
    df = df.copy()
    df["legal_score"] = scores
    return df.sort_values(by=["legal_score", "days_since_2000"], ascending=[False, False]).head(top_k)

# Function find news by keyword
def find_news_about(df, keyword):
    return df[df["Document"].str.lower().str.contains(keyword.lower())]

# Model Handling with OpenAI and Anthropic Architectures
def ask_model(system_prompt, user_prompt):
    if model_option.startswith("gpt"):
        if not openai_key:
            return " Missing OpenAI API key."
        client = OpenAI(api_key=openai_key)
        model_id = "gpt-4o" if "4o" in model_option else "gpt-5-nano"
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        res = client.chat.completions.create(model=model_id, messages=messages)
        return res.choices[0].message.content.strip()

    elif model_option.startswith("Claude"):
        if not anthropic_key:
            return "Missing Anthropic API key."
        client = Anthropic(api_key=anthropic_key)
        model_id = "claude-sonnet-4-20250514" if "Sonnet" in model_option else "claude-opus-4-20250514"
        res = client.messages.create(
            model=model_id,
            max_tokens=1000,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}]
        )
        return res.content[0].text.strip()
    else:
        return "Invalid model selected."

# Initialize Chat History
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
for role, msg in st.session_state.chat_history:
    st.chat_message(role).markdown(msg)

# New Input Query
query = st.chat_input("Ask a question like 'Find the most interesting news' or 'News about JPMorgan'")
if query:
    st.chat_message("user").markdown(query)
    st.session_state.chat_history.append(("user", query))

    query_lower = query.lower()

    if "interesting" in query_lower:
        top_news = get_top_ranked_news(df, top_k=5)
        context = "\n\n".join([
            f"{row['Date'].date()} — {row['company_name']}: {row['Document'][:300]}..."
            for _, row in top_news.iterrows()
        ])
        user_prompt = f"From the following news stories, identify and explain the most legally significant items for a global law firm:\n\n{context}"
        system_prompt = "You are a legal news analyst helping lawyers find regulatory and legal issues in business news."
        answer = ask_model(system_prompt, user_prompt)

    elif "news about" in query_lower:
        keyword = query_lower.split("news about")[-1].strip()
        if not keyword:
            answer = "❗ Please specify a company or topic after 'news about'."
        else:
            filtered = find_news_about(df, keyword)
            if filtered.empty:
                answer = f"No news found about '{keyword}'. Try another keyword."
            else:
                context = "\n\n".join([
                    f"{row['Date'].date()} — {row['company_name']}: {row['Document'][:300]}..."
                    for _, row in filtered.head(5).iterrows()
                ])
                user_prompt = f"Summarize and analyze news about {keyword}, focusing on legal or financial significance:\n\n{context}"
                system_prompt = "You are a legal assistant providing insights into news stories relevant to legal teams."
                answer = ask_model(system_prompt, user_prompt)

    else:
        context = "\n\n".join([
            f"{row['Date'].date()} — {row['company_name']}: {row['Document'][:300]}..."
            for _, row in df.head(10).iterrows()
        ])
        user_prompt = f"The user asked: '{query}'. Use the following news to answer:\n\n{context}"
        system_prompt = "You are a legal and financial news analyst."
        answer = ask_model(system_prompt, user_prompt)

    st.chat_message("assistant").markdown(answer)
    st.session_state.chat_history.append(("assistant", answer))
