import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import pickle
import os
import pandas as pd

# Set environment variables
os.environ["LLAMA_INDEX_LLM"] = "none"
os.environ["LLAMA_INDEX_EMBEDDING_MODEL"] = "huggingface"

# Setup embedding model
hf_embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5")
Settings.embed_model = hf_embed_model
Settings.llm = None

# Load index
with open('index.pkl', 'rb') as f:
    index = pickle.load(f)

query_engine = index.as_query_engine()

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.7,
    google_api_key="AIzaSyA7YdIMRXPIlHfPSsn3vN3ZkiffBQhhEy0"  # Replace with your actual API key
)

# Markdown table → JSON
def markdown_table_to_json(markdown_text: str) -> list:
    lines = [line.strip() for line in markdown_text.strip().split('\n') if line.strip().startswith('|')]
    if len(lines) < 2:
        return []

    headers = [h.strip() for h in lines[0].split('|')[1:-1]]
    rows = [[col.strip() for col in row.split('|')[1:-1]] for row in lines[2:]]

    data = []
    for row in rows:
        item = dict(zip(headers, row))
        data.append({
            "title": item.get("Assessment Name"),
            "url": item.get("URL"),
            "remote_testing": item.get("Remote Testing Support", "").lower() == "yes",
            "adaptive_irt": item.get("Adaptive/IRT Support", "").lower() == "yes",
            "duration": f"Approximate Completion Time: {item.get('Duration')}",
            "test_type": [x.strip() for x in item.get("Test Type", "").split(',')]
        })
    return data

# Gemini response
def get_markdown_response(user_input):
    context = query_engine.query(user_input)

    prompt = f"""
    You are an AI chatbot having a friendly and helpful conversation.

    Please answer in a markdown table with the following columns:
    - Assessment Name
    - URL
    - Remote Testing Support (Yes/No)
    - Adaptive/IRT Support (Yes/No)
    - Duration
    - Test Type

    User Query:
    {user_input}

    Context:
    {context}

    Format the response as a markdown table using | separators only.
    """

    response = llm.invoke(prompt)
    return response.content if hasattr(response, 'content') else str(response)

# Streamlit UI
st.set_page_config(page_title="SHL Assessment Recommender", layout="wide")
st.title("📊 SHL Assessment Recommendation Chatbot")

with st.form("query_form"):
    user_input = st.text_input("Ask something:", placeholder="e.g. Recommend assessments for leadership roles")
    submitted = st.form_submit_button("Submit")

if submitted and user_input:
    with st.spinner("Thinking..."):
        markdown_response = get_markdown_response(user_input)
        json_data = markdown_table_to_json(markdown_response)

        if json_data:
            st.subheader("📌 Recommended Assessments")
            df = pd.DataFrame(json_data)
            df_display = df.drop(columns=["remote_testing", "adaptive_irt", "test_type"])
            st.dataframe(df_display, use_container_width=True)

            with st.expander("🔍 Full JSON Output"):
                st.json(json_data)
        else:
            st.warning("No recommendations found. Please try rephrasing your query.")
