from flask import Flask, render_template, request, jsonify
from langchain_google_genai import ChatGoogleGenerativeAI
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import pickle
import os
import gdown

app = Flask(__name__)

# Set environment variables
os.environ["LLAMA_INDEX_LLM"] = "none"
os.environ["LLAMA_INDEX_EMBEDDING_MODEL"] = "huggingface"

# Download index.pkl from Google Drive if not present
def download_index():
    url = "https://drive.google.com/file/d/1JpmszxtWqc-jW_NwSMdSPdnYxAI_tuvF/view?usp=drivesdk"  # <-- Replace YOUR_FILE_ID
    output = "index.pkl"
    if not os.path.exists(output):
        print("Downloading index.pkl from Google Drive...")
        gdown.download(url, output, quiet=False)

download_index()

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

def markdown_table_to_html(markdown_text: str) -> str:
    lines = [line.strip() for line in markdown_text.strip().split('\n') if line.strip().startswith('|')]
    if len(lines) < 2:
        return "<p>No table data available.</p>"
    headers = [h.strip() for h in lines[0].split('|')[1:-1]]
    rows = [[col.strip() for col in row.split('|')[1:-1]] for row in lines[2:]]
    table_html = '<table class="styled-table"><thead><tr>'
    for header in headers:
        table_html += f'<th>{header}</th>'
    table_html += '</tr></thead><tbody>'
    for row in rows:
        table_html += '<tr>' + ''.join(f'<td>{cell}</td>' for cell in row) + '</tr>'
    table_html += '</tbody></table>'
    return table_html

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

@app.route("/")
def home():
    return render_template("chat.html")

@app.route("/chat", methods=["POST"])
def chat_api():
    data = request.get_json()
    user_input = data.get("message", "")
    if not user_input:
        return jsonify({"error": "Message is required"}), 400
    raw_markdown = get_markdown_response(user_input)
    html_table = markdown_table_to_html(raw_markdown)
    return jsonify({"response": html_table})

@app.route("/recom", methods=["POST"])
def recommendation_api():
    data = request.get_json()
    user_input = data.get("message", "")
    if not user_input:
        return jsonify({"error": "Message is required"}), 400
    raw_markdown = get_markdown_response(user_input)
    structured_data = markdown_table_to_json(raw_markdown)
    return jsonify(structured_data)

if __name__ == "__main__":
    app.run(debug=True)
