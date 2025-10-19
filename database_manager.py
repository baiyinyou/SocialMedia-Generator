# database_manager.py
import os
import requests
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.docstore.document import Document

# === 配置 ===
API_KEY = "0ce5a0b4abf7496cb06837bc8b6b85d3" 
PERSIST_DIR = "./vector_db"

def load_chroma():
    """加载持久化的 Chroma 向量数据库"""
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma(
        collection_name="linkedin_insight",
        embedding_function=embedding,
        persist_directory=PERSIST_DIR
    )

def update_database_from_api():
    """从 NewsAPI 拉取最新科技新闻并写入数据库"""
    print("Fetching new tech news from NewsAPI...")
    url = f"https://newsapi.org/v2/top-headlines?category=technology&language=en&apiKey={API_KEY}"
    resp = requests.get(url, timeout=15)
    data = resp.json()

    if "articles" not in data:
        print("⚠️ Failed to fetch or invalid API response.")
        return

    vectorstore = load_chroma()

    new_docs = []
    for article in data["articles"]:
        title = article.get("title", "")
        desc = article.get("description", "")
        content = article.get("content", "")
        combined = f"{title}\n{desc}\n{content}"
        if len(combined.strip()) > 30:
            new_docs.append(Document(page_content=combined))

    if not new_docs:
        print("No new articles found.")
        return

    vectorstore.add_documents(new_docs)
    vectorstore.persist()
    print(f"Added {len(new_docs)} new documents to ChromaDB.")
