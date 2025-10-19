import os
import requests
import urllib.request
import xml.etree.ElementTree as ET
from dotenv import load_dotenv
from langchain_community.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# --------------------------
# 🔍 1. NewsAPI Source
# --------------------------
def fetch_articles_from_newsapi(query: str, max_results=10):
    if not NEWS_API_KEY:
        print("Missing NEWS_API_KEY, skipping NewsAPI source.")
        return []
    
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "language": "en",
        "pageSize": max_results,
        "sortBy": "relevancy",
        "apiKey": NEWS_API_KEY
    }

    response = requests.get(url, params=params)
    data = response.json()

    if data.get("status") != "ok":
        print(f"NewsAPI error: {data}")
        return []

    texts = []
    for article in data.get("articles", []):
        title = article.get("title") or ""
        desc = article.get("description") or ""
        content = article.get("content") or ""
        source = article.get("url") or ""
        combined = f"{title}\n{desc}\n{content}\n(Source: {source})".strip()
        if len(combined) > 50:
            texts.append(combined)
    print(f"🗞️ Retrieved {len(texts)} articles from NewsAPI.")
    return texts


# --------------------------
# 📚 2. Arxiv Source
# --------------------------
def fetch_papers_from_arxiv(query: str, max_results=5):
    """
    从 Arxiv API 获取与 query 相关的论文摘要
    """
    url = f"http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results={max_results}"
    try:
        data = urllib.request.urlopen(url).read().decode("utf-8")
        root = ET.fromstring(data)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        entries = root.findall("atom:entry", ns)

        texts = []
        for entry in entries:
            title = entry.find("atom:title", ns).text or ""
            summary = entry.find("atom:summary", ns).text or ""
            link = entry.find("atom:id", ns).text or ""
            combined = f"[Paper] {title}\n{summary}\n(Source: {link})"
            texts.append(combined.strip())
        print(f"📘 Retrieved {len(texts)} papers from Arxiv.")
        return texts
    except Exception as e:
        print(f"⚠️ Arxiv fetch error: {e}")
        return []


# --------------------------
# 🔧 3. Build Vectorstore
# --------------------------
def build_online_vectorstore(topic: str):
    """
    联网获取 NewsAPI + Arxiv 数据，并构建临时向量数据库
    """
    print(f"Fetching online knowledge for topic: {topic}")
    news_texts = fetch_articles_from_newsapi(topic, max_results=10)
    paper_texts = fetch_papers_from_arxiv(topic, max_results=5)
    all_texts = news_texts + paper_texts

    if not all_texts:
        raise ValueError("No online content found.")

    docs = [Document(page_content=t) for t in all_texts]
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vs = FAISS.from_documents(chunks, embedding)

    print(f"Indexed {len(chunks)} chunks from online sources.")
    return vs
