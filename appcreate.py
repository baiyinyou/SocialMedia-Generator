# appcreate.py

import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from vision_embedding import embed_image          # 图像 → 向量
from imagegen import render_cover_image          # 图像 → 描述文字

def build_rag_model(blobs, img_files=None, persist_dir="./vector_db"):
    """构建多模态 RAG 向量数据库"""
    os.makedirs(persist_dir, exist_ok=True)
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma(
        collection_name="linkedin_insight",
        embedding_function=embedding,
        persist_directory=persist_dir
    )

    # --- 文本部分 ---
    if blobs:
        docs = [Document(page_content=b) for b in blobs if b.strip()]
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
        chunks = splitter.split_documents(docs)
        if chunks:
            vectorstore.add_documents(chunks)
        else:
            print("⚠️ Warning: No valid text chunks to add.")

    # --- 图像部分 ---
    if img_files:
        for f in img_files:
            try:
                caption = render_cover_image(f)
                caption_doc = Document(page_content=f"[Image Caption] {caption}")
                vectorstore.add_documents([caption_doc])

                img_emb = embed_image(f)
                vectorstore._collection.add(
                    embeddings=[img_emb],
                    documents=[f"[Image Embedding] {f.name}"]
                )
            except Exception as e:
                print(f"⚠️ Image processing failed: {e}")

    vectorstore.persist()
    return vectorstore

# === 文本生成部分 ===
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

PROMPT = ChatPromptTemplate.from_template("""
You are a multilingual content strategist. 
Based ONLY on the following CONTEXT (from both image and text), 
create a {platform} post in {lang} with the tone of {persona}. 
Length ≈ {length} characters. Emojis: {emojis}, Hashtags: {hashtags}.

[TOPIC HINT] {topic}

[CONTEXT]
{context}
""".strip())

def make_llm():
    """使用 Groq（Llama 3.1）生成器"""
    return ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model="llama-3.1-8b-instant")

def generate_multilang_posts(langs, persona, platform_target, context, length, emojis, hashtags, topic):
    """根据上下文生成多语言内容"""
    outputs = {}
    for lang in langs:
        llm = make_llm()
        chain = PROMPT | llm
        response = chain.invoke({
            "lang": lang,
            "persona": persona,
            "platform": platform_target,
            "length": length,
            "emojis": "on" if emojis else "off",
            "hashtags": "on" if hashtags else "off",
            "topic": topic or "n/a",
            "context": context
        })
        outputs[lang] = response.content.strip()
    return outputs
