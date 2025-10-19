# app.py
import streamlit as st
from main import run_pipeline

st.set_page_config(page_title="Insight Helper", page_icon="📝", layout="wide")
st.title("📝Insight Helper — Vision + Text RAG")

with st.sidebar:
    st.header("Configuration")
    langs = st.multiselect("Output Languages", ["zh", "en", "sv"], default=["zh", "en"])
    persona = st.selectbox("Persona", ["Professional", "Concise", "Analytical", "Friendly"], index=0)
    platform_target = st.selectbox("Platform", ["LinkedIn", "X/Twitter", "Short Video Script"], index=0)
    add_hashtags = st.checkbox("Add Hashtags", True)
    add_emojis = st.checkbox("Add Emojis", True)
    length = st.slider("Length", 200, 1200, 600, 50)
    mode = st.radio("Knowledge Source", ["Local Upload", "Online Search"], index=0)

st.subheader("① Provide Inputs")
img_files = st.file_uploader("Upload screenshots / charts", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
urls_text = st.text_area("Paste article links")
topic_hint = st.text_input("Optional topic hint", "Retrieval-Augmented Generation")

if st.button("✨ Generate Insights"):
    if mode == "Online Search":
        from online_retriever import build_online_vectorstore
        with st.spinner("Searching online sources..."):
            vs = build_online_vectorstore(topic_hint or "AI")
    else:
        from appcreate import build_rag_model
        vs = build_rag_model([], img_files)

    from appcreate import generate_multilang_posts
    from langchain_community.vectorstores import Chroma

    query = topic_hint or "Summarize key insights for professionals"
    retriever = vs.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke(query)
    context = "\n\n---\n\n".join([d.page_content for d in docs])

    outputs = generate_multilang_posts(
        langs, persona, platform_target, context, length, add_emojis, add_hashtags, topic_hint
    )

    for lang, text in outputs.items():
        st.markdown(f"#### {lang.upper()} · {platform_target}")
        st.text_area("Post", text, height=280)
