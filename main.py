# main.py
import streamlit as st
from datacleaning import clean_texts_pipeline
from appcreate import build_rag_model, generate_multilang_posts
from utils import ocr_image, fetch_article
from imagegen import render_cover_image

def run_pipeline(img_files, urls_text, langs, persona, platform_target,
                 add_hashtags, add_emojis, length, topic_hint):

    blobs = []

    with st.spinner("Extracting from images and URLs..."):
        if img_files:
            for f in img_files:
                blobs.append(ocr_image(f))
        for u in [u.strip() for u in urls_text.splitlines() if u.strip()]:
            blobs.append(fetch_article(u))

    if not blobs and not img_files:
        st.warning("Please provide at least one image or URL.")
        return

    with st.spinner("Cleaning text..."):
        blobs = clean_texts_pipeline(blobs)

    with st.spinner("Building multimodal vectorstore..."):
        vs = build_rag_model(blobs, img_files)

    query = topic_hint or "Summarize key insights for professionals"
    retriever = vs.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke(query)
    context = "\n\n---\n\n".join([d.page_content for d in docs])

    st.subheader("② Generated Results")
    outputs = generate_multilang_posts(
        langs, persona, platform_target, context,
        length, add_emojis, add_hashtags, topic_hint
    )

    for lang, text in outputs.items():
        st.markdown(f"#### {lang.upper()} · {platform_target}")
        st.text_area("Post", text, height=280)
        title = text.splitlines()[0][:60] if text.strip() else "Insight"
        cover = render_cover_image(title, topic_hint or "AI-driven insight")
        st.image(cover, caption="Auto-generated cover image")
