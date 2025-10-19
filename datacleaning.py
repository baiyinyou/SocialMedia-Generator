import re
from langdetect import detect
from sentence_transformers import SentenceTransformer
import numpy as np

def clean_text(text):
    """Basic cleaning pipeline."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"(Read more|Click here|Share this).*", "", text, flags=re.I)
    if len(text) < 40:
        return ""
    try:
        if detect(text) not in ["en", "zh"]:
            return ""
    except:
        pass
    return text.strip()

def deduplicate_texts(blobs, threshold=0.95):
    """Semantic deduplication via embeddings."""
    if len(blobs) <= 1:
        return blobs
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embs = model.encode(blobs, normalize_embeddings=True)
    keep = []
    for i, e in enumerate(embs):
        if all(np.dot(e, embs[j]) < threshold for j in keep):
            keep.append(i)
    return [blobs[i] for i in keep]

def clean_texts_pipeline(blobs):
    texts = [clean_text(b) for b in blobs if b.strip()]
    texts = [t for t in texts if t]
    if len(texts) > 3:
        texts = deduplicate_texts(texts)
    return texts
