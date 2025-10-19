# vision_embedding.py
from PIL import Image
from sentence_transformers import SentenceTransformer

clip_model = SentenceTransformer("clip-ViT-B-32")

def embed_image(image_file):
    """返回图像的 CLIP 向量（与文本共空间）"""
    img = Image.open(image_file).convert("RGB")
    emb = clip_model.encode([img], normalize_embeddings=True)
    return emb[0]
