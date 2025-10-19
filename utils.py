import requests
import pytesseract
from PIL import Image
import trafilatura

def ocr_image(file):
    image = Image.open(file).convert("RGB")
    return pytesseract.image_to_string(image, lang="eng+chi_sim").strip()

def fetch_article(url):
    resp = requests.get(url, timeout=15)
    text = trafilatura.extract(resp.text, include_links=False)
    return text.strip() if text else ""
