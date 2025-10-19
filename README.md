# Multimodal-SocialMedia-Generator

## 1. Overview

**Multimodal Social Media Insight Generator** is a Streamlit-based research prototype that combines text and vision understanding with Retrieval-Augmented Generation (RAG).  
It automatically generates multilingual, platform-ready LinkedIn posts based on:
- Uploaded screenshots or images (via OCR and vision captioning),
- Web articles or online news sources (via web scraping and API integration).

The system integrates multimodal embedding, vector retrieval, and generative language models (Groq Llama 3.1) to provide concise, contextualized content generation.

---

## 2. System Architecture

The system follows a modular, layered architecture:

```

User Input (Images / URLs / Keywords)
│
▼
[1] Data Preprocessing
├── OCR (pytesseract)
├── Web Content Extraction (trafilatura)
└── Text Cleaning & Deduplication

[2] Multimodal Embedding & Storage
├── Text Embedding (HuggingFace sentence-transformers)
├── Vision Captioning (BLIP)
├── Image Embedding (CLIP / custom embedding)
└── Persistent Vector DB (Chroma)

[3] Retrieval-Augmented Generation
├── Query Encoding and Top-k Retrieval
└── Contextual Prompt Construction (LangChain PromptTemplate)

[4] Multilingual Post Generation
├── LLM (ChatGroq, Llama-3.1-8b-instant)
├── Persona and Platform Conditioning
└── Output: Multi-language, Style-aligned Post Text

[5] Optional Visualization
├── Auto-generated Cover Image (PIL)
└── Streamlit UI Display

````

---

## 3. Key Components

| File | Description |
|------|--------------|
| `app.py` | Main Streamlit application interface; orchestrates all modules. |
| `appcreate.py` | Core RAG pipeline implementation (embedding, retrieval, generation). |
| `datacleaning.py` | Text normalization, HTML removal, OCR noise filtering, and deduplication. |
| `imagegen.py` | Generates simple platform cover images and performs image captioning. |
| `main.py` | Pipeline definition and integration logic. |
| `database_manager.py` | Supports external API integration (e.g., NewsAPI, ArXiv) for dynamic updates. |
| `requirements.txt` | Dependency list for reproducibility. |
| `.env.example` | Template for environment variable configuration (no API keys included). |

---

## 4. Pipeline Description

### 4.1 Input Stage
Users may provide:
- **Screenshots / charts** (uploaded locally)
- **News URLs** (e.g., BBC, Reuters, TechCrunch)
- **Topic hints** for retrieval expansion

### 4.2 Data Preprocessing
1. **OCR Extraction:** Text is extracted from uploaded images using Tesseract.  
2. **Web Scraping:** Clean text is retrieved from web links using `trafilatura`.  
3. **Data Cleaning:**  
   - Remove HTML, duplicated sentences, and short noise fragments.  
   - Language detection filters out non-English/Chinese content.  
4. **Optional Semantic Deduplication:** Similar embeddings are merged using cosine similarity.

### 4.3 Multimodal Vectorization
- Text is embedded via `sentence-transformers/all-MiniLM-L6-v2`.
- Each image generates:
  - A **caption** (via BLIP model)
  - An **embedding** vector (via CLIP or custom vision encoder)
- Both text and image vectors are stored persistently in a **Chroma** vector database.

### 4.4 Retrieval-Augmented Generation (RAG)
At query time:
1. A semantic query (topic hint or abstract summary request) is embedded.
2. Top-k similar text/image chunks are retrieved.
3. The retrieved context is concatenated to a structured prompt template.

### 4.5 Multilingual Generation
A unified prompt is passed to **Llama 3.1 via Groq API**, producing coherent, style-adapted posts in multiple languages:
- English (en)
- Chinese (zh)
- Swedish (sv)

Tone, platform, and length are configurable via the sidebar interface.

### 4.6 Visualization & Export
- The output posts are displayed in Streamlit with copy and download options.  
- A simple cover image (text + gradient background) is generated locally for visual branding.

---

## 5. Example Usage

### Local Run
```bash
# 1. Clone the repository
git clone https://github.com/<your_username>/Multimodal-LinkedIn-Insight-Generator.git
cd Multimodal-LinkedIn-Insight-Generator

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your Groq key to .env
echo "GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxx" > .env

# 5. Launch app
streamlit run app.py
````

---

## 6. Deployment

### 6.1 Streamlit Cloud

1. Push this repository to GitHub.
2. Go to [https://share.streamlit.io](https://share.streamlit.io).
3. Select your repo and specify `app.py` as the main entry point.
4. Add your environment variable in the "Secrets" section:

   ```
   GROQ_API_KEY = gsk_xxxxxxxxxxxxxx
   ```
5. Deploy and share your public app link.

### 6.2 Hugging Face Spaces

Alternatively, deploy on [Hugging Face Spaces](https://huggingface.co/spaces) with SDK = Streamlit.

---

## 7. Extensibility

This system is designed for extensibility:

* **Database Expansion:** Integrate ArXiv, NewsAPI, or academic datasets for live updates.
* **Vision Model Replacement:** Swap BLIP for CLIP, Florence-2, or LLaVA.
* **LLM Plug-in:** Replace Groq Llama with OpenAI GPT, Anthropic Claude, or local Mistral.
* **Cross-domain Applications:** Adapt the same pipeline for scientific insight summarization, brand monitoring, or news sentiment analysis.

---

## 8. Limitations

* The current vision encoder is limited to single-frame captioning.
* Multilingual coverage is optimized for English, Chinese, and Swedish.
* Online retrieval relies on external APIs; rate limits may apply.

---

## 9. License

MIT License — free for research and educational use.

---

## 10. Citation

If you use this project in academic or applied research, please cite:

```
@software{linkedinsight2025,
  title = {Multimodal LinkedIn Insight Generator — Vision + Text RAG},
  author = {Baiyinyou},
  year = {2025},
  url = {https://github.com/<your_username>/Multimodal-LinkedIn-Insight-Generator}
}
