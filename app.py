import gradio as gr
import requests
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
import json
import os

# -------------------------------------------------------
# CUSTOMIZE THIS SECTION
# -------------------------------------------------------

YOUR_NAME = "Your Name"
YOUR_TITLE = "Data Scientist / ML Practitioner"

# Your publications — PMIDs and DOIs
PMIDS = [
    "23669799",
    "25605531",
    "26306230",
    "26306225",
    "25991289",
    "29084368",
    "29218905",
    "30815180",
    "3125899",
    "27107452",
    "31837229",
]

DOIS = [
    "10.1007/978-3-030-18626-5_5",
    "10.1101/277145",
]

# -------------------------------------------------------
# FETCH PUBLICATION METADATA AT STARTUP
# -------------------------------------------------------

def fetch_pubmed(pmid: str) -> str:
    try:
        url = (
            f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
            f"?db=pubmed&id={pmid}&retmode=xml&rettype=abstract"
        )
        resp = requests.get(url, timeout=10)
        root = ET.fromstring(resp.text)

        title = root.findtext(".//ArticleTitle") or "Unknown Title"
        abstract = root.findtext(".//AbstractText") or "No abstract available."
        journal = root.findtext(".//Journal/Title") or ""
        year = root.findtext(".//PubDate/Year") or ""
        authors = root.findall(".//Author")
        author_list = []
        for a in authors[:3]:
            ln = a.findtext("LastName") or ""
            fn = a.findtext("ForeName") or ""
            if ln:
                author_list.append(f"{ln} {fn}".strip())
        author_str = ", ".join(author_list)
        if len(authors) > 3:
            author_str += " et al."

        return (
            f"PMID: {pmid}\n"
            f"Title: {title}\n"
            f"Authors: {author_str}\n"
            f"Journal: {journal} ({year})\n"
            f"Abstract: {abstract}"
        )
    except Exception as e:
        return f"[Could not fetch PMID {pmid}: {e}]"


def fetch_doi(doi: str) -> str:
    try:
        url = f"https://api.semanticscholar.org/graph/v1/paper/{doi}?fields=title,abstract,authors,year,venue"
        resp = requests.get(url, timeout=10)
        data = resp.json()

        title = data.get("title") or "Unknown Title"
        abstract = data.get("abstract") or "No abstract available."
        year = data.get("year") or ""
        venue = data.get("venue") or ""
        authors = data.get("authors", [])
        author_list = [a.get("name", "") for a in authors[:3]]
        author_str = ", ".join(author_list)
        if len(authors) > 3:
            author_str += " et al."

        return (
            f"DOI: {doi}\n"
            f"Title: {title}\n"
            f"Authors: {author_str}\n"
            f"Venue: {venue} ({year})\n"
            f"Abstract: {abstract}"
        )
    except Exception as e:
        return f"[Could not fetch DOI {doi}: {e}]"


print("Fetching publications from PubMed and Semantic Scholar...")
pub_texts = []

for pmid in PMIDS:
    print(f"  PubMed -> PMID {pmid}")
    pub_texts.append(fetch_pubmed(pmid))

for doi in DOIS:
    print(f"  DOI -> {doi}")
    pub_texts.append(fetch_doi(doi))

PUBLICATIONS_CONTEXT = "\n\n---\n\n".join(pub_texts)
print(f"Done. Loaded {len(pub_texts)} publications.")

# -------------------------------------------------------
# SYSTEM CONTEXT (injected into every query)
# -------------------------------------------------------

BACKGROUND = f"""You are representing {YOUR_NAME}, a {YOUR_TITLE}.
Answer questions from HR professionals, hiring managers, and colleagues
at corporations, universities, and government agencies.

Draw on the publication abstracts below to:
- Search for relevant findings when asked about specific topics
- Summarize overall research themes and expertise when asked broadly
- Translate technical content into plain, professional language

Be concise, confident, and professional. If something is outside the scope
of the publications, say so and pivot to what you can speak to.

PUBLICATIONS:
{PUBLICATIONS_CONTEXT}

GENERAL BACKGROUND:
- Machine learning, NLP, pharmacogenomics, EHR/biobank informatics, precision medicine
- Experience in VA / government health systems, academic research, translational informatics
- Proficient: Python, scikit-learn, TensorFlow/Keras, autoencoders, clustering, phenotyping pipelines
- Strong communicator across technical and non-technical audiences
"""

# -------------------------------------------------------
# CHAT — uses HuggingFace free Inference API (no key needed)
# -------------------------------------------------------

HF_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
HF_TOKEN = os.environ.get("HF_TOKEN", "")

def build_prompt(message: str, history: list) -> str:
    prompt = f"<s>[INST] <<SYS>>\n{BACKGROUND}\n<</SYS>>\n\n"
    for human, assistant in history[-4:]:  # keep last 4 turns for context
        prompt += f"{human} [/INST] {assistant} </s><s>[INST] "
    prompt += f"{message} [/INST]"
    return prompt


def chat(message: str, history: list) -> str:
    prompt = build_prompt(message, history)
    try:
        resp = requests.post(
            HF_API_URL,
            headers={
                "Content-Type": "application/json",
                **({"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}),
            },
            json={
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 512,
                    "temperature": 0.4,
                    "return_full_text": False,
                },
            },
            timeout=60,
        )
        result = resp.json()
        if isinstance(result, list) and result:
            return result[0].get("generated_text", "").strip()
        elif isinstance(result, dict) and "error" in result:
            return f"Model loading, please retry in a moment. ({result['error']})"
        return "Sorry, I couldn't generate a response. Please try again."
    except Exception as e:
        return f"Error reaching model: {e}"

# -------------------------------------------------------
# UI
# -------------------------------------------------------

with gr.Blocks(
    title=f"{YOUR_NAME} -- Instant Overview",
    theme=gr.themes.Soft(primary_hue="blue"),
    css="""
        .header { text-align: center; padding: 20px 0 10px 0; }
        .subheader { text-align: center; color: #555; margin-bottom: 20px; }
        footer { display: none !important; }
    """
) as demo:

    gr.HTML(f"""
        <div class="header">
            <h1>👋 {YOUR_NAME}</h1>
            <p class="subheader">{YOUR_TITLE} &mdash; Instant Overview</p>
        </div>
    """)

    gr.ChatInterface(
        fn=chat,
        chatbot=gr.Chatbot(
            height=420,
            label="Instant Overview",
            placeholder="Ask about my research, background, tools, or experience.",
        ),
        textbox=gr.Textbox(placeholder="Ask a question...", container=False),
        examples=[
            "Summarize your research background",
            "What have you published on machine learning?",
            "Tell me about your EHR and biobank work",
            "What is pharmacogenomics and what's your experience with it?",
            "How do you approach precision medicine?",
            "Why should we hire you?",
        ],
        retry_btn=None,
        undo_btn=None,
    )

if __name__ == "__main__":
    demo.launch()
