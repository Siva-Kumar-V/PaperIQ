"""
AI Research Paper Assistant with Enhanced RAG Q&A
Features:
- Upload PDF / DOCX / TXT
- Extract text & preprocess
- Extractive summarization
- RAG-based Q&A with vector embeddings and local LLM
- Rule-based entity extraction
- Methodology detection
- Suggested questions generator
- Visualizations: wordcloud and top-words bar chart
- Gradio multi-tab UI
"""

# =============================
# 0. Requirements
# =============================
!pip install PyPDF2 python-docx gradio nltk wordcloud matplotlib scikit-learn
!pip install sentence-transformers faiss-cpu transformers accelerate langchain langchain-community

import re
import io
import os
import random
from collections import Counter, defaultdict
import numpy as np
import PyPDF2
import docx
import gradio as gr
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from PIL import Image

# RAG imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from langchain_community.llms import HuggingFacePipeline

# Download required NLTK packages
try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)
    nltk.download('maxent_ne_chunker', quiet=True)
    nltk.download('words', quiet=True)
except:
    pass



# =============================
# Global state
# =============================
global_text = ""
qa_chain = None
vector_db = None
rag_initialized = False

# TF-IDF fallback
qa_vectorizer = None
qa_tfidf_matrix = None
qa_original_sentences = None

# =============================
# 1. File Upload & Text Extract
# =============================
def extract_text(file):
    """Extract text from PDF, DOCX, or TXT files."""
    text = ""
    path = None

    if hasattr(file, "name"):
        path = file.name
    elif isinstance(file, str) and os.path.exists(file):
        path = file
    elif isinstance(file, dict) and "name" in file:
        path = file["name"]
    else:
        path = str(file)

    if path.lower().endswith(".pdf"):
        try:
            with open(path, "rb") as fh:
                reader = PyPDF2.PdfReader(fh)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + " "
        except Exception as e:
            return f"Error reading PDF: {e}"

    elif path.lower().endswith(".docx"):
        try:
            doc = docx.Document(path)
            for para in doc.paragraphs:
                text += para.text + " "
        except Exception as e:
            return f"Error reading DOCX: {e}"

    elif path.lower().endswith(".txt"):
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                text = fh.read()
        except Exception as e:
            return f"Error reading TXT: {e}"
    else:
        return "Unsupported file type. Upload PDF/DOCX/TXT."

    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# =============================
# 2. Text Preprocessing
# =============================
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """Clean, tokenize, lemmatize, and remove stopwords."""
    text = re.sub(r'[^a-zA-Z0-9\s\.\,]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip().lower()
    tokens = nltk.word_tokenize(text)
    processed = [lemmatizer.lemmatize(tok) for tok in tokens if tok.isalpha() and tok not in stop_words]
    return ' '.join(processed), processed

# =============================
# 3. Summarization (extractive)
# =============================
def summarize_text(text, top_n=10):
    if not text or len(text) < 50:
        return "Document too short to summarize."

    sentences = re.split(r'(?<=[.!?]) +', text)
    processed_text, _ = preprocess_text(text)
    freq = Counter(processed_text.split())

    sentence_scores = {}
    for sent in sentences:
        words = [w for w in re.findall(r'\w+', sent.lower()) if w not in stop_words]
        if not words:
            continue
        sentence_scores[sent] = sum(freq.get(w, 0) for w in words) / len(words)

    top_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:top_n]
    return " ".join(top_sentences).strip()

# =============================
# 4. RAG-based Q&A Setup
# =============================
def setup_rag_qa(text):
    """Initialize RAG system with vector embeddings and local LLM."""
    global qa_chain, vector_db, rag_initialized

    if not text or len(text) < 100:
        return "Document too short for RAG setup."

    try:
        # Split text into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = splitter.split_text(text)

        # Create embeddings and vector store
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_db = FAISS.from_texts(chunks, embeddings)

        # Load local LLM
        model_name = "google/flan-t5-base"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        hf_pipeline = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=512,
            temperature=0.7,
            top_p=0.95
        )

        local_llm = HuggingFacePipeline(pipeline=hf_pipeline)

        # Build RetrievalQA chain
        retriever = vector_db.as_retriever(search_kwargs={"k": 3})
        qa_chain = RetrievalQA.from_chain_type(
            llm=local_llm,
            retriever=retriever,
            return_source_documents=False
        )

        rag_initialized = True
        return f"RAG Q&A initialized successfully with {len(chunks)} chunks."

    except Exception as e:
        rag_initialized = False
        return f"RAG setup failed: {e}. Using fallback TF-IDF method."

# =============================
# 5. TF-IDF Fallback Q&A
# =============================
def setup_tfidf_qa(text):
    """Setup TF-IDF based Q&A as fallback."""
    global qa_vectorizer, qa_tfidf_matrix, qa_original_sentences

    if not text:
        return "No text loaded."

    qa_original_sentences = [s.strip() for s in re.split(r'(?<=[.!?]) +', text) if len(s.strip()) > 10]
    processed_sentences = [preprocess_text(s)[0] for s in qa_original_sentences]

    qa_vectorizer = TfidfVectorizer()
    qa_tfidf_matrix = qa_vectorizer.fit_transform(processed_sentences)
    return "TF-IDF Q&A setup complete."

def answer_question_tfidf(question):
    """Fallback TF-IDF based Q&A."""
    global qa_vectorizer, qa_tfidf_matrix, qa_original_sentences

    try:
        if qa_vectorizer is None:
            return "Please upload a document first."

        proc_q = preprocess_text(question)[0]
        q_vec = qa_vectorizer.transform([proc_q])
        sims = cosine_similarity(q_vec, qa_tfidf_matrix).flatten()
        best_idx = int(np.argmax(sims))

        if sims[best_idx] < 0.1:
            return "Sorry, I couldn't find a relevant answer in the document."

        return qa_original_sentences[best_idx]
    except Exception as e:
        return f"Error in QA: {e}"

def answer_question(question):
    """Answer questions using RAG or fallback to TF-IDF."""
    global qa_chain, rag_initialized

    if rag_initialized and qa_chain is not None:
        try:
            response = qa_chain.run(question)
            return response
        except Exception as e:
            print(f"RAG error: {e}, falling back to TF-IDF")
            return answer_question_tfidf(question)
    else:
        return answer_question_tfidf(question)

# =============================
# 6. Entity Extraction
# =============================
def extract_entities(text, top_n=25):
    """Extract entities using POS tagging and noun phrase chunking."""
    if not text:
        return {}

    tokens = nltk.word_tokenize(text)
    tagged = nltk.pos_tag(tokens)

    grammar = r"NP: {<DT|JJ|NN.*>+}"
    cp = nltk.RegexpParser(grammar)
    tree = cp.parse(tagged)

    candidates = []
    for subtree in tree.subtrees(filter=lambda t: t.label() == 'NP'):
        words = [w for w, pos in subtree.leaves()]
        candidate = " ".join(words)
        if len([c for c in re.findall(r'\w+', candidate)]) > 0:
            candidates.append(candidate)

    cap_seqs = re.findall(r'\b([A-Z][a-z0-9]+(?:\s+[A-Z][a-z0-9]+)*)\b', text)
    candidates.extend(cap_seqs)

    norm = [re.sub(r'\s+', ' ', c).strip() for c in candidates]
    freq = Counter(norm)
    most = freq.most_common(top_n)
    return {item: count for item, count in most}

# =============================
# 7. Methodology Detection
# =============================
METHOD_KEYWORDS = [
    "dataset", "experiment", "evaluation", "methodology", "approach",
    "proposed", "framework", "analysis", "training", "testing",
    "implementation", "algorithm", "baseline", "architecture", "loss",
    "optimizer", "cross-validation", "ablation", "protocol"
]

def extract_methods(text):
    if not text:
        return []
    found = []
    lower = text.lower()
    for kw in METHOD_KEYWORDS:
        if kw in lower:
            found.append(kw)
    return sorted(set(found)) if found else ["No explicit methods detected."]

# =============================
# 8. Suggested Questions
# =============================
SUGGESTED_PROMPTS = [
    "What is the main contribution of the paper?",
    "Which dataset(s) did the authors use?",
    "What methodology or algorithm is proposed?",
    "How did the authors evaluate their approach?",
    "What are the main results and metrics reported?",
    "Are there any notable limitations or future work?",
    "What baseline methods were compared?",
    "What experiments were run and with what configuration?"
]

def generate_suggested_questions(text, top_k=5):
    questions = list(SUGGESTED_PROMPTS)

    methods = extract_methods(text)
    if methods and "No explicit methods detected." not in methods:
        for m in methods[:top_k]:
            questions.append(f"What does the paper mention about \"{m}\"?")

    entities = extract_entities(text, top_n=10)
    for ent in list(entities.keys())[:top_k]:
        questions.append(f"What role does \"{ent}\" play in the paper?")

    uniq = []
    for q in questions:
        if q not in uniq:
            uniq.append(q)
        if len(uniq) >= top_k:
            break
    return uniq

# =============================
# 9. Visualization
# =============================
def create_wordcloud_image(text, max_words=150):
    if not text:
        img = Image.new("RGB", (800, 500), color=(255,255,255))
        return img
    wc = WordCloud(width=800, height=500, background_color="white", max_words=max_words).generate(text)
    return wc.to_image()

def plot_top_words_bar(text, top_n=20):
    if not text:
        img = Image.new("RGB", (800, 400), color=(255,255,255))
        return img

    processed, tokens = preprocess_text(text)
    freq = Counter(tokens).most_common(top_n)
    words, counts = zip(*freq) if freq else ([], [])

    plt.figure(figsize=(10,5))
    plt.bar(range(len(words)), counts)
    plt.xticks(range(len(words)), words, rotation=45, ha='right')
    plt.title("Top words (lemmatized, stopwords removed)")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    return Image.open(buf)

# =============================
# 10. Document Continuation
# =============================
def continue_from_doc(text, user_input, window=50):
    if not text or not user_input:
        return None

    words = text.split()
    user_words = user_input.split()
    n = len(user_words)

    for i in range(len(words) - n):
        if words[i:i+n] == user_words:
            continuation = words[i+n:i+n+window]
            if not continuation:
                return None
            continuation_text = " ".join(continuation)
            if "." in continuation_text:
                continuation_text = continuation_text.split(".")[0] + "."
            return user_input + " " + continuation_text
    return None

# =============================
# 11. Chatbot Function
# =============================
FALLBACK_RESPONSES = [
    "I didn't quite get that. Ask me to *summarize*, ask a *question*, or provide a sentence to continue.",
    "Not sure what you mean 🤷. Want me to summarize the file instead?",
    "That doesn't look like something from the document. Can you rephrase?",
    "Hey 👋, Try: 'Summarize', 'What datasets used?', or paste a sentence from the document to continue it."
]

GREETINGS = ["hi","hello","hey","good morning","good evening","good afternoon"]

def chatbot(user_input):
    lower = user_input.lower().strip()

    if any(g in lower for g in GREETINGS):
        return random.choice([
            "Hello! 😊 I'm ready to help with the uploaded document.",
            "Hi 👋 — ask me to summarize, ask a question, or paste a sentence to continue.",
            "Hello there! 👋 Ready to explore your document?",
            "Hey! 😊 I can summarize your file, answer questions, or complete a sentence.",
            "Hi! 🚀 Ask me to *summarize*, *ask a question*, or type a sentence to complete."
        ])

    if "summarize" in lower:
        return summarize_text(global_text, top_n=10)

    if user_input.endswith("?"):
        return answer_question(user_input)

    cont = continue_from_doc(global_text, user_input)
    if cont:
        return cont

    return random.choice(FALLBACK_RESPONSES)

# =============================
# 12. Upload & Initialize
# =============================
def upload_and_train(file):
    global global_text

    try:
        txt = extract_text(file)
        if txt.startswith("Error") or txt.startswith("Unsupported"):
            return txt

        global_text = txt

        # Setup RAG Q&A
        rag_status = setup_rag_qa(global_text)

        # Setup TF-IDF fallback
        setup_tfidf_qa(global_text)

        return f"File processed successfully. {rag_status}"

    except Exception as e:
        return f"Upload error: {e}"

# =============================
# 13. Gradio UI
# =============================
with gr.Blocks(theme=gr.themes.Soft(), css="""
.gradio-container {max-width: 1100px; margin: auto;}
.tab-header {text-align:center; font-size:20px; margin:10px 0; font-weight:bold;}
.status-box {background:#e8f4ff; padding:10px; border-radius:8px; font-style:italic; border:1px solid #cce0ff;}
[role="tab"]:nth-of-type(1) { color: #FF9800 !important; font-weight: bold; }
[role="tab"]:nth-of-type(2) { color: #4CAF50 !important; font-weight: bold; }
[role="tab"]:nth-of-type(3) { color: #3F51B5 !important; font-weight: bold; }
[role="tab"]:nth-of-type(4) { color: #9C27B0 !important; font-weight: bold; }
[role="tab"]:nth-of-type(5) { color: #009688 !important; font-weight: bold; }
[role="tab"]:nth-of-type(6) { color: #795548 !important; font-weight: bold; }
.main-heading {
    font-size: 34px;
    font-weight: bold;
    background: linear-gradient(90deg, #FF9800, #3F51B5, #9C27B0);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
""") as app:

    gr.Markdown("""
    <div style="text-align:center">
        <h1> 🤖 <span class="main-heading">ȺƗ ℝɆŞɆɄɌƇĦ ₱₳₱ɆⱤ ȺŞŞƗŞŦ₳ŇŦ</span> </h1>
        <p style="color:gray;">Upload your research paper and explore summaries, entities, methods & more</p>
    </div>
    """)

    with gr.Tab("📂 Upload"):
        gr.Markdown("<div class='tab-header' style='color:#FF9800;'>Upload your Research Paper</div>")
        with gr.Row():
            file_input = gr.File(label="Choose PDF / DOCX / TXT", file_types=[".pdf", ".docx", ".txt"], file_count="single", type="filepath")
        with gr.Row():
            upload_btn = gr.Button("🚀 Process Document", variant="primary")
            clear_btn = gr.Button("🧹 Clear File")
        status = gr.Textbox(label="Status", elem_classes="status-box", interactive=False)

        upload_btn.click(upload_and_train, inputs=[file_input], outputs=[status])
        clear_btn.click(lambda: ("", ""), None, [file_input, status])

    with gr.Tab("💬 Chatbot"):
        gr.Markdown("<div class='tab-header' style='color:#4CAF50;'>Chat with your Document</div>")
        chatbot_box = gr.Chatbot(label="Chat with Document")
        with gr.Row():
            user_text = gr.Textbox(placeholder="Ask me something about the document...", show_label=False, scale=4)
            send_btn = gr.Button("Send", variant="primary", scale=1)
        clear_btn = gr.Button("🧹 Clear Chat")
        chat_history = gr.State([])

        def on_send(message, history):
            bot = chatbot(message)
            history = history + [(message, bot)]
            return history, history, ""

        send_btn.click(on_send, [user_text, chat_history], [chatbot_box, chat_history, user_text])
        user_text.submit(on_send, [user_text, chat_history], [chatbot_box, chat_history, user_text])
        clear_btn.click(lambda: ([], [], ""), None, [chatbot_box, chat_history, user_text])

    with gr.Tab("🔑 Entities"):
        gr.Markdown("<div class='tab-header' style='color:#3F51B5;'>Extracted Entities</div>")
        ent_btn = gr.Button("🔍 Extract Entities")
        ent_out = gr.Dataframe(headers=["Entity", "Frequency"], label="Top Entities", interactive=False)

        def ent_click():
            data = extract_entities(global_text, top_n=20)
            return [[k, v] for k, v in data.items()]

        ent_btn.click(ent_click, None, ent_out)

    with gr.Tab("⚙️ Methods"):
        gr.Markdown("<div class='tab-header' style='color:#9C27B0;'>Detected Methodologies</div>")
        with gr.Row():
            method_btn = gr.Button("Detect Methods")
            suggest_btn = gr.Button("💡 Suggest Questions")
        method_out = gr.HighlightedText(label="Methods Found")
        suggest_out = gr.Textbox(label="Suggested Questions", lines=7, interactive=False)

        method_btn.click(lambda: [(m, None) for m in extract_methods(global_text)], None, method_out)
        suggest_btn.click(lambda: "\n".join(generate_suggested_questions(global_text, top_k=7)), None, suggest_out)

    with gr.Tab("📊 Visualization"):
        gr.Markdown("<div class='tab-header' style='color:#009688;'>Visual Insights</div>")
        with gr.Row():
            wc_btn = gr.Button("Generate Wordcloud")
            bar_btn = gr.Button("Top Words Bar Chart")
        with gr.Row(equal_height=True):
            wc_img = gr.Image(type="pil", label="Wordcloud", scale=1)
            bar_img = gr.Image(type="pil", label="Top Words", scale=1)

        wc_btn.click(lambda: create_wordcloud_image(global_text), None, wc_img)
        bar_btn.click(lambda: plot_top_words_bar(global_text, top_n=20), None, bar_img)

    with gr.Tab("🛠 Utilities"):
        gr.Markdown("<div class='tab-header' style='color:#795548;'>Extra Tools</div>")
        with gr.Row():
            sum_btn = gr.Button("📄 Summarize Document")
            qa_q = gr.Textbox(label="Ask a direct question", placeholder="Type your question here...")
            qa_btn = gr.Button("Get Answer")
        sum_out = gr.Textbox(label="Summary", lines=5, interactive=False)
        qa_out = gr.Textbox(label="Answer", lines=3, interactive=False)

        sum_btn.click(lambda: summarize_text(global_text, top_n=10), None, sum_out)
        qa_btn.click(lambda q: answer_question(q), inputs=[qa_q], outputs=[qa_out])

if __name__ == "__main__":
    app.launch()