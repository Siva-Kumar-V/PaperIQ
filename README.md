# ȺƗ ℝɆŞɆɄɌƇĦ ₱₳₱ɆⱤ ȺŞŞƗŞŦ₳ŇŦ (AI Research Paper Assistant)

An interactive AI tool to **analyze, summarize, and query research papers** using both RAG (Retrieval-Augmented Generation) and TF-IDF methods. Upload PDF, DOCX, or TXT files, and explore summaries, entities, methodologies, and visual insights.

---

## 🌟 Features

- Upload PDF, DOCX, or TXT research papers  
- Extract text and preprocess it (cleaning, tokenization, lemmatization, stopword removal)  
- Extractive summarization of documents  
- **RAG-based Q&A** with vector embeddings and local LLM (Flan-T5)  
- TF-IDF fallback Q&A  
- Entity extraction using POS tagging and noun phrase chunking  
- Methodology detection with keyword matching  
- Suggested questions for paper exploration  
- Visual insights: Wordcloud and Top Words Bar Chart  
- Chatbot interface for interactive exploration  
- Gradio multi-tab UI for ease of use  

---

## ⚡ Installation

Clone this repository:

git clone <your-repo-url>
cd <repo-folder>
Install required dependencies:

bash
Copy code
pip install -r requirements.txt
If you don’t have requirements.txt, install manually:

bash
Copy code
pip install PyPDF2 python-docx gradio nltk wordcloud matplotlib scikit-learn
pip install sentence-transformers faiss-cpu transformers accelerate langchain langchain-community
Download required NLTK packages (done automatically in code):

python
Copy code
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
🚀 Usage
Run the application:

bash
Copy code
python app.py
Open the Gradio interface in your browser (e.g., http://127.0.0.1:7860).

Tabs Overview
Upload – Upload your research paper and process it

Chatbot – Ask questions or get summaries interactively

Entities – View extracted entities from the document

Methods – Detect methodologies and generate suggested questions

Visualization – Generate wordclouds and bar charts of top words

Utilities – Summarize the document and ask direct questions

📂 Supported File Types
PDF (.pdf)

Word Document (.docx)

Plain Text (.txt)

🤖 How it Works
Extracts text from the uploaded document

Preprocesses text (tokenization, lemmatization, stopword removal)

Builds RAG-based vector database and local LLM for Q&A

Falls back to TF-IDF for documents too small for RAG

Allows entity extraction and methodology detection

Generates visual insights (wordcloud & top words bar chart)

💡 Example Workflow
Upload a paper (PDF/DOCX/TXT)

Click Process Document

Ask questions in the Chatbot tab

Explore entities, methods, and visual insights

Generate summaries or answers in the Utilities tab

⚙️ Requirements
Python 3.9+

PyPDF2

python-docx

gradio

nltk

wordcloud

matplotlib

scikit-learn

sentence-transformers

faiss-cpu

transformers

accelerate

langchain

langchain-community

📚 References
LangChain – for RAG & embeddings

Hugging Face Transformers – local LLM

Gradio – UI for interaction

📝 License
MIT License © 2025 SIVA KUMAR VAVILAPALLI
