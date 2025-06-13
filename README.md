# AskTheDoc
AI agent which can read the doc and answer your quries

SAskTheDoc is an intelligent chatbot that can read and understand uploaded PDF documents, and then answer user questions based on the content. It leverages cutting-edge technologies such as Google Generative AI, LangChain, and FAISS for embedding and retrieval, all wrapped inside a sleek Streamlit UI.


### 🚀 Features
- 📚 Upload one or more PDFs
- 🔍 Extracts and preprocesses text from PDFs
- ✨ Displays word count and most frequent terms
- 💬 Asks questions about the documents
- 🤖 Uses Google Generative AI (Gemini) for intelligent answers
- 🧠 Named Entity Recognition (NER) with SpaCy
- 📊 Displays response time for each answer
- ☁️ Vector storage with FAISS
- 🛠 Built with Streamlit, LangChain, NLTK, SpaCy, and more

### 🧰 Tech Stack
- Frontend/UI: Streamlit
- LLM: Google Gemini (gemini-1.5-flash)
- Embeddings: GoogleGenerativeAIEmbeddings (embedding-001)
- Text Splitting & QA Chain: LangChain
- Vector Store: FAISS
- NER: SpaCy (en_core_web_sm)
- PDF Reader: PyPDF2
- Preprocessing: NLTK
- Visualization: WordCloud, Pandas

### 📝 Installation
```bash
git clone https://github.com/your-username/smart-guide-agent.git
cd smart-guide-agent
```

### 🔧 Install Requirements
```bash
pip install -r requirements.txt
```

### 📦 Required NLTK & SpaCy Models
```bash
python -m nltk.downloader punkt stopwords wordnet
python -m spacy download en_core_web_sm
```

### 🔐 Setup Google API Key
Get your API key from Google AI Studio.
Create a .env file in the root directory:
```
GOOGLE_API_KEY=your_google_api_key
```

### ▶️ Run the App
```bash
streamlit run app.py
```