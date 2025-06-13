import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.probability import FreqDist
import nltk
from nltk.translate.bleu_score import sentence_bleu
import spacy
from spacy import displacy
import os
from wordcloud import WordCloud
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from time import perf_counter
import numpy as np
import base64
from PIL import Image

# Ensure necessary NLTK data is downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load SpaCy model
NER = spacy.load("en_core_web_sm")

# Initialize WordNetLemmatizer
lemma = WordNetLemmatizer()

# Function to read the uploaded PDF documents
def get_text_from_pdfs(pdf_documents):
    text = ""
    for pdf in pdf_documents:
        pdf_reader = PdfReader(pdf)
        num_pages = len(pdf_reader.pages)
        #st.write(f"1. Number of pages in the PDF are {num_pages}")
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split text into chunks
def get_text_chunks_from_pdfs(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to store text chunks as vector embeddings using FAISS
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Function to start the conversation using Prompt
def start_conversation():
    prompt_template = """You are an expert in reading and understanding PDF documents. Read the uploaded documents
    carefully and answer the questions asked by users. Please do not give wrong answers. Return Result in Proper format.

    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to handle user questions and return response
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    try:
        # Load FAISS index with dangerous deserialization enabled
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question,k=3)
        chain = start_conversation()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        get_named_entity(response["output_text"])
        return response["output_text"]
    except FileNotFoundError:
        st.error("The FAISS index file was not found. Please ensure it has been created.")
    except ValueError as ve:
        st.error(f"Error during FAISS loading: {ve}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

# Function to preprocess PDF text
def preprocess_pdf_text(text):
    cleaned_text = re.sub(r'[^a-zA-Z\s]', '', text).lower()
    words = cleaned_text.split()
    stop_words = set(stopwords.words('english'))
    words = [lemma.lemmatize(word) for word in words if word not in stop_words]
    return words

# Function to count words
def count_words(words):
    return len(words)

# Function to get named entities
def get_named_entity(raw_text):
    text2 = NER(raw_text)
    entities = [(ent.text, ent.label_) for ent in text2.ents]
    entities_df = pd.DataFrame(entities, columns=['Entity', 'Label'])
    st.write("3. Named Entities:")
    st.write(entities_df)

# Main function
def main():
    load_dotenv()
    os.getenv("GOOGLE_API_KEY")
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

    logo = Image.open("ai-agent.jpg")  # Replace with your logo path

    # Adjust the size: Increase width and reduce height
    new_width = 800  # Set your desired width
    new_height = 400  # Set your desired height
    logo_resized = logo.resize((new_width, new_height))

    st.image(logo_resized)
    
    # Path to your image (either local or from a URL)

    st.header("Smart Guide :Agent:")
    with st.sidebar:
        st.markdown('''
        ## About
        This LLM QnA ChatBot app is built using:
        - LangChain Framework
        - Google Generative AI
        - FAISS Vector DB
        - Streamlit Application
        ''')

    pdf_docs = st.sidebar.file_uploader("Upload your PDFs here", accept_multiple_files=True)
    user_question = st.text_input("Ask any question about your Equipments:")

    if st.button("Submit"):
        with st.spinner("Processing"):
            if pdf_docs:
                raw_text = get_text_from_pdfs(pdf_docs)
                preprocessed_words = preprocess_pdf_text(raw_text)
                word_count = count_words(preprocessed_words)
                st.write(f"1. Total Word Count from PDF is {word_count}")
                freq_dist = FreqDist(preprocessed_words)
                frequent_words = freq_dist.most_common(10)
                st.write(f"2. Frequent words from PDF are {frequent_words}")
                text_chunks = get_text_chunks_from_pdfs(raw_text)
                get_vector_store(text_chunks)
                if user_question:
                    try:
                        tick = perf_counter()
                        response = user_input(user_question)
                        st.subheader("Response :")
                        st.write(response)
                        total_time = perf_counter() - tick
                        st.subheader(f"Response Time: {np.round(total_time, 3)} seconds")
                    except Exception:
                        st.error("Error occurred during processing.")
            else:
                st.error("Please upload at least one PDF file.")

if __name__ == '__main__':
    main()
