import os
import re
import webbrowser
from datetime import datetime

import joblib
import nltk
import numpy as np
import pandas as pd
import streamlit as st
from newspaper import Article
from PIL import Image

import google.generativeai as genai

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import RetrievalQAWithSourcesChain

# ----------------------------
# Streamlit config (must be first)
# ----------------------------
st.set_page_config(page_title="Fact Finder", layout="wide")

# ----------------------------
# Sidebar navigation
# ----------------------------
st.sidebar.image("logo.png", use_column_width=True)
st.sidebar.title("🚀 Navigation")

tabs = st.sidebar.radio(
    "Go to:",
    [
        "🏠 Home",
        "📰 Fake News Detector",
        "🧠 News Research",
        "❓ FAQ",
        "👨💻 Developed By",
    ],
)

# ----------------------------
# NLTK setup (download minimal data)
# ----------------------------
nltk_data_dir = os.path.expanduser("~/nltk_data")
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)

with st.spinner("Downloading NLTK data..."):
    nltk.download("punkt", download_dir=nltk_data_dir)
    nltk.download("stopwords", download_dir=nltk_data_dir)

# ----------------------------
# Gemini API key (use Streamlit Secrets)
# Put this in Streamlit Cloud -> App -> Settings -> Secrets:
# GEMINI_API_KEY = "YOUR_KEY"
# ----------------------------
GOOGLE_API_KEY = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", ""))
if not GOOGLE_API_KEY:
    st.warning("Missing GEMINI_API_KEY. Add it in Streamlit Secrets to enable Gemini features.")

genai.configure(api_key=GOOGLE_API_KEY) if GOOGLE_API_KEY else None
vision_model = genai.GenerativeModel("gemini-2.0-flash") if GOOGLE_API_KEY else None

# ----------------------------
# Fake News model load
# ----------------------------
@st.cache_resource
def load_model():
    vectorizer = joblib.load("vectorizer.pkl")
    model = joblib.load("fake_news_model.pkl")
    return vectorizer, model

vectorizer, fake_news_model = load_model()

def preprocess_text(text: str) -> str:
    stemmer = nltk.stem.PorterStemmer()
    words = re.sub(r"[^a-zA-Z]", " ", text).lower().split()
    sw = set(nltk.corpus.stopwords.words("english"))
    words = [stemmer.stem(w) for w in words if w not in sw]
    return " ".join(words)

def extract_text_from_url(url: str) -> str:
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        return f"Failed to extract content from the URL. Error: {e}"

# ----------------------------
# Pages
# ----------------------------
if tabs == "🏠 Home":
    st.title("Fact Finder: One Stop solution for all your NEWS!")
    st.markdown(
        """
**Explore tools:**
- 📰 Fake News Detector: Verify news credibility.
- 🧠 News Research: Ask questions over up to 3 URLs (RAG).
"""
    )
    st.image("ai_image.jpg", use_column_width=True)

elif tabs == "📰 Fake News Detector":
    st.title("📰 Fake News Detector")
    option = st.selectbox("Select input type:", ["Text", "URL", "Image"])

    user_input = ""

    if option == "Text":
        user_input = st.text_area("Enter news text here...")

    elif option == "URL":
        url_input = st.text_input("Enter the URL of the news article:")
        if st.button("Extract Text"):
            user_input = extract_text_from_url(url_input)
            st.session_state["user_input"] = user_input
            st.write("Extracted Text:")
            st.write(user_input)

    elif option == "Image":
        uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            if st.button("Extract Text"):
                if not vision_model:
                    st.error("Gemini API key not configured. Add GEMINI_API_KEY in Streamlit Secrets.")
                else:
                    with st.spinner("Sending image to Gemini..."):
                        try:
                            response = vision_model.generate_content(
                                ["Extract all text from this image. Output only the text.", image]
                            )
                            user_input = (response.text or "").strip()
                            st.session_state["user_input"] = user_input
                            st.write("Extracted Text:")
                            st.write(user_input)
                        except Exception as e:
                            st.error(f"An error occurred: {str(e)}")

    if st.button("Check News"):
        text_for_pred = st.session_state.get("user_input") or user_input
        if not text_for_pred:
            st.warning("Please provide valid input.")
        else:
            processed_text = preprocess_text(text_for_pred)
            input_vector = vectorizer.transform([processed_text])
            prediction = fake_news_model.predict(input_vector)[0]
            result = "Fake News" if prediction == 1 else "Real News"
            if prediction == 1:
                st.error(f"🛑 Prediction: {result}")
            else:
                st.success(f"✅ Prediction: {result}")

elif tabs == "🧠 News Research":
    st.title("🧠 News Research")
    st.write("Enter at most 3 URLs to process:")

    urls = [st.text_input(f"URL {i+1}", key=f"url_input_{i+1}") for i in range(3)]

    if st.button("Process URLs") and any(urls):
        if not GOOGLE_API_KEY:
            st.error("Gemini API key not configured. Add GEMINI_API_KEY in Streamlit Secrets.")
        else:
            with st.spinner("Processing URLs..."):
                loader = UnstructuredURLLoader(urls=urls)
                data = loader.load()

                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
                docs = text_splitter.split_documents(data)

                embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                faiss_index = FAISS.from_documents(docs, embeddings)
                faiss_index.save_local("faiss_index")

            st.success("Processing Complete. Ask your questions!")

    st.write("⭐ Important: Ask questions relevant to the processed URLs.")
    query = st.text_input("Ask a question:", placeholder="What do you want to know?")

    if query:
        if not GOOGLE_API_KEY:
            st.error("Gemini API key not configured. Add GEMINI_API_KEY in Streamlit Secrets.")
        else:
            vectorindex = FAISS.load_local(
                "faiss_index",
                GoogleGenerativeAIEmbeddings(model="models/embedding-001"),
                allow_dangerous_deserialization=True,
            )

            chain = RetrievalQAWithSourcesChain.from_llm(
                llm=ChatGoogleGenerativeAI(
                    model="gemini-1.5-flash",
                    temperature=0,
                    max_tokens=None,
                    timeout=None,
                    max_retries=2,
                ),
                retriever=vectorindex.as_retriever(),
            )

            response = chain({"question": query}, return_only_outputs=True)

            with st.expander("Sources"):
                st.write(response.get("sources", ""))

            st.success(f"Answer: {response.get('answer', '')}")

elif tabs == "❓ FAQ":
    st.title("❓ Frequently Asked Questions")
    st.write("Here are some common questions about the application:")
    st.markdown(
        """
- **What is this app for?**
  - Tools for detecting fake news and researching news topics using AI.
- **How does the Fake News Detector work?**
  - A trained ML model classifies news as real or fake.
"""
    )

elif tabs == "👨💻 Developed By":
    st.title("👨💻 Developed By")
    st.header("Milan Das")

    st.subheader("Connect with me:")
    if st.button("LinkedIn Profile"):
        webbrowser.open("https://www.linkedin.com/in/milan-das-41a45b251")

    if st.button("GitHub Profile"):
        webbrowser.open("https://github.com/milan070das")

    st.write("Don't hesitate to share your feedback!")
    image = Image.open("qr.jpg")
    image = image.resize((200, 200))
    st.image(image)

    if st.button("Feedback Form"):
        webbrowser.open("https://forms.gle/XMk5oLhjoAgXoFPT9")
