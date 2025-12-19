import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
import os
import webbrowser
from datetime import datetime
import pyttsx3
import speech_recognition as sr
import google.generativeai as genai
import joblib
from PIL import Image
from newspaper import Article
import platform

# LangChain-related imports are optional. Guard them so the app still runs
# when LangChain or its community/extensions are not installed.
ChatGoogleGenerativeAI = None
UnstructuredURLLoader = None
RecursiveCharacterTextSplitter = None
GoogleGenerativeAIEmbeddings = None
FAISS = None
RetrievalQAWithSourcesChain = None
langchain_import_error = None
try:
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    from langchain_community.document_loaders import UnstructuredURLLoader
    # text_splitter location changed across versions; try import and tolerate failure
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    except Exception:
        RecursiveCharacterTextSplitter = None
    from langchain_community.vectorstores import FAISS
    from langchain.chains import RetrievalQAWithSourcesChain
except Exception as e:
    langchain_import_error = e

# Set up Streamlit page configuration at the very beginning
st.set_page_config(page_title="Fact Finder", layout="wide")

# --- SECRET HANDLING FOR DEPLOYMENT ---
# Access GEMINI_API_KEY from Streamlit Secrets
# In Streamlit Cloud, go to Settings -> Secrets and add:
# GEMINI_API_KEY = "your_api_key_here"
API_KEY = st.secrets.get("GEMINI_API_KEY")

# Sidebar Navigation with Icons
try:
    st.sidebar.image("logo.png", use_column_width=True)
except:
    st.sidebar.title("Fact Finder")

st.sidebar.title("🚀 Navigation")
tabs = st.sidebar.radio("Go to:", [
    "🏠 Home",
    "📰 Fake News Detector",
    "🧠 News Research",
    "🤖 Newzie",
    "❓ FAQ",
    "👨‍💻 Developed By"
])

nltk_data_dir = os.path.expanduser('~/nltk_data')
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)

nltk.data.path.append(nltk_data_dir)
with st.spinner("Downloading NLTK data..."):
    nltk.download('punkt', download_dir=nltk_data_dir)
    nltk.download('stopwords', download_dir=nltk_data_dir)

# Configure Gemini API key
if not API_KEY:
    st.warning("⚠️ GEMINI_API_KEY not found in Streamlit Secrets. Please configure it in the dashboard.")
else:
    try:
        genai.configure(api_key=API_KEY)
    except Exception:
        pass

# Initialize Gemini model
model = None
available_models = ['gemini-2.5-flash', 'gemini-2.0-flash', 'gemini-1.5-flash']
if API_KEY:
    for gemini_model in available_models:
        try:
            model = genai.GenerativeModel(gemini_model)
            break
        except Exception:
            continue

if model is None and API_KEY:
    st.sidebar.error("❌ No Gemini model available")

# Initialize text-to-speech
try:
    engine = pyttsx3.init()
except Exception:
    engine = None

def say(text):
    """Converts text to speech."""
    if engine:
        engine.say(text)
        engine.runAndWait()

def take_command():
    """Captures user's voice input and converts it to text."""
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("🎤 Listening...")
        audio = r.listen(source)
        try:
            return r.recognize_google(audio, language='en-in').lower()
        except:
            return ""

def ai_response(prompt):
    """Generate AI response using available Gemini model."""
    try:
        if model is not None and hasattr(model, 'generate_content'):
            resp = model.generate_content(prompt)
            return getattr(resp, 'text', str(resp)).strip()
        return "Gemini client not available or not configured."
    except Exception as e:
        return f"Error calling Gemini: {e}"

# Load the trained model and vectorizer (Fake News Detector)
@st.cache_resource
def load_model():
    vectorizer = joblib.load("vectorizer.pkl")
    model = joblib.load("fake_news_model.pkl")
    return vectorizer, model

try:
    vectorizer, fake_news_model = load_model()
except:
    st.sidebar.warning("⚠️ Fake news model files not found in root directory.")

def preprocess_text(text):
    stemmer = nltk.stem.PorterStemmer()
    text = re.sub(r'[^a-zA-Z]', ' ', text).lower().split()
    try:
        stop_words = nltk.corpus.stopwords.words('english')
    except:
        stop_words = []
    text = [stemmer.stem(word) for word in text if word not in stop_words]
    return ' '.join(text)

def extract_text_from_url(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        return f"Failed to extract content from the URL. Error: {e}"

# Home Page
if tabs == "🏠 Home":
    st.title("Fact Finder: One Stop solution for all your NEWS!")
    st.markdown("""
    **Explore AI-driven tools:**
    - 📰 **Fake News Detector**: Verify news credibility.
    - 🧠 **News Research**: Get a quick view of your news article.
    - 🤖 **Newzie**: Interact with an AI chatbot powered by Gemini.
    """)
    try:
        st.image("ai_image.jpg", use_column_width=True)
    except:
        pass

# Fake News Detector
elif tabs == "📰 Fake News Detector":
    st.title("📰 Fake News Detector")
    option = st.selectbox("Select input type:", ["Text", "URL","Image"])
    user_input = None
    
    if option == "Text":
        user_input = st.text_area("Enter news text here...", key="text_input")
        st.session_state['user_input'] = user_input
        
    elif option == "URL":
        url_input = st.text_input("Enter the URL of the news article:", key="url_input")
        if st.button("Extract Text", key="extract_url"):
            with st.spinner("Extracting text from URL..."):
                user_input = extract_text_from_url(url_input)
                st.text_area("Extracted Text:", user_input, height=200, key="extracted_url_text")
                st.session_state['user_input'] = user_input
                
    elif option == "Image":
        uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"], key="image_uploader")
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            if st.button("Extract Text from Image", key="extract_image"):
                if not API_KEY:
                    st.error("Please provide a Gemini API Key in secrets to use image features.")
                else:
                    with st.spinner("Sending image to Gemini..."):
                        try:
                            if model is not None:
                                response = model.generate_content([
                                    "Extract all text from this image exactly as it appears. Do not add any extra commentary or explanations. Return only the extracted text.",
                                    image,
                                ])
                                user_input = getattr(response, 'text', '').strip()
                                st.text_area("Extracted Text:", user_input, height=200, key="extracted_image_text")
                                st.session_state['user_input'] = user_input
                            else:
                                st.warning("Gemini model not available for image processing.")
                        except Exception as e:
                            st.error(f"Image processing failed: {str(e)}")
    
    if st.button("🔍 Check News Authenticity", key="check_news"):
        if 'user_input' in st.session_state and st.session_state['user_input']:
            user_input = st.session_state['user_input']
            if user_input and not user_input.startswith("Failed to extract content"):
                try:
                    with st.spinner("Analyzing news..."):
                        processed_text = preprocess_text(user_input)
                        input_vector = vectorizer.transform([processed_text])
                        prediction = fake_news_model.predict(input_vector)[0]
                        probability = fake_news_model.predict_proba(input_vector)[0]
                        
                        result = "🛑 FAKE NEWS" if prediction == 1 else "✅ REAL NEWS"
                        confidence = max(probability) * 100
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            if prediction == 1:
                                st.error(f"{result}")
                            else:
                                st.success(f"{result}")
                        
                        with col2:
                            st.info(f"Confidence: {confidence:.1f}%")
                except NameError:
                    st.error("Model not loaded. Check sidebar warnings.")
            else:
                st.warning("Please provide valid news content.")
        else:
            st.warning("Please provide input first (text, URL, or image).")

# News Research
elif tabs == "🧠 News Research":
    st.title("🧠 News Research")
    
    if ChatGoogleGenerativeAI is None or GoogleGenerativeAIEmbeddings is None:
        st.error("❌ LangChain components not available.")
    elif not API_KEY:
        st.error("❌ API Key required for Research features. Please add it to Streamlit Secrets.")
    else:
        st.info("Enter up to 3 URLs to process:")
        col1, col2, col3 = st.columns(3)
        urls = []
        with col1:
            url1 = st.text_input("URL 1", key="url1")
            if url1: urls.append(url1)
        with col2:
            url2 = st.text_input("URL 2", key="url2")
            if url2: urls.append(url2)
        with col3:
            url3 = st.text_input("URL 3", key="url3")
            if url3: urls.append(url3)
        
        if st.button("🚀 Process URLs") and urls:
            with st.spinner("Processing URLs and building knowledge base..."):
                try:
                    loader = UnstructuredURLLoader(urls=urls)
                    data = loader.load()
                    
                    if RecursiveCharacterTextSplitter:
                        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    else:
                        text_splitter = None
                    
                    docs = text_splitter.split_documents(data) if text_splitter else data
                    
                    embeddings = GoogleGenerativeAIEmbeddings(
                        model="models/embedding-001",
                        google_api_key=API_KEY
                    )
                    faiss_index = FAISS.from_documents(docs, embeddings)
                    faiss_index.save_local("faiss_index")
                    st.session_state['index_ready'] = True
                    st.success("✅ Knowledge base ready! Ask questions below.")
                except Exception as e:
                    st.error(f"Processing failed: {str(e)}")
        
        if 'index_ready' in st.session_state and st.session_state['index_ready']:
            st.write("⭐ Ask sensible questions relevant to the processed articles:")
            query = st.text_input('Your question:', placeholder="What are the key points discussed?")
            
            if query:
                with st.spinner("Searching knowledge base..."):
                    try:
                        embeddings = GoogleGenerativeAIEmbeddings(
                            model="models/embedding-001",
                            google_api_key=API_KEY
                        )
                        vectorindex = FAISS.load_local(
                            "faiss_index",
                            embeddings,
                            allow_dangerous_deserialization=True
                        )
                        
                        llm = ChatGoogleGenerativeAI(
                            model=available_models[0] if model else "gemini-1.5-flash",
                            temperature=0,
                            google_api_key=API_KEY
                        )
                        
                        chain = RetrievalQAWithSourcesChain.from_llm(
                            llm=llm,
                            retriever=vectorindex.as_retriever()
                        )
                        response = chain({"question": query}, return_only_outputs=True)
                        
                        st.success(f"**Answer:** {response['answer']}")
                        
                        with st.expander("📚 Sources"):
                            for source, content in response.get('sources', {}).items():
                                if content:
                                    st.write(f"**{source}:**")
                                    st.write(content[:1000] + "..." if len(content) > 1000 else content)
                    except Exception as e:
                        st.error(f"Query failed: {str(e)}")

# AI Assistant
elif tabs == "🤖 Newzie":
    st.title("🤖 Newzie - Your AI News Assistant")
    
    if not API_KEY:
        st.error("❌ API Key required for Newzie. Please add it to Streamlit Secrets.")
    else:
        current_model_name = next((m for m in available_models if model), "Unknown")
        st.info(f"🧠 Powered by: {current_model_name}")
        
        query = st.text_input("Ask me anything about news:", placeholder="What's trending? Analyze this headline...")
        st.info('⭐ Pro tip: Ask about current events, news analysis, or fact-checking!')
        
        if query:
            if "time" in query.lower() or "date" in query.lower():
                strfTime = datetime.now().strftime("%H:%M:%S")
                today = datetime.now()
                formatted_date = today.strftime('%B %d, %Y')
                st.success(f"**Time:** {strfTime} | **Date:** {formatted_date}")
            
            elif "weather" in query.lower():
                st.info("Weather info coming soon! Try asking about news topics for now.")
            
            else:
                with st.spinner("Newzie is thinking..."):
                    response = ai_response(query)
                    st.success(f"🤖 **Newzie:** {response}")

# FAQ Section
elif tabs == "❓ FAQ":
    st.title("❓ Frequently Asked Questions")
    faq_data = {
        "What models does this use?": "Gemini-2.5-flash (primary), with fallbacks to 2.0-flash and 1.5-flash.",
        "How does Fake News work?": "ML model trained on news datasets using TF-IDF + Logistic Regression.",
        "Voice input supported?": "Voice input depends on browser permissions and local system libraries.",
        "News Research requires?": "LangChain + up to 3 URLs for RAG-based Q&A.",
        "API Key needed?": "Yes, configure GEMINI_API_KEY in the Streamlit Cloud Secrets dashboard."
    }
    
    for question, answer in faq_data.items():
        with st.expander(question):
            st.write(answer)

# Developed By Section
elif tabs == "👨‍💻 Developed By":
    st.title("👨‍💻 Developed By")
    st.header("Milan Das")
    st.subheader("Connect with me 😄:")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💼 LinkedIn"):
            webbrowser.open('https://www.linkedin.com/in/milan-das-41a45b251')
    with col2:
        if st.button("🐙 GitHub"):
            webbrowser.open('https://github.com/milan070das')
    
    st.markdown("---")
    st.write("😄 **Share feedback to help us improve!**")
    
    col3, col4 = st.columns([1,2])
    with col3:
        try:
            image = Image.open("qr.jpg")
            st.image(image, caption="Scan for more info", width=200)
        except:
            st.write("📱 QR Code")
    
    with col4:
        if st.button("📝 Feedback Form", use_container_width=True):
            webbrowser.open('https://forms.gle/XMk5oLhjoAgXoFPT9')


