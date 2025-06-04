# 🕵️‍♂️ FactFinder — AI-Powered News Verification and Research Assistant

FactFinder is an intelligent web application designed to help users detect fake news, research trusted news sources, get AI-driven assistance, and access frequently asked questions about misinformation. Built with modern AI models and intuitive interfaces, FactFinder empowers users to navigate the complex media landscape with confidence.

---

## ✨ Key Features

- 🔍 **Fake News Detection:** Analyze news articles to determine their authenticity using advanced NLP models.
- 📰 **News Research:** Search and summarize relevant news articles on any topic.
- 🤖 **AI Assistant:** Get AI-generated answers and insights related to news and misinformation.
- ❓ **FAQ Section:** Access a curated set of frequently asked questions about fake news, misinformation, and media literacy.
- 📂 **Multi-module System:** Organized interface separating key functions for ease of use.
- ⚡ **Responsive UI:** Clean and interactive design built with Streamlit.

---

## 🛠️ Tech Stack

| Layer          | Technology                    |
|----------------|------------------------------|
| Frontend & UI  | Streamlit                    |
| NLP Models     | Gemini, Transformers   |
| Data Handling  | Python, Pandas, Requests     |
| Backend Logic  | Python                      |
| Fake News Detection Models | Custom-trained NLP classifiers |

---

## 📁 Project Structure

```plaintext
FactFinder/
├── app.py                      # Main Streamlit app entry point
├── fake_news_detection.py      # Fake news detection model and logic
├── news_research.py            # News search and summarization utilities
├── ai_assistant.py             # GPT-powered AI assistant module
├── faq_data.json               # FAQ content data
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
