# 🧰 AI Multi-Tool Dashboard — Unified AI Application Suite

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red?style=flat-square&logo=streamlit)
![OpenAI](https://img.shields.io/badge/API-OpenAI-green?style=flat-square&logo=openai)
![Status](https://img.shields.io/badge/Stage-Complete-green?style=flat-square)

---

👤 **Author:** [Ashutosh](https://github.com/Ashutosh-AIBOT) · [LinkedIn](https://www.linkedin.com/in/ashutosh1975271/)
💼 **Portfolio:** [ashutosh-portfolio-kappa.vercel.app](https://ashutosh-portfolio-kappa.vercel.app/)

---

## 🧠 What This Does

A single Streamlit dashboard that brings together multiple
AI capabilities — text generation, image understanding,
voice interaction, and document analysis — in one unified
interface. No switching between apps.

---

## ✨ Tools Inside

| Tool | What It Does |
|------|-------------|
| 💬 Text Chat | LLM-powered conversation with memory |
| 🖼️ Image Analysis | Upload image → AI describes and analyzes |
| 🎙️ Voice Input | Speak to the assistant via microphone |
| 📄 Document Chat | Upload PDF → ask questions about it |
| 🔍 Web Search | AI-augmented web search and summarization |

---

## 🏗️ Architecture
```
Streamlit Multi-Page App
        ↓
Page Router → Select AI Tool
        ↓
Tool Modules
  ├── text_chat.py      → LLM chat with memory
  ├── image_analysis.py → Vision model integration
  ├── voice_input.py    → Whisper transcription
  ├── doc_chat.py       → RAG on uploaded PDFs
  └── web_search.py     → Search + summarize
        ↓
Shared API Layer
  → OpenAI / HuggingFace API calls
  → Centralized error handling
  → Session state management
```

---

## ⚡ Quick Start
```bash
git clone https://github.com/Ashutosh-AIBOT/ai-tools-multi-app-dashboard.git
cd ai-tools-multi-app-dashboard
pip install -r requirements.txt
cp .env.example .env   # add your API keys
streamlit run app.py
```

---

## 🛠️ Tech Stack

`Python` `Streamlit` `OpenAI API` `Whisper` `LangChain` `PyPDF2` `HuggingFace` `Git`

---

## 🌐 Links

| Resource | URL |
|----------|-----|
| 🐙 GitHub | [github.com/Ashutosh-AIBOT](https://github.com/Ashutosh-AIBOT) |
| 🔗 LinkedIn | [linkedin.com/in/ashutosh1975271](https://www.linkedin.com/in/ashutosh1975271/) |
| 💼 Portfolio | [ashutosh-portfolio-kappa.vercel.app](https://ashutosh-portfolio-kappa.vercel.app/) |

---

## 👤 Author
**Ashutosh** · B.Tech Electronics Engineering · Batch 2026
[GitHub](https://github.com/Ashutosh-AIBOT) · [LinkedIn](https://www.linkedin.com/in/ashutosh1975271/)

> *"One dashboard. Every AI tool. Zero switching."*
```
