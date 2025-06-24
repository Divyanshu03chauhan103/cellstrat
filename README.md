# 🧠 MediMind – AI-Powered Healthcare Assistant

**MediMind** is a modular, AI-driven healthcare platform built around three specialized agents designed to extract insights, support operational decision-making, and answer medical questions in real-time using LLMs and Retrieval-Augmented Generation (RAG).

> 🎥 [Watch Demo Video](https://drive.google.com/file/d/1nOyM2Kefq_uoqo9UDfer5Dj9hfFCgL-6/view?usp=sharing)

---

## 🧩 System Overview

### 🤖 Core Agents

#### 1. 🧠 DiagnosticsAgent
- Processes **medical documents (PDFs)**.
- Extracts clinical information using NLP and LLM-based summarization.
- Converts unstructured text into usable insights for operational analysis.

#### 2. 🏥 HospitalOperationsAgent
- Analyzes **hospital operational data (Excel files)**.
- Uses **Python REPL Tool** to generate executable code for calculations and analysis.
- Returns actionable insights based on interpreted output.

#### 3. 💬 AskAI
- Answers open-ended queries.
- Combines web search, web scraping, and RAG for contextual, up-to-date responses.
- Queries are processed using LLMs, and results are pulled from vector databases and internet sources.

---

## 🔄 Architecture & Flow

- Each agent operates independently and is triggered based on the query type.
- **FastAPI** serves as the backend orchestrator.
- **React + TypeScript** powers the dynamic frontend interface.
- **MongoDB** is used for data storage, including logs, responses, and optional user metadata.

📌 Flow Highlights:
- PDF data → `DiagnosticsAgent` → Clinical insights  
- Excel data → `HospitalOperationsAgent` → Operational recommendations  
- Natural language queries → `AskAI` → Real-time contextual answers


---

## 🧪 Core Technologies & Concepts

- ✅ **Retrieval-Augmented Generation (RAG)** – Document-based contextual insights
- ✅ **Multi-Agent System** – Distributed AI workflows (LangChain, CrewAI)
- ✅ **Vector Databases** – FAISS / ChromaDB / Weaviate
- ✅ **Web Scraping** – Real-time structured data extraction
- ✅ **Natural Language Processing (NLP)** – spaCy, NLTK
- ✅ **Task Queue & Orchestration** – Celery + Redis

---

## 🧱 Stack Breakdown

### ⚙️ Frameworks & Libraries

| Category        | Tools/Frameworks |
|----------------|------------------|
| Agents & LLMs  | LangChain, CrewAI, Hugging Face Transformers |
| Backend        | FastAPI |
| Frontend       | React.js, TypeScript |
| NLP            | spaCy, NLTK |
| Data Handling  | Pandas, OpenPyXL |
| Deployment     | Docker |
| Orchestration  | Celery, Redis |

---

## 🌐 APIs & Integrations

- 🔍 **SerperDevTool** – Google Search API for real-time information
- 📚 **LangChain RAG** – Knowledge base-backed generation
- 🦆 **DuckDuckGo Search API** – Privacy-first medical info retrieval
- 📄 **PDFMiner / PyMuPDF** – PDF document parsing
- 🧠 **LangSmith** – Agent performance monitoring and debugging
- 📊 **OpenTelemetry** – Agent trace logging
- 🔐 **Firebase (Optional)** – User authentication and session control
- 🌐 **WebSockets** – Real-time UI communication





