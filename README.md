# AI Librarian: Voice-Interactive Book Recommendation System

## 📖 Project Overview

AI Librarian is an innovative full-stack web application designed to revolutionize how library users discover and interact with books. The system features an AI librarian assistant named **"Lite" (ไลท์)** that provides intelligent, voice-interactive support for book recommendations and general conversations.

Moving beyond traditional keyword searches, AI Librarian leverages advanced AI techniques including **RAG (Retrieval-Augmented Generation)** architecture to provide accurate, context-aware book recommendations based on natural language queries, significantly reducing AI "hallucinations" and ensuring factual accuracy.

This project was proudly developed as a final project for the Computer Technology department.

---

## ✨ Key Features

-   **� Dual-Mode Conversation:** Two conversation modes for different needs:
    -   **Book Mode:** Uses RAG architecture to provide accurate book recommendations from the knowledge base.
    -   **Chat Mode:** General conversation mode for topics beyond books using the LLM's general knowledge.
-   **�🗣️ Voice & Text Interaction:** Seamlessly interact with the AI using both voice commands (via Web Speech API) and text input.
-   **🧠 RAG Architecture:** Employs a robust Retrieval-Augmented Generation (RAG) architecture to provide answers grounded in a curated knowledge base, significantly reducing "hallucinations" and ensuring factual accuracy.
-   **🔍 Semantic Search:** Discovers books based on the *meaning* and *context* of the user's query, rather than just matching keywords, powered by a highly efficient FAISS vector store.
-   **🌐 Bilingual Support:** Designed to understand and respond effectively in both Thai and English, featuring intelligent language detection and appropriate Text-to-Speech (TTS) voice selection.
-   **🚀 Hybrid AI Model Integration:** Strategically utilizes the Groq API for lightning-fast language detection and initial understanding, combined with the Google Gemini API for high-quality, nuanced response generation.
-   **🎙️ Continuous Voice Mode:** Offers a hands-free, continuous conversational experience—after the AI finishes speaking, the system automatically returns to listening mode for seamless back-and-forth dialogue.
-   **🎨 Modern & Responsive UI:** Presents a clean, user-friendly interface with toggles for light/dark themes, language preferences, audio playback controls, and more.

---

## � User Satisfaction Results

Based on user testing with 10 participants (primarily students and educators), the AI Librarian system achieved an **overall satisfaction score of 4.54 out of 5** (Excellent rating).

| Category | Score | Rating |
|----------|-------|--------|
| **UI Clarity & Ease of Use** | 4.70 | Excellent |
| **Book Recommendations Quality** | 4.70 | Excellent |
| **TTS Voice Quality** | 4.60 | Excellent |
| **Hands-free Experience** | 4.60 | Excellent |
| **Overall Satisfaction** | 4.60 | Excellent |
| **Likelihood to Use Again** | 4.60 | Excellent |
| **Response Speed** | 4.30 | Very Good |
| **STT Accuracy** | 4.30 | Very Good |

---

## �🛠️ Technology Stack

-   **Backend:** Python, Flask, LangChain, Gunicorn (for production-like environment if desired)
-   **AI & Machine Learning:** Google Gemini API, Groq API, FAISS (Vector Database), `sentence-transformers` (for embeddings), `edge-tts` (for Text-to-Speech)
-   **Frontend:** HTML5, CSS3, Vanilla JavaScript, Web Speech API (for Speech-to-Text and browser TTS), `marked.js` (for Markdown rendering)
-   **Database:** JSONL (for raw book data storage), FAISS (for high-performance vector indexing)
-   **Development & Deployment:** Python `venv`, Git, GitHub, VS Code

---

## 🚀 Getting Started: Setup and Installation

Follow these detailed steps to set up and run the AI Librarian project on your local machine.

### Prerequisites

Before you begin, ensure you have the following installed:

1.  **Python 3.12+**: Download and install from [python.org](https://www.python.org/downloads/). Ensure you check "Add Python to PATH" during installation on Windows.
2.  **Git**: Download and install from [git-scm.com](https://git-scm.com/downloads).
3.  **Command Line Access**: Familiarity with using your system's terminal (Command Prompt/PowerShell on Windows, Terminal on macOS/Linux).
4.  **(Recommended for Windows Users)** **WSL 2 (Windows Subsystem for Linux)** with an Ubuntu distribution for optimal compatibility and performance.
    * **Installation Guide for WSL 2 & Ubuntu:** Open PowerShell as Administrator and run `wsl --install`. Follow on-screen prompts and restart your computer. Install Ubuntu from the Microsoft Store if not automatically added. Update Ubuntu: `sudo apt update && sudo apt upgrade -y`.

### 1. Clone the Repository

Open your terminal (or WSL Ubuntu terminal) and clone the project repository:

```bash
git clone [https://github.com/ratthanan-dev/AI-Librarian.git]
cd app_ai_librarian
