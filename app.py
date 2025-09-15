# app.py (Final Version - Fixed Event Loop Error)
# Main Backend Server for AI Librarian V5.4

import os
import json
import asyncio
import time
import threading
import re
import logging
from pathlib import Path
from hashlib import md5
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify, send_from_directory

# LangChain & Google Generative AI components
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

# Edge TTS for Text-to-Speech
import edge_tts

# --- 1. CONFIGURATION & INITIALIZATION ---

logging.basicConfig(level=logging.INFO)
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY_NAME") # <-- อาจจะต้องเปลี่ยน Key ให้ตรงกับ .env ของคุณ

BASE_DIR = Path(__file__).parent
INDEX_DIR = BASE_DIR / "faiss_index"
INDEX_PATH = INDEX_DIR / "book_index"
AUDIO_DIR = BASE_DIR / "static" / "audio"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, template_folder='templates', static_folder='static')
app.logger.setLevel(logging.INFO)

# --- Global Variables ---
db = None
embeddings_model = None
llm_gemini_flash = None
language_router_chain = None
chains = {"th": {}, "en": {}}

# --- 2. MODEL & VECTOR STORE LOADING ---

def load_models_and_db():
    global db, embeddings_model, llm_gemini_flash, language_router_chain, chains
    app.logger.info("--- 🚀 Initializing AI Librarian Backend... ---")
    try:
        app.logger.info("1. Loading Embedding Model (text-embedding-004)...")
        embeddings_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=GOOGLE_API_KEY)

        app.logger.info(f"2. Loading FAISS Vector Store from: {INDEX_PATH}")
        db = FAISS.load_local(str(INDEX_PATH), embeddings_model, allow_dangerous_deserialization=True)
        app.logger.info("   ✅ FAISS Index loaded successfully.")

        app.logger.info("3. Initializing LLM (Gemini 1.5 Flash)...")
        llm_gemini_flash = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY, temperature=0.7)
        app.logger.info("   ✅ LLM initialized.")

        app.logger.info("4. Building LangChain processing chains for all languages...")
        from prompts import AI_LANGUAGE_ROUTER_PROMPT, PROMPTS
         
        language_router_prompt = ChatPromptTemplate.from_template(AI_LANGUAGE_ROUTER_PROMPT)
        language_router_chain = language_router_prompt | llm_gemini_flash | JsonOutputParser()

        for lang_code, lang_prompts in PROMPTS.items():
            app.logger.info(f"   Building chains for language: '{lang_code}'...")
            
            rag_prompt = ChatPromptTemplate.from_messages([("system", lang_prompts["RAG_LIBRARIAN"]), ("user", "Context:\n{context}\n\nQuestion:\n{question}")])
            chains[lang_code]["rag_chat"] = rag_prompt | llm_gemini_flash | StrOutputParser()
            
            general_prompt = ChatPromptTemplate.from_messages([("system", lang_prompts["GENERAL_LIBRARIAN"]), ("user", "{question}")])
            chains[lang_code]["general_chat"] = general_prompt | llm_gemini_flash | StrOutputParser()

            voice_rag_prompt = ChatPromptTemplate.from_messages([("system", lang_prompts["VOICE_RAG_LIBRARIAN"]), ("user", "Context:\n{context}\n\nQuestion:\n{question}")])
            chains[lang_code]["rag_voice"] = voice_rag_prompt | llm_gemini_flash | StrOutputParser()

            voice_general_prompt = ChatPromptTemplate.from_messages([("system", lang_prompts["VOICE_GENERAL_LIBRARIAN"]), ("user", "{question}")])
            chains[lang_code]["general_voice"] = voice_general_prompt | llm_gemini_flash | StrOutputParser()
         
        app.logger.info("   ✅ All chains built successfully.")
        app.logger.info("--- ✨ AI Librarian is ready to serve! ---")

    except Exception as e:
        app.logger.error(f"❌ FATAL ERROR during initialization: {e}", exc_info=True)
        db = None

# --- 3. BACKGROUND TASK: AUDIO FILE CLEANUP ---
def cleanup_audio_files():
    while True:
        try:
            now = time.time()
            for filename in os.listdir(AUDIO_DIR):
                file_path = os.path.join(AUDIO_DIR, filename)
                if os.path.isfile(file_path) and (now - os.path.getmtime(file_path)) > 600:
                    os.remove(file_path)
        except Exception as e:
            app.logger.error(f"Error during audio cleanup: {e}")
        time.sleep(300)

# --- 4. FLASK API ENDPOINTS ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    if db is None: return jsonify({"error": "System is not ready, initialization failed."}), 503

    data = request.json
    query = data.get("query")
    mode = data.get("mode", "general")
    app.logger.info(f"[/chat] Received query: '{query}' in mode: '{mode}'")

    if not query: return jsonify({"error": "Query is missing"}), 400

    try:
        router_result = language_router_chain.invoke({"question": query})
        detected_language = router_result.get("language", "th")
        app.logger.info(f"[/chat] Language detected: '{detected_language}'")

        answer = ""
        if mode == "rag":
            app.logger.info(f"[/chat] RAG mode. Searching with filter '{detected_language}'...")
            docs = db.similarity_search(query, k=4, filter={"language": detected_language})
            context = "\n\n---\n\n".join([doc.page_content for doc in docs])
            
            selected_chain = chains[detected_language]["rag_chat"]
            answer = selected_chain.invoke({"context": context, "question": query})
        else:
            app.logger.info(f"[/chat] General mode. Generating chat response...")
            selected_chain = chains[detected_language]["general_chat"]
            answer = selected_chain.invoke({"question": query})
         
        app.logger.info(f"[/chat] Final answer generated successfully.")
        return jsonify({"answer": answer, "language": detected_language})

    except Exception as e:
        app.logger.error(f"Error in /chat endpoint: {e}", exc_info=True)
        return jsonify({"error": "An internal server error occurred."}), 500

# --- PWA Endpoints ---
@app.route('/manifest.json')
def serve_manifest():
    return send_from_directory('static', 'manifest.json')

@app.route('/sw.js')
def serve_sw():
    return send_from_directory('static', 'sw.js', mimetype='application/javascript')

@app.route('/voice_mode_ask', methods=['POST'])
async def voice_mode_ask():
    if db is None: return jsonify({"error": "System is not ready, initialization failed."}), 503
     
    data = request.json
    query = data.get("query")
    mode = data.get("mode", "general")
    app.logger.info(f"[/voice] Received query: '{query}', mode: {mode}")

    if not query: return jsonify({"error": "Query is missing"}), 400

    try:
        # ### <<< CHANGE START: REVERT TO SYNCHRONOUS LANGCHAIN CALLS >>> ###
        # ใช้ .invoke() แบบปกติแทน .ainvoke() เพื่อหลีกเลี่ยงปัญหา Event Loop
        router_result = language_router_chain.invoke({"question": query})
        # ### <<< CHANGE END >>> ###
        detected_language = router_result.get("language", "th")
        app.logger.info(f"[/voice] Language detected: '{detected_language}'")

        text_answer = ""
        if mode == "rag":
            app.logger.info(f"[/voice] RAG mode. Searching with filter '{detected_language}'...")
            docs = db.similarity_search(query, k=3, filter={"language": detected_language})
            context = "\n\n---\n\n".join([doc.page_content for doc in docs])

            selected_chain = chains[detected_language]["rag_voice"]
            # ### <<< CHANGE START: REVERT TO SYNCHRONOUS LANGCHAIN CALLS >>> ###
            text_answer = selected_chain.invoke({"context": context, "question": query})
            # ### <<< CHANGE END >>> ###
        else:
            app.logger.info(f"[/voice] General mode. Generating voice response...")
            selected_chain = chains[detected_language]["general_voice"]
            # ### <<< CHANGE START: REVERT TO SYNCHRONOUS LANGCHAIN CALLS >>> ###
            text_answer = selected_chain.invoke({"question": query})
            # ### <<< CHANGE END >>> ###

        app.logger.info(f"[/voice] Generated text: '{text_answer[:70]}...'")

        voice = "th-TH-NiwatNeural" if detected_language == "th" else "en-US-GuyNeural"
        filename = f"{md5(text_answer.encode()).hexdigest()}.mp3"
        filepath = AUDIO_DIR / filename
        audio_url = f"/static/audio/{filename}"
        
        try:
            if not filepath.exists():
                app.logger.info(f"[/voice] Attempting to generate TTS with voice: {voice}")
                # การเรียกใช้ edge-tts ยังคงเป็น await เพราะไลบรารีนี้เป็น async-native
                communicate = edge_tts.Communicate(text_answer, voice)
                await communicate.save(str(filepath))
                app.logger.info(f"[/voice] Successfully generated audio file: {filename}")
        except Exception as tts_error:
            app.logger.error(f"❌ FAILED to generate TTS audio for voice '{voice}'. Error: {tts_error}", exc_info=True)
            audio_url = ""
        
        return jsonify({"answer": text_answer, "audio_url": audio_url})

    except Exception as e:
        app.logger.error(f"Error in /voice_mode_ask endpoint: {e}", exc_info=True)
        return jsonify({"error": "An internal server error occurred."}), 500

@app.route('/tts', methods=['POST'])
async def text_to_speech():
    data = request.json
    text = data.get("text")
    language = data.get("language", "th")
    if not text: return jsonify({"error": "Text is missing"}), 400
    
    voice = "th-TH-NiwatNeural" if language == "th" else "en-US-GuyNeural"
    filename = f"{md5(text.encode()).hexdigest()}.mp3"
    filepath = AUDIO_DIR / filename
    audio_url = f"/static/audio/{filename}"

    try:
        if not filepath.exists():
            app.logger.info(f"[/tts] Attempting to generate TTS with voice: {voice}")
            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(str(filepath))
            app.logger.info(f"[/tts] Successfully generated audio file: {filename}")
        return jsonify({"audio_url": audio_url})
    except Exception as tts_error:
        app.logger.error(f"❌ FAILED to generate TTS audio for voice '{voice}'. Error: {tts_error}", exc_info=True)
        return jsonify({"audio_url": ""})

# --- 5. STARTUP ---

if __name__ == '__main__':
    load_models_and_db()
    cleanup_thread = threading.Thread(target=cleanup_audio_files, daemon=True)
    cleanup_thread.start()
    app.run(host='0.0.0.0', port=5000, debug=True)

