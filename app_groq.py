# app_groq.py
import os
import json
import asyncio
import time
import threading
import logging
import re
from pathlib import Path
from hashlib import md5

# Flask & Web Components
from flask import Flask, render_template, request, jsonify, send_from_directory

# LangChain Components
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

# Edge TTS (primary) & gTTS (fallback)
import edge_tts
from gtts import gTTS

# ### <<< CHANGE: นำเข้า config ใหม่แทนการโหลด .env เอง >>> ###
from config import api_keys 

# --- 1. CONFIGURATION & INITIALIZATION ---

logging.basicConfig(level=logging.INFO)

# === HELPER FUNCTION: Remove Emoji from text for TTS ===
def remove_emoji(text):
    """Remove emoji characters from text to prevent TTS errors."""
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002702-\U000027B0"  # dingbats
        "\U000024C2-\U0001F251" 
        "]+", 
        flags=re.UNICODE
    )
    return emoji_pattern.sub('', text)

# หมายเหตุ: เราไม่ต้องตรวจสอบ Key ที่นี่แล้ว เพราะ Class KeyManager ใน config.py 
# จะทำหน้าที่ตรวจสอบและ Raise ValueError ให้ถ้าไม่มีคีย์

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
llm_groq_router = None
language_router_chain = None
chains = {"th": {}, "en": {}}

# --- 2. MODEL & VECTOR STORE LOADING ---

def load_models_and_db():
    global db, embeddings_model, llm_gemini_flash, llm_groq_router, language_router_chain, chains
    app.logger.info("--- 🚀 Initializing AI Librarian Backend (with Key Manager)... ---")
    try:
        # ### <<< CHANGE: เรียกใช้คีย์จาก api_keys >>> ###
        
        app.logger.info("1. Loading Embedding Model...")
        # เรียกใช้ get_google_key() ครั้งที่ 1 สำหรับ Embeddings
        embeddings_model = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004", 
            google_api_key=api_keys.get_google_key() 
        )

        app.logger.info(f"2. Loading FAISS Vector Store from: {INDEX_PATH}")
        db = FAISS.load_local(str(INDEX_PATH), embeddings_model, allow_dangerous_deserialization=True)
        app.logger.info("   ✅ FAISS Index loaded successfully.")

        app.logger.info("3. Initializing Main LLM (Google Gemini 2.5 Flash)...")
        # เรียกใช้ get_google_key() ครั้งที่ 2 สำหรับ Main LLM (จะได้คีย์ตัวถัดไปใน List โดยอัตโนมัติ)
        llm_gemini_flash = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            google_api_key=api_keys.get_google_key(), 
            temperature=0.7
        )
        app.logger.info("   ✅ Main LLM initialized.")
        
        app.logger.info("4. Initializing Router LLM (Groq Llama3-8B)...")
        llm_groq_router = ChatGroq(
            model="llama-3.1-8b-instant",
            groq_api_key=api_keys.get_groq_key(),
            temperature=0
        )
        app.logger.info("   ✅ Router LLM initialized.")

        app.logger.info("5. Building LangChain processing chains...")
        from prompts import AI_LANGUAGE_ROUTER_PROMPT, PROMPTS
         
        language_router_prompt = ChatPromptTemplate.from_template(AI_LANGUAGE_ROUTER_PROMPT)
        language_router_chain = language_router_prompt | llm_groq_router | JsonOutputParser()
        
        for lang_code, lang_prompts in PROMPTS.items():
            app.logger.info(f"   - Building Main Chains for language: '{lang_code}'...")
            
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
        # ใช้ Groq Router (คีย์ถูกใส่ไว้แล้วตอน init)
        router_result = language_router_chain.invoke({"question": query})
        detected_language = router_result.get("language", "th")
        app.logger.info(f"[/chat] Language detected: '{detected_language}'")

        answer = ""
        if mode == "rag":
            docs = db.similarity_search(query, k=4, filter={"language": detected_language})
            context = "\n\n---\n\n".join([doc.page_content for doc in docs])
            selected_chain = chains[detected_language]["rag_chat"]
            answer = selected_chain.invoke({"context": context, "question": query})
        else:
            selected_chain = chains[detected_language]["general_chat"]
            answer = selected_chain.invoke({"question": query})
         
        return jsonify({"answer": answer, "language": detected_language})

    except Exception as e:
        app.logger.error(f"Error in /chat endpoint: {e}", exc_info=True)
        return jsonify({"error": "An internal server error occurred."}), 500

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
    
    if not query: return jsonify({"error": "Query is missing"}), 400

    # === DEBUG: Voice Input ===
    print("="*60)
    print(f"🎤 [VOICE_MODE_ASK] INPUT RECEIVED")
    print(f"   📝 Query: {query}")
    print(f"   🎯 Mode: {mode}")
    print("="*60)

    try:
        router_result = language_router_chain.invoke({"question": query})
        detected_language = router_result.get("language", "th")
        
        # === DEBUG: Language Detection ===
        print(f"   🌐 Detected Language: {detected_language}")
        
        text_answer = ""
        if mode == "rag":
            docs = db.similarity_search(query, k=3, filter={"language": detected_language})
            context = "\n\n---\n\n".join([doc.page_content for doc in docs])
            
            # === DEBUG: RAG Context ===
            print(f"   📚 RAG Mode - Found {len(docs)} documents")
            
            selected_chain = chains[detected_language]["rag_voice"]
            text_answer = selected_chain.invoke({"context": context, "question": query})
        else:
            # === DEBUG: General Mode ===
            print(f"   💬 General Mode")
            
            selected_chain = chains[detected_language]["general_voice"]
            text_answer = selected_chain.invoke({"question": query})

        # === DEBUG: LLM Response ===
        print(f"   🤖 LLM Response (first 100 chars): {text_answer[:100]}...")
        print(f"   📏 Response Length: {len(text_answer)} chars")

        voice = "th-TH-NiwatNeural" if detected_language == "th" else "en-US-GuyNeural"
        filename = f"{md5(text_answer.encode()).hexdigest()}.mp3"
        filepath = AUDIO_DIR / filename
        audio_url = f"/static/audio/{filename}"
        
        # === DEBUG: TTS Generation ===
        print(f"   🔊 TTS Voice: {voice}")
        print(f"   📁 Audio File: {filename}")
        print(f"   📍 Audio Path: {filepath}")
        
        try:
            if not filepath.exists():
                print(f"   ⏳ Generating new audio file...")
                # Remove emoji before TTS
                clean_text = remove_emoji(text_answer)
                print(f"   🧹 Clean text (no emoji): {clean_text[:50]}...")
                
                # Try Edge TTS first, fallback to gTTS
                try:
                    communicate = edge_tts.Communicate(clean_text, voice)
                    await communicate.save(str(filepath))
                    print(f"   ✅ Audio file generated successfully (Edge TTS)!")
                except Exception as edge_error:
                    print(f"   ⚠️ Edge TTS failed: {edge_error}")
                    print(f"   🔄 Falling back to gTTS...")
                    # Use gTTS as fallback
                    tts_lang = 'th' if detected_language == 'th' else 'en'
                    tts = gTTS(text=clean_text, lang=tts_lang)
                    tts.save(str(filepath))
                    print(f"   ✅ Audio file generated successfully (gTTS)!")
            else:
                print(f"   ♻️ Using cached audio file")
        except Exception as tts_error:
            print(f"   ❌ TTS ERROR: {tts_error}")
            app.logger.error(f"TTS Error: {tts_error}")
            audio_url = ""
        
        # === DEBUG: Final Output ===
        print("="*60)
        print(f"🔈 [VOICE_MODE_ASK] OUTPUT")
        print(f"   📝 Answer: {text_answer[:100]}...")
        print(f"   🔗 Audio URL: {audio_url}")
        print("="*60)
        
        return jsonify({"answer": text_answer, "audio_url": audio_url})

    except Exception as e:
        print(f"   ❌ EXCEPTION: {e}")
        app.logger.error(f"Error in /voice_mode_ask endpoint: {e}", exc_info=True)
        return jsonify({"error": "An internal server error occurred."}), 500

@app.route('/tts', methods=['POST'])
async def text_to_speech():
    data = request.json
    text = data.get("text")
    language = data.get("language", "th")
    
    # === DEBUG: TTS Input ===
    print("="*60)
    print(f"🔊 [TTS] INPUT RECEIVED")
    print(f"   📝 Text (first 100 chars): {text[:100] if text else 'None'}...")
    print(f"   🌐 Language: {language}")
    print("="*60)
    
    if not text: return jsonify({"error": "Text is missing"}), 400
    
    voice = "th-TH-NiwatNeural" if language == "th" else "en-US-GuyNeural"
    filename = f"{md5(text.encode()).hexdigest()}.mp3"
    filepath = AUDIO_DIR / filename
    audio_url = f"/static/audio/{filename}"
    
    # === DEBUG: TTS Generation ===
    print(f"   🔊 Voice: {voice}")
    print(f"   📁 Filename: {filename}")
    print(f"   📍 Filepath: {filepath}")

    try:
        if not filepath.exists():
            print(f"   ⏳ Generating new audio file...")
            # Remove emoji before TTS
            clean_text = remove_emoji(text)
            print(f"   🧹 Clean text (no emoji): {clean_text[:50]}...")
            
            # Try Edge TTS first, fallback to gTTS
            try:
                communicate = edge_tts.Communicate(clean_text, voice)
                await communicate.save(str(filepath))
                print(f"   ✅ Audio file generated successfully (Edge TTS)!")
            except Exception as edge_error:
                print(f"   ⚠️ Edge TTS failed: {edge_error}")
                print(f"   🔄 Falling back to gTTS...")
                # Use gTTS as fallback
                tts_lang = 'th' if language == 'th' else 'en'
                tts = gTTS(text=clean_text, lang=tts_lang)
                tts.save(str(filepath))
                print(f"   ✅ Audio file generated successfully (gTTS)!")
        else:
            print(f"   ♻️ Using cached audio file")
        
        # === DEBUG: TTS Output ===
        print(f"   🔗 Audio URL: {audio_url}")
        print("="*60)
        
        return jsonify({"audio_url": audio_url})
    except Exception as tts_error:
        print(f"   ❌ TTS ERROR: {tts_error}")
        return jsonify({"audio_url": ""})

# --- 5. STARTUP ---

if __name__ == '__main__':
    load_models_and_db()
    cleanup_thread = threading.Thread(target=cleanup_audio_files, daemon=True)
    cleanup_thread.start()
    app.run(host='0.0.0.0', port=5000, debug=True)