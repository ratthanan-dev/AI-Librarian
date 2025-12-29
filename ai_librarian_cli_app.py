# ai_librarian_cli_app.py
# เวอร์ชัน Command Line สำหรับการศึกษาการทำงานของ AI Librarian V5.5 (Multi-Key Support)
# โค้ดนี้ถูกดัดแปลงให้ใช้แกนหลักเดียวกับ app_groq.py และ config.py

# --- ส่วนที่ 1: การนำเข้าเครื่องมือที่จำเป็น (Imports) ---
import os
import json
import asyncio
import platform

# เครื่องมือจาก LangChain, Google และ Groq
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# ### <<< CHANGE: เพิ่ม ChatGroq >>> ###
from langchain_groq import ChatGroq 
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

# ### <<< CHANGE: นำเข้า config และ prompts >>> ###
from config import api_keys 
from prompts import AI_LANGUAGE_ROUTER_PROMPT, PROMPTS

# เครื่องมือสำหรับระบบเสียง
import torch
import sounddevice as sd
import soundfile as sf
import edge_tts
from transformers import pipeline as hf_pipeline, AutoTokenizer, AutoModelForSpeechSeq2Seq, AutoFeatureExtractor

# --- ส่วนที่ 2: การตั้งค่าพื้นฐานของโปรแกรม (Configurations) ---

# หมายเหตุ: เราตัดส่วนตรวจสอบ os.getenv ออก เพราะ config.py จัดการให้แล้ว

# ตั้งค่าสำหรับระบบเสียง
WHISPER_MODEL_NAME = "openai/whisper-small"
VOICE_NAME_TH = "th-TH-NiwatNeural"
VOICE_NAME_EN = "en-US-GuyNeural"
AUDIO_FILE_INPUT = "user_input.wav"
AUDIO_FILE_OUTPUT = "ai_output.mp3"
SAMPLE_RATE = 16000
RECORD_DURATION = 7

# --- ส่วนที่ 3: โหลดโมเดลและสร้าง Chain (หัวใจของ AI) ---

db = None
llm_gemini_flash = None
llm_groq_router = None # ### <<< CHANGE: เพิ่มตัวแปร Global สำหรับ Groq >>> ###
language_router_chain = None
whisper_pipe = None

# เก็บ Chain แยกตามภาษา
chains = {
    "th": {},
    "en": {}
}

def load_models_and_chains():
    """
    ฟังก์ชันสำหรับโหลดทุกอย่างที่ AI ต้องใช้ในการทำงาน
    รองรับการสลับ Key อัตโนมัติผ่าน config.api_keys
    """
    global db, llm_gemini_flash, llm_groq_router, language_router_chain, chains, whisper_pipe
    print("\n--- 🚀 กำลังเตรียมระบบ AI Librarian (CLI Version with Multi-Keys)... ---")

    try:
        # 1. โหลด Embedding Model และ FAISS Index
        print("1. กำลังโหลด Embedding Model และ FAISS Index...")
        # ### <<< CHANGE: ใช้ api_keys.get_google_key() >>> ###
        embeddings_model = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004", 
            google_api_key=api_keys.get_google_key()
        )
        
        faiss_index_path = "faiss_index/book_index"
        if not os.path.exists(faiss_index_path):
            raise FileNotFoundError("ไม่พบโฟลเดอร์ faiss_index/book_index กรุณารัน data_preparation.py ก่อน")
        
        db = FAISS.load_local(faiss_index_path, embeddings_model, allow_dangerous_deserialization=True)
        print("   ✅ คลังความรู้พร้อมใช้งาน")

        # 2. โหลด LLM หลัก (Gemini 1.5 Flash)
        print("2. กำลังโหลด LLM หลัก (Gemini 1.5 Flash)...")
        # ### <<< CHANGE: ใช้ api_keys.get_google_key() (จะได้คีย์ถัดไป) >>> ###
        llm_gemini_flash = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            google_api_key=api_keys.get_google_key(), 
            temperature=0.7
        )
        print("   ✅ Main LLM (Gemini) พร้อมใช้งาน")

        # 3. โหลด Router LLM (Groq Llama3)
        print("3. กำลังโหลด Router LLM (Groq Llama3)...")
        llm_groq_router = ChatGroq(
            model="llama-3.1-8b-instant", 
            groq_api_key=api_keys.get_groq_key(),
            temperature=0
        )
        print("   ✅ Router LLM (Groq) พร้อมใช้งาน")

        # 4. สร้าง Chain การทำงานต่างๆ
        print("4. กำลังสร้าง Chain การทำงาน...")

        # Chain 4.1: ผู้เชี่ยวชาญด้านภาษา (Language Router) โดยใช้ Groq
        # ### <<< CHANGE: ใช้ llm_groq_router แทน Gemini >>> ###
        router_prompt = ChatPromptTemplate.from_template(AI_LANGUAGE_ROUTER_PROMPT)
        language_router_chain = router_prompt | llm_groq_router | JsonOutputParser()

        # Chain 4.2: วนลูปสร้าง Chain สำหรับแต่ละภาษา (ใช้ Gemini)
        for lang_code, lang_prompts in PROMPTS.items():
            print(f"   - กำลังสร้าง Chain สำหรับภาษา '{lang_code.upper()}'...")
            
            # บรรณารักษ์สำหรับค้นหาหนังสือ (RAG Chain)
            rag_prompt = ChatPromptTemplate.from_messages([
                ("system", lang_prompts["RAG_LIBRARIAN"]),
                ("user", "Context:\n{context}\n\nQuestion:\n{question}")
            ])
            chains[lang_code]["rag"] = rag_prompt | llm_gemini_flash | StrOutputParser()

            # บรรณารักษ์สำหรับคุยเล่น (General Chain)
            general_prompt = ChatPromptTemplate.from_messages([
                ("system", lang_prompts["GENERAL_LIBRARIAN"]),
                ("user", "{question}")
            ])
            chains[lang_code]["general"] = general_prompt | llm_gemini_flash | StrOutputParser()

        print("   ✅ Chain ทั้งหมดพร้อมใช้งาน")

        # 5. โหลดโมเดล Whisper
        print(f"5. กำลังโหลด Whisper Model ({WHISPER_MODEL_NAME})...")
        whisper_processor = AutoTokenizer.from_pretrained(WHISPER_MODEL_NAME)
        whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(WHISPER_MODEL_NAME)
        whisper_feature_extractor = AutoFeatureExtractor.from_pretrained(WHISPER_MODEL_NAME)
        whisper_pipe = hf_pipeline(
            "automatic-speech-recognition",
            model=whisper_model,
            tokenizer=whisper_processor,
            feature_extractor=whisper_feature_extractor,
            device=0 if torch.cuda.is_available() else -1,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        print("   ✅ Whisper Model พร้อมใช้งาน")
        print("--- ✨ ระบบ AI Librarian พร้อมแล้ว! ---")

    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาดร้ายแรงระหว่างการเตรียมระบบ: {e}")
        exit()

# --- ส่วนที่ 4: ฟังก์ชันเสริมต่างๆ (Helpers) ---
# (ส่วนนี้เหมือนเดิมทุกประการ)

def record_audio(duration=RECORD_DURATION, samplerate=SAMPLE_RATE, filename=AUDIO_FILE_INPUT):
    print(f"\n🎤 กำลังบันทึกเสียง {duration} วินาที... เริ่มพูดได้เลยครับ!")
    try:
        recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
        sd.wait()
        sf.write(filename, recording, samplerate)
        print(f"   บันทึกเสียงลงไฟล์ '{filename}' เรียบร้อย")
        return filename
    except Exception as e:
        print(f"❌ Error Recording Audio: {e}")
        return None

async def text_to_speech(text, lang, filename=AUDIO_FILE_OUTPUT):
    print("   🔊 กำลังสร้างเสียงพูด...")
    voice_name = VOICE_NAME_TH if lang == "th" else VOICE_NAME_EN
    try:
        communicate = edge_tts.Communicate(text, voice_name)
        await communicate.save(filename)
        print(f"   สร้างไฟล์เสียง '{filename}' เรียบร้อย")
        return filename
    except Exception as e:
        print(f"   ❌ ไม่สามารถสร้างไฟล์เสียงได้: {e}")
        return None

def play_audio(filename=AUDIO_FILE_OUTPUT):
    if not filename or not os.path.exists(filename):
        print("   ⚠️ ไม่พบไฟล์เสียงที่จะเล่น")
        return
    
    print("   ▶️  กำลังเล่นเสียงตอบกลับ...")
    
    try:
        # ตรวจสอบว่าเป็น WSL (Windows Subsystem for Linux) หรือไม่
        is_wsl = "microsoft" in platform.uname().release.lower()
        
        if is_wsl:
            # ถ้าเป็น WSL ให้ใช้ cmd.exe ของ Windows สั่งเปิดไฟล์ (Non-blocking)
            # วิธีนี้จะเปิด Media Player ของ Windows ขึ้นมาเล่นและไม่ทำให้ Python ค้าง
            os.system(f'cmd.exe /c start "" "{filename}"')
            
        elif os.name == 'posix':
            # เช็คว่ามี afplay (macOS) หรือไม่
            if os.system("which afplay > /dev/null") == 0:
                os.system(f"afplay '{filename}'")
            else:
                # ถ้าเป็น Linux ปกติ ให้ใช้ mpg123 (ต้องติดตั้งก่อน: sudo apt install mpg123)
                # เติม & > /dev/null 2>&1 เพื่อให้เล่นแบบ Background ไม่ต้องรอจบ
                os.system(f"mpg123 -q '{filename}' > /dev/null 2>&1")
                
        elif os.name == 'nt':
            # ถ้าเป็น Windows ปกติ
            os.system(f"start {filename}")
            
    except Exception as e:
        print(f"   เกิดข้อผิดพลาดในการเล่นเสียง: {e}")

async def speech_to_text(audio_file):
    if not audio_file: return ""
    print("   🧠 กำลังแปลงเสียงเป็นข้อความ...")
    try:
        result = whisper_pipe(audio_file)
        text = result["text"].strip()
        print(f"   คุณพูดว่า: '{text}'")
        return text
    except Exception as e:
        print(f"   เกิดข้อผิดพลาดในการแปลงเสียง: {e}")
        return ""


# --- ส่วนที่ 5: ฟังก์ชันการทำงานหลัก (Main Loop) ---

async def main():
    load_models_and_chains()
    print("\n=======================================================")
    print("      ยินดีต้อนรับสู่ AI Librarian (Command Line)")
    print("=======================================================")
    print("ผมคือ 'ไลท์' ผู้ช่วยบรรณารักษ์ AI ของคุณครับ")
    print("พิมพ์ 'exit' หรือ '0' เพื่อจบการสนทนาได้ทุกเมื่อ")

    while True:
        print("\nกรุณาเลือกโหมดที่ต้องการคุยกับผมครับ:")
        print("  1: 📚 โหมดแนะนำหนังสือ (RAG)")
        print("  2: 💬 โหมดคุยเล่นทั่วไป")
        mode_choice = input("เลือกโหมด (1/2): ").strip()

        if mode_choice.lower() in ['exit', '0']: break
        if mode_choice == '1': selected_mode = 'rag'
        elif mode_choice == '2': selected_mode = 'general'
        else:
            print("   ตัวเลือกไม่ถูกต้อง กรุณาลองใหม่ครับ"); continue

        print(f"\n--- 📚 เข้าสู่โหมด: {selected_mode.upper()} ---")
        print("\nเลือกวิธีการส่งคำถามครับ:")
        print("  1: ⌨️ พิมพ์ข้อความ")
        print("  2: 🎙️  พูดใส่ไมโครโฟน")
        input_choice = input("เลือกวิธี (1/2): ").strip()

        user_query = ""
        if input_choice == '1':
            user_query = input("\n✏️  พิมพ์คำถามของคุณ: ").strip()
        elif input_choice == '2':
            audio_file = record_audio()
            user_query = await speech_to_text(audio_file)
        else:
            print("   ตัวเลือกไม่ถูกต้อง"); continue

        if not user_query:
            print("   ไม่ได้รับคำถาม หรือแปลงเสียงไม่สำเร็จ กรุณาลองใหม่ครับ"); continue
        if user_query.lower() in ['exit', '0']: break

        print("\n" + "-"*20 + " กำลังประมวลผล " + "-"*20)
        try:
            # 1. ใช้ Language Router ตรวจจับภาษา (ใช้ Groq)
            print("1. กำลังตรวจสอบภาษาของคำถาม (via Groq)...")
            router_result = language_router_chain.invoke({"question": user_query})
            detected_lang = router_result.get("language", "th")
            print(f"   ภาษาที่ตรวจจับได้: {detected_lang.upper()}")

            # 2. เลือก Chain และดำเนินการตามโหมด
            ai_response = ""
            source_documents = []
            selected_chain = chains[detected_lang][selected_mode]

            if selected_mode == 'rag':
                print("2. กำลังค้นหาหนังสือที่เกี่ยวข้อง (RAG)...")
                # กรองด้วยภาษาที่ตรวจจับได้
                docs = db.similarity_search(user_query, k=4, filter={"language": detected_lang})
                context = "\n\n".join([doc.page_content for doc in docs])
                source_documents = docs
                
                print("3. กำลังสร้างคำตอบ (via Gemini)...")
                ai_response = selected_chain.invoke({"context": context, "question": user_query})
            else: # general
                print("2. กำลังเตรียมคำตอบ (via Gemini)...")
                ai_response = selected_chain.invoke({"question": user_query})

            # 3. แสดงผลและเล่นเสียง
            print("\n--- 👨‍💻 คำตอบจากไลท์ ---")
            print(ai_response)
            print("------------------------")

            audio_file = await text_to_speech(ai_response, detected_lang)
            play_audio(audio_file)

            if source_documents:
                print("\n--- 📚 ข้อมูลอ้างอิงจาก ---")
                unique_sources = set(doc.metadata.get('source', 'N/A') for doc in source_documents)
                for source in unique_sources:
                    print(f"  - {source}")

        except Exception as e:
            print(f"\n❌ เกิดข้อผิดพลาดที่ไม่คาดคิด: {e}")

        print("\n" + "="*50)

    print("\n👋 ขอบคุณที่ใช้บริการ AI Librarian ครับ แล้วพบกันใหม่!")

# --- ส่วนที่ 6: จุดเริ่มต้นการทำงานของโปรแกรม ---
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nโปรแกรมถูกยกเลิกโดยผู้ใช้ 👋")