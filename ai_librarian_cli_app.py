# ai_librarian_cli_app.py
# เวอร์ชัน Command Line สำหรับการศึกษาการทำงานของ AI Librarian V5.4
# โค้ดนี้ถูกดัดแปลงให้ใช้แกนหลัก (Core Logic) เดียวกับ app.py เวอร์ชันล่าสุด
# เพื่อให้นักเรียนสามารถทำความเข้าใจการทำงานของ Backend ได้โดยไม่ต้องรันเว็บเซิร์ฟเวอร์

# --- ส่วนที่ 1: การนำเข้าเครื่องมือที่จำเป็น (Imports) ---
import os
import json
import asyncio

# เครื่องมือจาก LangChain และ Google
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser # ### <<< CHANGE: เพิ่ม JsonOutputParser >>> ###

# ### <<< CHANGE: นำเข้าโครงสร้างพรอมต์ใหม่ >>> ###
# นำเข้า Dictionary `PROMPTS` ที่มีพรอมต์ทั้งหมด แยกตามภาษา
from prompts import AI_LANGUAGE_ROUTER_PROMPT, PROMPTS

# เครื่องมือสำหรับระบบเสียง
import torch
import sounddevice as sd
import soundfile as sf
import edge_tts
from transformers import pipeline as hf_pipeline, AutoTokenizer, AutoModelForSpeechSeq2Seq, AutoFeatureExtractor

# --- ส่วนที่ 2: การตั้งค่าพื้นฐานของโปรแกรม (Configurations) ---

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY_NNG") # ตรวจสอบให้แน่ใจว่า Key นี้ตรงกับไฟล์ .env ของคุณ

if not GOOGLE_API_KEY:
    print("❌ Error: ไม่พบ GOOGLE_API_KEY_NNG ในไฟล์ .env")
    exit()

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
language_router_chain = None
whisper_pipe = None
# ### <<< CHANGE: เปลี่ยนโครงสร้างการเก็บ Chain ใหม่ >>> ###
# เราจะใช้ Dictionary เพื่อเก็บ Chain ที่สร้างไว้ล่วงหน้าสำหรับแต่ละภาษา
chains = {
    "th": {},
    "en": {}
}

def load_models_and_chains():
    """
    ฟังก์ชันสำหรับโหลดทุกอย่างที่ AI ต้องใช้ในการทำงาน
    เหมือนกับส่วน Initialization ใน app.py เวอร์ชันล่าสุด
    """
    global db, llm_gemini_flash, language_router_chain, chains, whisper_pipe
    print("\n--- 🚀 กำลังเตรียมระบบ AI Librarian (เวอร์ชัน Command Line)... ---")

    try:
        # 1. โหลด Embedding Model และ FAISS Index (เหมือนเดิม)
        print("1. กำลังโหลด Embedding Model และ FAISS Index...")
        embeddings_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=GOOGLE_API_KEY)
        faiss_index_path = "faiss_index/book_index"
        if not os.path.exists(faiss_index_path):
            raise FileNotFoundError("ไม่พบโฟลเดอร์ faiss_index/book_index กรุณารัน data_preparation.py ก่อน")
        db = FAISS.load_local(faiss_index_path, embeddings_model, allow_dangerous_deserialization=True)
        print("   ✅ คลังความรู้พร้อมใช้งาน")

        # 2. โหลด LLM หลัก (Gemini 1.5 Flash)
        print("2. กำลังโหลด LLM หลัก (Gemini 1.5 Flash)...")
        llm_gemini_flash = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY, temperature=0.7)
        print("   ✅ LLM พร้อมใช้งาน")

        # 3. สร้าง Chain การทำงานต่างๆ
        print("3. กำลังสร้าง Chain การทำงาน...")

        # ### <<< CHANGE: ปรับปรุง Language Router >>> ###
        # Chain 3.1: ผู้เชี่ยวชาญด้านภาษา (Language Router)
        # เราใช้ JsonOutputParser เพื่อให้ได้ผลลัพธ์เป็น Dictionary ที่สะอาดโดยอัตโนมัติ
        router_prompt = ChatPromptTemplate.from_template(AI_LANGUAGE_ROUTER_PROMPT)
        language_router_chain = router_prompt | llm_gemini_flash | JsonOutputParser()

        # ### <<< CHANGE: สร้าง Chain สำหรับทุกภาษาแบบไดนามิก >>> ###
        # Chain 3.2: วนลูปสร้าง Chain สำหรับแต่ละภาษาที่กำหนดไว้ใน prompts.py
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

        # 4. โหลดโมเดล Whisper (เหมือนเดิม)
        print(f"4. กำลังโหลด Whisper Model ({WHISPER_MODEL_NAME})...")
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
# ฟังก์ชัน record_audio, text_to_speech, play_audio, speech_to_text คงเดิม ไม่มีการเปลี่ยนแปลง
def record_audio(duration=RECORD_DURATION, samplerate=SAMPLE_RATE, filename=AUDIO_FILE_INPUT):
    print(f"\n🎤 กำลังบันทึกเสียง {duration} วินาที... เริ่มพูดได้เลยครับ!")
    recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
    sd.wait()
    sf.write(filename, recording, samplerate)
    print(f"   บันทึกเสียงลงไฟล์ '{filename}' เรียบร้อย")
    return filename

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
        if os.name == 'posix':
            os.system(f"afplay '{filename}'" if os.system("which afplay > /dev/null") == 0 else f"mpg123 -q '{filename}'")
        elif os.name == 'nt':
            os.system(f"start {filename}")
    except Exception as e:
        print(f"   เกิดข้อผิดพลาดในการเล่นเสียง: {e}")

async def speech_to_text(audio_file):
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
            print("   ไม่ได้รับคำถาม กรุณาลองใหม่ครับ"); continue
        if user_query.lower() in ['exit', '0']: break

        print("\n" + "-"*20 + " กำลังประมวลผล " + "-"*20)
        try:
            # ### <<< CHANGE: ปรับปรุงกระบวนการทั้งหมดให้เหมือน app.py >>> ###
            # 1. ใช้ Language Router ตรวจจับภาษา
            print("1. กำลังตรวจสอบภาษาของคำถาม...")
            router_result = language_router_chain.invoke({"question": user_query})
            detected_lang = router_result.get("language", "th")
            print(f"   ภาษาที่ตรวจจับได้: {detected_lang.upper()}")

            # 2. เลือก Chain และดำเนินการตามโหมด
            ai_response = ""
            source_documents = []

            # 2.1 เลือก Chain ที่ถูกต้องจาก Dictionary ตามภาษาและโหมด
            selected_chain = chains[detected_lang][selected_mode]

            if selected_mode == 'rag':
                print("2. กำลังค้นหาหนังสือที่เกี่ยวข้อง (RAG)...")
                # 2.2 ค้นหาข้อมูลจาก FAISS โดย "กรองด้วยภาษา" ที่ตรวจจับได้
                docs = db.similarity_search(user_query, k=4, filter={"language": detected_lang})
                context = "\n\n".join([doc.page_content for doc in docs])
                source_documents = docs
                
                print("3. กำลังสร้างคำตอบ...")
                ai_response = selected_chain.invoke({"context": context, "question": user_query})
            else: # general
                print("2. กำลังเตรียมคำตอบ (General Chat)...")
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
