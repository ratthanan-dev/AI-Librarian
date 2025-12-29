# data_preparation.py
# เวอร์ชัน: Multi-Key, Batch Processing, Rich Metadata

import os
import json
import time
import shutil
from pathlib import Path
from tqdm import tqdm

# LangChain components
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
# นำเข้า Config ใหม่
from config import api_keys

# --- Configuration ---

BATCH_SIZE = 10  # จำนวนหนังสือที่จะประมวลผลก่อนบันทึก 1 ครั้ง (ป้องกันแรมเต็มและ Error)
DATA_DIR = Path("data")
INDEX_DIR = Path("faiss_index")
DATA_FILE = DATA_DIR / "all_books.jsonl"
INDEX_PATH = INDEX_DIR / "book_index"
PROGRESS_FILE = DATA_DIR / "progress.log"

# สร้างโฟลเดอร์ถ้ายังไม่มี
DATA_DIR.mkdir(exist_ok=True)
INDEX_DIR.mkdir(exist_ok=True)

# --- Helper Functions ---

def load_progress():
    if not PROGRESS_FILE.exists():
        return set()
    with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f)

def save_progress(book_ids):
    with open(PROGRESS_FILE, "a", encoding="utf-8") as f:
        for bid in book_ids:
            f.write(f"{bid}\n")

def get_embeddings_model():
    """สร้าง Embedding Model โดยใช้คีย์จาก Config"""
    return GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=api_keys.get_google_key(), # ใช้คีย์จาก KeyManager
        request_timeout=120
    )

# --- Main Script ---

def main():
    print("🚀 Starting AI Librarian Data Preparation...")
    print(f"📦 Batch Size: {BATCH_SIZE}")

    if not DATA_FILE.exists():
        print(f"❌ Error: Data file not found at {DATA_FILE}")
        return

    # โหลดรายการหนังสือที่ทำเสร็จแล้ว
    processed_books = load_progress()
    if processed_books:
        print(f"🔄 Found {len(processed_books)} books already processed. Resuming...")

    # เตรียม Embedding Model
    embeddings = get_embeddings_model()

    # เตรียมตัวแปรสำหรับ Batch
    batch_docs = []
    batch_book_ids = []
    
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()

    total_lines = len(lines)
    
    for i, line in enumerate(tqdm(lines, desc="Processing Books")):
        try:
            book = json.loads(line)
            book_id = book.get("book_id")

            # ข้ามถ้าทำไปแล้ว
            if not book_id or book_id in processed_books:
                continue

            # --- ดึงข้อมูลส่วนกลาง (Metadata) ---
            # เราจะแปลง List เป็น String เพื่อเก็บใน Metadata (FAISS ไม่ชอบ List)
            authors_th = ", ".join(book.get('author_th', []))
            authors_en = ", ".join(book.get('author_en', []))
            category_th = book.get('category_th', 'General')
            category_en = book.get('category_en', 'General')
            year = str(book.get('publication_year', ''))

            # --- 1. Process Thai Data (TH) ---
            title_th = book.get("title_th")
            if title_th:
                # 1.1 Book Overview (TH)
                key_points_th = "\n".join(f"- {p}" for p in book.get("key_points_th", []))
                
                # ปรับปรุง Content: ใส่ Category เพื่อให้ AI เข้าใจบริบทกว้างๆ
                overview_content_th = (
                    f"หนังสือ: {title_th}\n"
                    f"ผู้แต่ง: {authors_th}\n"
                    f"หมวดหมู่: {category_th}\n"  # <--- เพิ่มหมวดหมู่ในเนื้อหา
                    f"ปีที่พิมพ์: {year}\n\n"
                    f"บทสรุปย่อ:\n{book.get('summary_th', '')}\n\n"
                    f"ประเด็นสำคัญ:\n{key_points_th}"
                )
                
                # ปรับปรุง Metadata: เพิ่มข้อมูลสำหรับการ Filter
                meta_th = {
                    "book_id": book_id,
                    "language": "th",
                    "source": title_th,
                    "type": "overview",
                    "author": authors_th,
                    "category": category_th,
                    "year": year
                }
                batch_docs.append(Document(page_content=overview_content_th, metadata=meta_th))

                # 1.2 Chapters (TH)
                for chapter in book.get("chapters_th", []):
                    chap_content_th = (
                        f"หนังสือ: {title_th} (หมวด: {category_th})\n"
                        f"บทที่ {chapter.get('chapter_number')}: {chapter.get('title')}\n\n"
                        f"เนื้อหาโดยสรุป:\n{chapter.get('summary', '')}"
                    )
                    chap_meta_th = meta_th.copy()
                    chap_meta_th.update({
                        "type": "chapter",
                        "chapter_num": chapter.get('chapter_number')
                    })
                    batch_docs.append(Document(page_content=chap_content_th, metadata=chap_meta_th))

            # --- 2. Process English Data (EN) ---
            title_en = book.get("title_en")
            if title_en:
                # 2.1 Book Overview (EN)
                key_points_en = "\n".join(f"- {p}" for p in book.get("key_points_en", []))
                
                overview_content_en = (
                    f"Book: {title_en}\n"
                    f"Author: {authors_en}\n"
                    f"Category: {category_en}\n" # <--- Context Injection
                    f"Year: {year}\n\n"
                    f"Summary:\n{book.get('summary_en', '')}\n\n"
                    f"Key Points:\n{key_points_en}"
                )
                
                meta_en = {
                    "book_id": book_id,
                    "language": "en",
                    "source": title_en,
                    "type": "overview",
                    "author": authors_en,
                    "category": category_en,
                    "year": year
                }
                batch_docs.append(Document(page_content=overview_content_en, metadata=meta_en))

                # 2.2 Chapters (EN)
                for chapter in book.get("chapters_en", []):
                    chap_content_en = (
                        f"Book: {title_en} (Category: {category_en})\n"
                        f"Chapter {chapter.get('chapter_number')}: {chapter.get('title')}\n\n"
                        f"Content Summary:\n{chapter.get('summary', '')}"
                    )
                    chap_meta_en = meta_en.copy()
                    chap_meta_en.update({
                        "type": "chapter",
                        "chapter_num": chapter.get('chapter_number')
                    })
                    batch_docs.append(Document(page_content=chap_content_en, metadata=chap_meta_en))

            # เพิ่ม ID ลงใน Batch tracker
            batch_book_ids.append(book_id)

            # --- Batch Saving Logic ---
            # บันทึกเมื่อครบ Batch หรือเป็นหนังสือเล่มสุดท้าย
            if len(batch_book_ids) >= BATCH_SIZE or i == total_lines - 1:
                if batch_docs:
                    try:
                        print(f"\n💾 Saving batch of {len(batch_book_ids)} books ({len(batch_docs)} chunks)...")
                        
                        if INDEX_PATH.exists():
                            # โหลดของเดิมมาบวกเพิ่ม
                            db = FAISS.load_local(str(INDEX_PATH), embeddings, allow_dangerous_deserialization=True)
                            db.add_documents(batch_docs)
                        else:
                            # สร้างใหม่ถ้ายังไม่มี
                            db = FAISS.from_documents(batch_docs, embeddings)
                        
                        # บันทึกลง Disk
                        db.save_local(str(INDEX_PATH))
                        
                        # บันทึก Progress
                        save_progress(batch_book_ids)
                        
                        # Reset Batch
                        batch_docs = []
                        batch_book_ids = []
                        
                        # พักหายใจ 2 วินาที เพื่อลดภาระ API
                        time.sleep(2) 
                        
                    except Exception as e:
                        print(f"❌ Error saving batch: {e}")
                        # กรณี Error อาจจะเลือกที่จะหยุด หรือข้าม ไปก่อน
                        # ในที่นี้เลือกที่จะไม่บันทึก Progress ของ Batch นี้เพื่อให้ลองใหม่รอบหน้า
                        batch_docs = []
                        batch_book_ids = []

        except json.JSONDecodeError:
            print(f"⚠️ Invalid JSON at line {i}")
            continue
        except Exception as e:
            print(f"⚠️ Unexpected error processing line {i}: {e}")
            continue

    print("\n✅ All Data Processing Complete!")

if __name__ == "__main__":
    main()