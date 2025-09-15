# data_preparation.py (เวอร์ชันปรับปรุงสำหรับ Language-Specific RAG)
# สคริปต์สำหรับกระบวนการ ETL (Extract, Transform, Load)
# หน้าที่: อ่านข้อมูล, สร้าง Chunks แบบแยกภาษาพร้อม Metadata, สร้าง Embeddings, และบันทึกเป็น FAISS Index

import os
import json
import time
import shutil
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

# LangChain components
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.docstore.document import Document

# --- Configuration ---

# V V V --- สวิตช์ควบคุม --- V V V
USE_PROGRESS_LOG = False
# ^ ^ ^ --- สวิตช์ควบคุม --- ^ ^ ^

DATA_DIR = Path("data")
INDEX_DIR = Path("faiss_index")
DATA_FILE = DATA_DIR / "all_books.jsonl"
INDEX_PATH = INDEX_DIR / "book_index"
PROGRESS_FILE = DATA_DIR / "progress.log"

DATA_DIR.mkdir(exist_ok=True)
INDEX_DIR.mkdir(exist_ok=True)

# --- Helper Functions (คงเดิม) ---

def load_progress():
    if not USE_PROGRESS_LOG or not PROGRESS_FILE.exists():
        return set()
    with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f)

def save_progress(book_id):
    if not USE_PROGRESS_LOG:
        return
    with open(PROGRESS_FILE, "a", encoding="utf-8") as f:
        f.write(f"{book_id}\n")

# --- Main Script ---

def main():
    print("🚀 Starting AI Librarian Data Preparation Script (Language-Specific RAG Version)...")
    print(f"🔄 Progress Tracking Mode: {'ENABLED' if USE_PROGRESS_LOG else 'DISABLED'}")

    if not USE_PROGRESS_LOG:
        print("🔥 Forcing a full rebuild...")
        if INDEX_PATH.exists():
            shutil.rmtree(INDEX_PATH)
            print("🗑️  Deleted existing FAISS index.")
        if PROGRESS_FILE.exists():
            os.remove(PROGRESS_FILE)
            print("🗑️  Deleted progress log file.")

    load_dotenv()
    google_api_key = os.getenv("GOOGLE_API_KEY_NNG") # <-- อาจจะต้องเปลี่ยน Key ให้ตรงกับ .env ของคุณ
    if not google_api_key:
        print("❌ Error: GOOGLE_API_KEY_NNG not found in .env file.")
        return

    if not DATA_FILE.exists():
        print(f"❌ Error: Data file not found at {DATA_FILE}")
        return

    print("✨ Initializing embedding model (text-embedding-004)...")
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=google_api_key,
            request_timeout=120
        )
    except Exception as e:
        print(f"❌ Error initializing embedding model: {e}")
        return

    processed_books = load_progress()
    if processed_books:
        print(f"✅ Found {len(processed_books)} books already processed. They will be skipped.")

    all_docs = []
    new_book_count = 0
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    print("📚 Processing books from all_books.jsonl...")
    for line in tqdm(lines, desc="Processing Books"):
        try:
            book = json.loads(line)
            book_id = book.get("book_id")

            if not book_id or book_id in processed_books:
                continue

            ### <<< CHANGE START: NEW CHUNKING LOGIC >>> ###

            # --- 1. Process Thai Data ---
            title_th = book.get("title_th")
            if title_th:
                # 1.1 Create Thai Book Overview Chunk
                key_points_th_text = "\n".join(f"- {point}" for point in book.get("key_points_th", []))
                overview_content_th = (
                    f"ภาพรวมหนังสือ: {title_th}\n"
                    f"โดย: {', '.join(book.get('author_th', []))}\n"
                    f"บทสรุป: {book.get('summary_th', '')}\n\n"
                    f"ประเด็นสำคัญ:\n{key_points_th_text}"
                )
                overview_metadata_th = {
                    "language": "th",
                    "book_id": book_id,
                    "source": title_th,
                    "chunk_type": "book_overview"
                }
                all_docs.append(Document(page_content=overview_content_th, metadata=overview_metadata_th))

                # 1.2 Create Thai Chapter Chunks
                for chapter in book.get("chapters_th", []):
                    chapter_content_th = (
                        f"จากหนังสือ: {title_th}\n"
                        f"บทที่ {chapter.get('chapter_number', 'N/A')}: {chapter.get('title', '')}\n\n"
                        f"สรุปเนื้อหา: {chapter.get('summary', '')}"
                    )
                    chapter_metadata_th = {
                        "language": "th",
                        "book_id": book_id,
                        "source": title_th,
                        "chunk_type": "chapter",
                        "chapter": chapter.get('chapter_number', 0)
                    }
                    all_docs.append(Document(page_content=chapter_content_th, metadata=chapter_metadata_th))

            # --- 2. Process English Data ---
            title_en = book.get("title_en")
            if title_en:
                # 2.1 Create English Book Overview Chunk
                key_points_en_text = "\n".join(f"- {point}" for point in book.get("key_points_en", []))
                overview_content_en = (
                    f"Book Overview: {title_en}\n"
                    f"By: {', '.join(book.get('author_en', []))}\n"
                    f"Summary: {book.get('summary_en', '')}\n\n"
                    f"Key Points:\n{key_points_en_text}"
                )
                overview_metadata_en = {
                    "language": "en",
                    "book_id": book_id,
                    "source": title_en,
                    "chunk_type": "book_overview"
                }
                all_docs.append(Document(page_content=overview_content_en, metadata=overview_metadata_en))

                # 2.2 Create English Chapter Chunks
                for chapter in book.get("chapters_en", []):
                    chapter_content_en = (
                        f"From the book: {title_en}\n"
                        f"Chapter {chapter.get('chapter_number', 'N/A')}: {chapter.get('title', '')}\n\n"
                        f"Content Summary: {chapter.get('summary', '')}"
                    )
                    chapter_metadata_en = {
                        "language": "en",
                        "book_id": book_id,
                        "source": title_en,
                        "chunk_type": "chapter",
                        "chapter": chapter.get('chapter_number', 0)
                    }
                    all_docs.append(Document(page_content=chapter_content_en, metadata=chapter_metadata_en))
            
            ### <<< CHANGE END >>> ###
            
            new_book_count += 1
            save_progress(book_id)

        except json.JSONDecodeError:
            print(f"⚠️ Warning: Skipping a line due to invalid JSON format. Content: {line[:100]}")
            continue

    if not all_docs:
        print("\n✅ No new books to process. The index is already up to date.")
        return

    print(f"\nCreated {len(all_docs)} new document chunks from {new_book_count} new book(s).")
    
    max_retries = 5
    retry_delay = 5

    for attempt in range(max_retries):
        try:
            if INDEX_PATH.exists() and USE_PROGRESS_LOG:
                print("📂 Loading existing FAISS index to add new data...")
                db = FAISS.load_local(str(INDEX_PATH), embeddings, allow_dangerous_deserialization=True)
                print(f"➕ Adding {len(all_docs)} new documents to the index...")
                db.add_documents(all_docs)
            else:
                print("✨ Creating new FAISS index from scratch...")
                db = FAISS.from_documents(all_docs, embeddings)
            
            print("💾 Saving index to disk...")
            db.save_local(str(INDEX_PATH))
            print("🎉 Success! FAISS index has been created/updated successfully.")
            break

        except Exception as e:
            print(f"🔥 An error occurred during embedding/indexing (Attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                print("❌ Failed to create/update index after multiple retries.")
                break

if __name__ == "__main__":
    main()