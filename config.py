# config.py
import os
from dotenv import load_dotenv

# สั่งให้ไลบรารีโหลดค่าต่างๆ จากไฟล์ .env เข้ามาในระบบ
load_dotenv()

# อ่านค่า API Key จากตัวแปรที่โหลดมา
# แล้วเก็บไว้ในตัวแปรใหม่ของ Python ชื่อ API_KEY
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

GOOGLE_API_KEY_NNG = os.getenv("GOOGLE_API_KEY_NNG")

GOOGLE_API_KEY_NNG = os.getenv("GOOGLE_API_KEY_NNG2")

GOOGLE_API_KEY_RTN = os.getenv("GOOGLE_API_KEY_RTN")

GOOGLE_API_KEY_GEZ = os.getenv("GOOGLE_API_KEY_GGEZ")

GOOGLE_API_KEY_GEZ = os.getenv("GOOGLE_API_KEY_GGEZ2")

GOOGLE_API_KEY_GEZ = os.getenv("GOOGLE_API_KEY_NAME")