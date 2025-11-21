# config.py
import os
import itertools
from dotenv import load_dotenv

# โหลดค่าจาก .env
load_dotenv()

class KeyManager:
    def __init__(self):
        # โหลดและแปลง String เป็น List ทันทีที่เริ่มทำงาน
        self.groq_keys = self._load_and_parse("GROQ_API_KEYS")
        self.google_keys = self._load_and_parse("GOOGLE_API_KEYS")
        
        # สร้างตัวหมุนคีย์ (Iterator) เพื่อให้สลับคีย์ใช้อัตโนมัติ
        self._groq_cycle = itertools.cycle(self.groq_keys) if self.groq_keys else None
        self._google_cycle = itertools.cycle(self.google_keys) if self.google_keys else None

    def _load_and_parse(self, env_var_name):
        """ฟังก์ชันช่วยสำหรับอ่านค่าและแยกด้วยเครื่องหมาย ,"""
        keys_str = os.getenv(env_var_name, "")
        if not keys_str:
            return []
        # แยกด้วย , และลบช่องว่างซ้ายขวาออก (เผื่อเผลอใส่มา)
        return [k.strip() for k in keys_str.split(",") if k.strip()]

    def get_groq_key(self):
        """ดึงคีย์ Groq ตัวถัดไป"""
        if not self._groq_cycle:
            raise ValueError("ไม่พบ GROQ_API_KEYS ในไฟล์ .env")
        return next(self._groq_cycle)

    def get_google_key(self):
        """ดึงคีย์ Google ตัวถัดไป"""
        if not self._google_cycle:
            raise ValueError("ไม่พบ GOOGLE_API_KEYS ในไฟล์ .env")
        return next(self._google_cycle)

# สร้าง instance ของคลาสไว้ให้ไฟล์อื่นเรียกใช้ได้เลย
api_keys = KeyManager()