# test_tts.py - Debug script for Edge TTS
import asyncio
import edge_tts

async def test_tts():
    """Test Edge TTS with different voices and texts."""
    
    test_cases = [
        ("th-TH-NiwatNeural", "สวัสดีครับ ทดสอบ"),
        ("th-TH-PremwadeeNeural", "สวัสดีค่ะ ทดสอบ"),
        ("en-US-GuyNeural", "Hello, this is a test."),
        ("en-US-JennyNeural", "Hello, this is a test."),
    ]
    
    print("=" * 60)
    print("🔊 Edge TTS Test Script")
    print("=" * 60)
    
    for voice, text in test_cases:
        print(f"\n🎤 Testing voice: {voice}")
        print(f"   📝 Text: {text}")
        
        try:
            filename = f"test_{voice.replace('-', '_')}.mp3"
            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(filename)
            print(f"   ✅ SUCCESS! Saved as: {filename}")
        except Exception as e:
            print(f"   ❌ FAILED! Error: {e}")
    
    print("\n" + "=" * 60)
    print("🔍 Fetching available Thai voices...")
    print("=" * 60)
    
    try:
        voices = await edge_tts.list_voices()
        thai_voices = [v for v in voices if v["Locale"].startswith("th-")]
        print(f"\n📋 Found {len(thai_voices)} Thai voices:")
        for v in thai_voices:
            print(f"   - {v['ShortName']} ({v['Gender']})")
    except Exception as e:
        print(f"   ❌ Error listing voices: {e}")

if __name__ == "__main__":
    asyncio.run(test_tts())
