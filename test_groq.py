"""
Quick test: verifies Groq API key works correctly.
Run: python test_groq.py
"""
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GROQ_API_KEY", "")
model   = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

print(f"GROQ_API_KEY loaded: {'YES (' + api_key[:8] + '...)' if api_key and api_key != 'your_groq_api_key_here' else 'NO ❌ - add your key to .env'}")
print(f"GROQ_MODEL: {model}")

if not api_key or api_key == "your_groq_api_key_here":
    print("\n❌ ERROR: Set your GROQ_API_KEY in the .env file first.")
    print("   Get a FREE key at: https://console.groq.com")
    exit(1)

print("\nTesting Groq API call...")
try:
    from groq import Groq
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Say 'Groq is working!' in exactly 5 words."}],
        max_tokens=50,
    )
    reply = response.choices[0].message.content
    print(f"\n✅ SUCCESS! Groq replied: {reply}")
    print("\nYour API key is valid. Restart the server: python app.py")
except Exception as e:
    print(f"\n❌ FAILED: {e}")
    print("   Check your API key and try again.")
