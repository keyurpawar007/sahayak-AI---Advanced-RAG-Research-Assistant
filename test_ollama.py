import requests
import time

def test_ollama():
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "llama3.2",
        "prompt": "Say 'Ollama is ready'",
        "stream": False
    }
    
    print(f"Connecting to Ollama at {url}...")
    start = time.time()
    try:
        response = requests.post(url, json=payload, timeout=30)
        end = time.time()
        if response.status_code == 200:
            print(f"SUCCESS! Response: {response.json().get('response')}")
            print(f"Time taken: {end - start:.2f}s")
        else:
            print(f"FAILED with status {response.status_code}: {response.text}")
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    test_ollama()
