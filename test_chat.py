import requests
import json

def test_chat():
    url = "http://localhost:8000/api/chat"
    payload = {"query": "test", "session_id": "test_session"}
    headers = {"Content-Type": "application/json"}
    
    print(f"Connecting to {url}...")
    try:
        response = requests.post(url, json=payload, headers=headers, stream=True, timeout=60)
        print(f"Status Code: {response.status_code}")
        print(f"Headers: {response.headers}")
        
        for line in response.iter_lines():
            if line:
                print(f"Chunk: {line.decode('utf-8')}")
                if "done" in line.decode('utf-8'):
                    break
    except Exception as e:
        print(f"CONNECTION ERROR: {e}")

if __name__ == "__main__":
    test_chat()
