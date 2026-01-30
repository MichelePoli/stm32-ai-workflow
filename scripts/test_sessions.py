import requests
import json
import time
import threading

URL = "http://127.0.0.1:8000/stream"

def send_request(user_id, session_id, message):
    payload = {
        "messages": [{"role": "user", "content": message}],
        "user_id": user_id,
        "session_id": session_id
    }
    print(f"[{user_id}:{session_id}] Inviando: {message}")
    response = requests.post(URL, json=payload, stream=True)
    
    full_response = ""
    is_finished = False
    for line in response.iter_lines():
        if line:
            data = json.loads(line.decode("utf-8"))
            if data["type"] == "markdown":
                full_response += data["content"]
            elif data["type"] == "progress":
                print(f"[{user_id}:{session_id}] Progresso: {data['content']}")
            elif data["type"] == "status" and data["event"] == "completed":
                print(f"[{user_id}:{session_id}] ✨ PIPELINE COMPLETATA!")
                is_finished = True
    
    print(f"[{user_id}:{session_id}] Risposta Finale: {full_response[:100]}...")
    if is_finished:
        print(f"[{user_id}:{session_id}] ✅ Verifica terminata con successo.")

if __name__ == "__main__":
    # Test 1: Utente Alpha, Sessione 1
    t1 = threading.Thread(target=send_request, args=("user-alpha", "session-1", "Voglio creare un progetto firmware per STM32F4"))
    
    # Test 2: Utente Beta, Sessione A
    t2 = threading.Thread(target=send_request, args=("user-beta", "session-A", "Analizza un modello MobileNet"))
    
    t1.start()
    time.sleep(1) # Piccolo delay per vedere i log separati
    t2.start()
    
    t1.join()
    t2.join()
    
    print("\n✅ Test completato. Controlla i log del server per vedere i diversi thread_id!")
