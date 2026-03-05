import requests
import json
import time

URL = "http://130.192.40.61/stream"

def send_request(user_id, session_id, message):
    payload = {
        "messages": [{"role": "user", "content": message}],
        "user_id": user_id,
        "session_id": session_id
    }
    print(f"\n--- [{user_id}:{session_id}] INVIO: {message} ---")
    response = requests.post(URL, json=payload, stream=True)
    
    full_response = ""
    for line in response.iter_lines():
        if line:
            data = json.loads(line.decode("utf-8"))
            if data["type"] == "markdown":
                full_response += data["content"]
            elif data["type"] == "status" and data["event"] == "completed":
                print(f"[{user_id}:{session_id}] ✨ PIPELINE COMPLETATA")
    
    print(f"[{user_id}:{session_id}] RISPOSTA: {full_response[:200]}...")

if __name__ == "__main__":
    USER = "michele_test"
    
    # STEP 1: Prima sessione - Diciamo all'assistente qualcosa da ricordare
    # In questo caso, il workflow route_request estrarrà info dal messaggio
    send_request(USER, "session-001", "Voglio creare un progetto firmware per una board STM32H7")
    
    print("\nAspetto 2 secondi per il salvataggio su Redis...")
    time.sleep(2)
    
    # STEP 2: Seconda sessione - Verifichiamo se ricorda
    # Usiamo una NUOVA session_id. Il server dovrebbe caricare il profilo da Redis.
    send_request(USER, "session-002", "Quale board stavo usando?")
    
    print("\n✅ Test terminato. Controlla i log del server per vedere 'Profilo utente caricato'!")
