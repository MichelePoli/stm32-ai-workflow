import asyncio
import aiohttp
import time
import json
import argparse

SERVER_URL = "http://127.0.0.1:8000/stream"

async def simulate_user(session_client: aiohttp.ClientSession, user_num: int, message: str):
    """Simula una singola richiesta chat utente al server."""
    user_id = f"test_user_{user_num}"
    session_id = f"stress-session-{int(time.time())}"
    
    payload = {
        "messages": [{"role": "user", "content": message}],
        "context": {},
        "user_id": user_id,
        "session_id": session_id
    }
    
    print(f"[User {user_num}] Inizio richiesta: '{message}'")
    start_time = time.time()
    
    try:
        async with session_client.post(SERVER_URL, json=payload) as response:
            if response.status != 200:
                print(f"[User {user_num}] ❌ Errore API: HTTP {response.status}")
                return False, 0
                
            # Leggiamo lo stream NDJSON
            async for line in response.content:
                if line:
                    decoded_line = line.decode('utf-8').strip()
                    try:
                        data = json.loads(decoded_line)
                        if data.get("type") == "error":
                            print(f"[User {user_num}] ❌ Errore dal server: {data.get('content')}")
                        elif data.get("type") == "status" and data.get("event") == "completed":
                            # Completato con successo!
                            pass
                    except json.JSONDecodeError:
                        pass
                        
            end_time = time.time()
            elapsed = end_time - start_time
            print(f"[User {user_num}] ✅ Completato in {elapsed:.2f} secondi.")
            return True, elapsed
            
    except Exception as e:
        print(f"[User {user_num}] ❌ Eccezione: {e}")
        return False, 0

async def main():
    parser = argparse.ArgumentParser(description="Stress test per l'API STM32 AI Workflow")
    parser.add_argument("-c", "--concurrency", type=int, default=3, help="Numero di utenti concorrenti")
    args = parser.parse_args()
    
    print(f"🚀 Avvio Stress Test con {args.concurrency} utenti concorrenti...")
    print(f"Target: {SERVER_URL}")
    print("-" * 50)
    
    # Mix di domande generiche o richieste che non triggerano lunghissimi job NNI
    # Per stressare Triton usiamo task di routing o chat generale che richiedono l'LLM
    test_message = "Qual è la differenza tra quantizzazione low e high in STEdgeAI?"
    
    start_total = time.time()
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i in range(args.concurrency):
            tasks.append(simulate_user(session, i + 1, test_message))
            
        results = await asyncio.gather(*tasks)
    
    end_total = time.time()
    
    # Statistiche
    successes = sum(1 for r, _ in results if r)
    failures = args.concurrency - successes
    latencies = [l for r, l in results if r]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    
    print("-" * 50)
    print("📊 RISULTATI STRESS TEST")
    print(f"Utenti Totali:     {args.concurrency}")
    print(f"Successi:          {successes}")
    print(f"Fallimenti:        {failures}")
    print(f"Tempo Totale:      {end_total - start_total:.2f} sec")
    print(f"Latenza Media:     {avg_latency:.2f} sec/utente")
    
if __name__ == "__main__":
    asyncio.run(main())
