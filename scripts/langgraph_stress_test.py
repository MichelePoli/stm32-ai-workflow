import asyncio
import aiohttp
import json
import time
import argparse

async def simulate_user_flow(session, base_url, user_id):
    """
    Simula l'interazione completa di un utente con l'endpoint FastAPI /stream
    di LangGraph. L'utente invia un messaggio "analizza il firmware" (o simile,
    che tu configurerai in modo che attraversi tutto il graph in bypass).
    """
    url = f"{base_url}/stream"
    start_time = time.time()
    
    # Prompt generico che dovrebbe triggerare tutto il mega-workflow bypassato
    payload = {
        "messages": [
            {"role": "user", "content": "Analizza il firmware e procedi con AI, Customization e Integration"}
        ],
        "user_id": f"stress_user_{user_id:02d}",
        "session_id": "stress_session"
    }

    # payload = {
    #     "messages": [
    #         {"role": "user", "content": "reset"}
    #     ],
    #     "user_id": f"stress_user_{user_id:02d}",
    #     "session_id": "stress_session"
    # } 
    # # solo se vuoi fare reset delle informazioni in redis commenta il payload di sopra e usa questa.
    
    print(f"[Utente {user_id:02d}] 🚀 Inizio esecuzione workflow...")
    try:
        # Usa timeout alti, questo flusso compila file e genera AI, 
        # potrebbe volerci molto tempo.
        async with session.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=800) as response:
            
            # Leggiamo lo stream SSE (Server-Sent Events) o NDJSON
            nodes_visited = []
            final_response = ""
            
            async for line in response.content:
                line = line.decode('utf-8').strip()
                if not line:
                    continue
                    
                # Format SSE: "data: {...}"
                if line.startswith("data: "):
                    line = line[6:]
                    
                try:
                    event = json.loads(line)
                    event_type = event.get("type")
                    
                    if event_type == "progress":
                        nodes_visited.append(event.get("content"))
                        # print(f"[Utente {user_id:02d}] 🔄 {event.get('content')}") # decommenta per debug
                    elif event_type == "markdown":
                        final_response += event.get("content", "")
                    elif event_type == "status" and event.get("event") == "completed":
                        break
                except json.JSONDecodeError:
                    pass
            
            latency = time.time() - start_time
            print(f"[Utente {user_id:02d}] ✅ Graph Completato in {latency:.2f}s (Nodi visitati: {len(nodes_visited)})")
            return True, latency, len(nodes_visited), len(final_response)
            
    except asyncio.TimeoutError:
         print(f"[Utente {user_id:02d}] ❌ Timeout dopo 800s")
         return False, 800, 0, 0
    except Exception as e:
        print(f"[Utente {user_id:02d}] ❌ Eccezione: {str(e)}")
        return False, time.time() - start_time, 0, 0

async def main():
    parser = argparse.ArgumentParser(description="LangGraph / FastAPI Stress Test (Simula esecuzione Workflow Completo Bypassato)")
    parser.add_argument("--url", default="http://localhost:8000", help="URL base di FastAPI (Server.py)")
    parser.add_argument("--users", type=int, default=5, help="Numero di utenti (flussi) paralleli")
    args = parser.parse_args()

    print(f"\n🚀 Inizio Stress Test LangGraph / FastAPI")
    print(f"🎯 Url Base: {args.url}")
    print(f"👥 Flussi concorrenti totali: {args.users}")
    # print("ATTENZIONE: Assicurati di aver commentato tutti gli '__interrupt__' in graph.py e nei workflow!")
    print("-" * 50)
    
    start_total = time.time()
    
    # Nessun timeout globale: 
    # la singola request ha un timeout interno di 800 secondi
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i in range(args.users):
            # Sfalsa leggermente la partenza degli utenti di 0.5s per simulare un arrivo realistico
            await asyncio.sleep(0.5) 
            tasks.append(simulate_user_flow(session, args.url, i+1))
        
        results = await asyncio.gather(*tasks)

    end_total = time.time()
    
    successes = [r for r in results if r[0]]
    failures = [r for r in results if not r[0]]
    
    print("\n📊 --- RISULTATI STRESS TEST LANGGRAPH ---")
    print(f"Tempo totale esecuzione app: {end_total - start_total:.2f} secondi")
    print(f"Tasso di successo:           {len(successes)}/{args.users} ({(len(successes)/args.users)*100:.1f}%)")
    
    if successes:
        lats = [r[1] for r in successes]
        nodes = [r[2] for r in successes]
        print(f"\nLatenza Workflow MEDIA:      {sum(lats)/len(lats):.2f} sec")
        print(f"Nodi LangGraph visitati:     ~{sum(nodes)/len(nodes):.0f} per utente")
    
    if failures:
        print(f"⚠️ Ci sono stati {len(failures)} errori (guarda i log sopra)")

if __name__ == '__main__':
    asyncio.run(main())
