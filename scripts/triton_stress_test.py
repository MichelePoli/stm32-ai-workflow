import asyncio
import aiohttp
import time
import json
import argparse
import statistics

async def infer(session, base_url, payload, user_id, model):
    """
    Simula un utente che invia una richiesta a Triton per uno specifico modello.
    Restituisce: (successo, latenza, lunghezza_testo, id_utente, modello)
    """
    url = f"{base_url}/v2/models/{model}/infer"
    start_time = time.time()
    try:
        async with session.post(url, json=payload, headers={"Content-Type": "application/json"}) as response:
            res_text = await response.text()
            if response.status == 200:
                res = json.loads(res_text)
                
                text_len = 0
                try:
                    # Estrae il testo generato dalla risposta standard del backend Python di Triton
                    output_data = res["outputs"][0]["data"][0]
                    content = output_data if isinstance(output_data, str) else output_data.decode("utf-8")
                    text_len = len(content)
                except Exception:
                    pass
                
                latency = time.time() - start_time
                print(f"[Utente {user_id:02d} | {model}] ✅ Risposta in {latency:.2f}s ({text_len} chars)")
                return True, latency, text_len, user_id, model
            else:
                print(f"[Utente {user_id:02d} | {model}] ❌ Errore HTTP {response.status}: {res_text[:100]}")
                return False, time.time() - start_time, 0, user_id, model
    except Exception as e:
        print(f"[Utente {user_id:02d} | {model}] ❌ Eccezione: {e}")
        return False, time.time() - start_time, 0, user_id, model

async def main():
    parser = argparse.ArgumentParser(description="Triton LLM Stress Test (Simula N utenti concorrenti)")
    parser.add_argument("--url", default="http://130.192.40.61", help="URL base di Triton (senza /v2/...)")
    parser.add_argument("--models", default="mistral", help="Nomi dei modelli caricati su Triton separati da virgola (es. mistral,gpt-oss-20b)")
    parser.add_argument("--users", type=int, default=10, help="Numero totale di utenti concorrenti da simulare")
    parser.add_argument("--prompt", default="Spiega in modo dettagliato come funziona una Convolutional Neural Network (CNN) per la classificazione di immagini. Scrivi circa 300 parole.", help="Prompt complesso per far generare molti token")
    args = parser.parse_args()

    # Prepara array di modelli
    models = [m.strip() for m in args.models.split(',')]
    base_url = args.url.rstrip('/')
    
    # Payload nel formato atteso dal triton_client.py
    # Manca la formattazione esatta dei template, ma approssimiamo una chat standard ChatML/Llama3
    payload = {
        "inputs": [
            {
                "name": "PROMPT",
                "shape": [1, 1],
                "datatype": "BYTES",
                "data": [f"System: You are an AI assistant.\nUser: {args.prompt}\nAssistant: "]
            }
        ]
    }

    print(f"\n🚀 Inizio Stress Test su Triton")
    print(f"🎯 Url Base: {base_url}")
    print(f"🧠 Modelli in test: {', '.join(models)}")
    print(f"👥 Utenti concorrenti totali: {args.users}")
    print(f"📝 Lunghezza prompt: {len(args.prompt)} caratteri")
    print("-" * 50)
    
    import urllib.request
    
    # Carica tutti i modelli richiesti sequenzialmente
    for model in models:
        load_endpoint = f"{base_url}/v2/repository/models/{model}/load"
        print(f"⏳ Richiesta caricamento modello '{model}' in VRAM (potrebbe richiedere minuti)...")
        try:
            req = urllib.request.Request(load_endpoint, method="POST")
            with urllib.request.urlopen(req, timeout=180) as response:
                pass
            print(f"✅ Modello '{model}' caricato con successo!")
        except Exception as e:
            print(f"⚠️ Caricamento saltato/fallito per '{model}' (magari è già in VRAM): {e}")

    # Pausa di assestamento
    await asyncio.sleep(2)

    start_total = time.time()
    
    # Timeout di 5 minuti
    timeout = aiohttp.ClientTimeout(total=300) 
    
    async with aiohttp.ClientSession(timeout=timeout) as session:
        # Crea N task sparpagliandoli sui vari modelli (Round-Robin)
        tasks = []
        for i in range(args.users):
            assigned_model = models[i % len(models)]
            tasks.append(infer(session, base_url, payload, i+1, assigned_model))
        
        # Aspetta che tutti finiscano
        results = await asyncio.gather(*tasks)

    end_total = time.time()
    total_duration = end_total - start_total
    
    # Estrazione Metriche
    successes = [r for r in results if r[0]]
    failures = [r for r in results if not r[0]]
    latencies = [r[1] for r in successes]
    total_chars_generated = sum([r[2] for r in successes])
    
    print("\n📊 --- RISULTATI DELLO STRESS TEST ---")
    print(f"Tempo totale di esecuzione: {total_duration:.2f} secondi")
    print(f"Tasso di successo:          {len(successes)}/{args.users} ({(len(successes)/args.users)*100:.1f}%)")
    
    if latencies:
        print(f"Latenza MIN:                {min(latencies):.2f} sec")
        print(f"Latenza MAX:                {max(latencies):.2f} sec")
        print(f"Latenza MEDIA:              {statistics.mean(latencies):.2f} sec")
        print(f"Latenza MEDIANA:            {statistics.median(latencies):.2f} sec")
        
        # Approssimazione: 1 token = ~4 caratteri 
        approx_tokens = total_chars_generated / 4
        tps = approx_tokens / total_duration
        print(f"Lunghezza media risposta:   {total_chars_generated/len(successes):.0f} caratteri")
        print(f"Throughput globale:         ~{tps:.1f} tokens/secondo globali")
        
        # Statistiche per modello
        print("\n📈 Latenze medie per modello:")
        for model in models:
            model_lats = [r[1] for r in successes if r[4] == model]
            if model_lats:
                print(f"  - {model}: {statistics.mean(model_lats):.2f} sec ({len(model_lats)} utenti)")
    
    if failures:
        print(f"⚠️ Ci sono stati {len(failures)} errori (guarda i log sopra)")

if __name__ == '__main__':
    asyncio.run(main())



# Nota: il file model_repository/gpt-oss-20b/1/model.py fisicamente presente sul server HPP ha già il parametro gpu_memory_utilization=0.45 cablato dentro il codice. Non c'è alcun bisogno di fare l'override tramite payload JSON !