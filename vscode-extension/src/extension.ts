
import * as vscode from 'vscode';

// Definiamo l'interfaccia per i messaggi del server
interface ServerResponse {
    type: 'markdown' | 'progress' | 'error' | 'debug';
    content: string;
}

const SERVER_URL = 'http://127.0.0.1:8000/stream';

export function activate(context: vscode.ExtensionContext) {

    console.log('Attivazione estensione STM32 AI Assistant');

    // 1. Registra il Chat Participant
    const handler: vscode.ChatRequestHandler = async (request: vscode.ChatRequest, context: vscode.ChatContext, stream: vscode.ChatResponseStream, token: vscode.CancellationToken) => {

        try {
            // Messaggio iniziale di progress
            // Nota: l'API stream.progress non è documentata come standard metodo dello stream in tutte le versioni, 
            // ma useremo un semplice messaggio markdown se necessario o verificheremo @types/vscode.
            // Per sicurezza, usiamo markdown "Thinking..." se progress non va, ma VS Code Insiders ha progress.
            // Assumiamo che il server gestisca il flusso.

            stream.markdown('Contatto il Brain STM32... \n\n');

            // 2. Prepara il payload per il server
            const messages = [
                // Aggiungi la history se necessario (qui semplificato)
                { role: 'user', content: request.prompt }
            ];

            // 3. Esegui richiesta POST al server Python
            // Nota: Node 18+ ha fetch nativo.
            const response = await fetch(SERVER_URL, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    messages: messages,
                    context: {
                        // Inserire qui eventuali info sul file aperto o selezione
                    }
                })
            });

            if (!response.ok) {
                throw new Error(`Errore server: ${response.status} ${response.statusText}`);
            }

            if (!response.body) {
                throw new Error("Nessuna risposta dal server");
            }

            // 4. Leggi lo stream (NDJSON o SSE)
            // Usiamo un lettore di stream testuale
            const reader = response.body.getReader();
            const decoder = new TextDecoder("utf-8");
            let buffer = "";

            while (true) {
                if (token.isCancellationRequested) {
                    console.log("Richiesta cancellata dall'utente.");
                    break;
                }

                const { done, value } = await reader.read();
                if (done) break;

                const chunk = decoder.decode(value, { stream: true });
                buffer += chunk;

                // Processa linee (NDJSON)
                const lines = buffer.split("\n");
                // Mantieni l'ultimo frammento se incompleto
                buffer = lines.pop() || "";

                for (const line of lines) {
                    if (!line.trim()) continue;
                    if (line.startsWith("data: ")) {
                        // Gestione SSE se il server usa 'data: '
                        // Ma il nostro server invia JSON raw o NDJSON per semplicità nel codice precedente.
                        // Se il server usa NDJSON puro:
                    }

                    try {
                        // Tenta di pulire eventuali prefissi SSE se presenti
                        const cleanLine = line.replace(/^data: /, "");
                        const data = JSON.parse(cleanLine) as ServerResponse;

                        if (data.type === 'markdown') {
                            stream.markdown(data.content);
                        } else if (data.type === 'progress') {
                            stream.markdown(`* ${data.content}...\n`);
                        } else if (data.type === 'error') {
                            stream.markdown(`\n> **Errore**: ${data.content}\n`);
                        } else if (data.type === 'debug') {
                            console.log(`[SERVER DEBUG] ${data.content}`);
                        }

                    } catch (e) {
                        console.warn("Errore parsing JSON chunk:", line, e);
                    }
                }
            }

        } catch (err) {
            if (err instanceof Error) {
                stream.markdown(`\n**Errore di connessione**: Impossibile contattare il server Python.\nAssicurati che ` + "`server.py`" + ` sia in esecuzione su porta 8000.\n\nDettaglio: ${err.message}`);
            } else {
                stream.markdown(`\n**Errore sconosciuto** durante la comunicazione con l'backend.`);
            }
        }
    };

    const helper = vscode.chat.createChatParticipant('stm32-ai.assistant', handler);
    helper.iconPath = new vscode.ThemeIcon('chip'); // Icona appropriata

    context.subscriptions.push(helper);
}

export function deactivate() { }
