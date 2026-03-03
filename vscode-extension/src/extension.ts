
import * as vscode from 'vscode';

// Definiamo l'interfaccia per i messaggi del server
interface ServerResponse {
    type: 'markdown' | 'progress' | 'error' | 'debug';
    content: string;
}

const SERVER_URL = 'http://127.0.0.1:8000/stream';

export function activate(context: vscode.ExtensionContext) {

    console.log('Attivazione estensione STM32 AI Assistant');

    // -----------------------------------------------------------------------
    // SESSION ID: generato UNA SOLA VOLTA all'attivazione dell'estensione.
    // Fisso per tutta la durata di questa finestra VS Code → stesso thread Redis
    // (= memoria conversazionale persistente tra i messaggi).
    // Aprire una nuova finestra VS Code genera un nuovo SESSION_ID → utente separato.
    // -----------------------------------------------------------------------
    const SESSION_ID = `vscode-session-${Date.now()}`;
    console.log(`STM32 AI: session attiva → ${SESSION_ID}`);

    // -----------------------------------------------------------------------
    // 1. REGISTRA IL CHAT PARTICIPANT
    // -----------------------------------------------------------------------
    // Questo handler viene chiamato da VS Code quando l'utente scrive "@stm32ai ..."
    // request: contiene il prompt dell'utente
    // stream: è il canale per inviare risposte progressive (testo, markdown, chip)
    // -----------------------------------------------------------------------
    const handler: vscode.ChatRequestHandler = async (request: vscode.ChatRequest, context: vscode.ChatContext, stream: vscode.ChatResponseStream, token: vscode.CancellationToken) => {

        try {
            // Feedback immediato per dire "Sto pensando..."
            stream.markdown('Contatto il Brain STM32... \n\n');

            // -----------------------------------------------------------------------
            // 2. PREPARAZIONE PAYLOAD
            // -----------------------------------------------------------------------
            // Costruiamo il JSON da mandare al nostro server Python (server.py)
            // L'API FastAPI si aspetta { messages: [...], context: {...} }
            const messages = [
                // Qui potremmo aggiungere la history della chat precedente se volessimo
                { role: 'user', content: request.prompt }
            ];

            // Recuperiamo info sul contesto (es. file aperto) se necessario
            // const activeEditor = vscode.window.activeTextEditor;

            // -----------------------------------------------------------------------
            // 3. CHIAMATA AL SERVER PYTHON (FastAPI)
            // -----------------------------------------------------------------------
            // Usiamo 'fetch' per una richiesta POST allo stream endpoint.
            // session_id è fisso per questa finestra (SESSION_ID), garantendo
            // continuità di memoria tra i messaggi dello stesso utente.
            const bodyPayload = JSON.stringify({
                messages: messages,
                context: {
                    // "activeFile": activeEditor?.document.fileName  // Esempio futuro
                },
                user_id: "michele",
                session_id: SESSION_ID
            });

            const response = await fetch(SERVER_URL, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: bodyPayload
            });

            if (!response.ok) {
                throw new Error(`Errore server: ${response.status} ${response.statusText}`);
            }

            if (!response.body) {
                throw new Error("Nessuna risposta dal server");
            }

            // -----------------------------------------------------------------------
            // 4. LETTURA DELLO STREAM (NDJSON)
            // -----------------------------------------------------------------------
            // Il server invia pezzi di JSON separati da newline.
            // Esempio:
            // {"type": "progress", "content": "Analizzando..."}
            // {"type": "markdown", "content": "Ciao **Michele**!"}
            // -----------------------------------------------------------------------
            const reader = response.body.getReader();
            const decoder = new TextDecoder("utf-8");
            let buffer = "";

            while (true) {
                // Controllo se l'utente ha cliccato "Cancel" nella UI di Chat
                if (token.isCancellationRequested) {
                    console.log("Richiesta cancellata dall'utente.");
                    break;
                }

                const { done, value } = await reader.read();
                if (done) break;

                const chunk = decoder.decode(value, { stream: true });
                buffer += chunk;

                // Splitta per gestire il "newline delimited JSON"
                const lines = buffer.split("\n");
                // L'ultima parte potrebbe essere incompleta, la rimettiamo nel buffer
                buffer = lines.pop() || "";

                for (const line of lines) {
                    if (!line.trim()) continue;

                    try {
                        // Pulizia eventuale prefisso SSE "data: "
                        const cleanLine = line.replace(/^data: /, "");
                        const data = JSON.parse(cleanLine) as ServerResponse;

                        // -----------------------------------------------------------------------
                        // 5. RENDERING NELLA CHAT UI
                        // -----------------------------------------------------------------------
                        if (data.type === 'markdown') {
                            // Aggiunge testo Markdown alla risposta
                            stream.markdown(data.content);
                        } else if (data.type === 'progress') {
                            // Mostra un bullet point di progresso (o API progress nativa se disponibile)
                            stream.markdown(`* *${data.content}*\n`);
                        } else if (data.type === 'error') {
                            stream.markdown(`\n> ❌ **Errore**: ${data.content}\n`);
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
