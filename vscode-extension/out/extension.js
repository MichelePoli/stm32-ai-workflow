"use strict";
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.activate = activate;
exports.deactivate = deactivate;
const vscode = require("vscode");
const SERVER_URL = 'http://127.0.0.1:8000/stream';
function activate(context) {
    console.log('Attivazione estensione STM32 AI Assistant');
    // -----------------------------------------------------------------------
    // 1. REGISTRA IL CHAT PARTICIPANT
    // -----------------------------------------------------------------------
    // Questo handler viene chiamato da VS Code quando l'utente scrive "@stm32ai ..."
    // request: contiene il prompt dell'utente
    // stream: è il canale per inviare risposte progressive (testo, markdown, chip)
    // -----------------------------------------------------------------------
    const handler = (request, context, stream, token) => __awaiter(this, void 0, void 0, function* () {
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
            // Questo endpoint restituisce una risposta "Transfer-Encoding: chunked" (NDJSON)
            // Passiamo ID utente e sessione (hardcoded per ora, ma espandibile)
            // Generiamo un session_id univoco per non avere conflitti in Redis se apriamo più VS Code
            const uniqueSessionId = `vscode-session-${Date.now()}`;
            const bodyPayload = JSON.stringify({
                messages: messages,
                context: {
                // "activeFile": activeEditor?.document.fileName  // Esempio futuro
                },
                user_id: "michele",
                session_id: uniqueSessionId
            });
            const response = yield fetch(SERVER_URL, {
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
                const { done, value } = yield reader.read();
                if (done)
                    break;
                const chunk = decoder.decode(value, { stream: true });
                buffer += chunk;
                // Splitta per gestire il "newline delimited JSON"
                const lines = buffer.split("\n");
                // L'ultima parte potrebbe essere incompleta, la rimettiamo nel buffer
                buffer = lines.pop() || "";
                for (const line of lines) {
                    if (!line.trim())
                        continue;
                    try {
                        // Pulizia eventuale prefisso SSE "data: "
                        const cleanLine = line.replace(/^data: /, "");
                        const data = JSON.parse(cleanLine);
                        // -----------------------------------------------------------------------
                        // 5. RENDERING NELLA CHAT UI
                        // -----------------------------------------------------------------------
                        if (data.type === 'markdown') {
                            // Aggiunge testo Markdown alla risposta
                            stream.markdown(data.content);
                        }
                        else if (data.type === 'progress') {
                            // Mostra un bullet point di progresso (o API progress nativa se disponibile)
                            stream.markdown(`* *${data.content}*\n`);
                        }
                        else if (data.type === 'error') {
                            stream.markdown(`\n> ❌ **Errore**: ${data.content}\n`);
                        }
                        else if (data.type === 'debug') {
                            console.log(`[SERVER DEBUG] ${data.content}`);
                        }
                    }
                    catch (e) {
                        console.warn("Errore parsing JSON chunk:", line, e);
                    }
                }
            }
        }
        catch (err) {
            if (err instanceof Error) {
                stream.markdown(`\n**Errore di connessione**: Impossibile contattare il server Python.\nAssicurati che ` + "`server.py`" + ` sia in esecuzione su porta 8000.\n\nDettaglio: ${err.message}`);
            }
            else {
                stream.markdown(`\n**Errore sconosciuto** durante la comunicazione con l'backend.`);
            }
        }
    });
    const helper = vscode.chat.createChatParticipant('stm32-ai.assistant', handler);
    helper.iconPath = new vscode.ThemeIcon('chip'); // Icona appropriata
    context.subscriptions.push(helper);
}
function deactivate() { }
//# sourceMappingURL=extension.js.map