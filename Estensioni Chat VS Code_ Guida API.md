# **L'ecosistema delle API di AI in Visual Studio Code: Architettura, Integrazione e Sviluppo di Chat Participant e Language Model Tools**

L'evoluzione degli ambienti di sviluppo integrati (IDE) ha raggiunto un punto di svolta fondamentale con l'integrazione dell'intelligenza artificiale generativa direttamente nel flusso di lavoro del programmatore. Visual Studio Code, attraverso la sua architettura estensibile, ha introdotto un set di interfacce di programmazione (API) progettate per trasformare l'editor da un semplice strumento di scrittura in un collaboratore attivo e consapevole del contesto.1 Questo passaggio non rappresenta soltanto un'aggiunta incrementale di funzionalità, ma un cambiamento di paradigma nel modo in cui le estensioni interagiscono con il codice, l'utente e i modelli di linguaggio di grandi dimensioni (LLM). Le API di Chat di VS Code, i Chat Participants e le Language Model API costituiscono le fondamenta di questo nuovo ecosistema, permettendo agli sviluppatori di costruire assistenti specializzati che possono comprendere l'intento dell'utente, accedere alla semantica del workspace e agire in modo autonomo per risolvere problemi complessi.1

## **L'architettura dell'estendibilità AI e la centralizzazione dell'intelligenza**

L'approccio di Visual Studio Code all'estendibilità AI si basa sulla separazione tra l'interfaccia utente, la logica dell'estensione e il fornitore del modello di linguaggio. In precedenza, un'estensione che desiderava integrare capacità AI doveva gestire autonomamente le connessioni agli endpoint dei modelli, la gestione delle chiavi API e il monitoraggio del consumo di token. Questo portava a un'esperienza utente frammentata, con ogni estensione che richiedeva configurazioni diverse e potenzialmente introduceva rischi per la sicurezza dei dati.1 L'attuale architettura risolve questo problema centralizzando l'accesso ai modelli attraverso l'infrastruttura di VS Code, spesso mediata da GitHub Copilot. L'editor agisce come un orchestratore che fornisce un accesso sicuro e standardizzato, garantendo che le estensioni possano invocare capacità di ragionamento avanzate senza dover implementare l'intera pipeline di comunicazione con l'LLM.1  
Questo modello di "intelligenza come servizio interno all'IDE" garantisce una coerenza sistematica. Se un utente ha configurato un modello specifico o dispone di un abbonamento aziendale a un fornitore di AI, tutte le estensioni autorizzate possono beneficiare di tale potenza computazionale in modo trasparente.4 Dal punto di vista tecnico, l'estensione non comunica direttamente con i server esterni; invece, interagisce con il processo dell'Extension Host di VS Code, il quale instrada le richieste al modello selezionato dall'utente, gestendo contemporaneamente il consenso, la telemetria e il monitoraggio delle risorse.1 Tale meccanismo permette agli sviluppatori di concentrarsi sulla creazione di valore specifico per il dominio, lasciando all'IDE il compito di gestire la complessità dell'infrastruttura AI.

## **Chat Participants: Gli esperti di dominio nella vista Chat**

Il concetto di Chat Participant rappresenta l'elemento centrale dell'interazione utente nella nuova interfaccia di chat di VS Code. Un Chat Participant è un assistente specializzato, un "esperto di dominio" che può essere invocato esplicitamente dall'utente per gestire richieste relative a un particolare argomento o tecnologia.1 Mentre un modello di linguaggio generico può rispondere a domande ampie, un partecipante come @vscode possiede una conoscenza specifica delle API dell'editor, mentre un partecipante creato da terze parti potrebbe essere un esperto di un framework come React, di un database come MongoDB o di una pipeline CI/CD specifica.1

### **Meccanismi di registrazione e identificazione statica**

La creazione di un Chat Participant inizia con la sua definizione nel manifesto dell'estensione, il file package.json. Questa registrazione statica è fondamentale perché permette a VS Code di popolare l'interfaccia utente e i suggerimenti della chat prima ancora che l'estensione stessa venga attivata.1 Le proprietà definite in questa fase determinano come l'utente percepirà e interagirà con l'assistente.

| Proprietà | Descrizione Tecnica | Requisiti di Naming |
| :---- | :---- | :---- |
| id | Identificativo univoco globale per il partecipante all'interno dell'ecosistema VS Code. | Deve avere come prefisso il nome dell'estensione (es. my-ext.expert). 1 |
| name | Il termine breve utilizzato dall'utente per invocare il partecipante tramite il simbolo @. | Solo caratteri alfanumerici minuscoli, trattini e underscore. 1 |
| fullName | Il nome completo visualizzato nel titolo della risposta generata dal partecipante. | Si raccomanda l'uso del Title Case (es. "Code Tutor"). 1 |
| description | Una breve spiegazione della funzione del partecipante, usata come testo segnaposto. | Sentence case, senza punteggiatura finale, lunghezza ridotta per la UI. 1 |
| isSticky | Flag booleano che determina se il partecipante deve rimanere attivo dopo una risposta. | Impostare su true per facilitare conversazioni iterative multi-turno. 1 |

La proprietà isSticky ha un impatto significativo sull'ergonomia. In un flusso di lavoro tipico, se un utente sta lavorando intensamente con un assistente per il refactoring, non dovrebbe essere costretto a digitare @refactor prima di ogni messaggio. La persistenza del partecipante trasforma la sessione di chat in un dialogo focalizzato, dove il contesto del partecipante rimane in primo piano finché l'utente non decide di cambiare interlocutore.1

### **Implementazione del Request Handler e logica di attivazione**

Una volta registrato nel manifesto, il comportamento dinamico del partecipante viene definito nel codice TypeScript dell'estensione durante la fase di attivazione. La funzione vscode.chat.createChatParticipant viene utilizzata per collegare l'ID statico definito nel package.json a un gestore di richieste (ChatRequestHandler).1 Questo gestore è una funzione asincrona che viene invocata ogni volta che l'utente invia un messaggio indirizzato al partecipante.  
Il gestore riceve quattro parametri fondamentali che costituiscono il contesto dell'interazione:

1. request: Contiene il testo digitato dall'utente (prompt), eventuali comandi slash e informazioni sulla posizione della chat (ad esempio, se l'utente sta interagendo dalla barra laterale o da una chat in linea nell'editor).1  
2. context: Fornisce l'accesso alla cronologia della conversazione corrente, permettendo al partecipante di comprendere i riferimenti ai messaggi precedenti.1  
3. stream: Un oggetto fondamentale per la gestione della risposta, che consente di inviare contenuti all'utente in modalità streaming, migliorando la percezione di reattività dell'interfaccia.1  
4. token: Un token di cancellazione che deve essere monitorato per interrompere le operazioni computazionalmente onerose nel caso in cui l'utente chiuda la chat o avvii una nuova richiesta.1

## **Language Model API: Il motore computazionale dell'intelligenza**

Se i Chat Participants definiscono "chi" risponde, la Language Model API definisce "come" viene generata l'intelligenza. Questa API espone le capacità degli LLM alle estensioni, permettendo loro di processare il linguaggio naturale in modi che vanno oltre la semplice risposta in chat, includendo suggerimenti per rinominare variabili, spiegazioni di errori di compilazione o generazione di documentazione.1

### **Selezione dinamica e gestione dei modelli**

L'accesso ai modelli linguistici non è un'operazione statica. Poiché l'ecosistema di VS Code supporta diversi fornitori e versioni di modelli (come GPT-4o, Claude 3.5 Sonnet, o1-mini), le estensioni devono scoprire quali modelli sono disponibili nell'ambiente dell'utente.1 Attraverso vscode.lm.selectChatModels, lo sviluppatore può richiedere modelli che corrispondano a determinati criteri di filtraggio, come la famiglia del modello o il produttore.1

| Famiglia Modello | Capacità Tipiche | Casi d'Uso Consigliati |
| :---- | :---- | :---- |
| gpt-4o | Ragionamento avanzato, visione, multilingua. | Task complessi di architettura e analisi del codice. 1 |
| gpt-4o-mini | Velocità elevata, latenza ridotta, costi contenuti. | Suggerimenti in tempo reale, spiegazioni brevi. 1 |
| claude-3.5-sonnet | Eccellente comprensione del contesto, stile naturale. | Scrittura di documentazione e refactoring logico. 1 |
| o1-mini | Ottimizzato per il ragionamento logico-matematico. | Risoluzione di bug complessi e algoritmi. 1 |

È fondamentale implementare una logica di selezione "difensiva". Se un'estensione richiede specificamente gpt-4o ma l'utente ha accesso solo a modelli meno potenti o di fornitori diversi, l'estensione deve essere in grado di degradare le proprie funzionalità con grazia o informare l'utente sulla necessità di un modello superiore.1 La selezione del modello dovrebbe sempre avvenire in risposta a un'azione avviata dall'utente, poiché VS Code potrebbe richiedere un'autenticazione o un consenso esplicito prima di concedere l'accesso alle API del modello.1

### **Costruzione dei Prompt e gestione dei token**

La qualità del risultato fornito dall'AI dipende intrinsecamente dal "Prompt Engineering". VS Code fornisce la classe LanguageModelChatMessage per rappresentare i messaggi inviati al modello. Attualmente, il sistema supporta principalmente due tipi di ruoli: User per le istruzioni e le richieste, e Assistant per includere la cronologia delle risposte precedenti come contesto.1 Sebbene molti modelli supportino nativamente i "System Messages" per definire la personalità e i vincoli del modello, in VS Code queste istruzioni vengono spesso integrate nel primo messaggio utente o gestite tramite wrapper di alto livello come la libreria @vscode/prompt-tsx.1  
La gestione dei token è un altro aspetto critico. Ogni modello ha un limite massimo di token in input (ad esempio, 64K per GPT-4o). Se un'estensione tenta di inviare l'intero contenuto di un workspace come contesto, la richiesta fallirà.1 Gli sviluppatori devono quindi implementare strategie di potatura del contesto, selezionando solo le informazioni più rilevanti per la richiesta corrente. La libreria TSX per i prompt facilita questo compito permettendo di dichiarare la gerarchia di importanza dei vari componenti del prompt e lasciando alla libreria il compito di tagliare le parti meno prioritarie per rientrare nei limiti del modello selezionato.1

## **Streaming e interfacce ricche: La potenza di ChatResponseStream**

La comunicazione tra l'estensione e l'utente nella chat di VS Code non deve essere vista come uno scambio di semplici stringhe di testo. L'interfaccia ChatResponseStream trasforma la risposta in un flusso dinamico di "parti" diverse, ognuna con una funzione specifica nell'interfaccia utente.9

### **Tipologie di contenuti e interazioni**

L'utilizzo dello streaming non serve solo a mostrare le parole mentre vengono generate, ma a fornire un feedback continuo sull'avanzamento dei compiti complessi.

* **Markdown**: È il componente principale per il testo e i frammenti di codice. Utilizzando stream.markdown(), l'estensione può formattare la risposta con grassetti, elenchi e blocchi di codice evidenziati sintatticamente.1  
* **Progress**: Indispensabile per i task che richiedono tempo (come la ricerca nel workspace o l'invocazione di API esterne). stream.progress() visualizza un indicatore di attività con un messaggio testuale, rassicurando l'utente che l'assistente sta lavorando e non è bloccato.1  
* **Button**: Permette di inserire pulsanti cliccabili che attivano comandi di VS Code. Ad esempio, una risposta che suggerisce una correzione potrebbe includere un pulsante "Applica Correzione" che apre l'editor e inserisce il codice suggerito.1  
* **References e Anchors**: Consentono di creare collegamenti interattivi a file o posizioni specifiche nel codice. Utilizzando ChatResponseReferencePart, l'estensione può mostrare quali file ha consultato per generare la risposta, permettendo all'utente di navigare istantaneamente alla fonte.9  
* **File Trees**: Per i comandi che generano nuove strutture di progetto o directory, stream.push(new ChatResponseFileTreePart(...)) visualizza un albero di file interattivo, fornendo una visione d'insieme immediata del risultato.9

Questa ricchezza di output è ciò che differenzia un'estensione di VS Code da una semplice interfaccia web per LLM. L'assistente è "dentro" l'editor e può interagire con i suoi elementi nativi, rendendo la transizione tra la conversazione e l'azione quasi istantanea.

## **Language Model Tools: Verso l'AI Agentica**

Una delle innovazioni più significative introdotte recentemente è il concetto di "Agent mode" e l'uso dei Language Model Tools. Mentre i Chat Participants sono esperti con cui parlare, i Tools sono strumenti che l'IA può decidere di impugnare per compiere azioni concrete.2 In modalità agente, l'LLM non risponde solo all'utente, ma pianifica una serie di passaggi e invoca autonomamente questi strumenti per raccogliere informazioni o modificare il workspace.11

### **Differenze concettuali e sinergie**

È fondamentale distinguere tra un partecipante e un tool per progettare correttamente l'estensione. Un Chat Participant ha il controllo totale sulla conversazione; riceve il prompt, decide quale modello usare e come formattare la risposta finale.1 Un Language Model Tool, invece, è un'unità funzionale atomica (ad esempio, "leggi il contenuto di un file" o "esegui un test unitario") che viene registrata nell'editor e messa a disposizione dell'orchestratore AI.2

| Caratteristica | Chat Participant | Language Model Tool |
| :---- | :---- | :---- |
| **Punto di Invocazione** | Menziome esplicita con @. 1 | Invocazione automatica dall'LLM basata sull'intento. 3 |
| **Simbolo UI** | @ (es. @terminal). 6 | \# (es. \#codebase o \#file). 6 |
| **Responsabilità** | Gestione end-to-end della conversazione. 1 | Esecuzione di un sotto-task specifico. 2 |
| **Configurazione** | Definizione dei comandi slash (/). 1 | Definizione dello schema di input (JSON Schema). 3 |

### **Il ciclo di invocazione dei tool (Tool-calling flow)**

Quando un'estensione contribuisce con un tool, essa non scrive il codice che "chiama" il tool. Invece, fornisce a VS Code una descrizione del tool e uno schema dei parametri accettati.3 Il processo segue un'iterazione precisa:

1. **Analisi dell'Intento**: Copilot riceve il prompt dell'utente e identifica quali tool (nativi o di estensioni) potrebbero essere utili per risolverlo.3  
2. **Generazione dei Parametri**: L'LLM non esegue il codice, ma genera un oggetto JSON contenente i valori per i parametri definiti nello schema del tool.3  
3. **Esecuzione Protetta**: VS Code riceve la richiesta di invocazione del tool. Se il tool è considerato sensibile (come l'esecuzione di comandi nel terminale), l'IDE mostra un dialogo di conferma all'utente.13  
4. **Integrazione del Risultato**: L'output del tool viene rimandato all'LLM, che lo utilizza come nuovo contesto per decidere se il compito è concluso o se sono necessari ulteriori passaggi.3

Questo approccio permette di creare workflow "agentici" dove l'IA può, ad esempio, cercare un errore in un log, trovare il file corrispondente, applicare una patch e verificare la correzione eseguendo i test, il tutto orchestrando strumenti diversi in sequenza.11

## **Variabili di Chat e gestione del contesto tramite \# Mentions**

Un assistente AI è utile solo se può accedere alle informazioni corrette. VS Code introduce il concetto di "Chat Variables" e "\# Mentions" per permettere all'utente e alle estensioni di iniettare contesto specifico nella conversazione.14 Le menzioni con il cancelletto funzionano come scorciatoie per entità del workspace che l'LLM deve "leggere" per fornire una risposta accurata.

### **Tipi di contesto e strumenti di ricerca**

L'IDE fornisce diverse variabili di contesto integrate che coprono le esigenze più comuni:

* \#file: Permette di includere il contenuto di un file specifico aperto o cercato nel workspace.6  
* \#codebase: È uno degli strumenti più potenti, poiché attiva una ricerca semantica (spesso basata su embedding) per trovare le parti del codice più rilevanti rispetto alla domanda dell'utente, superando i limiti della ricerca testuale tradizionale.6  
* \#terminal: Include l'output recente del terminale, fondamentale per chiedere spiegazioni su errori di compilazione o di esecuzione.6  
* \#selection: Si riferisce al blocco di codice attualmente selezionato dall'utente nell'editor attivo.6

Le estensioni possono contribuire con variabili personalizzate, permettendo all'utente di includere dati provenienti da fonti esterne (come ticket Jira, documentazione interna o dati di telemetria) direttamente nel prompt della chat.15 Questo meccanismo garantisce che il modello non lavori nel vuoto, ma sia ancorato alla realtà specifica del progetto in esame.

### **Implementazione della disambiguazione**

Per rendere l'interfaccia ancora più intelligente, VS Code supporta la "Participant Detection". Se un utente pone una domanda senza specificare un partecipante con @, l'IDE analizza la query e cerca di capire se uno dei partecipanti registrati è particolarmente adatto a rispondere.1 Questo avviene grazie alla proprietà disambiguation nel package.json, dove lo sviluppatore può definire:

* **Category**: Il dominio di competenza (es. "database", "testing").  
* **Description**: Una spiegazione dettagliata dei tipi di domande gestite.  
* **Examples**: Una lista di domande campione (es. "Come connetto MongoDB?") che aiutano il sistema di routing a mappare l'intento dell'utente al partecipante corretto.1

## **UX Guidelines: Progettare per la coerenza e la fiducia**

L'integrazione dell'AI nell'interfaccia utente richiede un'attenzione particolare alla trasparenza e alla facilità d'uso. VS Code fornisce linee guida rigorose per assicurare che le estensioni mantengano uno standard qualitativo elevato e non disorientino l'utente.1

### **Convenzioni di denominazione e visibilità**

La coerenza visiva è garantita da regole precise sulla presentazione dei partecipanti e dei comandi slash. Il nome del partecipante (quello dopo @) deve essere sempre in minuscolo per uniformità con i comandi di sistema, mentre il nome completo visualizzato nelle risposte deve usare il Title Case e riflettere il branding dell'estensione o dell'azienda.1 La descrizione deve essere breve e scritta in "Sentence case", evitando la punteggiatura finale, per fungere efficacemente da segnaposto nel campo di input della chat.1

### **Interattività e Follow-up**

Un errore comune nello sviluppo di bot e assistenti è quello di fornire una risposta "morta" che non invita all'azione successiva. VS Code incoraggia l'uso delle "Follow-up Questions". Dopo una risposta, il partecipante può suggerire all'utente i passi logici successivi. Ad esempio, dopo aver spiegato un errore, potrebbe suggerire: "Vuoi che generi un test unitario per riprodurre questo bug?".1 Questi suggerimenti dovrebbero essere formulati come domande naturali o istruzioni chiare, rendendo l'interazione fluida e guidata.1

### **Trasparenza nelle operazioni di background**

Poiché l'elaborazione dei modelli linguistici può essere lenta e soggetta a errori, è vitale mantenere l'utente informato. L'uso dei messaggi di progresso e la visualizzazione chiara dei tool invocati permettono all'utente di capire cosa sta succedendo "sotto il cofano".1 Inoltre, VS Code implementa un sistema di feedback integrato (pollici su/giù) che le estensioni dovrebbero monitorare tramite l'evento onDidReceiveFeedback. Questi dati sono essenziali per calcolare metriche di successo e rifinire i prompt nel tempo.1

## **Language Model Chat Provider: Portare il proprio modello in VS Code**

Mentre la maggior parte delle estensioni consuma modelli esistenti, VS Code offre anche la possibilità di diventare un fornitore di modelli tramite la **Language Model Chat Provider API**.5 Questa funzionalità è pensata per aziende che hanno sviluppato modelli proprietari o che desiderano integrare modelli locali per motivi di privacy e conformità.

### **Processo di registrazione e implementazione**

Per registrare un fornitore di modelli, l'estensione deve dichiarare il punto di contribuzione languageModelChatProviders nel package.json, specificando un identificativo del fornitore (vendor) e un nome visualizzato.5 Successivamente, deve implementare l'interfaccia LanguageModelChatProvider, che richiede tre metodi fondamentali:

1. provideLanguageModelChatInformation: Fornisce a VS Code l'elenco dei modelli disponibili, incluse le loro capacità (come il supporto alle immagini o al tool calling) e i limiti di token.5  
2. provideLanguageModelChatResponse: È il metodo che gestisce l'effettiva generazione della risposta, ricevendo l'array di messaggi e restituendo un flusso di parti della risposta.5  
3. provideTokenCount: Implementa la logica per contare i token in base al tokenizer specifico del modello, permettendo all'IDE e alle altre estensioni di gestire correttamente i limiti di contesto.5

Questa API apre la porta a scenari di "Local AI", dove un'estensione può far girare un modello come Llama 3 localmente sulla macchina dell'utente tramite strumenti come Ollama, esponendolo però attraverso le interfacce standard di VS Code in modo che tutte le altre estensioni possano utilizzarlo come se fosse un modello cloud.1

## **Tutorial Pratico: Creare un "Code Tutor" Chat Participant**

Per comprendere appieno l'interazione tra questi componenti, analizziamo il processo di creazione di un partecipante tutorial chiamato @tutor. L'obiettivo di questo assistente è guidare gli studenti senza fornire risposte dirette, agendo come un mentore pedagogico.8

### **Passaggio 1: Generazione del Progetto**

Il punto di partenza è l'uso dello strumento Yeoman con il generatore ufficiale di VS Code.

Bash

npx \--package yo \--package generator-code \-- yo code

Durante la configurazione, si seleziona "New Extension (TypeScript)" e si assegna il nome "Code Tutor".8

### **Passaggio 2: Definizione del Manifesto**

Nel file package.json, si aggiunge il punto di contributo per il partecipante. Notare l'uso di isSticky per mantenere la sessione attiva.

JSON

"contributes": {  
  "chatParticipants":  
    }  
  \]  
}

8

### **Passaggio 3: Implementazione del Prompt di Base**

Nel file extension.ts, si definisce la personalità del tutor. Questo "System Prompt" istruisce l'IA a non dare mai la soluzione completa, ma a porre domande stimolanti e fornire piccoli indizi.8 È qui che risiede l'intelligenza specifica del dominio.

### **Passaggio 4: Gestione della Cronologia (Context)**

Per permettere una conversazione reale, l'estensione deve recuperare i messaggi precedenti.

TypeScript

const history \= context.history.map(turn \=\> {  
    if (turn instanceof vscode.ChatRequestTurn) {  
        return vscode.LanguageModelChatMessage.User(turn.prompt);  
    } else {  
        return vscode.LanguageModelChatMessage.Assistant(turn.response.map(r \=\> r.value).join(''));  
    }  
});

Questi messaggi vengono poi inseriti nell'array inviato alla Language Model API per garantire che il modello ricordi cosa è stato detto in precedenza.8

### **Passaggio 5: Streaming della Risposta**

L'handler invia la richiesta al modello e instrada i frammenti di testo allo stream della chat.

TypeScript

const \[model\] \= await vscode.lm.selectChatModels({ family: 'gpt-4o' });  
const request \= model.sendRequest(messages, {}, token);  
for await (const fragment of request.text) {  
    stream.markdown(fragment);  
}

Questo garantisce che l'utente veda la risposta generarsi in tempo reale, mantenendo l'interazione vivace.4

## **Evoluzione Recente e Prospettive (Gennaio 2026\)**

L'ecosistema delle API AI di Visual Studio Code è in rapida evoluzione. Gli ultimi aggiornamenti di gennaio 2026 hanno introdotto diverse ottimizzazioni che migliorano l'usabilità e la potenza degli strumenti agentici.17

### **Novità dell'aggiornamento 1.109**

Le modifiche recenti indicano una tendenza verso una maggiore autonomia degli agenti e una migliore gestione del terminale.

* **Gestione del Terminale**: È stato introdotto un nuovo strumento per attendere il completamento dei comandi in background, permettendo agli agenti AI di procedere con la logica solo dopo che un'operazione lunga (come un npm install o una build) è terminata con successo.17  
* **Interfaccia del Tool Picker**: Il selettore degli strumenti è stato spostato in una posizione dedicata, rendendo più facile per l'utente abilitare o disabilitare specifici tool (come il fetch web o l'accesso al file system) per la sessione corrente.13  
* **Performance delle Sessioni Lunghe**: Sono stati apportati miglioramenti significativi alla gestione della memoria per le chat che crescono in lunghezza, riducendo la latenza e migliorando la reattività anche dopo centinaia di messaggi.17  
* **Supporto a Claude Code**: È stata integrata la capacità di gestire sub-agenti, permettendo a un agente principale di delegare compiti a unità specializzate in background.17

### **L'integrazione di MCP (Model Context Protocol)**

Un'altra direzione fondamentale è l'adozione dell'MCP, uno standard aperto che permette ai modelli AI di connettersi a dati e strumenti esterni attraverso un'interfaccia unificata.13 VS Code ora permette di registrare server MCP che espongono tool specializzati (ad esempio, per interrogare database SQL o navigare su internet) che vengono automaticamente rilevati dall'agente di chat.13 Questo significa che gli sviluppatori possono creare strumenti che funzionano non solo in VS Code, ma in qualsiasi client compatibile con MCP, promuovendo un'interoperabilità senza precedenti nell'ecosistema AI.2

## **Considerazioni su Sicurezza e Sviluppo Responsabile**

L'accesso a capacità AI avanzate solleva questioni cruciali riguardanti la privacy dei dati e la sicurezza del codice. VS Code implementa diverse barriere protettive per garantire che l'automazione non comprometta l'integrità del sistema dell'utente.

### **Approvazione dei Comandi e Sandbox**

Per impostazione predefinita, le azioni che modificano lo stato del sistema (come l'esecuzione di comandi nel terminale) richiedono l'approvazione esplicita dell'utente.13 Sebbene sia possibile configurare l'auto-approvazione per comandi considerati "sicuri", l'IDE avverte costantemente dei rischi di "Prompt Injection", dove un file maligno nel workspace potrebbe contenere istruzioni nascoste atte a ingannare l'agente AI facendogli eseguire comandi dannosi.13  
Gli sviluppatori di estensioni devono aderire alle policy di utilizzo accettabile di fornitori come GitHub Copilot. È fondamentale evitare l'uso delle API AI per scopi proibiti, come la generazione di codice malevolo o l'estrazione non autorizzata di dati personali.1 La trasparenza rimane il pilastro fondamentale: l'utente deve sempre essere consapevole di quando l'AI è attiva e di quali dati sta consultando.

## **Sintesi e Conclusione**

Le API di AI in Visual Studio Code rappresentano una delle infrastrutture più sofisticate mai messe a disposizione dei creatori di estensioni. Attraverso l'uso sinergico di Chat Participants, Language Model Tools e una gestione intelligente del contesto, è possibile costruire assistenti che non si limitano a rispondere a domande, ma partecipano attivamente al processo creativo della programmazione.1  
L'architettura attuale bilancia abilmente tre necessità contrastanti: la potenza computazionale degli LLM, la coerenza e la semplicità dell'interfaccia utente, e la sicurezza dell'ambiente di esecuzione.1 Per gli sviluppatori, questo significa avere accesso a un framework che astrae la complessità dei modelli linguistici, permettendo di focalizzarsi sull'integrazione profonda con l'editor e sulla risoluzione dei problemi specifici del proprio dominio.1  
Con l'avanzare delle capacità agentiche e l'adozione di protocolli aperti come MCP, il ruolo delle estensioni in VS Code continuerà a evolversi. L'IDE non è più solo un posto dove scrivere codice, ma un ecosistema di intelligenze collaborate dove ogni strumento contribuisce con la propria conoscenza specifica per rendere lo sviluppo software più veloce, più sicuro e, in ultima analisi, più umano.

#### **Bibliografia**

1. Chat Participant API \- Visual Studio Code, accesso eseguito il giorno gennaio 29, 2026, [https://code.visualstudio.com/api/extension-guides/ai/chat](https://code.visualstudio.com/api/extension-guides/ai/chat)  
2. AI extensibility in VS Code | Visual Studio Code Extension API, accesso eseguito il giorno gennaio 29, 2026, [https://code.visualstudio.com/api/extension-guides/ai/ai-extensibility-overview](https://code.visualstudio.com/api/extension-guides/ai/ai-extensibility-overview)  
3. Language Model Tool API | Visual Studio Code Extension API, accesso eseguito il giorno gennaio 29, 2026, [https://code.visualstudio.com/api/extension-guides/ai/tools](https://code.visualstudio.com/api/extension-guides/ai/tools)  
4. Language Model API \- Visual Studio Code, accesso eseguito il giorno gennaio 29, 2026, [https://code.visualstudio.com/api/extension-guides/ai/language-model](https://code.visualstudio.com/api/extension-guides/ai/language-model)  
5. Language Model Chat Provider API \- Visual Studio Code, accesso eseguito il giorno gennaio 29, 2026, [https://code.visualstudio.com/api/extension-guides/ai/language-model-chat-provider](https://code.visualstudio.com/api/extension-guides/ai/language-model-chat-provider)  
6. GitHub Copilot in VS Code cheat sheet, accesso eseguito il giorno gennaio 29, 2026, [https://code.visualstudio.com/docs/copilot/reference/copilot-vscode-features](https://code.visualstudio.com/docs/copilot/reference/copilot-vscode-features)  
7. vscode/package.json at main · mongodb-js/vscode \- GitHub, accesso eseguito il giorno gennaio 29, 2026, [https://github.com/mongodb-js/vscode/blob/main/package.json](https://github.com/mongodb-js/vscode/blob/main/package.json)  
8. Build a code tutorial chat participant with the Chat API \- Visual Studio Code, accesso eseguito il giorno gennaio 29, 2026, [https://code.visualstudio.com/api/extension-guides/ai/chat-tutorial](https://code.visualstudio.com/api/extension-guides/ai/chat-tutorial)  
9. VS Code API | Visual Studio Code Extension API, accesso eseguito il giorno gennaio 29, 2026, [https://code.visualstudio.com/api/references/vscode-api\#ChatResponseStream](https://code.visualstudio.com/api/references/vscode-api#ChatResponseStream)  
10. VS Code API | Visual Studio Code Extension API, accesso eseguito il giorno gennaio 29, 2026, [https://code.visualstudio.com/api/references/vscode-api](https://code.visualstudio.com/api/references/vscode-api)  
11. Get started with chat in VS Code, accesso eseguito il giorno gennaio 29, 2026, [https://code.visualstudio.com/docs/copilot/chat/copilot-chat](https://code.visualstudio.com/docs/copilot/chat/copilot-chat)  
12. Planning in VS Code chat, accesso eseguito il giorno gennaio 29, 2026, [https://code.visualstudio.com/docs/copilot/chat/chat-planning](https://code.visualstudio.com/docs/copilot/chat/chat-planning)  
13. Use tools in chat \- Visual Studio Code, accesso eseguito il giorno gennaio 29, 2026, [https://code.visualstudio.com/docs/copilot/chat/chat-tools](https://code.visualstudio.com/docs/copilot/chat/chat-tools)  
14. Manage context for AI \- Visual Studio Code, accesso eseguito il giorno gennaio 29, 2026, [https://code.visualstudio.com/docs/copilot/chat/copilot-chat-context](https://code.visualstudio.com/docs/copilot/chat/copilot-chat-context)  
15. Chat Participant API | Visual Studio Code Extension API, accesso eseguito il giorno gennaio 29, 2026, [https://code.visualstudio.com/api/extension-guides/ai/chat\#chat-variables](https://code.visualstudio.com/api/extension-guides/ai/chat#chat-variables)  
16. src/vs/workbench/contrib/chat/common/chatAgents.ts at main \- sarvex/vscode \- CodeFactor, accesso eseguito il giorno gennaio 29, 2026, [https://www.codefactor.io/repository/github/sarvex/vscode/source/main/src/vs/workbench/contrib/chat/common/chatAgents.ts](https://www.codefactor.io/repository/github/sarvex/vscode/source/main/src/vs/workbench/contrib/chat/common/chatAgents.ts)  
17. January 2026 Insiders (version 1.109) \- Visual Studio Code, accesso eseguito il giorno gennaio 29, 2026, [https://code.visualstudio.com/updates/v1\_109](https://code.visualstudio.com/updates/v1_109)



-----------
##**Si può integrare un progetto sviluppato in LangGraph all'interno di vsCode tramite queste estensioni chat VScode? **+

Sì, è assolutamente possibile integrare un progetto sviluppato in **LangGraph** (specificamente tramite la versione JavaScript/TypeScript, **LangGraph.js**) all'interno di un'estensione per Visual Studio Code. Questa integrazione permette di trasformare un workflow complesso, ciclico e multi-agente in un assistente interattivo direttamente accessibile dalla barra laterale o dall'editor.

L'integrazione avviene principalmente attraverso la creazione di un **Chat Participant** che funge da interfaccia tra l'utente e la logica del grafo.

### Perché l'integrazione è efficace

1. **Compatibilità del Runtime:** Le estensioni di VS Code vengono eseguite in un processo chiamato *Extension Host*, che utilizza **Node.js**. Poiché LangGraph.js è una libreria Node.js nativa, può essere importata ed eseguita direttamente all'interno dell'estensione senza necessità di server esterni.
2. **Orchestrazione vs Risposta:** Mentre un semplice modello di linguaggio (LLM) risponde in modo lineare, LangGraph permette al tuo Chat Participant di eseguire cicli di "ragionamento-azione-osservazione" prima di restituire la risposta finale all'utente.
3. **Accesso alle API di VS Code:** Poiché il codice del grafo gira nell'estensione, i nodi del tuo grafo possono invocare direttamente i comandi di VS Code, leggere file dal workspace o interagire con il terminale.



### Come realizzare l'integrazione

Per integrare il tuo progetto LangGraph, devi seguire questi passaggi tecnici:

#### 1. Configurazione del Chat Participant

Nel file `package.json` dell'estensione, definisci il partecipante che l'utente invocherà (ad esempio `@my-graph-agent`) :

```json
"contributes": {
  "chatParticipants":
}

```

#### 2. Implementazione del Request Handler

Nel file `extension.ts`, crea un gestore che riceve il prompt dell'utente e lo passa al tuo grafo LangGraph. È fondamentale utilizzare le capacità di **streaming** per non far attendere l'utente durante l'esecuzione dei nodi.

* **Scelta della modalità di streaming:** LangGraph supporta diverse modalità. Per una chat, la modalità `messages` è ideale perché permette di inviare i token del modello all'utente in tempo reale.
* **Bridge dello Stream:** Devi convogliare i chunk generati da `graph.stream()` verso l'oggetto `vscode.ChatResponseStream`.

#### 3. Gestione dello Stato e del Contesto

Puoi iniettare il contesto di VS Code (come i file menzionati con `#file` o il codice selezionato) direttamente nello stato iniziale del tuo grafo (`GraphState`). Questo permette al grafo di "vedere" il codice su cui l'utente sta lavorando.

### Scenari di integrazione avanzata

* **Agent Mode (Tools):** Invece di un intero grafo, puoi esporre singoli nodi o funzioni del tuo progetto LangGraph come **Language Model Tools**. In questo modo, sarà l'istanza principale di Copilot a decidere quando invocare il tuo "strumento LangGraph" tramite il simbolo `#`.


* **Integrazione Python:** Se il tuo progetto LangGraph è scritto in Python, l'integrazione diretta è più complessa poiché l'estensione VS Code è in JS. In questo caso, è necessario esporre il grafo Python tramite un server locale (usando il *LangGraph CLI* o *LangSmith Agent Server*) e far sì che l'estensione VS Code agisca come client tramite `langgraph-sdk`.

### Esempi esistenti

Esistono già estensioni nel Marketplace, come **LangGraph Visualizer**, che dimostrano la profonda sinergia tra questo framework e l'IDE, permettendo di visualizzare l'esecuzione del grafo, impostare breakpoint sui nodi e monitorare i cambiamenti di stato in tempo reale durante il debug.



