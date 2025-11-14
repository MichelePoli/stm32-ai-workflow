# ============================================================================
# WORKFLOW 4: WEB RESEARCH E RICERCA INFORMAZIONI ONLINE
# ============================================================================
# Modulo dedicato alla ricerca online di informazioni su board STM32, modelli AI
# e best practices di ottimizzazione
#
# Responsabilità:
#   - Classificazione tipo di ricerca (ai_model, board_selection, optimization, documentation)
#   - Esecuzione ricerche via Google Search / LLM
#   - Formattazione risultati per l'utente
#
# Dipendenze: langgraph, langchain, agno.tools, requests

import os
import logging
from typing import Literal, Optional

from langchain_ollama import ChatOllama
from agno.models.groq import Groq
from agno.agent import Agent
from agno.tools.googlesearch import GoogleSearchTools
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from src.assistant.configuration import Configuration
from src.assistant.state import MasterState

logger = logging.getLogger(__name__)

# ============================================================================
# EXTRACTION SCHEMAS - WORKFLOW 4
# ============================================================================

class SearchClassification(BaseModel):
    """Classificazione del tipo di ricerca richiesta"""
    search_type: Literal[
        "ai_model",
        "board_selection",
        "optimization",
        "documentation",
        "none"
    ] = Field(description="Tipo di ricerca richiesta dall'utente")
    
    search_query: str = Field(
        description="Query specifica da cercare online"
    )
    
    reasoning: str = Field(
        description="Spiegazione della classificazione"
    )


# ============================================================================
# EXTRACTION INSTRUCTIONS - WORKFLOW 4
# ============================================================================

search_classification_instructions = """Sei un classificatore di ricerche per un sistema STM32 + AI.

Analizza la richiesta dell'utente e classifica il tipo di ricerca necessaria:

1. **ai_model**: Cercare modelli di AI compatibili con STM32
   - Esempi: "quali modelli CNN leggeri per STM32?", "MobileNet vs SqueezeNet per STM32"
   - Keywords: modello, network, CNN, RNN, rete neurale, intelligenza artificiale
   
2. **board_selection**: Aiutare a scegliere una board STM32
   - Esempi: "quale STM32 per un progetto con AI?", "STM32H7 vs STM32F4"
   - Keywords: board, scelta, quale, differenza, confronto, memoria, performance
   
3. **optimization**: Ottimizzazione e compressione AI su STM32
   - Esempi: "come comprimere il modello?", "quantizzazione su STM32"
   - Keywords: ottimizzazione, quantizzazione, compressione, pruning, embedded
   
4. **documentation**: Documentazione generale, tutorial, best practices
   - Esempi: "come compilare per STM32?", "guide STEdgeAI", "tutorial"
   - Keywords: documentazione, tutorial, guide, come, best practice, risorse

5. **none**: Nessuno dei precedenti o richiesta non valida
   - Esempi: "ciao", "non so", richieste completamente non correlate

Rispondi SEMPRE in formato JSON con tre campi:
- "search_type": uno tra "ai_model", "board_selection", "optimization", "documentation", "none"
- "search_query": la query da cercare online (in inglese, dettagliata e specifica)
- "reasoning": spiegazione della classificazione (max 100 caratteri)

Se search_type è "none", puoi mettere search_query e reasoning come stringhe vuote.

Esempi:

Input: "Quali modelli leggeri posso usare per la classificazione di immagini su STM32H7?"
Output: {
  "search_type": "ai_model",
  "search_query": "lightweight image classification models STM32H7 embedded TensorFlow",
  "reasoning": "Richiesta esplicita di modelli AI per STM32, task ben definito"
}

Input: "Confronta STM32F4 e STM32H7 per un progetto con inference AI"
Output: {
  "search_type": "board_selection",
  "search_query": "STM32F4 vs STM32H7 comparison memory performance AI inference",
  "reasoning": "Confronto tra board STM32, focus su compatibilità AI"
}

Input: "Come quantizzare un modello TensorFlow per STM32?"
Output: {
  "search_type": "optimization",
  "search_query": "TensorFlow model quantization INT8 STM32 embedded optimization",
  "reasoning": "Domanda su tecniche di ottimizzazione/compressione per embedded"
}

Input: "Dove trovo la documentazione ufficiale di STEdgeAI?"
Output: {
  "search_type": "documentation",
  "search_query": "STEdgeAI official documentation tutorial guide STMicroelectronics",
  "reasoning": "Richiesta di documentazione e risorse ufficiali"
}

Input: "Ciao come stai?"
Output: {
  "search_type": "none",
  "search_query": "",
  "reasoning": "Richiesta non correlata al sistema STM32+AI"
}
"""


# ============================================================================
# PROMPTS DINAMICI PER RICERCA
# ============================================================================

SEARCH_PROMPTS = {
    "ai_model": """
Ricerca informazioni su modelli AI compatibili con STM32.
Query: {search_query}

Per ogni modello trovato, fornisci:
1. Nome modello
2. Framework (TensorFlow, PyTorch, ONNX, etc.)
3. Dimensione in KB
4. Compatibilità STM32 (quali MCU?)
5. Link alla documentazione
6. Livello di quantizzazione consigliato
7. Performance (inference time, accuracy)
8. Casi d'uso tipici

Sii conciso e pratico per sviluppatori embedded.
    """,
    
    "board_selection": """
Ricerca informazioni su board STM32.
Query: {search_query}

Per ogni board trovata, fornisci:
1. Nome board (es. STM32F4, STM32H7, STM32U5)
2. Memoria FLASH (KB)
3. RAM (KB)
4. Velocità clock (MHz)
5. Periferiche principali (ADC, DAC, PWM, I2C, SPI, etc.)
6. Prezzo approssimativo (USD)
7. Casi d'uso consigliati
8. Dove acquistarla (distributori principali)

Compara almeno 3 board se rilevante.
    """,
    
    "optimization": """
Ricerca tecniche di ottimizzazione AI su STM32.
Query: {search_query}

Fornisci:
1. Tecniche di compressione disponibili (quantizzazione, pruning, distillazione)
2. Livelli di quantizzazione (INT8, INT16, FP16, etc.) e impatto
3. Trade-off accuratezza vs dimensione modello
4. Tool di ottimizzazione (STEdgeAI, TensorFlow Lite, TVM, etc.)
5. Benchmark di performance (latenza, throughput, memory)
6. Best practices e checklist di ottimizzazione
7. Link a risorse ufficiali e tutorial

Includi metriche concrete (es. "da 5MB a 200KB con quantizzazione INT8").
    """,
    
    "documentation": """
Ricerca documentazione e guide STM32.
Query: {search_query}

Fornisci:
1. Link a documentazione ufficiale STMicroelectronics
2. Tutorial passo-passo per il tuo argomento
3. Esempi di codice su GitHub
4. FAQ comuni e problemi risolti
5. Community forum e risorse (StackOverflow, ST Community, etc.)
6. Video tutorial (YouTube, Udemy, Coursera, etc.)
7. Libri consigliati se rilevante

Prioritizza fonti ufficiali e recenti.
    """
}


# ============================================================================
# WORKFLOW 4: WEB RESEARCH (OTTIMIZZATO CON PROMPT DINAMICO)
# ============================================================================

def classify_search(state: MasterState, config: dict) -> MasterState:
    """Classifica il tipo di ricerca richiesta dall'utente."""
    
    logger.info(f"🔍 Classificazione ricerca: {state.message}")
    
    try:
        cfg = Configuration.from_runnable_config(config)
        
        llm = ChatOllama(
            model=cfg.local_llm,
            temperature=cfg.llm_temperature,
            num_ctx=cfg.llm_context_window
        )
        
        llm_classifier = llm.with_structured_output(SearchClassification)
        
        result = llm_classifier.invoke([
            SystemMessage(content=search_classification_instructions),
            HumanMessage(content=f"Richiesta: {state.message}")
        ])
        
        state.search_type = result.search_type
        state.search_query = result.search_query
        
        logger.info(f"✓ Tipo ricerca: {state.search_type}")
        logger.info(f"  Query: {state.search_query}")
        logger.info(f"  Reasoning: {result.reasoning}")
        
        if state.search_type == "none":
            logger.warning("Nessun tipo di ricerca riconosciuto")
            state.route = "unknown"
        
    except Exception as e:
        logger.error(f"❌ Errore classificazione ricerca: {str(e)}")
        logger.exception(e)
        state.route = "unknown"
    
    return state


def search_type_decision(state: MasterState) -> Literal["execute_web_search", "clarify"]:
    """Routing semplificato: se il tipo di ricerca è valido, vai a execute_web_search."""
    if state.search_type in ["ai_model", "board_selection", "optimization", "documentation"]:
        logger.info(f"→ Esecuzione ricerca: {state.search_type}")
        return "execute_web_search"
    else:
        logger.warning(f"⚠️  Tipo ricerca non valido: {state.search_type}")
        return "clarify"


def execute_web_search(state: MasterState, config: dict) -> MasterState:
    """
    Nodo unico di ricerca che adatta il prompt dinamicamente.
    Molto più elegante che avere 4 nodi separati.
    """
    
    logger.info(f"🔍 Ricerca web: tipo={state.search_type}, query={state.search_query}")
    
    try:
        # Ottieni il prompt dinamico basato sul tipo di ricerca
        base_prompt = SEARCH_PROMPTS.get(state.search_type, SEARCH_PROMPTS["documentation"])
        search_prompt = base_prompt.format(search_query=state.search_query)
        
        logger.info(f"📋 Prompt utilizzato per {state.search_type} (lunghezza: {len(search_prompt)} char)")
        
        # Inizializza Agno Agent con Google Search
        agent = Agent(
            model=Ollama(id="mistral"),
            tools=[GoogleSearchTools()],
            show_tool_calls=False,
            markdown=True
        )

        # Esegui la ricerca
        logger.info(f"🌐 Esecuzione ricerca con Agno Agent...")
        response = agent.run(search_prompt)

        #vedi se usa i tools
        # ✅ DEBUG: STAMPA INFORMAZIONI SUI TOOL
        # print("\n" + "="*70)
        # print("🔍 DEBUG: Tool Calls")
        # print("="*70)
        
        # # Controlla gli attributi della response
        # if hasattr(response, 'formatted_tool_calls'):
        #     print(f"✅ Tool Calls: {response.formatted_tool_calls}")
        # else:
        #     print(f"❌ NO formatted_tool_calls")
        
        # if hasattr(response, 'tools'):
        #     print(f"Tools usati: {response.tools}")
        # else:
        #     print(f"❌ NO tools attribute")
        
        # if hasattr(response, 'messages'):
        #     print(f"Messages count: {len(response.messages)}")
        #     for i, msg in enumerate(response.messages):
        #         if hasattr(msg, 'tool_calls') and msg.tool_calls:
        #             print(f"  Message {i}: ✅ Ha tool_calls: {msg.tool_calls}")
        #         else:
        #             print(f"  Message {i}: ❌ NO tool_calls")
        # else:
        #     print(f"❌ NO messages")
        #fine debug tools

        
        state.search_results = response.content if response else "Nessun risultato trovato"
        state.web_research_success = True
        
        logger.info(f"✓ Ricerca completata ({len(state.search_results)} caratteri)")
        
    except Exception as e:
        logger.error(f"❌ Errore ricerca web: {str(e)}")
        logger.exception(e)
        state.search_results = f"Errore nella ricerca: {str(e)}"
        state.web_research_success = False
    
    return state


def finalize_search(state: MasterState, config: dict) -> MasterState:
    """Nodo finale che presenta i risultati della ricerca."""
    
    if state.web_research_success:
        print("\n" + "="*70)
        print(f"📊 RISULTATI RICERCA: {state.search_type.upper()}")
        print("="*70)
        print(state.search_results)
        print("="*70 + "\n")
        logger.info("✓ Ricerca completata con successo")
    else:
        print(f"\n❌ Errore durante la ricerca:\n{state.search_results}\n")
        logger.error(f"Ricerca fallita: {state.search_results}")
    
    return state

