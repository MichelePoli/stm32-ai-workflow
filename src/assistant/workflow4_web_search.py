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
from agno.agent import Agent
from agno.tools.duckduckgo import DuckDuckGoTools
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from src.assistant.configuration import Configuration
from src.assistant.state import MasterState

from agno.tools.duckduckgo import DuckDuckGoTools
from agno.models.ollama import Ollama
import asyncio

# ============================================================================
# DEEPEVAL INTEGRATION
# ============================================================================
def _evaluate_summary_sync(
    research_topic: str,
    running_summary: str,
    web_research_results: str
) -> dict:
    """
    Evaluation Synchrone in thread separato.
    Usa metriche compatibili con web search (Faithfulness, AnswerRelevancy).
    """
    print("\n🔍 Running DeepEval evaluation with Ollama (deepseek-r1:latest)...\n")
    
    try:
        # ===== CRITICAL: SET ENVIRONMENT VARIABLES FIRST =====
        import os
        os.environ["DEEPEVAL_RESULTS_FOLDER"] = ""
        os.environ["DEEPEVAL_DISABLE_TELEMETRY"] = "1"
        os.environ["DEEPEVAL_SKIP_PROMPTS_CACHE"] = "1"
        
        from deepeval import evaluate
        from deepeval.models import OllamaModel
        from deepeval.metrics import (
            FaithfulnessMetric,
            AnswerRelevancyMetric,
            ContextualRelevancyMetric,
            HallucinationMetric
        )
        from deepeval.test_case import LLMTestCase
        
        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
        # Initialize Ollama model
        ollama_model = OllamaModel(
            model="deepseek-r1:latest", 
            base_url=base_url
        )
        
        # Define metrics 
        faithfulness = FaithfulnessMetric(threshold=0.55, model=ollama_model, async_mode=False) 
        relevancy = AnswerRelevancyMetric(threshold=0.55, model=ollama_model, async_mode=False)
        # ContextualRelevancy e Hallucination ORA ATTIVI
        contextual_relevancy = ContextualRelevancyMetric(threshold=0.55, model=ollama_model, async_mode=False)
        hallucination = HallucinationMetric(threshold=0.55, model=ollama_model, async_mode=False)
        
        # Create test case
        # retrieval_context must be a LIST of strings. 
        # If input is a string, split it or wrap it. Ideally it comes as a list from the state.
        if isinstance(web_research_results, str):
             # Fallback if string: split by double newlines to simulate chunks
             retrieval_ctx = [chunk for chunk in web_research_results.split("\n\n") if len(chunk.strip()) > 50]
             if not retrieval_ctx: retrieval_ctx = [web_research_results[:2000]]
        else:
             retrieval_ctx = web_research_results # It's already a list

        test_case = LLMTestCase(
            input=research_topic,
            actual_output=running_summary,
            retrieval_context=retrieval_ctx, 
            context=retrieval_ctx # Used by HallucinationMetric as "ground truth"
        )
        
        # Run evaluation manually to avoid 'evaluate()' blocking IO/cache overhead
        metrics_results = {}
        
        # 1. Faithfulness
        try:
            faithfulness.measure(test_case)
            metrics_results["faithfulness"] = faithfulness.score
        except Exception as e:
            print(f"Error measuring faithfulness: {e}")
            metrics_results["faithfulness"] = 0

        # 2. Answer Relevancy
        try:
            relevancy.measure(test_case)
            metrics_results["answer_relevancy"] = relevancy.score
        except Exception as e:
            print(f"Error measuring relevancy: {e}")
            metrics_results["answer_relevancy"] = 0

        # 3. Contextual Relevancy
        try:
            contextual_relevancy.measure(test_case)
            metrics_results["contextual_relevancy"] = contextual_relevancy.score
        except Exception as e:
            print(f"Error measuring contextual relevancy: {e}")
            metrics_results["contextual_relevancy"] = 0

        # 4. Hallucination
        try:
            hallucination.measure(test_case)
            metrics_results["hallucination"] = hallucination.score
        except Exception as e:
            print(f"Error measuring hallucination: {e}")
            metrics_results["hallucination"] = 0
        
        return {
            "completed": True,
            "metrics": metrics_results
        }
        
    except Exception as e:
        print(f"\n❌ Evaluation error: {e}")
        return {"completed": False, "error": str(e)}



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

def classify_search(state: MasterState, config: RunnableConfig = None) -> MasterState:
    """Classifica il tipo di ricerca richiesta dall'utente."""
    
    logger.info(f"🔍 Classificazione ricerca: {state.message}")
    
    try:
        cfg = Configuration.from_runnable_config(config)
        
        from src.assistant.utils import get_llm
        llm_classifier = get_llm(
            config=config,
            structured_schema=SearchClassification,
        )
        
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


def execute_web_search(state: MasterState, config: RunnableConfig = None) -> MasterState:
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
        cfg = Configuration.from_runnable_config(config)
        agent = Agent(
            model=Ollama(id="mistral", base_url=cfg.ollama_base_url),
            tools=[DuckDuckGoTools()],
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
        
        # Populate search_results_list for DeepEval
        if state.search_results:
            # Simple heuristic: Split by paragraphs/double newlines to create "chunks"
            # In a real RAG with vector DB, these would be the retrieved docs.
            state.search_results_list = [
                chunk.strip() 
                for chunk in state.search_results.split("\n\n") 
                if len(chunk.strip()) > 20 # Filter out tiny noise
            ]
        
        state.web_research_success = True
        
        logger.info(f"✓ Ricerca completata ({len(state.search_results)} caratteri, {len(state.search_results_list)} chunks)")
        
    except Exception as e:
        logger.error(f"❌ Errore ricerca web: {str(e)}")
        logger.exception(e)
        state.search_results = f"Errore nella ricerca: {str(e)}"
        state.search_results_list = []
        state.web_research_success = False
    
    return state


def summarize_search_results(state: MasterState, config: RunnableConfig = None) -> MasterState:
    """
    Nodo che riassume i risultati della ricerca web.
    Input: state.search_results (RAW text)
    Output: state.search_summary (Processed summary)
    """
    logger.info("📝 Summarizing search results...")
    
    if not state.web_research_success:
        logger.warning("Skipping summary due to failed search.")
        return state

    try:
        cfg = Configuration.from_runnable_config(config)
        
        # Prompt di validazione/riassunto (English for better performance)
        summary_prompt = f"""
        You are an expert technical writer for STM32 embedded systems.
        
        Objective: Summarize the web search results to answer the user's question perfectly.
        
        User Question: {state.search_query}
        
        Raw Web Results:
        {state.search_results[:10000]}  # Limit context to avoid overflow
        
        Instructions:
        1. Analyze the web results carefully.
        2. Synthesize a clear, direct, and technical answer in English.
        3. Cite sources (URLs) if present in the results.
        4. If results are irrelevant to the query, state it clearly.
        5. Use Bullet Points for readability.
        
        Answer:
        """
        
        from src.assistant.utils import get_llm
        llm = get_llm(
            config=config,
            temperature=0.2 # Low temp for factual summary
        )
        
        response = llm.invoke(summary_prompt)
        state.search_summary = response.content
        logger.info(f"✓ Summary generato ({len(state.search_summary)} chars)")
        
    except Exception as e:
        logger.error(f"❌ Errore summary: {e}")
        state.search_summary = "Impossibile generare il riassunto. Consulta i risultati grezzi."
        
    return state


def finalize_search(state: MasterState, config: RunnableConfig = None) -> MasterState:
    """Nodo finale che presenta i risultati della ricerca (Riassunto + Eval)."""
    
    if state.web_research_success:
        print("\n" + "="*70)
        print(f"📊 RISULTATI RICERCA: {state.search_type.upper()}")
        print("="*70)
        # Mostra il riassunto, non i risultati grezzi
        print(state.search_summary) 
        print("="*70 + "\n")
        logger.info("✓ Ricerca completata con successo")
    else:
        print(f"\n❌ Errore durante la ricerca:\n{state.search_results}\n")
        logger.error(f"Ricerca fallita: {state.search_results}")
    
    # ===== DEEPEVAL EVALUATION =====
    if state.web_research_success:
        try:
            print("\n" + "="*70)
            print("⚖️  EVALUATING RESULT QUALITY (DeepEval)")
            print("="*70)
            
            # Context Separation for DeepEval:
            # Actual Output = Il riassunto generato (search_summary)
            # Retrieval Context = LISTA dei chunks (search_results_list)
            
            eval_result = asyncio.run(asyncio.to_thread(
                _evaluate_summary_sync,
                state.search_query,      # Input (User Query)
                state.search_summary,    # Actual Output (LLM Summary)
                state.search_results_list if state.search_results_list else state.search_results # Fallback
            ))
            
            if eval_result["completed"]:
                metrics = eval_result["metrics"]
                print(f"✅ Faithfulness Score:       {metrics.get('faithfulness', 0):.2f}")
                print(f"✅ Answer Relevancy Score:    {metrics.get('answer_relevancy', 0):.2f}")
                print(f"✅ Contextual Relevancy:      {metrics.get('contextual_relevancy', 0):.2f}")
                print(f"✅ Hallucination Score:       {metrics.get('hallucination', 0):.2f}")
            else:
                print(f"⚠️ Evaluation skipped: {eval_result.get('error')}")
                
            print("="*70 + "\n")
            
        except Exception as e:
            logger.warning(f"Evaluation failed: {e}")

    return state

