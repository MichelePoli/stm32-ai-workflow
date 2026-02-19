# ============================================================================
# GRAPH.PY - MAIN LLM AGENT ORCHESTRATION
# ============================================================================
# Orchestrazione principale dei 5 workflow tramite LangGraph
#
# Responsabilità:
#   - Routing primario tra i 5 workflow
#   - Nodi di decisione tra workflow sequenziali
#   - StateGraph building e compilation con SUBGRAPHS
#
# ARCHITETTURA MODULARE (SUBGRAPHS):
#   START → route_request → [firmware_flow | ai_flow | integration_flow | search_flow]
#

import os
import logging
import json
from typing import Literal
from datetime import datetime

from langgraph.graph import START, END, StateGraph
from langgraph.types import interrupt
import redis.asyncio as aioredis
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from src.assistant.configuration import Configuration
from src.assistant.state import MasterState, MasterInput

# ============================================================================
# WORKFLOW IMPORTS
# ============================================================================

# --- Workflow 1: STM32 Firmware Generation ---
from src.assistant.workflow1_firmware import (
    collect_project_info,
    search_and_install_stm32_package,
    check_package_installation,
    generate_cubemx_script,
    execute_generation,
    finalize_project,
)

# --- Workflow 2: AI Analysis ---
from src.assistant.workflow2_ai import (
    collect_analysis_info,
    choose_ai_task,
    choose_ai_model,
    download_model,
    run_analyze,
    run_validate,
    run_generate,
    finalize_analysis,
    search_recommendation_model,
    check_resource_constraints,
    resource_check_routing,
    handle_resource_failure,
    add_custom_model_procedure
)

# --- Workflow 3: Integration ---
from src.assistant.workflow3_integration import (
    collect_integration_info,
    scan_ai_files,
    copy_ai_files,
    modify_main_c,
    verify_integration,
    finalize_integration,
)

# --- Workflow 4: Web Search ---
from src.assistant.workflow4_web_search import (
    classify_search,
    search_type_decision,
    execute_web_search,
    summarize_search_results, # NEW NODE
    finalize_search,
)

# --- Workflow 5: Model Customization ---
from src.assistant.workflow5_customization import (
    inspect_model_architecture,
    ask_modification_intent,  
    decide_after_inspection,  
    retrieve_best_practices_for_architecture,
    ask_and_parse_user_modifications,
    collect_modification_confirmation,
    apply_user_customization,
    fine_tune_customized_model,
    validate_customized_model,
    save_customized_model_final,
    ask_continue_after_customization,
    optimize_hyperparameters_with_nni,    # NEW
    ask_optimization_preference,          # NEW
    optimization_routing,                 # NEW
    modification_confirmation_routing as customize_confirmation_routing
)

# --- Workflow 6: Synthetic Data ---
from src.assistant.workflow6_synthetic_data import (
    ask_synthetic_data_requirements,
    generate_synthetic_samples,
    validate_synthetic_data,
)

# --- Workflow 7: Dataset Selection ---
from src.assistant.workflow7_dataset import (
    decide_data_source,
    select_predefined_dataset,
    register_custom_dataset,
    download_dataset,
)


# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Silencing noisy internal framework logs
logging.getLogger("langgraph_api.server").setLevel(logging.WARNING)  # Silenzia log server interni
logging.getLogger("langgraph_storage.queue").setLevel(logging.WARNING) # Silenzia statistiche periodiche "Worker stats" e "Queue stats"
logging.getLogger("langgraph_api.metadata").setLevel(logging.ERROR)   # Nasconde errori persistenti di invio metadati a LangSmith
logging.getLogger("langsmith.client").setLevel(logging.ERROR)         # Silenzia errori 403 Forbidden di LangSmith (come il client di telemetria di LangChain che cercava di inviare dati ai server ufficiali senza avere una chiave API valida.)
logging.getLogger("httpx").setLevel(logging.WARNING)                 # Silenzia log delle richieste HTTP interne

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# ============================================================================
# SCHEMAS - ROUTING ONLY
# ============================================================================

class RouteDecision(BaseModel):
    """Schema per decisione di routing principale"""
    route: Literal["firmware", "ai_analysis", "integration", "web_research", "chat"] = Field(
        description="Il workflow da eseguire"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Livello di confidenza della decisione (0-1)"
    )
    reasoning: str = Field(
        description="Breve spiegazione della scelta"
    )

# ============================================================================
# EXTRACTION INSTRUCTIONS - ROUTING ONLY
# ============================================================================

router_instructions = """Sei un router intelligente per un sistema di sviluppo firmware STM32 con AI.

Il sistema ha quattro workflow principali:

1. **firmware**: Generazione progetto firmware STM32
   - Keywords: firmware, cubemx, stm32, progetto, board, .ioc, toolchain, generazione
   
2. **ai_analysis**: ESEGUIRE analisi, selezione o download di modelli
   - Usalo per: "Trova un modello per X", "Analizza Moiblenet", "Voglio scaricare YOLO"
   - Keywords: ai, modello, network, neurale, stedgeai, analyze, validate, generate, .h5
   - NON usare per domande tipo "Come converto...?", "Spiegami X" -> usa web_research
   
3. **integration**: Integrazione codice AI nel firmware
   - Keywords: integra, copia, merge, combina, main.c, include, linking
   
4. **web_research**: Ricerca online di informazioni, guide e tutorial
   - Usalo per: "Come converto X in Y?", "Documentazione su Z", "Errori comuni", "Confronto teorico"
   - Keywords: ricerca, informazioni, aiutami, quale, come, best practice, documentazione, convertire, spiegare

5. **chat**: Conversazione generale, saluti o domande sulla memoria utente
   - Usalo per: "Ciao", "Chi sei?", "Cosa stavo facendo?", "Qual'è la mia board preferita?"
   - Keywords: ciao, ricordi, profilo, chi sei, cosa ho fatto, memoria

**CONTESTO UTENTE (Profilo Persistente):**
Ti verrà fornito un "Profilo Utente" con informazioni sulle sessioni precedenti (board usata, MCU, ultimo modello). 
Usa queste informazioni se la richiesta dell'utente è ambigua o fa riferimento al passato (es: "Quale board stavo usando?", "Cosa ho fatto ieri?"). 
In questi casi di RECALL o conversazione, usa SEMPRE la route "chat".

Analizza la richiesta dell'utente e il suo profilo per determinare il workflow più appropriato.
Se la richiesta è ambigua e non riguarda la memoria, scegli il workflow più generale.

Rispondi SEMPRE in formato JSON con tre campi:
- "route": uno tra "firmware", "ai_analysis", "integration", "web_research", "chat"
- "confidence": numero tra 0.0 e 1.0
- "reasoning": breve spiegazione (max 100 caratteri)
"""

# ============================================================================
# PRIMARY ROUTING NODES
# ============================================================================

def route_request(state: MasterState, config: RunnableConfig = None) -> MasterState:
    """Router principale che classifica tra firmware, AI, integration, web_research"""
    
    logger.info(f"🔀 Routing messaggio: {state.message[:80]}...")

    
    # === GESTIONE RESET TOTALE ===
    msg_clean = state.message.lower().strip()
    if any(k == msg_clean for k in ["reset", "restart", "riparti"]):
        logger.info("🧹 Resetting ALL workflows state...")
        
        # --- AI State ---
        state.last_task = ""
        state.selected_model = None
        state.model_accepted = False
        state.model_discovery_method = ""
        state.available_models = []
        state.search_iterations = 0
        state.target = "stm32f401"
        state.compression = "high"
        state.analyze_success = False
        state.validate_success = False
        state.generate_success = False
        state.resource_check_result = "ok"
        state.ai_error_message = None
        
        # --- Firmware State ---
        state.board_name = "STM32F401VCHx"
        state.mcu_series = ""
        state.project_name = "MySTM32Project"
        state.toolchain = "STM32CubeIDE"
        state.ioc_file_path = None
        state.firmware_project_path = ""
        state.firmware_generation_success = False
        state.package_installation_success = False
        
        # --- Integration State ---
        state.integration_success = False
        state.firmware_project_dir = ""
        state.ai_code_dir = ""
        state.integration_error_message = None
        
        # --- Customization State ---
        state.customization_applied = False
        state.final_model_path = ""
        state.model_architecture = {}
        state.wants_model_modifications = False
        
        # --- Search & Data State ---
        state.search_results = ""
        state.search_summary = ""
        state.search_query = ""
        state.search_results_list = []
        state.dataset_source = ""
        state.synthetic_data_path = ""
        state.use_synthetic_data = False
        
        # --- Commmon ---
        state.user_response = ""
        state.persistent_context = {} # <--- PULIZIA MEMORIA A LUNGO TERMINE (Stato corrente)
        state.reset_profile = True    # <--- SEGNALE PER IL SERVER (Wipe su Redis)
        
        # Cambiamo il messaggio in modo che general_chat dia conferma del reset
        state.message = "[SYSTEM_MESSAGE] Il sistema è stato resettato COMPLETAMENTE. Saluta l'utente e conferma che hai dimenticato tutto, inclusa la sua board preferita e i progetti passati."
        state.route = "chat"
        
        logger.info("✓ Cleanup totale effettuato (HARD RESET). Routing a 'chat'.")
        return state
    
    try:
        # Gestisci config None (fallback a dict vuoto)
        if config is None:
            config = {}
            
        cfg = Configuration.from_runnable_config(config)
        
        # NOTA: Per VS Code integration, permettiamo esecuzione anche senza config completa
        # La validazione strict è necessaria solo per operazioni che richiedono file system
        # if not cfg.validate():
        #     logger.error("❌ Configurazione non valida!")
        #     state.route = "unknown"
        #     return state
        
        # Popola stato con valori da Configuration (se disponibili)
        state.st_email = cfg.st_email
        state.st_password = cfg.st_password
        state.base_dir = cfg.base_dir
        state.cubemx_path = cfg.cubemx_path
        state.model_path = cfg.ai_model_path
        state.ai_output_dir = cfg.ai_output_dir
        state.target = cfg.ai_target
        state.compression = cfg.ai_compression
        
        logger.info(f"✓ Configurazione caricata")
        
        # === ROUTING LLM ===
        from src.assistant.utils import get_llm
        llm_router = get_llm(
            config=config,
            structured_schema=RouteDecision,
            temperature=cfg.llm_temperature
        )
        
        # Includi profilo utente nella richiesta se presente
        user_info = f"\n\nPROFILO UTENTE: {state.persistent_context}" if state.persistent_context else ""
        
        result = llm_router.invoke([
            SystemMessage(content=router_instructions),
            HumanMessage(content=f"Richiesta: {state.message}{user_info}")
        ])
        
        state.route = result.route
        
        logger.info(f"✓ Route selezionata: {result.route}")
        logger.info(f"  Confidence: {result.confidence:.2f}")
        logger.info(f"  Reasoning: {result.reasoning}")
        
        if result.confidence < 0.6:
            logger.warning(f"⚠️  Bassa confidence ({result.confidence:.2f}), richiedo clarify")
            state.route = "unknown"
        
    except Exception as e:
        logger.error(f"❌ Errore routing: {str(e)}")
        logger.exception(e)
        state.route = "unknown"
    
    return state


def general_chat(state: MasterState, config: RunnableConfig = None) -> MasterState:
    """Nodo per conversazione generale e recall della memoria"""
    logger.info("💬 Avvio Workflow: Chat (General Assistant)")
    
    try:
        cfg = Configuration.from_runnable_config(config or {})
        from src.assistant.utils import get_llm
        llm = get_llm(
            config=config,
            temperature=0.7 # Leggermente più alta per chat
        )
        
        # Costruisci il prompt includendo la memoria storica
        user_memory = json.dumps(state.persistent_context, indent=2) if state.persistent_context else "Nessuna informazione precedente disponibile."
        
        instructions = f"""Sei l'Assistente AI per STM32. 
        
        Ecco le INFORMAZIONI SUL CONTESTO UTENTE (usale per rispondere a domande come "cosa stavo facendo?" o "quale board uso?"):
        {user_memory}
        
        ISTRUZIONI:
        1. Se l'utente chiede informazioni sul suo passato o sul progetto attuale, USA ESPLICITAMENTE i dati nel JSON sopra.
        2. Non inventare informazioni se non sono presenti nel contesto.
        3. Rispondi in modo amichevole e professionale.
        4. Parla sempre in ITALIANO.
        """
        
        logger.info(f"🧠 Context injected into Chat Prompt: {user_memory}")
        
        response = llm.invoke([
            SystemMessage(content=instructions),
            HumanMessage(content=state.message)
        ])
        
        state.message = response.content
        logger.info("✓ Risposta chat generata")
        
    except Exception as e:
        logger.error(f"❌ Errore in general_chat: {e}")
        state.message = "Spiacente, ho avuto un problema tecnico nel recuperare la nostra conversazione."

    return state


def route_decision(state: MasterState) -> Literal["firmware_flow", "ai_flow", "integration_flow", "search_flow", "chat", "clarify"]:
    """Routing condizionale principale verso SUBGRAPHS"""
    route_map = {
        "firmware": "firmware_flow",
        "ai_analysis": "ai_flow",
        "integration": "integration_flow",
        "web_research": "search_flow",
        "chat": "chat",
        "unknown": "clarify"
    }
    
    result = route_map.get(state.route, "clarify")
    logger.info(f"→ Routing verso Subgraph: {result}")
    return result


def clarify_request(state: MasterState, config: RunnableConfig = None) -> MasterState:
    """Gestisce richieste non chiare chiedendo all'utente"""
    prompt = {
        "instruction": "Non ho capito bene cosa vuoi fare. Scegli un'opzione:",
        "options": {
            "1": "Generare un nuovo progetto firmware STM32",
            "2": "Analizzare o scaricare un modello AI (X-CUBE-AI)",
            "3": "Integrare un modello AI in un progetto esistente",
            "4": "Cercare informazioni o guide online",
            "5": "Annulla e torna alla chat generale"
        }
    }
    
    # Interrupt attende che l'utente scelga un'opzione dall'estensione VS Code
    user_choice = interrupt(prompt)
    
    choice_map = {
        "1": "firmware", 
        "2": "ai_analysis", 
        "3": "integration", 
        "4": "web_research",
        "5": "chat"
    }
    
    state.route = choice_map.get(str(user_choice), "chat")
    
    # Aggiorniamo il messaggio in modo che al prossimo passo il router capisca l'intento
    if state.route != "chat":
        state.message = f"Voglio procedere con: {state.route}"
    
    logger.info(f"✓ Chiarimento ricevuto dall'utente: {state.route}")
    return state

# ============================================================================
# DECISION NODES - CONNECTING SUBGRAPHS
# ============================================================================

def decide_continue_to_ai(state: MasterState, config: RunnableConfig = None) -> MasterState:
    """
    Nodo di decisione dopo finalize_project.
    """
    
    logger.info("📋 Decisione: Continuare verso analisi AI?")
    
    # === ESTRATTORE LLM ===
    from src.assistant.utils import extract_user_response, get_llm
    cfg = Configuration.from_runnable_config(config)
    llm = get_llm(config)
    
    # Prompt semplificato per classificazione
    classification_prompt = """Analizza la risposta dell'utente e rispondi SOLO con una di queste due parole:
    
Se l'utente vuole continuare con l'analisi AI / X-CUBE-AI / ottimizzazione -> CONTINUARE
Se l'utente vuole fermarsi / ha finito -> TERMINARE

Rispondi SOLO con la parola, senza altro testo.
"""

    # --- Passo 1: Prova a usare il messaggio iniziale ---
    initial_decision = None
    if not state.user_response:
        res = llm.invoke([
            SystemMessage(content=classification_prompt),
            HumanMessage(content=f"Messaggio: {state.message}")
        ])
        decision_text = res.content.strip().upper()
        if "CONTINUARE" in decision_text: initial_decision = "CONTINUARE"
        elif "TERMINARE" in decision_text: initial_decision = "TERMINARE"
        
        if initial_decision == "CONTINUARE":
            logger.info("🤖 Intento di continuazione rilevato nel messaggio iniziale.")

    # --- Passo 2: Verifica e Interrupt ---
    if not initial_decision:
        resume_value = None
        if not state.user_response:
            prompt = {
                "instruction": "Il firmware è stato generato con successo! Vuoi continuare con l'analisi del modello AI o terminare qui?",
            }
            logger.info("⏸️ Intento di continuazione non chiaro, richiedo input...")
            resume_value = interrupt(prompt)
        
        # Dopo la ripresa: usa interrupt return value come priorità
        if resume_value and str(resume_value).strip():
            user_text = str(resume_value).strip()
        else:
            user_text = extract_user_response(state.user_response)
        state.user_response = ""
        
        res = llm.invoke([
            SystemMessage(content=classification_prompt),
            HumanMessage(content=f"Risposta: {user_text}")
        ])
        decision_text = res.content.strip().upper()
    else:
        decision_text = initial_decision

    # Interpreta la decisione
    if "CONTINUARE" in decision_text:
        logger.info("✓ CONTINUE - Going to AI Analysis")
        state.route = "continue_to_ai"
    else:
        logger.info("✓ TERMINATE - Ending flow")
        state.route = "end_workflow"
    
    return state


def decide_continue_to_integration(state: MasterState, config: RunnableConfig = None) -> MasterState:
    """
    Nodo di decisione dopo finalize_analysis.
    """
    
    logger.info("📋 Decisione: Continuare verso integrazione?")
    
    # === ESTRATTORE LLM ===
    from src.assistant.utils import extract_user_response, get_llm
    cfg = Configuration.from_runnable_config(config)
    llm = get_llm(config)
    
    classification_prompt = """Analizza la risposta dell'utente e rispondi SOLO con una di queste due parole:
    
Se l'utente vuole integrare il codice nel firmware / procedere con l'unione -> CONTINUARE
Se l'utente vuole fermarsi / ha finito -> TERMINARE

Rispondi SOLO con la parola, senza altro testo.
"""

    # --- Passo 1: Prova a usare il messaggio iniziale ---
    initial_decision = None
    if not state.user_response:
        res = llm.invoke([
            SystemMessage(content=classification_prompt),
            HumanMessage(content=f"Messaggio: {state.message}")
        ])
        decision_text = res.content.strip().upper()
        if "CONTINUARE" in decision_text: initial_decision = "CONTINUARE"
        elif "TERMINARE" in decision_text: initial_decision = "TERMINARE"
        
        if initial_decision == "CONTINUARE":
            logger.info("🤖 Intento di integrazione rilevato nel messaggio iniziale.")

    # --- Passo 2: Verifica e Interrupt ---
    if not initial_decision:
        resume_value = None
        if not state.user_response:
            prompt = {
                "instruction": "L'analisi AI è stata completata con successo! Vuoi continuare con l'integrazione del codice AI nel firmware o terminare qui?",
            }
            logger.info("⏸️ Intento di integrazione non chiaro, richiedo input...")
            resume_value = interrupt(prompt)
        
        # Dopo la ripresa: usa interrupt return value come priorità
        if resume_value and str(resume_value).strip():
            user_text = str(resume_value).strip()
        else:
            user_text = extract_user_response(state.user_response)
        state.user_response = ""
        
        res = llm.invoke([
            SystemMessage(content=classification_prompt),
            HumanMessage(content=f"Risposta: {user_text}")
        ])
        decision_text = res.content.strip().upper()
    else:
        decision_text = initial_decision

    # Interpreta la decisione
    if "CONTINUARE" in decision_text:
        logger.info("✓ CONTINUE - Going to Integration")
        state.route = "continue_to_integration"
    else:
        logger.info("✓ TERMINATE - Ending flow")
        state.route = "end_workflow"
    
    logger.info(f"📊 Final state.route: {state.route}")
    
    return state

def decision_continue_routing(state: MasterState) -> Literal["ai_flow", "integration_flow", "end"]:
    """Router per decision nodes inter-subgraph"""
    
    if state.route == "continue_to_ai":
        logger.info("→ Routing verso: ai_flow")
        return "ai_flow"
    elif state.route == "continue_to_integration":
        logger.info("→ Routing verso: integration_flow")
        return "integration_flow"
    elif state.route == "change_board":
        logger.info("→ Routing verso: firmware_flow (BACK)")
        return "firmware_flow"
    else:
        logger.info("→ Routing verso: END")
        return "end"


# ============================================================================
# SUBGRAPH BUILDERS
# ============================================================================


# ============================================================================
# SUBGRAPH BUILDERS
# ============================================================================

def inject_firmware_nodes(builder: StateGraph):
    """Adds Firmware Generation nodes and edges to the master builder"""
    builder.add_node("collect_project_info", collect_project_info)
    builder.add_node("search_and_install_stm32_package", search_and_install_stm32_package)
    builder.add_node("generate_cubemx_script", generate_cubemx_script)
    builder.add_node("execute_generation", execute_generation)
    builder.add_node("finalize_project", finalize_project)
    
    # Workflow flow (START is handled by master router)
    builder.add_edge("collect_project_info", "search_and_install_stm32_package")
    
    builder.add_conditional_edges(
        "search_and_install_stm32_package",
        check_package_installation,
        {
            "generate_cubemx_script": "generate_cubemx_script",
            "finalize_project": "finalize_project" # Skip if fail
        }
    )
    
    builder.add_edge("generate_cubemx_script", "execute_generation")
    builder.add_edge("execute_generation", "finalize_project")
    # Flow after finalize_project is handled by master builder (decide_continue_to_ai)


def inject_ai_analysis_nodes(builder: StateGraph):
    """Adds AI Analysis nodes and edges to the master builder"""
    
    # === NODES: BASIC AI FLOW ===
    builder.add_node("collect_analysis_info", collect_analysis_info)
    builder.add_node("choose_ai_task", choose_ai_task)
    builder.add_node("choose_ai_model", choose_ai_model)
    builder.add_node("search_recommendation_model", search_recommendation_model)
    builder.add_node("add_custom_model_procedure", add_custom_model_procedure)
    builder.add_node("download_model", download_model)
    
    # === NODES: CUSTOMIZATION & NNI ===
    # ... (rest of nodes same) ...
    builder.add_node("inspect_model_architecture", inspect_model_architecture)
    builder.add_node("ask_modification_intent", ask_modification_intent)
    builder.add_node("retrieve_best_practices_for_architecture", retrieve_best_practices_for_architecture)
    builder.add_node("ask_and_parse_user_modifications", ask_and_parse_user_modifications)
    builder.add_node("collect_modification_confirmation", collect_modification_confirmation)
    builder.add_node("apply_user_customization", apply_user_customization)
    
    # ... (rest of nodes same) ...
    builder.add_node("decide_data_source", decide_data_source)
    builder.add_node("select_predefined_dataset", select_predefined_dataset)
    builder.add_node("register_custom_dataset", register_custom_dataset)
    builder.add_node("download_dataset", download_dataset)
    builder.add_node("ask_synthetic_data_requirements", ask_synthetic_data_requirements)
    builder.add_node("generate_synthetic_samples", generate_synthetic_samples)
    builder.add_node("validate_synthetic_data", validate_synthetic_data)
    
    # Training & NNI
    builder.add_node("ask_optimization_preference", ask_optimization_preference)
    builder.add_node("optimize_hyperparameters_with_nni", optimize_hyperparameters_with_nni)
    builder.add_node("fine_tune_customized_model", fine_tune_customized_model)
    builder.add_node("validate_customized_model", validate_customized_model)
    builder.add_node("save_customized_model_final", save_customized_model_final)
    builder.add_node("ask_continue_after_customization", ask_continue_after_customization)
    
    # === NODES: ANALYSIS & GENERATION ===
    builder.add_node("run_analyze", run_analyze)
    builder.add_node("check_resource_constraints", check_resource_constraints)
    builder.add_node("handle_resource_failure", handle_resource_failure)
    builder.add_node("run_validate", run_validate)
    builder.add_node("run_generate", run_generate)
    builder.add_node("finalize_analysis", finalize_analysis)
    
    # ========================================================================
    # EDGES & ROUTING
    # ========================================================================
    
    builder.add_edge("collect_analysis_info", "choose_ai_task")
    builder.add_edge("choose_ai_task", "choose_ai_model")
    
    # 1. Model Discovery Routing
    def inner_model_selection_routing(state: MasterState) -> Literal["download_model", "search_recommendation_model", "add_custom_model_procedure"]:
        """Decide se procedere al download o alla ricerca avanzata o registrazione"""
        if state.model_discovery_method == "register_new":
            return "add_custom_model_procedure"
        if state.model_discovery_method == "search":
            return "search_recommendation_model"
        return "download_model"
    
    builder.add_conditional_edges(
        "choose_ai_model",
        inner_model_selection_routing,
        {
            "download_model": "download_model",
            "search_recommendation_model": "search_recommendation_model",
            "add_custom_model_procedure": "add_custom_model_procedure"
        }
    )
    
    builder.add_conditional_edges(
        "search_recommendation_model",
        inner_model_selection_routing,
        {
            "search_recommendation_model": "search_recommendation_model",
            "download_model": "download_model",
            "add_custom_model_procedure": "add_custom_model_procedure"
        }
    )
    
    # 2. Connection to Customization
    builder.add_edge("download_model", "inspect_model_architecture")
    builder.add_edge("add_custom_model_procedure", "download_model")
    builder.add_edge("inspect_model_architecture", "ask_modification_intent")
    
    # 3. Decision: Modify or Skip to Analyze?
    builder.add_conditional_edges(
        "ask_modification_intent",
        decide_after_inspection,
        {
            "retrieve_best_practices_for_architecture": "retrieve_best_practices_for_architecture",
            "run_analyze": "run_analyze" # Skip customization entirely
        }
    )
    
    builder.add_edge("retrieve_best_practices_for_architecture", "ask_and_parse_user_modifications")
    builder.add_edge("ask_and_parse_user_modifications", "collect_modification_confirmation")
    
    # 4. Confirmation Routing
    builder.add_conditional_edges(
        "collect_modification_confirmation",
        customize_confirmation_routing,
        {
            "ask_and_parse_user_modifications": "ask_and_parse_user_modifications", # Loop back
            "apply_user_customization": "apply_user_customization",
            "run_analyze": "run_analyze" # Skip to analyze
        }
    )
    
    builder.add_conditional_edges(
        "apply_user_customization",
        lambda state: "decide_data_source" if state.customization_applied else "ask_continue_after_customization",
        {
            "decide_data_source": "decide_data_source",
            "ask_continue_after_customization": "ask_continue_after_customization"
        }
    )
    
    # 5. Data Source Routing
    def inner_dataset_source_routing(state: MasterState) -> Literal["select_predefined_dataset", "register_custom_dataset", "ask_synthetic_data_requirements", "ask_optimization_preference"]:
        if state.dataset_source == "real":
            return "select_predefined_dataset"
        elif state.dataset_source == "register":
            return "register_custom_dataset"
        elif state.dataset_source == "synthetic":
            return "ask_synthetic_data_requirements"
        else:
            return "ask_optimization_preference"

    builder.add_conditional_edges(
        "decide_data_source",
        inner_dataset_source_routing,
        {
            "select_predefined_dataset": "select_predefined_dataset",
            "register_custom_dataset": "register_custom_dataset",
            "ask_synthetic_data_requirements": "ask_synthetic_data_requirements",
            "ask_optimization_preference": "ask_optimization_preference"
        }
    )
    
    builder.add_edge("register_custom_dataset", "download_dataset")
    
    builder.add_edge("select_predefined_dataset", "download_dataset")
    builder.add_edge("download_dataset", "ask_optimization_preference") 
    
    builder.add_edge("ask_synthetic_data_requirements", "generate_synthetic_samples")
    builder.add_edge("generate_synthetic_samples", "validate_synthetic_data")
    builder.add_edge("validate_synthetic_data", "ask_optimization_preference") 
    
    # 6. Optimization Preference (NNI vs Standard)
    builder.add_conditional_edges(
        "ask_optimization_preference",
        optimization_routing,
        {
            "fine_tune_customized_model": "fine_tune_customized_model",
            "optimize_hyperparameters_with_nni": "optimize_hyperparameters_with_nni"
        }
    )
    
    builder.add_edge("fine_tune_customized_model", "validate_customized_model")
    builder.add_edge("optimize_hyperparameters_with_nni", "validate_customized_model")
    
    builder.add_edge("validate_customized_model", "save_customized_model_final")
    builder.add_edge("save_customized_model_final", "ask_continue_after_customization")
    
    
    # 7. Continue After Customization Routing
    def inner_continue_after_customization_routing(state: MasterState) -> str:
        """Route to AI analysis or finalize based on user choice"""
        if state.continue_after_customization:
            return "run_analyze"
        else:
            return "finalize_analysis"
    
    builder.add_conditional_edges(
        "ask_continue_after_customization",
        inner_continue_after_customization_routing,
        {
            "run_analyze": "run_analyze",
            "finalize_analysis": "finalize_analysis"
        }
    )
    
    
    # 8. STEdgeAI Analysis & Resource Check
    builder.add_edge("run_analyze", "check_resource_constraints")
    
    builder.add_conditional_edges(
        "check_resource_constraints",
        resource_check_routing,
        {
            "run_validate": "run_validate",
            "run_generate": "run_generate",
            "choose_predefined_taskbased_model": "choose_ai_task",
            "run_analyze": "run_analyze",
            "handle_resource_failure": "handle_resource_failure"
        }
    )
    
    builder.add_conditional_edges(
        "handle_resource_failure",
        lambda state: state.route,
        {
            "change_model": "choose_ai_task",
            "change_board": "finalize_analysis" # Yield logic to inter-subgraph
        }
    )
    
    builder.add_edge("run_validate", "run_generate")
    builder.add_edge("run_generate", "finalize_analysis")


def inject_integration_nodes(builder: StateGraph):
    """Adds Integration nodes and edges to the master builder"""
    builder.add_node("collect_integration_info", collect_integration_info)
    builder.add_node("scan_ai_files", scan_ai_files)
    builder.add_node("copy_ai_files", copy_ai_files)
    builder.add_node("modify_main_c", modify_main_c)
    builder.add_node("verify_integration", verify_integration)
    builder.add_node("finalize_integration", finalize_integration)
    
    builder.add_edge("collect_integration_info", "scan_ai_files")
    
    # === CONDITIONAL EDGES: skip to finalize on failure ===
    builder.add_conditional_edges(
        "scan_ai_files",
        lambda state: "copy_ai_files" if state.scan_success else "finalize_integration",
        {
            "copy_ai_files": "copy_ai_files",
            "finalize_integration": "finalize_integration"
        }
    )
    
    builder.add_conditional_edges(
        "copy_ai_files",
        lambda state: "modify_main_c" if state.copy_success else "finalize_integration",
        {
            "modify_main_c": "modify_main_c",
            "finalize_integration": "finalize_integration"
        }
    )
    
    builder.add_edge("modify_main_c", "verify_integration")
    builder.add_edge("verify_integration", "finalize_integration")


def inject_web_search_nodes(builder: StateGraph):
    """Adds Web Search nodes and edges to the master builder"""
    builder.add_node("classify_search", classify_search)
    builder.add_node("execute_web_search", execute_web_search)
    builder.add_node("summarize_search_results", summarize_search_results)
    builder.add_node("finalize_search", finalize_search)
    
    builder.add_conditional_edges(
        "classify_search",
        search_type_decision,
        {
            "execute_web_search": "execute_web_search",
            "clarify": END # Exits to master graph to handle clarification
        }
    )
    
    builder.add_edge("execute_web_search", "summarize_search_results")
    builder.add_edge("summarize_search_results", "finalize_search")
    builder.add_edge("finalize_search", END)


# ============================================================================
# MASTER GRAPH 
# ============================================================================
builder = StateGraph(
    MasterState,
    input=MasterInput,
    config_schema=Configuration
)

# === ROUTER & COMMON NODES ===
builder.add_node("route_request", route_request)
builder.add_node("clarify", clarify_request)
builder.add_node("general_chat", general_chat)

# === INJECT WORKFLOW NODES ===
inject_firmware_nodes(builder)
inject_ai_analysis_nodes(builder)
inject_integration_nodes(builder)
inject_web_search_nodes(builder)

# === DECISION NODES (Inter-Workflow) ===
builder.add_node("decide_continue_to_ai", decide_continue_to_ai)
builder.add_node("decide_continue_to_integration", decide_continue_to_integration)


# ============================================================================
# EDGES CONFIGURATION
# ============================================================================

# === ENTRY POINT ===
builder.add_edge(START, "route_request")

# === PRIMARY ROUTING ===
builder.add_conditional_edges(
    "route_request",
    route_decision,
    {
        "firmware_flow": "collect_project_info",
        "ai_flow": "collect_analysis_info",
        "integration_flow": "collect_integration_info",
        "search_flow": "classify_search",
        "chat": "general_chat",
        "clarify": "clarify"
    }
)

builder.add_edge("clarify", "route_request")

# === FIRMWARE BRANCH CONNECTION ===
builder.add_edge("finalize_project", "decide_continue_to_ai")

builder.add_conditional_edges(
    "decide_continue_to_ai",
    decision_continue_routing,
    {
        "ai_flow": "collect_analysis_info",
        "firmware_flow": "collect_project_info",
        "end": END
    }
)

# === AI BRANCH CONNECTION ===
builder.add_conditional_edges(
    "finalize_analysis",
    lambda state: "firmware_flow" if state.route == "change_board" else "decide_continue_to_integration",
    {
        "firmware_flow": "collect_project_info",
        "decide_continue_to_integration": "decide_continue_to_integration"
    }
)

builder.add_conditional_edges(
    "decide_continue_to_integration",
    decision_continue_routing,
    {
        "integration_flow": "collect_integration_info",
        "end": END
    }
)

# === OTHER BRANCHES END ===
builder.add_edge("finalize_integration", END)
builder.add_edge("general_chat", END)

# === REDIS CLIENTS ===
# Helper per risolvere l'URL di Redis (Docker vs Local)
def get_redis_url():
    # Priorità 1: Variabile d'ambiente REDIS_URL (es: redis://redis:6379)
    env_url = os.environ.get("REDIS_URL")
    if env_url:
        return env_url
    
    # Priorità 2: Check se siamo in Docker (servizio "redis" invece di "localhost")
    if os.path.exists("/.dockerenv"):
        return "redis://redis:6379"
    
    # Fallback: Localhost
    return "redis://localhost:6379"

REDIS_URL_FOR_APP = get_redis_url()

# Client per il profilo utente (stringhe/JSON) - Async
redis_client = aioredis.from_url(REDIS_URL_FOR_APP, decode_responses=True)

# Client per il checkpointer (raw bytes) - Async
checkpointer_redis = aioredis.from_url(REDIS_URL_FOR_APP, decode_responses=False)

# Nota: memory e graph devono essere inizializzati dentro un event loop (es: startup di FastAPI)
# per evitare "RuntimeError: no running event loop"

# Per LangGraph CLI / Dev visualization (senza persistenza Redis)
graph = builder.compile()
