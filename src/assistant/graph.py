# ============================================================================
# GRAPH.PY - MAIN LLM AGENT ORCHESTRATION
# ============================================================================
# Orchestrazione principale dei 5 workflow tramite LangGraph
#
# Responsabilità:
#   - Routing primario tra i 5 workflow
#   - Nodi di decisione tra workflow sequenziali
#   - StateGraph building e compilation
#
# ARCHITETTURA SEMPLIFICATA:
#   START → route_request → [firmware | ai | integration | web_search | clarify]
#        → decide_continue_* → [prossimo workflow | END]

import os
import logging
from typing import Literal
from datetime import datetime

from langgraph.graph import START, END, StateGraph
from langgraph.types import interrupt
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
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
    choose_predefined_taskbased_model,
    download_model,
    run_analyze,
    run_validate,
    run_generate,
    finalize_analysis,
    search_recommendation_model,
    model_selection_routing,
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
)


# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logging.getLogger("langgraph_api.server").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# ============================================================================
# SCHEMAS - ROUTING ONLY
# ============================================================================

class RouteDecision(BaseModel):
    """Schema per decisione di routing principale"""
    route: Literal["firmware", "ai_analysis", "integration", "web_research"] = Field(
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
   
2. **ai_analysis**: Analisi e generazione codice AI
   - Keywords: ai, modello, network, neurale, stedgeai, analyze, validate, generate, .h5, quantizzazione
   
3. **integration**: Integrazione codice AI nel firmware
   - Keywords: integra, copia, merge, combina, main.c, include, linking
   
4. **web_research**: Ricerca online di informazioni
   - Keywords: ricerca, informazioni, aiutami, quale, come, best practice, documentazione

Analizza la richiesta dell'utente e determina quale workflow è più appropriato.
Se la richiesta è ambigua, scegli il workflow più generale.

Rispondi SEMPRE in formato JSON con tre campi:
- "route": uno tra "firmware", "ai_analysis", "integration", "web_research"
- "confidence": numero tra 0.0 e 1.0
- "reasoning": breve spiegazione (max 100 caratteri)
"""

# ============================================================================
# PRIMARY ROUTING NODES
# ============================================================================

def route_request(state: MasterState, config: dict) -> MasterState:
    """Router principale che classifica tra firmware, AI, integration, web_research"""
    
    logger.info(f"🔀 Routing messaggio: {state.message[:80]}...")
    
    try:
        cfg = Configuration.from_runnable_config(config)
        
        if not cfg.validate():
            logger.error("❌ Configurazione non valida!")
            state.route = "unknown"
            return state
        
        # Popola stato con valori da Configuration
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
        llm = ChatOllama(
            model=cfg.local_llm,
            temperature=cfg.llm_temperature,
            num_ctx=cfg.llm_context_window
        )
        
        llm_router = llm.with_structured_output(RouteDecision)
        
        result = llm_router.invoke([
            SystemMessage(content=router_instructions),
            HumanMessage(content=f"Richiesta: {state.message}")
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


def route_decision(state: MasterState) -> Literal["firmware_branch", "ai_branch", "integration_branch", "search_branch", "clarify"]:
    """Routing condizionale principale"""
    route_map = {
        "firmware": "firmware_branch",
        "ai_analysis": "ai_branch",
        "integration": "integration_branch",
        "web_research": "search_branch",
        "unknown": "clarify"
    }
    
    result = route_map.get(state.route, "clarify")
    logger.info(f"→ Routing verso: {result}")
    return result


def clarify_request(state: MasterState, config: dict) -> MasterState:
    """Gestisce richieste non chiare"""
    prompt = {
        "instruction": "La tua richiesta non è chiara. Specifica cosa vuoi fare (1-4):",
        "options": {
            "1": "Generare firmware STM32",
            "2": "Analizzare modello AI",
            "3": "Integrare AI nel firmware",
            "4": "Ricerca informazioni online"
        }
    }
    user_choice = interrupt(prompt)
    
    choice_map = {"1": "firmware", "2": "ai_analysis", "3": "integration", "4": "web_research"}
    state.route = choice_map.get(str(user_choice), "firmware")
    
    logger.info(f"✓ Chiarimento ricevuto: {state.route}")
    return state

# ============================================================================
# DECISION NODES - COLLEGA RAMI SEQUENZIALI
# ============================================================================

def decide_continue_to_ai(state: MasterState, config: dict) -> MasterState:
    """Decisione dopo finalize_project"""
    
    logger.info("📋 Continuo verso AI analysis?")
    
    prompt = {
        "instruction": "Firmware generato! Continui con analisi AI? (sì/no)",
    }
    
    user_response = interrupt(prompt)
    
    if isinstance(user_response, dict):
        user_text = user_response.get("response", user_response.get("input", str(user_response)))
    else:
        user_text = str(user_response).lower()
    
    # LLM per analizzare la risposta
    cfg = Configuration.from_runnable_config(config)
    llm = ChatOllama(model=cfg.local_llm, temperature=0)
    
    response = llm.invoke([
        SystemMessage(content="Rispondi SOLO con: CONTINUARE o TERMINARE"),
        HumanMessage(content=f"Risposta: '{user_text}'")
    ])
    
    decision = response.content.strip().upper()
    
    if "CONTINUARE" in decision or "SÌ" in decision:
        state.route = "continue_to_ai"
    else:
        state.route = "end_workflow"
    
    return state


def decide_continue_to_integration(state: MasterState, config: dict) -> MasterState:
    """Decisione dopo finalize_analysis"""
    
    logger.info("📋 Continuo verso integrazione?")
    
    prompt = {
        "instruction": "Analisi AI completata! Continui con integrazione? (sì/no)",
    }
    
    user_response = interrupt(prompt)
    
    if isinstance(user_response, dict):
        user_text = user_response.get("response", user_response.get("input", str(user_response)))
    else:
        user_text = str(user_response).lower()
    
    cfg = Configuration.from_runnable_config(config)
    llm = ChatOllama(model=cfg.local_llm, temperature=0)
    
    response = llm.invoke([
        SystemMessage(content="Rispondi SOLO con: CONTINUARE o TERMINARE"),
        HumanMessage(content=f"Risposta: '{user_text}'")
    ])
    
    decision = response.content.strip().upper()
    
    if "CONTINUARE" in decision or "SÌ" in decision:
        state.route = "continue_to_integration"
    else:
        state.route = "end_workflow"
    
    return state


def decision_continue_routing(state: MasterState) -> Literal["collect_analysis_info", "collect_integration_info", "end"]:
    """Router per decision nodes"""
    
    if state.route == "continue_to_ai":
        logger.info("→ Routing verso: collect_analysis_info")
        return "collect_analysis_info"
    elif state.route == "continue_to_integration":
        logger.info("→ Routing verso: collect_integration_info")
        return "collect_integration_info"
    else:
        logger.info("→ Routing verso: END")
        return "end"


def modification_confirmation_routing(state: MasterState) -> Literal["apply_user_customization", "ask_and_parse_user_modifications"]:
    """Router per modifiche customizzazione"""
    
    if state.modification_confirmed:
        return "apply_user_customization"
    else:
        return "ask_and_parse_user_modifications"


def continue_after_customization_routing(state: MasterState) -> Literal["run_analyze", "end"]:
    """Router dopo customizzazione"""
    
    if state.continue_after_customization:
        return "run_analyze"
    else:
        return "end"

# ============================================================================
# MASTER GRAPH 
# ============================================================================
builder = StateGraph(
    MasterState,
    input=MasterInput,
    config_schema=Configuration
)

# === ROUTER NODE ===
builder.add_node("route_request", route_request)
builder.add_node("clarify", clarify_request)

# === WORKFLOW 1: FIRMWARE ===
builder.add_node("collect_project_info", collect_project_info)
builder.add_node("search_and_install_stm32_package", search_and_install_stm32_package)
builder.add_node("generate_cubemx_script", generate_cubemx_script)
builder.add_node("execute_generation", execute_generation)
builder.add_node("finalize_project", finalize_project)

# === DECISION NODE 1 ===
builder.add_node("decide_continue_to_ai", decide_continue_to_ai)

# === WORKFLOW 2: AI ANALYSIS - CON MODEL DISCOVERY ===
builder.add_node("collect_analysis_info", collect_analysis_info)
builder.add_node("choose_predefined_taskbased_model", choose_predefined_taskbased_model)
builder.add_node("search_recommendation_model", search_recommendation_model)
builder.add_node("download_model", download_model)

# === WORKFLOW 5: MODEL CUSTOMIZATION (ENHANCED VERSION) ===
# Architettura
builder.add_node("inspect_model_architecture", inspect_model_architecture)
builder.add_node("ask_modification_intent", ask_modification_intent)  # ✅ NUOVO
builder.add_node("retrieve_best_practices_for_architecture", retrieve_best_practices_for_architecture)

# User interaction & parsing
builder.add_node("ask_and_parse_user_modifications", ask_and_parse_user_modifications)
builder.add_node("collect_modification_confirmation", collect_modification_confirmation)

# Applicazione modifiche
builder.add_node("apply_user_customization", apply_user_customization)

# Fine-tuning e validazione
builder.add_node("fine_tune_customized_model", fine_tune_customized_model)
builder.add_node("validate_customized_model", validate_customized_model)

# Salvataggio e decision
builder.add_node("save_customized_model_final", save_customized_model_final)
builder.add_node("ask_continue_after_customization", ask_continue_after_customization)

# === WORKFLOW 2 CONTINUAZIONE ===
builder.add_node("run_analyze", run_analyze)
builder.add_node("run_validate", run_validate)
builder.add_node("run_generate", run_generate)
builder.add_node("finalize_analysis", finalize_analysis)

# === DECISION NODE 2 ===
builder.add_node("decide_continue_to_integration", decide_continue_to_integration)

# === WORKFLOW 3: INTEGRATION ===
builder.add_node("collect_integration_info", collect_integration_info)
builder.add_node("scan_ai_files", scan_ai_files)
builder.add_node("copy_ai_files", copy_ai_files)
builder.add_node("modify_main_c", modify_main_c)
builder.add_node("verify_integration", verify_integration)
builder.add_node("finalize_integration", finalize_integration)

# === WORKFLOW 4: WEB RESEARCH ===
builder.add_node("classify_search", classify_search)
builder.add_node("execute_web_search", execute_web_search)
builder.add_node("finalize_search", finalize_search)

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
        "firmware_branch": "collect_project_info",
        "ai_branch": "collect_analysis_info",
        "integration_branch": "collect_integration_info",
        "search_branch": "classify_search",
        "clarify": "clarify"
    }
)

builder.add_edge("clarify", "route_request")

# ============================================================================
# === RAMO 1: FIRMWARE + DECISION ===
# ============================================================================
builder.add_edge("collect_project_info", "search_and_install_stm32_package")

builder.add_conditional_edges(
    "search_and_install_stm32_package",
    check_package_installation,
    {
        "generate_cubemx_script": "generate_cubemx_script",
        "finalize_project": "finalize_project"
    }
)

builder.add_edge("generate_cubemx_script", "execute_generation")
builder.add_edge("execute_generation", "finalize_project")
builder.add_edge("finalize_project", "decide_continue_to_ai")

# === DECISION ROUTING 1: After Firmware ===
builder.add_conditional_edges(
    "decide_continue_to_ai",
    decision_continue_routing,
    {
        "collect_analysis_info": "collect_analysis_info",
        "end": END
    }
)

# ============================================================================
# === RAMO 2: AI + MODEL CUSTOMIZATION ===
# ============================================================================
builder.add_edge("collect_analysis_info", "choose_predefined_taskbased_model")

builder.add_conditional_edges(
    "choose_predefined_taskbased_model",
    model_selection_routing,
    {
        "search_recommendation_model": "search_recommendation_model",
        "download_model": "download_model"
    }
)

# Loop per ricerca multipla di modelli
builder.add_conditional_edges(
    "search_recommendation_model",
    model_selection_routing,
    {
        "search_recommendation_model": "search_recommendation_model",
        "download_model": "download_model"
    }
)

# Dopo download, vai a ispezionamento architettura
builder.add_edge("download_model", "inspect_model_architecture")

# ============================================================================
# === WORKFLOW 5: MODEL CUSTOMIZATION FLOW ===
# ============================================================================

# Fase 0: Ispezionamento e decisione intenzione di modifica
builder.add_edge("inspect_model_architecture", "ask_modification_intent")  # ✅ NUOVO

# ✅ NUOVO ROUTER CONDIZIONALE: Decide se customizzare o skip diretto ad analyze
builder.add_conditional_edges(
    "ask_modification_intent",
    decide_after_inspection,  # Router function
    {
        "retrieve_best_practices_for_architecture": "retrieve_best_practices_for_architecture",  # Se vuole modifiche
        "run_analyze": "run_analyze"  # Se skip diretto ad analyze
    }
)

# Fase 1: Analisi e suggerimenti (solo se entra qui)
builder.add_edge("retrieve_best_practices_for_architecture", "ask_and_parse_user_modifications")

# Fase 2: Parsing richieste utente
builder.add_edge("ask_and_parse_user_modifications", "collect_modification_confirmation")

# Fase 3: Loop - se l'utente non è soddisfatto, ritorna indietro
def modification_confirmation_routing(state: MasterState) -> str:
    """
    Route basato su modifica_confirmed.
    - Se modification_confirmed=False: l'utente vuole modificare la richiesta
    - Se modification_confirmed=True: l'utente è soddisfatto, procedi
    """
    if not state.modification_confirmed:
        return "ask_and_parse_user_modifications"
    else:
        return "apply_user_customization"

builder.add_conditional_edges(
    "collect_modification_confirmation",
    modification_confirmation_routing,
    {
        "ask_and_parse_user_modifications": "ask_and_parse_user_modifications",
        "apply_user_customization": "apply_user_customization"
    }
)

# Fase 4: Applicazione modifiche all'architettura
builder.add_edge("apply_user_customization", "fine_tune_customized_model")

# Fase 5: Fine-tuning con dataset reale
builder.add_edge("fine_tune_customized_model", "validate_customized_model")

# Fase 6: Validazione modello
builder.add_edge("validate_customized_model", "save_customized_model_final")

# Fase 7: Salvataggio definitivo con metadata
builder.add_edge("save_customized_model_final", "ask_continue_after_customization")

# Fase 8: Decisione: proseguire con AI analysis o terminare
def continue_after_customization_routing(state: MasterState) -> str:
    """
    Route dopo customization.
    - Se continue_after_customization=True: prosegui con X-CUBE-AI analysis
    - Se continue_after_customization=False: termina
    """
    if state.continue_after_customization:
        return "run_analyze"
    else:
        return "end"

builder.add_conditional_edges(
    "ask_continue_after_customization",
    continue_after_customization_routing,
    {
        "run_analyze": "run_analyze",
        "end": END
    }
)

# ============================================================================
# === WORKFLOW 2 CONTINUAZIONE: X-CUBE-AI ANALYSIS ===
# ============================================================================
builder.add_edge("run_analyze", "run_validate")
builder.add_edge("run_validate", "run_generate")
builder.add_edge("run_generate", "finalize_analysis")
builder.add_edge("finalize_analysis", "decide_continue_to_integration")

# === DECISION ROUTING 2: After AI Analysis ===
builder.add_conditional_edges(
    "decide_continue_to_integration",
    decision_continue_routing,
    {
        "collect_integration_info": "collect_integration_info",
        "end": END
    }
)

# ============================================================================
# === RAMO 3: INTEGRATION (FIRMWARE + AI CODE) ===
# ============================================================================
builder.add_edge("collect_integration_info", "scan_ai_files")
builder.add_edge("scan_ai_files", "copy_ai_files")
builder.add_edge("copy_ai_files", "modify_main_c")
builder.add_edge("modify_main_c", "verify_integration")
builder.add_edge("verify_integration", "finalize_integration")
builder.add_edge("finalize_integration", END)

# ============================================================================
# === RAMO 4: WEB RESEARCH ===
# ============================================================================
builder.add_conditional_edges(
    "classify_search",
    search_type_decision,
    {
        "execute_web_search": "execute_web_search",
        "clarify": "clarify"
    }
)

builder.add_edge("execute_web_search", "finalize_search")
builder.add_edge("finalize_search", END)

# ============================================================================
# === COMPILE ===
# ============================================================================
graph = builder.compile()