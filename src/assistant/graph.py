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
    modification_confirmation_routing as customize_confirmation_routing # Alias to avoid name conflict if needed
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
    download_dataset,
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


def route_decision(state: MasterState) -> Literal["firmware_flow", "ai_flow", "integration_flow", "search_flow", "clarify"]:
    """Routing condizionale principale verso SUBGRAPHS"""
    route_map = {
        "firmware": "firmware_flow",
        "ai_analysis": "ai_flow",
        "integration": "integration_flow",
        "web_research": "search_flow",
        "unknown": "clarify"
    }
    
    result = route_map.get(state.route, "clarify")
    logger.info(f"→ Routing verso Subgraph: {result}")
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
    
    # Default: option 2 (AI analysis)
    if not user_choice or str(user_choice).strip() == "":
        user_choice = "2"
    
    choice_map = {"1": "firmware", "2": "ai_analysis", "3": "integration", "4": "web_research"}
    state.route = choice_map.get(str(user_choice), "firmware")
    
    logger.info(f"✓ Chiarimento ricevuto: {state.route}")
    return state

# ============================================================================
# DECISION NODES - CONNECTING SUBGRAPHS
# ============================================================================

def decide_continue_to_ai(state: MasterState, config: dict) -> MasterState:
    """Decisione dopo finalize_project"""
    
    logger.info("📋 Continuo verso AI analysis?")
    
    try:
        project_path = state.firmware_project_path
        if not project_path:
             project_path = "tuo progetto"
    except:
        project_path = "nuovo progetto"

    prompt = {
        "instruction": f"Firmware generato in {project_path}! Continui con analisi AI? (sì/no)",
    }
    
    user_response = interrupt(prompt)
    
    if isinstance(user_response, dict):
        user_text = user_response.get("response", user_response.get("input", str(user_response)))
    else:
        user_text = str(user_response).lower()
    
    #Default: continue to AI
    if not user_text or user_text.strip() == "":
        user_text = "sì"
    
    # Check semplice
    if "sì" in user_text.lower() or "si" in user_text.lower() or "yes" in user_text.lower():
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
    
    # Default: continue to integration
    if not user_text or user_text.strip() == "":
        user_text = "sì"
    
    if "sì" in user_text.lower() or "si" in user_text.lower() or "yes" in user_text.lower():
        state.route = "continue_to_integration"
    else:
        state.route = "end_workflow"
    
    return state


def decision_continue_routing(state: MasterState) -> Literal["ai_flow", "integration_flow", "end"]:
    """Router per decision nodes inter-subgraph"""
    
    if state.route == "continue_to_ai":
        logger.info("→ Routing verso: ai_flow")
        return "ai_flow"
    elif state.route == "continue_to_integration":
        logger.info("→ Routing verso: integration_flow")
        return "integration_flow"
    else:
        logger.info("→ Routing verso: END")
        return "end"


# ============================================================================
# SUBGRAPH BUILDERS
# ============================================================================

def build_firmware_graph():
    """Builds the Firmware Generation Subgraph"""
    workflow = StateGraph(MasterState, config_schema=Configuration)
    
    workflow.add_node("collect_project_info", collect_project_info)
    workflow.add_node("search_and_install_stm32_package", search_and_install_stm32_package)
    # check_package_installation is a conditional edge function, not a node
    workflow.add_node("generate_cubemx_script", generate_cubemx_script)
    workflow.add_node("execute_generation", execute_generation)
    workflow.add_node("finalize_project", finalize_project)
    
    workflow.add_edge(START, "collect_project_info")
    workflow.add_edge("collect_project_info", "search_and_install_stm32_package")
    
    # Routing conditional for package installed/not installed usually handled by node return logic
    # Here assuming standard flow:
    workflow.add_conditional_edges(
        "search_and_install_stm32_package",
        check_package_installation,
        {
            "generate_cubemx_script": "generate_cubemx_script",
            "finalize_project": "finalize_project" # Skip if fail
        }
    )
    
    workflow.add_edge("generate_cubemx_script", "execute_generation")
    workflow.add_edge("execute_generation", "finalize_project")
    workflow.add_edge("finalize_project", END)
    
    return workflow.compile()


def build_customization_graph():
    """Builds the Model Customization Subgraph (Workflow 5 + 6 + 7)"""
    workflow = StateGraph(MasterState, config_schema=Configuration)
    
    # Nodes
    workflow.add_node("inspect_model_architecture", inspect_model_architecture)
    workflow.add_node("ask_modification_intent", ask_modification_intent)
    workflow.add_node("retrieve_best_practices_for_architecture", retrieve_best_practices_for_architecture)
    workflow.add_node("ask_and_parse_user_modifications", ask_and_parse_user_modifications)
    workflow.add_node("collect_modification_confirmation", collect_modification_confirmation)
    workflow.add_node("apply_user_customization", apply_user_customization)
    
    # Dataset & Synthetic
    workflow.add_node("decide_data_source", decide_data_source)
    workflow.add_node("select_predefined_dataset", select_predefined_dataset)
    workflow.add_node("download_dataset", download_dataset)
    workflow.add_node("ask_synthetic_data_requirements", ask_synthetic_data_requirements)
    workflow.add_node("generate_synthetic_samples", generate_synthetic_samples)
    workflow.add_node("validate_synthetic_data", validate_synthetic_data)
    
    # Training & Save
    workflow.add_node("fine_tune_customized_model", fine_tune_customized_model)
    workflow.add_node("validate_customized_model", validate_customized_model)
    workflow.add_node("save_customized_model_final", save_customized_model_final)
    workflow.add_node("ask_continue_after_customization", ask_continue_after_customization)
    
    # Edges
    workflow.add_edge(START, "inspect_model_architecture")
    workflow.add_edge("inspect_model_architecture", "ask_modification_intent")
    
    workflow.add_conditional_edges(
        "ask_modification_intent",
        decide_after_inspection,
        {
            "retrieve_best_practices_for_architecture": "retrieve_best_practices_for_architecture",
            "run_analyze": END # Exits this subgraph, will be handled by master graph to go to AI flow
        }
    )
    
    workflow.add_edge("retrieve_best_practices_for_architecture", "ask_and_parse_user_modifications")
    workflow.add_edge("ask_and_parse_user_modifications", "collect_modification_confirmation")
    
    workflow.add_conditional_edges(
        "collect_modification_confirmation",
        customize_confirmation_routing,
        {
            "ask_and_parse_user_modifications": "ask_and_parse_user_modifications",
            "apply_user_customization": "apply_user_customization",
            "run_analyze": END # Exits customization
        }
    )
    
    workflow.add_edge("apply_user_customization", "decide_data_source")
    
    # Data Source Routing
    def dataset_source_routing(state: MasterState) -> Literal["select_predefined_dataset", "ask_synthetic_data_requirements", "fine_tune_customized_model"]:
        if state.dataset_source == "real":
            return "select_predefined_dataset"
        elif state.dataset_source == "synthetic":
            return "ask_synthetic_data_requirements"
        else:
            return "fine_tune_customized_model"

    workflow.add_conditional_edges(
        "decide_data_source",
        dataset_source_routing,
        {
            "select_predefined_dataset": "select_predefined_dataset",
            "ask_synthetic_data_requirements": "ask_synthetic_data_requirements",
            "fine_tune_customized_model": "fine_tune_customized_model"
        }
    )
    
    workflow.add_edge("select_predefined_dataset", "download_dataset")
    workflow.add_edge("download_dataset", "fine_tune_customized_model") # Direct connection
    
    workflow.add_edge("ask_synthetic_data_requirements", "generate_synthetic_samples")
    workflow.add_edge("generate_synthetic_samples", "validate_synthetic_data")
    workflow.add_edge("validate_synthetic_data", "fine_tune_customized_model")
    
    workflow.add_edge("fine_tune_customized_model", "validate_customized_model")
    workflow.add_edge("validate_customized_model", "save_customized_model_final")
    workflow.add_edge("save_customized_model_final", "ask_continue_after_customization")
    
    # End customization
    workflow.add_edge("ask_continue_after_customization", END)
    
    return workflow.compile()


def build_ai_analysis_graph():
    """Builds the AI Analysis Subgraph (Workflow 2), includes connection to Customization"""
    workflow = StateGraph(MasterState, config_schema=Configuration)
    
    # Nodes
    workflow.add_node("collect_analysis_info", collect_analysis_info)
    workflow.add_node("choose_predefined_taskbased_model", choose_predefined_taskbased_model)
    workflow.add_node("search_recommendation_model", search_recommendation_model)
    workflow.add_node("download_model", download_model)
    
    # HERE WE INSERT THE CUSTOMIZATION SUBGRAPH AS A NODE
    workflow.add_node("customization_flow", build_customization_graph())
    
    workflow.add_node("run_analyze", run_analyze)
    workflow.add_node("run_validate", run_validate)
    workflow.add_node("run_generate", run_generate)
    workflow.add_node("finalize_analysis", finalize_analysis)
    
    # Edges
    workflow.add_edge(START, "collect_analysis_info")
    workflow.add_edge("collect_analysis_info", "choose_predefined_taskbased_model")
    
    # Discovery Routing
    workflow.add_conditional_edges(
        "choose_predefined_taskbased_model",
        model_selection_routing,
        {
            "search_recommendation_model": "search_recommendation_model",
            "download_model": "download_model"
        }
    )
    
    workflow.add_conditional_edges(
        "search_recommendation_model",
        model_selection_routing,
        {
            "search_recommendation_model": "search_recommendation_model",
            "download_model": "download_model"
        }
    )
    
    # Dopo download, passa sempre per customization flow (che internamente può skippare)
    workflow.add_edge("download_model", "customization_flow")  
    
    # Dopo customization (con o senza modifiche), si va ad analyze
    workflow.add_edge("customization_flow", "run_analyze")
    
    workflow.add_edge("run_analyze", "run_validate")
    workflow.add_edge("run_validate", "run_generate")
    workflow.add_edge("run_generate", "finalize_analysis")
    workflow.add_edge("finalize_analysis", END)
    
    return workflow.compile()


def build_integration_graph():
    """Builds Integration Subgraph"""
    workflow = StateGraph(MasterState, config_schema=Configuration)
    
    workflow.add_node("collect_integration_info", collect_integration_info)
    workflow.add_node("scan_ai_files", scan_ai_files)
    workflow.add_node("copy_ai_files", copy_ai_files)
    workflow.add_node("modify_main_c", modify_main_c)
    workflow.add_node("verify_integration", verify_integration)
    workflow.add_node("finalize_integration", finalize_integration)
    
    workflow.add_edge(START, "collect_integration_info")
    workflow.add_edge("collect_integration_info", "scan_ai_files")
    workflow.add_edge("scan_ai_files", "copy_ai_files")
    workflow.add_edge("copy_ai_files", "modify_main_c")
    workflow.add_edge("modify_main_c", "verify_integration")
    workflow.add_edge("verify_integration", "finalize_integration")
    workflow.add_edge("finalize_integration", END)
    
    return workflow.compile()


def build_web_search_graph():
    """Builds Web Search Subgraph"""
    workflow = StateGraph(MasterState, config_schema=Configuration)
    
    workflow.add_node("classify_search", classify_search)
    workflow.add_node("execute_web_search", execute_web_search)
    workflow.add_node("finalize_search", finalize_search)
    
    workflow.add_edge(START, "classify_search")
    
    workflow.add_conditional_edges(
        "classify_search",
        search_type_decision,
        {
            "execute_web_search": "execute_web_search",
            "clarify": END # Exits to master graph to handle clarification
        }
    )
    
    workflow.add_edge("execute_web_search", "finalize_search")
    workflow.add_edge("finalize_search", END)
    
    return workflow.compile()


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

# === SUBGRAPH NODES ===
builder.add_node("firmware_flow", build_firmware_graph())
builder.add_node("ai_flow", build_ai_analysis_graph())
builder.add_node("integration_flow", build_integration_graph())
builder.add_node("search_flow", build_web_search_graph())

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
        "firmware_flow": "firmware_flow",
        "ai_flow": "ai_flow",
        "integration_flow": "integration_flow",
        "search_flow": "search_flow",
        "clarify": "clarify"
    }
)

builder.add_edge("clarify", "route_request")

# === FIRMWARE BRANCH CONNECTION ===
builder.add_edge("firmware_flow", "decide_continue_to_ai")

builder.add_conditional_edges(
    "decide_continue_to_ai",
    decision_continue_routing,
    {
        "ai_flow": "ai_flow",
        "end": END
    }
)

# === AI BRANCH CONNECTION ===
builder.add_edge("ai_flow", "decide_continue_to_integration")

builder.add_conditional_edges(
    "decide_continue_to_integration",
    decision_continue_routing,
    {
        "integration_flow": "integration_flow",
        "end": END
    }
)

# === OTHER BRANCHES END ===
builder.add_edge("integration_flow", END)
builder.add_edge("search_flow", END)

# === COMPILE ===
graph = builder.compile()