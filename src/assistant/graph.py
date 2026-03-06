# ============================================================================
# GRAPH.PY - MAIN LLM AGENT ORCHESTRATION
# ============================================================================
# Main orchestration of the 5 workflows via LangGraph
#
# Responsibilities:
#   - Primary routing between the 5 workflows
#   - Decision nodes between sequential workflows
#   - StateGraph building and compiling with SUBGRAPHS
#
# MODULAR ARCHITECTURE (SUBGRAPHS):
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
logging.getLogger("langgraph_api.server").setLevel(logging.WARNING)  # Silence internal server logs
logging.getLogger("langgraph_storage.queue").setLevel(logging.WARNING) # Silence periodic "Worker stats" and "Queue stats"
logging.getLogger("langgraph_api.metadata").setLevel(logging.ERROR)   # Hide persistent errors sending metadata to LangSmith
logging.getLogger("langsmith.client").setLevel(logging.ERROR)         # Silence 403 Forbidden errors from LangSmith
logging.getLogger("httpx").setLevel(logging.WARNING)                 # Silence HTTP requests logs

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# ============================================================================
# SCHEMAS - ROUTING ONLY
# ============================================================================

class RouteDecision(BaseModel):
    """Schema for main routing decision"""
    route: Literal["firmware", "ai_analysis", "integration", "web_research", "chat"] = Field(
        description="The workflow to execute"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence level of the decision (0-1)"
    )
    reasoning: str = Field(
        description="Brief explanation of the choice"
    )

# ============================================================================
# EXTRACTION INSTRUCTIONS - ROUTING ONLY
# ============================================================================

router_instructions = """You are an intelligent router for an STM32 firmware development system with AI.

The system has four main workflows:

1. **firmware**: STM32 firmware project generation
   - Keywords: firmware, cubemx, stm32, project, board, .ioc, toolchain, generation
   
2. **ai_analysis**: EXECUTE analysis, selection, or download of models
   - Use it for: "Find a model for X", "Analyze Mobilenet", "I want to download YOLO"
   - Keywords: ai, model, network, neural, stedgeai, analyze, validate, generate, .h5
   - DO NOT use for questions like "How do I convert...?", "Explain X" -> use web_research
   
3. **integration**: Integrate AI code into firmware
   - Keywords: integrate, copy, merge, combine, main.c, include, linking
   
4. **web_research**: Online search for information, guides, and tutorials
   - Use it for: "How do I convert X to Y?", "Documentation on Z", "Common errors", "Theoretical comparison"
   - Keywords: search, information, help me, which, how, best practice, documentation, convert, explain

5. **chat**: General conversation, greetings, or questions about user memory
   - Use it for: "Hello", "Who are you?", "What was I doing?", "What is my favorite board?"
   - Keywords: hello, hi, remember, profile, who are you, what did I do, memory

**USER CONTEXT (Persistent Profile):**
You will be provided with a "User Profile" containing information about previous sessions (used board, MCU, last model). 
Use this information if the user's request is ambiguous or refers to the past (e.g., "Which board was I using?", "What did I do yesterday?"). 
In these cases of RECALL or conversation, ALWAYS use the "chat" route.

Analyze the user's request and their profile to determine the most appropriate workflow.
If the request is ambiguous and does not concern memory, choose the more general workflow.

ALWAYS answer in JSON format with three fields:
- "route": one of "firmware", "ai_analysis", "integration", "web_research", "chat"
- "confidence": number between 0.0 and 1.0
- "reasoning": brief explanation (max 100 characters)
"""

# ============================================================================
# PRIMARY ROUTING NODES
# ============================================================================

def route_request(state: MasterState, config: RunnableConfig = None) -> MasterState:
    """Main router that classifies between firmware, AI, integration, web_research"""
    
    logger.info(f"🔀 Routing message: {state.message[:80]}...")
    
    # Clear transient response so stale integration/finalizer summaries don't persist
    # across sessions (they would otherwise be re-emitted by server.py for every node).
    state.response = ""

    
    # === TOTAL RESET HANDLING ===
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
        state.persistent_context = {} # <--- LONG TERM MEMORY CLEANUP (Current state)
        state.reset_profile = True    # <--- SIGNAL FOR SERVER (Wipe on Redis)
        
        # Change the message so general_chat confirms the reset
        state.message = "[SYSTEM_MESSAGE] The system has been COMPLETELY reset. Greet the user and confirm that you have forgotten everything, including their favorite board and past projects."
        state.route = "chat"
        
        logger.info("✓ Total cleanup performed (HARD RESET). Routing to 'chat'.")
        return state
    
    try:
        # Handle None config (fallback to empty dict)
        if config is None:
            config = {}
            
        cfg = Configuration.from_runnable_config(config)
        
        # NOTE: For VS Code integration, we allow execution even without full config
        # Strict validation is only needed for operations requiring file system
        # if not cfg.validate():
        #     logger.error("❌ Invalid configuration!")
        #     state.route = "unknown"
        #     return state
        
        # Populate state with Configuration values (if available)
        state.st_email = cfg.st_email
        state.st_password = cfg.st_password
        state.base_dir = cfg.base_dir
        state.cubemx_path = cfg.cubemx_path
        state.model_path = cfg.ai_model_path
        state.ai_output_dir = cfg.ai_output_dir
        state.target = cfg.ai_target
        state.compression = cfg.ai_compression
        
        # Synchronize state with long term memory if not already set
        # At the start of each request, synchronize your profile data (like saved board) with the graph's working variables
        if state.persistent_context:
            if not state.board_name: 
                state.board_name = state.persistent_context.get("board_name")
                if state.board_name: logger.info(f"🔄 Board loaded from memory: {state.board_name}")
            if not state.mcu_series: 
                state.mcu_series = state.persistent_context.get("mcu_series")
            if not state.project_name:
                state.project_name = state.persistent_context.get("project_name")
        
        logger.info(f"✓ Configuration loaded")
        
        # === LLM ROUTING ===
        from src.assistant.utils import get_llm
        llm_router = get_llm(
            config=config,
            structured_schema=RouteDecision,
            temperature=cfg.llm_temperature
        )
        
        # Include user profile in request if present
        user_info = f"\n\nUSER PROFILE: {state.persistent_context}" if state.persistent_context else ""
        
        result = llm_router.invoke([
            SystemMessage(content=router_instructions),
            HumanMessage(content=f"Request: {state.message}{user_info}")
        ])
        
        # Normalize: support both Pydantic model (result.route) and dict (result['route'])
        # JsonOutputParser can return a dict even when given pydantic_object=...
        if isinstance(result, dict):
            result = RouteDecision(**result)
        
        state.route = result.route
        
        logger.info(f"✓ Route selected: {result.route}")
        logger.info(f"  Confidence: {result.confidence:.2f}")
        logger.info(f"  Reasoning: {result.reasoning}")
        
        confidence_threshold = 0.4 if result.route == "chat" else 0.6
        if result.confidence < confidence_threshold:
            logger.warning(f"⚠️  Low confidence ({result.confidence:.2f}), requesting clarify")
            state.route = "unknown"
        
    except Exception as e:
        # -----------------------------------------------------------------------
        # FALLBACK: Mistral sometimes returns a Python dict with single quotes
        # instead of valid JSON with double quotes, e.g.: {'route': 'chat', ...}
        # LangChain's JsonOutputParser fails on this format.
        # We try ast.literal_eval as a recover before giving up.
        # -----------------------------------------------------------------------
        err_msg = str(e)
        if "Invalid json output:" in err_msg:
            try:
                import ast
                raw_dict_str = err_msg.split("Invalid json output:")[-1].strip()
                # ast.literal_eval correctly handles Python single quotes
                recovered = ast.literal_eval(raw_dict_str)
                result = RouteDecision(**recovered)
                state.route = result.route
                logger.warning(f"⚠️ Router: Malformed JSON recovered via ast.literal_eval → route={result.route}")
            except Exception as recover_e:
                logger.error(f"❌ Routing error: {err_msg}")
                logger.error(f"   Recover via ast failed: {recover_e}")
                logger.exception(e)
                state.route = "unknown"
        else:
            logger.error(f"❌ Routing error: {err_msg}")
            logger.exception(e)
            state.route = "unknown"

    
    return state


def general_chat(state: MasterState, config: RunnableConfig = None) -> MasterState:
    """Node for general conversation and memory recall"""
    logger.info("💬 Start Workflow: Chat (General Assistant)")
    
    try:
        cfg = Configuration.from_runnable_config(config or {})
        from src.assistant.utils import get_llm
        llm = get_llm(
            config=config,
            temperature=0.7 # Slightly higher for chat
        )
        
        # Bypass LLM for reset message
        if state.message and state.message.startswith("[SYSTEM_MESSAGE] The system has been COMPLETELY reset"):
            state.message = "🧠 Bzz... beep... memory successfully wiped!\n\nI've forgotten all your preferences, boards, and previous models. From now on we start with a blank slate.\n\nHow can I help you today?"
            logger.info("✓ Hardcoded reset response generated (LLM bypassed)")
            return state
            
        # Build a summary of the current situation (short + long term memory)
        session_info = {
            "current_board": state.board_name or "Not selected yet",
            "mcu_series": state.mcu_series or "Not detected",
            "project_path": state.firmware_project_path or "No project generated",
            "ai_model": state.selected_model.get("name") if state.selected_model else "None",
            "last_workflow": state.route,
            "last_operation_date": state.timestamp
        }
        
        user_memory = json.dumps(state.persistent_context, indent=2) if state.persistent_context else "No historical info."
        current_context = json.dumps(session_info, indent=2)
        
        instructions = f"""You are the EXPERT AI Assistant for STM32. 

ESSENTIAL RULES:
1. ONLY ANSWER the user's latest question.
2. NEVER generate 'User:', 'Assistant:' or conversation summaries.
3. If you don't know information about the board or project from the data below, ask the user.
4. Answer in ENGLISH briefly and technically.

CONTEXT DATA:
- Historical Memory: {user_memory}
- Current Session: {current_context}
"""
        
        logger.info(f"🧠 Context injected into Chat Prompt: {user_memory}")
        
        response = llm.invoke([
            SystemMessage(content=instructions),
            HumanMessage(content=state.message)
        ])
        
        state.message = response.content
        logger.info("✓ Chat response generated")
        
    except Exception as e:
        logger.error(f"❌ Error in general_chat: {e}")
        state.message = "Sorry, I had a technical problem retrieving our conversation."

    return state


def route_decision(state: MasterState) -> Literal["firmware_flow", "ai_flow", "integration_flow", "search_flow", "chat", "clarify"]:
    """Conditional routing to SUBGRAPHS"""
    route_map = {
        "firmware": "firmware_flow",
        "ai_analysis": "ai_flow",
        "integration": "integration_flow",
        "web_research": "search_flow",
        "chat": "chat",
        "unknown": "clarify"
    }
    
    result = route_map.get(state.route, "clarify")
    logger.info(f"→ Routing to Subgraph: {result}")
    return result


def clarify_request(state: MasterState, config: RunnableConfig = None) -> MasterState:
    """Handles ambiguous requests by asking the user"""
    prompt = {
        "instruction": "I didn't quite get what you want to do. Choose an option:",
        "options": {
            "1": "Generate a new STM32 firmware project",
            "2": "Analyze or download an AI model (X-CUBE-AI)",
            "3": "Integrate an AI model into an existing project",
            "4": "Search for information or online guides",
            "5": "Cancel and return to general chat"
        }
    }
    
    # Interrupt waits for user to choose an option from VS Code extension
    user_choice = interrupt(prompt)
    
    choice_map = {
        "1": "firmware", 
        "2": "ai_analysis", 
        "3": "integration", 
        "4": "web_research",
        "5": "chat"
    }
    
    state.route = choice_map.get(str(user_choice), "chat")
    
    # Update message so that in the next step the router understands intent
    if state.route != "chat":
        state.message = f"I want to proceed with: {state.route}"
    
    logger.info(f"✓ Clarification received from user: {state.route}")
    return state

# ============================================================================
# DECISION NODES - CONNECTING SUBGRAPHS
# ============================================================================

def decide_continue_to_ai(state: MasterState, config: RunnableConfig = None) -> MasterState:
    """
    Decision node after finalize_project.
    Asks the user if they want to continue with AI analysis, unless
    the original message explicitly requested it.
    """
    
    logger.info("📋 Decision: Continue to AI analysis?")
    
    from src.assistant.utils import extract_user_response, get_llm
    cfg = Configuration.from_runnable_config(config)
    llm = get_llm(config)
    
    classification_prompt = """Analyze the user's response and reply ONLY with one of these two words:
    
If the user wants to continue with AI analysis / X-CUBE-AI / optimization -> CONTINUE
If the user wants to stop / is done / does not mention AI -> TERMINATE

Answer ONLY with the word, without any other text.
"""

    # --- Step 1: Check if the ORIGINAL message EXPLICITLY requested AI ---
    # ONLY in this case do we skip the interrupt. "TERMINATE" on original message
    # doesn't mean user wants to stop: it just means it wasn't clear,
    # so we still have to ask.
    initial_continue = False
    if not state.user_response:
        res = llm.invoke([
            SystemMessage(content=classification_prompt),
            HumanMessage(content=f"Message: {state.message}")
        ])
        decision_text = res.content.strip().upper()
        if "CONTINUE" in decision_text:
            initial_continue = True
            logger.info("🤖 Explicit continue intent detected in initial message.")

    if initial_continue:
        # User already asked for AI analysis: proceed without interrupting
        logger.info("✓ CONTINUE - Going to AI Analysis (detected from original message)")
        state.route = "continue_to_ai"
        return state

    # --- Step 2: Interrupt - always ask the user ---
    if not state.user_response:
        prompt = {
            "instruction": "✅ Firmware successfully generated! Do you want to continue with AI model analysis (X-CUBE-AI) or stop here?",
        }
        logger.info("⏸️ Asking user if they want to continue with AI analysis...")
        resume_value = interrupt(prompt)
    else:
        resume_value = None

    # --- Step 3: Classify user response ---
    if resume_value and str(resume_value).strip():
        user_text = str(resume_value).strip()
    else:
        user_text = extract_user_response(state.user_response)
    state.user_response = ""

    res = llm.invoke([
        SystemMessage(content=classification_prompt),
        HumanMessage(content=f"Response: {user_text}")
    ])
    decision_text = res.content.strip().upper()

    if "CONTINUE" in decision_text:
        logger.info("✓ CONTINUE - Going to AI Analysis")
        state.route = "continue_to_ai"
    else:
        logger.info("✓ TERMINATE - Ending flow")
        state.route = "end_workflow"
    
    return state




def decide_continue_to_integration(state: MasterState, config: RunnableConfig = None) -> MasterState:
    """
    Decision node after finalize_analysis.
    ALWAYS asks the user if they want to continue with integration.
    Does not use LLM fast-path on original message because the previous context
    might be ambiguous (e.g. AI message != integration intent).
    """
    
    logger.info("📋 Decision: Continue to Integration?")
    
    from src.assistant.utils import extract_user_response, get_llm
    llm = get_llm(config)
    
    classification_prompt = """Analyze the user's response and reply ONLY with one of these two words:
    
If the user wants to integrate the code into firmware / proceed with merging -> CONTINUE
If the user wants to stop / is done -> TERMINATE

Answer ONLY with the word, without any other text.
"""

    # --- Interrupt: ALWAYS ask the user ---
    # We do not use LLM fast-path on original message: initial message
    # was about AI/firmware, not integration, and would cause false CONTINUEs.
    if not state.user_response:
        prompt = {
            "instruction": "✅ AI Analysis complete! Do you want to continue integrating AI code into firmware (STM32CubeMX merge) or stop here?",
        }
        logger.info("⏸️ Asking user if they want to continue with integration...")
        resume_value = interrupt(prompt)
    else:
        resume_value = None

    # --- Classify response ---
    if resume_value and str(resume_value).strip():
        user_text = str(resume_value).strip()
    else:
        user_text = extract_user_response(state.user_response)
    state.user_response = ""

    res = llm.invoke([
        SystemMessage(content=classification_prompt),
        HumanMessage(content=f"Response: {user_text}")
    ])
    decision_text = res.content.strip().upper()

    if "CONTINUE" in decision_text:
        logger.info("✓ CONTINUE - Going to Integration")
        state.route = "continue_to_integration"
    else:
        logger.info("✓ TERMINATE - Ending flow")
        state.route = "end_workflow"
    
    logger.info(f"📊 Final state.route: {state.route}")
    
    return state


def decision_continue_routing(state: MasterState) -> Literal["ai_flow", "integration_flow", "end"]:
    """Router for inter-subgraph decision nodes"""
    
    if state.route == "continue_to_ai":
        logger.info("→ Routing to: ai_flow")
        return "ai_flow"
    elif state.route == "continue_to_integration":
        logger.info("→ Routing to: integration_flow")
        return "integration_flow"
    elif state.route == "change_board":
        logger.info("→ Routing to: firmware_flow (BACK)")
        return "firmware_flow"
    else:
        logger.info("→ Routing to: END")
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
        """Decide whether to proceed with download or advanced search or registration"""
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
# Helper to resolve Redis URL (Docker vs Local)
def get_redis_url():
    # Priority 1: REDIS_URL environment variable (e.g. redis://redis:6379)
    env_url = os.environ.get("REDIS_URL")
    if env_url:
        return env_url
    
    # Priority 2: Check if we are in Docker (service "redis" instead of "localhost")
    if os.path.exists("/.dockerenv"):
        return "redis://redis:6379"
    
    # Fallback: Localhost
    return "redis://localhost:6379"

REDIS_URL_FOR_APP = get_redis_url()

# Client for user profile (strings/JSON) - Async
redis_client = aioredis.from_url(REDIS_URL_FOR_APP, decode_responses=True)

# Client for checkpointer (raw bytes) - Async
checkpointer_redis = aioredis.from_url(REDIS_URL_FOR_APP, decode_responses=False)

# Note: memory and graph must be initialized inside an event loop (e.g. FastAPI startup)
# to avoid "RuntimeError: no running event loop"

# For LangGraph CLI / Dev visualization (without Redis persistence)
graph = builder.compile()
