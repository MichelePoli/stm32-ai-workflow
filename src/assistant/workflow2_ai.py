# ============================================================================
# WORKFLOW 2: AI ANALYSIS WITH MODEL DISCOVERY AND CUSTOMIZATION
# ============================================================================
# Module dedicated to AI model analysis and STEdgeAI code generation
#
# Responsibilities:
#   - Collecting AI configuration (target MCU, compression)
#   - Model discovery (predefined, online search, fallback)
#   - Downloading models from GitHub/Google
#   - Model customization (architecture, fine-tuning, quantization)
#   - STEdgeAI analyze/validate/generate
#
# Dependencies: langgraph, langchain, stedgeai, tensorflow, requests

import os
import subprocess
import shutil
import re
import json
import logging
import requests
import h5py
import tensorflow as tf
from github import Github
import difflib

from typing import Optional, Literal, List
from datetime import datetime

from tensorflow.keras.models import Model, load_model, model_from_json

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from src.assistant.configuration import Configuration
from src.assistant.state import MasterState
from src.assistant.utils import get_llm, extract_user_response

# from agno.tools.googlesearch import GoogleSearchTools # Module missing in new agno
from agno.tools.duckduckgo import DuckDuckGoTools # Alternative
from agno.models.ollama import Ollama
from agno.agent import Agent


logger = logging.getLogger(__name__)

# ============================================================================
# EXTRACTION SCHEMAS - WORKFLOW 2
# ============================================================================

class AnalysisInfoExtraction(BaseModel):
    """Schema for extracting target MCU and compression"""
    target: Optional[str] = Field(
        default=None,
        description="Target MCU (e.g., stm32f401, stm32h743, stm32u5)"
    )
    compression: Optional[str] = Field(
        default=None,
        description="Compression level (low, medium, high, very_high)"
    )


class TaskSelectionExtraction(BaseModel):
    """Extracts task selection from natural response"""
    task: Optional[str] = Field(
        default=None,
        description="Selected task (technical key of the category)"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Classification confidence"
    )


class ModelSelectionExtraction(BaseModel):
    """Extracts model selection from natural response"""
    model_index: Optional[int] = Field(
        default=None,
        description="Selected model index (1-based)"
    )
    model_accepted: bool = Field(
        default=False,
        description="Did the user accept the model?"
    )
    wants_another_search: bool = Field(
        default=False,
        description="Does the user want another search?"
    )


class ModelFeedbackExtraction(BaseModel):
    """Extracts user feedback on the proposed model"""
    model_accepted: bool = Field(
        default=False,
        description="True if the user accepts the proposed model"
    )
    wants_another_search: bool = Field(
        default=False,
        description="True if the user wants another/different search"
    )
    wants_default: bool = Field(
        default=False,
        description="True if the user wants the default model/terminate search"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Classification confidence (0-1)"
    )


class ResolutionExtraction(BaseModel):
    """Extracts decision post resource failure"""
    decision: str = Field(
        description="Action to take: change_board or change_model"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Choice confidence"
    )


class SearchResultExtraction(BaseModel):
    """Extracts downloadable AI models (.h5, .keras, .onnx, .tflite)"""
    model_name: str = Field(description="Model name (e.g., MobileNetV2 128)")
    download_url: Optional[str] = Field(
        default=None,
        description="Direct URL to .h5, .keras, .onnx, or .tflite file"
    )
    model_size: Optional[str] = Field(default=None, description="Size (e.g., 5.7MB)")
    accuracy: Optional[str] = Field(default=None, description="Accuracy (e.g., 64%)")
    inference_time: Optional[str] = Field(default=None, description="Time (e.g., 40ms (STM32H7))")
    is_valid: bool = Field(
        default=False,
        description="True only if download_url is present and not None"
    )


# ============================================================================
# EXTRACTION INSTRUCTIONS - WORKFLOW 2
# ============================================================================

analysis_info_extraction_instructions = """You are an information extractor for AI analysis configuration.

Analyze the user's response and extract the following fields:

1. **target**: Target MCU to optimize the model for
     Common values: "stm32f401", "stm32f4", "stm32h743", "stm32h7", "stm32u5", "stm32u575"
     → If not specified: null

2. **compression**: Compression level for the model
     Common values: "low", "medium", "high", "very_high"
     → If not specified: null

Examples:
- Input: "STM32H743 with medium compression"
  Output: {"target": "stm32h743", "compression": "medium"}

- Input: "F4, high compression"
  Output: {"target": "stm32f4", "compression": "high"}

ALWAYS reply in valid JSON format.
"""

# Dynamically extracted instructions

model_selection_instructions = """Analyze the user's response regarding the specific model selection.
The user is choosing a model from a numbered list.

CRITICAL RULES:
1. If the response is a number (e.g. "1", "2"), strictly map it to that index (model_index).
2. "model_accepted" must be true if the user chooses a model from the list.
3. If the user rejects all or writes "no" / "none", set wants_another_search: true and model_accepted: false.
4. If the user wants to use a default or doesn't know, set model_accepted: false and wants_another_search: false.

ALWAYS reply with a valid JSON object with this exact structure:
{
  "model_index": 1,
  "model_accepted": true,
  "wants_another_search": false
}
"""

model_feedback_extraction_instructions = """Analyze the user's feedback on the proposed model.

Classify the response into one of these categories:

1. **model_accepted**: The user ACCEPTS the proposed model
   Examples: "yes", "perfect", "ok", "fine", "I accept", "tell me how to download it"

2. **wants_another_search**: The user wants to SEARCH FOR ANOTHER MODEL
   Examples: "no", "I don't like it", "search for another", "nope", "too big"

3. **wants_default**: The user wants the DEFAULT MODEL or TERMINATES
   Examples: "default", "stop searching", "stop", "predefined", "terminate"

ALWAYS reply with a valid JSON object with this exact structure:
{
  "model_accepted": true,
  "wants_another_search": false,
  "wants_default": false,
  "confidence": 0.95
}

IMPORTANT: Only ONE of the three (model_accepted, wants_another_search, wants_default) can be true!
"""

search_result_extraction_instructions = """Extract ONLY these 5 fields from the search result:

1. **model_name**: The name of the model (e.g., MobileNetV2 128)
2. **download_url**: The URL to download the file (.h5, .keras, .onnx, .tflite) (extract from parentheses if Markdown)
3. **model_size**: The size of the file (e.g., 5.7MB)
4. **accuracy**: The accuracy of the model (e.g., 64%)
5. **inference_time**: The inference time (e.g., 40ms (STM32H7))

IMPORTANT: Look for links ending with .h5, .keras, .onnx, or .tflite.
If you see [text](https://...) extract the URL from the parentheses (the second one)

ALWAYS reply with a valid JSON object with exactly these fields:
{
  "model_name": "MobileNetV2 128",
  "download_url": "https://url.com/model.h5",
  "model_size": "5.7MB",
  "accuracy": "64%",
  "inference_time": "40ms (STM32H7)",
  "is_valid": true
}
"""

# Attenzione: search_result_extraction_instructions è diverso da research_prompt. Serve per estrarre i risultati trovati, non per fare la ricerca!!

# ============================================================================
# PREDEFINED_MODELS - REAL URLs (Verified)
# ============================================================================


def get_resource_path(filename: str) -> str:
    """Returns the absolute path of a resource in the resources folder."""
    # First search in src/assistant/resources relative to this file
    base_dir = os.path.dirname(os.path.abspath(__file__))
    res_path = os.path.join(base_dir, "resources", filename)
    return res_path

def load_predefined_models() -> dict:
    """Loads predefined models from the JSON file."""
    path = get_resource_path("predefined_models.json")
    if not os.path.exists(path):
        logger.warning(f"⚠️ Models registry not found at {path}, returning empty.")
        return {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"❌ Error loading models: {e}")
        return {}

def save_predefined_models(models: dict):
    """Saves models to the JSON file."""
    path = get_resource_path("predefined_models.json")
    try:
        # Ensure the folder exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(models, f, indent=4, ensure_ascii=False)
        logger.info(f"✅ Models registry updated: {path}")
    except Exception as e:
        logger.error(f"❌ Error saving models: {e}")

# Initialize PREDEFINED_MODELS dynamically (but load it every time if we want to be dynamic at runtime)
PREDEFINED_MODELS = load_predefined_models()
# <- with .h5 and not .tflite


# ============================================================================
# WORKFLOW 2 NODES
# ============================================================================
def collect_analysis_info(state: MasterState, config: RunnableConfig = None) -> MasterState:
    """
    Collects ONLY MCU target and compression.
    Model selection is handled in subsequent nodes!
    """
    
    logger.info("📋 Collecting AI analysis configuration...")
    
    cfg = Configuration.from_runnable_config(config)
    
    prompt = {
        "instruction": """AI Analysis Configuration with STEdgeAI

Specify (briefly):
1. Target MCU (STM32F4, STM32H7, STM32U5, etc.)
2. Compression level: low, medium, high, very_high (optional, default: high)

Examples:
- "STM32H743"
- "F4 with high compression"
- "STM32U5 medium"
        """,
    }
    
    # === IDEMPOTENCY CHECK ===
    # If we already have target and compression (e.g., injected from config or resume), we skip.
    # CRITICAL: Do not skip if the target is the default one but the board is different!
    board_target = None
    if state.board_name:
        # Simple map for known boards or series extraction
        b_low = state.board_name.lower()
        targets = ["f0", "f1", "f2", "f3", "f4", "f7", "h5", "h7", "l0", "l1", "l4", "l5", "u5", "g0", "g4", "w5", "c0", "n6"]
        for t in targets:
            if t in b_low:
                board_target = f"stm32{t}"
                break
    
    # If the current board suggests a different target from the saved one, DO NOT skip
    if state.target and state.compression and not state.user_response:
        if board_target and board_target != state.target:
            logger.info(f"🔄 Reset AI target for board alignment: {state.target} -> {board_target}")
            state.target = board_target
        else:
            logger.info(f"⏭️  Idempotency: Target '{state.target}' and Compression '{state.compression}' already present. Skipping interrupt.")
            return state

    from src.assistant.utils import extract_user_response, get_llm
    
    # --- Step 1: Try to use the initial message ---
    # We search if the user has already specified a board/target in the start command
    initial_target = None
    if not state.user_response:
        # Fast heuristic analysis of the initial message
        msg_low = state.message.lower()
        targets = ["f0", "f1", "f2", "f3", "f4", "f7", "h5", "h7", "l0", "l1", "l4", "l5", "u5", "g0", "g4", "w5", "c0", "n6"]
        for t in targets:
            if t in msg_low:
                initial_target = f"stm32{t}"
                break
    
    # --- Step 2: Verification and Interrupt ---
    # Force interrupt if the intent is not crystal clear in the first message
    if not initial_target:
        resume_value = None
        if not state.user_response:
            # Suggestion from profile
            last_series = state.persistent_context.get("mcu_series", "F4") if state.persistent_context else "F4"
            dynamic_prompt = {
                "instruction": prompt["instruction"],
                "suggestion": f"💡 Last time you worked on the **{last_series}** series. Do you want to continue with this one or change?"
            }
            logger.info("⏸️ Interrupting for AI analysis config with profile suggestion.")
            # resume_value = interrupt(dynamic_prompt)
            resume_value = "STM32H7A3ZI" # BYPASS
        
        # After resuming: use interrupt return value as priority
        if resume_value and str(resume_value).strip():
            user_text = str(resume_value).strip()
        else:
            user_text = extract_user_response(state.user_response)
        state.user_response = ""
    else:
        # We already have the target from the initial message
        user_text = state.message
        logger.info(f"✓ Target '{initial_target}' detected in initial message.")

    # --- Step 3: Inheritance and Parsing ---
    if not user_text or user_text.strip() == "" or "previous" in user_text.lower() or "like before" in user_text.lower() or "profile" in user_text.lower() or "precedente" in user_text.lower() or "quella di" in user_text.lower():
        # Recover mcu_series from current state OR from persistent memory
        current_series = state.mcu_series
        if not current_series and state.persistent_context:
             current_series = state.persistent_context.get("mcu_series")
        
        if current_series and current_series.strip():
            series_to_target = {
                "F0": "stm32f0", "F1": "stm32f1", "F2": "stm32f2", "F3": "stm32f3", 
                "F4": "stm32f4", "F7": "stm32f7", "H5": "stm32h5", "H7": "stm32h7",
                "L0": "stm32l0", "L1": "stm32l1", "L4": "stm32l4", "L5": "stm32l5",
                "U5": "stm32u5", "G0": "stm32g0", "G4": "stm32g4", "W5": "stm32w5",
                "C0": "stm32c0", "N6": "stm32n6"
            }
            target_mcu = series_to_target.get(current_series.upper(), "stm32f4")
            user_text = f"{target_mcu}, high compression"
            logger.info(f"📋 Applied profile configuration: {target_mcu}")
        else:
            user_text = "STM32F4, high compression"
    
    logger.info(f"📝 User input RAW: '{user_text}'")
    
    # === EXTRACT TARGET AND COMPRESSION ===
    
    llm_extractor = get_llm(
        config=config,
        structured_schema=AnalysisInfoExtraction,
        temperature=0
    )
    
    extraction_result = llm_extractor.invoke([
        SystemMessage(content=analysis_info_extraction_instructions),
        HumanMessage(content=f"User response: {user_text}")
    ])
    
    state.target = extraction_result.target or state.target or "stm32h743"
    state.compression = extraction_result.compression or state.compression or "high"
    state.ai_output_dir = cfg.ai_output_dir
    
    os.makedirs(state.ai_output_dir, exist_ok=True)
    
    logger.info(f"✓ Configuration extracted:")
    logger.info(f"  Target: {state.target}")
    logger.info(f"  Compression: {state.compression}")
    
    return state


# ============================================================================
# NODE: CHOOSE FROM PREDEFINED MODELS (TASK-BASED)
# ============================================================================
def choose_predefined_taskbased_model(state: MasterState, config: RunnableConfig = None) -> MasterState:
    """
    Shows predefined models with LLM parsing.
    Saves the task for intelligent fallback.
    Uses PREDEFINED_MODELS as the sole source.
    """
    
def choose_ai_task(state: MasterState, config: RunnableConfig = None) -> MasterState:
    """
    Node 1: Chooses the TASK (e.g. Image Classification)
    Handles the interrupt for the main menu.
    """
    logger.info("📋 Choosing AI Task...")
    
    # Reload models from JSON
    global PREDEFINED_MODELS
    PREDEFINED_MODELS = load_predefined_models()
    
    cfg = Configuration.from_runnable_config(config)
    llm = get_llm(config=config, temperature=0)
    
    categories = list(PREDEFINED_MODELS.keys())
    prompt_lines = ["--- PREDEFINED MODELS (Optimized and Guaranteed) ---", "Select a category to see ready-to-use models:\n"]
    mapping = {}
    for i, cat in enumerate(categories, 1):
        desc = PREDEFINED_MODELS[cat].get("description", cat)
        prompt_lines.append(f"{i}. {desc}")
        mapping[str(i)] = cat
    
    prompt_lines.append("\n--- OTHER OPTIONS ---")
    reg_idx = len(categories) + 1
    prompt_lines.append(f"{reg_idx}. Register YOUR custom local model (already present on disk)")
    mapping[str(reg_idx)] = "register_new"
    
    other_idx = reg_idx + 1
    prompt_lines.append(f"{other_idx}. ONLINE Search (Look for new models on GitHub/Google)")
    mapping[str(other_idx)] = "other"
    
    prompt_text = "\n".join(prompt_lines) + f"\n\nReply with the number (1-{other_idx}) or describe what you want to do."
    
    # === IDEMPOTENCY & INTERRUPT ===
    if state.last_task and state.last_task != "other" and not state.user_response:
        logger.info(f"⏭️  Idempotency: Task '{state.last_task}' already selected.")
        return state

    resume_value = None
    if not state.user_response or state.user_response.strip() == "":
        logger.info("⏸️ Interrupting for AI task selection.")
        # resume_value = interrupt({"instruction": prompt_text})
        resume_value = "1" # BYPASS
        # logger.info("⏭️  BYPASS: Automatic task selection -> '1' (Classification)")
        # user_text = "1"
    
    # Use interrupt return value as priority
    if resume_value and str(resume_value).strip():
        user_text = str(resume_value).strip()
    else:
        user_text = extract_user_response(state.user_response).strip()
    state.user_response = "" # Clear after use
    if not user_text: user_text = "1"
    
    logger.info(f"📥 User response received: '{user_text}'")
    
    # === EXTRACT TASK WITH LLM ===
    mapping_text = "\n".join([f'- "{k}" -> {v}' for k, v in mapping.items()])
    dynamic_instructions = f"""Analyze the user's response and determine the requested AI task.
Strictly use the numerical mapping if the user has entered a number.

Menu displayed to the user:
{prompt_text}

EXPLICIT MAPPING:
{mapping_text}

RULES:
1. If the user replies with a number present in the mapping, return the corresponding task.
2. If the user describes an action, map to the closest category.
3. If the user wants something not present or a search, use "other".
4. The "confidence" value must be 1.0 for exact numerical matches.

Reply in JSON format: {{"task": "...", "confidence": 0.0-1.0}}
"""

    llm_extractor = get_llm(
        config=config,
        structured_schema=TaskSelectionExtraction,
        temperature=0
    )
    task_result = llm_extractor.invoke([
        SystemMessage(content=dynamic_instructions),
        HumanMessage(content=f"User response: {user_text}")
    ])
    
    logger.info(f"🤖 LLM Extraction: task='{task_result.task}', confidence={task_result.confidence}")
    
    state.last_task = task_result.task
    logger.info(f"✓ Task selected: {state.last_task}")
    
    if state.last_task == "register_new":
        state.model_discovery_method = "register_new"
    elif state.last_task == "other" or task_result.confidence < 0.5:
        state.model_discovery_method = "search"
        state.search_iterations = 0
    else:
        state.model_discovery_method = "taskbased"
        
    return state


def choose_ai_model(state: MasterState, config: RunnableConfig = None) -> MasterState:
    """
    Node 2: Chooses the specific MODEL from the task catalog.
    Handles the interrupt for the models list.
    """
    if state.model_discovery_method != "taskbased":
        return state

    logger.info(f"📋 Choosing Model for task '{state.last_task}'...")
    
    # === IDEMPOTENCY CHECK ===
    if state.selected_model and not state.user_response:
        logger.info(f"⏭️  Idempotency: Model '{state.selected_model['name']}' already selected.")
        return state

    task_info = PREDEFINED_MODELS.get(state.last_task)
    if not task_info:
        state.model_discovery_method = "search"
        return state

    available_models = task_info["models"]
    state.available_models = available_models
    flash_limit, _ = get_mcu_limits(state.target)
    
    model_options_text = []
    for i, model in enumerate(available_models, 1):
        size_bytes = parse_size_str(model['size'])
        flash_ratio = size_bytes / flash_limit
        status_icon = "✅" if flash_ratio <= 1.0 else ("⚠️" if flash_ratio <= 8.0 else "❌")
        status_note = "Fits" if flash_ratio <= 1.0 else (f"Compressible ({flash_ratio:.1f}x)" if flash_ratio <= 8.0 else f"Too Large ({flash_ratio:.1f}x)")
        
        import os
        filename = model.get('local_filename', model['url'])
        ext = os.path.splitext(filename)[1].upper() or "N/D"
        model_options_text.append(f"{i}. {model['name']} {status_icon} [{ext}] ({model['size']} - {status_note})")
    
    models_list = "\n".join(model_options_text)
    prompt_text = f"Which model do you want to use for {task_info['description']}?\n\nAvailable options:\n{models_list}\n{len(available_models)+1}. None of these (online search)\n\nReply with the number."
    
    resume_value = None
    if not state.user_response or state.user_response.strip() == "":
        logger.info("⏸️ Interrupting for AI model selection.")
        # resume_value = interrupt({"instruction": prompt_text})
        resume_value = "mobilenetv2 128" # BYPASS
        # logger.info("⏭️  BYPASS: Automatic model selection -> '2' (MobileNetV1)")
        # model_text = "2"
    
    # Use interrupt return value as priority
    if resume_value and str(resume_value).strip():
        model_text = str(resume_value).strip()
    else:
        model_text = extract_user_response(state.user_response).strip()
    state.user_response = "" # Clear after use
    
    # === CHOICE EXTRACTION ===
    cfg = Configuration.from_runnable_config(config)
    llm_model_extractor = get_llm(
        config=config,
        structured_schema=ModelSelectionExtraction,
        temperature=0
    )
    
    logger.info(f"📥 User response for model: '{model_text}'")
    
    model_result = llm_model_extractor.invoke([
        SystemMessage(content=model_selection_instructions),
        HumanMessage(content=f"Available models:\n{models_list}\n\nUser response: {model_text}")
    ])
    
    logger.info(f"🤖 LLM Model Extraction: index={model_result.model_index}, accepted={model_result.model_accepted}, search_again={model_result.wants_another_search}")
    
    if model_result.model_accepted and model_result.model_index:
        model_idx = model_result.model_index - 1
        if 0 <= model_idx < len(available_models):
            state.selected_model = available_models[model_idx]
            state.model_accepted = True
            logger.info(f"✓ Chosen model: {state.selected_model['name']}")
        else:
            state.model_discovery_method = "search"
    elif model_result.wants_another_search:
        state.model_discovery_method = "search"
        state.search_iterations = 0
    else:
        fallback_model = get_task_based_default_model(state.last_task)
        if fallback_model:
            state.selected_model = fallback_model
            state.model_accepted = True
        else:
            state.model_discovery_method = "default"

    return state


# ============================================================================
# MAIN NODE for model search !
# ============================================================================
def search_recommendation_model(state: MasterState, config: RunnableConfig = None) -> MasterState:
    """
    ✅ MAIN NODE: Model search with intelligent fallback
    
    TYPE HINTS: state: MasterState, config: RunnableConfig → MasterState
    
    Flow:
    1. GitHub (hybrid Python+LLM) - counts iteration
    2. Google (fallback) - DOES NOT count iteration
    3. Interrupt for user confirmation
    4. Return to "search" in routing (max 3 iterations)
    5. Task-based default - ONLY after 3 failed iterations
    """
    
    logger.info("=" * 70)
    logger.info(f"🔍 MODEL SEARCH [Iter {state.search_iterations + 1}/3]")
    logger.info(f"   Task: {state.last_task} | Target: {state.target}")
    logger.info("=" * 70)
    
    # ====================================================================
    # PHASE 1: GITHUB (hybrid) - COUNTS ITERATION
    # ====================================================================
    logger.info(f"\n📍 PHASE 1: GitHub (hybrid) - Iter {state.search_iterations + 1}/3")
    
    github_result = search_h5_file_in_repo_hybrid(
        repo_path="STMicroelectronics/stm32ai-modelzoo",
        task=state.last_task,
        target_mcu=state.target,
        config=config
    )
    
    if github_result and github_result.get('url_raw'):
        logger.info(f"✓ GitHub: Found and validated!")
        logger.info(f"  {github_result['name']}")
        
        state.selected_model = {
            'name': github_result['name'],
            'url': github_result['url_raw'],
            'local_filename': github_result.get('local_filename'),
            'source': github_result.get('source'),
            'selection_method': github_result.get('selection_method')
        }
        state.model_discovery_method = "github_hybrid"
        state.search_iterations += 1
        
        # ✅ INTERRUPT: Ask user confirmation
        logger.info(f"\n✓ MODEL FOUND - Requesting user confirmation...")
        
        # Extract format
        import os
        filename = github_result.get('local_filename', github_result['url_raw'])
        _, ext = os.path.splitext(filename)
        ext = ext.upper() if ext else "N/D"

        prompt = {
            "instruction": f"""AI Model found for {state.last_task}

📦 Details:
- Name: {github_result['name']}
- Format: {ext}
- Size: {github_result.get('size', 'N/A')}
- Source: {github_result.get('source', 'GitHub')}
- Method: {github_result.get('selection_method', 'N/A')}

🔗 URL: {github_result['url_raw']}

❓ Do you accept this model? (reply: yes/no or si/no)
- 'yes' or 'si': Proceed with download
- 'no': Continue searching for other models""",
        }
        
        # user_confirmation = interrupt(prompt)
        user_confirmation = "yes" # BYPASS
        
        # Handle dict or string
        if isinstance(user_confirmation, dict):
            confirmation_text = str(user_confirmation.get("response", user_confirmation.get("input", ""))).lower().strip()
        else:
            confirmation_text = str(user_confirmation).lower().strip()
        
        # Default: accept model (si)
        if not confirmation_text or confirmation_text.strip() == "":
            confirmation_text = "si"
        
        logger.info(f"📝 User response: '{confirmation_text}'")
        
        # Accept if: si, yes, ok, accetto, conferma, y, sì
        accepted_keywords = ["si", "yes", "ok", "accetto", "conferma", "y", "sì"]
        
        # If ACCEPTS → return state (remains github_hybrid/google_search/taskbased_fallback)
        if any(keyword in confirmation_text for keyword in accepted_keywords):
            logger.info(f"✓ Model ACCEPTED by user")
            return state  # ← Goes to download

        # If REJECTS → returns to loop
        else:
            logger.warning(f"❌ Model REJECTED by user")
            state.model_discovery_method = "search"  # ← Back to loop, continue search if iterations remain

    else:
        logger.warning(f"❌ GitHub failed")
    
    state.search_iterations += 1
    
    # ====================================================================
    # PHASE 2: GOOGLE (fallback) - NO ITERATION COUNT
    # ====================================================================
    logger.info(f"\n📍 PHASE 2: Google (fallback, NO iter++)")
    
    if state.search_iterations <= 3:
        google_result = search_via_google_tools_hybrid(state, config)
        
        if google_result['success'] and google_result['url_valid']:
            logger.info(f"✓ Google: Found and validated!")
            logger.info(f"  {google_result['model']['name']}")
            
            state.selected_model = google_result['model']
            state.model_discovery_method = "google_search"
            
            # ✅ INTERRUPT: Ask user confirmation (also for Google)
            logger.info(f"\n✓ MODEL FOUND (Google) - Requesting user confirmation...")
            
            # Extract format
            import os
            filename = google_result['model'].get('local_filename', google_result['model']['url'])
            _, ext = os.path.splitext(filename)
            ext = ext.upper() if ext else "N/D"
            
            prompt = {
                "instruction": f"""AI Model found for {state.last_task}

📦 Details:
- Name: {google_result['model']['name']}
- Format: {ext}
- Size: {google_result['model'].get('size', 'N/A')}
- Source: {google_result['model'].get('source', 'Google Search')}

🔗 URL: {google_result['model']['url']}

❓ Do you accept this model? (reply: yes/no or si/no)
- 'yes' or 'si': Proceed with download
- 'no': Continue searching for other models""",
            }
            
            # user_confirmation = interrupt(prompt)
            user_confirmation = "yes" # BYPASS
            
            if isinstance(user_confirmation, dict):
                confirmation_text = str(user_confirmation.get("response", user_confirmation.get("input", ""))).lower().strip()
            else:
                confirmation_text = str(user_confirmation).lower().strip()
            
            logger.info(f"📝 User response: '{confirmation_text}'")
            
            accepted_keywords = ["si", "yes", "ok", "accetto", "conferma", "y", "sì"]
            
            if any(keyword in confirmation_text for keyword in accepted_keywords):
                logger.info(f"✓ Model ACCEPTED by user")
                logger.info("=" * 70)
                return state  # ← Proceed to download
            else:
                logger.warning(f"❌ Model REJECTED by user - Continue search")
        else:
            logger.warning(f"❌ Google: Failed")
    
    # ====================================================================
    # PHASE 3: VERIFY ITERATIONS
    # ====================================================================
    
    if state.search_iterations < 3:
        # ✅ RETURN TO ROUTING WITH "search" - NEXT ATTEMPT
        logger.info(f"\n📍 PHASE 3: Iteration {state.search_iterations}/3 completed")
        logger.info(f"   ↻ Returning to routing for next attempt...")
        
        state.model_discovery_method = "search"  # ← BACK TO LOOP
        
        logger.info("=" * 70)
        return state
    
    # ====================================================================
    # PHASE 4: MAX ITERATIONS REACHED - TASK-BASED FALLBACK
    # ====================================================================
    else:
        logger.warning(f"\n⚠️  PHASE 4: Max iterations reached (3/3)")
        logger.info(f"   → Activating task-based fallback...")
        
        fallback_model = get_task_based_default_model(state.last_task)
        
        if fallback_model:
            logger.info(f"✓ Fallback found: {fallback_model['name']}")
            
            state.selected_model = fallback_model
            state.model_discovery_method = "taskbased_fallback"
            
            # ✅ AUTO-ACCEPT fallback (no interrupt — LangGraph re-executes node
            # from start on every resume, preventing interrupt from receiving 'yes' value)
            import os
            filename = fallback_model.get('local_filename', fallback_model.get('url', ''))
            _, ext = os.path.splitext(filename)
            ext = ext.upper() if ext else "N/D"
            
            logger.info(f"\n✓ FALLBACK MODEL AUTO-SELECTED")
            logger.info(f"  Name: {fallback_model['name']}")
            logger.info(f"  Format: {ext} | Size: {fallback_model.get('size', 'N/A')}")
            logger.info(f"  URL: {fallback_model.get('url', 'N/A')}")
            logger.info("=" * 70)
            return state  # ← Proceed to download automatically
        else:
            logger.warning(f"❌ No fallback available")
            
            cfg = Configuration.from_runnable_config(config)
            state.model_path = cfg.ai_model_path
            state.model_discovery_method = "default"
        
        logger.info("=" * 70)
        return state


def model_selection_routing(state: MasterState) -> Literal["run_analyze", "download_model", "search_recommendation_model", "add_custom_model_procedure"]:
    """
    Intelligent routing after model selection.
    Handles the search loop up to max 3 attempts and new model registration.
    """
    
    logger.info(f"\n🔄 ROUTING DECISION:")
    logger.info(f"   discovery_method: {state.model_discovery_method}")
    logger.info(f"   search_iterations: {state.search_iterations}")
    
    # ====================================================================
    # CASE 0: Register new model
    # ====================================================================
    if state.model_discovery_method == "register_new":
        logger.info("   → Registering new model, goes to add_custom_model_procedure")
        return "add_custom_model_procedure"

    # ====================================================================
    # CASE 1: Default model (no search)
    # ====================================================================
    if state.model_discovery_method == "default":
        logger.info("   → Model ready/selected, goes to download_model for inspection")
        return "download_model"
    
    # ====================================================================
    # CASE 2: Active search (returns to loop if rejected)
    # ====================================================================
    elif state.model_discovery_method == "search":
        if state.search_iterations < 3:
            logger.info(f"   → Search in progress ({state.search_iterations}/3), returning to search_recommendation_model")
            return "search_recommendation_model"
        else:
            logger.info(f"   → Max iterations (3/3) reached, goes to run_analyze (default)")
            return "run_analyze"
    
    # ====================================================================
    # CASE 3: Model found and ACCEPTED
    # ====================================================================
    else:  # github_hybrid, google_search, taskbased_fallback
        logger.info(f"   → {state.model_discovery_method} ACCEPTED by user, goes to download_model")
        return "download_model"

# ============================================================================
# PART 1 of model search: HYBRID GITHUB SEARCH (Python + LLM with Structured Output)
# ============================================================================

def search_h5_file_in_repo_hybrid( # fundamental 
    repo_path: str,
    task: str,
    target_mcu: Optional[str] = None,
    config: RunnableConfig = None,
    max_depth: int = 5  # ← DEPTH LIMIT
) -> Optional[dict]:
    """
    Search for .h5 files with hybrid approach (OPTIMIZED)
    
    ✅ Improvements:
    - Depth limit to avoid infinite loops
    - Detailed logging to find blockages
    - Early exit on found files
    - Virtualized timeout on GitHub API
    """
    
    try:
        logger.info(f"🔗 HYBRID GitHub Search: {task}")
        
        # STEP 1: PYTHON → Repo scan
        logger.info(f"→ STEP 1: Repo scan (Python)...")
        
        token = os.getenv("GITHUB_ACCESS_TOKEN")
        if not token:
            logger.error("❌ GITHUB_ACCESS_TOKEN not set!")
            return None
        
        try:
            g = Github(token)
            repo = g.get_repo(repo_path)
            logger.info(f"✓ Connected to {repo_path}")
        except Exception as e:
            logger.error(f"❌ Connection error: {str(e)[:80]}")
            return None
        
        # ✅ TASK → FOLDER
        folder = task.lower().replace(" ", "_")
        
        try:
            root_contents = repo.get_contents(folder)
            logger.info(f"✓ Folder found: {folder}/")
        except Exception as e:
            logger.error(f"❌ Folder not found: {folder}")
            logger.error(f"   Details: {str(e)[:80]}")
            return None
        
        h5_files: List[dict] = []
        items_checked = 0  # Counter for debugging
        
        def scan_repo(contents_list, depth=0):
            """
            Scans repo and collects .h5 files
            ✅ OPTIMIZED: Early exit, limit depth, logging
            """
            nonlocal items_checked
            
            if depth >= max_depth:
                logger.debug(f"  ⚠️  Max depth ({max_depth}) reached, stop")
                return
            
            try:
                for item in contents_list:
                    items_checked += 1
                    
                    # Log every 50 items
                    if items_checked % 50 == 0:
                        logger.info(f"  📊 Scanned {items_checked} items ({len(h5_files)} .h5 found)...")
                    
                    try:
                        if item.type == "dir":
                            logger.debug(f"  {'  ' * depth}📁 Dir: {item.name}")
                            
                            try:
                                sub = repo.get_contents(item.path)
                                scan_repo(sub, depth + 1)
                            except Exception as e:
                                logger.debug(f"  {'  ' * depth}⚠️  Error reading {item.path}: {type(e).__name__}")
                                continue
                        
                        elif item.type == "file" and any(item.name.endswith(ext) for ext in [".h5", ".keras", ".onnx", ".tflite"]):
                            # FIX FOR CUSTOMIZATION: If the user wants to customize the model, we MUST restrict the search to .h5 or .keras 
                            # Otherwise workflow5 will crash trying to structural-edit an ONNX or TFLite model.
                            # For simplicity we assume that if we are doing a repo search, it is safer to stick to native Keras models by default 
                            # if we intend to apply some workflow on them later.
                            # But since `state` isn't passed here, we'll just highly penalize non-Keras models in the LLM scoring later.
                            description = extract_description(item.name, item.path)
                            h5_files.append({
                                'name': item.name,
                                'path': item.path,
                                'size': item.size if hasattr(item, 'size') else 0,
                                'description': description,
                                'folder': item.path.rsplit('/', 1)[0] if '/' in item.path else folder
                            })
                            logger.debug(f"  {'  ' * depth}✅ File found: {item.name}")
                            
                            # ✅ EARLY EXIT if enough files found
                            if len(h5_files) >= 20:  # Practical limit
                                logger.info(f"  ℹ️  Found {len(h5_files)} files, stopping search")
                                return
                    
                    except Exception as e:
                        logger.debug(f"  ⚠️  Error parsing item {item.name}: {type(e).__name__}")
                        continue
            
            except Exception as e:
                logger.error(f"❌ Error during scan_repo: {str(e)[:100]}")
                import traceback
                logger.debug(traceback.format_exc())
        
        logger.info(f"→ Beginning recursive scan...")
        scan_repo(root_contents)
        
        logger.info(f"✓ Scan completed: {items_checked} items, {len(h5_files)} .h5 files found")
        
        if not h5_files:
            logger.warning(f"❌ No .h5 files found after {items_checked} checks")
            return None
        
        logger.info(f"✓ Found {len(h5_files)} .h5 files")
        for f in h5_files[:5]:
            logger.info(f"  - {f['name']} ({format_bytes(f['size'])}) [{f['description']}]")
        
        if len(h5_files) > 5:
            logger.info(f"  ... and {len(h5_files) - 5} more files")
        
        # STEP 2: LLM → Sophisticated selection
        logger.info(f"→ STEP 2: Reasoning with LLM (structured)...")
        
        selected_file = llm_select_best_model(
            h5_files=h5_files,
            task=task,
            target_mcu=target_mcu or "STM32H7",
            config=config
        )
        
        if not selected_file:
            logger.warning(f"❌ LLM failed, using first file fallbacl")
            selected_file = h5_files[0]
            selection_method = "fallback_first"
        else:
            selection_method = "llm_reasoning"
            logger.info(f"✓ LLM selected: {selected_file['name']}")
        
        # STEP 3: PYTHON → URL and Validation
        logger.info(f"→ STEP 3: URL Construction and validation...")
        
        url_raw = f"https://raw.githubusercontent.com/{repo_path}/main/{selected_file['path']}"
        logger.info(f"🔗 URL: {url_raw[:70]}...")
        
        is_valid = validate_model_url_quick(url_raw)
        
        if not is_valid:
            logger.warning(f"❌ Un-downloadable URL (404?)")
            
            # Fallback: try other files
            for alt_file in h5_files[1:3]:
                logger.info(f"→ Alternative attempt: {alt_file['name']}...")
                alt_url = f"https://raw.githubusercontent.com/{repo_path}/main/{alt_file['path']}"
                
                if validate_model_url_quick(alt_url):
                    logger.info(f"✓ Formatter valid!")
                    selected_file = alt_file
                    url_raw = alt_url
                    is_valid = True
                    break
        
        if not is_valid:
            logger.error(f"❌ No valid URLs found")
            return None
        
        logger.info(f"✓ URL validated! Size: {format_bytes(selected_file['size'])}")
        
        return {
            'name': selected_file['name'],
            'url_raw': url_raw,
            'path': selected_file['path'],
            'size': selected_file['size'],
            'selection_method': selection_method,
            'source': 'github',
            'local_filename': selected_file['name']
        }
    
    except Exception as e:
        logger.error(f"❌ Error: {str(e)[:150]}")
        import traceback
        logger.debug(traceback.format_exc())
        return None


def llm_select_best_model(
    h5_files: List[dict],
    task: str,
    target_mcu: str,
    config: RunnableConfig = None
) -> Optional[dict]:
    """
    LLM reasons and selects the best .h5 file
    ✅ STRUCTURED: Forces format with Pydantic
    """
    
    try:
        logger.info(f"→ Sending to LLM ({len(h5_files)} file(s))...")
        
        h5_list_text = "\n".join([
            f"{i+1}. {f['name']:40} | {format_bytes(f['size']):>10} | {f['description']}"
            for i, f in enumerate(h5_files)
        ])
        
        prompt = f"""You are an AI model expert for STM32 embedded systems.

REQUESTED TASK: {task}
TARGET MCU: {target_mcu}

AVAILABLE FILES IN REPO:
{h5_list_text}

⚠️ CRITICAL INSTRUCTIONS:
1. Analyze ALL models (.h5, .keras, .onnx, .tflite)
2. Choose the BEST one for the task (consider: compatibility, size, architecture)
3. Return a JSON object with the key "selected_index" containing the 1-based index (1-{len(h5_files)})

SCORING:
- Exact match task: +100
- Known architecture (resnet, yolo, mobilenet, efficientnet): +50
- Is .h5 or .keras file: +1000  (CRITICAL: Always prefer native Keras files over .onnx or .tflite to allow structural editing later)
- Size < 10MB: +25
- Size < 1MB: +50
"""
        
        logger.debug(f"LLM Prompt: {prompt[:350]}...")
        
        # Use centralized LLM setup
        from src.assistant.utils import get_llm
        llm = get_llm(config)
        
        # ✅ STRUCTURED OUTPUT - Force format
        class ModelSelection(BaseModel):
            selected_index: int = Field(
                description=f"MANDATORY 1-based index (1-{len(h5_files)}). Nothing else.",
                ge=1,  # Min 1
                le=len(h5_files)  # Max len(h5_files)
            )
        
        llm_selector = llm.with_structured_output(ModelSelection)
        
        logger.info(f"→ Sending prompt to LLM...")
        
        selection = llm_selector.invoke([
            SystemMessage(content="""You are a model selection task.
YOU MUST answer ONLY with a valid JSON object.
Example: {"selected_index": 5}
No text, no explanations.
If in doubt, choose the smallest and most stable model."""),
            HumanMessage(content=prompt)
        ])
        
        logger.info(f"📊 LLM Selection:")
        logger.info(f"  Index: {selection.selected_index}")
        
        # Convert 1-based → 0-based
        idx_0based = selection.selected_index - 1
        
        if idx_0based < 0 or idx_0based >= len(h5_files):
            logger.warning(f"❌ Index out of bounds: {selection.selected_index}")
            logger.warning(f"   Fallback: choosing the first file")
            return h5_files[0]
        
        selected_file = h5_files[idx_0based]
        logger.info(f"✓ LLM chose file #{selection.selected_index}: {selected_file['name']}")
        logger.info(f"  Size: {format_bytes(selected_file['size'])}")
        logger.info(f"  Description: {selected_file['description']}")
        
        return selected_file
    
    except Exception as e:
        logger.error(f"❌ LLM selection error: {str(e)[:100]}")
        import traceback
        logger.debug(traceback.format_exc())
        
        # Smart fallback: prefer .h5/.keras, non-empty files, sorted by size ascending
        valid_files = [f for f in h5_files if f.get('size', 0) > 1000]  # skip 0-byte/corrupt files
        if valid_files:
            keras_files = [f for f in valid_files if f['name'].endswith(('.h5', '.keras'))]
            best = sorted(keras_files or valid_files, key=lambda x: x.get('size', 0))
            logger.warning(f"→ Smart fallback: choosing '{best[0]['name']}' ({format_bytes(best[0].get('size', 0))})")
            return best[0]
        
        logger.warning(f"→ Fallback: choosing the first file")
        return h5_files[0] if h5_files else None


# ============================================================================
# PART 2: GOOGLE FALLBACK SEARCH (with LLM Structured Extraction)
# ============================================================================

def search_via_google_tools_hybrid(
    state: MasterState,
    config: RunnableConfig
) -> dict:
    """
    Google Search as fallback (DOES NOT increment iterations)
    Uses SearchResultExtraction with structured output
    """
    
    try:
        logger.info(f"🔍 Google Search (fallback, NO iter++)...")
        
        google_prompt = f"""Search AI models (.h5, .keras, .onnx, .tflite) for STM32
Target: {state.target}
Task: {state.last_task}

Criteria:
1. GitHub Raw or Hugging Face links
2. Direct download
3. File compatible with STM32 X-CUBE-AI

Return exactly this JSON format:
- Name: [model_title]
- URL: [downloadable_link]
- Size: [MB]
- Accuracy: [%]
- Inference: [ms]
"""
        
        logger.info(f"→ Google Agent...")
        
        google_agent = Agent(
            model=Ollama(id="mistral"),
            tools=[DuckDuckGoTools()],
            instructions=[
                "Search .h5 files for STM32",
                "Direct GitHub /raw/ links",
                "Do not invent URLs"
            ],
        )

        
        google_response = google_agent.run(google_prompt)
        response_text = (
            google_response.content 
            if hasattr(google_response, 'content') 
            else str(google_response)
        )
        
        logger.info(f"📊 Google Response: {response_text[:250]}...")
        
        if "NOT_FOUND" in response_text.upper():
            logger.warning(f"❌ Google: Not found")
            return {'success': False, 'url_valid': False, 'model': None}
        
        # ✅ EXTRACT WITH LLM STRUCTURED OUTPUT (NOT regex!)
        logger.info(f"→ Extraction with SearchResultExtraction...")
        
        cfg = Configuration.from_runnable_config(config)
        
        llm_extractor = get_llm(config, structured_schema=SearchResultExtraction)
        
        try:
            search_extraction = llm_extractor.invoke([
                SystemMessage(content=search_result_extraction_instructions),
                HumanMessage(content=f"Google search result:\n\n{response_text}")
            ])
            
            logger.info(f"📊 LLM Extraction:")
            logger.info(f"  Model: {search_extraction.model_name}")
            logger.info(f"  URL: {search_extraction.download_url[:60] if search_extraction.download_url else 'None'}...")
            logger.info(f"  Size: {search_extraction.model_size}")
            logger.info(f"  Valid: {search_extraction.is_valid}")
            
        except Exception as e:
            logger.error(f"❌ LLM extraction failed: {str(e)[:100]}")
            import traceback
            logger.debug(traceback.format_exc())
            return {'success': False, 'url_valid': False, 'model': None}
        
        # ✅ VALIDATION
        if not search_extraction.is_valid or not search_extraction.download_url:
            logger.warning(f"❌ Invalid URL from LLM extraction")
            return {'success': False, 'url_valid': False, 'model': None}
        
        logger.info(f"🔗 Validating URL...")
        is_valid = validate_model_url_quick(search_extraction.download_url)
        
        if is_valid:
            logger.info(f"✓ Google: VALID URL!")
            return {
                'success': True,
                'url_valid': True,
                'model': {
                    'name': search_extraction.model_name,
                    'url': search_extraction.download_url,
                    'local_filename': search_extraction.model_name.replace(" ", "_") + os.path.splitext(search_extraction.download_url)[1],
                    'size': search_extraction.model_size,
                    'accuracy': search_extraction.accuracy,
                    'inference_time': search_extraction.inference_time,
                    'source': 'google'
                }
            }
        else:
            logger.warning(f"❌ Google: Un-downloadable URL (404?)")
            return {'success': True, 'url_valid': False, 'model': None}
    
    except Exception as e:
        logger.error(f"❌ Google exception: {str(e)[:100]}")
        import traceback
        logger.debug(traceback.format_exc())
        return {'success': False, 'url_valid': False, 'model': None}


# ============================================================================
# PARTE 3 della ricerca di modelli: VALIDAZIONE E UTILITY
# ============================================================================

def validate_model_url_quick(url: str, timeout: int = 5) -> bool:
    """Validazione rapida via HEAD request"""
    
    if not url or not any(url.endswith(ext) for ext in ['.h5', '.keras', '.onnx', '.tflite']):
        logger.warning(f"⚠️  URL non supportato: {url[:50]}")
        return False
    
    try:
        logger.debug(f"  → HEAD request a {url[:50]}...")
        response = requests.head(
            url,
            timeout=timeout,
            allow_redirects=True,
            headers={'User-Agent': 'Mozilla/5.0'}
        )
        
        if response.status_code == 200:
            content_length = response.headers.get('content-length')
            if content_length:
                size_mb = int(content_length) / (1024 * 1024)
                logger.debug(f"  ✓ 200 OK | {size_mb:.1f} MB")
            else:
                logger.debug(f"  ✓ 200 OK")
            return True
        
        elif response.status_code == 404:
            logger.warning(f"  ❌ 404 Not Found")
            return False
        else:
            logger.warning(f"  ⚠️  HTTP {response.status_code}")
            return False
    
    except requests.exceptions.Timeout:
        logger.warning(f"  ❌ Timeout ({timeout}s)")
        return False
    except requests.exceptions.ConnectionError:
        logger.warning(f"  ❌ Connection error")
        return False
    except Exception as e:
        logger.warning(f"  ❌ {str(e)[:50]}")
        return False


def extract_description(filename: str, path: str) -> str:
    """
    Extracts readable description from filename
    Example: "mobilenet_v2_224_224.h5" → "Mobilenet V2 224 224"
    """
    
    # Rimuovi estensione comune
    name = re.sub(r'\.(h5|keras|onnx|tflite)$', '', filename, flags=re.IGNORECASE)
    name = re.sub(r'_+', ' ', name)
    name = name.title()
    
    name = name.replace(" Tfs", " TFS")
    name = name.replace(" Tflite", " TFLite")
    name = name.replace(" Onnx", " ONNX")
    name = name.replace(" V2", " V2")
    name = name.replace(" V1", " V1")
    
    return name


def format_bytes(bytes_val: int) -> str:
    """
    Formats bytes into readable format
    Example: 1048576 → "1.0MB"
    """
    
    if bytes_val == 0:
        return "0B"
    
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.1f}{unit}"
        bytes_val /= 1024.0
    
    return f"{bytes_val:.1f}TB"


def parse_size_str(size_str: str) -> int:
    """
    Converts size string (e.g., "14.0MB") to bytes.
    Handles KB, MB, GB.
    """
    s = size_str.strip().upper()
    multiplier = 1
    
    if "KB" in s:
        multiplier = 1024
        s = s.replace("KB", "")
    elif "MB" in s:
        multiplier = 1024 * 1024
        s = s.replace("MB", "")
    elif "GB" in s:
        multiplier = 1024 * 1024 * 1024
        s = s.replace("GB", "")
    elif "B" in s:
        s = s.replace("B", "")
        
    try:
        return int(float(s.strip()) * multiplier)
    except ValueError:
        return 0


def get_task_based_default_model(task: str) -> Optional[dict]:
    """Returns the first available model for the task from PREDEFINED_MODELS"""
    
    if task not in PREDEFINED_MODELS:
        logger.warning(f"⚠️  Task not found: {task}")
        for task_key, info in PREDEFINED_MODELS.items():
            if info.get("models"):
                return info["models"][0]
        return None
    
    task_info = PREDEFINED_MODELS[task]
    models = task_info.get("models", [])
    
    if not models:
        logger.warning(f"⚠️  No models for task: {task}")
        return None
    
    default_model = models[0]
    logger.info(f"✓ Default model for '{task}': {default_model['name']}")
    
    return default_model



# ============================================================================
# LEGACY ENVIRONMENT SUPPORT
# ============================================================================

ARCHITECTURE_ENV_MAP = {
    'mobilenet': 'stm32legacy',
    'resnet': 'stm32legacy',
    'vgg': 'stm32legacy',
    'efficientnet': 'stm32legacy',
    'inception': 'stm32legacy',
    'yolo': 'stm32legacy',
    'har': 'stm32legacy',
    'custom': 'stm32legacy',
}

# CONDA_PYTHON_PATHS removed in favor of config.get_python_path()

def detect_architecture_from_model(model_path: str) -> str:
    """Detects architecture from model name"""
    model_name = os.path.basename(model_path).lower()
    if 'mobilenet' in model_name: return 'mobilenet'
    elif 'resnet' in model_name: return 'resnet'
    elif 'vgg' in model_name: return 'vgg'
    elif 'efficient' in model_name: return 'efficientnet'
    elif 'inception' in model_name: return 'inception'
    elif 'yolo' in model_name: return 'yolo'
    elif 'har' in model_name or 'activity' in model_name: return 'har'
    else: return 'custom'

def execute_in_environment(python_code: str, python_path: str, timeout: int = 60) -> dict:
    """Executes code in subprocess with specific python"""
    if not python_path:
        raise Exception("python_path required")
    
    result = subprocess.run(
        [python_path, '-c', python_code],
        capture_output=True,
        text=True,
        timeout=timeout,
        env={**os.environ, 'TF_CPP_MIN_LOG_LEVEL': '3'}
    )
    return {
        'success': result.returncode == 0,
        'stdout': result.stdout.strip(),
        'stderr': result.stderr.strip(),
        'returncode': result.returncode
    }

def inspect_model_via_legacy_env(model_path: str, config: RunnableConfig = None) -> Optional[dict]:
    """
    Inspects model using legacy env (to avoid Keras 3 crashes with old models)
    Returns dict with architecture info or None if it fails.
    """
    if not model_path.endswith(('.h5', '.keras')):
        logger.info(f"ℹ️  Skip legacy inspection for non-Keras format: {os.path.basename(model_path)}")
        return {
            'input_shape': 'Unknown',
            'output_shape': 'Unknown',
            'n_layers': 0,
            'total_params': 0,
            'trainable_params': 0,
            'model_size_mb': os.path.getsize(model_path) / (1024*1024),
            'has_batchnorm': False,
            'has_dropout': False,
            'format': os.path.splitext(model_path)[1]
        }

    try:
        arch = detect_architecture_from_model(model_path)
        
        cfg = Configuration.from_runnable_config(config)
        
        # Choose environment: .keras -> stm32 (Keras 3), .h5 -> stm32legacy (Keras 2) or stm32
        if model_path.endswith('.keras'):
            env_name = 'stm32'
        else:
            env_name = ARCH_ENVIRONMENT_MAP.get(arch, 'stm32legacy')
        
        python_path = cfg.get_python_path(env_name)
        
        if not python_path or "NOT_FOUND" in python_path:
            logger.warning(f"⚠️  Python path not found for {env_name}: {python_path}")
            return None
            
        logger.info(f"🔄 Inspecting via subprocess ({env_name})...")
        
        script = f"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import json
import sys

# GPU Memory Limit (Prevent OOM during inspection)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_virtual_device_configuration(
                gpu,
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)]
            )
    except RuntimeError:
        pass

try:
    model = tf.keras.models.load_model(r'{model_path}', compile=False)
    
    trainable = int(sum([tf.size(w).numpy() for w in model.trainable_weights]))
    
    info = {{
        'input_shape': str(model.input_shape),
        'output_shape': str(model.output_shape),
        'n_layers': len(model.layers),
        'total_params': int(model.count_params()),
        'trainable_params': trainable,
        'model_size_mb': os.path.getsize(r'{model_path}') / (1024*1024),
        'has_batchnorm': any(['BatchNormalization' in l.__class__.__name__ for l in model.layers]),
        'has_dropout': any(['Dropout' in l.__class__.__name__ for l in model.layers]),
        'env_used': '{env_name}'
    }}
    print("JSON_START" + json.dumps(info) + "JSON_END")
except Exception as e:
    print(f"ERROR:{{str(e)}}")
    sys.exit(1)
"""
        res = execute_in_environment(script, python_path, timeout=30)
        
        if res['success'] and "JSON_START" in res['stdout']:
            json_str = res['stdout'].split("JSON_START")[1].split("JSON_END")[0]
            return json.loads(json_str)
        else:
            logger.warning(f"⚠️  Legacy inspection failed: {res['stderr'][:100]}")
            return None
            
    except Exception as e:
        logger.warning(f"⚠️  Legacy inspection exception: {e}")
        return None

# ============================================================================
# NODE 3: DOWNLOAD MODEL
# ============================================================================
def download_model(state: MasterState, config: RunnableConfig = None) -> MasterState:
    """
    Wrapper to download the model from state.selected_model.
    Called by routing after online search is accepted.
    """
    
    logger.info("📥 download_model node (wrapper) started...")
    
    if not state.selected_model:
        logger.error("❌ selected_model not found!")
        cfg = Configuration.from_runnable_config(config)
        state.model_path = cfg.ai_model_path
        state.model_discovery_method = "default"
        return state
    
    logger.info(f"📦 Download: {state.selected_model['name']}")
    
    # ✅ CALL download_model_to_cache WITH the model
    state = download_model_to_cache(state, config, state.selected_model)
    
    return state

def download_model_to_cache(state: MasterState, config: RunnableConfig, model: dict) -> MasterState:
    """
    Download model with intelligent skip + ROBUST ANALYSIS
    """
    
    logger.info(f"📥 Downloading model: {model['name']}...")
    
    cache_dir = os.path.expanduser("~/.stm32_ai_models")
    os.makedirs(cache_dir, exist_ok=True)
    
    model_filename = model.get("local_filename")
    
    if not model_filename:
        logger.error("❌ local_filename not found in model!")
        cfg = Configuration.from_runnable_config(config)
        state.model_path = cfg.ai_model_path
        return state
    
    # ===== SECURITY: Sanitize filename to prevent path traversal =====
    from src.assistant.utils import sanitize_filename
    model_filename = sanitize_filename(model_filename)
    logger.debug(f"Sanitized filename: {model_filename}")
    
    cached_path = os.path.join(cache_dir, model_filename)
    
    # === CACHE VERIFICATION ===
    
    if os.path.exists(cached_path) and os.path.isfile(cached_path):
        logger.info(f"✓ Model in cache: {cached_path}")
        logger.info(f"  Size: {os.path.getsize(cached_path) / (1024*1024):.1f} MB")
        state.model_path = cached_path
        
        # ✅ PRINT MODEL ARCHITECTURE - ROBUST MODE
        logger.info(f"\n📋 MODEL ARCHITECTURE ANALYSIS (from cache)")
        logger.info("=" * 80)
        # ✅ OPTIMIZED ALGORITHM (USER REQUEST)
        # 1. Legacy Env Subprocess (First attempt)
        # 2. HDF5 Raw (Fallback)
        # 3. NO standard load_model()
        
        legacy_info = inspect_model_via_legacy_env(cached_path, config)
            
        if legacy_info:
            logger.info(f"✓ Analysis successful (via stm32legacy)!")
            logger.info(f"  Input: {legacy_info.get('input_shape')}")
            logger.info(f"  Output: {legacy_info.get('output_shape')}")
            logger.info(f"  Params: {legacy_info.get('total_params'):,}")
            if 'model_size_mb' in legacy_info:
                logger.info(f"  Size: {legacy_info['model_size_mb']:.2f} MB")
            logger.info(f"  BN: {'Yes' if legacy_info.get('has_batchnorm') else 'No'} | Dropout: {'Yes' if legacy_info.get('has_dropout') else 'No'}")
            state.model_info = legacy_info
            state.model_architecture = legacy_info # Sync for workflow5 compatibility
        else:
            logger.warning(f"⚠️  Legacy subprocess failed, trying HDF5 fallback...")
            
        # ← SECOND ATTEMPT: Raw HDF5 read (more robust, only if .h5 or .keras)
        if cached_path.endswith(('.h5', '.keras')):
            try:
                with h5py.File(cached_path, 'r') as f:
                    logger.info(f"\n📋 INTERNAL ANALYSIS (HDF5/Keras)")
                    logger.info(f"  Keys in file: {list(f.keys())}")
                    
                    if 'model_config' in f.attrs:
                        config_str = f.attrs['model_config']
                        if isinstance(config_str, bytes):
                            config_str = config_str.decode('utf-8')
                        config_dict = json.loads(config_str)
                        logger.info(f"  Model class: {config_dict.get('class_name', 'Unknown')}")
                        logger.info(f"  Backend: {config_dict.get('backend', 'Unknown')}")
                    
                    if 'model_weights' in f:
                        weights_group = f['model_weights']
                        n_layers = len(list(weights_group.keys()))
                        logger.info(f"  Number of layer groups: {n_layers}")
                    
                    logger.info("=" * 80 + "\n")
            except Exception as e2:
                logger.warning(f"⚠️  HDF5 analysis failed: {str(e2)[:100]}")
        else:
            logger.info(f"📋 Detected {os.path.splitext(cached_path)[1]} format. Structural analysis skipped.")
        
        return state
    
    # === PRIORITY 1: Direct URL ===
    
    direct_url = model.get("url")
    
    if direct_url:
        try:
            logger.info(f"📥 [1/2] Direct URL attempt: {direct_url[:80]}...")
            
            response = requests.get(direct_url, stream=True, timeout=30, allow_redirects=True)
            
            if response.status_code == 404:
                logger.warning(f"⚠️  URL returns 404 (Not Found)")
                return None
            else:
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                
                with open(cached_path, 'wb') as f:
                    downloaded = 0
                    last_log = 0
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total_size:
                                pct = (downloaded / total_size) * 100
                                if pct >= last_log + 20:
                                    last_log = (int(pct / 20)) * 20
                                    logger.info(f"  ⬇️  {last_log}%")
                
                # ✅ POST-DOWNLOAD VERIFICATION
                actual_size = os.path.getsize(cached_path)
                if actual_size == 0:
                    logger.error(f"❌ Download failed: Saved file is empty (0 bytes)!")
                    if os.path.exists(cached_path): os.remove(cached_path)
                    return None
                    
                logger.info(f"✓ Download completed! Size: {actual_size / (1024*1024):.1f} MB")
                
                # ===== SECURITY: Verify file integrity (SHA256) =====
                from src.assistant.utils import verify_file_integrity
                expected_hash = model.get("sha256")  # Optional field in model dict
                if expected_hash:
                    logger.info(f"🔐 Verifying file integrity...")
                    if not verify_file_integrity(cached_path, expected_hash):
                        logger.error(f"❌ Integrity check FAILED! Removing corrupted file.")
                        os.remove(cached_path)
                        raise SecurityError("Downloaded file failed SHA256 verification")
                    logger.info(f"✓ Integrity verified!")
                else:
                    logger.warning(f"⚠️  No SHA256 hash provided - skipping integrity check")
                
                # ✅ PRINT ARCHITECTURE - ROBUST MODE (same as above)
                logger.info(f"\n📋 MODEL ARCHITECTURE ANALYSIS (just downloaded)")
                logger.info("=" * 80)
                # ✅ OPTIMIZED ALGORITHM (USER REQUEST)
                # 1. Legacy Env Subprocess (First attempt)
                # 2. HDF5 Raw (Fallback)
                
                legacy_info = inspect_model_via_legacy_env(cached_path, config)
                
                if legacy_info:
                    logger.info(f"✓ Analysis successful (via {legacy_info.get('env_used', 'unknown')})!")
                    logger.info(f"  Input: {legacy_info['input_shape']}")
                    state.model_info = legacy_info
                    state.model_architecture = legacy_info # Sync for workflow5 compatibility
                else:
                    logger.warning(f"⚠️  Legacy subprocess failed, trying HDF5...")
                    try:
                        with h5py.File(cached_path, 'r') as f:
                            logger.info(f"  File contains: {list(f.keys())}")
                            if 'model_weights' in f:
                                logger.info(f"  Weight layers available")
                    except Exception as e2:
                        logger.warning(f"⚠️  HDF5 analysis failed: {str(e2)[:100]}")
                
                state.model_path = cached_path
                return state
            
        except Exception as e:
            logger.warning(f"⚠️  Download failed: {type(e).__name__}")
            if os.path.exists(cached_path):
                os.remove(cached_path)
    
    # === PRIORITY 2: Task-Based Fallback ===
    
    logger.error(f"❌ Download failed")
    cfg = Configuration.from_runnable_config(config)
    last_task = state.__dict__.get("last_task")
    
    if last_task:
        logger.info(f"🔄 Trying task-based fallback: {last_task}")
        fallback_model = get_task_based_default_model(last_task)
        
        if fallback_model:
            logger.info(f"✓ Fallback model: {fallback_model['name']}")
            fallback_url = fallback_model.get("url")
            
            if fallback_url:
                try:
                    logger.info(f"📥 Downloading fallback...")
                    fallback_filename = fallback_model.get("local_filename", f"fallback_{fallback_model['name'][:20]}.h5")
                    fallback_path = os.path.join(cache_dir, fallback_filename)
                    
                    response = requests.get(fallback_url, stream=True, timeout=30, allow_redirects=True)
                    
                    if response.status_code == 200:
                        total_size = int(response.headers.get('content-length', 0))
                        
                        with open(fallback_path, 'wb') as f:
                            downloaded = 0
                            last_log = 0
                            for chunk in response.iter_content(chunk_size=8192):
                                if chunk:
                                    f.write(chunk)
                                    downloaded += len(chunk)
                                    if total_size:
                                        pct = (downloaded / total_size) * 100
                                        if pct >= last_log + 20:
                                            last_log = (int(pct / 20)) * 20
                                            logger.info(f"  ⬇️  {last_log}%")
                        
                        logger.info(f"✓ Fallback download completed!")
                        state.model_path = fallback_path
                        state.model_discovery_method = "taskbased_fallback"
                        state.selected_model = fallback_model
                        
                        return state
                
                except Exception as e:
                    logger.warning(f"⚠️  Fallback download failed: {type(e).__name__}")
    
    logger.warning(f"⚠️  All fallbacks exhausted")
    state.model_path = cfg.ai_model_path
    state.model_discovery_method = "default"
    
    return state


# ============================================================================
# HELPER: GET DEFAULT MODEL BY TASK
# ============================================================================
def get_task_based_default_model(task_name: str) -> Optional[dict]:
    """
    Returns the first available model for the specific task.
    Intelligent fallback: if the user searched for "image_classification" 
    and the search fails, uses the first MobileNetV2 from PREDEFINED_MODELS
    """
    
    if task_name not in PREDEFINED_MODELS:
        logger.warning(f"⚠️  Task not found: {task_name}, using generic fallback")
        # Fallback to the first available model of any task
        for task, info in PREDEFINED_MODELS.items():
            if info["models"]:
                return info["models"][0]
        return None
    
    task_info = PREDEFINED_MODELS[task_name]
    models = task_info.get("models", [])
    
    if not models:
        logger.warning(f"⚠️  No models available for task: {task_name}")
        return None
    
    default_model = models[0]  # Take the first (lightest/fastest)
    logger.info(f"✓ Default model for '{task_name}': {default_model['name']}")
    
    return default_model


# ============================================================================
# ROUTING DECISION - DECIDES WHICH NODE TO USE
# ============================================================================

def model_selection_routing(state: MasterState) -> Literal[
    "run_analyze", 
    "download_model", 
    "search_recommendation_model" 
]:
    """
    Router that decides the next step after model selection.
    
    Now also supports the CUSTOMIZATION branch.
    """
    
    logger.info(f"📍 model_selection_routing:")
    logger.info(f"   - discovery_method: {state.model_discovery_method}")
    logger.info(f"   - search_iterations: {state.search_iterations}")
    logger.info(f"   - wants_customization: {getattr(state, 'wants_customization', False)}")
    
    # ============================================================
    # CASE 1: Default model (no search)
    # ============================================================
    if state.model_discovery_method == "default":
        logger.info("→ Default model, goes directly to analyze")
        return "run_analyze"
    
    # ============================================================
    # CASE 2: In search, search loop still available
    # ============================================================
    elif state.model_discovery_method == "search":
        if state.search_iterations < 3:
            logger.info(f"→ Search loop ({state.search_iterations}/3), searching again")
            return "search_recommendation_model"
        else:
            logger.info("→ Max search iterations reached, goes to analyze")
            return "run_analyze"
    
    # ============================================================
    # CASE 3: Model found (github, google_search, taskbased_fallback)
    # ============================================================
    else:
            logger.info("→ Model found, goes to download_model")
            return "download_model"


def run_analyze(state: MasterState, config: RunnableConfig = None) -> MasterState:
    """
    ✨ Analyzes the model (customized OR original)
    
    Logic:
    - If customized: final_model_path
    - Otherwise: model_path (default)
    """
    
    logger.info("🔍 Executing model analysis...")
    cfg = Configuration.from_runnable_config(config)
    
    try:
        # ===== DETERMINE MODEL =====
        # Try final first, otherwise use original
        model_path = state.final_model_path if state.customization_applied else state.model_path
        model_type = "CUSTOMIZED" if state.customization_applied else "ORIGINAL"
        
        if not model_path or not os.path.exists(model_path):
            logger.error(f"❌ Model not found: {model_path}")
            state.analyze_success = False
            state.ai_error_message = f"Model not found: {model_path}"
            return state
        
        # ✅ FIX FOR Keras 3 (Environment 'stm32'): Convert to TFLite for stedgeai compatibility
        # stedgeai v2.x does not directly support Keras 3 models (even if saved as .h5)
        # We detect Keras 3 if the file is .keras OR if we know we are in 'stm32' environment
        is_keras3 = model_path.endswith('.keras') or state.conda_env == 'stm32'
        
        if is_keras3:
            logger.info("⚡ Keras 3 model detected. Starting TFLite conversion for stedgeai compatibility...")
            tflite_path = model_path.replace('.keras', '.tflite').replace('.h5', '.tflite') # The .replace() function is called twice in a row. 
            # First pass: Looks for .keras and replaces it with .tflite.
            # Second pass: Takes the result of the first and looks for .h5, replacing it with .tflite. 
            # This ensures that the final file has a .tflite extension regardless of the original format (.keras or .h5).
            
            if not os.path.exists(tflite_path): # If the file is already present, the system skips the entire conversion block. It means the conversion was already executed previously.
                conversion_script = f"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ.pop('TF_USE_LEGACY_KERAS', None)  # Allow keras3 to load cleanly

import tensorflow as tf

try:
    # Try keras3 first (env 'stm32'), fall back to tf.keras (env 'stm32legacy')
    try:
        import keras
        model = keras.models.load_model(r'{model_path}', compile=False)
    except Exception:
        model = tf.keras.models.load_model(r'{model_path}', compile=False)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(r'{tflite_path}', 'wb') as f:
        f.write(tflite_model)
    print("CONVERSION_OK")
except Exception as e:
    print(f"CONVERSION_ERROR:{{e}}")
"""
                python_path = cfg.get_python_path('stm32') # Usa env Keras 3
                # Use subprocess.run directly to avoid any state/type confusion
                import tempfile
                with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmpf:
                    tmpf.write(conversion_script)
                    tmp_script = tmpf.name
                try:
                    conv_result = subprocess.run(
                        [python_path, tmp_script],
                        capture_output=True, text=True, timeout=120
                    )
                    conv_stdout = conv_result.stdout + conv_result.stderr
                finally:
                    if os.path.exists(tmp_script):
                        os.remove(tmp_script)
                
                if "CONVERSION_OK" not in conv_stdout:
                    logger.error(f"❌ TFLite conversion failed: {conv_stdout[:500]}")
                    state.analyze_success = False
                    state.ai_error_message = f"TFLite conversion failed for Keras 3 model."
                    return state
                
                logger.info(f"✅ Conversion completed: {tflite_path}")
            
            # Use TFLite for analysis
            model_path = tflite_path
            state.model_path = tflite_path # Update the state so subsequent nodes use it
        
        logger.info(f"  Model ({model_type}): {model_path}")
        
        # ===== OUTPUT DIR =====
        analyze_dir = os.path.join(state.ai_output_dir, "report_analyze")
        os.makedirs(analyze_dir, exist_ok=True)
        
        # ===== EXECUTE =====
        cmd = [
            "stedgeai", "analyze",
            "--model", model_path,
            "--target", state.target,
            "--output", analyze_dir
        ]
        
        # ✅ FIX: Add compression if specified. 
        if state.compression: # fundamental. X-CUBE-AI has built-in quantization capabilities. If required by user, X-CUBE-AI uses this parameter to automatically apply compression/quantization techniques during analysis and C code generation.
             cmd.extend(["--compression", state.compression])
             logger.info(f"  Compression: {state.compression}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        state.analyze_success = (result.returncode == 0)
        
        if state.analyze_success:
            logger.info(f"✓ Analyze completed")
            state.analyze_report_dir = analyze_dir
            
            # ✅ REGISTRATION COMMIT: Save the model to the permanent catalog only if technical analysis succeeded.
            # This avoids registering broken links or models unsupported by ST tools.
            if state.is_new_registration and state.pending_model_entry:
                try:
                    # Load current catalog from JSON file
                    models = load_predefined_models()
                    new_entry = state.pending_model_entry.copy()
                    
                    # Extract category (e.g., image_classification) and remove it from model data
                    category = new_entry.pop("category", "other")
                    
                    # Create category in catalog if it doesn't exist yet
                    if category not in models:
                        models[category] = {
                            "description": category.replace("_", " ").title(),
                            "models": []
                        }
                    
                    # Anti-duplicate check: save only if URL is not already present in the category
                    if not any(m['url'] == new_entry['url'] for m in models[category]['models']):
                        models[category]["models"].append(new_entry)
                        save_predefined_models(models) # Physical write to disk (predefined_models.json)
                        logger.info(f"💾 Model '{new_entry['name']}' saved to permanent catalog.")
                    
                    # Reset state: registration concluded successfully
                    state.is_new_registration = False # Reset flag
                    state.pending_model_entry = None  # Clean up
                    
                except Exception as ex:
                    logger.error(f"⚠️ Error saving to catalog: {ex}")
        else:
            state.ai_error_message = result.stderr.strip() or f"Return code {result.returncode}"
            logger.error(f"✗ Analyze failed: {state.ai_error_message[:500]}")
    
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        state.analyze_success = False
        state.ai_error_message = str(e)
    
    return state


def run_validate(state: MasterState, config: RunnableConfig = None) -> MasterState:
    validate_file = os.path.join(state.ai_output_dir, "network_validate_report.txt")
    os.makedirs(os.path.dirname(validate_file), exist_ok=True)
    cmd = [
        "stedgeai", "validate",
        "--model", state.model_path,
        "--target", state.target,
        "--output", validate_file
    ]
    if state.compression:
        cmd.extend(["--compression", state.compression])
    res = subprocess.run(cmd, capture_output=True, text=True)
    state.validate_success = (res.returncode == 0)
    if not state.validate_success:
        state.ai_error_message = res.stderr.strip() or f"Return code {res.returncode}"
    state.validate_report = validate_file
    logger.info("✓ Validate completed" if state.validate_success else f"✗ Validate failed: {state.ai_error_message}")
    return state


def run_generate(state: MasterState, config: RunnableConfig = None) -> MasterState:
    code_dir = os.path.join(state.ai_output_dir, "code_resnet")
    os.makedirs(code_dir, exist_ok=True)
    cmd = [
        "stedgeai", "generate",
        "--model", state.model_path,
        "--target", state.target,
        "--output", code_dir
    ]
    
    if state.compression:
        cmd.extend(["--compression", state.compression])
    res = subprocess.run(cmd, capture_output=True, text=True)
    state.generate_success = (res.returncode == 0)
    if not state.generate_success:
        state.ai_error_message = res.stderr.strip() or f"Return code {res.returncode}"
    state.generate_code_dir = code_dir
    state.ai_code_dir = code_dir
    logger.info("✓ Generate completed" if state.generate_success else f"✗ Generate failed: {state.ai_error_message}")
    return state


def finalize_analysis(state: MasterState, config: RunnableConfig = None) -> MasterState:
    if state.analyze_success and state.validate_success and state.generate_success:
        print("✓ AI Analysis completed!")
        print(f" - Analyze report in: {state.analyze_report_dir}")
        print(f" - Validate report in: {state.validate_report}")
        print(f" - Generated code in: {state.generate_code_dir}")
    else:
        print(f"✗ AI Error: {state.ai_error_message}")
    return state


# ============================================================================
# NEW RESOURCE CONSTRAINT CHECK LOGIC
# ============================================================================


def get_mcu_limits(target_mcu: str) -> tuple[int, int]:
    """
    Returns (flash_limit_bytes, ram_limit_bytes) for the target MCU.
    Approximate but safe (conservative) values.
    """
    target = target_mcu.lower()
    
    if "stm32f4" in target or "f401" in target:
        # STM32F401: 256KB Flash, 64KB RAM
        return (256 * 1024, 64 * 1024)
    elif "stm32h7" in target or "h743" in target:
        # STM32H743: 2MB Flash, ~1MB RAM (for safe contiguous activations)
        return (2 * 1024 * 1024, 1024 * 1024)
    elif "stm32u5" in target:
        # STM32U5: 2MB Flash, 786KB RAM
        return (2 * 1024 * 1024, 768 * 1024)
    elif "stm32l4" in target:
         # STM32L4: 1MB Flash, 128KB RAM
        return (1024 * 1024, 128 * 1024)
    else:
        # Default safe fallback (assume F4)
        logger.warning(f"⚠️ Target MCU not recognized: {target_mcu}. Using default limits (F4).")
        return (256 * 1024, 64 * 1024)


def check_resource_constraints(state: MasterState, config: RunnableConfig = None) -> MasterState:
    """
    Analyzes the STEdgeAI report to verify if the model fits in the MCU.
    """
    logger.info("⚖️  Checking Resource Constraints...")
    
    if not state.analyze_success:
        logger.warning("⚠️  Analysis failed, impossible to verify constraints.")
        state.ai_error_message = (
            "Impossible to analyze the model with ST tools.\n"
            "This usually happens for unsupported models or conversion errors.\n"
            "The automation will return to model selection to allow you to choose another one."
        )
        state.resource_check_result = "error"
        return state

    report_path = os.path.join(state.analyze_report_dir, "network_analyze_report.txt")
    if not os.path.exists(report_path):
        # Fallback: search for any .txt file in the dir
        try:
            files = [f for f in os.listdir(state.analyze_report_dir) if f.endswith(".txt")]
            if files:
                report_path = os.path.join(state.analyze_report_dir, files[0])
            else:
                logger.error("❌ Report file not found.")
                state.resource_check_result = "error"
                return state
        except Exception:
             logger.error("❌ Report dir not found.")
             state.resource_check_result = "error"
             return state

    # Parse Report
    ram_usage = 0
    flash_usage = 0
    
    try:
        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # Search for pattern like: "activations  : 4917696 bytes" or "weights      : 8833768 bytes"
            # Or more complex patterns depending on version. We look for "activations" and "weights" / "total"
            
            # Example report:
            #  activations size   : 4917696 bytes (4802.44 KiB)
            #  weights size       : 8833768 bytes (8626.73 KiB)
            #  macc               : ...
            
            ram_match = re.search(r'(?i)activations\s*(?:size)?\s*:\s*(\d+)', content)
            if ram_match:
                ram_usage = int(ram_match.group(1))
            
            flash_match = re.search(r'(?i)weights\s*(?:size)?\s*:\s*(\d+)', content)
            if flash_match:
                flash_usage = int(flash_match.group(1))

            # If 0, try alternative patterns (total ram/flash report table)
            if ram_usage == 0:
                 ram_match = re.search(r'(?i)ram\s*:\s*(\d+)', content)
                 if ram_match: ram_usage = int(ram_match.group(1))

            if flash_usage == 0:
                 flash_match = re.search(r'(?i)flash\s*:\s*(\d+)', content)
                 if flash_match: flash_usage = int(flash_match.group(1))

    except Exception as e:
        logger.error(f"❌ Error parsing report: {e}")
        state.resource_check_result = "error"
        return state

    state.ram_usage = ram_usage
    state.flash_usage = flash_usage

    # Check Limits
    flash_limit, ram_limit = get_mcu_limits(state.target)
    
    logger.info(f"📊 Usage: RAM={format_bytes(ram_usage)} / {format_bytes(ram_limit)}")
    logger.info(f"📊 Usage: Flash={format_bytes(flash_usage)} / {format_bytes(flash_limit)}")
    
    ram_ratio = ram_usage / ram_limit
    flash_ratio = flash_usage / flash_limit
    max_ratio = max(ram_ratio, flash_ratio)
    
    # ✅ INTELLIGENT COMPRESSION ESCALATION LOGIC
    # Removed "very_high" as it caused stedgeai crash (unsupported flag)
    COMPRESSION_LEVELS = ["low", "medium", "high"]
    
    if max_ratio <= 1.0:
        # Model fits perfectly
        logger.info("✅ Resources OK (Fits in MCU)")
        state.resource_check_result = "ok"
        state.needs_compression_retry = False
        
    elif max_ratio <= 8.0:
        # Model doesn't fit, but compression might help (up to 8x overflow)
        logger.warning(f"⚠️  Resources Overflow ({max_ratio:.1f}x)")
        
        # Try to escalate compression
        try:
            current_idx = COMPRESSION_LEVELS.index(state.compression)
        except ValueError:
            current_idx = 1  # Default to "medium" if unknown
        
        if current_idx < len(COMPRESSION_LEVELS) - 1:
            # Can escalate to higher compression
            next_compression = COMPRESSION_LEVELS[current_idx + 1]
            logger.warning(f"🔄 Auto-retry with compression: {next_compression}")
            logger.info(f"   Current: {state.compression} → Next: {next_compression}")
            
            state.compression = next_compression
            state.needs_compression_retry = True
            state.resource_check_result = "retry"
            
        else:
            # Already at maximum compression (high)
            logger.error(f"❌ Still doesn't fit even with max compression ({state.compression})")
            logger.error(f"   Model requires {max_ratio:.1f}x more resources than available")
            state.resource_check_result = "critical"
            state.needs_compression_retry = False
            
    else:
        # Model is WAY too big (>8x overflow) - compression won't help
        logger.error(f"❌ Resources CRITICAL (Overflow {max_ratio:.1f}x) -> Model too big")
        logger.error(f"   The overflow is {max_ratio:.1f}x, which is beyond the capacity of even maximum compression (~8-10x).")
        logger.warning(f"   Skipping auto-retry to avoid wasting processing time on an impossible fit.")
        state.resource_check_result = "critical"
        state.needs_compression_retry = False
        
    # Set error message for critical failures
    if state.resource_check_result == "critical":
        state.ai_error_message = (
            f"Model requires {format_bytes(ram_usage)} RAM / {format_bytes(flash_usage)} Flash. "
            f"Target {state.target} has only {format_bytes(ram_limit)} RAM / {format_bytes(flash_limit)} Flash."
        )
        # Reset for new model selection
        state.model_discovery_method = "search" 
        state.search_iterations = 0 

    return state

# Intelligent Logic:
# -Fits: Proceed.
# -Warning (<8x overflow): Activate higher compression and retry.
# -Critical (>8x overflow): Block all and ask to choose a smaller model (e.g. ResNet -> MobileNet)


def resource_check_routing(state: MasterState) -> Literal["run_analyze", "run_validate", "run_generate", "choose_predefined_taskbased_model", "handle_resource_failure"]:
    """
    Decides the route based on constraints.
    Also handles automatic retry with higher compression.
    """
    res = getattr(state, "resource_check_result", "ok")

    # ── CASE: handle_resource_failure has already processed the decision ──────────
    # resource_check_result is reset to "resolved" by the function itself.
    # This avoids the loop when LangGraph re-executes the routing from the checkpoint
    # on a new VSCode session (otherwise "critical" → handle_resource_failure
    # → new interrupt → infinite loop).
    if res == "resolved":
        route = getattr(state, "route", "change_model")
        logger.info(f"✅ resource_check_routing: already resolved, following route='{route}'")
        # change_board returns to collect_project_info → use "run_validate" as passthrough
        # Actually the graph after handle_resource_failure has its edges, this
        # "resolved" case should never be reached in a normal run.
        # We handle it anyway for safety.
        return "choose_predefined_taskbased_model"

    # ✅ NEW: Check if we need to retry with higher compression
    if state.needs_compression_retry and res == "retry":
        logger.info(f"🔄 Routing back to analyze with compression: {state.compression}")
        return "run_analyze"  # Re-analyze with new compression level

    if res == "ok":
        return "run_validate"

    elif res == "warning":
        return "run_generate"

    else:  # critical or error
        if not getattr(state, "analyze_success", True):
            logger.error(f"❌ Technical Error during analysis: {getattr(state, 'ai_error_message', 'Unknown')}")
        else:
            logger.error("🚫 Model rejected due to hardware constraints.")
            logger.error(f"""⛔ MODEL TOO BIG FOR {state.target}!
                
Resource Details:
- RAM Required: {format_bytes(state.ram_usage)} (Max: {format_bytes(get_mcu_limits(state.target)[1])})
- Flash Required: {format_bytes(state.flash_usage)} (Max: {format_bytes(get_mcu_limits(state.target)[0])})

The automation returns to model selection forcing a more appropriate choice.""")

        return "handle_resource_failure"



def handle_resource_failure(state: MasterState, config: RunnableConfig = None) -> MasterState:
    """
    Asks the user if they want to change board or model after a resource failure.
    """
    logger.info("📋 Post-error resource decision: Change Board or Change Model?")
    
    ram_str   = format_bytes(state.ram_usage)   if getattr(state, 'ram_usage', 0)   else "N/A"
    flash_str = format_bytes(state.flash_usage) if getattr(state, 'flash_usage', 0) else "N/A"
    ram_lim, flash_lim = get_mcu_limits(getattr(state, 'target', ''))  if getattr(state, 'target', '') else (0, 0)
    target_str = getattr(state, 'target', 'MCU')

    prompt = {
        "instruction": (
            f"The model requires {ram_str} RAM and {flash_str} Flash, "
            f"but {target_str} only supports {format_bytes(ram_lim)} RAM / {format_bytes(flash_lim)} Flash.\n\n"
            "Change Board or Change Model?\n"
            "0 → Change Board (choose a more powerful MCU)\n"
            "1 → Change Model (choose a lighter model)\n"
            "2 → Stop Execution (End Workflow)"
        ),
        "options": [
            "Change Microcontroller (Board)",
            "Choose another AI model",
            "Stop Execution"
        ]
    }
    
    # user_response = interrupt(prompt)
    user_response = "2" # BYPASS End Workflow
    
    if isinstance(user_response, dict):
        user_text = user_response.get("response", user_response.get("input", str(user_response)))
    else:
        user_text = str(user_response)

    user_lower = user_text.lower()

    # Robust keyword-based classification (does not depend on LLM)
    BOARD_KEYWORDS  = ["board", "scheda", "mcu", "micro", "microcontroller", "change board", "0"]
    MODEL_KEYWORDS  = ["model", "modello", "other model", "choose", "light", "lighter", "1"]
    END_KEYWORDS    = ["end", "stop", "fine", "esci", "quit", "2"]

    board_score = sum(1 for kw in BOARD_KEYWORDS if kw in user_lower)
    model_score = sum(1 for kw in MODEL_KEYWORDS if kw in user_lower)
    end_score = sum(1 for kw in END_KEYWORDS if kw in user_lower)

    if end_score > 0:
        decision = "end_workflow"
    elif board_score > model_score:
        decision = "change_board"
    elif model_score > 0:
        decision = "change_model"
    else:
        # Fallback to LLM only if keywords are not enough
        try:
            llm_extractor = get_llm(config, structured_schema=ResolutionExtraction)
            analysis_prompt = f"""Analyze the response and choose between 'change_board' or 'change_model' or 'end_workflow'.
User response: "{user_text}"
Board keywords: board, scheda, mcu → change_board
Model keywords: modello, model, lighter → change_model
End keywords: stop, fine, end → end_workflow"""
            result = llm_extractor.invoke([
                SystemMessage(content="You are a technical intent classifier."),
                HumanMessage(content=analysis_prompt)
            ])
            decision = result.decision.lower()
            logger.info(f"🤖 LLM Decision: {decision} (confidence: {result.confidence:.2f})")
        except Exception as e:
            logger.warning(f"⚠️ LLM not available ({e}). Using 'change_model' as default.")
            decision = "change_model"

    logger.info(f"✅ Final decision: {decision} (input: '{user_text}')")

    if "board" in decision:
        state.route = "change_board"
        state.board_name = None
        state.mcu_series = ""
        logger.info("🧹 Reset board state for microcontroller change.")
    elif "model" in decision:
        state.route = "change_model"
        state.last_task = None
        state.selected_model = None
        state.model_discovery_method = "taskbased"
        state.model_accepted = False
        state.search_iterations = 0
        logger.info("🧹 Reset AI selection state for model change.")
    else:
        state.route = "end_workflow"
        logger.info("🛑 End workflow requested.")

    # IMPORTANT: reset resource_check_result to "resolved".
    # If LangGraph resumes from interrupt with a new VSCode session,
    # resource_check_routing is re-executed from the checkpoint. If the value
    # was still "critical", it routes back to handle_resource_failure
    # creating an infinite loop. With "resolved" the routing doesn't go there.
    state.resource_check_result = "resolved"

    return state


def add_custom_model_procedure(state: MasterState, config: RunnableConfig = None) -> MasterState:
    """
    Procedure to add a new model to the catalog.
    """
    logger.info("🆕 Starting new model registration procedure...")
    
    # 1. Ask details to the user
    prompt = {
        "instruction": """New AI Model Registration

Provide the following details separated by comma:
1. Category (e.g.: image_classification, object_detection, audio)
2. Model Name (e.g.: MobileNetV3 Small)
3. GitHub Link (URL Raw .h5, .onnx, .tflite, .keras)

Example:
"image_classification, MobileNetV3, https://github.com/.../model.keras"
        """
    }
    
    # user_response = interrupt(prompt)
    user_response = "mobilenetv1" # BYPASS
    if isinstance(user_response, dict):
        user_text = user_response.get("response", user_response.get("input", str(user_response)))
    else:
        user_text = str(user_response)
        
    # 2. Parsing with LLM
    cfg = Configuration.from_runnable_config(config)
    llm = get_llm(config)
    
    extraction_prompt = f"""Extract the details of the new model from the following user response:
"{user_text}"

Reply in JSON format with these fields:
- "category": category (lowercase, snake_case)
- "name": model name
- "url": complete Raw GitHub link
- "is_valid": true if the data seems sensible
"""
    
    response = llm.invoke(extraction_prompt)
    try:
        # Clean response if LLM puts markdown
        clean_content = response.content.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean_content)
    except Exception as e:
        logger.error(f"❌ Error parsing new model data: {e}")
        return state

    if not data.get("is_valid") or not data.get("url"):
        logger.error("❌ Invalid model data or missing URL.")
        return state

    # 3. URL and Metadata Validation
    url = data["url"]
    logger.info(f"🔍 Validating URL: {url}")
    
    try:
        res = requests.head(url, timeout=5, allow_redirects=True)
        if res.status_code == 200:
            size_bytes = int(res.headers.get('Content-Length', 0))
            size_str = format_bytes(size_bytes) if size_bytes > 0 else "N/A"
        else:
            logger.warning(f"⚠️ URL responds with status {res.status_code}. Proceed anyway?")
            size_str = "N/A"
    except Exception as e:
        logger.warning(f"⚠️ URL connection error: {e}")
        size_str = "N/A"

    # 4. Update Catalog
    models = load_predefined_models()
    category = data["category"]
    
    if category not in models:
        models[category] = {
            "description": category.replace("_", " ").title(),
            "models": []
        }
    
    new_entry = {
        "name": data["name"],
        "local_filename": data["name"].lower().replace(" ", "_") + os.path.splitext(url)[1],
        "size": size_str,
        "accuracy": "N/A (User Provided)",
        "inference_time": "N/A",
        "url": url,
        "category": category # Temporary to save it later
    }
    
    logger.info(f"⏳ Model '{data['name']}' waiting for technical validation...")
    
    # Set the new model as selected to proceed immediately
    state.selected_model = new_entry
    state.pending_model_entry = new_entry
    state.is_new_registration = True
    
    state.model_path = "" # Will be downloaded in download_model node
    state.model_discovery_method = "default" # Pretend it's predefined now
    
    return state
