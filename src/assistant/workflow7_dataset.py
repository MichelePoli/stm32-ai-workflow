# ============================================================================
# WORKFLOW 7: DATASET SELECTION & MANAGEMENT
# ============================================================================
# Module for selecting and downloading real datasets
#
# Responsibilities:
#   - Ask the user for the data source (Real, Synthetic, Both)
#   - Show a menu of predefined datasets based on the task (Audio/Vision)
#   - Download real datasets (e.g. CIFAR-10, MNIST, SpeechCommands)
#
# Dependencies: tensorflow, keras, requests

import os
import shutil
import logging
import json
from typing import Literal, Optional, List, Dict, Any
from datetime import datetime

from langgraph.types import interrupt
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableConfig
from typing import Literal, Optional, List, Dict, Any, Union

class DatasetRegistration(BaseModel):
    """Schema for registering a new dataset via URL"""
    name: str = Field(description="Readable name of the dataset")
    key: str = Field(description="Unique key (snake_case, e.g. my_custom_data)")
    category: Literal["vision", "audio", "object_detection", "human_activity_recognition"] = Field(
        description="Dataset category"
    )
    url: str = Field(description="Direct URL for download (zip, tar.gz)")
    description: str = Field(description="Brief description of the dataset")
    expected_shape: Optional[List[int]] = Field(
        default=None, 
        description="Expected shape of inputs (e.g. [224, 224, 3]). Optional."
    )

from src.assistant.configuration import Configuration
from src.assistant.state import MasterState
import numpy as np

import requests
import tarfile
import zipfile
import io
from tqdm import tqdm
import tensorflow as tf

logger = logging.getLogger(__name__)

# ============================================================================
# RESOURCE HELPERS
# ============================================================================

def get_resource_path(filename: str) -> str:
    """Returns the absolute path of a resource in the resources folder."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    res_path = os.path.join(base_dir, "resources", filename)
    return res_path

def load_dataset_catalog() -> dict:
    """Loads the dataset catalog from the JSON file."""
    path = get_resource_path("predefined_datasets.json")
    if not os.path.exists(path):
        logger.warning(f"⚠️ Dataset catalog not found in {path}, returning empty.")
        return {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"❌ Error loading dataset catalog: {e}")
        return {}

def load_dataset_mapping() -> dict:
    """Loads the model-to-dataset mapping from the JSON file."""
    path = get_resource_path("dataset_mapping.json")
    if not os.path.exists(path):
        logger.warning(f"⚠️ Dataset mapping not found in {path}, returning empty.")
        return {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"❌ Error loading dataset mapping: {e}")
        return {}

def save_dataset_catalog(catalog: dict):
    """Saves the dataset catalog to the JSON file."""
    path = get_resource_path("predefined_datasets.json")
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(catalog, f, indent=4, ensure_ascii=False)
    except Exception as e:
        logger.error(f"❌ Error saving dataset catalog: {e}")

def validate_url(url: str) -> bool:
    """Checks if a URL is reachable."""
    try:
        # User Agent to avoid blocks
        headers = {"User-Agent": "Mozilla/5.0 (STM32-Agent)"}
        response = requests.head(url, timeout=5, allow_redirects=True, headers=headers)
        if response.status_code >= 400:
            # Fallback to GET if HEAD is blocked (some servers do this)
            response = requests.get(url, timeout=5, stream=True, headers=headers)
        return response.status_code < 400
    except Exception:
        return False

# ============================================================================
# NODES
# ============================================================================

def decide_data_source(state: MasterState, config: RunnableConfig = None) -> MasterState:
    """Asks the user which data source to use"""
    
    logger.info("📊 Dataset Source Selection")
    
    # === LLM EXTRACTOR ===
    from src.assistant.utils import extract_user_response, get_llm
    cfg = Configuration.from_runnable_config(config)
    llm = get_llm(config)
    
    # Simplified prompt for source classification
    source_classification_prompt = """Classify the data source from the user's request.
    
1. REAL: The user wants to use famous or existing datasets (CIFAR, MNIST, etc.)
2. REGISTER: The user wants to provide a URL or register a new dataset
3. SYNTHETIC: The user wants to generate data artificially

Answer ONLY with one word: REAL, REGISTER, or SYNTHETIC. If unsure: null.
"""

    # --- Step 1: Try to detect from the initial message with keyword check ---
    # We DO NOT use the LLM here: it was too aggressive and "guessed" SYNTHETIC
    # even when the user had not expressly stated it.
    # Only explicit keywords in the message trigger auto-selection.
    initial_source = None
    if not state.user_response and state.message:
        msg_low = state.message.lower()
        # Explicit keywords for each source
        real_keywords    = ["real", "reale", "cifar", "mnist", "dataset", "predefined", "predefinit", "esistente", "existing"]
        # NOTE: Do NOT use generic verbs like "genera"/"generat" here — they match "Genera firmware"!
        # Only use specific synthetic-data terminology.
        synth_keywords   = ["synthetic", "sintetico", "artificiale", "artificial", "dati sintetici", "synthetic data", "generate data", "genera dati"]
        register_keywords = ["register", "url", "http", "link", "registra", "aggiungi", "add"]
        
        if any(k in msg_low for k in synth_keywords):
            initial_source = "synthetic"
        elif any(k in msg_low for k in register_keywords):
            initial_source = "register"
        elif any(k in msg_low for k in real_keywords):
            initial_source = "real"
        # If no keyword found → initial_source remains None → asks the user
        
        if initial_source:
            logger.info(f"🤖 Source detected in the initial message: {initial_source}")
        else:
            logger.info("ℹ️ No source detected in the message — asking the user")


    # --- Step 2: Verification and Interrupt ---
    if not initial_source:
        resume_value = None
        if not state.user_response:
            prompt = {
                "instruction": """Which data source do you want to use for fine-tuning?

Options:
1. **Real Dataset**: Select from predefined ones (CIFAR, MNIST, SpeechCommands)
2. **Register New**: Add via direct URL
3. **Synthetic Data**: Generate artificial data now (sine, noise, etc.)

What do you prefer? (1, 2 or 3)""",
            }
            # Suggestion if the user has a past preference
            last_source = state.persistent_context.get("last_dataset_source", "Synthetic is recommended for fast testing.") if state.persistent_context else "Synthetic is recommended for fast testing."
            prompt["suggestion"] = f"💡 Last time you used: **{last_source}**."
            
        if not state.user_response:
            logger.info("⏸️ Interrupting for data source decision.")
            # resume_value = interrupt(prompt)
            resume_value = "1" # BYPASS - Real Dataset
            user_text = str(resume_value).strip().lower() if resume_value else ""
        else:
            # After resuming: use interrupt return value as priority
            if resume_value and str(resume_value).strip():
                user_text = str(resume_value).strip().lower()
            else:
                user_text = extract_user_response(state.user_response).lower()
        state.user_response = ""
        
        if "1" in user_text or "real" in user_text or "predefini" in user_text:
            state.dataset_source = "real"
        elif "2" in user_text or "register" in user_text or "url" in user_text:
            state.dataset_source = "register"
        elif "3" in user_text or "synthetic" in user_text or "gener" in user_text:
            state.dataset_source = "synthetic"
        else:
            # Re-invoke LLM on the response if not trivial
            res = llm.invoke([
                SystemMessage(content=source_classification_prompt),
                HumanMessage(content=f"Response: {user_text}")
            ])
            source_text = res.content.strip().upper()
            if "REAL" in source_text: state.dataset_source = "real"
            elif "REGISTER" in source_text: state.dataset_source = "register"
            elif "SYNTHETIC" in source_text: state.dataset_source = "synthetic"
            else: state.dataset_source = "synthetic" # Default
    else:
        state.dataset_source = initial_source

    logger.info(f"✅ Selected: {state.dataset_source}")
    return state


def register_custom_dataset(state: MasterState, config: RunnableConfig = None) -> MasterState:
    """Allows the user to register a new dataset by providing a URL"""
    
    logger.info("➕ Registering new dataset...")
    
    prompt_text = """Provide the information for the new dataset.
Required format:
- Name: [Dataset Name]
- Key: [snake_case_key]
- Category: [vision/audio/object_detection/human_activity_recognition]
- URL: [Direct URL to the .zip or .tar.gz file]
- Description: [Brief description]
"""
    
    from src.assistant.utils import extract_user_response
    resume_value = None
    if not state.user_response or state.user_response.strip() == "":
        # resume_value = interrupt({
        #     "instruction": prompt_text,
        #     "hint": "You can write in natural language, I will extract the data."
        # })
        resume_value = "fruit_360" # BYPASS
    
    # Use interrupt return value as priority
    if resume_value and str(resume_value).strip():
        user_input = str(resume_value).strip()
    else:
        user_input = extract_user_response(state.user_response)
    state.user_response = "" # Clear
    
    logger.info(f"📝 User response: {user_input[:100]}")
    
    # Parse with LLM structured output
    from src.assistant.utils import get_llm
    llm = get_llm(config, structured_schema=DatasetRegistration)
    try:
        info = llm.invoke([
            SystemMessage(content="You are an MLOps expert. Extract the dataset information from the user input."),
            HumanMessage(content=user_input)
        ])
        
        logger.info(f"🧐 URL Validation: {info.url}")
        if not validate_url(info.url):
            logger.warning(f"⚠️ URL unreachable or invalid: {info.url}")
            # We don't block, but warn
            
        # Update catalog
        catalog = load_dataset_catalog()
        cat_key = info.category
        if cat_key not in catalog:
            catalog[cat_key] = {"description": cat_key.replace("_", " ").title(), "datasets": {}}
        
        catalog[cat_key]["datasets"][info.key] = {
            "name": info.name,
            "description": info.description,
            "url": info.url,
            "type": "audio" if cat_key == "audio" else "image" if cat_key != "human_activity_recognition" else "sensor",
            "size": "N/A (Custom)",
            "expected_shape": info.expected_shape,
            "note": "Added by user via URL"
        }
        
        save_dataset_catalog(catalog)
        logger.info(f"✅ Dataset '{info.name}' registered successfully in '{cat_key}'")
        
        # Set as selected
        state.real_dataset_name = info.key
        state.dataset_source = "real" # Proceed towards download
        
    except Exception as e:
        logger.error(f"❌ Error during registration: {e}")
        state.ai_error_message = f"Registration failed: {e}"
        state.dataset_source = "synthetic" # Fallback
        
    return state


def select_predefined_dataset(state: MasterState, config: RunnableConfig = None) -> MasterState:
    """
    Shows dataset menu based on the selected model's task.
    Automatically determines the most appropriate task_type.
    """
    
    logger.info("📊 Smart dataset selection based on model...")
    
    # ===== STEP 1: Determine task_type from selected model =====
    task_type = "vision"  # Default
    preferred_datasets = []
    mapping_notes = ""
    
    # Load dynamic mapping and catalog
    mapping_catalog = load_dataset_mapping()
    dataset_catalog = load_dataset_catalog()
    
    # Search mapping from saved last_task (user selected task)
    if state.last_task:
        mapping = mapping_catalog.get(state.last_task)
        if mapping:
            task_type = mapping["task_type"]
            preferred_datasets = mapping["preferred_datasets"]
            mapping_notes = mapping.get("notes", "")
            logger.info(f"✓ Task type determined by model: {task_type}")
            logger.info(f"  Preferred datasets: {preferred_datasets}")
        else:
            logger.warning(f"⚠️ Task '{state.last_task}' not mapped, using heuristics")
    
    # ===== Fallback heuristics (backward compatibility) =====
    if not preferred_datasets:
        logger.info("  Using keyword-based heuristics on task...")
        # Check if last_task is empty before using 'in' operator
        if state.last_task:
            if "audio" in state.last_task or "speech" in state.last_task or "sound" in state.last_task:
                task_type = "audio"
                preferred_datasets = ["speech_commands", "esc50", "fsdd"]
            elif "image" in state.last_task or "object" in state.last_task or "vision" in state.last_task:
                task_type = "vision"
                preferred_datasets = ["cifar10", "mnist"]
            elif "human" in state.last_task or "activity" in state.last_task or "har" in state.last_task:
                task_type = "human_activity_recognition"
                preferred_datasets = ["uci_har", "wisdm"]
            elif "detection" in state.last_task:
                task_type = "object_detection"
                preferred_datasets = ["roboflow_vehicles", "pascal_voc_2012"]
        else:
            # Default if last_task is empty
            logger.warning("⚠️ state.last_task is empty, using default vision")
            task_type = "vision"
            preferred_datasets = ["cifar10", "mnist"]
    
    # ===== STEP 2: Input shape compatibility check (optional) =====
    if state.model_architecture:
        input_shape = state.model_architecture.get('input_shape')
        if input_shape:
            logger.info(f"  Model input shape: {input_shape}")
            
            # Advanced heuristics based on input shape
            if isinstance(input_shape, (list, tuple)) and len(input_shape) == 3:
                h, w, c = input_shape
                
                # Audio spectrograms: typically small and mono-channel
                if c == 1 and (h < 100 or w < 100):
                    logger.info(f"  ✓ Input shape {input_shape} suggests audio (spectrogram)")
                    if task_type == "vision":  # Only if not already audio
                        task_type = "audio"
                        preferred_datasets = ["speech_commands", "fsdd", "esc50"]
                
                # HAR: 1D input or very small
                elif len(input_shape) == 2 or (h < 50 and w < 50):
                    logger.info(f"  ⚠️ Input shape {input_shape} might be for HAR (sensor data)")
    
    # ===== STEP 3: Select dataset from catalog =====
    category_info = dataset_catalog.get(task_type, dataset_catalog.get("vision", {}))
    options = category_info.get("datasets", {})
    
    if not options:
        logger.error(f"❌ No dataset found for task_type '{task_type}'")
        # Fallback to vision
        task_type = "vision"
        category_info = dataset_catalog.get("vision", {})
        options = category_info.get("datasets", {"cifar10": {}})
        preferred_datasets = ["cifar10"]
    
    # ===== STEP 4: Sort datasets (preferred first) =====
    # Show preferred datasets first, then the others
    all_keys = list(options.keys())
    
    # Filter preferred that actually exist in the catalog
    valid_preferred = [k for k in preferred_datasets if k in all_keys]
    other_keys = [k for k in all_keys if k not in valid_preferred]
    
    valid_keys = valid_preferred + other_keys
    
    # ===== STEP 5: Build menu with badges for recommended datasets =====
    menu_text = f"\n{'='*70}\n"
    menu_text += f"📊 REAL DATASETS FOR: {task_type.upper().replace('_', ' ')}\n"
    menu_text += f"{'='*70}\n"
    logger.info(f"  Menu datasets: {valid_keys}") # Debug log
    
    if mapping_notes:
        menu_text += f"💡 Note: {mapping_notes}\n\n"
    
    menu_text += "Choose a dataset:\n\n"
    
    for idx, key in enumerate(valid_keys, 1):
        info = options[key]
        
        # Badge for recommended datasets
        badge = "⭐ RECOMMENDED" if key in valid_preferred else ""
        note = info.get('note', '')
        name = info.get('name', key.replace('_', ' ').title())
        
        menu_text += f"{idx}. {name} ({key}): {info['description']}\n"
        menu_text += f"   📦 Size: {info['size']}"
        if badge:
            menu_text += f"  {badge}"
        menu_text += "\n"
        if note:
            menu_text += f"   💬 {note}\n"
        menu_text += "\n"
    
    menu_text += f"{'='*70}\n"
    
    # ===== STEP 6: Show selected model information =====
    if state.selected_model:
        model_name = state.selected_model.get('name', 'N/A')
        menu_text += f"\n🤖 Model: {model_name}\n"
    
    prompt = {
        "instruction": menu_text,
        "valid_options": ["(Type number / name)"] + valid_keys,
        "hint": "Enter the dataset number or name (e.g. 1 or cifar10)"
    }
    
    # === LLM EXTRACTOR ===
    from src.assistant.utils import extract_user_response, get_llm
    llm = get_llm(config)
    
    # --- Step 1: Try to use the initial message ---
    initial_selection = None
    if not state.user_response:
        # Check if one of the dataset names is in the trigger message
        msg_low = state.message.lower()
        for key in valid_keys:
            if key.lower() in msg_low:
                initial_selection = key
                logger.info(f"🤖 Dataset '{key}' detected in the initial message.")
                break

    # --- Step 2: Verification and Interrupt ---
    if not initial_selection:
        resume_value = None
        if not state.user_response:
            # Suggestion if the user has a past preference
            last_ds = state.persistent_context.get("last_real_dataset", "None") if state.persistent_context else "None"
            if last_ds != "None" and last_ds in valid_keys:
                prompt["suggestion"] = f"💡 Last time you used: **{last_ds}**. Do you want to use the same?"
            
        if not state.user_response:
            logger.info("⏸️ Interrupting for dataset selection.")
            # resume_value = interrupt(prompt)
            resume_value = "1" # BYPASS
            selection = str(resume_value).strip().lower() if resume_value else ""
        else:
            # Use interrupt return value as priority
            if resume_value and str(resume_value).strip():
                selection = str(resume_value).strip().lower()
            else:
                selection = extract_user_response(state.user_response).lower().strip()
        state.user_response = ""
    else:
        selection = initial_selection

    # (Number/name parsing logic remains the same and flows below)
    
    # ===== STEP 8: Parse user response =====
    # Default: first recommended dataset (or first available)
    if not selection or selection.strip() == "":
        selection = valid_keys[0] if valid_keys else "cifar10"
        logger.info(f"  No selection, using default: {selection}")
    
    # Fuzzy matching: search by name or by number
    selected_key = None
    
    # Try to interpret as number
    try:
        idx = int(selection) - 1
        if 0 <= idx < len(valid_keys):
            selected_key = valid_keys[idx]
            logger.info(f"  ✓ Dataset selected by index {idx+1}: {selected_key}")
    except ValueError:
        pass
    
    # If not a number, search by partial match in the name
    if not selected_key:
        for key in valid_keys:
            if key in selection or selection in key:
                selected_key = key
                logger.info(f"  ✓ Dataset selected by match: {selected_key}")
                break
    
    # Fallback: use first available
    if not selected_key:
        selected_key = valid_keys[0] if valid_keys else "cifar10"
        logger.warning(f"⚠️ Unrecognized dataset '{selection}', using default: {selected_key}")
    
    # ===== STEP 9: Model-dataset compatibility check =====
    if state.model_architecture and selected_key:
        compatibility_ok = check_dataset_model_compatibility(
            state.model_architecture.get('input_shape'),
            selected_key,
            task_type
        )
        if not compatibility_ok:
            logger.warning("⚠️ Dataset preprocessing/resizing might be necessary")
    
    # ===== STEP 10: Save selection =====
    state.real_dataset_name = selected_key
    logger.info(f"✅ Final selected dataset: {selected_key}")
    logger.info(f"   Task type: {task_type}")
    
    return state


def check_dataset_model_compatibility(model_input_shape, dataset_name: str, task_type: str) -> bool:
    """
    Checks if the dataset is compatible with the model input.
    Uses metadata in the catalog if present.
    """
    
    logger.info(f"🔍 Compatibility check: {dataset_name} vs {model_input_shape}")
    
    # 1. Load catalog to find expected shape
    catalog = load_dataset_catalog()
    dataset_info = None
    for category in catalog.values():
        if dataset_name in category.get("datasets", {}):
            dataset_info = category["datasets"][dataset_name]
            break
            
    if not dataset_info:
        logger.warning(f"⚠️ Dataset '{dataset_name}' not found in catalog for compatibility check.")
        return True # Proceed anyway
        
    expected_shape = dataset_info.get("expected_shape")
    
    if not model_input_shape:
        logger.info("  ℹ️  Model input shape not available, skipping compatibility check")
        return True
    
    # ===== If dataset has variable shape (None), always OK =====
    if expected_shape is None:
        logger.info(f"  ✓ Dataset '{dataset_name}' has variable dimensions (supports preprocessing)")
        return True
    
    # ===== Convert model_input_shape to tuple for comparison =====
    model_shape_tuple = None
    if isinstance(model_input_shape, list):
        model_shape_tuple = tuple(model_input_shape)
    elif isinstance(model_input_shape, tuple):
        model_shape_tuple = model_input_shape
    elif isinstance(model_input_shape, str):
        # Try parsing string like "(None, 224, 224, 3)"
        try:
            import ast
            parsed = ast.literal_eval(model_input_shape)
            if isinstance(parsed, (list, tuple)):
                model_shape_tuple = tuple(parsed)
        except:
            pass
            
    if model_shape_tuple is None:
        logger.warning(f"  ⚠️ Unrecognized input shape format: {type(model_input_shape)} ({model_input_shape})")
        return True # Proceed anyway
    
    # ===== Compare dimensions =====
    if expected_shape == model_shape_tuple:
        logger.info(f"  ✓✓ Perfect compatibility: dataset {expected_shape} = model {model_shape_tuple}")
        return True
    
    # ===== Different shape → needs resize =====
    logger.warning(f"  ⚠️ Shape incompatibility:")
    logger.warning(f"     Dataset '{dataset_name}': {expected_shape}")
    logger.warning(f"     Model requires: {model_shape_tuple}")
    
    # Specific suggestions
    if task_type == "vision":
        logger.info(f"  💡 Solution: Use resizing layer or preprocessing to adapt {expected_shape} → {model_shape_tuple}")
    elif task_type == "audio":
        logger.info(f"  💡 Solution: Modify spectrogram processing parameters (target_shape)")
    elif task_type in ["human_activity_recognition", "object_detection"]:
        logger.info(f"  💡 Solution: Configure window size or use data augmentation with resize")
    
    return False


def download_dataset(state: MasterState, config: RunnableConfig = None) -> MasterState:
    """Downloads the selected dataset using the dynamic catalog"""
    
    dataset_name = state.real_dataset_name
    logger.info(f"📥 Starting dataset download: {dataset_name}...")
    
    # Check disk space
    if not check_disk_space(state.base_dir, required_gb=5.0):
        # We proceed anyway but warn
        pass

    # Setup dir
    dataset_dir = os.path.join(state.base_dir, "data", "real_datasets", dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)
    state.real_dataset_path = dataset_dir
    
    # 1. Retrieve info from catalog
    catalog = load_dataset_catalog()
    dataset_info = None
    category_name = None
    
    for cat, info in catalog.items():
        if dataset_name in info.get("datasets", {}):
            dataset_info = info["datasets"][dataset_name]
            category_name = cat
            break
            
    if not dataset_info:
        logger.error(f"❌ Dataset '{dataset_name}' not found in catalog.")
        state.ai_error_message = f"Dataset {dataset_name} not found."
        return state

    url = dataset_info.get("url")
    keras_name = dataset_info.get("keras_name")
    tfds_name = dataset_info.get("tfds_name")
    
    try:
        # A. Keras Built-in
        if keras_name:
            logger.info(f"📦 Using Keras built-in dataset: {keras_name}")
            import tensorflow as tf
            if keras_name == "cifar10":
                (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
            elif keras_name == "mnist":
                (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
            elif keras_name == "fashion_mnist":
                (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
            else:
                raise ValueError(f"Keras dataset {keras_name} not directly supported.")
                
            np.save(os.path.join(dataset_dir, "x_train.npy"), x_train)
            np.save(os.path.join(dataset_dir, "y_train.npy"), y_train)
            np.save(os.path.join(dataset_dir, "x_test.npy"), x_test)
            np.save(os.path.join(dataset_dir, "y_test.npy"), y_test)
            logger.info(f"✅ Dataset saved in {dataset_dir}")

        # B. URL Download (Generic Archive)
        elif url:
            if "roboflow.com" in url:
                # Manual logic for Roboflow (already existing)
                logger.info(f"📥 Roboflow dataset detected")
                logger.info(f"")
                logger.info(f"⚠️  USER ACTION REQUIRED:")
                logger.info(f"   To download this dataset:")
                logger.info(f"")
                logger.info(f"   1. Visit: {url}")
                logger.info(f"   2. Create Roboflow account (free)")
                logger.info(f"   3. Select format: COCO JSON")
                logger.info(f"   4. Download and extract to: {dataset_dir}")
                logger.info(f"")
                
                # Save instructions
                with open(os.path.join(dataset_dir, "DOWNLOAD_INSTRUCTIONS.txt"), "w") as f:
                    f.write(f"Dataset: {dataset_name}\n")
                    f.write(f"Roboflow URL: {url}\n\n")
                    f.write(f"Manual Download Instructions:\n")
                    f.write(f"1. Visit: {url}\n")
                    f.write(f"2. Create free Roboflow account\n")
                    f.write(f"3. Select format: COCO JSON\n")
                    f.write(f"4. Download and extract to: {dataset_dir}\n\n")
                    kaggle_alt = dataset_info.get("kaggle_alternative", "")
                    if kaggle_alt:
                        f.write(f"Alternative (Kaggle):\n{kaggle_alt}\n")
                
                logger.warning(f"⚠️  Manual download required. Instructions saved in DOWNLOAD_INSTRUCTIONS.txt")
                
                # Save minimal metadata
                metadata = {
                    "dataset_name": dataset_name,
                    "url": url,
                    "note": dataset_info.get("note", ""),
                    "manual_download_required": True
                }
                with open(os.path.join(dataset_dir, "metadata.json"), "w") as f:
                    json.dump(metadata, f, indent=2)
            else:
                logger.info(f"⬇️ Download via URL: {url}")
                archive_name = url.split("/")[-1]
                archive_path = os.path.join(dataset_dir, archive_name)
                
                if not os.path.exists(archive_path):
                    download_file(url, archive_path)
                else:
                    logger.info(f"✅ Archive found: {archive_path}")
                
                extract_dir = os.path.join(dataset_dir, "extracted")
                if not os.path.exists(extract_dir):
                    logger.info(f"📦 Extracting to {extract_dir}...")
                    extract_archive(archive_path, extract_dir)
                else:
                    logger.info(f"✅ Extracted dir found")
                
                # Specific processing based on category
                processing_success = True
                if category_name == "audio":
                    logger.info("🎵 Processing audio spectrograms...")
                    process_speech_commands(extract_dir, dataset_dir)
                    logger.info(f"✅ Audio dataset processed")
                elif category_name == "human_activity_recognition":
                    logger.info("⌚ HAR dataset ready (extracted)")
                    # Metadata saving
                    metadata = {
                        "dataset_name": dataset_name,
                        "download_date": datetime.now().isoformat(),
                        "url": url,
                        "type": "sensor_data",
                        "note": dataset_info.get("note", "")
                    }
                    with open(os.path.join(dataset_dir, "metadata.json"), "w") as f:
                        json.dump(metadata, f, indent=2)
                else:
                    # If category is vision or object_detection (images), process automatically
                    if category_name in ["vision", "object_detection"]:
                        logger.info(f"🖼️  Attempting automatic processing for images dataset...")
                        process_generic_vision_dataset(extract_dir, dataset_dir)
                    else:
                        logger.info(f"✅ Generic dataset ready in {extract_dir}")
                
                # 🧹 CLEANUP: Remove extracted folder and archive to save space
                try:
                    if os.path.exists(extract_dir):
                        logger.info(f"🧹 Cleanup: Removed {extract_dir}")
                        shutil.rmtree(extract_dir)
                    if os.path.exists(archive_path):
                        logger.info(f"🧹 Cleanup: Removed {archive_path}")
                        os.remove(archive_path)
                except Exception as cleanup_err:
                    logger.warning(f"⚠️ Error during cleanup: {cleanup_err}")

        # C. TFDS (TensorFlow Datasets)
        elif tfds_name:
            logger.info(f"📥 Loading via TFDS: {tfds_name}")
            
            try:
                import tensorflow_datasets as tfds
                
                # Download dataset con tfds (automatic)
                logger.info(f"⬇️  Loading from TFDS: {tfds_name}")
                logger.info(f"   This may take a while for first download (~{dataset_info.get('size', 'unknown size')})...")
                
                # Load dataset with info
                ds_train, ds_info = tfds.load(
                    tfds_name,
                    split='train',
                    with_info=True,
                    data_dir=dataset_dir  # Save in specific directory
                )
                
                # Check if validation split exists
                splits = ds_info.splits
                ds_validation = None
                if 'validation' in splits:
                    ds_validation = tfds.load(
                        tfds_name,
                        split='validation',
                        data_dir=dataset_dir
                    )
                elif 'test' in splits: # Fallback to test if no validation
                    ds_validation = tfds.load(
                        tfds_name,
                        split='test',
                        data_dir=dataset_dir
                    )
                
                logger.info(f"✅ TFDS {tfds_name} loaded successfully!")
                logger.info(f"   Train samples: {ds_info.splits['train'].num_examples}")
                if ds_validation:
                    val_split_name = 'validation' if 'validation' in splits else 'test'
                    logger.info(f"   {val_split_name.capitalize()} samples: {ds_info.splits[val_split_name].num_examples}")
                logger.info(f"   Features: {ds_info.features}")
                
                # Save metadata
                metadata = {
                    "dataset_name": dataset_name,
                    "download_date": datetime.now().isoformat(),
                    "source": "TensorFlow Datasets (tfds)",
                    "tfds_name": tfds_name,
                    "num_train": int(ds_info.splits['train'].num_examples),
                    "num_validation": int(ds_info.splits.get('validation', ds_info.splits.get('test', {'num_examples': 0}))['num_examples']),
                    "features": str(ds_info.features),
                    "note": dataset_info.get("note", "")
                }
                
                with open(os.path.join(dataset_dir, "metadata.json"), "w") as f:
                    json.dump(metadata, f, indent=2)
                
                # Save info on how to use the dataset
                with open(os.path.join(dataset_dir, "USAGE_INFO.txt"), "w") as f:
                    f.write(f"{dataset_name} Dataset (via TensorFlow Datasets)\n\n")
                    f.write(f"To load this dataset in your code:\n\n")
                    f.write(f"import tensorflow_datasets as tfds\n\n")
                    f.write(f"# Load train split\n")
                    f.write(f"ds_train = tfds.load('{tfds_name}', split='train', data_dir='{dataset_dir}')\n\n")
                    if ds_validation:
                        f.write(f"# Load {val_split_name} split\n")
                        f.write(f"ds_validation = tfds.load('{tfds_name}', split='{val_split_name}', data_dir='{dataset_dir}')\n\n")
                    f.write(f"Features:\n{ds_info.features}\n")
                
                logger.info(f"✅ TFDS {tfds_name} setup completed")
                logger.info(f"💡 Usage instructions saved in USAGE_INFO.txt")
                if category_name == "object_detection":
                    logger.info(f"⚠️  Note: Dataset includes bounding boxes and segmentation masks")
                    
            except ImportError:
                logger.error("❌ tensorflow_datasets not installed!")
                logger.info("   Install with: pip install tensorflow-datasets")
                raise
            
    except Exception as e:
        logger.error(f"❌ Error during download/processing: {e}")
        state.ai_error_message = str(e)
        # Fallback dummy file
        with open(os.path.join(dataset_dir, "README.txt"), "w") as f:
            f.write(f"Dataset {dataset_name} download/processing failed: {e}")
        
    return state


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def download_file(url: str, dest_path: str):
    """Downloads file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024 * 1024 # 1MB
    
    with open(dest_path, 'wb') as f, tqdm(
        desc=dest_path,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(block_size):
            size = f.write(data)
            bar.update(size)

def extract_archive(file_path: str, extract_to: str):
    """Extracts .tar.gz or .zip"""
    os.makedirs(extract_to, exist_ok=True)
    if file_path.endswith("tar.gz") or file_path.endswith(".tgz"):
        with tarfile.open(file_path, "r:gz") as tar:
            tar.extractall(path=extract_to)
    elif file_path.endswith(".zip"):
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)

def audio_to_spectrogram(file_path: str, target_shape=(32, 32)) -> Optional[np.ndarray]:
    """
    Reads a WAV file, computes STFT spectrogram, resizes to target_shape.
    Returns (H, W, 1) array normalized [0,1].
    """
    try:
        # 1. READ WAV FILE (Load & Resample)
        # Decodes the WAV file. Note: In a real scenario, this should also handle resampling
        # if the input rate != 16kHz (e.g. using tfio.audio.resample).
        audio_binary = tf.io.read_file(file_path)
        audio, sample_rate = tf.audio.decode_wav(audio_binary)
        
        # 2. FIX LENGTH (Padding/Truncation)
        # Ensures all inputs have exactly the same length (1 second @ 16kHz = 16000 samples).
        # This is CRITICAL for Neural Networks which expect fixed-size input tensors.
        desired_samples = 16000
        audio = tf.squeeze(audio, axis=-1) # Remove channel dim: (N, 1) -> (N,)
        
        if tf.shape(audio)[0] < desired_samples:
            # If too short: Pad with zeros (silence) at the end
            paddings = [[0, desired_samples - tf.shape(audio)[0]]]
            audio = tf.pad(audio, paddings)
        else:
            # If too long: Truncate to the first 1 second
            audio = audio[:desired_samples]
            
        # 3. FEATURE EXTRACTION (STFT Spectrogram)
        # Converts time-domain signal (waveform) to frequency-domain (image).
        # Parameters (frame_length, frame_step) control time/freq resolution.
        # Output shape approx: (124, 129)
        stft = tf.signal.stft(audio, frame_length=255, frame_step=128)
        spectrogram = tf.abs(stft)
        
        # Add channel dim for CNN -> (Time, Freq, 1)
        # This makes it compatible with 2D Convolution layers (like an image).
        spectrogram = tf.expand_dims(spectrogram, axis=-1)
        
        # 4. RESIZE (Adaptation to Model Input)
        # Resizes the large spectrogram to a smaller 32x32 image.
        # This drastically reduces RAM usage on STM32 while keeping key features.
        spectrogram = tf.image.resize(spectrogram, target_shape)
        
        # 5. NORMALIZATION
        # Scales values to [0, 1] range. 
        # Essential for model convergence and 8-bit quantization later.
        max_val = tf.reduce_max(spectrogram)
        if max_val > 0:
            spectrogram = spectrogram / max_val
            
        return spectrogram.numpy()
        
    except Exception as e:
        # logger.warning(f"Error processing {file_path}: {e}")
        return None

def process_speech_commands(extract_dir: str, output_dir: str, target_shape=(32, 32)):
    """
    Processes Google Speech Commands dataset.
    Structure: extracted/speech_commands_v0.02/word/file.wav
    """
    # Find real root (often there is an intermediate folder)
    # For speech commands v0.02 it's usually direct or in a folder
    # We look for folders which are labels (e.g. "yes", "no", "up")
    
    # Keywords to use (subset for simplicity or all)
    # We use the 10 standard + silence/unknown if we want, but for now we take the existing folders
    # Filter system folders or files
    
    root_search = extract_dir
    # If there is only one folder inside extracted, enter there
    entries = os.listdir(extract_dir)
    if len(entries) == 1 and os.path.isdir(os.path.join(extract_dir, entries[0])):
        root_search = os.path.join(extract_dir, entries[0])
        
    logger.info(f"📂 Scanning {root_search} for classes...")
    
    classes = [d for d in os.listdir(root_search) 
               if os.path.isdir(os.path.join(root_search, d)) and d != "_background_noise_"]
    classes.sort()
    
    logger.info(f"✓ Found {len(classes)} classes: {classes}")
    
    X = []
    y = []
    
    # Limit samples per class for speed/memory if needed
    MAX_SAMPLES_PER_CLASS = 500 
    
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    
    for cls_name in classes:
        cls_dir = os.path.join(root_search, cls_name)
        files = [f for f in os.listdir(cls_dir) if f.endswith('.wav')]
        
        # Shuffle and limit
        import random
        random.shuffle(files)
        files = files[:MAX_SAMPLES_PER_CLASS]
        
        logger.info(f"  Processing class '{cls_name}' ({len(files)} samples)...")
        
        for f in files:
            wav_path = os.path.join(cls_dir, f)
            spec = audio_to_spectrogram(wav_path, target_shape)
            if spec is not None:
                X.append(spec)
                y.append(class_to_idx[cls_name])
                
    # Convert to numpy
    X = np.array(X, dtype='float32')
    y = np.array(y, dtype='int32')
    
    logger.info(f"✓ Processed Total: {len(X)} samples. Shape: {X.shape}")
    
    # Save
    np.save(os.path.join(output_dir, "x_train.npy"), X)
    np.save(os.path.join(output_dir, "y_train.npy"), y)
    
    # Save class names mapping
    with open(os.path.join(output_dir, "classes.json"), "w") as f:
        json.dump(class_to_idx, f)

def process_generic_vision_dataset(extract_dir: str, output_dir: str, target_shape=(224, 224), max_samples=5000):
    """
    Scans an extracted folder for images and converts them to .npy.
    Infers classes from subfolders.
    """
    logger.info(f"📁 Generic image scanning in {extract_dir}...")
    
    # Supported extensions
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    
    # 1. Find all images and map classes
    image_paths = []
    for root, dirs, files in os.walk(extract_dir):
        for f in files:
            if f.lower().endswith(valid_exts):
                image_paths.append(os.path.join(root, f))
                
    if not image_paths:
        logger.warning("⚠️  No images found in the extracted archive.")
        return

    # 2. Class inference from parent folder name
    # Assume structure: root/class/image.jpg
    path_to_class = {}
    for p in image_paths:
        cls_name = os.path.basename(os.path.dirname(p))
        if not cls_name or cls_name == os.path.basename(extract_dir):
            cls_name = "default_class"
        path_to_class[p] = cls_name
        
    classes = sorted(list(set(path_to_class.values())))
    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    
    logger.info(f"✓ Found {len(image_paths)} images across {len(classes)} classes.")
    logger.info(f"✓ Classes: {classes[:10]} {'...' if len(classes) > 10 else ''}")

    # 3. Shuffle and Limit for performance/memory
    import random
    random.shuffle(image_paths)
    image_paths = image_paths[:max_samples]
    
    X = []
    y = []
    
    logger.info(f"⚙️  Processing {len(image_paths)} samples...")
    
    X = []
    y = []
    
    for p in image_paths:
        try:
            img = tf.io.read_file(p)
            img = tf.image.decode_image(img, channels=3, expand_animations=False)
            img = tf.image.resize(img, target_shape[:2])
            # Save in uint8 [0, 255] to save 4x disk space (float32 -> uint8)
            img = tf.cast(img, tf.uint8)
            X.append(img.numpy())
            y.append(class_to_idx[path_to_class[p]])
        except Exception as e:
            continue
            
    if not X:
        logger.error("❌ Error: No valid images processed.")
        return
        
    X = np.array(X, dtype='uint8')
    y = np.array(y, dtype='int32')
    
    # 4. Save
    np.save(os.path.join(output_dir, "x_train.npy"), X)
    np.save(os.path.join(output_dir, "y_train.npy"), y)
    
    # Manual split for validation (20%)
    split_idx = int(len(X) * 0.8)
    np.save(os.path.join(output_dir, "x_test.npy"), X[split_idx:])
    np.save(os.path.join(output_dir, "y_test.npy"), y[split_idx:])
    
    with open(os.path.join(output_dir, "classes.json"), "w") as f:
        json.dump(class_to_idx, f, indent=2)
        
    logger.info(f"✅ Processing completed. Saved {len(X)} samples (uint8) in {output_dir}")

def check_disk_space(path: str, required_gb: float = 5.0) -> bool:
    """Checks if there is enough disk space."""
    import shutil
    total, used, free = shutil.disk_usage(path)
    free_gb = free / (2**30)
    if free_gb < required_gb:
        logger.warning(f"⚠️  Critical disk space: {free_gb:.2f} GB available. Required: {required_gb} GB.")
        return False
    return True
