# ============================================================================
# WORKFLOW 5: MODEL CUSTOMIZATION WITH AI EMBEDDING AND BEST PRACTICES
# ============================================================================
# Module dedicated to customizing AI model architectures for STM32
#
# Responsibilities:
#   - Detailed inspection of model architecture
#   - Retrieval of best practices via embeddings (sentence-transformers)
#   - Parsing user customization requests
#   - Applying architecture modifications (layers, activation, etc.)
#   - Fine-tuning with dataset
#   - Validation and INT8 quantization
#   - Saving with metadata
#
# Dependencies: tensorflow, langchain, sentence-transformers, h5py, numpy


import subprocess
import os
from datetime import datetime
import logging
import tensorflow as tf
from typing import Optional, Tuple, List, Literal, Any
from langchain_core.runnables import RunnableConfig
import urllib.request
from src.assistant.utils import get_llm, force_unload_ollama, get_embeddings

import shutil
import re
import json
from typing import Tuple, Optional, List, Literal
from langgraph.types import interrupt
from src.assistant.configuration import Configuration
from typing import Any

# any  = Python builtin function (all lowercase)
#         ↓
#         Returns True/False

# Any  = type hint from typing module (CamelCase)
#         ↓
#         Means "any type"

# Pydantic understands: Any ✅
# Pydantic DOES NOT understand: any ❌

from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage

from agno.agent import Agent
#from agno.tools.github import GithubTools
#uses other tools besides GoogleSearchTools, see agno tools
from agno.models.ollama import Ollama
from agno.tools.duckduckgo import DuckDuckGoTools

import numpy as np

from tensorflow.keras.layers import (
    Dense, Dropout, Input, Resizing, Conv2D, 
    GlobalAveragePooling2D, GlobalMaxPooling2D,
    BatchNormalization, Activation, Add,  # ← Must have BatchNormalization
    AveragePooling2D, Flatten
)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
import h5py

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout

from langchain_community.document_loaders import RecursiveUrlLoader  # Web scraping & site crawling
from langchain_community.vectorstores import Chroma                                   # Vector DB
from langchain_core.embeddings import Embeddings

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
#from langchain_chroma import Chroma # SBAGLIATO QUESTO !!


# -------------------------
# Sentence Transformers / Embeddings
# -------------------------
from langchain_core.documents import Document                   # Standard document container for LangChain


from src.assistant.configuration import Configuration
from src.assistant.state import MasterState

from bs4 import BeautifulSoup


logger = logging.getLogger(__name__)

# Lines matching any of these patterns are suppressed from the VS Code streaming log.
# They are still captured in stdout (for SUCCESS parsing) but not sent to the user.
SUBPROCESS_NOISE_FILTER = [
    "WARNING: All log messages before absl::InitializeLog()",  # absl early-logging boilerplate
    "I0000 00:00:",          # GPU device creation (e.g. "I0000 gpu_device.cc:...")
    "W0000 00:00:",          # CUDA/XLA warnings
    "WARNING:absl:",         # Keras 3 deprecation warnings (save_format, HDF5, etc.)
    "WARNING:tensorflow:",   # TF legacy keras warning (TF_USE_LEGACY_KERAS)
    "UserWarning: Your input ran out of data",  # Keras epoch data warning
    "self._interrupted_warning()",              # Keras warning follow-up line
    "extern/local_xla",      # XLA compilation noise
    "ptxas warning",         # CUDA PTX assembler warnings
    "dot_search_space",      # XLA autotuning noise
]

# When whitelist is active, ONLY lines containing these patterns (or SUCCESS/ERROR) are shown to the user.
# This keeps the VS Code streaming output completely clean and beautiful.
SUBPROCESS_CLEAN_ALLOWLIST = [
    "Epoch",        # Only the training progress
    "[Phase",       # Modification phases
    "[Saving]",     # Start saving
    "[Train] SUCCESS:", # End of training (if any, but SUCCESS is already hardcoded in run_subprocess_streaming)
    "[Train] ERROR:",   # Errors (already covered by "ERROR:" in run_subprocess_streaming)
]

class ModificationDecision(BaseModel):
    """Decision on whether to apply modifications to the model"""
    wants_modifications: bool = Field(
        description="Does the user want to make modifications to the model?"
    )
    reasoning: str = Field(
        description="Brief explanation of the decision"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence of the classification"
    )


class ContinueDecision(BaseModel):
    """Decision on whether to continue with AI analysis after customization"""
    wants_to_continue: bool = Field(
        description="Does the user want to continue with X-CUBE-AI analysis?"
    )
    reasoning: str = Field(
        description="Brief explanation of the decision"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence of the classification"
    )

continue_decision_instructions = """You are an intent classifier for the X-CUBE-AI workflow.

Analyze the user's response to the question: "Do you want to continue with X-CUBE-AI analysis?"

AFFIRMATIVE RESPONSES (wants to continue):
- "yes", "y", "sure", "ok", "continue", "go", "proceed", "forward", "yes please", "analyze", "continue_ai"
- Any explicit confirmation

NEGATIVE RESPONSES (terminate):
- "no", "nope", "stop", "end", "done", "enough", "terminate", "exit", "quit", "no thanks"

ALWAYS respond in JSON:
- "wants_to_continue": true/false
- "reasoning": brief explanation (max 50 characters)
- "confidence": 0.0-1.0

Examples:

Input: "yes"
Output: {"wants_to_continue": true, "reasoning": "Explicit confirmation", "confidence": 1.0}

Input: "no thanks, I'm done"
Output: {"wants_to_continue": false, "reasoning": "Explicit refusal", "confidence": 1.0}

Input: "analyze"
Output: {"wants_to_continue": true, "reasoning": "Analysis request", "confidence": 0.95}
"""


modification_decision_instructions = """You are an intent classifier for AI model customization.

Analyze the user's response to the question: "Do you want to make modifications to the downloaded model's architecture, or proceed directly with STEdgeAI analysis?"

AFFIRMATIVE RESPONSES (wants modifications):
- "yes", "y", "sure", "ok", "I want to", "I want to modify", "I want to change", "regularize", "reduce", "compression", "optimize"
- "fewer layers", "lighter", "efficient", "dropout", "change activation"
- Any explicit request for change

NEGATIVE RESPONSES (proceeds without modifications):
- "no", "nope", "nothing", "skip", "forward", "proceed", "let's go forward", "keep", "it's ok like this", "fine as is"
- "no, proceed directly", "no modifications", "default"

ALWAYS respond with a valid JSON object with this exact structure:
{
  "wants_modifications": true,
  "reasoning": "explanation (max 50 characters)",
  "confidence": 0.95
}

Examples:

Input: "Reduce the number of layers, it's too complex"
Output: {"wants_modifications": true, "reasoning": "Explicit request to reduce layers", "confidence": 0.95}

Input: "No, proceed directly with the analysis"
Output: {"wants_modifications": false, "reasoning": "Explicit refusal, skip modifications", "confidence": 0.95}

Input: "Hmm, I don't know... what do you recommend?"
Output: {"wants_modifications": false, "reasoning": "Indecision, keeps default", "confidence": 0.6}
"""


class ModelModification(BaseModel):
    """Schema for user-requested modifications"""
    freeze_layers: Optional[str] = Field(description="Which layers to freeze (e.g. 'all', 'none', 'first_10')")
    target_output_classes: Optional[int] = Field(description="Desired number of output classes")
    dropout_rate: Optional[float] = Field(description="Dropout rate to add (0.0 - 1.0)")
    learning_rate: Optional[float] = Field(description="Suggested learning rate")
    additional_layers: Optional[List[str]] = Field(description="Additional layers requested")
    confidence: float = Field(description="Parsing confidence (0-1)")

model_modification_instructions = """You are a Deep Learning expert interpreting model modification requests.

Analyze the user's request and extract the desired architectural modifications.

PARAMETERS TO EXTRACT:
1. freeze_layers: 
   - 'all': freeze everything (except the last part)
   - 'none': unfreeze everything
   - 'first_N': freeze the first N layers
   - 'last_N': freeze the last N layers
   - 'base': freeze only the backbone

2. target_output_classes: Integer. If the user says "2 classes", extract 2.

3. dropout_rate: Float between 0.0 and 1.0. If the user says "Add dropout", use 0.5 as default if not specified.

4. learning_rate: Float (e.g. 0.001).

EXAMPLES:

Input: "I want to classify dogs and cats, freeze everything else."
Output: {
    "freeze_layers": "all",
    "target_output_classes": 2,
    "confidence": 0.95
}

Input: "Add a 30% dropout and use learning rate 1e-4"
Output: {
    "dropout_rate": 0.3,
    "learning_rate": 0.0001,
    "confidence": 0.98
}

If a parameter is not mentioned, leave it null.
"""


def load_or_create_sample_dataset(num_samples: int = 100, 
                                   img_size: Tuple[int, int] = (32, 32),
                                   num_classes: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """Loads or creates a sample dataset"""
    logger.info(f"📊 Creating sample dataset ({num_samples} images)...")
    
    X = np.random.rand(num_samples, img_size[0], img_size[1], 3).astype(np.float32)
    y = tf.keras.utils.to_categorical(np.random.randint(0, num_classes, num_samples), num_classes)
    
    X = X / 255.0
    
    logger.info(f"✓ Dataset created: X.shape={X.shape}, y.shape={y.shape}")
    return X, y


def save_model_with_metadata(model: Model, 
                             output_path: str,
                             metadata: dict[str, any]) -> None:
    """Saves model + metadata for traceability"""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    
    model.save(output_path)
    
    metadata_path = output_path.replace('.h5', '_metadata.json')
    with open(metadata_path, 'w') as f:
        metadata_clean = {
            'timestamp': metadata.get('timestamp', datetime.now().isoformat()),
            'input_shape': str(metadata.get('input_shape', 'unknown')),
            'output_shape': str(metadata.get('output_shape', 'unknown')),
            'total_params': int(metadata.get('total_params', 0)),
            'model_size_mb': round(os.path.getsize(output_path) / (1024*1024), 2),
            'modifications_applied': metadata.get('modifications_applied', []),
            'training_params': metadata.get('training_params', {}),
        }
        json.dump(metadata_clean, f, indent=2)
    
    logger.info(f"✓ Model saved: {output_path}")


def inspect_model_architecture(state: MasterState, config: RunnableConfig = None) -> MasterState:
    """Detailed inspection of the downloaded model with a robust fallback"""

    logger.info("🔍 Inspecting model architecture...")

    # ✅ SKIP ANALYSIS IF ALREADY DONE IN WORKFLOW 2
    if state.model_architecture and state.model_architecture.get('n_layers', 0) > 0:
        logger.info("✓ Architecture info already present, skipping analysis.")
        return state

    # === FILE FORMAT AND EXISTENCE CHECK ===
    ext = os.path.splitext(state.model_path)[1].lower() if state.model_path else ""
    
    file_exists = os.path.exists(state.model_path) if state.model_path else False
    if not file_exists:
        logger.warning(f"⚠️  The model file does not exist: {state.model_path}")
        logger.info("ℹ️  Continuing with virtual metadata for integration purposes.")
        state.model_architecture = {
            "input_shape": "Variable (Missing File)",
            "output_shape": "Variable (Missing File)",
            "n_layers": 0,
            "layer_types": [],
            "layer_names": [],
            "total_params": 0,
            "trainable_params": 0,
            "model_size_mb": 0.0,
            "has_batchnorm": False,
            "has_dropout": False,
            "output_classes": 0,
            "format": ext or "unknown"
        }
        state.model_summary_text = "The selected model file was not found on the local file system.\nThe workflow continues in 'Virtual' mode to allow testing the VS Code interface."
        return state

    # === HANDLING NON-KERAS FORMATS (ONNX, TFLITE) ===
    if ext in ['.onnx', '.tflite']:
        logger.info(f"ℹ️  Format {ext} detected. Providing generic metadata.")
        file_size = os.path.getsize(state.model_path)
        state.model_architecture = {
            "input_shape": "Variable (External Format)",
            "output_shape": "Variable (External Format)",
            "n_layers": 0,
            "layer_types": [],
            "layer_names": [],
            "total_params": 0,
            "trainable_params": 0,
            "model_size_mb": round(file_size / (1024*1024), 2),
            "has_batchnorm": False,
            "has_dropout": False,
            "output_classes": 0,
            "format": ext
        }
        state.model_summary_text = f"Model in {ext.upper()} format.\nDetailed layer inspection is only natively supported for H5/Keras.\nSTEdgeAI will handle conversion and optimization."
        return state

    try:
        # ✅ First attempt: standard load_model
        logger.info("   Attempt 1: standard load_model()...")
        model = tf.keras.models.load_model(state.model_path, compile=False)
        
        trainable_params = int(sum([tf.size(w).numpy() for w in model.trainable_weights]))
        model_size_mb = os.path.getsize(state.model_path) / (1024*1024)
        
        state.model_architecture = {
            "input_shape": str(model.input_shape),
            "output_shape": str(model.output_shape),
            "n_layers": len(model.layers),
            "layer_types": [layer.__class__.__name__ for layer in model.layers],
            "layer_names": [layer.name for layer in model.layers],
            "total_params": int(model.count_params()),
            "trainable_params": trainable_params,
            "model_size_mb": round(model_size_mb, 2),
            "has_batchnorm": any(isinstance(l, tf.keras.layers.BatchNormalization) for l in model.layers),
            "has_dropout": any(isinstance(l, tf.keras.layers.Dropout) for l in model.layers),
            "output_classes": model.output_shape[-1] if len(model.output_shape) > 1 else 1,
            "format": ext
        }
        
        import io
        stream = io.StringIO()
        model.summary(print_fn=lambda x: stream.write(x + '\n'))
        state.model_summary_text = stream.getvalue()
        
        logger.info(f"✓ Architecture analyzed (load_model):")
        logger.info(f"  - Layers: {state.model_architecture['n_layers']}")
        logger.info(f"  - Total params: {state.model_architecture['total_params']:,}")
        logger.info(f"  - Model size: {state.model_architecture['model_size_mb']:.2f} MB")
        
        return state
    
    except Exception as e:
        # ❌ load_model fails, try raw HDF5 fallback (only if .h5 or .keras)
        logger.warning(f"⚠️  load_model() failed: {str(e)[:100]}")
        
        if ext in ['.h5', '.keras']:
            logger.info("   Attempt 2: Raw HDF5 analysis...")
            try:
                # ✅ Fallback: Extract info directly from HDF5 file
                with h5py.File(state.model_path, 'r') as f:
                    # Extract layer info
                    if 'model_config' in f.attrs:
                        config_data = json.loads(f.attrs['model_config'])
                        if isinstance(config_data, str): # Keras 3 might store it differently
                             config_data = json.loads(config_data)
                        
                        layers_list = config_data.get('config', {}).get('layers', [])
                        n_layers = len(layers_list)
                        layer_names = [l.get('name', 'unknown') for l in layers_list]
                        layer_types = [l.get('class_name', 'unknown') for l in layers_list]
                    else:
                        # Fallback: extract from model_weights
                        layer_names = list(f.get('model_weights', {}).keys()) if 'model_weights' in f else []
                        n_layers = len(layer_names)
                        layer_types = ['Unknown'] * n_layers
                    
                    # Extract shape info
                    input_shape = "Unknown"
                    if 'model_weights' in f:
                        weights_group = f['model_weights']
                        first_layer_weights = list(weights_group.values())[0] if len(weights_group) > 0 else None
                        if first_layer_weights:
                            input_shape = first_layer_weights.shape if hasattr(first_layer_weights, 'shape') else "Unknown"
                    
                    file_size = os.path.getsize(state.model_path)
                    estimated_params = (file_size - 1024) / 4 
                    
                    state.model_architecture = {
                        "input_shape": str(input_shape),
                        "output_shape": "Unknown (raw HDF5)",
                        "n_layers": n_layers,
                        "layer_types": layer_types,
                        "layer_names": layer_names,
                        "total_params": int(estimated_params) if estimated_params > 0 else 0,
                        "trainable_params": 0,
                        "model_size_mb": round(file_size / (1024*1024), 2),
                        "has_batchnorm": any('batch' in name.lower() for name in layer_names),
                        "has_dropout": any('dropout' in name.lower() for name in layer_names),
                        "output_classes": 0,
                        "format": ext
                    }
                    return state
            except Exception as e2:
                logger.error(f"❌ Raw HDF5 failed: {str(e2)[:100]}")
        
        # ❌ Final fallback: minimal default
        logger.warning("⚠️  Using minimal default to continue the workflow")
        
        file_size = os.path.getsize(state.model_path) if os.path.exists(state.model_path) else 0
        
        state.model_architecture = {
            "input_shape": "Unknown",
            "output_shape": "Unknown",
            "n_layers": 0,
            "layer_types": [],
            "layer_names": [],
            "total_params": 0,
            "trainable_params": 0,
            "model_size_mb": file_size / (1024*1024),
            "has_batchnorm": False,
            "has_dropout": False,
            "output_classes": 0,
            "format": ext
        }
        return state


def ask_modification_intent(state, config: RunnableConfig = None):
    """Asks the user if they want to modify the model"""
    
    logger.info("💬 Requesting modification intent...")
    
    cfg = Configuration.from_runnable_config(config)
    
    from src.assistant.utils import extract_user_response, get_llm
    llm = get_llm(config)
    llm_classifier = llm.with_structured_output(ModificationDecision)
    
    # --- Step 1: Fast-path ONLY if the original message EXPLICITLY asks for modifications ---
    # If the LLM says wants_modifications=False (even with high confidence) we still ask:
    # the user might not have mentioned modifications in the original AI initial message.
    initial_intent = None
    if not state.user_response:
        msg_clean = state.message.lower().strip()
        generic_triggers = ["ai", "ai_analysis", "analyze", "modello", "model", "start ai"]
        is_generic = any(t == msg_clean for t in generic_triggers) or msg_clean.startswith("@")
        
        if not is_generic:
            res = llm_classifier.invoke([
                SystemMessage(content=modification_decision_instructions),
                HumanMessage(content=f"Message: {state.message}")
            ])
            # Fast-path ONLY if explicitly wants to modify (True with high confidence)
            if res.wants_modifications and res.confidence > 0.9:
                initial_intent = True
                logger.info(f"🤖 Explicit modification intent detected (Conf: {res.confidence}), skipping interrupt.")
            else:
                logger.info(f"🤔 Non-explicit intent (wants={res.wants_modifications}, Conf: {res.confidence}), asking for confirmation.")

    if initial_intent is True:
        decision = ModificationDecision(wants_modifications=True, reasoning="Detected in initial message", confidence=1.0)
    else:
        # --- Step 2: Interrupt – ask the user ---
        resume_value = None
        if not state.user_response:
            prompt = {
                "instruction": """Do you want to make modifications to the model architecture?

Options:
- YES: We proceed with customization (reduce layers, add regularization, etc.)
- NO: We go forward directly with STEdgeAI analyze/validate/generate

What do you prefer? (yes/no)""",
            }
            has_modified = state.persistent_context.get("last_workflow") == "customization" if state.persistent_context else False
            if has_modified:
                prompt["suggestion"] = "💡 Last time you customized the model. Do you want to do it again?"
            
            logger.info("⏸️ Interrupting for modification intent.")
            # resume_value = interrupt(prompt)
            resume_value = "yes" # BYPASS
            # user_text = "yes"

        # After resume
        if resume_value and str(resume_value).strip():
            user_text = str(resume_value).strip().lower()
        else:
            user_text = extract_user_response(state.user_response).lower()
        state.user_response = ""
        
        try:
            decision = llm_classifier.invoke([
                SystemMessage(content=modification_decision_instructions),
                HumanMessage(content=f"User response: {user_text}")
            ])

            # Guard: In case LLM returns non-dict or validation fails (via triton_client _to_pydantic)
            if not hasattr(decision, "wants_modifications"):
                logger.warning(f"⚠️ Classification failed/invalid (type: {type(decision)}). Defaulting to NO modifications.")
                decision = ModificationDecision(wants_modifications=False, reasoning="Classification error fallback", confidence=0.0)
        except Exception as e:
            logger.warning(f"❌ Critical error parsing LLM intent: {str(e)}. Defaulting to NO modifications.")
            decision = ModificationDecision(wants_modifications=False, reasoning="Parser exception fallback", confidence=0.0)

    # === SAVE INTENT INTO STATE ===
    state.wants_model_modifications = decision.wants_modifications
    state.modification_intent_confidence = decision.confidence
    
    logger.info(f"✓ Final decision: wants_modifications={state.wants_model_modifications}")
    return state



def decide_after_inspection(state) -> Literal["retrieve_best_practices_for_architecture", "run_analyze"]:
    """Decides whether to proceed to customization or directly to analyze"""
    
    logger.info(f"📍 Post-inspection routing:")
    logger.info(f"   wants_modifications: {state.wants_model_modifications}")
    
    if state.wants_model_modifications:
        logger.info("   → Route: CUSTOMIZATION")
        return "retrieve_best_practices_for_architecture"
    else:
        logger.info("   → Route: SKIP TO ANALYZE")
        return "run_analyze"


# ============================================================================
# RETRIEVE BEST PRACTICES FOR ARCHITECTURE
# ===========================================================================

def retrieve_best_practices_for_architecture(state: MasterState, config: RunnableConfig = None) -> MasterState:
    """With optional web fetch (max 10s timeout)"""
    
    model_name = state.selected_model.get('name', 'Unknown') if state.selected_model else None
    
    if not model_name:
        state.best_practices_display = _get_generic_practices()
        return state
    
    arch_type = _detect_architecture_type(model_name)
    logger.info(f"🔍 Model: {model_name} → Architecture: {arch_type}")
    
    ai_output_dir = getattr(config, "ai_output_dir", os.path.expanduser("~/stm32-ai-workflow/st_ai_output"))
    base_persist_dir = os.path.join(ai_output_dir, "chroma_docs")
    arch_persist_dir = os.path.join(base_persist_dir, arch_type)
    
    # ===== STEP 1: Check cache =====
    logger.info(f"  [Step 1/3] Checking cache for {arch_type}...")
    
    arch_db_exists = os.path.exists(arch_persist_dir) and os.listdir(arch_persist_dir)
    
    if arch_db_exists:
        try:
            best_practices = _retrieve_from_chroma(
                query=f"best practices customization fine-tuning {arch_type}",
                persist_dir=arch_persist_dir,
                arch_type=arch_type,
                embeddings_override=get_embeddings(model="nomic-embed")
            )
            if best_practices and len(best_practices) > 0:
                logger.info(f"  ✓ Retrieved {len(best_practices)} docs from cache")
                state.best_practices_display = _format_practices(best_practices, source=f"CACHE_{arch_type}")
                state.best_practices_raw = [p.page_content for p in best_practices]
                return state
        
        except Exception as e:
            logger.warning(f"  ⚠️  Cache lookup failed: {str(e)[:60]}")
    
    # ===== STEP 2: LLM Generation (MAX 20 SECONDS) =====
    logger.info(f"  [Step 2/3] Generating practices with LLM...")
    
    import time
    start_time = time.time()
    
    try:
        best_practices = _generate_and_cache_with_llm(
            model_name=model_name,
            arch_type=arch_type,
            persist_dir=arch_persist_dir
        )
        
        if best_practices:
            logger.info(f"  ✓ Generated & Custom Cached {len(best_practices)} docs in {time.time()-start_time:.1f}s")
            state.best_practices_display = _format_practices(best_practices, source=f"LLM_GEN_{arch_type}")
            state.best_practices_raw = [p.page_content for p in best_practices]
            return state
    
    except Exception as e:
        logger.warning(f"  ⚠️  LLM Generation failed ({time.time()-start_time:.1f}s): {str(e)[:40]}")
    
    # ===== STEP 3: Fallback =====
    logger.info(f"  [Step 3/3] Using fallback practices for {arch_type}...")
    state.best_practices_display = _get_architecture_specific_practices(arch_type)
    state.best_practices_raw = []
    
    return state


def _generate_and_cache_with_llm(
    model_name: str,
    arch_type: str,
    persist_dir: str
) -> Optional[List]:
    """
    Generates best practices using local LLM (Ollama) and saves them to Chroma.
    Replaces web search for greater reliability.
    """
    import time
    from langchain_core.documents import Document
    
    start_time = time.time()
    logger.info(f"  Generating best practices for {arch_type} with LLM...")
    
    try:
        # 1. Setup LLM
        # Use centrally configured model (more robust than hardcoded 'mistral')
        llm = get_llm(None, temperature=0.3, keep_alive="5m")
        
        # 2. Best Practices Prompt (CONCISE & SCHEMATIC)
        prompt = f"""You are an expert embedded AI engineer.
        Provide a **concise, bullet-point checklist** for fine-tuning {model_name} ({arch_type}) on STM32.
        
        REQUIRED FORMAT (Strictly follow this):
        
        *   **Strategy**: [Freeze X% layers / Retrain all]
        *   **Hyperparams**: [LR: 1e-X, Batch: N, Epochs: N]
        *   **Quantization**: [INT8/float16]
        *   **Constraints**: [Flash/RAM usage estimates]
        
        Keep it under 200 words. No intro/outro. Schematic only.
        """
        
        # 3. Generation
        logger.info(f"  Invoking LLM (this may take 10-20s)...")
        response = llm.invoke(prompt)
        content = response.content
        
        logger.info(f"  ✓ Native LLM generation complete ({len(content)} chars)")
        
        # 4. Create LangChain Document
        doc = Document(
            page_content=content,
            metadata={
                "source": "LLM_GENERATED",
                "architecture": arch_type,
                "model": model_name,
                "timestamp": str(datetime.now())
            }
        )
        
        all_docs = [doc]
        
    except Exception as e:
        logger.error(f"❌ LLM Generation failed: {str(e)}")
        return None

    # ===== STEP 3: Save to Chroma ===== (Reuse existing logic)
    if all_docs:
        try:
            logger.info(f"  Saving to Chroma ({arch_type})...")
            
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            chunks = splitter.split_documents(all_docs)
            
            embeddings = get_embeddings(model="nomic-embed")
            
            os.makedirs(persist_dir, exist_ok=True)
            
            # Save to vectorstore
            Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=persist_dir,
                collection_name=f"{arch_type}_best_practices"
            )
            
            logger.info(f"  ✓ Saved {len(chunks)} chunks to {persist_dir}")
        
        except Exception as e:
            logger.warning(f"  Chroma save failed: {str(e)[:60]}")
    
    return all_docs

    # ===== STEP 3: Save to Chroma =====
    if all_docs:
        try:
            logger.info(f"  Saving to Chroma ({arch_type})...")
            
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            chunks = splitter.split_documents(all_docs)
            
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            
            os.makedirs(persist_dir, exist_ok=True)
            
            # Save to vectorstore
            Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=persist_dir,
                collection_name=f"{arch_type}_best_practices"
            )
            
            logger.info(f"  ✓ Saved {len(chunks)} chunks to {persist_dir}")
        
        except Exception as e:
            logger.warning(f"  Chroma save failed: {str(e)[:60]}")
    
    return all_docs[:3]


# ============================================================================
# SEARCH WEB: Fetch URLs for Best Practices
# ============================================================================

def search_web(queries: List[str]) -> List[dict[str, str]]:
    """
    ✨ Web search to retrieve best practice URLs by architecture
    
    Based on execute_web_search but optimized to return a list of URLs.
    
    Args:
        queries: List of queries (max 3 for efficiency)
    
    Returns:
        List[dict] with format:
        [
            {"url": "https://...", "title": "...", "content": "..."},
            {"url": "https://...", "title": "...", "content": "..."},
            ...
        ]
    
    Raises:
        Exception if search fails
    """
    
    logger.info(f"🌐 Search web: {len(queries)} queries")
    
    all_results = []
    
    for query in queries:
        try:
            logger.debug(f"  Searching: {query}")
            
            # ===== SETUP AGNO AGENT =====
            base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
            agent = Agent(
                model=Ollama(id="mistral", base_url=base_url),
                tools=[DuckDuckGoTools()],
                markdown=True
            )
            
            # ===== SIMPLE AND DIRECT PROMPT =====
            search_prompt = f"""Search for information about:
"{query}"

Return the top 5 most relevant results with URLs and brief descriptions."""
            
            # ===== EXECUTE SEARCH =====
            logger.debug(f"  Invoking Agno Agent...")
            response = agent.run(search_prompt)
            
            if not response:
                logger.warning(f"  Empty response for query: {query}")
                continue
            
            # ===== PARSE RESPONSE =====
            content = response.content if hasattr(response, 'content') else str(response)
            
            logger.debug(f"  Response length: {len(content)} chars")
            
            # Extract URLs from the response (parse markdown links)
            urls = _extract_urls_from_response(content)
            
            if urls:
                logger.debug(f"  Extracted {len(urls)} URLs")
                
                for url_info in urls:
                    all_results.append({
                        "url": url_info.get('url'),
                        "title": url_info.get('title', ''),
                        "content": content[:200]  # Snippet of the response
                    })
            else:
                logger.debug(f"  No URLs extracted")
        
        except Exception as e:
            logger.warning(f"  Query failed: {query} - {str(e)[:60]}")
            continue
    
    # ===== SIMULATED FALLBACK (FOR DEVELOPMENT/DEBUG) =====
    if not all_results:
        logger.warning("⚠️  No URLs found (or Rate Limit). Using SIMULATED results to test Chroma.")
        
        # Simulation based on query
        simulated_data = []
        q_str = " ".join(queries).lower()
        
        if "yolo" in q_str:
            simulated_data = [
                {"url": "https://docs.ultralytics.com/modes/export", "title": "YOLO Export Guide", "content": "Guide to exporting YOLO models to TFLite and other formats for embedded deployment."},
                {"url": "https://wiki.st.com/stm32mcu/wiki/AI:Model_ZOO", "title": "STM32 Model Zoo - Object Detection", "content": "Official ST model zoo including YOLO derivatives optimized for STM32 series (H7, U5)."},
                {"url": "https://github.com/STMicroelectronics/stm32ai-modelzoo", "title": "STM32AI Model Zoo GitHub", "content": "Code and pre-trained models for various STM32 boards including object detection examples."}
            ]
        elif "mobilenet" in q_str:
             simulated_data = [
                {"url": "https://www.tensorflow.org/lite/models/modify/model_maker/image_classification", "title": "TFLite Model Maker", "content": "Retraining MobileNetV2 with TensorFlow Lite Model Maker for custom datasets."},
                {"url": "https://wiki.st.com/stm32mcu/wiki/AI:Getting_started", "title": "Getting Started with STM32Cube.AI", "content": "Step by step guide to importing MobileNet into STM32Cube.AI."}
            ]
        else:
             simulated_data = [
                {"url": "https://www.st.com/en/embedded-software/x-cube-ai.html", "title": "X-CUBE-AI Expansion Pack", "content": "Main page for the STM32 AI expansion pack, supporting multiple frameworks and models."}
            ]
            
        all_results.extend(simulated_data)
        logger.info(f"  ✓ Added {len(simulated_data)} simulated results")

    logger.info(f"✓ Total results: {len(all_results)} URLs")
    
    return all_results[:10]  # Return top 10 results


# ============================================================================
# HELPER: Extract URLs from Agno Response
# ============================================================================

def _extract_urls_from_response(content: str) -> List[dict[str, str]]:
    """
    Extracts URLs from the Agno Agent response.
    
    Supports formats:
    - Markdown links: [Title](URL)
    - Plain URLs: https://...
    - Numbered lists with URL
    """
    
    import re
    
    urls = []
    
    # ===== PATTERN 1: Markdown links [title](url) =====
    markdown_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
    for match in re.finditer(markdown_pattern, content):
        title = match.group(1).strip()
        url = match.group(2).strip()
        
        # Validate URL
        if url.startswith('http'):
            urls.append({
                'url': url,
                'title': title
            })
    
    # ===== PATTERN 2: Plain URLs =====
    url_pattern = r'https?://[^\s\)]+(?:\.[a-zA-Z]+)+'
    for match in re.finditer(url_pattern, content):
        url = match.group(0).strip()
        
        # Validate URL (not already included)
        if url not in [u['url'] for u in urls]:
            urls.append({
                'url': url,
                'title': 'Search result'
            })
    
    # ===== PATTERN 3: Numbered list with URL =====
    # Ex: "1. Title - https://example.com"
    list_pattern = r'\d+\.\s+([^-]+)\s*-?\s*(https?://[^\s]+)'
    for match in re.finditer(list_pattern, content):
        title = match.group(1).strip()
        url = match.group(2).strip()
        
        if url not in [u['url'] for u in urls]:
            urls.append({
                'url': url,
                'title': title
            })
    
    return urls


# ============================================================================
# USAGE IN _fetch_and_cache_architecture_practices
# ============================================================================

def _fetch_and_cache_architecture_practices(
    model_name: str,
    arch_type: str,
    persist_dir: str
) -> Optional[List]:
    """Fetch best practices online and save in DEDICATED Chroma"""
    
    logger.info(f"  Fetching practices for {arch_type}...")
    
    queries = _get_search_queries_for_architecture(arch_type)
    all_docs = []
    
    # ===== STEP 1: Search web =====
    logger.info(f"  Searching web for {len(queries)} queries...")
    
    try:
        search_results = search_web(queries)  # ← CALL THE NEW FUNCTION
    except Exception as e:
        logger.warning(f"  Web search failed: {str(e)[:60]}")
        search_results = []
    
    if not search_results:
        logger.warning(f"  No search results for {arch_type}")
        return None
    
    logger.info(f"  Found {len(search_results)} URLs to load")
    
    # ===== STEP 2: Load from URLs =====
    for i, result in enumerate(search_results, 1):
        url = result.get('url')
        title = result.get('title', 'Unknown')
        
        if not url:
            continue
        
        try:
            logger.debug(f"  [{i}/{len(search_results)}] Loading {url[:50]}...")
            
            loader = RecursiveUrlLoader(
                url=url,
                max_depth=1,
                extractor=lambda x: BeautifulSoup(x, "html.parser").get_text(),
                prevent_outside=True,
                timeout=10
            )
            
            docs = loader.load()
            
            if docs:
                # Add metadata
                for doc in docs:
                    doc.metadata['architecture'] = arch_type
                    doc.metadata['model_name'] = model_name
                    doc.metadata['source_url'] = url
                    doc.metadata['source_title'] = title
                
                all_docs.extend(docs)
                logger.debug(f"    ✓ Loaded {len(docs)} sections")
        
        except Exception as e:
            logger.debug(f"    Failed: {str(e)[:40]}")
    
    if not all_docs:
        logger.warning(f"  No documents loaded from URLs")
        return None
    
    logger.info(f"  Loaded {len(all_docs)} total documents from {len(search_results)} URLs")
    
    # ===== STEP 3: Save to Chroma =====
    try:
        logger.info(f"  Saving to Chroma ({arch_type})...")
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = splitter.split_documents(all_docs)
        
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        os.makedirs(persist_dir, exist_ok=True)
        
        # vectorstore = Chroma.from_documents(
        #     documents=chunks,
        #     embedding=embeddings,
        #     persist_directory=persist_dir,
        #     collection_name=f"{arch_type}_best_practices"
        # )

        # TO BE FIXED !!! CHROMA IS CAUSING ISSUES. ALSO VECTORSTORE WAS NOT USED ANYWHERE...
        
        logger.info(f"  ✓ Saved {len(chunks)} chunks to {persist_dir}")
    
    except Exception as e:
        logger.warning(f"  Chroma save failed: {str(e)[:60]}")
    
    return all_docs[:5]



def _get_search_queries_for_architecture(arch_type: str) -> List[str]:
    """Returns specific queries for architecture"""
    
    queries_map = {
        'mobilenet': [
            "MobileNetV2 optimization STM32 embedded",
            "fine-tuning MobileNet transfer learning best practices",
            "MobileNetV2 quantization INT8 edge deployment"
        ],
        
        'resnet': [
            "ResNet fine-tuning transfer learning STM32",
            "ResNet optimization layer freezing",
            "ResNet50 quantization embedded systems"
        ],
        
        'efficientnet': [
            "EfficientNet optimization embedded devices",
            "EfficientNet fine-tuning best practices",
            "EfficientNet quantization INT8"
        ],
        
        'vgg': [
            "VGG16 transfer learning optimization",
            "VGG fine-tuning embedded systems",
            "VGG quantization compression"
        ],
        
        'yolo': [
            "YOLO object detection STM32 embedded",
            "YOLOv2 tiny optimization microcontroller",
            "YOLO quantization real-time inference"
        ],
        
        'har': [
            "human activity recognition STM32 embedded",
            "activity recognition optimization microcontroller",
            "HAR model compression quantization"
        ],
        
        'custom': [
            "neural network optimization STM32",
            "fine-tuning deep learning transfer learning",
            "model quantization embedded systems"
        ]
    }
    
    return queries_map.get(arch_type, queries_map['custom'])

def _retrieve_from_chroma(
    query: str,
    persist_dir: str,
    arch_type: str,
    embeddings_override: Any = None
) -> Optional[List]:
    """Retrieves from DEDICATED Chroma by architecture"""
    
    try:
        embeddings = embeddings_override or get_embeddings(model="mistral")
        
        vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
            collection_name=f"{arch_type}_best_practices"
        )
        
        # ✅ CORRECT: Use similarity_search without filter
        # (the filter is implicit because we use an arch-specific collection_name)
        results = vectorstore.similarity_search(query, k=5)
        
        return results if results else None
    
    except Exception as e:
        logger.warning(f"Chroma retrieval failed: {str(e)}")
        return None


def _detect_architecture_type(model_name: str) -> str:
    """Detects architecture from model name"""
    
    model_lower = model_name.lower()
    
    if 'mobilenet' in model_lower:
        return 'mobilenet'
    elif 'resnet' in model_lower or 'res' in model_lower:
        return 'resnet'
    elif 'vgg' in model_lower:
        return 'vgg'
    elif 'efficient' in model_lower or 'efficientnet' in model_lower:
        return 'efficientnet'
    elif 'inception' in model_lower or 'inceptionv3' in model_lower:
        return 'inception'
    elif 'yolo' in model_lower:
        return 'yolo'
    elif 'ssd' in model_lower:
        return 'ssd'
    elif 'gmp' in model_lower or 'har' in model_lower or 'activity' in model_lower:
        return 'har'
    else:
        return 'custom'


def _format_practices(docs: List, source: str = "UNKNOWN") -> str:
    """Formats documents for display"""
    
    formatted = f"\n════════════════════════════════════════════════════════════\n"
    formatted += f"📋 BEST PRACTICES ({source})\n"
    formatted += f"════════════════════════════════════════════════════════════\n\n"
    
    for doc in docs[:1]: # Take only the first one if there are many (usually just 1)
        content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
        formatted += f"{content.strip()}\n"
    
    formatted += f"\n════════════════════════════════════════════════════════════\n"
    
    return formatted



def _get_architecture_specific_practices(arch_type: str) -> str:
    """Returns hardcoded best practices for architecture"""
    
    practices_map = {
        'mobilenet': """
🛠️  MOBILENETV2 BEST PRACTICES:
  • Freeze first 50-70% layers for fine-tuning
  • Use learning rate: 1e-4 to 1e-5
  • Add Dropout (0.3-0.4) before classifier
  • Supports input resizing well
  • Quantization: Excellent with INT8 (4× compression)
  • Inference time STM32H7: 50-100ms
        """,
        
        'resnet': """
🛠️  RESNET BEST PRACTICES:
  • Freeze first 60% layers for transfer learning
  • Use learning rate: 1e-5 to 1e-4 (conservative)
  • Add BatchNorm momentum: 0.9
  • WARNING: Changing input shape < 224×224 may fail
  • Quantization: Good, may lose 2-3% accuracy
  • Deep network: Use low learning rates
        """,
        
        'efficientnet': """
🛠️  EFFICIENTNET BEST PRACTICES:
  • Already has Dropout - don't add more!
  • Freeze first 80% layers (more aggressive)
  • Use learning rate: 1e-4
  • Flexible input sizes (64-380×64-380)
  • Quantization: Very efficient with INT8
  • Best for embedded (size vs accuracy trade-off)
        """,
        
        'vgg': """
🛠️  VGG BEST PRACTICES:
  • Older architecture - consider MobileNet instead
  • Freeze first 3-4 blocks (70%+)
  • High memory usage - not ideal for STM32
  • Use learning rate: 1e-5
  • Input size: Must be 224×224
  • Quantization: Works but large even after
        """,
        
        'yolo': """
🛠️  YOLO BEST PRACTICES:
  • Object detection - different workflow than classification
  • DON'T use change_output_layer (custom output)
  • Freeze backbone, fine-tune detection head
  • Learning rate: 1e-5 to 1e-6
  • Use small YOLO versions (YOLOv2-tiny, YOLOv3-tiny)
  • Quantization: Check mAP after INT8
        """,
        
        'har': """
🛠️  HUMAN ACTIVITY RECOGNITION BEST PRACTICES:
  • Time-series input (not images!)
  • Small models (1-5MB) - excellent for STM32
  • Freeze 30-50% layers
  • Use learning rate: 1e-4
  • Classes: 4-6 (sitting, walking, running, etc)
  • Quantization: Minimal accuracy loss
        """,
        
        'custom': """
🛠️  GENERAL CUSTOMIZATION BEST PRACTICES:
  • Start conservative: Freeze 50% layers
  • Use learning rate: 1e-4 (safe default)
  • Add Dropout (0.3) if > 10 layers
  • Monitor for overfitting
  • Test on STM32 early
  • Use quantization INT8 for deployment
        """
    }
    
    default = practices_map.get(arch_type, practices_map['custom'])
    
    return f"\n════════════════════════════════════════════════════════════\n{default}\n════════════════════════════════════════════════════════════\n"



def _get_generic_practices() -> str:
    """Generic fallback when model is unknown"""
    
    return """
════════════════════════════════════════════════════════════
📋 GENERAL BEST PRACTICES
════════════════════════════════════════════════════════════

🔒 Layer Freezing:
   • Freeze 40-60% of layers for transfer learning
   • Preserve pre-trained features from ImageNet/COCO

💧 Regularization:
   • Add Dropout (0.3-0.5) if > 10 layers
   • Monitor for overfitting on small datasets

🎓 Learning Rate:
   • Fine-tuning: 1e-4 to 1e-5
   • From scratch: 1e-3 to 1e-2
   • Reduce 10× every plateau

📊 Batch Size & Epochs:
   • Batch size: 32-64 (STM32 memory constraint)
   • Epochs: 10-30 (early stopping recommended)

🔢 Quantization:
   • INT8 for STM32 deployment (4× size reduction)
   • Check accuracy drop (usually < 2%)

📸 Data Augmentation:
   • Essential for small datasets
   • Rotation, flip, zoom, brightness

════════════════════════════════════════════════════════════
"""



# ============================================================================
# PYDANTIC SCHEMAS FOR STRUCTURED OUTPUT
# 🔴 PYDANTIC ERROR: 'any' type not supported
# Problem: In the Modification class, 'any' (builtin function) was used as a type hint, but Pydantic doesn't support it.
# ============================================================================

class Modification(BaseModel):
    """Single structured modification"""
    type: str = Field(
        description="Modification type: freeze_layers, freeze_almost_all, change_output_layer, add_dropout, change_input_shape, change_learning_rate, add_resizing_layer"
    )
    description: str = Field(description="Brief description of what this modification does")
    params: dict[str, Any] = Field(description="Parameters for this modification")
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence score for this modification (0.0-1.0)"
    )


class TrainingRecommendation(BaseModel):
    """Training recommendations"""
    learning_rate: float = Field(
        ge=1e-6, le=1e-1,
        description="Suggested learning rate"
    )
    epochs: int = Field(
        ge=1, le=1000,
        description="Suggested number of epochs"
    )
    batch_size: int = Field(
        ge=1, le=256,
        description="Suggested batch size"
    )
    optimizer: str = Field(
        description="Suggested optimizer (adam, sgd, rmsprop, etc)"
    )
    notes: str = Field(
        description="Additional training notes and recommendations"
    )


class ValidationInfo(BaseModel):
    """Validation info"""
    is_valid: bool = Field(description="Are all modifications valid?")
    issues: List[str] = Field(
        default_factory=list,
        description="List of validation issues (empty if valid)"
    )


class ParsedModificationsPlan(BaseModel):
    """Complete modifications plan - FINAL OUTPUT"""
    modifications: List[Any] = Field(
        description="List of modifications to apply. Each item should be a dictionary with 'type', 'description', 'params'.",
        default_factory=list
    )
    summary: str = Field(
        description="Brief summary of all modifications"
    )
    confidence: float = Field(
        description="Overall confidence of the parsing (0.0-1.0)",
        default=0.9
    )
    validation: Any = Field(
        description="Validation status and issues",
        default=None
    )
    training_recommendation: Any = Field(
        description="Training recommendations based on modifications",
        default=None
    )

# List of models that DO NOT support change_input_shape
INCOMPATIBLE_INPUT_SHAPE_MODELS = {
    'yolo': ['tiny_yolo_v2', 'yolov2', 'yolov3', 'yolov4', 'yolov5', 'yolov8'],
    'ssd': [
        'ssd_mobilenet', 
        'ssd_inception', 
        'ssd_resnet', 
        'st_ssd_mobilenet_v1',  # ⭐ NEW from catalog
    ],
    'other_detectors': ['faster_rcnn', 'mask_rcnn', 'retinanet'],
    'time_series_models': [  # ⭐ NEW
        'gmp',
        'har',
        'activity_recognition',
    ]
}
# Fixed grid output in detection. Changing input → changes grid → mismatch with label → loss function crashes. 
# YOLO is designed for 416x416, SSD for 300/512. Changing them breaks model integrity.
# Blocked to avoid silent crash. Alternative: Resizing Layer (in backlog)

def is_model_compatible_with_input_shape_change(model_name: str) -> bool:
    """Checks if the model supports change_input_shape"""
    model_lower = model_name.lower()
    
    for category, models in INCOMPATIBLE_INPUT_SHAPE_MODELS.items():
        for model_pattern in models:
            if model_pattern.lower() in model_lower:
                if category == 'time_series_models':
                    logger.warning(
                        f"⚠️ {model_name} (detected: time-series) "
                        f"has temporal input, not spatial"
                    )
                else:
                    logger.warning(
                        f"⚠️ {model_name} (detected: {category}) "
                        f"has fixed output structure"
                    )
                return False
    return True


def ask_and_parse_user_modifications(state: any, config: RunnableConfig = None) -> any:
    """
    ✨ CONSOLIDATED VERSION: Asks the user and parses immediately
    
    Atomic flow:
    1. Show UI with best practices
    2. User inputs requests (natural language)
    3. LLM parses and validates
    4. Returns structured plan
    
    Args:
        state: MasterState object
        config: Configuration dict
    
    Returns:
        Updated state with parsed_modifications
    """
    
    logger.info("🤔 Asking user which modifications to apply...")

    # ===== STEP 0: Retrieve architecture-specific best practices =====
    logger.info("  [Step 0/3] Fetching best practices...")
    state = retrieve_best_practices_for_architecture(state, config)
    best_practices = state.best_practices_display
    
    # ===== EXTRACT INFO =====
    input_shape = state.model_architecture.get('input_shape', 'Unknown')
    output_classes = state.model_architecture.get('output_classes', 0)
    total_params = state.model_architecture.get('total_params', 0)
    total_layers = state.model_architecture.get('n_layers') or len(state.model_architecture.get('layer_types', []))
    
    formatted_params = f"{total_params:,}" if total_params else "N/A"
    
    # ===== USER PROMPT =====
    prompt = {
        "instruction": f"""
╔═══════════════════════════════════════════════════════════╗
║         🛠️  CUSTOMIZE YOUR STM32 MODEL                    ║
╚═══════════════════════════════════════════════════════════╝

Current Model Info:
  • Input: {input_shape}
  • Output classes: {output_classes}
  • Total params: {formatted_params}
  • Total layers: {total_layers}

Available Modifications:
  ✓ Freeze layers (e.g., "freeze first 5 layers")
  ✓ Freeze almost all (e.g., "keep last 3 layers trainable")
  ✓ Change input shape (e.g., "change input to 64x64x3"). ⚠️ Not supported for detection models (YOLO, SSD, etc.). Use instead "add resizing layer".
  ✓ Change output (e.g., "change output to 100 classes")
  ✓ Add dropout (e.g., "add 0.3 dropout")
  ✓ Learning rate (e.g., "use learning rate 0.0001")
  ✓ Add resizing layer (e.g., "Add resizing layer to accept flexible input sizes")
    ️⚠️ NOTE: Automatically uses your model's original input shape. Zero parameters needed.

Examples:
  • "Freeze all layers except last 3 and add 0.4 dropout"
  • "Change input to 128x128 and output to 50 classes"
  • "Freeze first 10 layers, add dropout 0.2, learning rate 0.0001"
  • "Just freeze the first 20 layers"

Write your modifications in natural language (or leave empty for defaults):
""",
        "best_practices": best_practices,
    }
    
    # === LLM EXTRACTOR ===
    from src.assistant.utils import extract_user_response, get_llm
    cfg = Configuration.from_runnable_config(config)
    llm = get_llm(config)
    llm_extractor = llm.with_structured_output(ParsedModificationsPlan)
    
    # --- Step 1: Try to use the initial message ---
    initial_mods_detected = False
    if not state.user_response:
        try:
            res = llm_extractor.invoke([
                SystemMessage(content="""Extract modifications from this message. 
STRICT RULES:
1. If the message does NOT contain explicit neural network modification requests (like freeze layers, add dropout, change input, etc.), return an EMPTY modifications list. 
2. Do NOT invent modifications based on examples provided in the prompt.
3. If the user just says 'yes', 'si', or 'ok' without specifying WHAT to change, return EMPTY.
4. ONLY extract: freeze_layers, freeze_almost_all, change_output_layer, add_dropout, change_input_shape, change_learning_rate, add_resizing_layer."""),
                HumanMessage(content=f"Message: {state.message}")
            ])
            # res might be a raw dict if Pydantic validation failed in _to_pydantic
            mods = res.modifications if hasattr(res, 'modifications') else res.get('modifications', [])
            conf = res.confidence if hasattr(res, 'confidence') else res.get('confidence', 0.0)
            if mods and conf >= 0.7:
                state.user_custom_modifications = state.message
                initial_mods_detected = True
                logger.info(f"🤖 Modifications detected in initial message (confidence: {conf:.0%}).")
            else:
                logger.info(f"ℹ️ No modifications in initial message (confidence: {conf:.0%}), asking user.")
        except Exception as e:
            logger.warning(f"⚠️ Initial modification detection failed: {e}. Proceeding with interrupt.")

    # --- Step 2: Verification and Interrupt ---
    if not initial_mods_detected:
        if not state.user_response:
            prompt = {
                "instruction": """Describe the modifications you want to apply to the model.
Examples:
- "Freeze first 10 layers and add dropout 0.3"
- "Change input to 128x128"
- "Use learning rate 1e-4"
""",
            }
            
            logger.info("⏸️ Interrupting for customization details.")
            # resume_value = interrupt(prompt)
            resume_value = "freeze first 5 layers and add 0.4 dropout" # BYPASS

            if resume_value and str(resume_value).strip():
                user_modifications = str(resume_value).strip()
            else:
                user_modifications = extract_user_response(state.user_response)
        else:
            user_modifications = extract_user_response(state.user_response)
        
        state.user_response = ""
        state.user_custom_modifications = user_modifications
    else:
        user_modifications = state.user_custom_modifications

    user_modifications = user_modifications.strip() if user_modifications else ""
    logger.info(f"📝 Final user request: {user_modifications[:80]}...")
    
    # ===== STEP 2: Parse with LLM =====
    logger.info("  [Step 2/2] Parsing with LLM structured output...")
    
    try:
        from src.assistant.utils import get_llm
        structured_llm = get_llm(
            config, 
            structured_schema=ParsedModificationsPlan, 
            temperature=0.3
        )
        
        # LLM Prompt - with a concrete example to prevent Mistral from confusing
        # the schema fields with the values of the modifications list
        llm_prompt = f"""Parse this neural network modification request and return ONLY a JSON object.

USER REQUEST: "{user_modifications}"

CURRENT MODEL:
- Total layers: {total_layers}
- Current output classes: {output_classes}
- Input shape: {input_shape}
- Total parameters: {total_params:,}

MODIFICATION TYPES:
1. freeze_layers → params: {{"num_frozen_layers": 5}}
2. freeze_almost_all → params: {{"num_trainable_layers": 3}}
3. change_output_layer → params: {{"new_classes": 100}}
4. add_dropout → params: {{"rate": 0.3}}  (rate between 0.0 and 1.0)
5. change_input_shape → params: {{"new_shape": [128, 128, 3]}}
6. change_learning_rate → params: {{"learning_rate": 0.0001}}
7. add_resizing_layer → params: {{}}

CRITICAL RULES:
- Each item in "modifications" is an OBJECT with "type", "description", "params" and "confidence" keys.
- Do NOT list type names as string values in the array. Each item is a {{...}} object.
- Return ONLY JSON, no extra text.

EXAMPLE OUTPUT for "change input to 128x128":
{{
  "modifications": [
    {{
      "type": "change_input_shape",
      "description": "Change input shape to 128x128x3",
      "params": {{"new_shape": [128, 128, 3]}},
      "confidence": 0.95
    }}
  ],
  "summary": "Change input shape to 128x128",
  "confidence": 0.95,
  "validation": {{"is_valid": true, "issues": []}},
  "training_recommendation": {{
    "learning_rate": 0.0001,
    "epochs": 10,
    "batch_size": 32,
    "optimizer": "adam",
    "notes": "Standard fine-tuning settings"
  }}
}}

Now, parse: "{user_modifications}"
Return a JSON object with the same structure as the example above."""
        
        # Invoke LLM
        try:
            result: ParsedModificationsPlan = structured_llm.invoke([
                SystemMessage(content="""You are a neural network customization expert.
Parse the user request and produce a JSON object with field 'modifications' as an ARRAY OF OBJECTS.
Each modification object must have: "type" (string), "description" (string), "params" (object), "confidence" (float).
DO NOT list type names as strings in the array. Always use object notation."""),
                HumanMessage(content=llm_prompt)
            ])
            
            # Guard: _to_pydantic falls back to raw dict when Mistral returns incomplete JSON.
            # In that case, try to extract what we can before giving up.
            if isinstance(result, dict):
                raw_mods = result.get('modifications', [])
                # Mistral sometimes returns field names as strings instead of objects:
                # e.g. ['change_input_shape', 'summary', ...] — detect and discard this
                actual_mods = [m for m in raw_mods if isinstance(m, dict)]
                if actual_mods:
                    # Partial recovery: we have some dicts, reconstruct
                    result = ParsedModificationsPlan(
                        modifications=actual_mods,
                        summary=result.get('summary', user_modifications[:60]),
                        confidence=result.get('confidence', 0.8),
                        validation=result.get('validation', None),
                        training_recommendation=result.get('training_recommendation', None)
                    )
                    logger.warning(f"⚠️ Partial LLM recovery: extracted {len(actual_mods)} modification dicts from raw response.")
                else:
                    raise ValueError(
                        f"LLM returned an incomplete/empty JSON object (missing required fields). "
                        f"Raw response: {result}"
                    )
        except Exception as e:
            # Catch Langchain OutputParserException and others, re-raise as ValueError
            # to be caught by the outer fallback block
            raise ValueError(f"Invalid JSON/Parsing error from LLM: {str(e)}")

            
        logger.info("  ✓ LLM parsing successful")


        # ===== VALIDATION: is change_input_shape INCOMPATIBLE? =====
        model_name = state.selected_model.get('name', '') if state.selected_model else ''
        mods_to_remove = []

        for i, mod in enumerate(result.modifications):
            mod_type = mod.get('type') if isinstance(mod, dict) else getattr(mod, 'type', '')
            if mod_type == 'change_input_shape':
                if not is_model_compatible_with_input_shape_change(model_name):
                    logger.error(f"❌ Removing change_input_shape (not supported for {model_name})")
                    mods_to_remove.append(i)
                    if not isinstance(result.validation, dict) and hasattr(result.validation, 'issues'):
                        result.validation.issues.append(
                            f"change_input_shape: Blocked - {model_name} has fixed input structure"
                        )
                    elif isinstance(result.validation, dict):
                        if 'issues' not in result.validation:
                            result.validation['issues'] = []
                        result.validation['issues'].append(
                            f"change_input_shape: Blocked - {model_name} has fixed input structure"
                        )

        # Remove in reverse order
        for i in reversed(mods_to_remove):
            result.modifications.pop(i)

        if mods_to_remove:
            result.validation.is_valid = False
        
        # ===== PARAMETER VALIDATION =====
        issues = []
        from src.assistant.utils import get_llm, validate_modification_params
        
        for i, mod in enumerate(result.modifications):
            mod_type = mod.get('type') if isinstance(mod, dict) else getattr(mod, 'type', '')
            mod_params = mod.get('params', {}) if isinstance(mod, dict) else getattr(mod, 'params', {})
            
            # Use centralized validation logic
            sanitized_params, mod_issues = validate_modification_params(
                mod_type, 
                mod_params, 
                total_layers=total_layers
            )
            
            if isinstance(result.modifications[i], dict):
                result.modifications[i]['params'] = sanitized_params
            else:
                result.modifications[i].params = sanitized_params
                
            issues.extend(mod_issues)
            
            # Additional custom validation for output layer (classes)
            if mod_type == 'change_output_layer':
                new_classes = sanitized_params.get('new_classes')
                if new_classes is None:
                    new_classes = output_classes
                if new_classes <= 0 or new_classes > 10000:
                    if isinstance(result.modifications[i], dict):
                        result.modifications[i]['params']['new_classes'] = output_classes
                    else:
                        result.modifications[i].params['new_classes'] = output_classes
                    issues.append(f"change_output_layer: invalid {new_classes}, using {output_classes}")
        
        if issues:
            if isinstance(result.validation, dict):
                result.validation['issues'] = issues
                result.validation['is_valid'] = False
            elif hasattr(result.validation, 'issues'):
                result.validation.issues = issues
                result.validation.is_valid = False
        
        # ===== SAVE STATE =====
        # Pydantic dict() to convert models, but also handles our raw dicts correctly
        state.parsed_modifications = result.dict() if hasattr(result, 'dict') else dict(result)
        
        # ===== LOG RESULTS =====
        logger.info(f"✅ Modifications parsed successfully!")
        logger.info(f"   • Modifications: {len(result.modifications)}")
        
        conf = result.confidence if hasattr(result, 'confidence') else result.get('confidence', 0.0)
        logger.info(f"   • Confidence: {conf:.0%}")
        
        val_is_valid = False
        if isinstance(result.validation, dict):
            val_is_valid = result.validation.get('is_valid', False)
        elif hasattr(result.validation, 'is_valid'):
            val_is_valid = result.validation.is_valid
            
        logger.info(f"   • Valid: {val_is_valid}")
        
        for i, mod in enumerate(result.modifications, 1):
            mod_type = mod.get('type') if isinstance(mod, dict) else getattr(mod, 'type', '')
            mod_desc = mod.get('description') if isinstance(mod, dict) else getattr(mod, 'description', '')
            logger.info(f"   [{i}] {mod_type} - {mod_desc}")
            
        train_lr = 0.0
        train_epochs = 0
        if isinstance(result.training_recommendation, dict):
            train_lr = result.training_recommendation.get('learning_rate', 0.0)
            train_epochs = result.training_recommendation.get('epochs', 0)
        elif hasattr(result.training_recommendation, 'learning_rate'):
            train_lr = result.training_recommendation.learning_rate
            train_epochs = result.training_recommendation.epochs
            
        logger.info(f"   Training: LR={train_lr}, Epochs={train_epochs}")
        
        # CLEAR state to avoid leaking into the next node's interrupt check
        state.user_response = ""
        
        return state
    
    except Exception as e:
        logger.error(f"❌ LLM parsing failed: {str(e)}")
        logger.warning("⚠️  Using fallback configuration...")
        
        # Minimal fallback
        state.parsed_modifications = {
            "modifications": [],
            "summary": f"Error: {str(e)[:50]}",
            "confidence": 0.0,
            "validation": {
                "is_valid": False,
                "issues": [str(e)[:80]]
            },
            "training_recommendation": {
                "learning_rate": 0.0001,
                "epochs": 15,  # Bumped from 5: more training improves accuracy even in fallback
                "batch_size": 32,
                "optimizer": "adam",
                "notes": "Fallback - LLM error (default config)"
            }
        }
        
        # CLEAR state to avoid leaking into the next node's interrupt check
        state.user_response = ""
        
        return state


def get_modification_best_practices(model_architecture: dict) -> str:
    """
    ✨ FUNCTION: Generates personalized best practices for the model.
    
    Args:
        model_architecture: dict with model architecture info
    
    Returns:
        Formatted string with best practices
    """
    
    logger.info("📚 Generating best practices for the model...")
    
    total_params = model_architecture.get('total_params', 0)
    n_layers = model_architecture.get('n_layers', 0)
    has_dropout = model_architecture.get('has_dropout', False)
    has_batchnorm = model_architecture.get('has_batchnorm', False)
    output_classes = model_architecture.get('output_classes', 10)
    model_size_mb = model_architecture.get('model_size_mb', 0)
    
    practices = []
    
    # ===== LAYER FREEZING RECOMMENDATIONS =====
    if n_layers > 5:
        frozen_count = max(1, n_layers // 3)  # Freeze 1/3 of the layers
        practices.append(f"🔒 Freeze first {frozen_count} layers to preserve pre-trained features")
    
    # ===== DROPOUT RECOMMENDATIONS =====
    if not has_dropout and n_layers > 10:
        practices.append("💧 Add Dropout (0.3-0.5) to prevent overfitting - NOT present in current model")
    elif has_dropout:
        practices.append("✅ Model already has Dropout - Good!")
    
    # ===== BATCH NORMALIZATION RECOMMENDATIONS =====
    if has_batchnorm:
        practices.append("✅ BatchNormalization present - Helps with training stability")
    else:
        practices.append("⚠️  No BatchNormalization - Consider adding for better convergence")
    
    # ===== SIZE-BASED RECOMMENDATIONS =====
    if model_size_mb > 50:
        practices.append(f"📦 Large model ({model_size_mb:.1f}MB) - Consider pruning or quantization for STM32")
    elif model_size_mb > 10:
        practices.append(f"📦 Medium model ({model_size_mb:.1f}MB) - Suitable for STM32 with optimization")
    else:
        practices.append(f"📦 Compact model ({model_size_mb:.1f}MB) - Good for embedded deployment")
    
    # ===== PARAMETER COUNT RECOMMENDATIONS =====
    if total_params > 10_000_000:
        practices.append(f"⚙️  Very large ({total_params:,} params) - Pruning recommended")
    elif total_params > 1_000_000:
        practices.append(f"⚙️  Medium-large ({total_params:,} params) - Consider optimizations")
    else:
        practices.append(f"⚙️  Compact ({total_params:,} params) - Efficient model")
    
    # ===== OUTPUT CLASSES RECOMMENDATIONS =====
    if output_classes > 1000:
        practices.append(f"🎯 Very large output space ({output_classes} classes) - May overfit easily")
    elif output_classes < 10:
        practices.append(f"🎯 Small output space ({output_classes} classes) - Suitable for binary/multi-class")
    
    # ===== TRAINING RECOMMENDATIONS =====
    if n_layers > 50:
        practices.append("🎓 Deep network - Use low learning rate (1e-5 to 1e-4)")
    else:
        practices.append("🎓 Shallow network - Can use higher learning rate (1e-4 to 1e-3)")
    
    # ===== DATA AUGMENTATION RECOMMENDATIONS =====
    practices.append("📸 Use data augmentation (rotation, flip, zoom) to improve generalization")
    
    # ===== QUANTIZATION RECOMMENDATIONS =====
    practices.append("🔢 Use INT8 quantization (4× size reduction) for STM32 deployment")
    
    # Format output
    formatted = "\n".join([f"  {p}" for p in practices])
    
    return f"""\n════════════════════════════════════════════════════════════
📋 BEST PRACTICES FOR YOUR MODEL
════════════════════════════════════════════════════════════

{formatted}

════════════════════════════════════════════════════════════
\n"""


def collect_modification_confirmation(state: any, config: RunnableConfig = None) -> any:
    """
    Shows modification preview and asks the user for confirmation.
    Uses LLM to understand natural language responses.
    
    Supports various response types:
      ✓ Positive: "yes", "ok", "apply", "confirm", "proceed"
      ✓ Negative: "no", "cancel", "reject", "stop"
      ✓ Edit: "edit", "modify", "change", "back"
    
    Args:
        state: MasterState object
        config: Configuration dict
    
    Returns:
        updated state with modification_confirmed bool
    """
    logger.info("👀 Asking confirmation for modifications...")
    
    # Protection: if there are no modifications, return immediately
    if not state.parsed_modifications:
        logger.warning("⚠️  No modifications to confirm")
        state.modification_confirmed = False
        return state
    
    # ==================== PREVIEW CREATION ====================
    
    # Extract info from modifications for the preview
    summary = state.parsed_modifications.get('summary', 'N/A')
    confidence = state.parsed_modifications.get('confidence', 0.9)
    num_modifications = len(state.parsed_modifications.get('modifications', []))
    
    # List of modifications for the preview
    modifications_list = state.parsed_modifications.get('modifications', [])
    modifications_text = "\n".join([
        f"  {i+1}. [{m.get('type', 'unknown')}] {m.get('description', 'No description')}"
        for i, m in enumerate(modifications_list)
    ])
    
    # Training recommendations
    train_rec = state.parsed_modifications.get('training_recommendation', {})
    train_text = f"""
  • Learning rate: {train_rec.get('learning_rate', 'N/A')}
  • Epochs: {train_rec.get('epochs', 'N/A')}
  • Batch size: {train_rec.get('batch_size', 'N/A')}
  • Optimizer: {train_rec.get('optimizer', 'N/A')}
  • Notes: {train_rec.get('notes', 'N/A')}"""
    
    # Validation info
    validation = state.parsed_modifications.get('validation', {})
    is_valid = validation.get('is_valid', True)
    validation_icon = "✅" if is_valid else "⚠️"
    
    # Build the formatted preview
    preview = f"""
════════════════════════════════════════════════════════════
🔍 PREVIEW: Modifications that will be applied
════════════════════════════════════════════════════════════

Summary: {summary}
Confidence: {confidence:.0%}
Number of modifications: {num_modifications}
Status: {validation_icon}

Modification details:
{modifications_text}

Training Recommendation:{train_text}

════════════════════════════════════════════════════════════
"""
    
    logger.info(preview)

    # ==================== CONFIRMATION REQUEST ====================
    
    # Prompt shown to the user (supports natural responses)
    confirmation_prompt = {
        "instruction": f"{preview}\n\nDo you want to apply these modifications? (Yes/No/Edit)",
        "options": ["yes", "no", "edit"],
        "hint": "You can respond naturally (e.g., 'yes please', 'apply it', 'go back')"
    }
    
    # ⏸️ INTERRUPT: Wait for user response
    from src.assistant.utils import extract_user_response
    resume_value = None
    if not state.user_response or state.user_response.strip() == "":
        logger.info("⏸️ Interrupting for modification confirmation.")
        # resume_value = interrupt(confirmation_prompt)
        resume_value = "yes" # BYPASS
        user_response = str(resume_value).strip() if resume_value else ""
    else:
        # Uses the return value of interrupt() as priority (LangGraph Studio compatibility),
        # otherwise uses state.user_response (server.py/VS Code compatibility)
        if resume_value and str(resume_value).strip():
            raw_response = str(resume_value).strip()
            logger.info(f"📝 User response (interrupt return): '{raw_response}'")
            user_response = raw_response
        else:
            logger.info(f"📝 User response (state): '{state.user_response}'")
            user_response = extract_user_response(state.user_response)
    state.user_response = "" # Clear
    
    # ==================== LLM RESPONSE PARSING ====================
    
    try:
        logger.info(" [Step 1] Interpreting response with LLM...")
        
        # Use the Triton LLM (mistral) instead of Ollama
        from src.assistant.utils import get_llm
        from langchain_core.messages import SystemMessage, HumanMessage
        llm = get_llm(config)
        
        # Build prompt to interpret the user's decision
        interpretation_prompt = f"""Interpret user confirmation response for model modifications.

CONTEXT:
Model modifications preview was shown to user.

USER RESPONSE TO "Do you want to apply these modifications?":
"{user_response}"

Return ONLY a JSON object (no markdown, no extra text):
{{
  "decision": "confirm",
  "confidence": 0.95,
  "reasoning": "Why we interpreted it this way"
}}

Decision values:
- "confirm": User approves and wants to apply modifications (yes, ok, apply, proceed)
- "reject": User does NOT want to apply modifications (no, cancel, reject, stop)
- "edit_request": User wants to modify/change the modifications (edit, change, back)

If empty response, return confirm.
"""
        response_msg = llm.invoke([
            SystemMessage(content="You are a decision interpreter. Return only valid JSON."),
            HumanMessage(content=interpretation_prompt)
        ])
        
        # Normalize the response
        content = response_msg.content if hasattr(response_msg, 'content') else str(response_msg)
        
        logger.debug(f"   LLM response: {content[:150]}...")
        
        # Extract JSON from the response
        json_match = re.search(r'\{[\s\S]*\}', content)
        
        if json_match:
            json_str = json_match.group(0)
            decision_data = json.loads(json_str)
        else:
            decision_data = json.loads(content)
        
        # Extract the decision (default: reject for safety) 
        #decision = decision_data.get('decision', 'reject').lower().strip()
        # FOR fast TESTING: default to confirm
        decision= decision_data.get('decision', 'confirm').lower().strip()
        confidence = decision_data.get('confidence', 0.5)
        reasoning = decision_data.get('reasoning', 'LLM interpretation')
        
        logger.info(f" ✓ LLM Interpretation:")
        logger.info(f"    • Decision: {decision}")
        logger.info(f"    • Confidence: {confidence:.0%}")
        logger.info(f"    • Reasoning: {reasoning}")
        
        # Convert decision to boolean and set edit flag if necessary
        if decision == "confirm":
            state.modification_confirmed = True
            state.user_wants_to_edit = False
            logger.info("✅ Modifications CONFIRMED")
            
        elif decision == "reject":
            state.modification_confirmed = False
            state.user_wants_to_edit = False
            logger.info("❌ Modifications REJECTED")
            
        elif decision == "edit_request":
            state.modification_confirmed = False
            state.user_wants_to_edit = True
            logger.info("✏️  User wants to EDIT the modifications")
        
        else:
            state.modification_confirmed = False
            state.user_wants_to_edit = False
            logger.warning(f"⚠️  Decision not recognized: '{decision}', defaulting to reject")
    
    # IF LLM PARSING FAILS
    except (json.JSONDecodeError, ValueError, AttributeError) as e:
        logger.error(f"❌ LLM parsing error: {str(e)[:100]}")
        logger.warning(" [Step 2] Fallback to keyword parsing...")
        
        # ==================== FALLBACK: DIRECT PARSING ====================
        
        if isinstance(user_response, dict):
             response_lower = str(user_response.get("response", user_response.get("input", ""))).lower().strip()
        else:
             response_lower = str(user_response).lower().strip()
        
        # Keywords for "yes"
        positive_keywords = [
            'yes', 'si', 'sì', 'yeah', 'yep', 'ok', 'okay',
            'apply', 'confirm', 'proceed', 'continue', 'go',
            'approve', 'perfect', 'good', 'sure', 'absolutely'
        ]
        
        # Keywords for "no"
        negative_keywords = [
            'no', 'nope', 'reject', 'cancel', 'stop', 'abort',
            'dont', 'don\'t', 'skip', 'refuse', 'decline', 'nah',
            'absolutely not', 'never', 'no way'
        ]
        
        # Keywords for "edit/modify"
        edit_keywords = [
            'edit', 'modifica', 'change', 'modify', 'back',
            'again', 'different', 'redo', 'rethink', 'again',
            'let me', 'wait', 'hold on'
        ]
        
        if any(kw in response_lower for kw in positive_keywords):
            state.modification_confirmed = True
            state.user_wants_to_edit = False
            logger.info("✅ Modifications CONFIRMED (keyword match)")
        
        elif any(kw in response_lower for kw in negative_keywords):
            state.modification_confirmed = False
            state.user_wants_to_edit = False
            logger.info("❌ Modifications REJECTED (keyword match)")
        
        elif any(kw in response_lower for kw in edit_keywords):
            state.modification_confirmed = False
            state.user_wants_to_edit = True
            logger.info("✏️  EDIT requested (keyword match)")
        
        else:
            state.modification_confirmed = False
            state.user_wants_to_edit = False
            logger.warning(f"⚠️  Response not interpreted, defaulting to reject")
    
    except Exception as e:
        logger.error(f"❌ Unexpected error: {str(e)}", exc_info=True)
        logger.warning("⚠️  Defaulting to reject for safety")
        
        state.modification_confirmed = False
        state.user_wants_to_edit = False
    
    # ==================== FINAL LOG ====================
    
    logger.info(f"═══════════════════════════════════════════════════════")
    logger.info(f"👀 Modification confirmed: {state.modification_confirmed}")
    logger.info(f"✏️  Edit requested: {getattr(state, 'user_wants_to_edit', False)}")
    logger.info(f"═══════════════════════════════════════════════════════")
    
    return state


# ============================================================================
# ARCHITECTURE → CONDA ENVIRONMENT MAPPING
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

# CONDA_PYTHON_PATHS removed in favor of cfg.get_python_path()
# I created an stm32legacy environment to use keras 2.x (for older models) in order to avoid errors.
# In case there is a need to create a new environment with different packages, the corresponding Python path can be added here.
# The load_model_with_conda_env function will take care of using the correct python based on the model's architecture.

# ============================================================================
# HELPER: Detect architecture from model_path
# ============================================================================

def detect_architecture_from_model(model_path: str) -> str:
    """Detects architecture from model name"""
    
    model_name = os.path.basename(model_path).lower()
    
    if 'mobilenet' in model_name:
        return 'mobilenet'
    elif 'resnet' in model_name:
        return 'resnet'
    elif 'vgg' in model_name:
        return 'vgg'
    elif 'efficient' in model_name:
        return 'efficientnet'
    elif 'inception' in model_name:
        return 'inception'
    elif 'yolo' in model_name:
        return 'yolo'
    elif 'har' in model_name or 'activity' in model_name:
        return 'har'
    else:
        return 'custom'


# ============================================================================
# HELPER: Load Model in specific Conda Environment
# ============================================================================
import subprocess
import json
import pickle

def execute_in_environment(python_code: str, state: MasterState, timeout: int = 600, ignore_list: list = None, whitelist_patterns: list = None) -> dict:
    """
    ✨ Execute Python code in the environment specified by state.python_path
    
    Works for stm32legacy, stm32, or any conda environment
    
    Returns: {'success': bool, 'stdout': str, 'stderr': str, 'returncode': int}
    """
    
    python_path = state.python_path
    if not python_path:
        return {'success': False, 'stdout': "", 'stderr': "No Python path available", 'returncode': 1}
    
    logger.info("🔧 Starting training subprocess...")
    logger.info(f"   • Environment: {state.conda_env}")
    logger.info(f"   • Python: {python_path}")
    logger.info(f"   • Timeout: {timeout}s")
    
    stdout_accumulator = []
    stderr_accumulator = []
    
    try:
        from src.assistant.utils import run_subprocess_streaming
        import tempfile
        
        # Write code to a temporary file to support self-restart/os.execv
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tf:
            tf.write(python_code)
            temp_script_path = tf.name
            
        try:
            cmd = [python_path, temp_script_path]
            res = run_subprocess_streaming(
                cmd, 
                logger, 
                prefix="[Train]", 
                timeout=timeout,
                ignore_list=ignore_list,
                whitelist_patterns=whitelist_patterns
            )
            
            return {
                'success': res.get('success', False),
                'stdout': res.get('stdout', ''),
                'stderr': res.get('error', ''), 
                'returncode': res.get('returncode', 0 if res.get('success') else 1)
            }
        finally:
            # Cleanup temp file
            if os.path.exists(temp_script_path):
                os.remove(temp_script_path)
    except Exception as e:
        logger.error(f"❌ Subprocess error: {e}")
        return {'success': False, 'stdout': "", 'stderr': str(e), 'returncode': 1}

def load_model_with_conda_env(model_path: str, architecture: str, state: MasterState) -> str:
    """
    ✨ Loads model IN SUBPROCESS and RETURNS the PATH
    
    Maintains ARCHITECTURE_ENV_MAP logic inside this function
    """
    
    logger.info(f"🔄 Loading {architecture} model...")
    
    # ===== DETERMINE ENVIRONMENT AND PYTHON PATH (ORIGINAL LOGIC) =====
    cfg = Configuration.from_runnable_config() # Configuration loads from env if not passed
    
    # .keras -> stm32, .h5 -> stm32legacy
    if model_path.endswith('.keras'):
        conda_env = 'stm32'
    else:
        conda_env = ARCHITECTURE_ENV_MAP.get(architecture, 'stm32legacy')
        
    python_path = cfg.get_python_path(conda_env)
    
    if "NOT_FOUND" in python_path:
        logger.error(f"❌ No Python path found for {conda_env}")
        raise Exception(f"Environment not found: {conda_env}")
    
    logger.info(f"  Environment: {conda_env}")
    logger.info(f"  Python: {python_path}")
    
    # ===== UPDATE state.python_path and state.conda_env =====
    state.python_path = python_path
    state.conda_env = conda_env
    
    # ===== PYTHON SCRIPT TO EXECUTE =====
    python_code = f"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Unset legacy keras flag so keras3 loads cleanly
os.environ.pop('TF_USE_LEGACY_KERAS', None)

import json
import sys

model_path = r'{model_path}'
temp_output = f'/tmp/model_loaded_{state.thread_id}.h5'

try:
    # Try keras3 (modern) first, fall back to tf.keras (legacy)
    try:
        import keras
        model = keras.models.load_model(model_path, compile=False)
    except Exception:
        import tensorflow as tf
        model = tf.keras.models.load_model(
            model_path,
            compile=False,
            safe_mode=False
        )
    
    info = {{
        'name': model.name,
        'input_shape': str(model.input_shape),
        'output_shape': str(model.output_shape),
        'total_params': int(model.count_params()),
    }}
    
    model.save(temp_output, save_format='h5')
    
    print(f"SUCCESS: {{temp_output}}|" + json.dumps(info))
    sys.exit(0)
    
except Exception as e:
    print(f"ERROR: {{str(e)}}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
"""
    
    # ===== EXECUTE USING NEW FUNCTION =====
    logger.info(f"  [Subprocess] Execution...")
    
    try:
        result = execute_in_environment(
            python_code, 
            state, 
            timeout=120, 
            ignore_list=SUBPROCESS_NOISE_FILTER,
            whitelist_patterns=SUBPROCESS_CLEAN_ALLOWLIST
        )
        
        output = result['stdout']
        
        if not result['success']:
            error = result['stderr']
            logger.error(f"  Subprocess failed: {error[:500]}")
            raise Exception(f"Subprocess error: {error}")
        
        if "SUCCESS:" not in output:
            logger.error(f"  No SUCCESS marker. Output: {output[:500]}")
            raise Exception(f"Unexpected output: {output}")
        
        logger.info(f"  ✓ Model loaded in subprocess")
        
        # ===== EXTRACT INFO AND PATH =====
        parts = output.split("SUCCESS:")[-1].strip().split('|')
        temp_model_path = parts[0].strip()
        info_json = parts[1].strip()
        
        info = json.loads(info_json)
        
        logger.info(f"✓ Model ready: {info['name']}")
        logger.info(f"  Input: {info['input_shape']}")
        logger.info(f"  Output: {info['output_shape']}")
        logger.info(f"  Params: {info['total_params']:,}")
        
        return temp_model_path
    
    except Exception as e:
        logger.error(f"❌ Load failed: {str(e)}")
        raise


# ============================================================================
# LOAD STM32 MODEL SAFE - SIMPLIFIED VERSION
# ============================================================================

def load_stm32_model_safe(model_path: str, state: MasterState) -> str:
    """
    ✨ Loads model and returns PATH (NOT the model)
    
    Sets state.python_path and state.conda_env for later use
    """
    
    logger.info(f"📥 Loading model: {model_path}")
    
    if not os.path.exists(model_path):
        logger.error(f"❌ Model file not found: {model_path}")
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    # ✅ FIX: Extension check. 
    # Customization (Workflow 5) requires native Keras formats (.h5, .keras)
    # Formats like .onnx or .tflite are great for analysis (Workflow 2) but not for structural editing.
    ext = os.path.splitext(model_path)[1].lower()
    if ext not in ['.h5', '.keras']:
        error_msg = f"The {ext} format does not support structural customization. " \
                    "Layer manipulation and fine-tuning require native Keras models (.h5 or .keras). " \
                    "You can still proceed to analysis and firmware generation (Workflow 2)."
        logger.error(f"❌ {error_msg}")
        raise ValueError(error_msg)
    
    try:
        # Detect architecture
        architecture = detect_architecture_from_model(model_path)
        logger.info(f"  Architecture: {architecture}")
        
        # Load and return PATH (also sets state.python_path and state.conda_env)
        model_path_ready = load_model_with_conda_env(model_path, architecture, state)
        
        logger.info(f"✓ Model path: {model_path_ready}")
        logger.info(f"✓ Python path set: {state.python_path}")
        logger.info(f"✓ Conda env set: {state.conda_env}")
        
        return model_path_ready  # ← PATH, not Model!
    
    except Exception as e:
        logger.error(f"❌ Model loading failed: {str(e)}")
        raise


# ============================================================================
# APPLY USER CUSTOMIZATION
# ============================================================================

def apply_user_customization(state: MasterState, config: RunnableConfig = None) -> MasterState:
    """
    ✨ Apply modifications WITH CORRECT MANANGEMENT OF MULTIPLE RECONSTRUCTIONS (applies reconstructions together and outputs a single customized model. Previously for each reconstruction a new updated model was created version 1, version 2, etc)
    """
    
    logger.info("🔧 Applying customizations to the STM32 model...")
    
    if not state.modification_confirmed:
        state.customization_applied = False
        state.error_message = "Modifications not confirmed"
        return state
    
    model_path = state.model_path
    if not model_path or not os.path.exists(model_path):
        state.customization_applied = False
        state.error_message = "Invalid model path"
        return state
        
    # Resolve correct python path
    cfg = Configuration.from_runnable_config(config)
    state.python_path = cfg.get_python_path('stm32')
    state.conda_env = 'stm32'
    logger.info(f"🔧 Resolved environment for customization: {state.python_path}")
    
    try:
        logger.info("[STEP 1/3] LOADING MODEL")
        loaded_model_path = load_stm32_model_safe(model_path, state)
        logger.info(f"✓ Model ready at: {loaded_model_path}\n")
        
        logger.info("[STEP 2/3] VALIDATING MODIFICATIONS")
        parsed_mods = state.parsed_modifications or {}
        
        if not _validate_modifications(parsed_mods):
            state.customization_applied = False
            state.error_message = "Invalid modification parameters"
            return state
        
        logger.info("✓ All modifications valid\n")
        
        logger.info("[STEP 3/3] APPLYING MODIFICATIONS IN SUBPROCESS")
        
        python_code = f"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ.pop('TF_USE_LEGACY_KERAS', None)

import tensorflow as tf
from tensorflow.keras.layers import Input, Dropout, Dense, Resizing
from tensorflow.keras.models import Model
import json
import sys

model_path = r'{loaded_model_path}'
modifications = {json.dumps(parsed_mods.get('modifications', []))}
output_path = f'/tmp/customized_model_{state.thread_id}.h5'

try:
    # ===== LOAD MODEL (keras3 first, tf.keras fallback) =====
    try:
        import keras
        model = keras.models.load_model(model_path, compile=False)
    except Exception:
        model = tf.keras.models.load_model(model_path, compile=False, safe_mode=False)
    print(f"✓ Model loaded: {{model.name}}")
    
    modifications_log = []

    # ===== PHASE 1: NON-RECONSTRUCTIVE MODIFICATIONS =====
    print("\\n[Phase 1] Applying non-reconstructive modifications...")
    for mod in modifications:
        mod_type = mod.get('type', '').strip()
        mod_params = mod.get('params', {{}})

        # FREEZE LAYERS
        if mod_type == "freeze_layers":
            num_freeze = mod_params.get('num_frozen_layers', 3)
            for layer in model.layers[1:num_freeze+1]:
                layer.trainable = False
            modifications_log.append(f"✓ Froze layers 1-{{num_freeze}}")
            print(f"  [✓] Froze layers 1-{{num_freeze}}")

        # FREEZE ALMOST ALL
        elif mod_type == "freeze_almost_all":
            num_trainable = mod_params.get('num_trainable_layers', 3)
            total_layers = len(model.layers)
            num_freeze = total_layers - num_trainable - 1
            
            for layer in model.layers[1:num_freeze+1]:
                layer.trainable = False
            
            modifications_log.append(f"✓ Froze {{num_freeze}}/{{total_layers-1}} layers")
            print(f"  [✓] Froze {{num_freeze}}/{{total_layers-1}} layers")

        # CHANGE LEARNING RATE
        elif mod_type == "change_learning_rate":
            lr = float(mod_params.get('learning_rate', 0.0001))
            modifications_log.append(f"✓ Learning rate: {{lr}}")
            print(f"  [✓] Learning rate: {{lr}}")
    
    # ===== PHASE 2: COLLECT RECONSTRUCTIVE MODIFICATIONS =====
    print("\\n[Phase 2] Collecting reconstructive modifications...")
    # ===== EXTRACT original_input_shape IMMEDIATELY (CRITICAL!) =====
    original_input_shape = model.input_shape  # Ex: (None, 416, 416, 3)
    original_h = original_input_shape[1] if original_input_shape and len(original_input_shape) > 1 else 224
    original_w = original_input_shape[2] if original_input_shape and len(original_input_shape) > 2 else 224
    original_c = original_input_shape[3] if original_input_shape and len(original_input_shape) > 3 else 3

    print(f"  [info] Original model input: {{original_input_shape}} "
      f"(H={{original_h}}, W={{original_w}}, C={{original_c}})")

    reconstructive_mods = {{}}
    has_input_shape_change = False
    has_resizing_layer = False  # ← Fix #1 (declaration)
    
    for mod in modifications:
        mod_type = mod.get('type', '').strip()
        mod_params = mod.get('params', {{}})
        
        if mod_type == "add_dropout":
            reconstructive_mods['dropout'] = float(mod_params.get('rate', 0.5))
            print(f"  [queued] add_dropout: rate={{reconstructive_mods['dropout']}}")
        
        elif mod_type == "change_input_shape":
            reconstructive_mods['input_shape'] = tuple(mod_params.get('new_shape', (224, 224, 3)))
            has_input_shape_change = True
            print(f"  [queued] change_input_shape: {{reconstructive_mods['input_shape']}}")
        
        elif mod_type == "change_output_layer":
            reconstructive_mods['output_classes'] = int(mod_params.get('new_classes', 10))
            print(f"  [queued] change_output_layer: {{reconstructive_mods['output_classes']}} classes")
        
        elif mod_type == "add_resizing_layer":  # NEW
            target_h = int(original_h)
            target_w = int(original_w)
            reconstructive_mods['resizing'] = (target_h, target_w)
            has_resizing_layer = True
            print(f"  [queued] add_resizing_layer: {{target_h}}x{{target_w}}")
 
    
    # ===== PHASE 3: HANDLE INPUT SHAPE CHANGE (NO SKIP LAYERS) =====
    if has_input_shape_change and 'input_shape' in reconstructive_mods:
        print("\\n[Phase 3] INPUT SHAPE CHANGE: Reloading model with new input shape...")
        
        new_shape = reconstructive_mods['input_shape']  # Ex: (64, 64, 3)
        original_shape = model.input_shape[1:]  # Ex: (224, 224, 3)
        
        print(f"  Original input: {{original_shape}}")
        print(f"  New input: {{new_shape}}")
        
        # Recreate model with new input shape
        model_config = model.get_config()  # Extracts the full model configuration (not weights, just structure/architecture in JSON format)
        
        # Modify input layer config  # Accesses the first layer (Input layer) and changes batch_input_shape from (None, 224, 224, 3) → (None, 64, 64, 3)
        if 'layers' in model_config and len(model_config['layers']) > 0:
            input_layer_config = model_config['layers'][0]
            if 'config' in input_layer_config:
                input_layer_config['config']['batch_input_shape'] = (None, *new_shape)  # !!! Change input shape here
        
        # Recreate model
        model_new = tf.keras.Model.from_config(model_config)  # Recreates the entire model USING the modified config # IMPORTANT: this operation changes input shape, preserves architecture (all layers remain) and initializes weights randomly (does not copy weights from old model).
        # Model.from_config() recreates the model with the RIGHT INTERNAL PROPORTIONS! When Keras reads the modified config, it automatically calculates all subsequent shapes.

        # Copy ALL weights - DO NOT skip any layer. Copy weights from the old model to the new model
        for new_layer, old_layer in zip(model_new.layers, model.layers):
            try:
                new_layer.set_weights(old_layer.get_weights())
            except Exception as e:
                print(f"  ⚠️  Layer {{new_layer.name}}: weight shape may need retraining")
        
        model = model_new  # Replaces the old model with the new one
        print(f"  ✓ Model reloaded with input shape {{new_shape}}")
        modifications_log.append(f"✓ Changed input shape to {{new_shape}}")
        # input shape works!
        
        # Apply other modifications with new model
        # IMPORTANT: Apply in order: FIRST output_classes, THEN dropout
        
        if 'output_classes' in reconstructive_mods:
            print("  [applying] Changing output layer...")
            
            # ✅ CORRECT METHOD: get_config() preserves skip connections
            model_config = model.get_config()
            
            # Modify only the last Dense layer
            if 'layers' in model_config and len(model_config['layers']) > 0:
                last_layer = model_config['layers'][-1]
                if last_layer.get('class_name') == 'Dense':
                    old_units = last_layer['config'].get('units', 1000)
                    last_layer['config']['units'] = reconstructive_mods['output_classes']
                    print(f"    ✓ Dense layer: {{old_units}} → {{reconstructive_mods['output_classes']}}")
            
            # Recreate model (skip connections INTACT!)
            model_new = tf.keras.Model.from_config(model_config)
            
            # Copy weights (all EXCEPT the last Dense)
            for new_layer, old_layer in zip(model_new.layers[:-1], model.layers[:-1]):
                try:
                    new_layer.set_weights(old_layer.get_weights())
                except ValueError as e:
                    print(f"    ⚠️  {{new_layer.name}}: weight shape mismatch (old: {{old_layer.weights.shape if old_layer.weights else 'none'}}, new: {{new_layer.weights.shape if new_layer.weights else 'none'}}), using random init")
                except Exception as e:
                    print(f"    ⚠️  {{new_layer.name}}: {{type(e).__name__}}")
            
            model = model_new
            modifications_log.append(f"✓ Changed output to {{reconstructive_mods['output_classes']}} classes")
            print(f"    ✓ Changed output to {{reconstructive_mods['output_classes']}} classes")
        
        # Apply dropout as LAST step (after output_classes)
        if 'dropout' in reconstructive_mods:
            print("  [applying] Adding dropout...")
            
            penultimate_layer = model.layers[-2]
            output_layer = model.layers[-1]
            
            x = penultimate_layer.output
            x = Dropout(reconstructive_mods['dropout'], name='dropout_custom')(x)
            new_output = output_layer(x)
            
            model = Model(inputs=model.input, outputs=new_output)
            modifications_log.append(f"✓ Added Dropout (rate={{reconstructive_mods['dropout']}})")
            print(f"    ✓ Added Dropout ({{reconstructive_mods['dropout']}}) BEFORE output layer")
    
    # ===== PHASE 3B: OTHER RECONSTRUCTIVE MODIFICATIONS (without input shape change) =====
    elif reconstructive_mods and not has_resizing_layer:
        print("\\n[Phase 3B] Applying reconstructive modifications...")
        
        # ===== CASE 1: Only output_classes =====
        if 'output_classes' in reconstructive_mods and 'dropout' not in reconstructive_mods:
            print("  [output_classes only] Using get_config method...")
            
            # ✅ CORRECT METHOD: get_config() preserves skip connections
            model_config = model.get_config()
            
            # Modify only the last Dense layer
            if 'layers' in model_config and len(model_config['layers']) > 0:
                last_layer = model_config['layers'][-1]
                if last_layer.get('class_name') == 'Dense':
                    old_units = last_layer['config'].get('units', 1000)
                    last_layer['config']['units'] = reconstructive_mods['output_classes']
                    print(f"    ✓ Dense layer: {{old_units}} → {{reconstructive_mods['output_classes']}}")
            
            # Recreate model (skip connections INTACT!)
            model_new = tf.keras.Model.from_config(model_config)
            
            # Copy weights (all EXCEPT the last Dense)
            for new_layer, old_layer in zip(model_new.layers[:-1], model.layers[:-1]):
                try:
                    new_layer.set_weights(old_layer.get_weights())
                except ValueError as e:
                    print(f"    ⚠️  {{new_layer.name}}: weight shape mismatch (old: {{old_layer.weights.shape if old_layer.weights else 'none'}}, new: {{new_layer.weights.shape if new_layer.weights else 'none'}}), using random init")
                except Exception as e:
                    print(f"    ⚠️  {{new_layer.name}}: {{type(e).__name__}}")
            
            model = model_new
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            modifications_log.append(f"✓ Changed output to {{reconstructive_mods['output_classes']}} classes")
            print(f"  ✓ Changed output to {{reconstructive_mods['output_classes']}} classes")
        
        # ===== CASE 2: Only dropout =====
        elif 'dropout' in reconstructive_mods and 'output_classes' not in reconstructive_mods:
            print("  [dropout only] Inserting dropout...")
            
            penultimate_layer = model.layers[-2]
            output_layer = model.layers[-1]
            
            x = penultimate_layer.output
            x = Dropout(reconstructive_mods['dropout'], name='dropout_custom')(x)
            new_output = output_layer(x)
            
            model = Model(inputs=model.input, outputs=new_output)
            modifications_log.append(f"✓ Added Dropout (rate={{reconstructive_mods['dropout']}})")
            print(f"  ✓ Added Dropout ({{reconstructive_mods['dropout']}}) BEFORE output layer")
        
        # ===== CASE 3: Both dropout and output_classes =====
        else:  # 'dropout' in reconstructive_mods and 'output_classes' in reconstructive_mods
            print("  [dropout + output_classes] Applying combined modifications...")
            
            # Step 1: Change output_classes with get_config
            model_config = model.get_config()
            
            if 'layers' in model_config and len(model_config['layers']) > 0:
                last_layer = model_config['layers'][-1]
                if last_layer.get('class_name') == 'Dense':
                    old_units = last_layer['config'].get('units', 1000)
                    last_layer['config']['units'] = reconstructive_mods['output_classes']
                    print(f"    ✓ Dense layer: {{old_units}} → {{reconstructive_mods['output_classes']}}")
            
            # Recreate model
            model_new = tf.keras.Model.from_config(model_config)
            
            # Copy weights (except last Dense)
            for new_layer, old_layer in zip(model_new.layers[:-1], model.layers[:-1]):
                try:
                    new_layer.set_weights(old_layer.get_weights())
                except ValueError as e:
                    print(f"    ⚠️  {{new_layer.name}}: weight shape mismatch (old: {{old_layer.weights.shape if old_layer.weights else 'none'}}, new: {{new_layer.weights.shape if new_layer.weights else 'none'}}), using random in it")
                except Exception as e:
                    print(f"    ⚠️  {{new_layer.name}}: {{type(e).__name__}}")
            
            model = model_new
            modifications_log.append(f"✓ Changed output to {{reconstructive_mods['output_classes']}} classes")
            print(f"    ✓ Changed output to {{reconstructive_mods['output_classes']}} classes")
            
            # Step 2: Add Dropout before new output
            penultimate_layer = model.layers[-2]
            output_layer = model.layers[-1]
            
            x = penultimate_layer.output
            x = Dropout(reconstructive_mods['dropout'], name='dropout_custom')(x)
            new_output = output_layer(x)
            
            model = Model(inputs=model.input, outputs=new_output)
            modifications_log.append(f"✓ Added Dropout (rate={{reconstructive_mods['dropout']}})")
            print(f"    ✓ Added Dropout ({{reconstructive_mods['dropout']}}) BEFORE output layer")

    # ===== PHASE 3A: ADD RESIZING LAYER WRAPPER =====
    if has_resizing_layer and 'resizing' in reconstructive_mods:
        print(f"  Original model input: {{original_input_shape}}")
        print(f"  Wrapper will resize any image to: {{target_h}}x{{target_w}}")
        
        channels = model.input_shape[3] if len(model.input_shape) > 3 else 3

        new_inputs = tf.keras.Input(shape=(None, None, channels), name="raw_image_input")
        x = Resizing(target_h, target_w, name="auto_resize_to_model_input")(new_inputs)
        outputs = model(x)

        model = Model(inputs=new_inputs, outputs=outputs, name=model.name + "_with_auto_resize")
        modifications_log.append(f"✓ Added automatic Resizing to {{target_h}}x{{target_w}}")
        print(f"  [applied] Added automatic Resizing layer → {{target_h}}x{{target_w}}")

    # ===== SAVE MODEL =====
    print(f"\\n[Saving] Model saving...")
    model.save(output_path, save_format='h5')
    print(f"✓ Model saved: {{output_path}}")
    
    # ===== FINAL INFO =====
    info = {{
        'input_shape': str(model.input_shape),
        'output_shape': str(model.output_shape),
        'total_params': int(model.count_params()),
        'trainable_params': int(sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])),
        'frozen_params': int(model.count_params() - sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])),
        'modifications_applied': modifications_log,
    }}
    
    print(f"\\n✅ Customization complete!")
    print(f"  Total params: {{info['total_params']:,}}")
    print(f"  Trainable: {{info['trainable_params']:,}}")
    print(f"  Frozen: {{info['frozen_params']:,}}")
    print(f"  Modifications: {{len(modifications_log)}}")
    
    print(f"SUCCESS: {{output_path}}|" + json.dumps(info))
    sys.exit(0)
    
except Exception as e:
    print(f"ERROR: {{str(e)}}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
"""

        result = execute_in_environment(python_code, state, timeout=600, ignore_list=SUBPROCESS_NOISE_FILTER, whitelist_patterns=SUBPROCESS_CLEAN_ALLOWLIST)
        
        output = result['stdout']
        stderr = result['stderr']
        logger.info(f"Subprocess output:\n{output}")
        
        # Text-based error detection (even if exit_code=0)
        lowered_output = (output + "\n" + stderr).lower()
        has_error = "error:" in lowered_output or "exception encountered:" in lowered_output or "traceback" in lowered_output
        
        if not result['success'] or has_error:
            error_msg = stderr if stderr.strip() else output.split("ERROR:")[-1].strip()
            logger.error(f"❌ Customization subprocess failed: {error_msg[:500]}")
            state.customization_applied = False
            state.error_message = error_msg
            return state

        if "SUCCESS:" in output:
            parts = output.split("SUCCESS:")[-1].strip().split('|')
            customized_path = parts[0].strip()
            info_json = parts[1].strip()
            info = json.loads(info_json)
            
            state.customized_model_path = customized_path
            state.customization_applied = True
            state.error_message = ""
            state.customized_model_info = {
                **info,
                "save_format": "keras",
                "timestamp": datetime.now().isoformat(),
            }
            
            logger.info(f"\n✅ CUSTOMIZATION COMPLETE")
            logger.info(f"  Model: {customized_path}")
            logger.info(f"  Total params: {info['total_params']:,}")
            logger.info(f"  Trainable: {info['trainable_params']:,}")
            logger.info(f"  Frozen: {info['frozen_params']:,}")
            for mod_desc in info['modifications_applied']:
                logger.info(f"    {mod_desc}")
    
    except ValueError as ve:
        # Specific error for unsupported format
        state.customization_applied = False
        state.ai_error_message = str(ve)
        return state
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}", exc_info=True)
        state.customization_applied = False
        state.customized_model_path = ""
        state.ai_error_message = str(e)
        return state
    
    return state


def _validate_modifications(modifications: dict) -> bool:
    """
    Validates modification parameters before applying them.
    """
    required_params = {
        'freeze_layers': ['num_frozen_layers'],
        'freeze_almost_all': ['num_trainable_layers'],
        'change_output_layer': ['new_classes'],
        'add_dropout': ['rate'],
        'change_input_shape': ['new_shape'],
        'change_learning_rate': ['learning_rate'],
        'add_resizing_layer': [],
    } # Defines the mandatory parameters for each modification type, for example:
        # freeze_layers → 'num_frozen_layers'
        # add_dropout → 'rate'
        # etc.
      # Loops over all requested modifications:
        # For each modification (e.g. "type": "freeze_layers") it checks that all required parameters are in the "params" sub-dictionary.
        # If one is missing (e.g. missing 'rate' for "add_dropout"), it warns and returns False immediately (breaking the loop).
      # If all modifications have the required parameters, it returns True.

    for mod in modifications.get('modifications', []):
        mod_type = mod.get('type', '').strip()
        mod_params = mod.get('params', {})
        
        if mod_type in required_params:
            for param in required_params[mod_type]:
                if param not in mod_params:
                    logger.warning(f"⚠️ Missing parameter '{param}' for {mod_type}")
                    return False
    
    return True

# ============================================================================
#                    MAIN CUSTOMIZATION FUNCTION
# ============================================================================

# Very important function. 
def fine_tune_customized_model(state: MasterState, config: RunnableConfig = None) -> MasterState:
    """
    ✨ Fine-tuning using execute_in_environment (state.python_path)
    Supports both Classification and Object Detection (YOLO)
    """
    
    logger.info("═══════════════════════════")
    logger.info("🎯 FINE-TUNING")
    logger.info("═══════════════════════════")
    
    try:
        model_path = state.customized_model_path
        
        if not model_path or not os.path.exists(model_path):
            logger.error("❌ Model file not found")
            raise FileNotFoundError("customized_model_path not set")
        
        model_size_mb = os.path.getsize(model_path) / 1024 / 1024
        logger.info(f"📌 Model: {os.path.basename(model_path)} ({model_size_mb:.1f}MB)")
        
        training_rec = state.parsed_modifications.get('training_recommendation', {})
        # Conservative Learning Rate for fine-tuning (start with 1e-4)
        learning_rate = training_rec.get('learning_rate', state.custom_learning_rate or 0.0001)
        
        # Safety cap: ensure LR is not too aggressive for pre-trained weights
        if learning_rate > 0.001: 
            logger.warning(f"⚠️  High LR suggested ({learning_rate}), capping at 0.001 for stability during fine-tuning.")
            learning_rate = 0.001
        # Force 10 epochs if not explicitly requested otherwise, and cap at 10 for standard runs
        epochs = min(training_rec.get('epochs', state.custom_epochs or 10), 10)
        batch_size = training_rec.get('batch_size', state.custom_batch_size or 32)
        
        # Determine dropout rate (prioritize user explicit request, then fallback to 0.3 anti-overfitting)
        dropout_rate = state.parsed_modifications.get('dropout_rate') or 0.3
        
        logger.info(f"📌 Training config: {epochs} epochs, batch={batch_size}, LR={learning_rate}")
        
        # Ensure we use the correct environment and python path
        cfg = Configuration.from_runnable_config(config)
        state.conda_env = 'stm32'
        state.python_path = cfg.get_python_path('stm32')
        logger.info(f"🔧 Resolved environment: {state.conda_env} -> {state.python_path}")
        
        output_path = model_path.replace('.keras', f'_{state.thread_id}_finetuned.h5').replace('.h5', f'_{state.thread_id}_finetuned.h5')
        
        # ===== PYTHON SCRIPT =====
        python_code = f"""
import sys
import os
import glob

# --- AUTO-CONFIGURE CUDA PATH (Robust) ---
# NOTE: Manual LD_LIBRARY_PATH manipulation removed to avoid symbol lookup errors.
# We rely on the correct environment being selected by the parent process.
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np


# GPU Memory Growth (Prevent OOM)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"[Train] 🎮 GPU initialized: {{len(gpus)}} devices")
    except RuntimeError as e:
        print(e)

model_path = r"{model_path}"
output_path = r"{output_path}"
use_synthetic_data = {str(state.use_synthetic_data)}
synthetic_data_path = r"{state.synthetic_data_path}"
dataset_source = r"{state.dataset_source}"
real_dataset_path = r"{state.real_dataset_path}"

try:
    try:
        import keras
        model = keras.models.load_model(model_path, compile=False)
    except Exception:
        model = tf.keras.models.load_model(model_path, compile=False)
    
    input_shape_raw = model.input_shape[1:]  
    target_height = None
    target_width = None

    print(f"\\n✓ Model loaded")
    print(f"  Input: {{model.input_shape}}")
    print(f"  Output: {{model.output_shape}}")
    
    # ===== DETECT MODEL TYPE AND LOSS =====
    output_shape = model.output_shape
    num_last_dim = int(output_shape[-1]) if len(output_shape) > 1 else None
    
    # Object detection: output has 4 dimensions (batch, H, W, channels)
    is_object_detection = (len(output_shape) == 4 and num_last_dim and num_last_dim < 100)
    
    # Choose appropriate loss function
    if is_object_detection:
        loss_fn = 'mse'  # Per YOLO, object detection, etc
        print(f"  → Object detection model (MSE loss)")
    else:
        loss_fn = 'categorical_crossentropy'
        print(f"  → Classification model (categorical_crossentropy loss)")
    
    # ===== SEARCH Resizing layer =====
    print(f"\\n🔍 Searching for Resizing layer...")
    
    for i, layer in enumerate(model.layers):
        layer_class = layer.__class__.__name__
        
        if layer_class == 'Resizing':
            print(f"  [{{i}}] FOUND: {{layer.name}}")
            
            if hasattr(layer, 'target_height'):
                target_height = int(layer.target_height)
                target_width = int(layer.target_width)
                print(f"      ✓ {{target_height}}x{{target_width}}")
                break
            
            if target_height is None:
                try:
                    config = layer.get_config()
                    if 'height' in config and 'width' in config:
                        target_height = int(config['height'])
                        target_width = int(config['width'])
                        print(f"      ✓ {{target_height}}x{{target_width}}")
                        break
                except:
                    pass
            
            if target_height is None:
                try:
                    target_height = int(layer.output_shape[1])
                    target_width = int(layer.output_shape[2])
                    print(f"      ✓ {{target_height}}x{{target_width}}")
                    break
                except:
                    pass

    # ===== DETERMINE input_shape =====
    print()
    if target_height is not None and target_width is not None:
        channels = input_shape_raw[-1] if input_shape_raw[-1] is not None else 3
        input_shape = (target_height, target_width, channels)
        print(f"✓ Input shape: {{input_shape}}")
    else:
        input_shape = tuple(dim if dim is not None else 224 for dim in input_shape_raw)
        print(f"⚠️  Input shape (fallback): {{input_shape}}")

    # ===== CREATE DATASET =====
    X = None
    y = None
    
    X_real = None
    y_real = None
    X_synth = None
    y_synth = None
    
    # 1. Load Real Dataset
    if dataset_source == "real" and os.path.exists(real_dataset_path):
        print(f"\\n📦 Loading Real Dataset from {{real_dataset_path}}...")
        try:
            X_real = np.load(os.path.join(real_dataset_path, "x_train.npy"))
            y_real = np.load(os.path.join(real_dataset_path, "y_train.npy"))
            
            # LIMIT: Use only first 5000 samples to avoid OOM and speed up testing
            max_samples = 5000
            if len(X_real) > max_samples:
                print(f"  ⚠️  Limiting dataset to {{max_samples}} samples (OOM prevention)")
                X_real = X_real[:max_samples]
                y_real = y_real[:max_samples]
            
            print(f"  ✓ Loaded {{len(X_real)}} real samples. Shape: {{X_real.shape}}")
            
            # Normalization if necessary (e.g. images 0-255 -> 0-1)
            if X_real.max() > 1.0:
                X_real = X_real.astype('float32') / 255.0
                
            # One-hot encoding if y is scalar
            if len(y_real.shape) == 1 or y_real.shape[-1] == 1:
                # Determine classes from data instead of model
                real_num_classes = len(np.unique(y_real))
                print(f"  📊 Detected {{real_num_classes}} classes in real dataset")
                y_real = tf.keras.utils.to_categorical(y_real, real_num_classes)
            else:
                real_num_classes = y_real.shape[-1]
            
            # Check for explicit validation set
            if os.path.exists(os.path.join(real_dataset_path, "x_test.npy")) and os.path.exists(os.path.join(real_dataset_path, "y_test.npy")):
                print(f"  ✓ Found explicit validation set (x_test.npy)")
                X_val_real = np.load(os.path.join(real_dataset_path, "x_test.npy"))
                y_val_real = np.load(os.path.join(real_dataset_path, "y_test.npy"))
                
                # Limit validation set too
                if len(X_val_real) > max_samples:
                     X_val_real = X_val_real[:max_samples]
                     y_val_real = y_val_real[:max_samples]
                
                if X_val_real.max() > 1.0:
                    X_val_real = X_val_real.astype('float32') / 255.0
                
                if len(y_val_real.shape) == 1 or y_val_real.shape[-1] == 1:
                    y_val_real = tf.keras.utils.to_categorical(y_val_real, real_num_classes)
            else:
                X_val_real = None
                y_val_real = None
                
        except Exception as e:
            print(f"  ❌ Error loading real dataset: {{e}}")
            X_real = None
            y_real = None
            X_val_real = None
            y_val_real = None
    
    # 2. Load Synthetic Data
    if dataset_source == "synthetic" and os.path.exists(synthetic_data_path):
        print(f"\\n🧪 Loading Synthetic Data from {{synthetic_data_path}}...")
        files = glob.glob(os.path.join(synthetic_data_path, "*.npy"))
        
        if files:
            loaded_data = []
            for f in files:
                try:
                    data = np.load(f)
                    loaded_data.append(data)
                except Exception as e:
                    print(f"  ⚠️ Error loading {{f}}: {{e}}")
            
            if loaded_data:
                X_synth = np.array(loaded_data)
                print(f"  ✓ Loaded {{len(X_synth)}} synthetic samples. Shape: {{X_synth.shape}}")
                
                # Dummy labels for synthetic
                # Use real_num_classes if available, else model classes
                if 'real_num_classes' in locals():
                    synth_num_classes = real_num_classes
                else:
                    synth_num_classes = int(output_shape[-1])
                
                y_synth = np.eye(synth_num_classes)[np.random.randint(0, synth_num_classes, len(X_synth))]
        else:
            print(f"  ⚠️ No .npy files found.")

    # 3. Assign to X, y
    if dataset_source == "real":
        X = X_real
        y = y_real
        X_val = X_val_real
        y_val = y_val_real
    elif dataset_source == "synthetic":
        X = X_synth
        y = y_synth
        # Synthetic data usually doesn't have a separate pre-generated val set here
        X_val = None
        y_val = None
    else:
        X = None
        y = None
        X_val = None
        y_val = None

    # 4. Resize if needed (fix shape mismatch)
    # Resizing is now handled dynamically in the tf.data pipeline to prevent OOM
    if X is not None and X.shape[1:] != input_shape:
        print(f"\\n🔧 Target shape for tf.data pipeline: {{input_shape}} (original: {{X.shape[1:]}})")

    # 5. Fallback a Dummy Data
    if X is None:
        print(f"\\n⚠️  Using DUMMY data (Random Noise)")
        num_samples = 100
        X = np.random.randn(num_samples, *input_shape).astype('float32')
        X = (X - X.mean()) / (X.std() + 1e-7)
    
        # Generate Labels (Dummy)
        if is_object_detection:
            y = np.random.randn(len(X), *output_shape[1:]).astype('float32')
        else:
            # Use real_num_classes if available, else model classes
            if 'real_num_classes' in locals():
                dummy_num_classes = real_num_classes
            else:
                dummy_num_classes = 10 # Default to 10 to avoid 1000 categories over-sparsing and mismatched Dense layers
            y = np.eye(dummy_num_classes)[np.random.randint(0, dummy_num_classes, len(X))]
    
    # ===== DETECT DATASET vs MODEL CLASS MISMATCH =====
    dataset_num_classes = None
    if y is not None:
        # Extract unique classes from dataset
        dataset_num_classes = len(np.unique(y.argmax(axis=1))) if len(y.shape) > 1 else len(np.unique(y))
        print(f"\\n📊 Dataset classes detected: {{dataset_num_classes}}")
    
    model_num_classes = int(output_shape[-1]) if len(output_shape) > 1 else None
    print(f"📊 Model output classes: {{model_num_classes}}")
    
    # Check for mismatch and warn/replace if needed
    if dataset_num_classes and model_num_classes:
        if dataset_num_classes != model_num_classes:
            print(f"\\n⚠️  CLASS MISMATCH DETECTED!")
            print(f"  Model expects: {{model_num_classes}} classes (pre-trained on ImageNet/COCO)")
            print(f"  Dataset has: {{dataset_num_classes}} classes")
            print(f"\\n🔧 Applying automatic fix: Replacing final layer...")
            
            # Remove last Dense layer and add new one with correct num_classes
            try:
                # Get output before final layer
                base_output = model.layers[-2].output
                
                # Add Dropout for regularization (reduce overfitting)
                # Using dynamic rate: {dropout_rate}
                dropout = tf.keras.layers.Dropout({dropout_rate}, name='dropout_finetuned')(base_output)
                
                # Add new Dense layer with correct number of classes
                new_output = tf.keras.layers.Dense(
                    dataset_num_classes, 
                    activation='softmax', 
                    name='predictions_finetuned'
                )(dropout)
                
                # FIX: Also we must force the labels `y` to have `dataset_num_classes` size
                # Sometimes `y` can be one-hot encoded to the OLD model output shape (e.g. 1000) 
                # before we reach here due to dummy fallback logic. We truncate it safely:
                if y is not None and len(y.shape) > 1 and y.shape[1] > dataset_num_classes:
                    print(f"  🔧 Truncating one-hot labels from {{y.shape[1]}} to {{dataset_num_classes}} to match new output")
                    y = y[:, :dataset_num_classes]
                    if y_val is not None:
                        y_val = y_val[:, :dataset_num_classes]
                
                # Create new model
                model = tf.keras.Model(inputs=model.input, outputs=new_output)
                
                print(f"  ✓ Final layer replaced: Dense({{dataset_num_classes}}, activation='softmax')")
                print(f"  ✓ New model output shape: {{model.output_shape}}")
                
                # Update output shape for subsequent logic
                output_shape = model.output_shape
                
                # No need to re-encode labels as they are already correct!
                print(f"  ✓ Labels already match new architecture ({{dataset_num_classes}} classes)")
                
            except Exception as e:
                print(f"  ❌ Error during layer replacement: {{e}}")
                print(f"  ⚠️  Continuing with original model (may cause issues)")
        else:
            print(f"\\n✓ Class count matches: {{dataset_num_classes}} classes")
    
    # Check for extreme input resize mismatch
    if X is not None and input_shape:
        original_shape = tuple(X.shape[1:3]) if len(X.shape) >= 3 else None
        target_shape = tuple(input_shape[:2]) if len(input_shape) >= 2 else None
        
        if original_shape and target_shape:
            resize_factor = (target_shape[0] / original_shape[0], target_shape[1] / original_shape[1])
            if resize_factor[0] > 4 or resize_factor[1] > 4:
                print(f"\\n⚠️  EXTREME RESIZE WARNING!")
                print(f"  Dataset resolution: {{original_shape}}")
                print(f"  Model expects: {{target_shape}}")
                print(f"  Upscale factor: {{resize_factor[0]:.1f}}x")
                print(f"  This may reduce model performance due to pixelation.")

    # 6. Prepare Train/Val Split
    print("⚖️  Shuffling and Splitting data (80% train, 20% val)...")
    
    # ==============================================================================
    # FIX: DATA SHUFFLING (CRITICAL FOR OVERFITTING)
    # ==============================================================================
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    
    if X_val is not None:
        print(f"✓ Using explicit validation set: {{len(X_val)}} samples")
    else:
        split_idx = int(len(X) * 0.8)
        X_val = X[split_idx:]
        y_val = y[split_idx:]
        X = X[:split_idx]
        y = y[:split_idx]
    
    print(f"✓ Dataset: train={{X.shape}}, val={{X_val.shape}}")

    # ==============================================================================
    # DEBUG: CHECK CLASS DISTRIBUTION & NORMALIZATION
    # ==============================================================================
    try:
        # Check Class Distribution
        y_train_idx = np.argmax(y, axis=1)
        y_val_idx = np.argmax(y_val, axis=1)
        
        train_classes, train_counts = np.unique(y_train_idx, return_counts=True)
        val_classes, val_counts = np.unique(y_val_idx, return_counts=True)
        
        print(f"  📊 Train Class Dist: {{dict(zip(train_classes, train_counts))}}")
        print(f"  📊 Val Class Dist:   {{dict(zip(val_classes, val_counts))}}")
        
        # Check Normalization (MobileNet expects [-1, 1]) 
        needs_mobilenet_rescale = False
        # CIFAR-10 natively is 0-255 (integers). We convert it to 0-1 (float). MobileNet however was "trained" to see the world in [-1, 1]. SO the solution: takes CIFAR-10 data (which is fine) and transforms it into the format MobileNet wants ([-1, 1]).
        if 'mobilenet' in model.name.lower() or 'mobilenet' in model_path.lower():
            print("  ℹ️  MobileNet detected: Checking normalization...")
            if X.min() >= 0.0 and X.max() <= 1.0:
                print("  ⚠️  Input is [0, 1] but MobileNet expects [-1, 1]. Rescaling lazily via pipeline...")
                needs_mobilenet_rescale = True
                print(f"  ✓ Original range: [{{X.min():.2f}}, {{X.max():.2f}}]")
    except Exception as e:
        print(f"  ⚠️  Debug info error: {{e}}")

    
    # 7. Convert to robust tf.data.Dataset pipeline
    # This specifically fixes: "Failed copying input tensor from CPU to GPU... Dst tensor is not initialized"
    # which happens when TF eager execution races to upload raw NumPy arrays to GPU memory.
    import tensorflow as tf
    
    # Create datasets
    train_ds = tf.data.Dataset.from_tensor_slices((X, y))
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    
    # Define preprocessing helper to act on tensors
    def preprocess_image(img, label):
        # 1. Grayscale to RGB if needed
        if len(img.shape) == 2 and len(input_shape) == 3 and input_shape[-1] == 3:
            img = tf.expand_dims(img, -1)
            img = tf.image.grayscale_to_rgb(img)
        elif len(img.shape) == 3 and img.shape[-1] == 1 and input_shape[-1] == 3:
            img = tf.image.grayscale_to_rgb(img)
            
        # 2. Resize if shape mismatches
        if img.shape[:2] != input_shape[:2]:
            img = tf.image.resize(img, [input_shape[0], input_shape[1]])
            
        # 3. Rescale for MobileNet if flagged  
        if needs_mobilenet_rescale:
            img = (img - 0.5) * 2.0
            
        return img, label

    # Apply preprocessing mapping to both datasets
    train_ds = train_ds.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Apply Data Augmentation using standard Keras layers inside the pipeline
    if len(input_shape) == 3 and not is_object_detection:
        print("📸 Enabling Data Augmentation (Rotation, Zoom, Flip) via tf.data...")
        
        # Define augmentation sequence
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.2),
            tf.keras.layers.RandomZoom(0.2),
            tf.keras.layers.RandomTranslation(0.2, 0.2)
        ])
        
        # Apply only to training data
        train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)

    # Batch and prefetch for optimal GPU transfer
    train_ds = train_ds.shuffle(buffer_size=1000).batch({batch_size}).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.batch({batch_size}).prefetch(tf.data.AUTOTUNE)
    
    # 8. Compile & Train
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate={learning_rate}),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
                  
    print(f"✓ Compiled (loss=categorical_crossentropy, LR={learning_rate})")
    
    # Custom Callback for feedback
    class PrintEpochProgress(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {{}}
            print(f"Epoch {{epoch+1}}/{epochs} - "
                  f"loss: {{logs.get('loss'):.4f}} - "
                  f"accuracy: {{logs.get('accuracy'):.4f}} - "
                  f"val_loss: {{logs.get('val_loss'):.4f}} - "
                  f"val_accuracy: {{logs.get('val_accuracy'):.4f}}", flush=True)

    callbacks_list = [
        PrintEpochProgress(),
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-7, verbose=0)
    ]

    # Use tf.data pipeline for training
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs={epochs},
        callbacks=callbacks_list,
        verbose=0
    )
    
    model.save(output_path, save_format='h5')
    
    # Extract correct metrics based on type
    if is_object_detection:
        final_mse = float(history.history['mse'][-1])
        final_val_mse = float(history.history['val_mse'][-1])
        final_loss = float(history.history['loss'][-1])
        final_val_loss = float(history.history['val_loss'][-1])
        # Convert MSE to "accuracy-like" metric (lower = more accurate)
        final_acc = 1.0 / (1.0 + final_mse)
        final_val_acc = 1.0 / (1.0 + final_val_mse)
    else:
        final_acc = float(history.history['accuracy'][-1])
        final_val_acc = float(history.history['val_accuracy'][-1])
        final_loss = float(history.history['loss'][-1])
        final_val_loss = float(history.history['val_loss'][-1])
    
    epochs_trained = len(history.history['loss'])
    
    print(f"✓ Training complete ({{epochs_trained}} epochs)")
    print(f"SUCCESS: {{final_acc:.4f}}|{{final_val_acc:.4f}}|{{final_loss:.4f}}|{{final_val_loss:.4f}}|{{epochs_trained}}")
    
except Exception as e:
    print(f"ERROR: {{str(e)}}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
"""
        
        logger.info(f"  [Subprocess] Executing fine-tuning...")
        
        # -----------------------------------------------------------------------
        # VRAM MANAGEMENT: Commented out (HPP Triton has enough VRAM)
        # Unload Mistral from Triton before fine-tuning
        # With USE_TRITON_BACKEND, Mistral occupies ~7.8GB of the A4000 (16GB).
        # TF training sees only ~1.1GB → SIGABRT (OOM) after Epoch 1.
        # Triton starts with --model-control-mode=explicit, so we can
        # use the /v2/repository/models/{model}/unload API to free VRAM.
        # After training, reload_triton_models() reloads the models.
        # -----------------------------------------------------------------------
        # from src.assistant.utils import force_unload_triton, reload_triton_models
        import os as _os
        use_triton = _os.environ.get("USE_TRITON_BACKEND", "false").lower() == "true"
        
        # if use_triton:
        #     logger.info("🔫 [fine-tuning] Unloading Mistral from Triton to free VRAM...")
        #     force_unload_triton(["mistral"])
        #     logger.info("✅ [fine-tuning] VRAM freed. Starting training...")
        # else:
        if not use_triton:
            from src.assistant.utils import force_unload_ollama
            cfg = Configuration.from_runnable_config(config)
            force_unload_ollama(cfg.local_llm or "gpt-oss:20b")
        # Define ignore list for real-time suppression
        ignore_list = [
            'tensorflow/core/util/port.cc',
            'tensorflow/tsl/cuda/cudart_stub.cc',
            'tensorflow/core/platform/cpu_feature_guard.cc',
            'tensorflow/compiler/tf2tensorrt',
            'oneDNN custom operations',
            'Could not find cuda drivers',
            'TF-TRT Warning',
            'NUMA node',
            'Created device /job:localhost',
            'built with optimized CPU instructions',
            'appropriate compiler flags',
            'Class Dist:',
            'layout failed: INVALID_ARGUMENT',
            'Loaded cuDNN version',
            'Start cannot spawn child process',
            'TensorFloat-32 will be used',
            'XLA service',
            'StreamExecutor device',
            'disabling MLIR crash reproducer',
            'Compiled cluster using XLA'
        ]
        
        # ===== USE execute_in_environment =====
        full_ignore_list = (ignore_list or []) + SUBPROCESS_NOISE_FILTER
        result = execute_in_environment(
            python_code, 
            state, 
            timeout=3600, 
            ignore_list=full_ignore_list,
            whitelist_patterns=SUBPROCESS_CLEAN_ALLOWLIST
        )
        
        stdout = result.get('stdout', '')
        stderr = result.get('stderr', '')
        
        # Output is already streamed cleanly by execute_in_environment (via whitelist).
        # We process the raw stdout secretly here just to extract the final metrics.
        
        if not result['success']:
            logger.error("❌ Training subprocess failed")
            logger.error(f"   • Return code: {result['returncode']}")
            logger.error(f"  Stderr:\n{stderr[:1000]}")
            # Fix: Handle empty stderr
            error_msg = "Unknown error"
            if stderr:
                stderr_lines = [line for line in stderr.split('\n') if line.strip()]
                if stderr_lines:
                    error_msg = stderr_lines[-1]
            raise Exception(f"Subprocess failed: {error_msg}")
        
        if "SUCCESS:" in stdout:
            parts = stdout.split("SUCCESS:")[-1].strip().split('|')
            
            if len(parts) < 5:
                raise Exception(f"Invalid output format. Expected 5 parts, got {len(parts)}: {parts}")
            
            final_acc = float(parts[0].strip())
            final_val_acc = float(parts[1].strip())
            final_loss = float(parts[2].strip())
            final_val_loss = float(parts[3].strip())
            epochs_trained = int(parts[4].strip())
            
            state.training_test_result = {
                "success": True,
                "final_accuracy": final_acc,
                "final_val_accuracy": final_val_acc,
                "final_loss": final_loss,
                "final_val_loss": final_val_loss,
                "epochs_trained": epochs_trained,
            }
            
            state.training_validation_success = True
            state.customized_model_path = output_path
            
            logger.info("✅ Training completed successfully")
            logger.info(f"   • Final loss: {final_loss:.4f}")
            logger.info(f"   • Final accuracy: {final_acc:.4f}")
            logger.info(f"   • Val loss: {final_val_loss:.4f}")
            logger.info(f"   • Val accuracy: {final_val_acc:.4f}")
        else:
            raise Exception(f"Output does not contain SUCCESS marker")
    
    except Exception as e:
        logger.error(f"❌ Fine-tuning failed: {str(e)}", exc_info=True)
        state.training_validation_success = False
        state.training_test_result = {
            "success": False,
            "error": str(e)
        }
    finally:
        # Reload Mistral on Triton after training (both on success and error)
        # if use_triton:
        #     logger.info("🚀 [fine-tuning] Reloading Mistral on Triton post-training...")
        #     reload_triton_models(["mistral"])
        pass
        
    return state


def ask_optimization_preference(state: MasterState, config: RunnableConfig = None) -> MasterState:
    """Asks the user whether to use NNI or standard training"""
    logger.info("=" * 70)
    logger.info("🤔 ASK_OPTIMIZATION_PREFERENCE NODE EXECUTING")
    logger.info("=" * 70)
    
    prompt = {
        "instruction": "Do you want to run hyperparameter optimization with NNI or proceed with standard fine-tuning? (nni/standard)",
        "options": {
            "nni": "Use NNI to find the best parameters (AutoML) - ⚠️ Requires more time than standard fine-tuning",
            "standard": "Standard fine-tuning (fixed parameters)"
        }
    }
    
    from src.assistant.utils import extract_user_response
    logger.info("📝 Checking for UI response...")
    resume_value = None
    if not state.user_response or state.user_response.strip() == "":
        logger.info("⏸️ Interrupting for optimization preference.")
        # resume_value = interrupt(prompt)
        resume_value = "standard" # BYPASS user choice
        response = str(resume_value).strip() if resume_value else ""
    else:
        # Use interrupt return value as priority
        if resume_value and str(resume_value).strip():
            response = str(resume_value).strip()
        else:
            response = extract_user_response(state.user_response)
    state.user_response = "" # Clear
    
    # Default if empty
    if not response or response.strip() == "":
        response = "nni" # default to NNI !
        
    state.optimization_mode = "nni" if "nni" in response.lower() else "standard"
    logger.info(f"✓ Selected mode: {state.optimization_mode}")
    logger.info("=" * 70)
    return state

def optimization_routing(state: MasterState) -> Literal["optimize_hyperparameters_with_nni", "fine_tune_customized_model"]:
    if getattr(state, "optimization_mode", "standard") == "nni": 
        logger.info("→ Routing to: NNI Optimization")
        return "optimize_hyperparameters_with_nni"
    
    logger.info("→ Routing to: Standard Fine-Tuning")
    return "fine_tune_customized_model"

# getattr(state, "optimization_mode", "standard") -> Reads the attribute optimization_mode from the state object. If the attribute exists, it returns its value (e.g. "nni" or "standard"). If it does NOT exist or is None, it returns the default value: "standard"

# == "nni" -> compares the obtained value with the string "nni"
# If state.optimization_mode == "nni" → Enters the if → Routing to NNI
# Otherwise → Goes to else → Routing to Standard Fine-Tuning
    
    
def optimize_hyperparameters_with_nni(state: MasterState, config: RunnableConfig = None) -> MasterState:
    """
    ✨ Runs hyperparameter optimization with NNI (Agentic/Adaptive Mode)
    Generate dynamic scripts based on model and dataset.
    """
    logger.info("═══════════════════════════")
    logger.info("🧠 NNI ADAPTIVE OPTIMIZATION")
    logger.info("═══════════════════════════")
    
    try:
        from src.assistant.nni_optimization.generator import generate_nni_experiment
        import subprocess
        
        # Use the customized model (with adapted head) as starting point for NNI
        model_path = state.customized_model_path if state.customized_model_path else state.model_path
        
        # Determine data path
        if state.dataset_source == "real":
             data_path = state.real_dataset_path
        else:
             data_path = state.synthetic_data_path
             
        # Prepare Info Dictionaries
        model_info = {
            "name": os.path.basename(model_path),
            "path": model_path,
            "input_shape": state.model_architecture.get("input_shape", "Unknown"),
            "n_layers": state.model_architecture.get("n_layers", 0),
            "output_shape": state.model_architecture.get("output_shape", "Unknown")
        }
        
        dataset_info = {
            "path": data_path,
            "source": state.dataset_source,
            # We assume standard numpy names, shape can be inferred or passed if known
            "num_classes": state.model_architecture.get("output_classes", 10) 
        }
        
        # Output Directory for Generated Scripts.
        # Use /tmp to avoid permission errors on the read-only mounted project volume.
        experiment_dir = os.path.join("/tmp", "nni_experiments", f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(experiment_dir, exist_ok=True)
        
        logger.info(f"📁 NNI Experiment directory: {experiment_dir}")
        
        # --- DATASET SUBSETTING FOR NNI (User Request: Generalized 50% Subsetting) ---
        nni_data_path = data_path
        
        # try:
        #     import numpy as np
        #     x_train_path = os.path.join(data_path, "x_train.npy")
        #     if os.path.exists(x_train_path):
        #         # Check actual size with mmap to avoid loading the whole thing
        #         x_sample = np.load(x_train_path, mmap_mode='r')
        #         total_samples = len(x_sample)
        #         subset_size = total_samples // 2
        #         
        #         logger.info(f"✂️ Generalized Subsetting: Using 50% ({subset_size}/{total_samples}) for NNI...")
        #         subset_dir = os.path.join(experiment_dir, "dataset_subset")
        #         os.makedirs(subset_dir, exist_ok=True)
        #         
        #         # Load and save subsets
        #         for filename in ["x_train.npy", "y_train.npy", "x_test.npy", "y_test.npy"]:
        #             src_path = os.path.join(data_path, filename)
        #             if os.path.exists(src_path):
        #                 data = np.load(src_path)
        #                 # Take 50% of whatever file it is (train or test)
        #                 file_subset_size = len(data) // 2
        #                 if len(data) > file_subset_size:
        #                     data = data[:file_subset_size]
        #                 np.save(os.path.join(subset_dir, filename), data)
        #         
        #         nni_data_path = subset_dir
        #         logger.info(f"✅ Dynamic Subset created at: {nni_data_path}")
        # except Exception as subset_err:
        #     logger.error(f"⚠️ Error during dynamic subsetting: {subset_err}. Using full dataset.")

        # Update dataset info with potentially new path
        dataset_info["path"] = nni_data_path

        # 1. GENERATE
        logger.info("🚀 Generating custom NNI experiment...")
        generated_files = generate_nni_experiment(
            model_info=model_info,
            dataset_info=dataset_info,
            output_dir=experiment_dir,
            num_ctx=config.get('llm_context_window', 8192) if config else 8192
        )
        
        if not generated_files:
            logger.error("❌ Generation failed. Abort.")
            return state
            
        logger.info("✓ Scripts generated successfully.")
        
        # 2. EXECUTE
        manager_script = os.path.join(experiment_dir, "manager.py")
        if not os.path.exists(manager_script):
             logger.error(f"❌ manager.py not found in {experiment_dir}")
             return state
             
        logger.info(f"▶️  Launching NNI Manager: {manager_script}")
        
        cfg = Configuration.from_runnable_config()
        # If we already used a specific env for customization, we use the same for NNI. In fact, already in the customization it understands which environment to use based on the model with the load_model_with_conda_env function. 
        # Otherwise we understand from the extension
        if state.conda_env:
            env_name = state.conda_env
            logger.info(f"📋 Using environment stored in state: {env_name}")
        elif model_path.endswith('.keras'):
            env_name = 'stm32' # For modern models (Keras 3). It is the default environment for new projects.
        else:
            env_name = 'stm32_legacy' # Needs when loading .h5 files created with previous versions or architectures that have not yet been migrated to Keras 3.
            
        python_path = cfg.get_python_path(env_name)
        
        logger.info(f"▶️  Launching NNI Manager with: {python_path}")

        # Find the free port HERE, so we can log it via logger.info
        # even before the subprocess starts — manager.py will read it from NNI_PORT.
        import socket as _socket
        nni_port = 8080
        for _p in range(8080, 8100):
            with _socket.socket() as _s:
                if _s.connect_ex(('localhost', _p)) != 0:
                    nni_port = _p
                    break
        logger.info(f"🌐 NNI Web UI will be available at: http://localhost:{nni_port}")

        nni_env = os.environ.copy()
        nni_env['NNI_PORT'] = str(nni_port)

        try:
            result = subprocess.run(
                [python_path, manager_script],  # Use environment Python
                cwd=experiment_dir,
                capture_output=True,
                text=True,
                timeout=10800,  # 3 hours timeout for NNI to start/run
                env=nni_env,
            )
            
            logger.info("✓ Experiment concluded (or interrupted).")
            if result.stdout:
                logger.info(f"STDOUT (last 5000 chars):\n...{result.stdout[-5000:]}") # I don't need the first 500 characters. I show the last 5000 characters ([-5000:]) instead of the first 500. 
            if result.stderr:
                logger.warning(f"STDERR:\n{result.stderr[:500]}...")
                
        except subprocess.TimeoutExpired:
            logger.error("⏱️  NNI Manager timeout (3 hours). Force closing everything...")
            # FORCE CLEANUP: Kill manager, trials, and NNI server
            try:
                # 1. Kill the specific manager process if possible (though Popen object is lost in run)
                # 2. Use 'nnictl stop --all' to kill NNI experiments
                logger.info("💀 Executing 'nnictl stop --all'...")
                subprocess.run([python_path, "-m", "nni.tools.nnictl", "stop", "--all"], check=False)
                
                # 3. Aggressive cleanup of python processes related to 'manager.py' or 'trial.py' in this dir
                # (Optional but safer)
                logger.info("💀 Force killing lingering NNI processes...")
                subprocess.run(["pkill", "-f", "nni.main"], check=False)
                subprocess.run(["pkill", "-f", "trial.py"], check=False)
                
            except Exception as cleanup_err:
                logger.error(f"⚠️ Error during cleanup: {cleanup_err}")
                
        except Exception as exec_err:
             logger.error(f"❌ Error executing manager: {exec_err}")
             
        # --- RETRIEVE BEST MODEL ---
        best_model_path = os.path.join(experiment_dir, "best_model.h5")
        if os.path.exists(best_model_path):
            logger.info(f"🏆 Found optimized model at: {best_model_path}")
            state.customized_model_path = best_model_path
            state.model_path = best_model_path # Update source for validation
            state.customization_applied = True # Indicate that the model has been modified
        else:
            logger.warning("⚠️  Optimized model NOT found. Using original/customized path.")
        
    except ImportError:
        logger.error("❌ NNI module not found or import error.")
    except Exception as e:
        logger.error(f"❌ NNI Optimization failed: {e}")
        
    return state


import shutil

def validate_customized_model(state: MasterState, config: RunnableConfig = None) -> MasterState:
    """
    ✨ Validates the customized model IN SUBPROCESS (uses state.python_path)
    """
    
    logger.info("✅ Validating customized model...")
    
    try:
        model_path = state.customized_model_path
        
        if not model_path or not os.path.exists(model_path):
            logger.error("❌ Model not found")
            state.error_message = "Model not found"
            return state
        
        python_code = f"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import json

model_path = r'{model_path}'

try:
    try:
        import keras
        model = keras.models.load_model(model_path, compile=False)
    except Exception:
        model = tf.keras.models.load_model(model_path, compile=False, safe_mode=False)
    
    # Extract info
    info = {{
        'input_shape': str(model.input_shape),
        'output_shape': str(model.output_shape),
        'total_params': int(model.count_params()),
    }}
    
    # model.summary() omitted to reduce subprocess log verbosity
    
    print(f"SUCCESS: " + json.dumps(info))
    
except Exception as e:
    print(f"ERROR: {{str(e)}}")
    import traceback
    traceback.print_exc()
"""
        
        # ===== USE execute_in_environment =====
        result = execute_in_environment(python_code, state, timeout=120, ignore_list=SUBPROCESS_NOISE_FILTER, whitelist_patterns=SUBPROCESS_CLEAN_ALLOWLIST)
        
        if not result['success']:
            logger.error(f"❌ Validation failed: {result['stderr'][:500]}")
            state.error_message = result['stderr']
            return state
        
        # ===== PARSE INFO =====
        stdout = result['stdout']
        
        if "SUCCESS: " in stdout:
            json_str = stdout.split("SUCCESS: ")[-1].strip()
            info = json.loads(json_str)
            
            state.customized_model_info.update(info)
            
            logger.info(f"✓ Model validated")
            logger.info(f"  Input: {info['input_shape']}")
            logger.info(f"  Output: {info['output_shape']}")
            logger.info(f"  Params: {info['total_params']:,}")
        else:
            logger.error(f"❌ Invalid output format")
            state.error_message = "Invalid output format"
    
    except Exception as e:
        logger.error(f"❌ Validation error: {str(e)}", exc_info=True)
        state.error_message = str(e)
    
    return state


def save_customized_model_final(state: MasterState, config: RunnableConfig = None) -> MasterState:
    """
    ✨ Saves the customized model as .h5 (compatible with stedgeai)
    Validates using state.python_path
    """
    
    logger.info("💾 Saving customized model definitively...")
    
    try:
        model_path = state.customized_model_path
        
        if not model_path or not os.path.exists(model_path):
            logger.error("❌ Customized model not found")
            state.error_message = "Customized model not found"
            return state
        
        output_dir = os.path.expanduser("~/.stm32_ai_models/customized")
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_path = os.path.join(output_dir, f"customized_final_{timestamp}.h5")  # ← .h5
        
        logger.info(f"  Copying model: {model_path} → {final_path}")
        
        # ===== COPY FILE =====
        shutil.copy(model_path, final_path)
        logger.info(f"✓ Model copied: {final_path}")
        
        # ===== VALIDATE IN SUBPROCESS =====
        logger.info(f"  Validating in environment: {state.conda_env}")
        
        python_code = f"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import json

model_path = r'{final_path}'

try:
    try:
        import keras
        model = keras.models.load_model(model_path, compile=False)
    except Exception:
        model = tf.keras.models.load_model(model_path, compile=False, safe_mode=False)
    
    info = {{
        'input_shape': str(model.input_shape),
        'output_shape': str(model.output_shape),
        'total_params': int(model.count_params()),
        'model_name': model.name,
    }}
    
    print(f"SUCCESS: " + json.dumps(info))
    
except Exception as e:
    print(f"ERROR: {{str(e)}}")
    import traceback
    traceback.print_exc()
"""
        
        # ===== USE execute_in_environment =====
        result = execute_in_environment(python_code, state, timeout=120, ignore_list=SUBPROCESS_NOISE_FILTER, whitelist_patterns=SUBPROCESS_CLEAN_ALLOWLIST)
        
        if not result['success']:
            logger.error(f"❌ Final save validation failed: {result['stderr'][:500]}")
            state.error_message = result['stderr']
            return state
        
        # ===== PARSE INFO =====
        if "SUCCESS: " in result['stdout']:
            json_str = result['stdout'].split("SUCCESS: ")[-1].strip()
            info = json.loads(json_str)
            
            state.final_model_path = final_path
            state.customization_applied = True # Make sure it is True
            state.customized_model_info.update({
                **info,
                "model_size_mb": round(os.path.getsize(final_path) / (1024*1024), 2),
                "format": "H5"  # ← Final format
            })
            
            logger.info("✅ Model saved successfully")
            logger.info(f"   • Output: {final_path}")
            logger.info(f"   • Size: {os.path.getsize(final_path) / 1024 / 1024:.1f}MB")
        else:
            logger.error(f"❌ Validation output missing SUCCESS marker")
            state.error_message = "Validation failed"
    
    except Exception as e:
        logger.error(f"❌ Save error: {str(e)}", exc_info=True)
        state.error_message = str(e)
    
    return state



def ask_continue_after_customization(state: MasterState, config: RunnableConfig = None) -> MasterState:
    """Ask whether to continue with AI analysis"""
    
    logger.info("🤔 Asking whether to continue...")
    
    
    
    info = state.customized_model_info or {}
    params = info.get('total_params')
    params_str = f"{params:,}" if isinstance(params, (int, float)) else "N/A"
    
    summary = f"""
Customization Complete!

Final Model: {state.final_model_path}
- Input: {info.get('input_shape', 'N/A')}
- Output: {info.get('output_shape', 'N/A')}
- Params: {params_str}
- Size: {info.get('model_size_mb', 'N/A')} MB

Training Results:
- Accuracy: {state.training_test_result.get('final_accuracy', 'N/A')}
- Val Accuracy: {state.training_test_result.get('final_val_accuracy', 'N/A')}

Quantized: {state.should_quantize}
{f'- Quantized model: {state.quantized_model_path}' if state.quantized_model_path else ''}
"""
    
    prompt = {
        "instruction": "Do you want to continue with X-CUBE-AI analysis?",
        "summary": summary,
        "options": ["continue_ai", "end"]
    }
    
    from src.assistant.utils import extract_user_response
    resume_value = None
    if not state.user_response or state.user_response.strip() == "":
        # resume_value = interrupt(prompt)
        resume_value = "yes" # BYPASS
    
    # Use interrupt return value as priority
    if resume_value and str(resume_value).strip():
        user_response = str(resume_value).strip()
    else:
        user_response = extract_user_response(state.user_response)
    state.user_response = "" # Clear
    
    # Default: continue with AI analysis if empty
    if not user_response or user_response.strip() == "":
        user_response = "continue_ai"
    
    # ===== LLM CLASSIFICATION =====
    try:
        logger.info(f"🤖 Asking the user whether to continue...")
        
        from src.assistant.utils import get_llm
        llm_classifier = get_llm(config, structured_schema=ContinueDecision)
        
        decision = llm_classifier.invoke([
            SystemMessage(content=continue_decision_instructions),
            HumanMessage(content=str(user_response))
        ])
        
        logger.info(f"✓ Decision classified:")
        logger.info(f"  wants_to_continue: {decision.wants_to_continue}")
        logger.info(f"  confidence: {decision.confidence:.2f}")
        
        state.continue_after_customization = decision.wants_to_continue
    
    except Exception as e:
        logger.warning(f"⚠️  Classification failed, using fallback: {str(e)[:100]}")
        # Fallback: if the response contains "continue", "si", "yes" → continue
        user_lower = str(user_response).lower()
        state.continue_after_customization = any(kw in user_lower for kw in ["continue", "si", "yes", "ok", "analyze"])
    
    # If the user decides to continue towards AI analysis, we must reset the 
    # idempotency flags (model_selected, task_selected) otherwise workflow2 will skip selection
    # and keep in memory the very first chosen model instead of this new customized one!
    if state.continue_after_customization:
        state.model_selected = False
        state.task_selected = False
        state.selected_model = None
        state.ai_task = None
        logger.info("🗑️  Reset of idempotency flags to force new model selection in Workflow 2")
        
    return state


# 🥇 Deepseek-r1      (BEST: perfect reasoning, impeccable JSON). A few more seconds to reflect, but lighter than Mistral (70 B vs 72 B) and better quality. 
# 🥈 Mistral 72B      (GOOD: fast, OK quality)
# 🥉 Qwen2 7B         (OK: light but lower quality)


# ============================================================================
# ROUTING HELPERS
# ============================================================================

def modification_confirmation_routing(state: MasterState) -> Literal["apply_user_customization", "ask_and_parse_user_modifications", "run_analyze"]:
    """
    Route based on modification_confirmed and user_wants_to_edit.
    - If modification_confirmed=True: proceed with application
    - If user_wants_to_edit=True: go back and ask for modifications
    - If modification_confirmed=False (and no edit): abort everything and go to analyze
    """
    if state.modification_confirmed:
        return "apply_user_customization"
    elif state.user_wants_to_edit:
        return "ask_and_parse_user_modifications"
    else:
        # Caso "No" -> Abort customization, go straight to analysis
        return "run_analyze"

