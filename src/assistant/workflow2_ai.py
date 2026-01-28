# ============================================================================
# WORKFLOW 2: AI ANALYSIS CON MODEL DISCOVERY E CUSTOMIZATION
# ============================================================================
# Modulo dedicato all'analisi dei modelli AI e generazione codice STEdgeAI
#
# Responsabilità:
#   - Raccolta configurazione AI (target MCU, compression)
#   - Model discovery (predefiniti, ricerca online, fallback)
#   - Download modelli da GitHub/Google
#   - Model customization (architecture, fine-tuning, quantization)
#   - STEdgeAI analyze/validate/generate
#
# Dipendenze: langgraph, langchain, stedgeai, tensorflow, requests

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

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from src.assistant.configuration import Configuration
from src.assistant.state import MasterState

from agno.tools.googlesearch import GoogleSearchTools
from agno.models.ollama import Ollama
from agno.agent import Agent  


logger = logging.getLogger(__name__)

# ============================================================================
# EXTRACTION SCHEMAS - WORKFLOW 2
# ============================================================================

class AnalysisInfoExtraction(BaseModel):
    """Schema per estrarre target MCU e compression"""
    target: Optional[str] = Field(
        default=None,
        description="Target MCU (es: stm32f401, stm32h743, stm32u5)"
    )
    compression: Optional[str] = Field(
        default=None,
        description="Livello di compressione (low, medium, high, very_high)"
    )


class TaskSelectionExtraction(BaseModel):
    """Estrae la scelta del task da risposta naturale"""
    task: Optional[str] = Field(
        default=None,
        description="Task selezionato (chiave tecnica della categoria)"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidenza della classificazione"
    )


class ModelSelectionExtraction(BaseModel):
    """Estrae la scelta del modello da risposta naturale"""
    model_index: Optional[int] = Field(
        default=None,
        description="Indice del modello selezionato (1-based)"
    )
    model_accepted: bool = Field(
        default=False,
        description="L'utente ha accettato il modello?"
    )
    wants_another_search: bool = Field(
        default=False,
        description="L'utente vuole un'altra ricerca?"
    )


class ModelFeedbackExtraction(BaseModel):
    """Estrae il feedback dell'utente sul modello proposto"""
    model_accepted: bool = Field(
        default=False,
        description="True se l'utente accetta il modello proposto"
    )
    wants_another_search: bool = Field(
        default=False,
        description="True se l'utente vuole un'altra ricerca/ricerca diversa"
    )
    wants_default: bool = Field(
        default=False,
        description="True se l'utente vuole il modello di default/termina ricerca"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidenza della classificazione (0-1)"
    )


class ResolutionExtraction(BaseModel):
    """Estrae la decisione post-fallimento risorse"""
    decision: str = Field(
        description="Azione da intraprendere: change_board o change_model"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidenza della scelta"
    )


class SearchResultExtraction(BaseModel):
    """Estrae modelli AI scaricabili (.h5, .keras, .onnx, .tflite)"""
    model_name: str = Field(description="Nome modello (es: MobileNetV2 128)")
    download_url: Optional[str] = Field(
        default=None,
        description="URL diretto a file .h5, .keras, .onnx o .tflite"
    )
    model_size: Optional[str] = Field(default=None, description="Size (es: 5.7MB)")
    accuracy: Optional[str] = Field(default=None, description="Accuracy (es: 64%)")
    inference_time: Optional[str] = Field(default=None, description="Tempo (es: 40ms (STM32H7))")
    is_valid: bool = Field(
        default=False,
        description="True solo se download_url è presente e non None"
    )


# ============================================================================
# EXTRACTION INSTRUCTIONS - WORKFLOW 2
# ============================================================================

analysis_info_extraction_instructions = """Sei un estrattore di informazioni per la configurazione dell'analisi AI.

Analizza la risposta dell'utente e estrai i seguenti campi:

1. **target**: Target MCU per cui ottimizzare il modello
     Valori comuni: "stm32f401", "stm32f4", "stm32h743", "stm32h7", "stm32u5", "stm32u575"
     → Se non specificato: null

2. **compression**: Livello di compressione per il modello
     Valori comuni: "low", "medium", "high", "very_high"
     → Se non specificato: null

Esempi:
- Input: "STM32H743 con compressione media"
  Output: {"target": "stm32h743", "compression": "medium"}

- Input: "F4, compressione alta"
  Output: {"target": "stm32f4", "compression": "high"}

Rispondi SEMPRE in formato JSON valido.
"""

# Istruzioni estratte dinamicamente

model_selection_instructions = """Analizza la risposta dell'utente sulla selezione del modello.

L'utente risponde a: "Quale modello vuoi usare? (1-N oppure 'no'/'ricerca')"

Esempi di risposte:
- "1" o "Il primo" → model_index: 1, model_accepted: true
- "2" o "Il secondo modello" → model_index: 2, model_accepted: true
- "no" / "Nessuno" / "Non mi piace" → model_accepted: false, wants_another_search: true
- "Usa il default" / "Default" → model_accepted: false, wants_another_search: false

Rispondi SEMPRE in formato JSON con:
- "model_index": numero intero (1-based) o null se non scelto
- "model_accepted": true se utente accetta, false altrimenti
- "wants_another_search": true se vuole cercare ancora, false se usa default
"""

model_feedback_extraction_instructions = """Analizza il feedback dell'utente sul modello proposto.

Classifica la risposta in una di queste categorie:

1. **model_accepted**: L'utente ACCETTA il modello proposto
   Esempi: "sì", "perfetto", "ok", "va bene", "accetto", "dimmi come scaricarlo"

2. **wants_another_search**: L'utente vuole CERCARE UN ALTRO MODELLO
   Esempi: "no", "non mi piace", "cerchiane un altro", "nope", "troppo grande"

3. **wants_default**: L'utente vuole il MODELLO DI DEFAULT o TERMINA
   Esempi: "default", "basta ricerche", "stop", "predefinito", "termina"

Rispondi SEMPRE in formato JSON con:
- "model_accepted": true/false
- "wants_another_search": true/false
- "wants_default": true/false
- "confidence": 0.0-1.0

IMPORTANTE: Solo UNO dei tre può essere true!
"""

search_result_extraction_instructions = """Estrai SOLO questi 5 campi dal risultato della ricerca:

1. **model_name**: Il nome del modello (es: MobileNetV2 128)
2. **download_url**: L'URL per scaricare il file (.h5, .keras, .onnx, .tflite) (estrarre dalle parentesi tonde se Markdown)
3. **model_size**: La dimensione del file (es: 5.7MB)
4. **accuracy**: L'accuratezza del modello (es: 64%)
5. **inference_time**: Il tempo di inferenza (es: 40ms (STM32H7))

IMPORTANTE: Cerca link che finiscono con .h5, .keras, .onnx o .tflite.
Se vedi [testo](https://...) estrai l'URL dalle parentesi tonde (il secondo)

Rispondi SEMPRE in formato JSON con esattamente questi campi:
{
  "model_name": "string",
  "download_url": "string o null",
  "model_size": "string o null",
  "accuracy": "string o null",
  "inference_time": "string o null",
  "is_valid": true/false
}
"""

# Attenzione: search_result_extraction_instructions è diverso da research_prompt. Serve per estrarre i risultati trovati, non per fare la ricerca!!

# ============================================================================
# PREDEFINED_MODELS - URL REALI (Verificati)
# ============================================================================


def get_resource_path(filename: str) -> str:
    """Restituisce il path assoluto di una risorsa nella cartella resources."""
    # Cerchiamo prima in src/assistant/resources relativo a questo file
    base_dir = os.path.dirname(os.path.abspath(__file__))
    res_path = os.path.join(base_dir, "resources", filename)
    return res_path

def load_predefined_models() -> dict:
    """Carica i modelli predefiniti dal file JSON."""
    path = get_resource_path("predefined_models.json")
    if not os.path.exists(path):
        logger.warning(f"⚠️ Registro modelli non trovato in {path}, ritorno vuoto.")
        return {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"❌ Errore caricamento modelli: {e}")
        return {}

def save_predefined_models(models: dict):
    """Salva i modelli nel file JSON."""
    path = get_resource_path("predefined_models.json")
    try:
        # Assicurati che la cartella esista
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(models, f, indent=4, ensure_ascii=False)
        logger.info(f"✅ Registro modelli aggiornato: {path}")
    except Exception as e:
        logger.error(f"❌ Errore salvataggio modelli: {e}")

# Inizializza PREDEFINED_MODELS dinamicamente (ma caricalo ogni volta se vogliamo essere dinamici al runtime)
PREDEFINED_MODELS = load_predefined_models()
# <- con .h5 e non .tflite


# ============================================================================
# NODI WORKFLOW 2
# ============================================================================
def collect_analysis_info(state: MasterState, config: dict) -> MasterState:
    """
    Raccoglie SOLO target MCU e compression.
    La selezione modello viene gestita nei nodi successivi !
    """
    
    logger.info("📋 Raccolta configurazione analisi AI...")
    
    cfg = Configuration.from_runnable_config(config)
    
    prompt = {
        "instruction": """Configurazione Analisi AI con STEdgeAI

Specifica (brevemente):
1. Target MCU (STM32F4, STM32H7, STM32U5, etc.)
2. Livello compressione: low, medium, high, very_high (opzionale, default: high)

Esempi:
- "STM32H743"
- "F4 con alta compressione"
- "STM32U5 medium"
        """,
    }
    
    # user_response = interrupt(prompt) # per adesso commentata per velocizzare
    user_response = "" # BYPASS
    if isinstance(user_response, dict):
        user_text = user_response.get("response", user_response.get("input", str(user_response)))
    else:
        user_text = str(user_response)
    
    # ✅ FIX: Eredita board context dal firmware workflow se disponibile
    if not user_text or user_text.strip() == "":
        # Se firmware è stato generato, usa quella board
        if state.mcu_series and state.mcu_series.strip():
            # Mappa serie MCU a target string per STEdgeAI
            series_to_target = {
                "F0": "stm32f0",
                "F1": "stm32f1",
                "F2": "stm32f2",
                "F3": "stm32f3",
                "F4": "stm32f4",
                "F7": "stm32f7",
                "H5": "stm32h5",
                "H7": "stm32h7",
                "L0": "stm32l0",
                "L1": "stm32l1",
                "L4": "stm32l4",
                "L5": "stm32l5",
                "U5": "stm32u5",
                "G0": "stm32g0",
                "G4": "stm32g4",
                "W5": "stm32w5",
                "C0": "stm32c0",
                "N6": "stm32n6"
            }
            target_mcu = series_to_target.get(state.mcu_series.upper(), "stm32f4")
            user_text = f"{target_mcu}, medium compression"
            logger.info(f"✓ Ereditato contesto firmware: board_name={state.board_name}, mcu_series={state.mcu_series} → target={target_mcu}")
        else:
            # Fallback se non c'è contesto firmware
            user_text = "STM32F4, medium compression"
            logger.info("ℹ️  Nessun contesto firmware, uso default STM32F4")
    
    logger.info(f"📝 User input RAW: '{user_text}'")
    
    # === ESTRAI TARGET E COMPRESSION ===
    
    llm = ChatOllama(
        model=cfg.local_llm,
        temperature=0,
        num_ctx=cfg.llm_context_window
    )
    
    llm_extractor = llm.with_structured_output(AnalysisInfoExtraction)
    
    extraction_result = llm_extractor.invoke([
        SystemMessage(content=analysis_info_extraction_instructions),
        HumanMessage(content=f"Risposta utente: {user_text}")
    ])
    
    state.target = extraction_result.target or state.target or "stm32h743"
    state.compression = extraction_result.compression or state.compression or "high"
    state.ai_output_dir = "./analisiAI"
    
    os.makedirs(state.ai_output_dir, exist_ok=True)
    
    logger.info(f"✓ Configurazione estratta:")
    logger.info(f"  Target: {state.target}")
    logger.info(f"  Compression: {state.compression}")
    
    return state


# ============================================================================
# NODO: SCEGLI DA MODELLI PREDEFINITI (TASK-BASED)
# ============================================================================
def choose_predefined_taskbased_model(state: MasterState, config: dict) -> MasterState:
    """
    Mostra modelli predefiniti con parsing LLM.
    Salva il task per fallback intelligente.
    Usa PREDEFINED_MODELS come unica fonte.
    """
    
    logger.info("📋 Scelta modello da catalogo predefinito...")
    
    # Ricarica modelli dal JSON per essere sicuri che siano aggiornati
    global PREDEFINED_MODELS
    PREDEFINED_MODELS = load_predefined_models()
    
    cfg = Configuration.from_runnable_config(config)
    llm = ChatOllama(
        model=cfg.local_llm,
        temperature=0,
        num_ctx=cfg.llm_context_window
    )
    
    # === STEP 1: COSTRUZIONE PROMPT DINAMICO ===
    categories = list(PREDEFINED_MODELS.keys())
    
    prompt_lines = ["Seleziona il task che vuoi fare:\n"]
    idx = 1
    mapping = {}
    
    for cat in categories:
        desc = PREDEFINED_MODELS[cat].get("description", cat)
        prompt_lines.append(f"{idx}. {desc}")
        mapping[str(idx)] = cat
        idx += 1
    
    # Opzioni fisse alla fine
    reg_idx = idx
    prompt_lines.append(f"{reg_idx}. Registra un NUOVO modello (fornisci dettagli nel prossimo step)")
    mapping[str(reg_idx)] = "register_new"
    
    other_idx = idx + 1
    prompt_lines.append(f"{other_idx}. Nessuno di questi (ricerca online)")
    mapping[str(other_idx)] = "other"
    
    prompt_text = "\n".join(prompt_lines)
    prompt_text += f"\n\nRispondi: 1-{other_idx} oppure descrivi il task"
    
    prompt = {"instruction": prompt_text}
    
    user_response = interrupt(prompt) # per adesso commentata per velocizzare
    #user_response = "" # BYPASS
    if isinstance(user_response, dict):
        user_text = user_response.get("response", user_response.get("input", str(user_response)))
    else:
        user_text = str(user_response).strip()
    
    # Default: image classification (option 1)
    if not user_text or user_text.strip() == "":
        user_text = "1"
    
    logger.info(f"📝 User task input: '{user_text}'")
    
    # === ESTRAI TASK CON LLM ===
    
    # Costruisci istruzioni dinamiche per l'LLM
    dynamic_instructions = f"""Analizza la risposta dell'utente e determina il task AI richiesto.
Il sistema presenta un menu dinamico:
{prompt_text}

MAPPING CRITICO:
"""
    for k, v in mapping.items():
        dynamic_instructions += f'- "{k}" -> {v}\n'
    
    dynamic_instructions += """
Rispondi sempre in formato JSON con:
- "task": la chiave tecnica del task (es: image_classification, register_new, etc.)
- "confidence": 0.0-1.0
"""

    llm_extractor = llm.with_structured_output(TaskSelectionExtraction)
    
    task_result = llm_extractor.invoke([
        SystemMessage(content=dynamic_instructions),
        HumanMessage(content=f"Risposta utente: {user_text}")
    ])
    
    logger.info(f"✓ Task estratto: {task_result.task} (confidence: {task_result.confidence:.2f})")
    
    selected_task = task_result.task
    
    # ✅ SALVA IL TASK NELLO STATE PER FALLBACK INTELLIGENTE
    state.last_task = selected_task
    logger.info(f"✓ Task salvato per fallback: {selected_task}")
    
    if selected_task == "register_new":
        logger.info("🆕 Utente vuole registrare un nuovo modello")
        state.model_discovery_method = "register_new"
        return state
        
    if selected_task == "other" or task_result.confidence < 0.5:
        logger.info("✓ Task non riconosciuto, va a ricerca online")
        state.model_discovery_method = "search"
        state.search_iterations = 0
        return state
    
    # === STEP 2: CARICA MODELLI DA PREDEFINED_MODELS ===
    
    task_info = PREDEFINED_MODELS.get(selected_task)
    
    if not task_info:
        logger.warning(f"⚠️  Task '{selected_task}' non trovato in PREDEFINED_MODELS")
        logger.info("→ Fallback a ricerca online")
        state.model_discovery_method = "search"
        state.search_iterations = 0
        return state
    
    available_models = task_info["models"]
    state.available_models = available_models
    
    logger.info(f"✓ Caricati {len(available_models)} modelli per task '{selected_task}'")
    
    # === STEP 3: MOSTRA MODELLI CON COMPATIBILITÀ ===
    
    # Recupera limiti MCU
    flash_limit, ram_limit = get_mcu_limits(state.target)
    
    print("\n" + "="*70)
    print(f"📦 MODELLI DISPONIBILI: {task_info['description']}")
    print(f"🎯 Target: {state.target} (Flash: {format_bytes(flash_limit)})")
    print("="*70)
    
    model_options_text = []
    
    for i, model in enumerate(available_models, 1):
        # Calcola compatibilità
        size_bytes = parse_size_str(model['size'])
        flash_ratio = size_bytes / flash_limit
        
        status_icon = "❓"
        status_note = ""
        
        if flash_ratio <= 1.0:
            status_icon = "✅"
            status_note = "Fits"
        elif flash_ratio <= 8.0:
            status_icon = "⚠️"
            status_note = f"Compressible ({flash_ratio:.1f}x)"
        else:
            status_icon = "❌"
            status_note = f"Too Large ({flash_ratio:.1f}x)"
            
        # Estrai formato dal nome file (url o local_filename)
        import os
        filename = model.get('local_filename', model['url'])
        _, ext = os.path.splitext(filename)
        ext = ext.upper() if ext else "N/D"

        print(f"\n{i}. {model['name']} {status_icon}")
        print(f"   📏 Dimensione: {model['size']} ({status_note}) | 📄 Formato: {ext}")
        print(f"   🎯 Accuratezza: {model['accuracy']}")
        print(f"   ⚡ Inferenza: {model['inference_time']}")
        
        model_options_text.append(f"{i}. {model['name']} {status_icon} [{ext}] ({model['size']} - {status_note})")
    
    print(f"\n{len(available_models)+1}. Nessuno di questi (ricerca online)")
    print("="*70 + "\n")
    
    # === STEP 4: CHIEDI MODELLO ===
    
    # Crea lista modelli per il prompt
    models_list = "\n".join(model_options_text)
    
    model_prompt = {
        "instruction": f"""Quale modello vuoi usare per {task_info['description']}?

Opzioni disponibili:
{models_list}
{len(available_models)+1}. Nessuno di questi (ricerca online)

Rispondi con: numero (1-{len(available_models)+1}) oppure descrivi
        """
    }
    
    model_response = interrupt(model_prompt) # per adesso commentata per velocizzare
    # model_response = "" # BYPASS
    
    if isinstance(model_response, dict):
        model_text = model_response.get("response", model_response.get("input", str(model_response)))
    else:
        model_text = str(model_response).strip()
    
    logger.info(f"📝 User model input: '{model_text}'")
    
    # === MANUALE: CHECK FUZZY MATCH SU NOME ===
    # Permette di scrivere "MobileNet V2" invece di "1"
    matched_model_index = None
    
    user_normalized = model_text.lower().replace(" ", "").replace("-", "").replace("_", "")
    
    # 1. Check Exact/Partial Index (es. "1", "1.")
    if model_text.strip().replace(".", "").isdigit():
        pass # Lascia all'LLM o gestisci dopo
        
    # 2. Check Name Match
    else:
        for i, model in enumerate(available_models, 1):
            name_normalized = model['name'].lower().replace(" ", "").replace("-", "").replace("_", "")
            
            # Match forte: una contiene l'altra
            if name_normalized in user_normalized or user_normalized in name_normalized:
                logger.info(f"✓ Fuzzy match trovato: '{model_text}' -> {model['name']} (Index {i})")
                matched_model_index = i
                break
    
    # Se abbiamo un match manuale, bypassiamo l'LLM o lo usiamo come conferma
    if matched_model_index is not None:
        model_result = ModelSelectionExtraction(
            model_index=matched_model_index, 
            model_accepted=True, 
            wants_another_search=False
        )
    else:
        # === FALLBACK: ESTRAI SCELTA CON LLM ===
        llm_model_extractor = llm.with_structured_output(ModelSelectionExtraction)
        
        model_result = llm_model_extractor.invoke([
            SystemMessage(content=model_selection_instructions),
            HumanMessage(content=f"Numero di modelli disponibili: {len(available_models)}\nRisposta utente: {model_text}")
        ])
    
    logger.info(f"✓ Scelta estratta:")
    logger.info(f"  model_index: {model_result.model_index}")
    logger.info(f"  model_accepted: {model_result.model_accepted}")
    logger.info(f"  wants_another_search: {model_result.wants_another_search}")
    
    # === STEP 5: APPLICA SCELTA ===
    
    if model_result.model_accepted and model_result.model_index:
        model_idx = model_result.model_index - 1
        
        if 0 <= model_idx < len(available_models):
            selected_model = available_models[model_idx]
            state.selected_model = selected_model
            state.model_discovery_method = "taskbased"
            state.model_accepted = True
            
            logger.info(f"✓ Modello selezionato: {selected_model['name']}")
            logger.info(f"  Size: {selected_model['size']}, Accuracy: {selected_model['accuracy']}")
            
            # ✅ STOP DOWNLOAD QUI - CI PENSA IL NODO SUCCESSIVO (download_model)
            # state = download_model_to_cache(state, config, selected_model)
            
        else:
            logger.warning(f"⚠️  Indice modello fuori range: {model_result.model_index}")
            logger.info("→ Fallback a ricerca online")
            state.model_discovery_method = "search"
            state.search_iterations = 0
    
    else:
        # Nessun modello predefinito accettato
        if model_result.wants_another_search:
            logger.info("✓ Utente vuole ricerca online")
            state.model_discovery_method = "search"
            state.search_iterations = 0
        else:
            logger.info("✓ Utente vuole default task-based")
            
            # ✅ USA IL PRIMO MODELLO DEL TASK COME DEFAULT (non config generico)
            fallback_model = get_task_based_default_model(selected_task)
            
            if fallback_model:
                logger.info(f"✓ Fallback task-based: {fallback_model['name']}")
                state.selected_model = fallback_model
                state.model_discovery_method = "taskbased_fallback"
                state.model_accepted = True
                
                # Download del fallback model - STOP, ci pensa il nodo successivo
                # state = download_model_to_cache(state, config, fallback_model)
            else:
                # Ultimo fallback: config generico
                logger.warning("⚠️  Nessun fallback task-based, uso config")
                state.model_path = cfg.ai_model_path
                state.model_discovery_method = "default"
    
    return state


# ============================================================================
# NODO PRINCIPALE per la ricerca modelli !
# ============================================================================
def search_recommendation_model(state: MasterState, config: dict) -> MasterState:
    """
    ✅ NODO PRINCIPALE: Ricerca modello con fallback intelligente
    
    TYPE HINTS: state: MasterState, config: dict → MasterState
    
    Flusso:
    1. GitHub (ibrido Python+LLM) - conta iterazione
    2. Google (fallback) - NON conta iterazione
    3. Interrupt per conferma utente
    4. Ritorno a "search" nel routing (max 3 iterazioni)
    5. Task-based default - SOLO dopo 3 iterazioni fallite
    """
    
    logger.info("=" * 70)
    logger.info(f"🔍 RICERCA MODELLO [Iter {state.search_iterations + 1}/3]")
    logger.info(f"   Task: {state.last_task} | Target: {state.target}")
    logger.info("=" * 70)
    
    # ====================================================================
    # FASE 1: GITHUB (ibrido) - CONTA ITERAZIONE
    # ====================================================================
    logger.info(f"\n📍 FASE 1: GitHub (ibrido) - Iter {state.search_iterations + 1}/3")
    
    github_result = search_h5_file_in_repo_hybrid(
        repo_path="STMicroelectronics/stm32ai-modelzoo",
        task=state.last_task,
        target_mcu=state.target,
        config=config
    )
    
    if github_result and github_result.get('url_raw'):
        logger.info(f"✓ GitHub: Trovato e validato!")
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
        
        # ✅ INTERRUPT: Chiedi conferma all'utente
        logger.info(f"\n✓ MODELLO TROVATO - Richiesta conferma utente...")
        
        # Estrai formato
        import os
        filename = github_result.get('local_filename', github_result['url_raw'])
        _, ext = os.path.splitext(filename)
        ext = ext.upper() if ext else "N/D"

        prompt = {
            "instruction": f"""Modello AI trovato per {state.last_task}

📦 Dettagli:
- Nome: {github_result['name']}
- Formato: {ext}
- Size: {github_result.get('size', 'N/A')}
- Source: {github_result.get('source', 'GitHub')}
- Method: {github_result.get('selection_method', 'N/A')}

🔗 URL: {github_result['url_raw']}

❓ Accetti questo modello? (rispondi: si/no oppure yes/no)
- 'si' o 'yes': Procedi con il download
- 'no': Continua la ricerca di altri modelli""",
        }
        
        user_confirmation = interrupt(prompt)
        
        # Gestisci dict o stringa
        if isinstance(user_confirmation, dict):
            confirmation_text = str(user_confirmation.get("response", user_confirmation.get("input", ""))).lower().strip()
        else:
            confirmation_text = str(user_confirmation).lower().strip()
        
        # Default: accept model (si)
        if not confirmation_text or confirmation_text.strip() == "":
            confirmation_text = "si"
        
        logger.info(f"📝 Risposta utente: '{confirmation_text}'")
        
        # Accetto se: si, yes, ok, accetto, conferma, y, sì
        accepted_keywords = ["si", "yes", "ok", "accetto", "conferma", "y", "sì"]
        
        # Se ACCETTA → return state (rimane github_hybrid/google_search/taskbased_fallback)
        if any(keyword in confirmation_text for keyword in accepted_keywords):
            logger.info(f"✓ Modello ACCETTATO dall'utente")
            return state  # ← Va al download

        # Se RIFIUTA → ritorna al loop
        else:
            logger.warning(f"❌ Modello RIFIUTATO dall'utente")
            state.model_discovery_method = "search"  # ← Torna al loop, continua ricerca se iterazioni rimaste

    else:
        logger.warning(f"❌ GitHub fallito")
    
    state.search_iterations += 1
    
    # ====================================================================
    # FASE 2: GOOGLE (fallback) - NON CONTA ITERAZIONE
    # ====================================================================
    logger.info(f"\n📍 FASE 2: Google (fallback, NO iter++)")
    
    if state.search_iterations <= 3:
        google_result = search_via_google_tools_hybrid(state, config)
        
        if google_result['success'] and google_result['url_valid']:
            logger.info(f"✓ Google: Trovato e validato!")
            logger.info(f"  {google_result['model']['name']}")
            
            state.selected_model = google_result['model']
            state.model_discovery_method = "google_search"
            
            # ✅ INTERRUPT: Chiedi conferma all'utente (anche per Google)
            logger.info(f"\n✓ MODELLO TROVATO (Google) - Richiesta conferma utente...")
            
            # Estrai formato
            import os
            filename = google_result['model'].get('local_filename', google_result['model']['url'])
            _, ext = os.path.splitext(filename)
            ext = ext.upper() if ext else "N/D"
            
            prompt = {
                "instruction": f"""Modello AI trovato per {state.last_task}

📦 Dettagli:
- Nome: {google_result['model']['name']}
- Formato: {ext}
- Size: {google_result['model'].get('size', 'N/A')}
- Source: {google_result['model'].get('source', 'Google Search')}

🔗 URL: {google_result['model']['url']}

❓ Accetti questo modello? (rispondi: si/no oppure yes/no)
- 'si' o 'yes': Procedi con il download
- 'no': Continua la ricerca di altri modelli""",
            }
            
            user_confirmation = interrupt(prompt)
            
            if isinstance(user_confirmation, dict):
                confirmation_text = str(user_confirmation.get("response", user_confirmation.get("input", ""))).lower().strip()
            else:
                confirmation_text = str(user_confirmation).lower().strip()
            
            logger.info(f"📝 Risposta utente: '{confirmation_text}'")
            
            accepted_keywords = ["si", "yes", "ok", "accetto", "conferma", "y", "sì"]
            
            if any(keyword in confirmation_text for keyword in accepted_keywords):
                logger.info(f"✓ Modello ACCETTATO dall'utente")
                logger.info("=" * 70)
                return state  # ← Procedi al download
            else:
                logger.warning(f"❌ Modello RIFIUTATO dall'utente - Continua ricerca")
        else:
            logger.warning(f"❌ Google: Fallito")
    
    # ====================================================================
    # FASE 3: VERIFICA ITERAZIONI
    # ====================================================================
    
    if state.search_iterations < 3:
        # ✅ RITORNA AL ROUTING CON "search" - PROSSIMO TENTATIVO
        logger.info(f"\n📍 FASE 3: Iterazione {state.search_iterations}/3 completata")
        logger.info(f"   ↻ Ritorno al routing per prossimo tentativo...")
        
        state.model_discovery_method = "search"  # ← TORNA AL LOOP
        
        logger.info("=" * 70)
        return state
    
    # ====================================================================
    # FASE 4: MAX ITERAZIONI RAGGIUNTO - FALLBACK TASK-BASED
    # ====================================================================
    else:
        logger.warning(f"\n⚠️  FASE 4: Max iterazioni raggiunto (3/3)")
        logger.info(f"   → Attivazione fallback task-based...")
        
        fallback_model = get_task_based_default_model(state.last_task)
        
        if fallback_model:
            logger.info(f"✓ Fallback trovato: {fallback_model['name']}")
            
            state.selected_model = fallback_model
            state.model_discovery_method = "taskbased_fallback"
            
            # ✅ INTERRUPT FINALE: Chiedi conferma anche per fallback
            logger.info(f"\n✓ MODELLO FALLBACK - Richiesta conferma utente...")
            
            # Estrai formato
            import os
            filename = fallback_model.get('local_filename', fallback_model.get('url', ''))
            _, ext = os.path.splitext(filename)
            ext = ext.upper() if ext else "N/D"
            
            prompt = {
                "instruction": f"""Modello di fallback per {state.last_task}

Dopo 3 tentativi di ricerca, ecco il modello di fallback:

📦 Dettagli:
- Nome: {fallback_model['name']}
- Formato: {ext}
- Size: {fallback_model.get('size', 'N/A')}
- Source: Task-based fallback

🔗 URL: {fallback_model.get('url', 'N/A')}

❓ Accetti questo modello? (rispondi: si/no)
- 'si': Procedi con il download
- 'no': Usa il modello generico dal config""",
            }
            
            user_confirmation = interrupt(prompt)
            
            if isinstance(user_confirmation, dict):
                confirmation_text = str(user_confirmation.get("response", user_confirmation.get("input", ""))).lower().strip()
            else:
                confirmation_text = str(user_confirmation).lower().strip()
            
            logger.info(f"📝 Risposta utente: '{confirmation_text}'")
            
            accepted_keywords = ["si", "yes", "ok", "accetto", "conferma", "y", "sì"]
            
            if any(keyword in confirmation_text for keyword in accepted_keywords):
                logger.info(f"✓ Modello ACCETTATO dall'utente")
                logger.info("=" * 70)
                return state  # ← Procedi al download
            else:
                logger.warning(f"❌ Modello RIFIUTATO - Uso config default")
                cfg = Configuration.from_runnable_config(config)
                state.model_path = cfg.ai_model_path
                state.model_discovery_method = "default"
        else:
            logger.warning(f"❌ Nessun fallback disponibile")
            
            cfg = Configuration.from_runnable_config(config)
            state.model_path = cfg.ai_model_path
            state.model_discovery_method = "default"
        
        logger.info("=" * 70)
        return state


def model_selection_routing(state: MasterState) -> Literal["run_analyze", "download_model", "search_recommendation_model", "add_custom_model_procedure"]:
    """
    Routing intelligente dopo selezione modello.
    Gestisce il loop di ricerca fino a max 3 tentativi e la registrazione di nuovi modelli.
    """
    
    logger.info(f"\n🔄 ROUTING DECISION:")
    logger.info(f"   discovery_method: {state.model_discovery_method}")
    logger.info(f"   search_iterations: {state.search_iterations}")
    
    # ====================================================================
    # CASO 0: Registrazione nuovo modello
    # ====================================================================
    if state.model_discovery_method == "register_new":
        logger.info("   → Registrazione nuovo modello, va a add_custom_model_procedure")
        return "add_custom_model_procedure"

    # ====================================================================
    # CASO 1: Default model (niente ricerca)
    # ====================================================================
    if state.model_discovery_method == "default":
        logger.info("   → Modello pronto/selezionato, va a download_model per ispezione")
        return "download_model"
    
    # ====================================================================
    # CASO 2: Ricerca attiva (ritorna al loop se rifiutato)
    # ====================================================================
    elif state.model_discovery_method == "search":
        if state.search_iterations < 3:
            logger.info(f"   → Ricerca in corso ({state.search_iterations}/3), ritorno a search_recommendation_model")
            return "search_recommendation_model"
        else:
            logger.info(f"   → Max iterazioni (3/3) raggiunto, vai a run_analyze (default)")
            return "run_analyze"
    
    # ====================================================================
    # CASO 3: Modello trovato e ACCETTATO
    # ====================================================================
    else:  # github_hybrid, google_search, taskbased_fallback
        logger.info(f"   → {state.model_discovery_method} ACCETTATO dall'utente, vai a download_model")
        return "download_model"

# ============================================================================
# PARTE 1 della ricerca di modelli: RICERCA GITHUB IBRIDA (Python + LLM con Structured Output)
# ============================================================================

def search_h5_file_in_repo_hybrid( #fondamentale 
    repo_path: str,
    task: str,
    target_mcu: Optional[str] = None,
    config: dict = None,
    max_depth: int = 5  # ← LIMITE DI PROFONDITÀ
) -> Optional[dict]:
    """
    Ricerca file .h5 con approccio ibrido (OTTIMIZZATO)
    
    ✅ Migliorie:
    - Limite di profondità per evitare loop infiniti
    - Logging dettagliato per trovare i blocchi
    - Early exit su file trovati
    - Timeout virtualizzato su API GitHub
    """
    
    try:
        logger.info(f"🔗 Ricerca GitHub IBRIDA: {task}")
        
        # STEP 1: PYTHON → Scansione repo
        logger.info(f"→ STEP 1: Scansione repo (Python)...")
        
        token = os.getenv("GITHUB_ACCESS_TOKEN")
        if not token:
            logger.error("❌ GITHUB_ACCESS_TOKEN non impostato!")
            return None
        
        try:
            g = Github(token)
            repo = g.get_repo(repo_path)
            logger.info(f"✓ Connesso a {repo_path}")
        except Exception as e:
            logger.error(f"❌ Errore connessione: {str(e)[:80]}")
            return None
        
        # ✅ TASK → FOLDER
        folder = task.lower().replace(" ", "_")
        
        try:
            root_contents = repo.get_contents(folder)
            logger.info(f"✓ Cartella trovata: {folder}/")
        except Exception as e:
            logger.error(f"❌ Cartella non trovata: {folder}")
            logger.error(f"   Dettagli: {str(e)[:80]}")
            return None
        
        h5_files: List[dict] = []
        items_checked = 0  # Counter per debugging
        
        def scan_repo(contents_list, depth=0):
            """
            Scansiona repo e raccoglie file .h5
            ✅ OTTIMIZZATO: Early exit, limit depth, logging
            """
            nonlocal items_checked
            
            if depth >= max_depth:
                logger.debug(f"  ⚠️  Max depth ({max_depth}) raggiunta, stop")
                return
            
            try:
                for item in contents_list:
                    items_checked += 1
                    
                    # Log ogni 50 item
                    if items_checked % 50 == 0:
                        logger.info(f"  📊 Scansionati {items_checked} item ({len(h5_files)} .h5 trovati)...")
                    
                    try:
                        if item.type == "dir":
                            logger.debug(f"  {'  ' * depth}📁 Dir: {item.name}")
                            
                            try:
                                sub = repo.get_contents(item.path)
                                scan_repo(sub, depth + 1)
                            except Exception as e:
                                logger.debug(f"  {'  ' * depth}⚠️  Errore lettura {item.path}: {type(e).__name__}")
                                continue
                        
                        elif item.type == "file" and any(item.name.endswith(ext) for ext in [".h5", ".keras", ".onnx", ".tflite"]):
                            description = extract_description(item.name, item.path)
                            h5_files.append({
                                'name': item.name,
                                'path': item.path,
                                'size': item.size if hasattr(item, 'size') else 0,
                                'description': description,
                                'folder': item.path.rsplit('/', 1)[0] if '/' in item.path else folder
                            })
                            logger.debug(f"  {'  ' * depth}✅ File trovato: {item.name}")
                            
                            # ✅ EARLY EXIT se trovi abbastanza file
                            if len(h5_files) >= 20:  # Limite pratico
                                logger.info(f"  ℹ️  Trovati {len(h5_files)} file, stop ricerca")
                                return
                    
                    except Exception as e:
                        logger.debug(f"  ⚠️  Errore item {item.name}: {type(e).__name__}")
                        continue
            
            except Exception as e:
                logger.error(f"❌ Errore durante scan_repo: {str(e)[:100]}")
                import traceback
                logger.debug(traceback.format_exc())
        
        logger.info(f"→ Inizio scansione ricorsiva...")
        scan_repo(root_contents)
        
        logger.info(f"✓ Scansione completata: {items_checked} item, {len(h5_files)} file .h5 trovati")
        
        if not h5_files:
            logger.warning(f"❌ Nessun file .h5 trovato dopo {items_checked} controlli")
            return None
        
        logger.info(f"✓ Trovati {len(h5_files)} file .h5")
        for f in h5_files[:5]:
            logger.info(f"  - {f['name']} ({format_bytes(f['size'])}) [{f['description']}]")
        
        if len(h5_files) > 5:
            logger.info(f"  ... e altri {len(h5_files) - 5} file")
        
        # STEP 2: LLM → Selezione sofisticata
        logger.info(f"→ STEP 2: Ragionamento con LLM (structured)...")
        
        selected_file = llm_select_best_model(
            h5_files=h5_files,
            task=task,
            target_mcu=target_mcu or "STM32H7",
            config=config
        )
        
        if not selected_file:
            logger.warning(f"❌ LLM fallito, uso primo file")
            selected_file = h5_files[0]
            selection_method = "fallback_first"
        else:
            selection_method = "llm_reasoning"
            logger.info(f"✓ LLM ha scelto: {selected_file['name']}")
        
        # STEP 3: PYTHON → URL e Validazione
        logger.info(f"→ STEP 3: Costruzione URL e validazione...")
        
        url_raw = f"https://raw.githubusercontent.com/{repo_path}/main/{selected_file['path']}"
        logger.info(f"🔗 URL: {url_raw[:70]}...")
        
        is_valid = validate_model_url_quick(url_raw)
        
        if not is_valid:
            logger.warning(f"❌ URL non scaricabile (404?)")
            
            # Fallback: prova altri file
            for alt_file in h5_files[1:3]:
                logger.info(f"→ Tentativo alternativo: {alt_file['name']}...")
                alt_url = f"https://raw.githubusercontent.com/{repo_path}/main/{alt_file['path']}"
                
                if validate_model_url_quick(alt_url):
                    logger.info(f"✓ Alternativo valido!")
                    selected_file = alt_file
                    url_raw = alt_url
                    is_valid = True
                    break
        
        if not is_valid:
            logger.error(f"❌ Nessun URL valido")
            return None
        
        logger.info(f"✓ URL validato! Size: {format_bytes(selected_file['size'])}")
        
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
        logger.error(f"❌ Errore: {str(e)[:150]}")
        import traceback
        logger.debug(traceback.format_exc())
        return None


def llm_select_best_model(
    h5_files: List[dict],
    task: str,
    target_mcu: str,
    config: dict = None
) -> Optional[dict]:
    """
    LLM ragiona e seleziona il migliore file .h5
    ✅ STRUTTURATO: Forza formato con Pydantic
    """
    
    try:
        logger.info(f"→ Invio a LLM ({len(h5_files)} file)...")
        
        h5_list_text = "\n".join([
            f"{i+1}. {f['name']:40} | {format_bytes(f['size']):>10} | {f['description']}"
            for i, f in enumerate(h5_files)
        ])
        
        prompt = f"""Sei un esperto di modelli AI per STM32 embedded systems.

TASK RICHIESTA: {task}
TARGET MCU: {target_mcu}

FILE DISPONIBILI NEL REPO:
{h5_list_text}

⚠️ ISTRUZIONI CRITICHE:
1. Analizza TUTTI i modelli (.h5, .keras, .onnx, .tflite)
2. Scegli il MIGLIORE per la task (considera: compatibilità, dimensione, architettura)
3. Ritorna SOLO il numero dell'indice (1-{len(h5_files)})
4. NON aggiungere altro testo

SCORING:
- Exact match task: +100
- Known architecture (resnet, yolo, mobilenet, efficientnet): +50
- Size < 10MB: +25
- Size < 1MB: +50

RISPOSTA - SOLO IL NUMERO:
{1}"""
        
        logger.debug(f"Prompt LLM: {prompt[:350]}...")
        
        # Use centralized LLM setup
        from src.assistant.utils import get_llm
        llm = get_llm(config)
        
        # ✅ STRUCTURED OUTPUT - Forza formato
        class ModelSelection(BaseModel):
            selected_index: int = Field(
                description=f"Indice 1-based OBBLIGATORIO (1-{len(h5_files)}). Niente altro.",
                ge=1,  # Minimo 1
                le=len(h5_files)  # Massimo len(h5_files)
            )
        
        llm_selector = llm.with_structured_output(ModelSelection)
        
        logger.info(f"→ Invio prompt a LLM...")
        
        selection = llm_selector.invoke([
            SystemMessage(content="""Tu sei un task di selezione modello.
DEVI rispondere SOLO con un JSON valido nel formato specificato.
Niente testo, niente spiegazioni.
Se hai dubbi, scegli il modello più piccolo e stabile."""),
            HumanMessage(content=prompt)
        ])
        
        logger.info(f"📊 LLM Selection:")
        logger.info(f"  Index: {selection.selected_index}")
        
        # Converti 1-based → 0-based
        idx_0based = selection.selected_index - 1
        
        if idx_0based < 0 or idx_0based >= len(h5_files):
            logger.warning(f"❌ Indice fuori range: {selection.selected_index}")
            logger.warning(f"   Fallback: seleziono il primo file")
            return h5_files[0]
        
        selected_file = h5_files[idx_0based]
        logger.info(f"✓ LLM scelto file #{selection.selected_index}: {selected_file['name']}")
        logger.info(f"  Size: {format_bytes(selected_file['size'])}")
        logger.info(f"  Description: {selected_file['description']}")
        
        return selected_file
    
    except Exception as e:
        logger.error(f"❌ Errore LLM selection: {str(e)[:100]}")
        import traceback
        logger.debug(traceback.format_exc())
        
        logger.warning(f"→ Fallback: seleziono il primo file")
        return h5_files[0] if h5_files else None


# ============================================================================
# PARTE 2: RICERCA GOOGLE FALLBACK (con LLM Structured Extraction)
# ============================================================================

def search_via_google_tools_hybrid(
    state: MasterState,
    config: dict
) -> dict:
    """
    Ricerca Google Search come fallback (NON incrementa iterazioni)
    Usa SearchResultExtraction con structured output
    """
    
    try:
        logger.info(f"🔍 Ricerca Google (fallback, NO iter++)...")
        
        google_prompt = f"""Ricerca modelli AI (.h5, .keras, .onnx, .tflite) per STM32
Target: {state.target}
Task: {state.last_task}

Criteri:
1. Link GitHub Raw o Hugging Face
2. Download diretto
3. File compatibile con STM32 X-CUBE-AI

Ritorna esattamente questo formato JSON:
- Nome: [titolo_modello]
- URL: [link_scaricabile]
- Size: [MB]
- Accuracy: [%]
- Inference: [ms]
"""
        
        logger.info(f"→ Google Agent...")
        
        google_agent = Agent(
            model=Ollama(id="mistral"),
            tools=[GoogleSearchTools()],
            instructions=[
                "Ricerca file .h5 per STM32",
                "Link GitHub /raw/ diretti",
                "Non inventare URL"
            ],
            show_tool_calls=True
        )
        
        google_response = google_agent.run(google_prompt)
        response_text = (
            google_response.content 
            if hasattr(google_response, 'content') 
            else str(google_response)
        )
        
        logger.info(f"📊 Google Response: {response_text[:250]}...")
        
        if "NOT_FOUND" in response_text.upper():
            logger.warning(f"❌ Google: Non trovato")
            return {'success': False, 'url_valid': False, 'model': None}
        
        # ✅ ESTRAI CON LLM STRUCTURED OUTPUT (NON regex!)
        logger.info(f"→ Estrazione con SearchResultExtraction...")
        
        cfg = Configuration.from_runnable_config(config)
        
        llm = ChatOllama(
            model=cfg.local_llm,
            temperature=0,
            num_ctx=cfg.llm_context_window
        )
        
        llm_extractor = llm.with_structured_output(SearchResultExtraction)
        
        try:
            search_extraction = llm_extractor.invoke([
                SystemMessage(content=search_result_extraction_instructions),
                HumanMessage(content=f"Risultato ricerca Google:\n\n{response_text}")
            ])
            
            logger.info(f"📊 LLM Extraction:")
            logger.info(f"  Model: {search_extraction.model_name}")
            logger.info(f"  URL: {search_extraction.download_url[:60] if search_extraction.download_url else 'None'}...")
            logger.info(f"  Size: {search_extraction.model_size}")
            logger.info(f"  Valid: {search_extraction.is_valid}")
            
        except Exception as e:
            logger.error(f"❌ Estrazione LLM fallita: {str(e)[:100]}")
            import traceback
            logger.debug(traceback.format_exc())
            return {'success': False, 'url_valid': False, 'model': None}
        
        # ✅ VALIDAZIONE
        if not search_extraction.is_valid or not search_extraction.download_url:
            logger.warning(f"❌ URL non valido da LLM extraction")
            return {'success': False, 'url_valid': False, 'model': None}
        
        logger.info(f"🔗 Validazione URL...")
        is_valid = validate_model_url_quick(search_extraction.download_url)
        
        if is_valid:
            logger.info(f"✓ Google: URL VALIDO!")
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
            logger.warning(f"❌ Google: URL non scaricabile (404?)")
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
    Estrae descrizione leggibile da nome file
    Esempio: "mobilenet_v2_224_224.h5" → "Mobilenet V2 224 224"
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
    Formatta bytes in formato leggibile
    Esempio: 1048576 → "1.0MB"
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
    Converte stringa dimensione (es. "14.0MB") in bytes.
    Gestisce KB, MB, GB.
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
    """Ritorna il primo modello disponibile per il task da PREDEFINED_MODELS"""
    
    if task not in PREDEFINED_MODELS:
        logger.warning(f"⚠️  Task non trovato: {task}")
        for task_key, info in PREDEFINED_MODELS.items():
            if info.get("models"):
                return info["models"][0]
        return None
    
    task_info = PREDEFINED_MODELS[task]
    models = task_info.get("models", [])
    
    if not models:
        logger.warning(f"⚠️  Nessun modello per task: {task}")
        return None
    
    default_model = models[0]
    logger.info(f"✓ Default model per '{task}': {default_model['name']}")
    
    return default_model



# ============================================================================
# LEGACY ENVIRONMENT SUPPORT
# ============================================================================

ARCHITECTURE_ENV_MAP = {
    'mobilenet': 'stm32_legacy',
    'resnet': 'stm32_legacy',
    'vgg': 'stm32_legacy',
    'efficientnet': 'stm32_legacy',
    'inception': 'stm32_legacy',
    'yolo': 'stm32_legacy',
    'har': 'stm32_legacy',
    'custom': 'stm32_legacy',
}

CONDA_PYTHON_PATHS = {
    'stm32_legacy': '/home/mrusso/miniconda3/envs/stm32_legacy/bin/python', #keras 2.x 
    'stm32': '/home/mrusso/miniconda3/envs/stm32/bin/python', # keras 3.x
}

def detect_architecture_from_model(model_path: str) -> str:
    """Detecta architettura dal nome modello"""
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
    """Esegue codice in subprocess con python specifico"""
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

def inspect_model_via_legacy_env(model_path: str) -> Optional[dict]:
    """
    Ispeziona modello usando env legacy (per evitare crash Keras 3 con modelli vecchi)
    Ritorna dict con info architettura o None se fallisce.
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
        
        # Scegli environment: .keras -> stm32 (Keras 3), .h5 -> stm32_legacy (Keras 2) o stm32
        if model_path.endswith('.keras'):
            env_name = 'stm32'
        else:
            env_name = ARCHITECTURE_ENV_MAP.get(arch, 'stm32_legacy')
        
        python_path = CONDA_PYTHON_PATHS.get(env_name)
        
        if not python_path or not os.path.exists(python_path):
            logger.warning(f"⚠️  Python path non trovato per {env_name}: {python_path}")
            return None
            
        logger.info(f"🔄 Inspecting via subprocess ({env_name})...")
        
        script = f"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import json
import sys

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
# NODO 3: DOWNLOAD MODELLO
# ============================================================================
def download_model(state: MasterState, config: dict) -> MasterState:
    """
    Wrapper per scaricare il modello dallo state.selected_model.
    Viene chiamato dal routing dopo ricerca online accettata.
    """
    
    logger.info("📥 Nodo download_model (wrapper) avviato...")
    
    if not state.selected_model:
        logger.error("❌ selected_model non trovato!")
        cfg = Configuration.from_runnable_config(config)
        state.model_path = cfg.ai_model_path
        state.model_discovery_method = "default"
        return state
    
    logger.info(f"📦 Download: {state.selected_model['name']}")
    
    # ✅ CHIAMA download_model_to_cache CON il modello
    state = download_model_to_cache(state, config, state.selected_model)
    
    return state

def download_model_to_cache(state: MasterState, config: dict, model: dict) -> MasterState:
    """
    Download modello con skip intelligente + ANALISI ROBUSTA
    """
    
    logger.info(f"📥 Download modello: {model['name']}...")
    
    cache_dir = os.path.expanduser("~/.stm32_ai_models")
    os.makedirs(cache_dir, exist_ok=True)
    
    model_filename = model.get("local_filename")
    
    if not model_filename:
        logger.error("❌ local_filename non trovato nel modello!")
        cfg = Configuration.from_runnable_config(config)
        state.model_path = cfg.ai_model_path
        return state
    
    # ===== SECURITY: Sanitize filename to prevent path traversal =====
    from src.assistant.utils import sanitize_filename
    model_filename = sanitize_filename(model_filename)
    logger.debug(f"Sanitized filename: {model_filename}")
    
    cached_path = os.path.join(cache_dir, model_filename)
    
    # === VERIFICA CACHE ===
    
    if os.path.exists(cached_path) and os.path.isfile(cached_path):
        logger.info(f"✓ Modello in cache: {cached_path}")
        logger.info(f"  Size: {os.path.getsize(cached_path) / (1024*1024):.1f} MB")
        state.model_path = cached_path
        
        # ✅ STAMPA ARCHITETTURA MODELLO - MODO ROBUSTO
        logger.info(f"\n📋 ANALISI ARCHITETTURA MODELLO (da cache)")
        logger.info("=" * 80)
        # ✅ ALGORITMO OTTIMIZZATO (RICHIESTA UTENTE)
        # 1. Legacy Env Subprocess (Primo tentativo)
        # 2. HDF5 Raw (Fallback)
        # 3. NO standard load_model()
        
        legacy_info = inspect_model_via_legacy_env(cached_path)
            
        if legacy_info:
            logger.info(f"✓ Analisi riuscita (via stm32_legacy)!")
            logger.info(f"  Input: {legacy_info.get('input_shape')}")
            logger.info(f"  Output: {legacy_info.get('output_shape')}")
            logger.info(f"  Params: {legacy_info.get('total_params'):,}")
            if 'model_size_mb' in legacy_info:
                logger.info(f"  Size: {legacy_info['model_size_mb']:.2f} MB")
            logger.info(f"  BN: {'Yes' if legacy_info.get('has_batchnorm') else 'No'} | Dropout: {'Yes' if legacy_info.get('has_dropout') else 'No'}")
            state.model_info = legacy_info
            state.model_architecture = legacy_info # Sync for workflow5 compatibility
        else:
            logger.warning(f"⚠️  Legacy subprocess fallito, provo fallback HDF5...")
            
        # ← SECONDO TENTATIVO: lettura raw HDF5 (più robusta, solo se .h5 o .keras)
        if cached_path.endswith(('.h5', '.keras')):
            try:
                with h5py.File(cached_path, 'r') as f:
                    logger.info(f"\n📋 ANALISI INTERNA (HDF5/Keras)")
                    logger.info(f"  Keys nel file: {list(f.keys())}")
                    
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
                logger.warning(f"⚠️  Analisi HDF5 fallita: {str(e2)[:100]}")
        else:
            logger.info(f"📋 Formato {os.path.splitext(cached_path)[1]} rilevato. Analisi strutturale saltata.")
        
        return state
    
    # === PRIORITY 1: URL Diretto ===
    
    direct_url = model.get("url")
    
    if direct_url:
        try:
            logger.info(f"📥 [1/2] Tentativo URL diretto: {direct_url[:80]}...")
            
            response = requests.get(direct_url, stream=True, timeout=30, allow_redirects=True)
            
            if response.status_code == 404:
                logger.warning(f"⚠️  URL restituisce 404")
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
                
                
                logger.info(f"✓ Download completato! Size: {os.path.getsize(cached_path) / (1024*1024):.1f} MB")
                
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
                
                # ✅ STAMPA ARCHITETTURA - MODO ROBUSTO (uguale a sopra)
                logger.info(f"\n📋 ANALISI ARCHITETTURA MODELLO (appena scaricato)")
                logger.info("=" * 80)
                # ✅ ALGORITMO OTTIMIZZATO (RICHIESTA UTENTE)
                # 1. Legacy Env Subprocess (Primo tentativo)
                # 2. HDF5 Raw (Fallback)
                
                legacy_info = inspect_model_via_legacy_env(cached_path)
                
                if legacy_info:
                    logger.info(f"✓ Analisi riuscita (via {legacy_info.get('env_used', 'unknown')})!")
                    logger.info(f"  Input: {legacy_info['input_shape']}")
                    state.model_info = legacy_info
                    state.model_architecture = legacy_info # Sync for workflow5 compatibility
                else:
                    logger.warning(f"⚠️  Legacy subprocess fallito, provo HDF5...")
                    try:
                        with h5py.File(cached_path, 'r') as f:
                            logger.info(f"  File contiene: {list(f.keys())}")
                            if 'model_weights' in f:
                                logger.info(f"  Peso layers disponibili")
                    except Exception as e2:
                        logger.warning(f"⚠️  Analisi HDF5 fallita: {str(e2)[:100]}")
                
                state.model_path = cached_path
                return state
            
        except Exception as e:
            logger.warning(f"⚠️  Download fallito: {type(e).__name__}")
            if os.path.exists(cached_path):
                os.remove(cached_path)
    
    # === PRIORITY 2: Task-Based Fallback ===
    
    logger.error(f"❌ Download fallito")
    cfg = Configuration.from_runnable_config(config)
    last_task = state.__dict__.get("last_task")
    
    if last_task:
        logger.info(f"🔄 Provo fallback task-based: {last_task}")
        fallback_model = get_task_based_default_model(last_task)
        
        if fallback_model:
            logger.info(f"✓ Fallback model: {fallback_model['name']}")
            fallback_url = fallback_model.get("url")
            
            if fallback_url:
                try:
                    logger.info(f"📥 Download fallback...")
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
                        
                        logger.info(f"✓ Fallback download completato!")
                        state.model_path = fallback_path
                        state.model_discovery_method = "taskbased_fallback"
                        state.selected_model = fallback_model
                        
                        return state
                
                except Exception as e:
                    logger.warning(f"⚠️  Fallback download fallito: {type(e).__name__}")
    
    logger.warning(f"⚠️  Tutti i fallback esauriti")
    state.model_path = cfg.ai_model_path
    state.model_discovery_method = "default"
    
    return state

# ============================================================================
# HELPER: GET DEFAULT MODEL BY TASK
# ============================================================================
def get_task_based_default_model(task_name: str) -> Optional[dict]:
    """
    Ritorna il primo modello disponibile per il task specifico.
    Fallback intelligente: se l'utente cercava "image_classification" 
    e la ricerca fallisce, usa il primo MobileNetV2 da PREDEFINED_MODELS
    """
    
    if task_name not in PREDEFINED_MODELS:
        logger.warning(f"⚠️  Task non trovato: {task_name}, uso generico")
        # Fallback al primo modello disponibile di qualsiasi task
        for task, info in PREDEFINED_MODELS.items():
            if info["models"]:
                return info["models"][0]
        return None
    
    task_info = PREDEFINED_MODELS[task_name]
    models = task_info.get("models", [])
    
    if not models:
        logger.warning(f"⚠️  Nessun modello per task: {task_name}")
        return None
    
    default_model = models[0]  # Prendi il primo (più leggero/veloce)
    logger.info(f"✓ Default model per '{task_name}': {default_model['name']}")
    
    return default_model


# ============================================================================
# ROUTING DECISION - DECIDE QUALE NODO USARE
# ============================================================================

def model_selection_routing(state: MasterState) -> Literal[
    "run_analyze", 
    "download_model", 
    "search_recommendation_model" 
]:
    """
    Router che decide il prossimo step dopo model selection.
    
    Ora supporta anche il ramo di CUSTOMIZZAZIONE.
    """
    
    logger.info(f"📍 model_selection_routing:")
    logger.info(f"   - discovery_method: {state.model_discovery_method}")
    logger.info(f"   - search_iterations: {state.search_iterations}")
    logger.info(f"   - wants_customization: {getattr(state, 'wants_customization', False)}")
    
    # ============================================================
    # CASE 1: Default model (nessuna ricerca)
    # ============================================================
    if state.model_discovery_method == "default":
        logger.info("→ Default model, vai direttamente ad analyze")
        return "run_analyze"
    
    # ============================================================
    # CASE 2: In ricerca, loop di ricerca ancora disponibile
    # ============================================================
    elif state.model_discovery_method == "search":
        if state.search_iterations < 3:
            logger.info(f"→ Loop ricerca ({state.search_iterations}/3), ricerca di nuovo")
            return "search_recommendation_model"
        else:
            logger.info("→ Max iterazioni ricerca raggiunte, vai ad analyze")
            return "run_analyze"
    
    # ============================================================
    # CASE 3: Modello trovato (github, google_search, taskbased_fallback)
    # ============================================================
    else:
            logger.info("→ Modello trovato, vai a download_model")
            return "download_model"


def run_analyze(state: MasterState, config: dict) -> MasterState:
    """
    ✨ Analizza il modello (customizzato O originale)
    
    Logica:
    - Se customizzato: final_model_path
    - Altrimenti: model_path (default)
    """
    
    logger.info("🔍 Eseguendo analisi del modello...")
    
    try:
        # ===== DETERMINA MODELLO =====
        # Prova finale prima, altrimenti usa originale
        model_path = state.final_model_path if state.customization_applied else state.model_path
        model_type = "CUSTOMIZZATO" if state.customization_applied else "ORIGINALE"
        
        if not model_path or not os.path.exists(model_path):
            logger.error(f"❌ Model not found: {model_path}")
            state.analyze_success = False
            state.ai_error_message = f"Model not found: {model_path}"
            return state
        
        # ✅ FIX PER .keras (Keras 3): Converti in TFLite se necessario
        # stedgeai v2.x ha bug con Keras 3 (es: Concatenate object has no attribute 'get_input_shape_at')
        if model_path.endswith('.keras'):
            logger.info("⚡ Rilevato modello Keras 3 (.keras). Avvio conversione TFLite per compatibilità stedgeai...")
            tflite_path = model_path.replace('.keras', '.tflite')
            
            if not os.path.exists(tflite_path): # Se il file è già presente, il sistema salta tutto il blocco di conversione. Significa che la conversione è già stata eseguita in precedenza.
                conversion_script = f"""
import tensorflow as tf
import os
try:
    model = tf.keras.models.load_model(r'{model_path}', compile=False)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(r'{tflite_path}', 'wb') as f:
        f.write(tflite_model)
    print("CONVERSION_OK")
except Exception as e:
    print(f"CONVERSION_ERROR:{{e}}")
"""
                python_path = CONDA_PYTHON_PATHS.get('stm32') # Usa env Keras 3
                res = execute_in_environment(conversion_script, python_path)
                
                if not res['success'] or "CONVERSION_OK" not in res['stdout']:
                    logger.error(f"❌ Conversione TFLite fallita: {res['stdout']} {res['stderr']}")
                    state.analyze_success = False
                    state.ai_error_message = f"TFLite conversion failed for Keras 3 model."
                    return state
                
                logger.info(f"✅ Conversione completata: {tflite_path}")
            
            # Usa il TFLite per l'analisi
            model_path = tflite_path
            state.model_path = tflite_path # Aggiorna lo stato così i nodi successivi lo usano
        
        logger.info(f"  Model ({model_type}): {model_path}")
        
        # ===== OUTPUT DIR =====
        analyze_dir = os.path.join(state.ai_output_dir, "report_analyze")
        os.makedirs(analyze_dir, exist_ok=True)
        
        # ===== ESEGUI =====
        cmd = [
            "stedgeai", "analyze",
            "--model", model_path,
            "--target", state.target,
            "--output", analyze_dir
        ]
        
        # ✅ FIX: Aggiungi compressione se specificata. 
        if state.compression: # fondamentale. X-CUBE-AI ha capacità di quantizzazione integrata. Se richiesto dall'utente, X-CUBE-AI usa questo parametro per applicare automaticamente tecniche di compressione/quantizzazione durante l'analisi e la generazione del codice C.
             cmd.extend(["--compression", state.compression])
             logger.info(f"  Compression: {state.compression}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        state.analyze_success = (result.returncode == 0)
        
        if state.analyze_success:
            logger.info(f"✓ Analyze completato")
            state.analyze_report_dir = analyze_dir
            
            # ✅ COMMIT REGISTRAZIONE: Salva il modello nel catalogo permanente solo se l'analisi tecnica ha avuto successo.
            # Questo evita di registrare link rotti o modelli non supportati dagli strumenti ST.
            if state.is_new_registration and state.pending_model_entry:
                try:
                    # Carica il catalogo attuale dal file JSON
                    models = load_predefined_models()
                    new_entry = state.pending_model_entry.copy()
                    
                    # Estrae la categoria (es: image_classification) e la rimuove dai dati del modello
                    category = new_entry.pop("category", "other")
                    
                    # Crea la categoria nel catalogo se non esiste ancora
                    if category not in models:
                        models[category] = {
                            "description": category.replace("_", " ").title(),
                            "models": []
                        }
                    
                    # Controllo anti-duplicati: salva solo se l'URL non è già presente nella categoria
                    if not any(m['url'] == new_entry['url'] for m in models[category]['models']):
                        models[category]["models"].append(new_entry)
                        save_predefined_models(models) # Scrittura fisica su disco (predefined_models.json)
                        logger.info(f"💾 Modello '{new_entry['name']}' salvato nel catalogo permanente.")
                    
                    # Reset dello stato: la registrazione è conclusa con successo
                    state.is_new_registration = False # Reset flag
                    state.pending_model_entry = None  # Pulisci
                    
                except Exception as ex:
                    logger.error(f"⚠️ Errore durante il salvataggio nel catalogo: {ex}")
        else:
            state.ai_error_message = result.stderr.strip() or f"Return code {result.returncode}"
            logger.error(f"✗ Analyze fallito: {state.ai_error_message[:500]}")
    
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        state.analyze_success = False
        state.ai_error_message = str(e)
    
    return state


def run_validate(state: MasterState, config: dict) -> MasterState:
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
    logger.info("✓ Validate completato" if state.validate_success else f"✗ Validate fallito: {state.ai_error_message}")
    return state


def run_generate(state: MasterState, config: dict) -> MasterState:
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
    logger.info("✓ Generate completato" if state.generate_success else f"✗ Generate fallito: {state.ai_error_message}")
    return state


def finalize_analysis(state: MasterState, config: dict) -> MasterState:
    if state.analyze_success and state.validate_success and state.generate_success:
        print("✓ Analisi AI completata!")
        print(f" - Report analyze in: {state.analyze_report_dir}")
        print(f" - Report validate in: {state.validate_report}")
        print(f" - Codice generato in: {state.generate_code_dir}")
    else:
        print(f"✗ Errore AI: {state.ai_error_message}")
    return state


# ============================================================================
# NEW RESOURCE CONSTRAINT CHECK LOGIC
# ============================================================================


def get_mcu_limits(target_mcu: str) -> tuple[int, int]:
    """
    Ritorna (flash_limit_bytes, ram_limit_bytes) per la MCU target.
    Valori approssimativi ma sicuri (conservativi).
    """
    target = target_mcu.lower()
    
    if "stm32f4" in target or "f401" in target:
        # STM32F401: 256KB Flash, 64KB RAM
        return (256 * 1024, 64 * 1024)
    elif "stm32h7" in target or "h743" in target:
        # STM32H743: 2MB Flash, ~1MB RAM (per attivazioni contigue safe)
        return (2 * 1024 * 1024, 1024 * 1024)
    elif "stm32u5" in target:
        # STM32U5: 2MB Flash, 786KB RAM
        return (2 * 1024 * 1024, 768 * 1024)
    elif "stm32l4" in target:
         # STM32L4: 1MB Flash, 128KB RAM
        return (1024 * 1024, 128 * 1024)
    else:
        # Default safe fallback (assumiamo F4)
        logger.warning(f"⚠️ Target MCU non riconosciuto: {target_mcu}. Uso limiti default (F4).")
        return (256 * 1024, 64 * 1024)


def check_resource_constraints(state: MasterState, config: dict) -> MasterState:
    """
    Analizza il report STEdgeAI per verificare se il modello ci sta.
    """
    logger.info("⚖️  Checking Resource Constraints...")
    
    if not state.analyze_success:
        logger.warning("⚠️  Analisi fallita, impossibile verificare constraints.")
        state.resource_check_result = "error"
        return state

    report_path = os.path.join(state.analyze_report_dir, "network_analyze_report.txt")
    if not os.path.exists(report_path):
        # Fallback: cerca qualsiasi file .txt nella dir
        try:
            files = [f for f in os.listdir(state.analyze_report_dir) if f.endswith(".txt")]
            if files:
                report_path = os.path.join(state.analyze_report_dir, files[0])
            else:
                logger.error("❌ Report file non trovato.")
                state.resource_check_result = "error"
                return state
        except Exception:
             logger.error("❌ Report dir non trovata.")
             state.resource_check_result = "error"
             return state

    # Parse Report
    ram_usage = 0
    flash_usage = 0
    
    try:
        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # Cerca pattern tipo: "activations  : 4917696 bytes" o "weights      : 8833768 bytes"
            # O pattern più complessi a seconda versioni. Cerchiamo "activations" e "weights" / "total"
            
            # Esempio report:
            #  activations size   : 4917696 bytes (4802.44 KiB)
            #  weights size       : 8833768 bytes (8626.73 KiB)
            #  macc               : ...
            
            ram_match = re.search(r'(?i)activations\s*(?:size)?\s*:\s*(\d+)', content)
            if ram_match:
                ram_usage = int(ram_match.group(1))
            
            flash_match = re.search(r'(?i)weights\s*(?:size)?\s*:\s*(\d+)', content)
            if flash_match:
                flash_usage = int(flash_match.group(1))

            # Se 0, prova pattern alternativi (totale ram/flash report table)
            if ram_usage == 0:
                 ram_match = re.search(r'(?i)ram\s*:\s*(\d+)', content)
                 if ram_match: ram_usage = int(ram_match.group(1))

            if flash_usage == 0:
                 flash_match = re.search(r'(?i)flash\s*:\s*(\d+)', content)
                 if flash_match: flash_usage = int(flash_match.group(1))

    except Exception as e:
        logger.error(f"❌ Errore parsing report: {e}")
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

# Logica Intelligente:
# -Fits: Procedi.
# -Warning (<4x overflow): Attiva compression='high' e riprova.
# -Critical (>4x overflow): Blocca tutto e chiede di scegliere un modello più piccolo (es. ResNet -> MobileNet)


def resource_check_routing(state: MasterState) -> Literal["run_analyze", "run_validate", "run_generate", "choose_predefined_taskbased_model"]:
    """
    Decide la route basata sui constraints.
    Gestisce anche il retry automatico con compressione più alta.
    """
    res = getattr(state, "resource_check_result", "ok")
    
    # ✅ NEW: Check if we need to retry with higher compression
    if state.needs_compression_retry and res == "retry":
        logger.info(f"🔄 Routing back to analyze with compression: {state.compression}")
        return "run_analyze"  # Re-analyze with new compression level
    
    if res == "ok":
        return "run_validate"
    
    elif res == "warning":
        return "run_generate"
        
    else: # critical or error
        # Notifica utente e torna alla scelta
        logger.error("🚫 Model rejected due to hardware constraints.")
        
        # LOG ONLY - NO INTERRUPT
        logger.error(f"""⛔ MODELLO TROPPO GRANDE PER {state.target}!
            
Dettagli Risorse:
- RAM Richiesta: {format_bytes(state.ram_usage)} (Max: {format_bytes(get_mcu_limits(state.target)[1])})
- Flash Richiesta: {format_bytes(state.flash_usage)} (Max: {format_bytes(get_mcu_limits(state.target)[0])})

L'automazione torna alla selezione modello forzando una scelta più appropriata.""")
        
        return "handle_resource_failure"
        # return "run_generate" # per alcuni test fatti la utilizzavo per forzare l'integrazione 


def handle_resource_failure(state: MasterState, config: dict) -> MasterState:
    """
    Chiede all'utente se vuole cambiare board o modello dopo un fallimento di risorse.
    """
    logger.info("📋 Decisione post-errore risorse: Cambio Board o Cambio Modello?")
    
    prompt = {
        "instruction": "Il modello è troppo grande per l'attuale MCU. Cosa vuoi fare?",
        "options": [
            "Cambia Microcontrollore (Board)",
            "Scegli un altro modello AI"
        ]
    }
    
    user_response = interrupt(prompt)
    
    if isinstance(user_response, dict):
        user_text = user_response.get("response", user_response.get("input", str(user_response)))
    else:
        user_text = str(user_response)
        
    cfg = Configuration.from_runnable_config(config)
    llm = ChatOllama(
        model=cfg.local_llm,
        temperature=0,
        num_ctx=cfg.llm_context_window
    )
    
    # === ESTRAI DECISIONE CON LLM (Robusto con Structured Output) ===
    llm_extractor = llm.with_structured_output(ResolutionExtraction)
    
    analysis_prompt = f"""Analizza la risposta dell'utente e determina l'azione da intraprendere.
L'utente ha visto queste opzioni:
0. Cambia Microcontrollore (Board)
1. Scegli un altro modello AI

Risposta utente: "{user_text}"

MAPPING:
- "0" o "board" o "scheda" -> change_board
- "1" o "modello" o "model" o "scelta" -> change_model

Rispondi con un JSON che contiene:
- "decision": "change_board" o "change_model"
- "confidence": 0.0-1.0
"""
    
    try:
        result = llm_extractor.invoke([
            SystemMessage(content="Sei un classificatore di intenti tecnico."),
            HumanMessage(content=analysis_prompt)
        ])
        
        decision = result.decision.lower()
        logger.info(f"🤖 Decisione LLM: {decision} (confidence: {result.confidence:.2f})")
    except Exception as e:
        logger.error(f"⚠️ Errore estrazione decisione: {e}. Fallback su change_model.")
        decision = "change_model"
    
    if "board" in decision:
        state.route = "change_board"
    else:
        state.route = "change_model"
        
    return state

def add_custom_model_procedure(state: MasterState, config: dict) -> MasterState:
    """
    Procedura per aggiungere un nuovo modello al catalogo.
    """
    logger.info("🆕 Inizio procedura registrazione nuovo modello...")
    
    # 1. Chiedi i dettagli all'utente
    prompt = {
        "instruction": """Registrazione Nuovo Modello AI

Fornisci i seguenti dettagli Separati da virgola:
1. Categoria (es: image_classification, object_detection, audio)
2. Nome Modello (es: MobileNetV3 Small)
3. Link GitHub (URL Raw .h5, .onnx, .tflite, .keras)

Esempio:
"image_classification, MobileNetV3, https://github.com/.../model.keras"
        """
    }
    
    user_response = interrupt(prompt)
    if isinstance(user_response, dict):
        user_text = user_response.get("response", user_response.get("input", str(user_response)))
    else:
        user_text = str(user_response)
        
    # 2. Parsing con LLM
    cfg = Configuration.from_runnable_config(config)
    llm = ChatOllama(model=cfg.local_llm, temperature=0)
    
    extraction_prompt = f"""Estrai i dettagli del nuovo modello dalla seguente risposta dell'utente:
"{user_text}"

Rispondi in formato JSON con questi campi:
- "category": categoria (in minuscolo, snake_case)
- "name": nome del modello
- "url": link GitHub Raw completo
- "is_valid": true se i dati sembrano sensati
"""
    
    response = llm.invoke(extraction_prompt)
    try:
        # Pulisci risposta se LLM mette markdown
        clean_content = response.content.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean_content)
    except Exception as e:
        logger.error(f"❌ Errore parsing dati nuovo modello: {e}")
        return state

    if not data.get("is_valid") or not data.get("url"):
        logger.error("❌ Dati modello non validi o URL mancante.")
        return state

    # 3. Validazione URL e Metadati
    url = data["url"]
    logger.info(f"🔍 Validando URL: {url}")
    
    try:
        res = requests.head(url, timeout=5, allow_redirects=True)
        if res.status_code == 200:
            size_bytes = int(res.headers.get('Content-Length', 0))
            size_str = format_bytes(size_bytes) if size_bytes > 0 else "N/A"
        else:
            logger.warning(f"⚠️ URL risponde con status {res.status_code}. Procedo comunque?")
            size_str = "N/A"
    except Exception as e:
        logger.warning(f"⚠️ Errore connessione URL: {e}")
        size_str = "N/A"

    # 4. Aggiorna Registro
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
        "category": category # Temporaneo per salvarlo dopo
    }
    
    logger.info(f"⏳ Modello '{data['name']}' in attesa di validazione tecnica...")
    
    # Imposta il nuovo modello come selezionato per procedere subito
    state.selected_model = new_entry
    state.pending_model_entry = new_entry
    state.is_new_registration = True
    
    state.model_path = "" # Verrà scaricato nel nodo download_model
    state.model_discovery_method = "default" # Fai finta che sia predefinito ora
    
    return state
