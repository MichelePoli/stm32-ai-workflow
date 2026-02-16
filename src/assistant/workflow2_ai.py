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

model_selection_instructions = """Analizza la risposta dell'utente sulla selezione del modello specifico.
L'utente sta scegliendo un modello da una lista numerata.

REGOLE CRITICAL:
1. Se la risposta è un numero (es. "1", "2"), mappa rigorosamente verso quell'indice (model_index).
2. "model_accepted" deve essere true se l'utente sceglie un modello dalla lista.
3. Se l'utente rifiuta tutti o scrive "no" / "nessuno", imposta wants_another_search: true e model_accepted: false.
4. Se l'utente vuole usare un default o non sa, imposta model_accepted: false e wants_another_search: false.

Output JSON:
- "model_index": int (1-based) o null
- "model_accepted": boolean
- "wants_another_search": boolean
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
def collect_analysis_info(state: MasterState, config: RunnableConfig = None) -> MasterState:
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
    
    # === IDEMPOTENCY CHECK ===
    # Se abbiamo già target e compression (es: iniettati da config o resume), saltiamo.
    # CRITICO: Non saltare se il target è quello di default ma la board è diversa!
    board_target = None
    if state.board_name:
        # Mappa semplice per board note o estrazione serie
        b_low = state.board_name.lower()
        targets = ["f0", "f1", "f2", "f3", "f4", "f7", "h5", "h7", "l0", "l1", "l4", "l5", "u5", "g0", "g4", "w5", "c0", "n6"]
        for t in targets:
            if t in b_low:
                board_target = f"stm32{t}"
                break
    
    # Se la board attuale suggerisce un target diverso da quello salvato, NON saltare
    if state.target and state.compression and not state.user_response:
        if board_target and board_target != state.target:
            logger.info(f"🔄 Reset target AI per allineamento board: {state.target} -> {board_target}")
            state.target = board_target
        else:
            logger.info(f"⏭️  Idempotenza: Target '{state.target}' e Compression '{state.compression}' già presenti. Salto interrupt.")
            return state

    from src.assistant.utils import extract_user_response, get_llm
    
    # --- Passo 1: Prova a usare il messaggio iniziale ---
    # Cerchiamo se l'utente ha già specificato una board/target nel comando di avvio
    initial_target = None
    if not state.user_response:
        # Analisi euristica veloce del messaggio iniziale
        msg_low = state.message.lower()
        targets = ["f0", "f1", "f2", "f3", "f4", "f7", "h5", "h7", "l0", "l1", "l4", "l5", "u5", "g0", "g4", "w5", "c0", "n6"]
        for t in targets:
            if t in msg_low:
                initial_target = f"stm32{t}"
                break
    
    # --- Passo 2: Verifica e Interrupt ---
    # Forza interruzione se l'intento non è cristallino nel primo messaggio
    if not initial_target:
        resume_value = None
        if not state.user_response:
            # Suggerimento dal profilo
            last_series = state.persistent_context.get("mcu_series", "F4") if state.persistent_context else "F4"
            dynamic_prompt = {
                "instruction": prompt["instruction"],
                "suggestion": f"💡 L'ultima volta hai lavorato su serie **{last_series}**. Vuoi continuare con questa o cambiare?"
            }
            logger.info("⏸️ Interrupting for AI analysis config with profile suggestion.")
            resume_value = interrupt(dynamic_prompt)
        
        # Dopo la ripresa: usa interrupt return value come priorità
        if resume_value and str(resume_value).strip():
            user_text = str(resume_value).strip()
        else:
            user_text = extract_user_response(state.user_response)
        state.user_response = ""
    else:
        # Abbiamo già il target dal messaggio iniziale
        user_text = state.message
        logger.info(f"✓ Target '{initial_target}' rilevato nel messaggio iniziale.")

    # --- Passo 3: Eredità e Parsing ---
    if not user_text or user_text.strip() == "" or "precedente" in user_text.lower() or "quella di" in user_text.lower() or "profilo" in user_text.lower():
        # Recupera mcu_series dallo stato corrente O dalla memoria persistente
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
            logger.info(f"📋 Applicata configurazione da profilo: {target_mcu}")
        else:
            user_text = "STM32F4, high compression"
    
    logger.info(f"📝 User input RAW: '{user_text}'")
    
    # === ESTRAI TARGET E COMPRESSION ===
    
    # === ESTRAI TARGET E COMPRESSION ===
    
    llm_extractor = get_llm(
        config=config,
        structured_schema=AnalysisInfoExtraction,
        temperature=0
    )
    
    extraction_result = llm_extractor.invoke([
        SystemMessage(content=analysis_info_extraction_instructions),
        HumanMessage(content=f"Risposta utente: {user_text}")
    ])
    
    state.target = extraction_result.target or state.target or "stm32h743"
    state.compression = extraction_result.compression or state.compression or "high"
    state.ai_output_dir = cfg.ai_output_dir
    
    os.makedirs(state.ai_output_dir, exist_ok=True)
    
    logger.info(f"✓ Configurazione estratta:")
    logger.info(f"  Target: {state.target}")
    logger.info(f"  Compression: {state.compression}")
    
    return state


# ============================================================================
# NODO: SCEGLI DA MODELLI PREDEFINITI (TASK-BASED)
# ============================================================================
def choose_predefined_taskbased_model(state: MasterState, config: RunnableConfig = None) -> MasterState:
    """
    Mostra modelli predefiniti con parsing LLM.
    Salva il task per fallback intelligente.
    Usa PREDEFINED_MODELS come unica fonte.
    """
    
def choose_ai_task(state: MasterState, config: RunnableConfig = None) -> MasterState:
    """
    Nodo 1: Sceglie il TASK (es. Classificazione immagini)
    Gestisce l'interrupt per il menu principale.
    """
    logger.info("📋 Scelta Task AI...")
    
    # Ricarica modelli dal JSON
    global PREDEFINED_MODELS
    PREDEFINED_MODELS = load_predefined_models()
    
    cfg = Configuration.from_runnable_config(config)
    cfg = Configuration.from_runnable_config(config)
    llm = get_llm(config=config, temperature=0)
    
    categories = list(PREDEFINED_MODELS.keys())
    prompt_lines = ["--- MODELLI PREDEFINITI (Ottimizzati e Garantiti) ---", "Seleziona una categoria per vedere i modelli pronti all'uso:\n"]
    mapping = {}
    for i, cat in enumerate(categories, 1):
        desc = PREDEFINED_MODELS[cat].get("description", cat)
        prompt_lines.append(f"{i}. {desc}")
        mapping[str(i)] = cat
    
    prompt_lines.append("\n--- ALTRE OPZIONI ---")
    reg_idx = len(categories) + 1
    prompt_lines.append(f"{reg_idx}. Registra un TUO modello locale (già presente sul disco)")
    mapping[str(reg_idx)] = "register_new"
    
    other_idx = reg_idx + 1
    prompt_lines.append(f"{other_idx}. Ricerca ONLINE (Cerca nuovi modelli su GitHub/Google)")
    mapping[str(other_idx)] = "other"
    
    prompt_text = "\n".join(prompt_lines) + f"\n\nRispondi con il numero (1-{other_idx}) o descrivi cosa vuoi fare."
    
    # === IDEMPOTENCY & INTERRUPT ===
    if state.last_task and state.last_task != "other" and not state.user_response:
        logger.info(f"⏭️  Idempotenza: Task '{state.last_task}' già selezionato.")
        return state

    resume_value = None
    if not state.user_response or state.user_response.strip() == "":
        # logger.info("⏸️ Interrupting for AI task selection.")
        # resume_value = interrupt({"instruction": prompt_text})
        logger.info("⏭️  BYPASS: Selezione automatica task -> '1' (Classificazione)")
        user_text = "1"
    else:
        # Usa interrupt return value come priorità
        if resume_value and str(resume_value).strip():
            user_text = str(resume_value).strip()
        else:
            user_text = extract_user_response(state.user_response).strip()
    state.user_response = "" # Clear after use
    if not user_text: user_text = "1"
    
    logger.info(f"📥 Risposta utente ricevuta: '{user_text}'")
    
    # === ESTRAI TASK CON LLM ===
    mapping_text = "\n".join([f'- "{k}" -> {v}' for k, v in mapping.items()])
    dynamic_instructions = f"""Analizza la risposta dell'utente e determina il task AI richiesto.
Usa rigorosamente il mapping numerico se l'utente ha inserito un numero.

Menu visualizzato all'utente:
{prompt_text}

MAPPING ESPLICITO:
{mapping_text}

REGOLE:
1. Se l'utente risponde con un numero presente nel mapping, ritorna il task corrispondente.
2. Se l'utente descrive un'azione, mappa verso la categoria più vicina.
3. Se l'utente vuole qualcosa di non presente o una ricerca, usa "other".
4. Il valore di "confidence" deve essere 1.0 per match numerici esatti.

Rispondi in formato JSON: {{"task": "...", "confidence": 0.0-1.0}}
"""

    llm_extractor = get_llm(
        config=config,
        structured_schema=TaskSelectionExtraction,
        temperature=0
    )
    task_result = llm_extractor.invoke([
        SystemMessage(content=dynamic_instructions),
        HumanMessage(content=f"Risposta utente: {user_text}")
    ])
    
    logger.info(f"🤖 LLM Extraction: task='{task_result.task}', confidence={task_result.confidence}")
    
    state.last_task = task_result.task
    logger.info(f"✓ Task selezionato: {state.last_task}")
    
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
    Nodo 2: Sceglie il MODELLO specifico dal catalogo del task.
    Gestisce l'interrupt per la lista modelli.
    """
    if state.model_discovery_method != "taskbased":
        return state

    logger.info(f"📋 Scelta Modello per task '{state.last_task}'...")
    
    # === IDEMPOTENCY CHECK ===
    if state.selected_model and not state.user_response:
        logger.info(f"⏭️  Idempotenza: Modello '{state.selected_model['name']}' già selezionato.")
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
    prompt_text = f"Quale modello vuoi usare per {task_info['description']}?\n\nOpzioni disponibili:\n{models_list}\n{len(available_models)+1}. Nessuno di questi (ricerca online)\n\nRispondi con il numero."
    
    resume_value = None
    if not state.user_response or state.user_response.strip() == "":
        # logger.info("⏸️ Interrupting for AI model selection.")
        # resume_value = interrupt({"instruction": prompt_text})
        logger.info("⏭️  BYPASS: Selezione automatica modello -> '2' (MobileNetV1)")
        model_text = "2"
    else:
        # Usa interrupt return value come priorità
        if resume_value and str(resume_value).strip():
            model_text = str(resume_value).strip()
        else:
            model_text = extract_user_response(state.user_response).strip()
    state.user_response = "" # Clear after use
    
    # === ESTRAZIONE SCELTA ===
    cfg = Configuration.from_runnable_config(config)
    llm_model_extractor = get_llm(
        config=config,
        structured_schema=ModelSelectionExtraction,
        temperature=0
    )
    
    logger.info(f"📥 Risposta utente per modello: '{model_text}'")
    
    model_result = llm_model_extractor.invoke([
        SystemMessage(content=model_selection_instructions),
        HumanMessage(content=f"Modelli disponibili: {len(available_models)}\nRisposta utente: {model_text}")
    ])
    
    logger.info(f"🤖 LLM Model Extraction: index={model_result.model_index}, accepted={model_result.model_accepted}, search_again={model_result.wants_another_search}")
    
    if model_result.model_accepted and model_result.model_index:
        model_idx = model_result.model_index - 1
        if 0 <= model_idx < len(available_models):
            state.selected_model = available_models[model_idx]
            state.model_accepted = True
            logger.info(f"✓ Modello scelto: {state.selected_model['name']}")
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
# NODO PRINCIPALE per la ricerca modelli !
# ============================================================================
def search_recommendation_model(state: MasterState, config: RunnableConfig = None) -> MasterState:
    """
    ✅ NODO PRINCIPALE: Ricerca modello con fallback intelligente
    
    TYPE HINTS: state: MasterState, config: RunnableConfig → MasterState
    
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
    config: RunnableConfig = None,
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
    config: RunnableConfig = None
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
    config: RunnableConfig
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
            tools=[DuckDuckGoTools()],
            instructions=[
                "Ricerca file .h5 per STM32",
                "Link GitHub /raw/ diretti",
                "Non inventare URL"
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
    'mobilenet': 'stm32legacy',
    'resnet': 'stm32legacy',
    'vgg': 'stm32legacy',
    'efficientnet': 'stm32legacy',
    'inception': 'stm32legacy',
    'yolo': 'stm32legacy',
    'har': 'stm32legacy',
    'custom': 'stm32legacy',
}

# CONDA_PYTHON_PATHS rimosso in favore di config.get_python_path()

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

def inspect_model_via_legacy_env(model_path: str, config: RunnableConfig = None) -> Optional[dict]:
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
        
        cfg = Configuration.from_runnable_config(config)
        
        # Scegli environment: .keras -> stm32 (Keras 3), .h5 -> stm32legacy (Keras 2) o stm32
        if model_path.endswith('.keras'):
            env_name = 'stm32'
        else:
            env_name = ARCH_ENVIRONMENT_MAP.get(arch, 'stm32legacy')
        
        python_path = cfg.get_python_path(env_name)
        
        if not python_path or "NOT_FOUND" in python_path:
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
def download_model(state: MasterState, config: RunnableConfig = None) -> MasterState:
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

def download_model_to_cache(state: MasterState, config: RunnableConfig, model: dict) -> MasterState:
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
        
        legacy_info = inspect_model_via_legacy_env(cached_path, config)
            
        if legacy_info:
            logger.info(f"✓ Analisi riuscita (via stm32legacy)!")
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
                logger.warning(f"⚠️  URL restituisce 404 (Not Found)")
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
                
                # ✅ VERIFICA POST-DOWNLOAD
                actual_size = os.path.getsize(cached_path)
                if actual_size == 0:
                    logger.error(f"❌ Download fallito: Il file salvato è vuoto (0 bytes)!")
                    if os.path.exists(cached_path): os.remove(cached_path)
                    return None
                    
                logger.info(f"✓ Download completato! Size: {actual_size / (1024*1024):.1f} MB")
                
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
                
                legacy_info = inspect_model_via_legacy_env(cached_path, config)
                
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


def run_analyze(state: MasterState, config: RunnableConfig = None) -> MasterState:
    """
    ✨ Analizza il modello (customizzato O originale)
    
    Logica:
    - Se customizzato: final_model_path
    - Altrimenti: model_path (default)
    """
    
    logger.info("🔍 Eseguendo analisi del modello...")
    cfg = Configuration.from_runnable_config(config)
    
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
        
        # ✅ FIX PER Keras 3 (Ambiente 'stm32'): Converti in TFLite per compatibilità stedgeai
        # stedgeai v2.x non supporta direttamente i modelli Keras 3 (anche se salvati come .h5)
        # Rileviamo Keras 3 se il file è .keras O se sappiamo di essere in ambiente 'stm32'
        is_keras3 = model_path.endswith('.keras') or state.conda_env == 'stm32'
        
        if is_keras3:
            logger.info("⚡ Rilevato modello Keras 3. Avvio conversione TFLite per compatibilità stedgeai...")
            tflite_path = model_path.replace('.keras', '.tflite').replace('.h5', '.tflite') # La funzione .replace() viene chiamata due volte di seguito. 
            # Primo passaggio: Cerca .keras e lo sostituisce con .tflite.
            # Secondo passaggio: Prende il risultato del primo e cerca .h5, sostituendolo con .tflite. 
            # Questo garantisce che il file finale abbia estensione .tflite indipendentemente dal formato originale (.keras o .h5).
            
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
                python_path = cfg.get_python_path('stm32') # Usa env Keras 3
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
    logger.info("✓ Validate completato" if state.validate_success else f"✗ Validate fallito: {state.ai_error_message}")
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
    logger.info("✓ Generate completato" if state.generate_success else f"✗ Generate fallito: {state.ai_error_message}")
    return state


def finalize_analysis(state: MasterState, config: RunnableConfig = None) -> MasterState:
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


def check_resource_constraints(state: MasterState, config: RunnableConfig = None) -> MasterState:
    """
    Analizza il report STEdgeAI per verificare se il modello ci sta.
    """
    logger.info("⚖️  Checking Resource Constraints...")
    
    if not state.analyze_success:
        logger.warning("⚠️  Analisi fallita, impossibile verificare constraints.")
        state.ai_error_message = (
            "Impossibile analizzare il modello con gli strumenti ST.\n"
            "Questo solitamente accade per modelli non supportati o errori di conversione.\n"
            "L'automazione tornerà alla selezione modello per permetterti di sceglierne un altro."
        )
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
        if not getattr(state, "analyze_success", True):
             logger.error(f"❌ Errore Tecnico durante l'analisi: {getattr(state, 'ai_error_message', 'Sconosciuto')}")
        else:
            logger.error("🚫 Model rejected due to hardware constraints.")
            
            # LOG ONLY - NO INTERRUPT
            logger.error(f"""⛔ MODELLO TROPPO GRANDE PER {state.target}!
                
Dettagli Risorse:
- RAM Richiesta: {format_bytes(state.ram_usage)} (Max: {format_bytes(get_mcu_limits(state.target)[1])})
- Flash Richiesta: {format_bytes(state.flash_usage)} (Max: {format_bytes(get_mcu_limits(state.target)[0])})

L'automazione torna alla selezione modello forzando una scelta più appropriata.""")
        
        return "handle_resource_failure"
        # return "run_generate" # per alcuni test fatti la utilizzavo per forzare l'integrazione 


def handle_resource_failure(state: MasterState, config: RunnableConfig = None) -> MasterState:
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
        # Reset board state to force new selection in collect_project_info
        state.board_name = None
        state.mcu_series = ""
        logger.info("🧹 Reset board state per cambio microcontrollore.")
    else:
        state.route = "change_model"
        # Reset AI state to force new selection
        state.last_task = None
        state.selected_model = None
        state.model_discovery_method = "taskbased"
        state.model_accepted = False
        state.search_iterations = 0
        logger.info("🧹 Reset AI selection state per cambio modello.")
        
    return state

def add_custom_model_procedure(state: MasterState, config: RunnableConfig = None) -> MasterState:
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
