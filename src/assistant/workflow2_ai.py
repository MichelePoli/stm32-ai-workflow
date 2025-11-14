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

from typing import Optional, Literal, List
from datetime import datetime

from tensorflow.keras.models import Model, load_model, model_from_json

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from src.assistant.configuration import Configuration
from src.assistant.state import MasterState

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
        description="Task selezionato: image_classification, object_detection, human_activity_recognition, other"
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


class SearchResultExtraction(BaseModel):
    """Estrae SOLO modelli .h5 scaricabili"""
    model_name: str = Field(description="Nome modello (es: MobileNetV2 128)")
    download_url: Optional[str] = Field(
        default=None,
        description="URL diretto a file .h5"
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

task_selection_instructions = """Analizza la risposta dell'utente e classifica quale task vuole fare:

1. image_classification: Classificazione immagini (immagini → categoria)
2. object_detection: Rilevamento oggetti (immagini → posizione oggetti)
3. human_activity_recognition: Riconoscimento delle attività umane (dati → attività)
4. other: Nessuno di questi / ricerca online

Esempi:
- "Classificazione" → image_classification
- "Oggetti" → object_detection
- "Attività" → human_activity_recognition
- "1" → image_classification
- "Non so" → other

Rispondi SEMPRE in formato JSON con:
- "task": uno tra image_classification, object_detection, human_activity_recognition, other
- "confidence": 0.0-1.0 (quanto sei sicuro)
"""

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
2. **download_url**: L'URL per scaricare il file .h5 (estrarre dalle parentesi tonde se Markdown)
3. **model_size**: La dimensione del file (es: 5.7MB)
4. **accuracy**: L'accuratezza del modello (es: 64%)
5. **inference_time**: Il tempo di inferenza (es: 40ms (STM32H7))

IMPORTANTE: Se vedi [testo](https://...) estrai l'URL dalle parentesi tonde (il secondo)

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

PREDEFINED_MODELS = {
    "image_classification": {
        "description": "Classificazione immagini",
        "models": [
            {
                "name": "MobileNetV2",
                "local_filename": "mobilenetv2_224.h5",
                "size": "8.4MB",
                "accuracy": "71%",
                "inference_time": "50ms (STM32H7)",
                "huggingface_repo": "STMicroelectronics/mobilenetv2",
                "huggingface_filename": "mobilenetv2_224.h5",
                "url": "https://github.com/STMicroelectronics/stm32ai-modelzoo/raw/main/image_classification/mobilenetv2/Public_pretrainedmodel_public_dataset/ImageNet/mobilenet_v2_1.0_224/mobilenet_v2_1.0_224.h5", # volendo si può aggiungere l'url per huggingface: "huggingface_url": "..."
            },
            {
                "name": "MobileNetV1",
                "local_filename": "mobilenetv1_128.h5",
                "size": "3.6MB",
                "accuracy": "65%",
                "inference_time": "30ms (STM32F4)",
                "huggingface_repo": "STMicroelectronics/mobilenetv1",
                "huggingface_filename": "mobilenetv1_128.h5",
                "url": "https://github.com/STMicroelectronics/stm32ai-modelzoo/raw/main/image_classification/mobilenetv1/Public_pretrainedmodel_public_dataset/ImageNet/mobilenet_v1_0.25_224/mobilenet_v1_0.25_224.h5",
            },
            {
                "name": "EfficientNetV2B0",
                "local_filename": "efficientnet_v2B0_224.h5",
                "size": "7.1MB",
                "accuracy": "80%",
                "inference_time": "140ms (STM32H7)",
                "huggingface_repo": "STMicroelectronics/efficientnetv2",
                "huggingface_filename": "efficientnet_v2B0_224.h5",
                "url": "https://github.com/STMicroelectronics/stm32ai-modelzoo/raw/main/image_classification/efficientnetv2/Public_pretrainedmodel_public_dataset/ImageNet/efficientnet_v2B0_224/efficientnet_v2B0_224.h5"
            }

        ]
    },
    "object_detection": {
        "description": "Rilevamento oggetti",
        "models": [
            {
                "name": "Tiny YOLOv2",
                "local_filename": "yolov2_tiny_416.h5",
                "size": "19.2MB",
                "accuracy": "35 mAP",
                "inference_time": "180ms (STM32H7)",
                "huggingface_repo": "STMicroelectronics/yolov2-tiny",
                "huggingface_filename": "yolov2_tiny_416.h5",
                "url": "https://github.com/STMicroelectronics/stm32ai-modelzoo/raw/main/object_detection/tiny_yolo_v2/ST_pretrainedmodel_public_dataset/coco_2017_person/tiny_yolo_v2_416/tiny_yolo_v2_416.h5"
            },
            {
                "name": "ST SSD MobileNet v1",
                "local_filename": "st_ssd_mobilenet_v1_256.h5",
                "size": "24.5MB",
                "accuracy": "35 mAP",
                "inference_time": "220ms (STM32H7)",
                "huggingface_repo": "STMicroelectronics/st-ssd-mobilenet-v1",
                "huggingface_filename": "st_ssd_mobilenet_v1_256.h5",
                "url": "https://github.com/STMicroelectronics/stm32ai-modelzoo/raw/main/object_detection/st_ssd_mobilenet_v1/ST_pretrainedmodel_public_dataset/coco_2017_person/st_ssd_mobilenet_v1_025_256/st_ssd_mobilenet_v1_025_256.h5"
            }
        ]
    },
    "Human Activity Recognition": {
        "description": "Human Activity Recognition",
        "models": [
            {
                "name": "GMP_WL (24)",
                "local_filename": "gmp_wl_24.h5",
                "size": "2.1MB",
                "accuracy": "95%",
                "inference_time": "20ms (STM32F4)",
                "huggingface_repo": "STMicroelectronics/har-wisdm",
                "huggingface_filename": "gmp_wl_24.h5",
                "url": "https://github.com/STMicroelectronics/stm32ai-modelzoo/raw/main/human_activity_recognition/gmp/ST_pretrainedmodel_public_dataset/WISDM/gmp_wl_24/gmp_wl_24.h5"
            },
            {
                "name": "GMP_WL (48)",
                "local_filename": "gmp_wl_48.h5",
                "size": "3.8MB",
                "accuracy": "96%",
                "inference_time": "25ms (STM32H7)",
                "huggingface_repo": "STMicroelectronics/har-wisdm",
                "huggingface_filename": "gmp_wl_48.h5",
                "url": "https://github.com/STMicroelectronics/stm32ai-modelzoo/raw/main/human_activity_recognition/gmp/ST_pretrainedmodel_public_dataset/WISDM/gmp_wl_48/gmp_wl_48.h5"
            }
        ]
    }


}
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
    
    user_response = interrupt(prompt)
    
    if isinstance(user_response, dict):
        user_text = user_response.get("response", user_response.get("input", str(user_response)))
    else:
        user_text = str(user_response)
    
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
    
    cfg = Configuration.from_runnable_config(config)
    llm = ChatOllama(
        model=cfg.local_llm,
        temperature=0,
        num_ctx=cfg.llm_context_window
    )
    
    # === STEP 1: CHIEDI TASK ===
    
    prompt = {
        "instruction": """Seleziona il task che vuoi fare:

1. Classificazione immagini (immagini → categoria)
2. Rilevamento oggetti (immagini → posizione oggetti)
3. Human Activity Recognition (sensori → attività fisica)
4. Nessuno di questi (ricerca online)

Rispondi: 1, 2, 3, 4 oppure descrivi il task
        """
    }
    
    user_response = interrupt(prompt)
    
    if isinstance(user_response, dict):
        user_text = user_response.get("response", user_response.get("input", str(user_response)))
    else:
        user_text = str(user_response).strip()
    
    logger.info(f"📝 User task input: '{user_text}'")
    
    # === ESTRAI TASK CON LLM ===
    
    llm_extractor = llm.with_structured_output(TaskSelectionExtraction)
    
    task_result = llm_extractor.invoke([
        SystemMessage(content=task_selection_instructions),
        HumanMessage(content=f"Risposta utente: {user_text}")
    ])
    
    logger.info(f"✓ Task estratto: {task_result.task} (confidence: {task_result.confidence:.2f})")
    
    selected_task = task_result.task
    
    # ✅ SALVA IL TASK NELLO STATE PER FALLBACK INTELLIGENTE
    state.last_task = selected_task
    logger.info(f"✓ Task salvato per fallback: {selected_task}")
    
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
    
    # === STEP 3: MOSTRA MODELLI ===
    
    print("\n" + "="*70)
    print(f"📦 MODELLI DISPONIBILI: {task_info['description']}")
    print("="*70)
    
    for i, model in enumerate(available_models, 1):
        print(f"\n{i}. {model['name']}")
        print(f"   📏 Dimensione: {model['size']}")
        print(f"   🎯 Accuratezza: {model['accuracy']}")
        print(f"   ⚡ Inferenza: {model['inference_time']}")
    
    print(f"\n{len(available_models)+1}. Nessuno di questi (ricerca online)")
    print("="*70 + "\n")
    
    # === STEP 4: CHIEDI MODELLO ===
    
    # Crea lista modelli per il prompt
    models_list = "\n".join([
        f"{i}. {model['name']} ({model['size']}, {model['accuracy']})"
        for i, model in enumerate(available_models, 1)
    ])
    
    model_prompt = {
        "instruction": f"""Quale modello vuoi usare per {task_info['description']}?

Opzioni disponibili:
{models_list}
{len(available_models)+1}. Nessuno di questi (ricerca online)

Rispondi con: numero (1-{len(available_models)+1}) oppure descrivi
        """
    }
    
    model_response = interrupt(model_prompt)
    
    if isinstance(model_response, dict):
        model_text = model_response.get("response", model_response.get("input", str(model_response)))
    else:
        model_text = str(model_response).strip()
    
    logger.info(f"📝 User model input: '{model_text}'")
    
    # === ESTRAI SCELTA MODELLO CON LLM ===
    
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
            
            # ✅ DOWNLOAD DIRETTO NEL NODO
            state = download_model_to_cache(state, config, selected_model)
            
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
                
                # Download del fallback model
                state = download_model_to_cache(state, config, fallback_model)
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
        
        prompt = {
            "instruction": f"""Modello AI trovato per {state.last_task}

📦 Dettagli:
- Nome: {github_result['name']}
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
            
            prompt = {
                "instruction": f"""Modello AI trovato per {state.last_task}

📦 Dettagli:
- Nome: {google_result['model']['name']}
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
            
            prompt = {
                "instruction": f"""Modello di fallback per {state.last_task}

Dopo 3 tentativi di ricerca, ecco il modello di fallback:

📦 Dettagli:
- Nome: {fallback_model['name']}
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


def model_selection_routing(state: MasterState) -> Literal["run_analyze", "download_model", "search_recommendation_model"]:
    """
    Routing intelligente dopo selezione modello.
    Gestisce il loop di ricerca fino a max 3 tentativi.
    """
    
    logger.info(f"\n🔄 ROUTING DECISION:")
    logger.info(f"   discovery_method: {state.model_discovery_method}")
    logger.info(f"   search_iterations: {state.search_iterations}")
    
    # ====================================================================
    # CASO 1: Default model (niente ricerca)
    # ====================================================================
    if state.model_discovery_method == "default":
        logger.info("   → Modello di default, va a run_analyze")
        return "run_analyze"
    
    # ====================================================================
    # CASO 2: Ricerca attiva (ritorna al loop se rifiutato)
    # ====================================================================
    elif state.model_discovery_method == "search":
        if state.search_iterations < 3:
            logger.info(f"   → Ricerca in corso ({state.search_iterations}/3), ritorno a search_recommendation_model")
            return "search_recommendation_model"
        else:
            logger.info(f"   → Max iterazioni (3/3) raggiunto, vai a run_analyze")
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
                        
                        elif item.type == "file" and item.name.endswith(".h5"):
                            description = extract_description(item.name, item.path)
                            h5_files.append({
                                'name': item.name,
                                'path': item.path,
                                'size': item.size if hasattr(item, 'size') else 0,
                                'description': description,
                                'folder': item.path.rsplit('/', 1)[0] if '/' in item.path else folder
                            })
                            logger.debug(f"  {'  ' * depth}✅ File .h5: {item.name}")
                            
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
1. Analizza TUTTI i modelli
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
        
        if config is None:
            logger.warning(f"⚠️  config non fornito, uso default Ollama")
            llm = ChatOllama(model="mistral", temperature=0)
        else:
            cfg = Configuration.from_runnable_config(config)
            llm = ChatOllama(
                model=cfg.local_llm,
                temperature=0,
                num_ctx=cfg.llm_context_window
            )
        
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
        
        google_prompt = f"""Ricerca file .h5 per STM32 AI

TASK: {state.last_task}
MCU: {state.target}

RICERCA:
1. Cerca nel repository STMicroelectronics stm32ai-modelzoo
2. Priorità: link GitHub /raw/ diretti
3. File .h5 per questa task

FORNISCI RISULTATO CON:
- Name: [modello]
- URL: [link_scaricabile_.h5]
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
                    'local_filename': search_extraction.model_name.replace(" ", "_") + ".h5",
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
    
    if not url or not url.endswith('.h5'):
        logger.warning(f"⚠️  URL non .h5: {url[:50]}")
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
    
    name = filename.replace(".h5", "")
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
    
    cached_path = os.path.join(cache_dir, model_filename)
    
    # === VERIFICA CACHE ===
    
    if os.path.exists(cached_path) and os.path.isfile(cached_path):
        logger.info(f"✓ Modello in cache: {cached_path}")
        logger.info(f"  Size: {os.path.getsize(cached_path) / (1024*1024):.1f} MB")
        state.model_path = cached_path
        
        # ✅ STAMPA ARCHITETTURA MODELLO - MODO ROBUSTO
        logger.info(f"\n📋 ANALISI ARCHITETTURA MODELLO (da cache)")
        logger.info("=" * 80)
        try:
            # ← PRIMO TENTATIVO: caricamento Keras standard
            keras_model = load_model(
                cached_path, 
                compile=False
            )
            logger.info(f"\n{keras_model.summary()}")
            
            # ✅ INFO AGGIUNTIVE
            logger.info(f"\n📊 STATISTICS:")
            logger.info(f"  Input shape: {keras_model.input_shape}")
            logger.info(f"  Output shape: {keras_model.output_shape}")
            logger.info(f"  Total parameters: {keras_model.count_params():,}")
            logger.info(f"  Trainable parameters: {sum([tf.size(w).numpy() for w in keras_model.trainable_weights]):,}")
            logger.info("=" * 80 + "\n")
            
            state.model_info = {
                "input_shape": str(keras_model.input_shape),
                "output_shape": str(keras_model.output_shape),
                "n_layers": len(keras_model.layers),
                "total_params": int(keras_model.count_params()),
                "trainable_params": int(sum([tf.size(w).numpy() for w in keras_model.trainable_weights]))
            }
            
        except Exception as e:
            logger.warning(f"⚠️  Primo tentativo fallito: {str(e)[:80]}")
            logger.info(f"   Provo analisi via HDF5 raw...")
            
            # ← SECONDO TENTATIVO: lettura raw HDF5 (più robusta)
            try:
                with h5py.File(cached_path, 'r') as f:
                    logger.info(f"\n📋 ANALISI HDF5 (raw)")
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
                        logger.info(f"  Layers: {list(weights_group.keys())[:20]}{'...' if n_layers > 20 else ''}") # stampa i primi 20 layers
                    
                    logger.info("=" * 80 + "\n")
                    
                    # Comunque continua - il file è caricabile per inferenza
                    
            except Exception as e2:
                logger.warning(f"⚠️  Analisi HDF5 fallita: {str(e2)[:100]}")
        
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
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total_size:
                                pct = (downloaded / total_size) * 100
                                logger.info(f"  ⬇️  {pct:.1f}%")
                
                logger.info(f"✓ Download completato! Size: {os.path.getsize(cached_path) / (1024*1024):.1f} MB")
                
                # ✅ STAMPA ARCHITETTURA - MODO ROBUSTO (uguale a sopra)
                logger.info(f"\n📋 ANALISI ARCHITETTURA MODELLO (appena scaricato)")
                logger.info("=" * 80)
                try:
                    keras_model = load_model(
                        cached_path, 
                        compile=False
                    )
                    logger.info(f"\n{keras_model.summary()}")
                    logger.info(f"\n📊 STATISTICS:")
                    logger.info(f"  Input shape: {keras_model.input_shape}")
                    logger.info(f"  Output shape: {keras_model.output_shape}")
                    logger.info(f"  Total parameters: {keras_model.count_params():,}")
                    logger.info(f"  Trainable parameters: {sum([tf.size(w).numpy() for w in keras_model.trainable_weights]):,}")
                    logger.info("=" * 80 + "\n")
                    
                    state.model_info = {
                        "input_shape": str(keras_model.input_shape),
                        "output_shape": str(keras_model.output_shape),
                        "n_layers": len(keras_model.layers),
                        "total_params": int(keras_model.count_params()),
                        "trainable_params": int(sum([tf.size(w).numpy() for w in keras_model.trainable_weights]))
                    }
                    
                except Exception as e:
                    logger.warning(f"⚠️  Analisi Keras fallita, provo HDF5...")
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
                            for chunk in response.iter_content(chunk_size=8192):
                                if chunk:
                                    f.write(chunk)
                                    downloaded += len(chunk)
                                    if total_size:
                                        pct = (downloaded / total_size) * 100
                                        logger.info(f"  ⬇️  {pct:.1f}%")
                        
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
    analyze_dir = os.path.join(state.ai_output_dir, "report_resnet")
    os.makedirs(analyze_dir, exist_ok=True)
    cmd = [
        "stedgeai", "analyze",
        "--model", state.model_path,
        "--target", state.target,
        "--output", analyze_dir
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    state.analyze_success = (res.returncode == 0)
    if not state.analyze_success:
        state.ai_error_message = res.stderr.strip() or f"Return code {res.returncode}"
    state.analyze_report_dir = analyze_dir
    logger.info("✓ Analyze completato" if state.analyze_success else f"✗ Analyze fallito: {state.ai_error_message}")
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
        "--compression", state.compression,
        "--output", code_dir
    ]
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

