# ============================================================================
# WORKFLOW 7: DATASET SELECTION & MANAGEMENT
# ============================================================================
# Modulo per la selezione e il download di dataset reali
#
# Responsabilità:
#   - Chiedere all'utente la fonte dei dati (Real, Synthetic, Both)
#   - Mostrare menu di dataset predefiniti basati sul task (Audio/Vision)
#   - Scaricare dataset reali (es. CIFAR-10, MNIST, SpeechCommands)
#
# Dipendenze: tensorflow, keras, requests

import os
import logging
import json
from typing import Literal, Optional, List, Dict, Any
from datetime import datetime

from langgraph.types import interrupt
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from typing import Literal, Optional, List, Dict, Any, Union

class DatasetRegistration(BaseModel):
    """Schema per la registrazione di un nuovo dataset via URL"""
    name: str = Field(description="Nome leggibile del dataset")
    key: str = Field(description="Chiave univoca (snake_case, es: my_custom_data)")
    category: Literal["vision", "audio", "object_detection", "human_activity_recognition"] = Field(
        description="Categoria del dataset"
    )
    url: str = Field(description="URL diretto per il download (zip, tar.gz)")
    description: str = Field(description="Breve descrizione del dataset")
    expected_shape: Optional[List[int]] = Field(
        default=None, 
        description="Shape atteso degli input (es: [224, 224, 3]). Opzionale."
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
    """Ritorna il path assoluto di una risorsa nella cartella resources."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    res_path = os.path.join(base_dir, "resources", filename)
    return res_path

def load_dataset_catalog() -> dict:
    """Carica il catalogo dataset dal file JSON."""
    path = get_resource_path("predefined_datasets.json")
    if not os.path.exists(path):
        logger.warning(f"⚠️ Catalogo dataset non trovato in {path}, ritorno vuoto.")
        return {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"❌ Errore caricamento catalogo dataset: {e}")
        return {}

def load_dataset_mapping() -> dict:
    """Carica il mapping model-to-dataset dal file JSON."""
    path = get_resource_path("dataset_mapping.json")
    if not os.path.exists(path):
        logger.warning(f"⚠️ Mapping dataset non trovato in {path}, ritorno vuoto.")
        return {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"❌ Errore caricamento mapping dataset: {e}")
        return {}

def save_dataset_catalog(catalog: dict):
    """Salva il catalogo dataset nel file JSON."""
    path = get_resource_path("predefined_datasets.json")
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(catalog, f, indent=4, ensure_ascii=False)
    except Exception as e:
        logger.error(f"❌ Errore salvataggio catalogo dataset: {e}")

def validate_url(url: str) -> bool:
    """Verifica se un URL è raggiungibile."""
    try:
        # User Agent per evitare blocchi
        headers = {"User-Agent": "Mozilla/5.0 (STM32-Agent)"}
        response = requests.head(url, timeout=5, allow_redirects=True, headers=headers)
        if response.status_code >= 400:
            # Fallback a GET se HEAD è bloccato (alcuni server lo fanno)
            response = requests.get(url, timeout=5, stream=True, headers=headers)
        return response.status_code < 400
    except Exception:
        return False

# ============================================================================
# NODES
# ============================================================================

def decide_data_source(state: MasterState, config: dict) -> MasterState:
    """Chiede all'utente quale fonte dati utilizzare"""
    
    logger.info("📊 Dataset Source Selection")
    
    prompt = {
        "instruction": "Quale dataset vuoi utilizzare per il fine-tuning?",
        "options": {
            "1": "Real Dataset (Seleziona dai predefiniti)",
            "2": "Register New Dataset (Aggiungi tramite URL)",
            "3": "Synthetic Data (Generato ora)"
        }
    }
    
    # user_response = interrupt(prompt)
    user_response = "1" # BYPASS
    if isinstance(user_response, dict):
        user_text = str(user_response.get("response", user_response.get("input", ""))).lower()
    else:
        user_text = str(user_response).lower()
    
    # Default: synthetic (no download)
    if not user_text or user_text.strip() == "":
        user_text = "1" # ho messo 1, giusto per velocizzare il test. ma va bene anche '2'
        
    if "1" in user_text or "real" in user_text:
        state.dataset_source = "real"
    elif "2" in user_text or "register" in user_text or "aggiungi" in user_text:
        state.dataset_source = "register"
    elif "3" in user_text or "synthetic" in user_text:
        state.dataset_source = "synthetic"
    else:
        # Default fallback
        logger.warning(f"⚠️  Scelta non riconosciuta '{user_text}', default a Synthetic")
        state.dataset_source = "synthetic"
        
    logger.info(f"✅ Selected: {state.dataset_source}")
    return state


def register_custom_dataset(state: MasterState, config: dict) -> MasterState:
    """Permette all'utente di registrare un nuovo dataset fornendo un URL"""
    
    logger.info("➕ Registrazione nuovo dataset...")
    
    prompt_text = """Fornisci le informazioni per il nuovo dataset.
Format richiesto:
- Nome: [Nome Dataset]
- Chiave: [chiave_snake_case]
- Categoria: [vision/audio/object_detection/human_activity_recognition]
- URL: [URL diretto al file .zip o .tar.gz]
- Descrizione: [Breve descrizione]
"""
    
    user_response = interrupt({
        "instruction": prompt_text,
        "hint": "Puoi scrivere in linguaggio naturale, estrarrò io i dati."
    })
    
    if isinstance(user_response, dict):
        user_input = str(user_response.get("response", user_response.get("input", "")))
    else:
        user_input = str(user_response)
    
    logger.info(f"📝 User response: {user_input[:100]}")
    
    # Parse con LLM structured output
    from src.assistant.utils import get_llm
    llm = get_llm(config, structured_schema=DatasetRegistration)
    try:
        info = llm.invoke([
            SystemMessage(content="Sei un esperto di MLOps. Estrai le informazioni del dataset dall'input utente."),
            HumanMessage(content=user_input)
        ])
        
        logger.info(f"🧐 Validazione URL: {info.url}")
        if not validate_url(info.url):
            logger.warning(f"⚠️ URL non raggiungibile o non valido: {info.url}")
            # Non blocchiamo, ma avvisiamo
            
        # Aggiornamento catalogo
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
            "note": "Aggiunto dall'utente via URL"
        }
        
        save_dataset_catalog(catalog)
        logger.info(f"✅ Dataset '{info.name}' registrato con successo in '{cat_key}'")
        
        # Imposta come selezionato
        state.real_dataset_name = info.key
        state.dataset_source = "real" # Prosegui verso download
        
    except Exception as e:
        logger.error(f"❌ Errore durante la registrazione: {e}")
        state.ai_error_message = f"Registrazione fallita: {e}"
        state.dataset_source = "synthetic" # Fallback
        
    return state


def select_predefined_dataset(state: MasterState, config: dict) -> MasterState:
    """
    Mostra menu dataset basato sul task del modello selezionato.
    Determina automaticamente il task_type più appropriato.
    """
    
    logger.info("📊 Selezione dataset intelligente basata sul modello...")
    
    # ===== STEP 1: Determina task_type dal modello selezionato =====
    task_type = "vision"  # Default
    preferred_datasets = []
    mapping_notes = ""
    
    # Carica mapping e catalogo dinamici
    mapping_catalog = load_dataset_mapping()
    dataset_catalog = load_dataset_catalog()
    
    # Cerca mapping dal last_task salvato (task selezionato dall'utente)
    if state.last_task:
        mapping = mapping_catalog.get(state.last_task)
        if mapping:
            task_type = mapping["task_type"]
            preferred_datasets = mapping["preferred_datasets"]
            mapping_notes = mapping.get("notes", "")
            logger.info(f"✓ Task type determinato dal modello: {task_type}")
            logger.info(f"  Preferred datasets: {preferred_datasets}")
        else:
            logger.warning(f"⚠️ Task '{state.last_task}' non mappato, uso euristica")
    
    # ===== Fallback euristica (backward compatibility) =====
    if not preferred_datasets:
        logger.info("  Usando euristica basata su keyword nel task...")
        # Check se last_task è vuoto prima di usare 'in' operator
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
            # Default se last_task è vuoto
            logger.warning("⚠️ state.last_task è vuoto, uso default vision")
            task_type = "vision"
            preferred_datasets = ["cifar10", "mnist"]
    
    # ===== STEP 2: Verifica compatibilità input shape (opzionale) =====
    if state.model_architecture:
        input_shape = state.model_architecture.get('input_shape')
        if input_shape:
            logger.info(f"  Model input shape: {input_shape}")
            
            # Euristica avanzata basata su input shape
            if isinstance(input_shape, (list, tuple)) and len(input_shape) == 3:
                h, w, c = input_shape
                
                # Audio spectrograms: tipicamente piccoli e mono-channel
                if c == 1 and (h < 100 or w < 100):
                    logger.info(f"  ✓ Input shape {input_shape} suggerisce audio (spectrogram)")
                    if task_type == "vision":  # Solo se non già audio
                        task_type = "audio"
                        preferred_datasets = ["speech_commands", "fsdd", "esc50"]
                
                # HAR: input 1D o molto piccolo
                elif len(input_shape) == 2 or (h < 50 and w < 50):
                    logger.info(f"  ⚠️ Input shape {input_shape} potrebbe essere per HAR (sensor data)")
    
    # ===== STEP 3: Seleziona dataset dal catalogo =====
    category_info = dataset_catalog.get(task_type, dataset_catalog.get("vision", {}))
    options = category_info.get("datasets", {})
    
    if not options:
        logger.error(f"❌ Nessun dataset trovato per task_type '{task_type}'")
        # Fallback a vision
        task_type = "vision"
        category_info = dataset_catalog.get("vision", {})
        options = category_info.get("datasets", {"cifar10": {}})
        preferred_datasets = ["cifar10"]
    
    # ===== STEP 4: Ordina dataset (preferred prima) =====
    # Mostra prima i dataset preferiti, poi gli altri
    all_keys = list(options.keys())
    
    # Filtra preferred che esistono effettivamente nel catalogo
    valid_preferred = [k for k in preferred_datasets if k in all_keys]
    other_keys = [k for k in all_keys if k not in valid_preferred]
    
    valid_keys = valid_preferred + other_keys
    
    # ===== STEP 5: Costruisci menu con badge per dataset consigliati =====
    menu_text = f"\n{'='*70}\n"
    menu_text += f"📊 DATASET REALI PER: {task_type.upper().replace('_', ' ')}\n"
    menu_text += f"{'='*70}\n"
    logger.info(f"  Menu datasets: {valid_keys}") # Debug log
    
    if mapping_notes:
        menu_text += f"💡 Note: {mapping_notes}\n\n"
    
    menu_text += "Scegli un dataset:\n\n"
    
    for idx, key in enumerate(valid_keys, 1):
        info = options[key]
        
        # Badge per dataset consigliati
        badge = "⭐ CONSIGLIATO" if key in valid_preferred else ""
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
    
    # ===== STEP 6: Mostra informazioni modello selezionato =====
    if state.selected_model:
        model_name = state.selected_model.get('name', 'N/A')
        menu_text += f"\n🤖 Modello selezionato: {model_name}\n"
    
    prompt = {
        "instruction": menu_text,
        # Prepend a dummy value at index 0 so that the UI indices [1, 2, 3...]
        # align with the human-readable menu [1, 2, 3...]
        "valid_options": ["(Digitare numero / nome)"] + valid_keys,
        "hint": "Inserisci il numero o il nome del dataset (es: 1 oppure cifar10)"
    }
    
    # user_response = interrupt(prompt)
    user_response = "fruit_360" # BYPASS
    if isinstance(user_response, dict):
        selection = str(user_response.get("response", user_response.get("input", ""))).lower().strip()
    else:
        selection = str(user_response).lower().strip()
    
    # ===== STEP 8: Parsing risposta utente =====
    # Default: primo dataset consigliato (o primo disponibile)
    if not selection or selection.strip() == "":
        selection = valid_keys[0] if valid_keys else "cifar10"
        logger.info(f"  Nessuna selezione, uso default: {selection}")
    
    # Fuzzy matching: cerca per nome o per numero
    selected_key = None
    
    # Prova a interpretare come numero
    try:
        idx = int(selection) - 1
        if 0 <= idx < len(valid_keys):
            selected_key = valid_keys[idx]
            logger.info(f"  ✓ Dataset selezionato per indice {idx+1}: {selected_key}")
    except ValueError:
        pass
    
    # Se non è un numero, cerca per match parziale nel nome
    if not selected_key:
        for key in valid_keys:
            if key in selection or selection in key:
                selected_key = key
                logger.info(f"  ✓ Dataset selezionato per match: {selected_key}")
                break
    
    # Fallback: usa il primo disponibile
    if not selected_key:
        selected_key = valid_keys[0] if valid_keys else "cifar10"
        logger.warning(f"⚠️ Dataset non riconosciuto '{selection}', uso default: {selected_key}")
    
    # ===== STEP 9: Verifica compatibilità modello-dataset =====
    if state.model_architecture and selected_key:
        compatibility_ok = check_dataset_model_compatibility(
            state.model_architecture.get('input_shape'),
            selected_key,
            task_type
        )
        if not compatibility_ok:
            logger.warning("⚠️ Potrebbe essere necessario preprocessing/resizing del dataset")
    
    # ===== STEP 10: Salva selezione =====
    state.real_dataset_name = selected_key
    logger.info(f"✅ Dataset finale selezionato: {selected_key}")
    logger.info(f"   Task type: {task_type}")
    
    return state


def check_dataset_model_compatibility(model_input_shape, dataset_name: str, task_type: str) -> bool:
    """
    Verifica se il dataset è compatibile con l'input del modello.
    Utilizza i metadata nel catalogo se presenti.
    """
    
    logger.info(f"🔍 Verifica compatibilità: {dataset_name} vs {model_input_shape}")
    
    # 1. Carica catalogo per trovare shape atteso
    catalog = load_dataset_catalog()
    dataset_info = None
    for category in catalog.values():
        if dataset_name in category.get("datasets", {}):
            dataset_info = category["datasets"][dataset_name]
            break
            
    if not dataset_info:
        logger.warning(f"⚠️ Dataset '{dataset_name}' non trovato nel catalogo per check compatibilità.")
        return True # Prosegui comunque
        
    expected_shape = dataset_info.get("expected_shape")
    
    if not model_input_shape:
        logger.info("  ℹ️  Input shape modello non disponibile, skip compatibilità check")
        return True
    
    # ===== Se dataset ha shape variabile (None), sempre OK =====
    if expected_shape is None:
        logger.info(f"  ✓ Dataset '{dataset_name}' ha dimensioni variabili (supporta preprocessing)")
        return True
    
    # ===== Converti model_input_shape in tuple per confronto =====
    model_shape_tuple = None
    if isinstance(model_input_shape, list):
        model_shape_tuple = tuple(model_input_shape)
    elif isinstance(model_input_shape, tuple):
        model_shape_tuple = model_input_shape
    elif isinstance(model_input_shape, str):
        # Prova a parseare stringa tipo "(None, 224, 224, 3)"
        try:
            import ast
            parsed = ast.literal_eval(model_input_shape)
            if isinstance(parsed, (list, tuple)):
                model_shape_tuple = tuple(parsed)
        except:
            pass
            
    if model_shape_tuple is None:
        logger.warning(f"  ⚠️ Input shape formato non riconosciuto: {type(model_input_shape)} ({model_input_shape})")
        return True # Prosegui comunque
    
    # ===== Confronta dimensioni =====
    if expected_shape == model_shape_tuple:
        logger.info(f"  ✓✓ Perfetta compatibilità: dataset {expected_shape} = modello {model_shape_tuple}")
        return True
    
    # ===== Shape diverso → serve resize =====
    logger.warning(f"  ⚠️ Incompatibilità shape:")
    logger.warning(f"     Dataset '{dataset_name}': {expected_shape}")
    logger.warning(f"     Modello richiede: {model_shape_tuple}")
    
    # Suggerimenti specifici
    if task_type == "vision":
        logger.info(f"  💡 Soluzione: Usa resizing layer o preprocessing per adattare {expected_shape} → {model_shape_tuple}")
    elif task_type == "audio":
        logger.info(f"  💡 Soluzione: Modifica parametri spectrogram processing (target_shape)")
    elif task_type in ["human_activity_recognition", "object_detection"]:
        logger.info(f"  💡 Soluzione: Configura window size o usa data augmentation con resize")
    
    return False


def download_dataset(state: MasterState, config: dict) -> MasterState:
    """Scarica il dataset selezionato utilizzando il catalogo dinamico"""
    
    dataset_name = state.real_dataset_name
    logger.info(f"📥 Avvio download dataset: {dataset_name}...")
    
    # Setup dir
    dataset_dir = os.path.join(state.base_dir, "data", "real_datasets", dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)
    state.real_dataset_path = dataset_dir
    
    # 1. Recupera info dal catalogo
    catalog = load_dataset_catalog()
    dataset_info = None
    category_name = None
    
    for cat, info in catalog.items():
        if dataset_name in info.get("datasets", {}):
            dataset_info = info["datasets"][dataset_name]
            category_name = cat
            break
            
    if not dataset_info:
        logger.error(f"❌ Dataset '{dataset_name}' non trovato nel catalogo.")
        state.ai_error_message = f"Dataset {dataset_name} non trovato."
        return state

    url = dataset_info.get("url")
    keras_name = dataset_info.get("keras_name")
    tfds_name = dataset_info.get("tfds_name")
    
    try:
        # A. Keras Built-in
        if keras_name:
            logger.info(f"📦 Utilizzo Keras built-in dataset: {keras_name}")
            import tensorflow as tf
            if keras_name == "cifar10":
                (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
            elif keras_name == "mnist":
                (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
            elif keras_name == "fashion_mnist":
                (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
            else:
                raise ValueError(f"Keras dataset {keras_name} non supportato direttamente.")
                
            np.save(os.path.join(dataset_dir, "x_train.npy"), x_train)
            np.save(os.path.join(dataset_dir, "y_train.npy"), y_train)
            np.save(os.path.join(dataset_dir, "x_test.npy"), x_test)
            np.save(os.path.join(dataset_dir, "y_test.npy"), y_test)
            logger.info(f"✅ Dataset salvato in {dataset_dir}")

        # B. URL Download (Generic Archive)
        elif url:
            if "roboflow.com" in url:
                # Logica manuale per Roboflow (già esistente)
                logger.info(f"📥 Dataset Roboflow rilevato")
                logger.info(f"")
                logger.info(f"⚠️  RICHIESTA AZIONE UTENTE:")
                logger.info(f"   Per scaricare questo dataset:")
                logger.info(f"")
                logger.info(f"   1. Visita: {url}")
                logger.info(f"   2. Crea account Roboflow (gratuito)")
                logger.info(f"   3. Seleziona formato: COCO JSON")
                logger.info(f"   4. Download ed estrai in: {dataset_dir}")
                logger.info(f"")
                
                # Salva istruzioni
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
                
                logger.warning(f"⚠️  Download manuale richiesto. Istruzioni salvate in DOWNLOAD_INSTRUCTIONS.txt")
                
                # Salva metadata minimale
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
                
                # Processing specifico basato sulla categoria
                if category_name == "audio":
                    logger.info("🎵 Processing audio spectrograms...")
                    # target_shape can be retrieved from dataset_info if needed, or use default
                    process_speech_commands(extract_dir, dataset_dir)
                    logger.info(f"✅ Audio dataset processed")
                elif category_name == "human_activity_recognition":
                    logger.info("⌚ HAR dataset pronto (estratto)")
                    # Metadata salvataggio
                    metadata = {
                        "dataset_name": dataset_name,
                        "download_date": datetime.now().isoformat(),
                        "url": url,
                        "type": "sensor_data",
                        "note": dataset_info.get("note", "")
                    }
                    with open(os.path.join(dataset_dir, "metadata.json"), "w") as f:
                        json.dump(metadata, f, indent=2)
                    logger.info(f"✅ HAR dataset downloaded")
                    logger.info(f"💡 Dataset contains raw sensor data (accelerometer/gyroscope)")
                    logger.info(f"⚠️  Preprocessing required: windowing, feature extraction, normalization")
                else:
                    logger.info(f"✅ Generic dataset pronto in {extract_dir}")
                    
                    # Se la categoria è vision o object_detection (immagini), processa automaticamente
                    if category_name in ["vision", "object_detection"]:
                        logger.info(f"🖼️  Tentativo di processing automatico per dataset immagini...")
                        process_generic_vision_dataset(extract_dir, dataset_dir)

        # C. TFDS (TensorFlow Datasets)
        elif tfds_name:
            logger.info(f"📥 Loading via TFDS: {tfds_name}")
            
            try:
                import tensorflow_datasets as tfds
                
                # Download dataset con tfds (automatico)
                logger.info(f"⬇️  Loading from TFDS: {tfds_name}")
                logger.info(f"   This may take a while for first download (~{dataset_info.get('size', 'unknown size')})...")
                
                # Load dataset con info
                ds_train, ds_info = tfds.load(
                    tfds_name,
                    split='train',
                    with_info=True,
                    data_dir=dataset_dir  # Salva in directory specifica
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
                
                # Salva metadata
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
                
                # Salva info su come usare il dataset
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
                
                logger.info(f"✅ TFDS {tfds_name} setup completato")
                logger.info(f"💡 Usage instructions saved in USAGE_INFO.txt")
                if category_name == "object_detection":
                    logger.info(f"⚠️  Note: Dataset includes bounding boxes and segmentation masks")
                    
            except ImportError:
                logger.error("❌ tensorflow_datasets not installed!")
                logger.info("   Install with: pip install tensorflow-datasets")
                raise
            
    except Exception as e:
        logger.error(f"❌ Errore durante download/processing: {e}")
        state.ai_error_message = str(e)
        # Fallback dummy file
        with open(os.path.join(dataset_dir, "README.txt"), "w") as f:
            f.write(f"Dataset {dataset_name} download/processing failed: {e}")
        
    return state


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def download_file(url: str, dest_path: str):
    """Scarica file con progress bar"""
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
    """Estrae .tar.gz o .zip"""
    os.makedirs(extract_to, exist_ok=True)
    if file_path.endswith("tar.gz") or file_path.endswith(".tgz"):
        with tarfile.open(file_path, "r:gz") as tar:
            tar.extractall(path=extract_to)
    elif file_path.endswith(".zip"):
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)

def audio_to_spectrogram(file_path: str, target_shape=(32, 32)) -> Optional[np.ndarray]:
    """
    Legge un WAV, calcola STFT spectrogram, ridimensiona a target_shape.
    Ritorna array (H, W, 1) normalizzato [0,1].
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
    Processa Google Speech Commands dataset.
    Struttura: extracted/speech_commands_v0.02/word/file.wav
    """
    # Trova root reale (spesso c'è una cartella intermedia)
    # Per speech commands v0.02 di solito è diretto o in una cartella
    # Cerchiamo cartelle che sono le label (es. "yes", "no", "up")
    
    # Keywords da usare (subset per semplicità o tutte)
    # Usiamo le 10 standard + silence/unknown se vogliamo, ma per ora prendiamo le cartelle presenti
    # Filtriamo cartelle di sistema o file
    
    root_search = extract_dir
    # Se c'è una sola cartella dentro extracted, entra lì
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
        
        # Shuffle e limit
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
    Scansiona una cartella estratta alla ricerca di immagini e le converte in .npy.
    Inferisce le classi dalle sottocartelle.
    """
    logger.info(f"📁 Scansione generica immagini in {extract_dir}...")
    
    # Estensioni supportate
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    
    # 1. Trova tutte le immagini e mappa le classi
    image_paths = []
    for root, dirs, files in os.walk(extract_dir):
        for f in files:
            if f.lower().endswith(valid_exts):
                image_paths.append(os.path.join(root, f))
                
    if not image_paths:
        logger.warning("⚠️  Nessuna immagine trovata nell'archivio estratto.")
        return

    # 2. Inferenza classi dal nome della cartella genitore
    # Assumiamo struttura: root/classe/immagine.jpg
    path_to_class = {}
    for p in image_paths:
        cls_name = os.path.basename(os.path.dirname(p))
        if not cls_name or cls_name == os.path.basename(extract_dir):
            cls_name = "default_class"
        path_to_class[p] = cls_name
        
    classes = sorted(list(set(path_to_class.values())))
    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    
    logger.info(f"✓ Trovate {len(image_paths)} immagini in {len(classes)} classi.")
    logger.info(f"✓ Classi: {classes[:10]} {'...' if len(classes) > 10 else ''}")

    # 3. Shuffle e Limite per performance/memoria
    import random
    random.shuffle(image_paths)
    image_paths = image_paths[:max_samples]
    
    X = []
    y = []
    
    logger.info(f"⚙️  Processing {len(image_paths)} campioni...")
    
    for p in image_paths:
        try:
            img = tf.io.read_file(p)
            img = tf.image.decode_image(img, channels=3, expand_animations=False)
            img = tf.image.resize(img, target_shape[:2])
            img = img / 255.0  # Normalizzazione [0,1]
            
            X.append(img.numpy())
            y.append(class_to_idx[path_to_class[p]])
        except Exception as e:
            # logger.debug(f"Salto file corrotto {p}: {e}")
            continue
            
    if not X:
        logger.error("❌ Errore: Nessuna immagine valida processata.")
        return
        
    X = np.array(X, dtype='float32')
    y = np.array(y, dtype='int32')
    
    # 4. Salvataggio
    np.save(os.path.join(output_dir, "x_train.npy"), X)
    np.save(os.path.join(output_dir, "y_train.npy"), y)
    
    # Split manuale per validazione (20%)
    split_idx = int(len(X) * 0.8)
    np.save(os.path.join(output_dir, "x_test.npy"), X[split_idx:])
    np.save(os.path.join(output_dir, "y_test.npy"), y[split_idx:])
    
    with open(os.path.join(output_dir, "classes.json"), "w") as f:
        json.dump(class_to_idx, f, indent=2)
        
    logger.info(f"✅ Processing completato. Salvati {len(X)} campioni in {output_dir}")
