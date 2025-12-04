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
# DATASET CATALOG
# ============================================================================

DATASET_CATALOG = {
    "audio": {
        "speech_commands": {
            "description": "Google Speech Commands (Keyword Spotting: Yes, No, Up, Down...)",
            "url": "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz",
            "type": "audio",
            "size": "2.3 GB"
        },
        "esc50": {
            "description": "ESC-50: 2000 Environmental Sounds (Rain, Chainsaw, Dog...)",
            "url": "https://github.com/karolpiczak/ESC-50/archive/master.zip",
            "type": "audio",
            "size": "600 MB"
        },
        "fsdd": {
            "description": "Free Spoken Digit Dataset (Spoken MNIST)",
            "url": "https://github.com/Jakobovski/free-spoken-digit-dataset/archive/v1.0.10.zip",
            "type": "audio",
            "size": "15 MB"
        }
    },
    "vision": {
        "cifar10": {
            "description": "CIFAR-10: 60k 32x32 color images (10 classes)",
            "keras_name": "cifar10",
            "type": "image",
            "size": "170 MB"
        },
        "mnist": {
            "description": "MNIST: Handwritten digits 28x28 grayscale",
            "keras_name": "mnist",
            "type": "image",
            "size": "12 MB"
        },
        "fashion_mnist": {
            "description": "Fashion-MNIST: Clothing items 28x28 grayscale",
            "keras_name": "fashion_mnist",
            "type": "image",
            "size": "30 MB"
        }
    },
    "object_detection": {
        "roboflow_vehicles": {
            "description": "Roboflow Vehicles: Cars, Trucks detection (COCO format)",
            "url": "https://universe.roboflow.com/roboflow-100/vehicles-q0vsv/dataset/1/download",
            "type": "image",
            "size": "~500 MB",
            "note": "Smaller, focused dataset with COCO-format annotations",
            "kaggle_alternative": "https://www.kaggle.com/datasets/solesensei/coco-minitrain-2017"
        },
        "pascal_voc_2012": {
            "description": "PASCAL VOC 2012: Object Detection (20 classes)",
            "tfds_name": "voc/2012",  # Usa TensorFlow Datasets invece di URL
            "type": "image",
            "size": "~1.9 GB",
            "note": "Downloaded via TensorFlow Datasets (tfds) - automatic"
        }
    },
    "human_activity_recognition": {
        "uci_har": {
            "description": "UCI HAR: 30 subjects, 6 activities (accelerometer + gyroscope)",
            "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip",
            "type": "sensor",
            "size": "~60 MB",
            "note": "Smartphone sensor data at 50Hz"
        },
        "wisdm": {
            "description": "WISDM: 51 subjects, 18 activities (Accelerometer + Gyroscope)",
            "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00507/wisdm-dataset.zip",
            "type": "sensor",
            "size": "~295 MB",
            "note": "UCI ML Repository - Direct download ZIP (20Hz sensor data)"
        }
    }
}

# ============================================================================
# MODEL TO DATASET MAPPING
# ============================================================================
# Maps PREDEFINED_MODELS task types to compatible datasets

MODEL_TO_DATASET_MAP = {
    "image_classification": {
        "preferred_datasets": ["cifar10", "mnist", "fashion_mnist"],
        "task_type": "vision",
        "notes": "Standard image classification datasets work well"
    },
    "object_detection": {
        "preferred_datasets": ["roboflow_vehicles", "pascal_voc_2012"],
        "task_type": "object_detection",
        "notes": "Requires bounding box annotations; Roboflow Vehicles is smaller and faster"
    },
    "human_activity_recognition": {
        "preferred_datasets": ["uci_har", "wisdm"],
        "task_type": "human_activity_recognition",
        "notes": "Sensor data (accelerometer/gyroscope); WISDM is simpler with fewer activities"
    },
    "audio_event_detection": {
        "preferred_datasets": ["speech_commands", "esc50", "fsdd"],
        "task_type": "audio",
        "notes": "Audio converted to spectrograms for vision models"
    }
}

# ============================================================================
# NODES
# ============================================================================

def decide_data_source(state: MasterState, config: dict) -> MasterState:
    """Chiede all'utente quale fonte dati utilizzare"""
    
    logger.info("📊 Dataset Source Selection")
    
    prompt = {
        "instruction": "Quale dataset vuoi utilizzare per il fine-tuning?",
        "options": {
            "1": "Real Dataset (CIFAR-10, SpeechCommands, etc.)",
            "2": "Synthetic Data (Generato ora)"
        }
    }
    
    user_response = interrupt(prompt)
    
    if isinstance(user_response, dict):
        user_text = str(user_response.get("response", user_response.get("input", ""))).lower()
    else:
        user_text = str(user_response).lower()
    
    # Default: synthetic (no download)
    if not user_text or user_text.strip() == "":
        user_text = "1" # ho messo 1, giusto per velocizzare il test. ma va bene anche '2'
        
    if "1" in user_text or "real" in user_text:
        state.dataset_source = "real"
    elif "2" in user_text or "synthetic" in user_text:
        state.dataset_source = "synthetic"
    else:
        # Default fallback
        logger.warning(f"⚠️  Scelta non riconosciuta '{user_text}', default a Synthetic")
        state.dataset_source = "synthetic"
        
    logger.info(f"✅ Selected: {state.dataset_source}")
    return state


def select_predefined_dataset(state: MasterState, config: dict) -> MasterState:
    """
    Mostra menu dataset basato sul task del modello selezionato.
    Usa MODEL_TO_DATASET_MAP per determinare automaticamente il task_type più appropriato.
    """
    
    logger.info("📊 Selezione dataset intelligente basata sul modello...")
    
    # ===== STEP 1: Determina task_type dal modello selezionato =====
    task_type = "vision"  # Default
    preferred_datasets = []
    mapping_notes = ""
    
    # Cerca mapping dal last_task salvato (task selezionato dall'utente)
    if state.last_task:
        mapping = MODEL_TO_DATASET_MAP.get(state.last_task)
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
                preferred_datasets = ["roboflow_vehicles", "pascal_voc_2012"]  # FIX: coco_minitrain non esiste più
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
    options = DATASET_CATALOG.get(task_type, DATASET_CATALOG.get("vision", {}))
    
    if not options:
        logger.error(f"❌ Nessun dataset trovato per task_type '{task_type}'")
        # Fallback a vision
        task_type = "vision"
        options = DATASET_CATALOG["vision"]
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
    menu_text += f"{'='*70}\n\n"
    
    if mapping_notes:
        menu_text += f"💡 Note: {mapping_notes}\n\n"
    
    menu_text += "Scegli un dataset:\n\n"
    
    for idx, key in enumerate(valid_keys, 1):
        info = options[key]
        
        # Badge per dataset consigliati
        badge = "⭐ CONSIGLIATO" if key in valid_preferred else ""
        note = info.get('note', '')
        
        menu_text += f"{idx}. {key}: {info['description']}\n"
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
        "valid_options": valid_keys,
        "hint": "Inserisci il numero o il nome del dataset (es: 1 oppure cifar10)"
    }
    
    # ===== STEP 7: Richiesta input utente =====
    user_response = interrupt(prompt)
    
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
    
    Args:
        model_input_shape: Input shape del modello (es: [224, 224, 3])
        dataset_name: Nome del dataset (es: 'cifar10')
        task_type: Tipo di task (es: 'vision', 'audio', 'human_activity_recognition')
    
    Returns:
        True se compatibile, False se serve preprocessing/resizing
    """
    
    if not model_input_shape:
        logger.info("  ℹ️  Input shape modello non disponibile, skip compatibilità check")
        return True
    
    # ===== Dimensioni standard dei dataset =====
    dataset_shapes = {
        # Vision
        "cifar10": (32, 32, 3),
        "mnist": (28, 28, 1),
        "fashion_mnist": (28, 28, 1),
        
        # Audio (dopo conversione a spectrogram)
        "speech_commands": (32, 32, 1),  # default processing
        "esc50": (64, 64, 1),  # tipico per ESC-50
        "fsdd": (28, 28, 1),  # small spectrograms
        
        # Object Detection (varia)
        "roboflow_vehicles": None,  # Multiple sizes, requires resizing - FIX: aggiornato da coco_minitrain
        "pascal_voc_2012": None,  # Multiple sizes, requires resizing
        
        # HAR (sensor data, varia molto)
        "uci_har": None,  # Time series, shape depends on window size
        "wisdm": None,  # Time series, shape depends on window size
    }
    
    expected_shape = dataset_shapes.get(dataset_name)
    
    # ===== Se dataset ha shape variabile (None), sempre OK =====
    if expected_shape is None:
        logger.info(f"  ✓ Dataset '{dataset_name}' ha dimensioni variabili (supporta preprocessing)")
        return True
    
    # ===== Converti model_input_shape in tuple per confronto =====
    if isinstance(model_input_shape, list):
        model_shape_tuple = tuple(model_input_shape)
    elif isinstance(model_input_shape, tuple):
        model_shape_tuple = model_input_shape
    else:
        logger.warning(f"  ⚠️ Input shape formato non riconosciuto: {type(model_input_shape)}")
        return True
    
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
    """Scarica il dataset selezionato"""
    
    dataset_name = state.real_dataset_name
    logger.info(f"📥 Downloading: {dataset_name}...")
    
    # Setup dir
    dataset_dir = os.path.join(state.base_dir, "data", "real_datasets", dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)
    state.real_dataset_path = dataset_dir
    
    # Logica specifica per Keras Datasets (Vision)
    if dataset_name in ["cifar10", "mnist", "fashion_mnist"]:
        try:
            import tensorflow as tf
            
            if dataset_name == "cifar10":
                (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
            elif dataset_name == "mnist":
                (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
            elif dataset_name == "fashion_mnist":
                (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
                
            # Salva in formato .npy per uniformità con il resto del sistema
            np.save(os.path.join(dataset_dir, "x_train.npy"), x_train)
            np.save(os.path.join(dataset_dir, "y_train.npy"), y_train)
            np.save(os.path.join(dataset_dir, "x_test.npy"), x_test)
            np.save(os.path.join(dataset_dir, "y_test.npy"), y_test)
            
            logger.info(f"✅ Dataset saved to {dataset_dir}")
            
        except Exception as e:
            logger.error(f"❌ Download failed: {e}")
            
    # Logica per Audio (Download URL)
    elif dataset_name in ["speech_commands", "esc50", "fsdd"]:
        try:
            # 1. Download
            url = DATASET_CATALOG["audio"][dataset_name]["url"]
            archive_name = url.split("/")[-1]
            archive_path = os.path.join(dataset_dir, archive_name)
            
            if not os.path.exists(archive_path):
                logger.info(f"⬇️  Downloading {url}...")
                download_file(url, archive_path)
            else:
                logger.info(f"✅ Archive found: {archive_path}")
                
            # 2. Extract
            extract_dir = os.path.join(dataset_dir, "extracted")
            if not os.path.exists(extract_dir):
                logger.info(f"📦 Extracting to {extract_dir}...")
                extract_archive(archive_path, extract_dir)
            else:
                logger.info(f"✅ Extracted dir found")
                
            # 3. Process to Spectrograms (.npy)
            logger.info("🎵 Processing audio to spectrograms...")
            
            # Parametri processing
            target_shape = (32, 32) # Resize spectrogram to 32x32 image
            
            if dataset_name == "speech_commands":
                process_speech_commands(extract_dir, dataset_dir, target_shape)
            elif dataset_name == "esc50":
                # TODO: Implement ESC-50 specific parsing
                logger.warning("⚠️ ESC-50 processing not yet implemented")
                pass
            elif dataset_name == "fsdd":
                # TODO: Implement FSDD specific parsing
                logger.warning("⚠️ FSDD processing not yet implemented")
                pass
                
            logger.info(f"✅ Audio dataset processed")
            
        except Exception as e:
            logger.error(f"❌ Error processing audio dataset: {e}")
            # Fallback dummy
            with open(os.path.join(dataset_dir, "README.txt"), "w") as f:
                f.write(f"Dataset {dataset_name} failed: {e}")
    
    # Logica per Object Detection (Download URL o TFDS)
    elif dataset_name in ["roboflow_vehicles", "pascal_voc_2012"]:
        try:
            dataset_info = DATASET_CATALOG["object_detection"][dataset_name]
            
            # ===== PASCAL VOC 2012: Usa TensorFlow Datasets =====
            if dataset_name == "pascal_voc_2012":
                logger.info(f"📥 Downloading PASCAL VOC 2012 via TensorFlow Datasets...")
                
                try:
                    import tensorflow_datasets as tfds
                    
                    # Download dataset con tfds (automatico)
                    logger.info(f"⬇️  Loading from TFDS: {dataset_info['tfds_name']}")
                    logger.info(f"   This may take a while for first download (~{dataset_info['size']})...")
                    
                    # Load dataset con info
                    ds_train, ds_info = tfds.load(
                        dataset_info['tfds_name'],
                        split='train',
                        with_info=True,
                        data_dir=dataset_dir  # Salva in directory specifica
                    )
                    
                    ds_validation = tfds.load(
                        dataset_info['tfds_name'],
                        split='validation',
                        data_dir=dataset_dir
                    )
                    
                    logger.info(f"✅ PASCAL VOC 2012 loaded successfully!")
                    logger.info(f"   Train samples: {ds_info.splits['train'].num_examples}")
                    logger.info(f"   Validation samples: {ds_info.splits['validation'].num_examples}")
                    logger.info(f"   Features: {ds_info.features}")
                    
                    # Salva metadata
                    metadata = {
                        "dataset_name": dataset_name,
                        "download_date": datetime.now().isoformat(),
                        "source": "TensorFlow Datasets (tfds)",
                        "tfds_name": dataset_info['tfds_name'],
                        "num_train": int(ds_info.splits['train'].num_examples),
                        "num_validation": int(ds_info.splits['validation'].num_examples),
                        "features": str(ds_info.features),
                        "note": dataset_info.get("note", "")
                    }
                    
                    with open(os.path.join(dataset_dir, "metadata.json"), "w") as f:
                        json.dump(metadata, f, indent=2)
                    
                    # Salva info su come usare il dataset
                    with open(os.path.join(dataset_dir, "USAGE_INFO.txt"), "w") as f:
                        f.write(f"PASCAL VOC 2012 Dataset (via TensorFlow Datasets)\n\n")
                        f.write(f"To load this dataset in your code:\n\n")
                        f.write(f"import tensorflow_datasets as tfds\n\n")
                        f.write(f"# Load train split\n")
                        f.write(f"ds_train = tfds.load('{dataset_info['tfds_name']}', split='train', data_dir='{dataset_dir}')\n\n")
                        f.write(f"# Load validation split\n")
                        f.write(f"ds_validation = tfds.load('{dataset_info['tfds_name']}', split='validation', data_dir='{dataset_dir}')\n\n")
                        f.write(f"Features:\n{ds_info.features}\n")
                    
                    logger.info(f"✅ Object Detection dataset (TFDS) setup completato")
                    logger.info(f"💡 Usage instructions saved in USAGE_INFO.txt")
                    logger.info(f"⚠️  Note: Dataset includes bounding boxes and segmentation masks")
                    
                except ImportError:
                    logger.error("❌ tensorflow_datasets not installed!")
                    logger.info("   Install with: pip install tensorflow-datasets")
                    raise
                    
            # ===== Roboflow Vehicles: Download manuale =====
            elif "roboflow.com" in dataset_info.get("url", ""):
                url = dataset_info["url"]
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
            
        except Exception as e:
            logger.error(f"❌ Error downloading object detection dataset: {e}")
            with open(os.path.join(dataset_dir, "README.txt"), "w") as f:
                f.write(f"Dataset {dataset_name} download failed: {e}")
    
    # Logica per Human Activity Recognition (Download URL)
    elif dataset_name in ["uci_har", "wisdm"]:
        try:
            # 1. Download
            url = DATASET_CATALOG["human_activity_recognition"][dataset_name]["url"]
            archive_name = url.split("/")[-1].replace("%20", "_")  # Fix URL encoding
            archive_path = os.path.join(dataset_dir, archive_name)
            
            if not os.path.exists(archive_path):
                logger.info(f"⬇️  Downloading {url}...")
                download_file(url, archive_path)
            else:
                logger.info(f"✅ Archive found: {archive_path}")
            
            # 2. Extract
            extract_dir = os.path.join(dataset_dir, "extracted")
            if not os.path.exists(extract_dir):
                logger.info(f"📦 Extracting to {extract_dir}...")
                extract_archive(archive_path, extract_dir)
            else:
                logger.info(f"✅ Extracted dir found")
            
            # 3. Save metadata
            metadata = {
                "dataset_name": dataset_name,
                "download_date": datetime.now().isoformat(),
                "url": url,
                "type": "sensor_data",
                "note": DATASET_CATALOG["human_activity_recognition"][dataset_name].get("note", "")
            }
            
            with open(os.path.join(dataset_dir, "metadata.json"), "w") as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"✅ HAR dataset downloaded")
            logger.info(f"💡 Dataset contains raw sensor data (accelerometer/gyroscope)")
            logger.info(f"⚠️  Preprocessing required: windowing, feature extraction, normalization")
            
        except Exception as e:
            logger.error(f"❌ Error downloading HAR dataset: {e}")
            with open(os.path.join(dataset_dir, "README.txt"), "w") as f:
                f.write(f"Dataset {dataset_name} download failed: {e}")
            
    else:
        logger.warning(f"⚠️ Dataset '{dataset_name}' non supportato per download automatico")
        with open(os.path.join(dataset_dir, "README.txt"), "w") as f:
            f.write(f"Dataset {dataset_name} requires manual download and processing")
            
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
        # 1. Read WAV
        audio_binary = tf.io.read_file(file_path)
        audio, sample_rate = tf.audio.decode_wav(audio_binary)
        
        # 2. Fix length (1 sec @ 16kHz = 16000 samples)
        # Se più lungo taglia, se più corto pad
        desired_samples = 16000
        audio = tf.squeeze(audio, axis=-1) # (N,)
        
        if tf.shape(audio)[0] < desired_samples:
            paddings = [[0, desired_samples - tf.shape(audio)[0]]]
            audio = tf.pad(audio, paddings)
        else:
            audio = audio[:desired_samples]
            
        # 3. STFT Spectrogram
        # frame_length=255, frame_step=128 -> output approx 124x129
        stft = tf.signal.stft(audio, frame_length=255, frame_step=128)
        spectrogram = tf.abs(stft)
        
        # Add channel dim -> (Time, Freq, 1)
        spectrogram = tf.expand_dims(spectrogram, axis=-1)
        
        # 4. Resize to target image shape (e.g. 32x32)
        spectrogram = tf.image.resize(spectrogram, target_shape)
        
        # 5. Normalize [0, 1]
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
