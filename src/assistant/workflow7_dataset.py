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
    }
}

# ============================================================================
# NODES
# ============================================================================

def decide_data_source(state: MasterState, config: dict) -> MasterState:
    """Chiede all'utente quale fonte dati utilizzare"""
    
    logger.info("📊 Selezione fonte dati...")
    
    prompt = {
        "instruction": "Quale dataset vuoi utilizzare per il fine-tuning?",
        "options": {
            "1": "Real Dataset (CIFAR-10, SpeechCommands, etc.)",
            "2": "Synthetic Data (Generato ora)",
            "3": "Both (Real + Synthetic)"
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
    elif "3" in user_text or "both" in user_text:
        state.dataset_source = "both"
    else:
        # Default fallback
        logger.warning(f"⚠️  Scelta non riconosciuta '{user_text}', default a Synthetic")
        state.dataset_source = "synthetic"
        
    logger.info(f"✓ Data Source: {state.dataset_source}")
    return state


def select_predefined_dataset(state: MasterState, config: dict) -> MasterState:
    """Mostra menu dataset basato sul task"""
    
    # Determina il tipo di task (audio vs vision)
    # Usiamo state.last_task o euristica sul modello
    task_type = "vision" # Default
    if "audio" in state.last_task or "speech" in state.last_task or "sound" in state.last_task:
        task_type = "audio"
    elif "image" in state.last_task or "object" in state.last_task or "vision" in state.last_task:
        task_type = "vision"
        
    # Se il modello ha input 1D/2D possiamo raffinare, ma per ora fidiamoci del task
    
    options = DATASET_CATALOG.get(task_type, DATASET_CATALOG["vision"])
    
    menu_text = "Scegli un dataset reale:\n"
    valid_keys = []
    for key, info in options.items():
        menu_text += f"- {key}: {info['description']} ({info['size']})\n"
        valid_keys.append(key)
        
    prompt = {
        "instruction": menu_text,
        "valid_options": valid_keys
    }
    
    user_response = interrupt(prompt)
    
    if isinstance(user_response, dict):
        selection = str(user_response.get("response", user_response.get("input", ""))).lower().strip() # può essere ad esempio: "1", "2", "3", "audio", "vision"
    else:
        selection = str(user_response).lower().strip()
    
    # Default: first dataset in the list
    if not selection or selection.strip() == "":
        selection = valid_keys[0]
        
    # Fuzzy matching semplice
    selected_key = None
    for key in valid_keys:
        if key in selection:
            selected_key = key
            break
            
    if not selected_key:
        # Fallback al primo
        selected_key = valid_keys[0]
        logger.warning(f"⚠️  Dataset non riconosciuto, uso default: {selected_key}")
        
    state.real_dataset_name = selected_key
    logger.info(f"✓ Dataset selezionato: {selected_key}")
    
    return state


def download_dataset(state: MasterState, config: dict) -> MasterState:
    """Scarica il dataset selezionato"""
    
    dataset_name = state.real_dataset_name
    logger.info(f"⬇️  Download dataset: {dataset_name}...")
    
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
            
            logger.info(f"✓ Dataset salvato in {dataset_dir}")
            
        except Exception as e:
            logger.error(f"❌ Errore download Keras dataset: {e}")
            
    # Logica per Audio (Download URL)
    elif dataset_name in ["speech_commands", "esc50", "fsdd"]:
        # Qui servirebbe logica di download e unzip
        # Per ora simuliamo o scarichiamo solo se piccolo (FSDD)
        logger.warning("⚠️  Download audio dataset completo non ancora implementato (richiede unzip e parsing)")
        # Placeholder: crea file dummy per testare il flusso
        with open(os.path.join(dataset_dir, "README.txt"), "w") as f:
            f.write(f"Dataset {dataset_name} should be here.")
            
    return state
