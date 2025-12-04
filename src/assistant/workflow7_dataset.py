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
    logger.info(f"✅ Dataset selected: {selected_key}")
    
    return state


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
                pass
            elif dataset_name == "fsdd":
                # TODO: Implement FSDD specific parsing
                pass
                
            logger.info(f"✅ Audio dataset processed")
            
        except Exception as e:
            logger.error(f"❌ Error processing audio dataset: {e}")
            # Fallback dummy
            with open(os.path.join(dataset_dir, "README.txt"), "w") as f:
                f.write(f"Dataset {dataset_name} failed: {e}")
            
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
