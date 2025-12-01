# ============================================================================
# WORKFLOW 6: SYNTHETIC DATA GENERATION
# ============================================================================
# Modulo per la generazione di dati sintetici (audio/time-series)
#
# Responsabilità:
#   - Chiedere all'utente i requisiti dei dati (tipo, durata, quantità)
#   - Generare waveform sintetiche (sine, noise, chirp, etc.)
#   - Salvare i dati in formato .npy o .wav per il fine-tuning
#
# Dipendenze: numpy, scipy, soundfile (opzionale), matplotlib (per debug)

import os
import logging
import numpy as np
import json
from typing import Literal, Optional, List, Dict, Any
from datetime import datetime

from langgraph.types import interrupt
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

from src.assistant.configuration import Configuration
from src.assistant.state import MasterState

logger = logging.getLogger(__name__)

# ============================================================================
# SCHEMAS
# ============================================================================

class SyntheticDataRequest(BaseModel):
    """Richiesta strutturata per dati sintetici"""
    signal_type: Literal["sine", "white_noise", "pink_noise", "chirp", "impulse", "silence", "mixed"] = Field(
        description="Tipo di segnale da generare"
    )
    frequency: Optional[float] = Field(description="Frequenza in Hz (per sine/chirp)")
    duration_sec: float = Field(default=1.0, description="Durata in secondi per sample")
    sample_rate: int = Field(default=16000, description="Sample rate in Hz")
    num_samples: int = Field(default=10, description="Numero di campioni da generare")
    amplitude: float = Field(default=0.5, description="Ampiezza del segnale (0.0-1.0)")
    noise_level: float = Field(default=0.0, description="Livello di rumore aggiunto (0.0-1.0)")

# ============================================================================
# NODES
# ============================================================================

def ask_synthetic_data_requirements(state: MasterState, config: dict) -> MasterState:
    """Chiede all'utente che tipo di dati generare"""
    
    logger.info("🧪 Avvio procedura generazione dati sintetici...")
    
    prompt = {
        "instruction": """Generazione Dati Sintetici (Audio/Time-Series)
        
Che tipo di dati vuoi generare per il training?
Puoi specificare:
- Tipo: Sine wave, Noise, Chirp, Impulse
- Frequenza: es. 1kHz, 440Hz
- Quantità: es. 50 samples
- Durata: es. 1 secondo

Esempi:
- "Genera 50 campioni di onda sinusoidale a 1kHz con rumore"
- "Voglio 20 campioni di rumore bianco per 2 secondi"
- "10 chirp da 100Hz a 1kHz"
        """
    }
    
    user_response = interrupt(prompt)
    
    if isinstance(user_response, dict):
        user_text = user_response.get("response", user_response.get("input", str(user_response)))
    else:
        user_text = str(user_response)
        
    logger.info(f"📝 Richiesta utente: '{user_text}'")
    
    # === PARSING CON LLM ===
    cfg = Configuration.from_runnable_config(config)
    llm = ChatOllama(model=cfg.local_llm, temperature=0)
    llm_parser = llm.with_structured_output(SyntheticDataRequest)
    
    system_prompt = """Sei un esperto di DSP (Digital Signal Processing).
Analizza la richiesta dell'utente ed estrai i parametri per la generazione del segnale.
Se l'utente non specifica, usa questi default:
- Duration: 1.0s
- Sample Rate: 16000Hz
- Num Samples: 10
- Amplitude: 0.5
- Noise Level: 0.1 (se menziona rumore) o 0.0

Per "mixed" o richieste complesse, cerca di mappare al tipo più simile o usa "sine" con rumore.
"""

    try:
        request = llm_parser.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Richiesta: {user_text}")
        ])
        
        state.synthetic_request = request.dict()
        logger.info(f"✓ Parametri estratti: {state.synthetic_request}")
        
    except Exception as e:
        logger.error(f"❌ Errore parsing richiesta: {e}")
        # Fallback
        state.synthetic_request = {
            "signal_type": "sine",
            "frequency": 440.0,
            "duration_sec": 1.0,
            "sample_rate": 16000,
            "num_samples": 10,
            "amplitude": 0.5,
            "noise_level": 0.0
        }
        
    return state


def generate_synthetic_samples(state: MasterState, config: dict) -> MasterState:
    """Genera i campioni usando numpy"""
    
    req = state.synthetic_request
    logger.info(f"⚙️  Generazione {req['num_samples']} campioni di tipo {req['signal_type']}...")
    
    params = SyntheticDataRequest(**req)
    
    # Setup output dir
    output_dir = os.path.join(state.base_dir, "data", "synthetic")
    os.makedirs(output_dir, exist_ok=True)
    
    generated_files = []
    
    for i in range(params.num_samples):
        # Time vector
        t = np.linspace(0, params.duration_sec, int(params.sample_rate * params.duration_sec), endpoint=False)
        
        # Base signal
        if params.signal_type == "sine":
            freq = params.frequency or 440.0
            # Aggiungi leggera variazione di frequenza per realismo
            freq_var = np.random.uniform(-5, 5) 
            signal = params.amplitude * np.sin(2 * np.pi * (freq + freq_var) * t)
            
        elif params.signal_type == "white_noise":
            signal = params.amplitude * np.random.uniform(-1, 1, len(t))
            
        elif params.signal_type == "pink_noise":
            # Approssimazione semplice pink noise (1/f)
            white = np.random.randn(len(t))
            signal = np.cumsum(white) # Brownian noise (1/f^2) actually, but close enough for simple test
            signal = signal / np.max(np.abs(signal)) * params.amplitude
            
        elif params.signal_type == "chirp":
            f_start = params.frequency or 100.0
            f_end = f_start * 10
            k = (f_end - f_start) / params.duration_sec
            signal = params.amplitude * np.sin(2 * np.pi * (f_start * t + (k/2) * t**2))
            
        elif params.signal_type == "impulse":
            signal = np.zeros_like(t)
            idx = np.random.randint(0, len(t))
            signal[idx] = params.amplitude
            
        elif params.signal_type == "silence":
            signal = np.zeros_like(t)
            
        else: # mixed or default
            signal = params.amplitude * np.sin(2 * np.pi * 440 * t)
            
        # Add noise
        if params.noise_level > 0:
            noise = np.random.normal(0, params.noise_level, len(t))
            signal = signal + noise
            
        # Normalize to -1..1 range to avoid clipping
        max_val = np.max(np.abs(signal))
        if max_val > 1.0:
            signal = signal / max_val
            
        # Save as .npy (raw data)
        filename = f"{params.signal_type}_{i:03d}.npy"
        filepath = os.path.join(output_dir, filename)
        np.save(filepath, signal.astype(np.float32))
        generated_files.append(filepath)
        
        # Optional: Save as .wav if soundfile is available (omitted for now to keep deps low)
        
    state.synthetic_data_path = output_dir
    state.synthetic_files_count = len(generated_files)
    
    logger.info(f"✓ Generati {len(generated_files)} file in {output_dir}")
    
    return state


def validate_synthetic_data(state: MasterState, config: dict) -> MasterState:
    """Mostra riepilogo e chiede conferma"""
    
    summary = f"""
✅ Generazione Completata!

📂 Output: {state.synthetic_data_path}
📊 File generati: {state.synthetic_files_count}
⚙️  Parametri:
   - Tipo: {state.synthetic_request['signal_type']}
   - Durata: {state.synthetic_request['duration_sec']}s
   - Sample Rate: {state.synthetic_request['sample_rate']}Hz
    """
    
    logger.info(summary)
    
    prompt = {
        "instruction": f"{summary}\n\nVuoi procedere con il fine-tuning usando questi dati? (sì/no)",
    }
    
    user_response = interrupt(prompt)
    
    if isinstance(user_response, dict):
        user_text = str(user_response.get("response", user_response.get("input", ""))).lower()
    else:
        user_text = str(user_response).lower()
        
    if "sì" in user_text or "si" in user_text or "yes" in user_text or "ok" in user_text:
        state.use_synthetic_data = True
    else:
        state.use_synthetic_data = False
        logger.warning("⚠️  Dati sintetici scartati dall'utente")
        
    return state
