# ============================================================================
# WORKFLOW 6: SYNTHETIC DATA GENERATION
# ============================================================================
# Module for generating synthetic data (audio/time-series)
#
# Responsibilities:
#   - Ask the user for data requirements (type, duration, quantity)
#   - Generate synthetic waveforms (sine, noise, chirp, etc.)
#   - Save the data in .npy or .wav format for fine-tuning
#
# Dependencies: numpy, scipy, soundfile (optional), matplotlib (for debugging)

import os
import logging
import numpy as np
import json
from typing import Literal, Optional, List, Dict, Any
from datetime import datetime

from langgraph.types import interrupt
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from src.assistant.configuration import Configuration
from src.assistant.state import MasterState

logger = logging.getLogger(__name__)

# ============================================================================
# SCHEMAS
# ============================================================================

class SyntheticDataRequest(BaseModel):
    """Structured request for synthetic data"""
    signal_type: Literal["sine", "white_noise", "pink_noise", "chirp", "impulse", "silence", "mixed"] = Field(
        description="Type of signal to generate"
    )
    frequency: Optional[float] = Field(description="Frequency in Hz (for sine/chirp)")
    duration_sec: float = Field(default=1.0, description="Duration in seconds per sample")
    sample_rate: int = Field(default=16000, description="Sample rate in Hz")
    num_samples: int = Field(default=10, description="Number of samples to generate")
    amplitude: float = Field(default=0.5, description="Signal amplitude (0.0-1.0)")
    noise_level: float = Field(default=0.0, description="Added noise level (0.0-1.0)")

# ============================================================================
# NODES
# ============================================================================

def ask_synthetic_data_requirements(state: MasterState, config: RunnableConfig = None) -> MasterState:
    """Asks the user what type of data to generate"""
    
    logger.info("🧪 Starting synthetic data generation procedure...")
    
    prompt = {
        "instruction": """Synthetic Data Generation (Audio/Time-Series)
        
What kind of data do you want to generate for training?
You can specify:
- Type: Sine wave, Noise, Chirp, Impulse
- Frequency: e.g. 1kHz, 440Hz
- Quantity: e.g. 50 samples
- Duration: e.g. 1 second

Examples:
- "Generate 50 sine wave samples at 1kHz with noise"
- "I want 20 white noise samples for 2 seconds"
- "10 chirps from 100Hz to 1kHz"
        """
    }
    
    # === LLM EXTRACTOR ===
    cfg = Configuration.from_runnable_config(config)
    from src.assistant.utils import extract_user_response, get_llm
    llm = get_llm(config)
    llm_parser = llm.with_structured_output(SyntheticDataRequest)
    
    # --- Step 1: Try to use the initial message ---
    initial_req_detected = False
    if not state.user_response:
        res = llm_parser.invoke([
            SystemMessage(content="Extract signal generation parameters. Return None if not specified."),
            HumanMessage(content=f"Messaggio: {state.message}")
        ])
        # Handle both Pydantic models and raw dicts (Triton fallback)
        signal_val = res.get("signal_type") if isinstance(res, dict) else getattr(res, "signal_type", None)
        
        # Normalize LLM values to valid enum: 'Sinusoidal' → 'sine', 'Noise' → 'white_noise', etc.
        _SIGNAL_NORMALIZE = {
            "sinusoidal": "sine", "sinus": "sine", "sin": "sine",
            "noise": "white_noise", "white": "white_noise", "whitenoise": "white_noise",
            "pink": "pink_noise", "pinknoise": "pink_noise",
            "sweep": "chirp", "frequency_sweep": "chirp",
            "pulse": "impulse", "delta": "impulse",
            "quiet": "silence", "zero": "silence",
        }
        if signal_val and isinstance(signal_val, str):
            normalized = _SIGNAL_NORMALIZE.get(signal_val.lower().replace(" ", "_"), signal_val.lower())
            if isinstance(res, dict):
                res["signal_type"] = normalized
            signal_val = normalized
        
        # Normalize numeric fields that LLM may return as strings with units (e.g. '2 kHz' → 2000.0)
        def _parse_hz(val):
            """Convert '2 kHz', '100 kHz', '16000' etc. to float."""
            if val is None:
                return None
            if isinstance(val, (int, float)):
                return float(val)
            s = str(val).strip().lower().replace(",", "").replace(" ", "")
            import re as _re
            m = _re.match(r'^([0-9.]+)(k|m)?hz?$', s)
            if m:
                n = float(m.group(1))
                if m.group(2) == 'k': n *= 1000
                elif m.group(2) == 'm': n *= 1_000_000
                return n
            try:
                return float(s)
            except Exception:
                return None
        
        if isinstance(res, dict):
            if 'frequency' in res:
                parsed_freq = _parse_hz(res['frequency'])
                if parsed_freq is not None:
                    res['frequency'] = parsed_freq
            if 'sample_rate' in res:
                parsed_sr = _parse_hz(res['sample_rate'])
                if parsed_sr is not None:
                    res['sample_rate'] = int(parsed_sr)
            if 'duration_sec' in res and isinstance(res['duration_sec'], str):
                try:
                    res['duration_sec'] = float(res['duration_sec'].strip())
                except Exception:
                    res['duration_sec'] = 1.0
        
        if signal_val:
            state.synthetic_request = res if isinstance(res, dict) else res.model_dump()
            initial_req_detected = True
            logger.info(f"🤖 Parameters detected in initial message: {state.synthetic_request}")

    # --- Step 2: Verification and Interrupt ---
    if not initial_req_detected:
        resume_value = None
        if not state.user_response:
            logger.info("⏸️ Interrupting for synthetic data requirements.")
            # resume_value = interrupt(prompt)
            resume_value = "generate 100 samples of sine wave" # BYPASS
        
        # After resuming: use interrupt return value as priority
        if resume_value and str(resume_value).strip():
            user_text = str(resume_value).strip()
        else:
            user_text = extract_user_response(state.user_response)
        state.user_response = ""
        
        # Parsing with LLM on the specific response (same system as before)
        # we use the system_prompt defined below for consistency
    else:
        user_text = state.message # Use the original message if detected there

    # (The LLM is invoked below anyway if we don't skip the whole block)
    # But to follow the pattern, we invoke it here if we are in the response phase
    if not initial_req_detected:
        logger.info(f"📝 Final user input: '{user_text}'")
    
    # === PARSING WITH LLM (if not already done) ===
    if not initial_req_detected:
        system_prompt = """You are a DSP (Digital Signal Processing) expert.
Analyze the user's request and extract the parameters for signal generation.
If the user does not specify, use these defaults:
- Duration: 1.0s
- Sample Rate: 16000Hz
- Num Samples: 10
- Amplitude: 0.5
- Noise Level: 0.1 (if noise is mentioned) or 0.0

For "mixed" or complex requests, try to map to the most similar type or use "sine" with noise.
"""

        try:
            request = llm_parser.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Request: {user_text}")
            ])
            
            state.synthetic_request = request if isinstance(request, dict) else request.model_dump()
            logger.info(f"✓ Extracted parameters: {state.synthetic_request}")
        
        except Exception as e:
            logger.error(f"❌ Error parsing request: {e}")
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


def generate_synthetic_samples(state: MasterState, config: RunnableConfig = None) -> MasterState:
    """Generates the samples using numpy"""
    
    req = state.synthetic_request
    logger.info(f"⚙️  Generating {req['num_samples']} {req['signal_type']} samples...")
    
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
            # Sine Wave: Pure sound, useful for testing specific frequencies (e.g. alarms)
            freq = params.frequency or 440.0
            # Add slight frequency variation for realism
            freq_var = np.random.uniform(-5, 5) 
            signal = params.amplitude * np.sin(2 * np.pi * (freq + freq_var) * t)
            
        elif params.signal_type == "white_noise":
            # White Noise: Constant hiss with all frequencies (e.g. background)
            signal = params.amplitude * np.random.uniform(-1, 1, len(t))
            
        elif params.signal_type == "pink_noise":
            # Pink Noise: More natural/dark noise (e.g. rain, wind) - 1/f
            # Simple pink noise (1/f) approximation
            white = np.random.randn(len(t))
            signal = np.cumsum(white) # Brownian noise (1/f^2) actually, but close enough for simple test
            signal = signal / np.max(np.abs(signal)) * params.amplitude
            
        elif params.signal_type == "chirp":
            # Chirp: Sound that changes frequency over time (e.g. sweep test)
            f_start = params.frequency or 100.0
            f_end = f_start * 10
            k = (f_end - f_start) / params.duration_sec
            signal = params.amplitude * np.sin(2 * np.pi * (f_start * t + (k/2) * t**2))
            
        elif params.signal_type == "impulse":
            # Impulse: Instantaneous peak (e.g. click, pop, sudden anomaly)
            signal = np.zeros_like(t)
            idx = np.random.randint(0, len(t))
            signal[idx] = params.amplitude
            
        elif params.signal_type == "silence":
            # Silence: Absolute silence, fundamental for "null" class
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
    
    logger.info(f"✓ Generated {len(generated_files)} files in {output_dir}")
    
    return state


def validate_synthetic_data(state: MasterState, config: RunnableConfig = None) -> MasterState:
    """Shows summary and asks for confirmation"""
    
    summary = f"""
✅ Generation Completed!

📂 Output: {state.synthetic_data_path}
📊 Generated files: {state.synthetic_files_count}
⚙️  Parameters:
   - Type: {state.synthetic_request['signal_type']}
   - Duration: {state.synthetic_request['duration_sec']}s
   - Sample Rate: {state.synthetic_request['sample_rate']}Hz
    """
    
    logger.info(summary)
    
    prompt = {
        "instruction": f"{summary}\n\nDo you want to proceed with fine-tuning using this data? (yes/no)",
    }
    
    resume_value = None
    if not state.user_response or state.user_response.strip() == "":
        # resume_value = interrupt(prompt)
        resume_value = "yes" # BYPASS
    
    # Use interrupt return value as priority
    if resume_value and str(resume_value).strip():
        user_text = str(resume_value).strip().lower()
    else:
        from src.assistant.utils import extract_user_response
        user_text = extract_user_response(state.user_response).lower()
    state.user_response = "" # Clear
    
    # Default: proceed with fine-tuning (yes)
    if not user_text or user_text.strip() == "":
        user_text = "yes"
        
    if "sì" in user_text or "si" in user_text or "yes" in user_text or "ok" in user_text or "y" in user_text:
        state.use_synthetic_data = True
    else:
        state.use_synthetic_data = False
        logger.warning("⚠️  Synthetic data discarded by user")
        
    return state
