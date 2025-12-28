import os
import logging
import json
import re
from typing import Dict, Any, Optional
from agno.agent import Agent
from agno.models.ollama import Ollama

# Configure logging
logger = logging.getLogger(__name__)

def normalize_code_escapes(text: str) -> str:
    """
    Normalize ALL escape sequences in code string.
    """
    if not isinstance(text, str):
        return text
    text = text.replace(r'\\n', '\\n')
    text = text.replace(r'\\t', '\\t')
    text = text.replace(r'\\r', '\\r')
    text = text.replace(r'\\"', '"')
    return text

def extract_files_from_code(code: str, verbose: bool = True) -> Dict[str, str]:
    """
    Extract individual files from code with # FILE: markers.
    More permissive regex to handle various LLM output formats.
    """
    if not code:
        return {}
        
    if '# FILE:' not in code and '#FILE:' not in code:
        if verbose:
            logger.warning("⚠️  No '# FILE:' markers found, returning as single file")
        return {"main.py": code}
    
    # More permissive regex:
    # - Allows optional space after #
    # - Allows anything after filename (e.g., comments)
    # - Captures filename with dots, underscores, dashes
    file_pattern = re.compile(r'^#\s*FILE:\s*([a-zA-Z0-9_\-\.]+)', re.MULTILINE | re.IGNORECASE)
    file_matches = list(file_pattern.finditer(code))
    
    if verbose:
        logger.info(f"   Found {len(file_matches)} file markers")
        for match in file_matches:
            logger.info(f"      - {match.group(1)}")
    
    if not file_matches:
        logger.warning("⚠️  # FILE: markers exist but regex didn't match, returning as single file")
        return {"main.py": code}
    
    files = {}
    for i, match in enumerate(file_matches):
        filename = match.group(1)
        start = match.end()
        
        # Find next file marker or end of string
        if i + 1 < len(file_matches):
            end = file_matches[i + 1].start()
        else:
            end = len(code)
        
        file_content = code[start:end].strip()
        files[filename] = file_content
        
    return files

def generate_nni_experiment(
    model_info: Dict[str, Any],
    dataset_info: Dict[str, Any],
    optimization_goal: str = "Maximize validation accuracy with minimal parameters",
    output_dir: str = "./nni_generated"
) -> Dict[str, str]:
    """
    Generates NNI experiment scripts (manager.py, trial.py) using an LLM Agent.
    
    Args:
        model_info: Dictionary containing model architecture details (layers, input_shape, etc.)
        dataset_info: Dictionary containing dataset details (path, shapes, classes)
        optimization_goal: The objective of the optimization
        output_dir: Directory to save generated files
        
    Returns:
        Dict mapping filenames to their generated content.
    """
    
    logger.info(f"🤖 Generating NNI experiment for model: {model_info.get('name', 'Unknown')}")
    
    # Construct Context Description
    context_desc = f"""
    TARGET MODEL:
    - Name: {model_info.get('name', 'Custom Model')}
    - Input Shape: {model_info.get('input_shape')}
    - Output Shape: {model_info.get('output_shape')}
    - Layers: {model_info.get('n_layers')}
    - File Path: {model_info.get('path')}
    
    TARGET DATASET:
    - Path: {dataset_info.get('path')}
    - Input Shape: {dataset_info.get('x_shape')}
    - Classes: {dataset_info.get('num_classes')}
    - Source: {dataset_info.get('source', 'numpy files')}
    
    OPTIMIZATION GOAL:
    {optimization_goal}
    """
    
    # Prompt Construction - ULTRA EXPLICIT
    prompt = f"""You are an expert NNI (Neural Network Intelligence) Engineer.

Your task: Generate EXACTLY TWO PYTHON FILES for an NNI hyperparameter optimization experiment.

{context_desc}

CRITICAL REQUIREMENTS:
🔴 YOU MUST GENERATE EXACTLY 2 FILES: manager.py AND trial.py
🔴 Both files are MANDATORY - DO NOT skip either one
🔴 Use the exact format shown below with # FILE: markers

FILE 1: manager.py
- Configure NNI experiment
- Define search space (learning_rate, batch_size, optimizer, freeze_mode)
- Set trial_command = "python trial.py"
- Launch experiment with experiment.run(port=8080)
- AFTER experiment: Get best params with experiment.export_top_trial()
- Trigger RETRAIN: Run "trial.py" with env variable RETRAIN_MODE='true' and best params

FILE 2: trial.py
- Import nni
- Check if os.environ['RETRAIN_MODE'] == 'true'
  - IF TRUE: Load params from os.environ['NNI_PARAMS']
  - IF FALSE: Get params from nni.get_next_parameter()
- Load model and data
- Resize input data
- Apply hyperparameters (freezing, optimizer, etc.)
- Train model
- IF RETRAIN_MODE: Save model to 'best_model.h5'
- IF NNI MODE: Report result with nni.report_final_result()

OUTPUT FORMAT (COPY THIS STRUCTURE EXACTLY):

# FILE: manager.py
```python
import nni
import os
import os
import sys
import json
import subprocess
from nni.experiment import Experiment

# Get absolute path to current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Define search space
search_space = {{
    'learning_rate': {{'_type': 'choice', '_value': [0.001, 0.0001, 0.00001]}},
    'batch_size': {{'_type': 'choice', '_value': [16, 32]}},
    'optimizer': {{'_type': 'choice', '_value': ['Adam', 'SGD']}},
    'freeze_mode': {{'_type': 'choice', '_value': ['freeze_base', 'train_all']}},
}}

# Create experiment
experiment = Experiment('local')
# CRITICAL: Use same Python interpreter for trials
experiment.config.trial_command = f'{{sys.executable}} trial.py'
experiment.config.trial_code_directory = current_dir  # Use absolute path
experiment.config.search_space = search_space
experiment.config.tuner.name = 'TPE'
experiment.config.tuner.class_args = {{'optimize_mode': 'maximize'}}
experiment.config.max_trial_number = 6
experiment.config.trial_concurrency = 3 # NNI lancerà 3 trial contemporaneamente

# Run with error handling
try:
    print(f"[NNI] Starting experiment in {{current_dir}}")
    print(f"[NNI] Web UI will be available at http://localhost:8080")
    experiment.run(port=8080, wait_completion=True)
    print("[NNI] Experiment completed successfully")
    
    # --- AUTO-RETRAIN BEST MODEL ---
    print("\n[NNI] 🏆 Exporting best trial...")
    best_trial = experiment.export_top_trial(top_k=1)[0]
    best_params = best_trial.parameter
    print(f"   • Best Params: {{best_params}}")
    
    print("\n[NNI] 💾 Retraining best model for saving...")
    
    # Prepare environment for retrain
    env = os.environ.copy()
    env['RETRAIN_MODE'] = 'true'
    env['NNI_PARAMS'] = json.dumps(best_params)
    
    # Re-run trial.py with best params
    subprocess.run(
        [sys.executable, 'trial.py'],
        cwd=current_dir,
        env=env,
        check=True
    )
    
    print(f"[NNI] ✅ Best model saved to: {{os.path.join(current_dir, 'best_model.h5')}}")

except Exception as e:
    print(f"[NNI] Error during experiment: {{e}}")
    import traceback
    traceback.print_exc()
finally:
    experiment.stop()
```


# FILE: trial.py
```python
import nni
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Check mode
IS_RETRAIN = os.environ.get('RETRAIN_MODE', 'false').lower() == 'true'

if IS_RETRAIN:
    print("[TRIAL] 🔄 Retraining mode activated!")
    params = json.loads(os.environ.get('NNI_PARAMS', '{}'))
else:
    # Standard NNI Mode
    params = nni.get_next_parameter()

# Get hyperparameters
lr = params.get('learning_rate', 0.001)
batch_size = params.get('batch_size', 32)
optimizer_name = params.get('optimizer', 'Adam')
freeze_mode = params.get('freeze_mode', 'freeze_base')

# Load data
x_train = np.load('{dataset_info.get('path')}/x_train.npy')
y_train = np.load('{dataset_info.get('path')}/y_train.npy')
x_test = np.load('{dataset_info.get('path')}/x_test.npy')
y_test = np.load('{dataset_info.get('path')}/y_test.npy')

# Load model
model = keras.models.load_model('{model_info.get('path')}')

# --- PARAMETER APPLICATION ---
# 1. Freeze Logic
if freeze_mode == 'freeze_base':
    print("Freezing base layers...")
    # Freeze all except last 5 layers
    for layer in model.layers[:-5]:
        layer.trainable = False
else:
    print("Unfreezing all layers...")
    for layer in model.layers:
        layer.trainable = True

# 2. Optimizer Selection
if optimizer_name == 'SGD':
    opt = keras.optimizers.SGD(learning_rate=lr)
else:
    opt = keras.optimizers.Adam(learning_rate=lr)

# --- SHAPE FIX: Resize images if needed ---
expected_shape = model.input_shape[1:3]  # (H, W)
print(f"Model expects: {{expected_shape}}, Data has: {{x_train.shape[1:3]}}")

def preprocess(x, y):
    # Resize to expected shape
    x = tf.image.resize(x, expected_shape)
    return x, y

# Create efficient tf.data pipeline
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_ds = train_ds.map(preprocess).shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
val_ds = val_ds.map(preprocess).batch(batch_size).prefetch(tf.data.AUTOTUNE)

# Compile
model.compile(optimizer=opt,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train
history = model.fit(train_ds, 
                    validation_data=val_ds,
                    epochs=3, verbose=0)

# Report (Only in NNI mode)
if not IS_RETRAIN:
    val_accuracy = history.history['val_accuracy'][-1]
    nni.report_final_result(val_accuracy)
else:
    # Save Model (Only in Retrain mode)
    output_path = 'best_model.h5'
    model.save(output_path)
    print(f"[TRIAL] ✅ Model saved to {os.path.abspath(output_path)}")
```

NOW GENERATE BOTH FILES FOLLOWING THIS EXACT FORMAT.
Start with # FILE: manager.py, then # FILE: trial.py.
DO NOT SKIP EITHER FILE.
"""
    
    # Initialize Agent with more powerful model
    agent = Agent(
        model=Ollama(id="gpt-oss:20b"),  # More powerful than mistral for complex instructions
        description="You are an AI specialized in writing NNI optimization code.",
        instructions="Return only valid Python code with # FILE markers. Generate BOTH manager.py and trial.py.",
        tools=[],
        show_tool_calls=False,
        markdown=False
    )
    
    # Generate Code
    logger.info("   ⏳ Waiting for LLM generation...")
    try:
        response = agent.run(prompt)
        content = getattr(response, "content", str(response))
        
        # Clean potential markdown wrappers
        content = normalize_code_escapes(content)
        
        # Extract files
        files = extract_files_from_code(content)
        
        if not files:
            logger.error("❌ No files extracted from LLM response")
            return {}
            
        # Save files
        os.makedirs(output_dir, exist_ok=True)
        for filename, file_content in files.items():
            # Remove markdown code blocks if present inside the file content
            file_content = file_content.replace("```python", "").replace("```", "")
            
            path = os.path.join(output_dir, filename)
            with open(path, "w") as f:
                f.write(file_content)
            logger.info(f"   ✓ Written: {path}")  # Show full path
            
        return files
        
    except Exception as e:
        logger.error(f"❌ Generation failed: {e}")
        return {}

if __name__ == "__main__":
    # Test stub
    model_dummy = {"name": "TestModel", "path": "model.h5", "input_shape": "(32,32,3)", "n_layers": 10}
    data_dummy = {"path": "./data", "x_shape": "(100,32,32,3)", "num_classes": 10}
    generate_nni_experiment(model_dummy, data_dummy)
