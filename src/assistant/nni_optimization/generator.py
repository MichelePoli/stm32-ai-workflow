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

def extract_files_from_code(code: str) -> Dict[str, str]:
    """
    Extract individual files from code with # FILE: markers.
    """
    if not code or '# FILE:' not in code:
        return {"main.py": code}
    
    file_pattern = re.compile(r'^# FILE:\s*(\w+(?:\.\w+)?)\s*$', re.MULTILINE)
    file_matches = list(file_pattern.finditer(code))
    
    files = {}
    for i, match in enumerate(file_matches):
        filename = match.group(1)
        start = match.end() + 1
        end = file_matches[i + 1].start() if i + 1 < len(file_matches) else len(code)
        files[filename] = code[start:end].strip()
        
    return files

def generate_nni_experiment(
    model_info: Dict[str, Any],
    dataset_info: Dict[str, Any],
    optimization_goal: str = "Maximize validation accuracy with minimal parameters",
    output_dir: str = "./nni_generated",
    model_id: str = "mistral"
) -> Dict[str, str]:
    """
    Generates NNI experiment scripts (manager.py, trial.py) using an LLM Agent.
    
    Args:
        model_info: Dictionary containing model architecture details (layers, input_shape, etc.)
        dataset_info: Dictionary containing dataset details (path, shapes, classes)
        optimization_goal: The objective of the optimization
        output_dir: Directory to save generated files
        model_id: LLM model to use (e.g., 'mistral', 'gpt-4')
        
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
    
    # Prompt Construction
    prompt = f"""You are an expert NNI (Neural Network Intelligence) Engineer.
    
    Your task is to write a COMPLETE, EXECUTABLE NNI experiment to optimize the TARGET MODEL on the TARGET DATASET.
    
    {context_desc}
    
    REQUIREMENTS:
    1. Generate TWO files:
       - `manager.py`: Configures the NNI Experiment, Search Space, and launches it.
       - `trial.py`: The trial code that loads the model, applies params, keeps the base frozen (or not), trains, and reports results.
    
    2. SEARCH SPACE Strategy:
       - Since the goal is "{optimization_goal}", define a search space that makes sense.
       - E.g., if optimizing architecture, try different numbers of dense layers or units.
       - E.g., if optimizing training, tune LR, Batch Size, Optimizer.
       
    3. IMPLEMENTATION DETAILS:
       - In `trial.py`:
         - Use `tensorflow` to load the model from `{model_info.get('path')}`.
         - Use `nni.get_next_parameter()` to get params.
         - Load data from `{dataset_info.get('path')}` (expect .npy files: x_train.npy, y_train.npy...).
         - Report final accuracy with `nni.report_final_result(acc)`.
       - In `manager.py`:
         - Use `nni.experiment.Experiment`.
         - Set `trial_command` to `python trial.py`.
         - Use `local` mode.
    
    FORMAT OUTPUT EXACTLY LIKE THIS:
    
    # FILE: manager.py
    ```python
    ... code ...
    ```
    
    # FILE: trial.py
    ```python
    ... code ...
    ```
    
    DO NOT use markdown for the whole block, just clear # FILE markers.
    """
    
    # Initialize Agent
    agent = Agent(
        model=Ollama(id=model_id),
        description="You are an AI specialized in writing NNI optimization code.",
        instructions="Return only valid Python code with # FILE markers.",
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
            logger.info(f"   ✓ Written {filename}")
            
        return files
        
    except Exception as e:
        logger.error(f"❌ Generation failed: {e}")
        return {}

if __name__ == "__main__":
    # Test stub
    model_dummy = {"name": "TestModel", "path": "model.h5", "input_shape": "(32,32,3)", "n_layers": 10}
    data_dummy = {"path": "./data", "x_shape": "(100,32,32,3)", "num_classes": 10}
    generate_nni_experiment(model_dummy, data_dummy)
