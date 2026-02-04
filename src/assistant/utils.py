"""
Utility functions for the STM32 AI Assistant.

This module provides centralized utilities for:
- LLM initialization and configuration
- File security (sanitization, integrity checks)
- Common helper functions
"""

import os
import re
import hashlib
import logging
import json
import urllib.request
from typing import Optional, Type, Dict, Any
from pydantic import BaseModel
from langchain_ollama import ChatOllama
from src.assistant.configuration import Configuration

logger = logging.getLogger(__name__)


# ============================================================================
# LLM UTILITIES
# ============================================================================

def get_llm(
    config: Optional[dict] = None,
    structured_schema: Optional[Type[BaseModel]] = None,
    temperature: float = 0
):
    """
    Centralized LLM initialization with consistent configuration.
    
    This replaces duplicated ChatOllama() calls across workflows,
    ensuring consistent model names and parameters.
    
    Args:
        config: RunnableConfig dict (optional). If None, uses default Configuration.
        structured_schema: Pydantic BaseModel class for structured output (optional)
        temperature: LLM temperature (default: 0 for deterministic output)
    
    Returns:
        ChatOllama instance, optionally with structured output schema
    
    Example:
        >>> # Simple LLM
        >>> llm = get_llm(config)
        >>> response = llm.invoke("Hello")
        
        >>> # Structured output
        >>> class MySchema(BaseModel):
        ...     answer: str
        >>> llm = get_llm(config, structured_schema=MySchema)
        >>> result = llm.invoke("What is 2+2?")
        >>> print(result.answer)
    """
    cfg = Configuration.from_runnable_config(config) if config else Configuration()
    
    llm = ChatOllama(
        model=cfg.local_llm,
        temperature=temperature,
        num_ctx=cfg.llm_context_window
    )
    
    if structured_schema:
        return llm.with_structured_output(structured_schema)
    
    return llm


def force_unload_ollama(model_name: str = "gpt-oss:20b"):
    """
    Force Ollama to unload the model from GPU to free up VRAM for training/NNI.
    Sends a request with keep_alive=0.
    """
    try:
        url = "http://localhost:11434/api/generate"
        data = {
            "model": model_name,
            "keep_alive": 0
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(data).encode('utf-8'),
            headers={'Content-Type': 'application/json'}
        )
        with urllib.request.urlopen(req) as response:
            logger.info(f"🔫 Ollama model '{model_name}' unloaded to free VRAM.")
            
    except Exception as e:
        logger.warning(f"⚠️ Failed to unload Ollama model: {e}")


# ============================================================================
# FILE SECURITY UTILITIES
# ============================================================================

def sanitize_filename(filename: str) -> str:
    """
    Remove path traversal attempts and dangerous characters from filename.
    
    This prevents security vulnerabilities like:
    - Path traversal: "../../../etc/passwd"
    - Command injection: "file.h5; rm -rf /"
    - Hidden files: ".bashrc"
    
    Args:
        filename: Raw filename from user/external source
    
    Returns:
        Sanitized filename safe for filesystem operations
    
    Examples:
        >>> sanitize_filename("../../../etc/passwd")
        'etc_passwd'
        >>> sanitize_filename("model; rm -rf /")
        'model__rm_-rf__'
        >>> sanitize_filename(".hidden_file.h5")
        '_hidden_file.h5'
    """
    # Remove directory separators (both Unix and Windows)
    safe_name = os.path.basename(filename)
    
    # Allow only alphanumeric, dots, underscores, hyphens
    safe_name = re.sub(r'[^a-zA-Z0-9._-]', '_', safe_name)
    
    # Prevent hidden files (starting with .)
    if safe_name.startswith('.'):
        safe_name = '_' + safe_name[1:]
    
    # Ensure not empty
    if not safe_name or safe_name == '_':
        safe_name = "unnamed_file"
    
    # Limit length to avoid filesystem issues
    if len(safe_name) > 255:
        name_part, ext = os.path.splitext(safe_name)
        safe_name = name_part[:250] + ext
    
    return safe_name


def compute_sha256(filepath: str) -> str:
    """
    Compute SHA256 hash of a file.
    
    Args:
        filepath: Path to file
    
    Returns:
        Hexadecimal SHA256 hash string (64 characters)
    
    Example:
        >>> hash_val = compute_sha256("/path/to/model.h5")
        >>> print(hash_val[:16])
        'a3d5e8f2c4b1...'
    """
    sha256 = hashlib.sha256()
    
    with open(filepath, 'rb') as f:
        # Read in chunks to handle large files efficiently
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    
    return sha256.hexdigest()


def verify_file_integrity(
    filepath: str, 
    expected_sha256: Optional[str],
    raise_on_failure: bool = False
) -> bool:
    """
    Verify file integrity against expected SHA256 hash.
    
    This prevents:
    - Corrupted downloads
    - Man-in-the-middle attacks
    - Tampered model files
    
    Args:
        filepath: Path to file to verify
        expected_sha256: Expected SHA256 hash (None to skip check)
        raise_on_failure: If True, raise SecurityError on mismatch
    
    Returns:
        True if hash matches or no expected hash provided, False otherwise
    
    Raises:
        SecurityError: If raise_on_failure=True and hash mismatch
    
    Example:
        >>> # With hash verification
        >>> verify_file_integrity(
        ...     "model.h5",
        ...     "a3d5e8f2c4b1..."
        ... )
        True
        
        >>> # Skip verification (no hash available)
        >>> verify_file_integrity("model.h5", None)
        True
    """
    if not expected_sha256:
        logger.warning(
            f"⚠️  No SHA256 provided for {os.path.basename(filepath)}, "
            f"skipping integrity check"
        )
        return True
    
    if not os.path.exists(filepath):
        logger.error(f"❌ File not found: {filepath}")
        return False
    
    actual = compute_sha256(filepath)
    expected_normalized = expected_sha256.lower().strip()
    
    if actual != expected_normalized:
        error_msg = (
            f"INTEGRITY CHECK FAILED for {os.path.basename(filepath)}!\n"
            f"  Expected SHA256: {expected_normalized}\n"
            f"  Actual SHA256:   {actual}\n"
            f"  This may indicate file corruption or a security threat."
        )
        logger.error(f"❌ {error_msg}")
        
        if raise_on_failure:
            raise SecurityError(error_msg)
        
        return False
    
    logger.info(f"✓ Integrity verified (SHA256: {actual[:16]}...)")
    return True


# ============================================================================
# MODIFICATION VALIDATION UTILITIES
# ============================================================================

def validate_modification_params(mod_type: str, params: dict, total_layers: int = 100) -> dict:
    """
    Validate and sanitize modification parameters before applying them.
    
    This provides a Second level of defense against LLM hallucination
    or malformed output.
    
    Args:
        mod_type: Type of modification (e.g., 'add_dropout')
        params: Dictionary of parameters for the modification
        total_layers: Total number of layers in the model (for capping)
        
    Returns:
        tuple: (sanitized_params, issues_list)
    """
    issues = []
    sanitized_params = params.copy()
    
    if mod_type == 'freeze_layers':
        num_frozen = params.get('num_frozen_layers')
        if num_frozen is None:
            sanitized_params['num_frozen_layers'] = 1
            issues.append("freeze_layers: num_frozen_layers was None, defaulting to 1")
        elif not isinstance(num_frozen, int):
            try:
                sanitized_params['num_frozen_layers'] = int(num_frozen)
            except:
                sanitized_params['num_frozen_layers'] = 1
                issues.append(f"freeze_layers: invalid type {type(num_frozen)}, defaulting to 1")
        
        # Cap to total layers
        if sanitized_params['num_frozen_layers'] >= total_layers:
            sanitized_params['num_frozen_layers'] = max(1, total_layers - 1)
            issues.append(f"freeze_layers: capped to {total_layers - 1}")
        elif sanitized_params['num_frozen_layers'] < 0:
            sanitized_params['num_frozen_layers'] = 0
            issues.append("freeze_layers: adjusted negative value to 0")

    elif mod_type == 'add_dropout':
        rate = params.get('rate')
        if rate is None:
            sanitized_params['rate'] = 0.3
            issues.append("add_dropout: rate was None, defaulting to 0.3")
        elif not isinstance(rate, (int, float)):
            sanitized_params['rate'] = 0.3
            issues.append("add_dropout: invalid rate type, defaulting to 0.3")
        elif not (0.0 <= sanitized_params['rate'] <= 1.0):
            sanitized_params['rate'] = 0.3
            issues.append(f"add_dropout: invalid rate {rate}, using 0.3")

    elif mod_type == 'change_learning_rate':
        lr = params.get('learning_rate')
        if lr is None or not isinstance(lr, (int, float)) or lr <= 0 or lr > 1:
            sanitized_params['learning_rate'] = 0.0001
            issues.append(f"change_learning_rate: invalid or missing LR, using 0.0001")

    return sanitized_params, issues


# ============================================================================
# SUBPROCESS UTILITIES
# ============================================================================

def run_subprocess_streaming(
    cmd: list, 
    logger_instance, 
    prefix: str = "[Subprocess]",
    timeout: int = 600,
    ignore_list: list = None
) -> dict:
    """
    Run a subprocess and stream its output to a logger in real-time.
    
    Args:
        cmd: List of command arguments
        logger_instance: Logger to output to
        prefix: Prefix for each logged line
        timeout: Execution timeout in seconds
        ignore_list: Optional list of strings to suppress from logs if found in a line
        
    Returns:
        Dictionary with success, stdout, and returncode
    """
    import subprocess
    import time
    
    # Truncate command log if too long to avoid "wall of code"
    full_cmd = ' '.join(cmd)
    if len(full_cmd) > 100:
        if '-c' in cmd:
            idx = cmd.index('-c')
            if idx + 1 < len(cmd):
                script = cmd[idx+1]
                first_line = script.strip().split('\n')[0][:60]
                num_lines = len(script.strip().split('\n'))
                summary = f"{cmd[0]} -c \"{first_line}...\" ({num_lines} lines)"
                logger_instance.info(f"🚀 Running: {summary}")
            else:
                logger_instance.info(f"🚀 Running: {full_cmd[:100]}...")
        else:
            logger_instance.info(f"🚀 Running: {full_cmd[:100]}...")
    else:
        logger_instance.info(f"🚀 Running: {full_cmd}")
    
    stdout_lines = []
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Merge stderr into stdout
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        start_time = time.time()
        
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            
            if line:
                clean_line = line.strip()
                if clean_line:
                    # Optional filtering
                    if ignore_list and any(x in clean_line for x in ignore_list):
                        stdout_lines.append(line)
                        continue
                        
                    logger_instance.info(f"  {prefix} {clean_line}")
                    stdout_lines.append(line)
            
            # Manual timeout check
            if time.time() - start_time > timeout:
                process.terminate()
                return {
                    'success': False,
                    'stdout': "".join(stdout_lines),
                    'error': "Timeout expired"
                }

        process.wait()
        return {
            'success': process.returncode == 0,
            'stdout': "".join(stdout_lines),
            'returncode': process.returncode
        }
        
    except Exception as e:
        return {
            'success': False,
            'stdout': "".join(stdout_lines),
            'error': str(e)
        }


# ============================================================================
# COMMON HELPERS
# ============================================================================

def extract_user_response(response: Any) -> str:
    """
    Extract a clean string from various possible interrupt response formats.
    
    LangGraph's interrupt() can return different types based on how the client
    sends data. This helper standardizes the extraction of the main text.
    
    Args:
        response: The raw response from interrupt()
        
    Returns:
        The extracted user string, or empty string if not found
    """
    from typing import Any
    
    if response is None:
        return ""
    if isinstance(response, (str, int, float)):
        return str(response).strip()
    
    if isinstance(response, dict):
        # Priority 1: Specific known keys
        for key in ["response", "message", "input", "text"]:
            if key in response and response[key]:
                return str(response[key]).strip()
        
        # Priority 2: If it's a single key dict, take the value
        if len(response) == 1:
            val = list(response.values())[0]
            if val:
                return str(val).strip()
    
    # Fallback: string representation
    return str(response).strip()


# ============================================================================
# EXCEPTIONS
# ============================================================================

class SecurityError(Exception):
    """Raised when a security check fails (integrity, path traversal, etc.)"""
    pass
