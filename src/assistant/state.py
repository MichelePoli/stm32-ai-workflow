# ============================================================================
# MASTER STATE
# ============================================================================

from dataclasses import dataclass, field
from typing import Optional, List
from datetime import datetime
from typing import Tuple, Optional, TypedDict, List, Literal
from pydantic import BaseModel, Field

@dataclass
class MasterState:
    # === INPUT MESSAGE ===
    message: str = ""
    route: str = ""
    
    # === CONFIGURAZIONE (riempiti da route_request) ===
    st_email: str = ""
    st_password: str = ""
    base_dir: str = ""
    cubemx_path: str = ""
    
    # === WEB RESEARCH ===
    search_type: str = ""
    search_query: str = ""
    search_results: str = ""
    web_research_success: bool = False
    
    # === WORKFLOW 1: STM32 FIRMWARE GENERATION ===
    ioc_file_path: Optional[str] = None
    board_name: Optional[str] = "STM32F401VCHx"
    mcu_series: str = ""
    project_name: str = "MySTM32Project"
    toolchain: str = "STM32CubeIDE"
    
    # Package Installation
    package_installation_success: bool = False
    package_error_message: Optional[str] = None
    
    firmware_script_path: str = ""
    firmware_project_path: str = ""
    firmware_script_content: str = ""
    firmware_generation_success: bool = False
    firmware_error_message: Optional[str] = None
    
    # === WORKFLOW 2: AI ANALYSIS ===
    model_path: str = ""
    target: str = "stm32f401"
    ai_output_dir: str = ""
    compression: str = "high"
    
    analyze_report_dir: str = ""
    validate_report: str = ""
    generate_code_dir: str = ""
    
    analyze_success: bool = False
    validate_success: bool = False
    generate_success: bool = False
    ai_error_message: Optional[str] = None

    # === DISCOVERY FIELDS ===
    model_discovery_method: str = ""  # "default", "taskbased", "recommendation", "search"
    available_models: List[dict] = field(default_factory=list)  # Modelli suggeriti
    selected_model: Optional[dict] = None  # Modello scelto
    search_iterations: int = 0  # Contatore ricerche online (max 3)
    model_accepted: bool = False  # Utente ha accettato il modello?
    last_task: str = ""  # image_classification, object_detection, human_activity_recognition
    custom_use_case: str = ""  # Nuovo: use case specifico utente

    # === WORKFLOW 3: INTEGRATION ===
    ai_code_dir: str = ""
    firmware_project_dir: str = ""
    network_name: str = "network"
    modify_main: bool = True
    
    ai_src_files: List[str] = field(default_factory=list)
    ai_header_files: List[str] = field(default_factory=list)
    firmware_src_dir: str = ""
    firmware_inc_dir: str = ""
    main_c_path: str = ""
    
    copy_success: bool = False
    main_modification_success: bool = False
    integration_success: bool = False
    integration_error_message: Optional[str] = None

    # === WORKFLOW 5: MODEL CUSTOMIZATION ===
    # Architecture inspection
    model_architecture: dict = field(default_factory=dict)  # Input/output shapes, n_layers, etc.
    model_summary_text: str = ""  # Testo completo del model.summary()

    wants_model_modifications: bool = False  
    modification_intent_confidence: float = 0.0  
    
    # Best practices
    best_practices_display: str = ""  # Formatted best practices per l'utente
    best_practices_raw: List[str] = field(default_factory=list)  # Raw best practices docs
    
    # User customization request
    user_custom_modifications: str = ""  # Prompt libero dell'utente
    
    # Parsed modifications
    parsed_modifications: Optional[dict] = None  # Structured modifications from LLM
    modification_confirmed: bool = False  # Utente ha confermato le modifiche?
    
    # Applied customization
    customized_model_path: str = ""  # Path del modello customizzato (.h5)
    customization_applied: bool = False  # Flag se le modifiche sono state applicate
    customized_model_info: dict = field(default_factory=dict)  # Info dopo customizzazione
    
    # Training validation
    training_test_result: dict = field(default_factory=dict)  # Risultati test training
    training_validation_success: bool = False
    
    # Final model
    final_model_path: str = ""  # Path del modello customizzato salvato definitivamente

    # Environment management
    python_path: str = ""  # Path del Python interpreter (es: /home/mrusso/miniconda3/envs/stm32_legacy/bin/python)
    conda_env: str = ""    # Nome ambiente conda (es: 'stm32_legacy', 'stm32')
    
    # Decision after customization
    continue_after_customization: bool = False  # Continua con AI analysis?
    
    # Training parameters
    custom_learning_rate: float = 0.0001  # Learning rate per fine-tuning
    custom_epochs: int = 20  # Epoche di training
    custom_batch_size: int = 32  # Batch size
    
    # Quantization (per STM32)
    should_quantize: bool = False  # Quantizzare per embedded?
    quantization_bit_width: int = 8  # INT8, INT16, etc.
    quantized_model_path: str = ""  # Path modello quantizzato (.tflite)

    # === COMMON ===
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))

    

# ============================================================================
# INPUT SCHEMA
# ============================================================================

class MasterInput(TypedDict, total=False):
    message: str

