# ============================================================================
# WORKFLOW 3: INTEGRATION AI CODICE NEL FIRMWARE STM32
# ============================================================================
# Modulo dedicato all'integrazione del codice AI generato nel progetto firmware
#
# Responsabilità:
#   - Raccolta percorsi progetto firmware e codice AI
#   - Scansione file AI (.c, .h) da cartella generata
#   - Copia file AI nella struttura del firmware (Src, Inc)
#   - Modifica main.c per includere init e inference call
#   - Verifica link e consistenza
#
# Dipendenze: langgraph, langchain, os, shutil, re

import os
import subprocess
import shutil
import re
import logging
from typing import Optional, List, Literal
from datetime import datetime

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from src.assistant.configuration import Configuration
from src.assistant.state import MasterState

logger = logging.getLogger(__name__)

# ============================================================================
# EXTRACTION SCHEMAS - WORKFLOW 3
# ============================================================================

class IntegrationInfoExtraction(BaseModel):
    """Schema per estrarre percorsi progetto da risposta naturale"""
    firmware_project_dir: Optional[str] = Field(
        default=None,
        description="Path completo al progetto firmware generato"
    )
    ai_code_dir: Optional[str] = Field(
        default=None,
        description="Path completo alla cartella contenente il codice AI generato"
    )


class ModificationConfirmation(BaseModel):
    """Conferma delle modifiche proposte"""
    proceed_with_modifications: bool = Field(
        description="L'utente vuole procedere con le modifiche proposte?"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidenza della decisione"
    )


# ============================================================================
# EXTRACTION INSTRUCTIONS - WORKFLOW 3
# ============================================================================

integration_info_extraction_instructions = """Sei un estrattore di percorsi per l'integrazione AI-Firmware.

Analizza la risposta dell'utente e estrai i seguenti campi:

1. **firmware_project_dir**: Path completo al progetto firmware generato
   Esempi:
   - "/Users/user/STM32CubeMX/MyProject"
   - "~/STM32Projects/MyProject"
   - "/home/user/firmware/project"
   → Se non specificato: null

2. **ai_code_dir**: Path completo alla cartella contenente il codice AI generato
   Esempi:
   - "./analisiAI/code_resnet"
   - "~/results/code_output"
   - "/tmp/ai_analysis/code"
   → Se non specificato: null

Rispondi SEMPRE in formato JSON valido, anche se alcuni campi sono null.

Esempi:
- Input: "Firmware in /home/user/MyProject, AI code in ./ai_output/code"
  Output: {"firmware_project_dir": "/home/user/MyProject", "ai_code_dir": "./ai_output/code"}

- Input: "Usa i percorsi default"
  Output: {"firmware_project_dir": null, "ai_code_dir": null}
"""

modification_confirmation_instructions = """Analizza la risposta dell'utente e classifica se vuole procedere:

L'utente risponde a: "Procediamo con le modifiche al firmware?"

Risposte di ACCETTAZIONE: "si", "sì", "yes", "ok", "procedi", "avanti", "continua"
Risposte di RIFIUTO: "no", "nope", "fermati", "stop", "rivedi", "no grazie"

Rispondi SEMPRE in formato JSON con:
- "proceed_with_modifications": true/false
- "confidence": 0.0-1.0
"""


# ============================================================================
# NODI WORKFLOW 3
# ============================================================================


def collect_integration_info(state: MasterState, config: RunnableConfig = None) -> MasterState:
    """
    Raccoglie info integrazione da risposta naturale dell'utente.
    La risposta viene analizzata da LLM per estrarre i path.
    Se i path sono già presenti nello state (da run precedente), usali.
    """
    
    prompt = {
        "instruction": """Configurazione Integrazione AI nel Firmware
            
Per favore specifica (in linguaggio naturale):
- Path completo al progetto firmware generato
- Path completo al codice AI generato

Esempi di path:
  Firmware: ~/STM32CubeMX/MySTM32Project
  AI code: ./analisiAI/code_resnet

Esempio risposta: "Integra il codice da ./analisiAI/code_resnet nel firmware di ~/STM32CubeMX/MySTM32Project"
            """,
    }
    
    # === ESTRATTORE LLM ===
    from src.assistant.utils import extract_user_response, get_llm
    cfg = Configuration.from_runnable_config(config)
    llm = get_llm(config)
    llm_extractor = llm.with_structured_output(IntegrationInfoExtraction)
    
    # --- Passo 1: Prova a usare il messaggio iniziale (se non siamo in fase di risposta) ---
    if not state.user_response:
        res = llm_extractor.invoke([
            SystemMessage(content=integration_info_extraction_instructions),
            HumanMessage(content=f"Messaggio: {state.message}")
        ])
        if res.firmware_project_dir: state.firmware_project_dir = res.firmware_project_dir
        if res.ai_code_dir: state.ai_code_dir = res.ai_code_dir

    # --- Passo 2: Verifica e Interrupt ---
    if not state.firmware_project_dir or not state.ai_code_dir:
        resume_value = None
        if not state.user_response:
            # Recupera path progetto dal profilo
            last_fw = state.persistent_context.get("last_project_path", "Nessuno") if state.persistent_context else "Nessuno"
            
            dynamic_prompt = {
                "instruction": prompt["instruction"],
                "suggestion": f"💡 Ho visto che l'ultimo progetto era: **{last_fw}**. Vuoi usare lo stesso path o uno nuovo?"
            }
            logger.info("⏸️ Interrupting for integration paths with profile suggestion.")
            resume_value = interrupt(dynamic_prompt)
        
        # Dopo la ripresa: usa interrupt return value come priorità
        if resume_value and str(resume_value).strip():
            user_text = str(resume_value).strip()
        else:
            user_text = extract_user_response(state.user_response)
        state.user_response = ""
        
        res = llm_extractor.invoke([
            SystemMessage(content=integration_info_extraction_instructions),
            HumanMessage(content=f"Risposta: {user_text}")
        ])
        
        # Gestione "USE_PROFILE" (Aggiungiamo istruzioni all'estrattore dopo)
        context_fw = state.persistent_context.get("last_project_path") if state.persistent_context else None
        
        # Logica euristica se LLM non ha istruzioni USE_PROFILE qui
        text_low = user_text.lower()
        if ("precedente" in text_low or "profilo" in text_low or "si" == text_low.strip()) and context_fw:
            state.firmware_project_dir = context_fw
            logger.info(f"📋 Applicato path progetto da profilo: {state.firmware_project_dir}")
        else:
            if res.firmware_project_dir: state.firmware_project_dir = res.firmware_project_dir
        
        if res.ai_code_dir: state.ai_code_dir = res.ai_code_dir

    # --- Passo 3: Finalizzazione e Validazione ---
    if not state.firmware_project_dir:
        # Fallback profilo se ancora vuoto
        state.firmware_project_dir = state.persistent_context.get("last_project_path") if state.persistent_context else None
    
    if not state.firmware_project_dir:
        raise ValueError("firmware_project_dir è obbligatorio per l'integrazione")
    
    if not state.ai_code_dir:
        raise ValueError("ai_code_dir è obbligatorio per l'integrazione")

    logger.info(f"✓ Configurazione finale: FW={state.firmware_project_dir}, AI={state.ai_code_dir}")
    
    # === ESPANDI I PATH (~ e variabili d'ambiente) ===
    
    firmware_project_expanded = os.path.expanduser(state.firmware_project_dir)
    ai_code_expanded = os.path.expanduser(state.ai_code_dir)
    
    logger.info(f"📂 Path espansi:")
    logger.info(f"  firmware_project_dir: {firmware_project_expanded}")
    logger.info(f"  ai_code_dir: {ai_code_expanded}")
    
    # === VERIFICA CHE I PATH ESISTANO ===
    
    if not os.path.exists(firmware_project_expanded):
        raise FileNotFoundError(f"❌ Progetto firmware non trovato: {state.firmware_project_dir}")
    
    if not os.path.exists(ai_code_expanded):
        raise FileNotFoundError(f"❌ Codice AI non trovato: {state.ai_code_dir}")
    
    logger.info("✓ Entrambi i path verificati")
    
    # === RILEVA STRUTTURA PROGETTO FIRMWARE ===
    
    proj_root = firmware_project_expanded
    entries = [e for e in os.listdir(proj_root) if not e.startswith('.')]
    
    # Se c'è una singola sottocartella, usala come root
    if len(entries) == 1 and os.path.isdir(os.path.join(proj_root, entries[0])):
        logger.info(f"📁 Rilevata sottocartella '{entries[0]}': usando come project root")
        proj_root = os.path.join(proj_root, entries[0])
        state.firmware_project_dir = proj_root
    
    logger.info(f"📂 Cercando struttura progetto in: {proj_root}")
    
    # === RILEVA LAYOUT PROGETTO (Src/Inc o Core/Src-Inc) ===
    
    std_src = os.path.join(proj_root, "Src")
    std_inc = os.path.join(proj_root, "Inc")
    core_src = os.path.join(proj_root, "Core", "Src")
    core_inc = os.path.join(proj_root, "Core", "Inc")
    
    if os.path.exists(std_src) and os.path.exists(std_inc):
        logger.info("✓ Struttura STM32 standard rilevata: Src/Inc")
        state.firmware_src_dir = std_src
        state.firmware_inc_dir = std_inc
        state.main_c_path = os.path.join(std_src, "main.c")
    elif os.path.exists(core_src) and os.path.exists(core_inc):
        logger.info("✓ Struttura STM32 Cube rilevata: Core/Src-Inc")
        state.firmware_src_dir = core_src
        state.firmware_inc_dir = core_inc
        state.main_c_path = os.path.join(core_src, "main.c")
    else:
        logger.error(f"❌ Struttura progetto non riconosciuta in {proj_root}")
        logger.error(f"   Cercate: Src/Inc o Core/Src-Inc")
        raise FileNotFoundError(f"Struttura progetto non riconosciuta in {proj_root}")
    
    # === VERIFICA FILE MAIN.C ===
    
    if not os.path.exists(state.main_c_path):
        logger.warning(f"⚠️  main.c non trovato in {state.main_c_path}")
        logger.warning(f"    Continuo comunque (potrebbe essere in altra locazione)")
    else:
        logger.info(f"✓ main.c trovato: {state.main_c_path}")
    
    # === VERIFICA CODICE AI ===
    
    ai_files = os.listdir(ai_code_expanded)
    c_files = [f for f in ai_files if f.endswith('.c')]
    h_files = [f for f in ai_files if f.endswith('.h')]
    
    logger.info(f"📂 Codice AI trovato:")
    logger.info(f"  File .c: {len(c_files)} ({', '.join(c_files[:3])}{'...' if len(c_files) > 3 else ''})")
    logger.info(f"  File .h: {len(h_files)} ({', '.join(h_files[:3])}{'...' if len(h_files) > 3 else ''})")
    
    if not c_files and not h_files:
        logger.warning("⚠️  Nessun file .c o .h trovato nella cartella AI code")
    
    logger.info("✓ Configurazione integrazione raccolta e validata")
    return state


def scan_ai_files(state: MasterState, config: RunnableConfig = None) -> MasterState:
    logger.info("Scansione file AI...")
    
    try:
        state.ai_src_files = []
        state.ai_header_files = []
        
        for file in os.listdir(state.ai_code_dir):
            file_path = os.path.join(state.ai_code_dir, file)
            if os.path.isfile(file_path):
                if file.endswith('.c'):
                    state.ai_src_files.append(file_path)
                elif file.endswith('.h'):
                    state.ai_header_files.append(file_path)
        
        logger.info(f"✓ Trovati {len(state.ai_src_files)} .c, {len(state.ai_header_files)} .h")
        
        if not state.ai_src_files and not state.ai_header_files:
            raise FileNotFoundError("Nessun file .c o .h trovato")
        
    except Exception as e:
        state.integration_error_message = f"Errore scansione: {str(e)}"
        logger.error(state.integration_error_message)
    
    return state


def copy_ai_files(state: MasterState, config: RunnableConfig = None) -> MasterState:
    logger.info("Copia file AI nel firmware...")
    
    try:
        for src_file in state.ai_src_files:
            filename = os.path.basename(src_file)
            dest_path = os.path.join(state.firmware_src_dir, filename)
            shutil.copy2(src_file, dest_path)
            logger.info(f"  Copiato: {filename}")
        
        for header_file in state.ai_header_files:
            filename = os.path.basename(header_file)
            dest_path = os.path.join(state.firmware_inc_dir, filename)
            shutil.copy2(header_file, dest_path)
            logger.info(f"  Copiato: {filename}")
        
        state.copy_success = True
        logger.info("✓ Copia completata")
        
    except Exception as e:
        state.copy_success = False
        state.integration_error_message = f"Errore copia: {str(e)}"
        logger.error(state.integration_error_message)
    
    return state


def modify_main_c(state: MasterState, config: RunnableConfig = None) -> MasterState:
    if not state.modify_main:
        logger.info("Modifica main.c saltata")
        state.main_modification_success = True
        return state
    
    logger.info("Modifica main.c...")
    
    try:
        if not os.path.exists(state.main_c_path):
            raise FileNotFoundError(f"File main.c non trovato: {state.main_c_path}")
        
        with open(state.main_c_path, 'r', encoding='utf-8') as f:
            main_content = f.read()
        
        backup_path = f"{state.main_c_path}.backup_{state.timestamp}"
        shutil.copy2(state.main_c_path, backup_path)
        logger.info(f"Backup creato: {backup_path}")
        
        net_name = state.network_name if state.network_name else "network"
        net_upper = net_name.upper()

        # 1. Includes
        includes_pattern = r'(\/\* USER CODE BEGIN Includes \*\/)'
        ai_includes = f'\n/* AI includes */\n#include "{net_name}.h"\n#include "{net_name}_data.h"\n'
        if re.search(includes_pattern, main_content) and f'#include "{net_name}.h"' not in main_content:
            main_content = re.sub(includes_pattern, r'\1' + ai_includes, main_content)

        # 2. Private Variables (PV)
        pv_pattern = r'(\/\* USER CODE BEGIN PV \*\/)'
        ai_pv = f'''
/* AI Variables */
static ai_handle {net_name} = AI_HANDLE_NULL;
static ai_u8 activations[AI_{net_upper}_DATA_ACTIVATIONS_SIZE];
static ai_float in_data[AI_{net_upper}_IN_1_SIZE];
static ai_float out_data[AI_{net_upper}_OUT_1_SIZE];

static ai_buffer ai_input[AI_{net_upper}_IN_NUM];
static ai_buffer ai_output[AI_{net_upper}_OUT_NUM];
'''
        if re.search(pv_pattern, main_content) and f'static ai_handle {net_name}' not in main_content:
             main_content = re.sub(pv_pattern, r'\1' + ai_pv, main_content)

        # 3. Initialization (BEGIN 2)
        init_pattern = r'(\/\* USER CODE BEGIN 2 \*\/)'
        ai_init = f'''
  /* AI Initialization */
  ai_error err;
  ai_network_params params;
  
  /* Get the weights/params from the data module */
  if (!ai_{net_name}_data_params_get(&params)) {{
      Error_Handler();  /* STM32 HAL standard error handler */
      /* Alternatively, implement custom error recovery */
  }}
  
  /* Set the activations buffer */
  params.activations = AI_HANDLE_PTR(activations);
  
  /* Create the network */
  err = ai_{net_name}_create(&{net_name}, AI_{net_upper}_DATA_CONFIG);
  if (err.type != AI_ERROR_NONE) {{
      Error_Handler();  /* Network creation failed */
      /* Error code available in err.type and err.code */
  }}
  
  /* Initialize the network */
  if (!ai_{net_name}_init({net_name}, &params)) {{
      Error_Handler();  /* Network initialization failed */
      /* Check params.activations buffer allocation */
  }}
  
  /* Initialize input/output buffers */
  ai_input[0] = ai_{net_name}_inputs_get({net_name}, NULL)[0];
  ai_output[0] = ai_{net_name}_outputs_get({net_name}, NULL)[0];
  
  ai_input[0].data = AI_HANDLE_PTR(in_data);
  ai_output[0].data = AI_HANDLE_PTR(out_data);
'''
        if re.search(init_pattern, main_content) and f'ai_{net_name}_create' not in main_content:
            main_content = re.sub(init_pattern, r'\1' + ai_init, main_content)

        # 4. Inference Loop (WHILE)
        while_pattern = r'(\/\* USER CODE BEGIN WHILE \*\/)'
        ai_while = f'''
    /* AI Inference */
    /* Fill in_data with sensor data (ADC, I2C, SPI, etc.)
       Example:
       HAL_ADC_Start(&hadc1);
       HAL_ADC_PollForConversion(&hadc1, HAL_MAX_DELAY);
       in_data[0] = (ai_float)HAL_ADC_GetValue(&hadc1) / 4095.0f;
       ... (fill remaining input features)
    */
    
    if (ai_{net_name}_run({net_name}, &ai_input[0], &ai_output[0]) != 1) {{
        /* Inference failed - blink error LED */
        HAL_GPIO_WritePin(LED_ERROR_GPIO_Port, LED_ERROR_Pin, GPIO_PIN_SET);
        HAL_Delay(100);
        HAL_GPIO_WritePin(LED_ERROR_GPIO_Port, LED_ERROR_Pin, GPIO_PIN_RESET);
    }}
    
    /* Process inference results
       Example for classification:
       int predicted_class = 0;
       ai_float max_prob = out_data[0];
       for (int i = 1; i < AI_{net_upper}_OUT_1_SIZE; i++) {{
           if (out_data[i] > max_prob) {{
               max_prob = out_data[i];
               predicted_class = i;
           }}
       }}
       // Use predicted_class for application logic (actuators, display, etc.)
    */
'''
        if re.search(while_pattern, main_content) and f'ai_{net_name}_run' not in main_content:
            main_content = re.sub(while_pattern, r'\1' + ai_while, main_content)

        with open(state.main_c_path, 'w', encoding='utf-8') as f:
            f.write(main_content)
        
        state.main_modification_success = True
        logger.info("✓ main.c modificato con logica di inferenza completa")
        
    except Exception as e:
        state.main_modification_success = False
        state.integration_error_message = f"Errore modifica main.c: {str(e)}"
        logger.error(state.integration_error_message)
    
    return state


def verify_integration(state: MasterState, config: RunnableConfig = None) -> MasterState:
    logger.info("Verifica integrazione...")
    
    try:
        all_files_copied = True
        for src_file in state.ai_src_files:
            filename = os.path.basename(src_file)
            dest_path = os.path.join(state.firmware_src_dir, filename)
            if not os.path.exists(dest_path):
                all_files_copied = False
                logger.error(f"File mancante: {dest_path}")
        
        for header_file in state.ai_header_files:
            filename = os.path.basename(header_file)
            dest_path = os.path.join(state.firmware_inc_dir, filename)
            if not os.path.exists(dest_path):
                all_files_copied = False
                logger.error(f"File mancante: {dest_path}")
        
        state.integration_success = (state.copy_success and all_files_copied and state.main_modification_success)
        
        if state.integration_success:
            logger.info("✓ Integrazione verificata")
        else:
            logger.error("✗ Verifica integrazione fallita")
        
    except Exception as e:
        state.integration_success = False
        state.integration_error_message = f"Errore verifica: {str(e)}"
        logger.error(state.integration_error_message)
    
    return state


def finalize_integration(state: MasterState, config: RunnableConfig = None) -> MasterState:
    if state.integration_success:
        print("✓ INTEGRAZIONE COMPLETATA CON SUCCESSO!")
        print(f"✓ File AI copiati: {len(state.ai_src_files)} .c, {len(state.ai_header_files)} .h")
        print(f"✓ main.c modificato")
        print(f"\nProgetto finale pronto in: {state.firmware_project_dir}")
        print("\nProssimi passi:")
        print(f"1. Compila: cd {state.firmware_project_dir} && make -j8")
        print(f"2. Flash su hardware STM32")
    else:
        print(f"✗ Integrazione fallita: {state.integration_error_message}")
    
    return state
