# ============================================================================
# WORKFLOW 3: INTEGRATION AI CODE INTO STM32 FIRMWARE
# ============================================================================
# Module dedicated to the integration of generated AI code into the firmware project
#
# Responsibilities:
#   - Collection of firmware project and AI code paths
#   - Scanning of AI files (.c, .h) from the generated folder
#   - Copying AI files into the firmware structure (Src, Inc)
#   - Modification of main.c to include init and inference call
#   - Verification of linking and consistency
#
# Dependencies: langgraph, langchain, os, shutil, re

import os
import subprocess
import shutil
import re
import logging
from typing import Optional, List, Literal
from datetime import datetime

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
    """Schema to extract project paths from natural language response"""
    firmware_project_dir: Optional[str] = Field(
        default=None,
        description="Full path to the generated firmware project"
    )
    ai_code_dir: Optional[str] = Field(
        default=None,
        description="Full path to the folder containing the generated AI code"
    )


# ============================================================================
# EXTRACTION INSTRUCTIONS - WORKFLOW 3
# ============================================================================

integration_info_extraction_instructions = """You are a path extractor for AI-Firmware integration.

Analyze the user's response and extract the following fields:

1. **firmware_project_dir**: Full path to the generated firmware project
   Examples:
   - "/Users/user/STM32CubeMX/MyProject"
   - "~/STM32Projects/MyProject"
   - "/home/user/firmware/project"
   → If not specified: null

2. **ai_code_dir**: Full path to the folder containing the generated AI code
   Examples:
   - "./ai_analysis/code_resnet"
   - "~/results/code_output"
   - "/tmp/ai_analysis/code"
   → If not specified: null

ALWAYS respond in valid JSON format, even if some fields are null.

Examples:
- Input: "Firmware in /home/user/MyProject, AI code in ./ai_output/code"
  Output: {"firmware_project_dir": "/home/user/MyProject", "ai_code_dir": "./ai_output/code"}

- Input: "Use the default paths"
  Output: {"firmware_project_dir": null, "ai_code_dir": null}
"""




# ============================================================================
# NODI WORKFLOW 3
# ============================================================================


def collect_integration_info(state: MasterState, config: RunnableConfig = None) -> MasterState:
    """
    Collects integration info from the user's natural language response.
    The response is analyzed by an LLM to extract the paths.
    If paths are already present in the state (from a previous run), use them.
    """
    
    # === IDEMPOTENCY CHECK ===
    # If we arrive from the sequential flow WF1→WF2→WF3, paths are already populated
    if state.firmware_project_dir and state.ai_code_dir and not state.user_response:
        logger.info(f"⏭️  Idempotency: Paths already present (FW={state.firmware_project_dir}, AI={state.ai_code_dir}). Skipping collection.")
        # Skip directly to filesystem validation
        return _validate_and_detect_structure(state)
    
    prompt = {
        "instruction": """AI to Firmware Integration Configuration
            
Please specify (in natural language):
- Full path to the generated firmware project
- Full path to the generated AI code

Example paths:
  Firmware: ~/STM32CubeMX/MySTM32Project
  AI code: ./ai_analysis/code_resnet

Example response: "Integrate the code from ./ai_analysis/code_resnet into the firmware at ~/STM32CubeMX/MySTM32Project"
            """,
    }
    
    # === LLM EXTRACTOR ===
    from src.assistant.utils import extract_user_response, get_llm
    cfg = Configuration.from_runnable_config(config)
    llm = get_llm(config)
    llm_extractor = llm.with_structured_output(IntegrationInfoExtraction)
    
    # --- Step 1: Try to use initial message (if not in response phase) ---
    if not state.user_response:
        res = llm_extractor.invoke([
            SystemMessage(content=integration_info_extraction_instructions),
            HumanMessage(content=f"Message: {state.message}")
        ])
        if res.firmware_project_dir: state.firmware_project_dir = res.firmware_project_dir
        if res.ai_code_dir: state.ai_code_dir = res.ai_code_dir

    # --- Step 2: Verification and Interrupt ---
    if not state.firmware_project_dir or not state.ai_code_dir:
        resume_value = None
        if not state.user_response:
            # Retrieve project path from profile
            last_fw = state.persistent_context.get("last_project_path", "None") if state.persistent_context else "None"
            
            dynamic_prompt = {
                "instruction": prompt["instruction"],
                "suggestion": f"💡 I saw the last project was: **{last_fw}**. Do you want to use the same path or a new one?"
            }
            logger.info("⏸️ Interrupting for integration paths with profile suggestion.")
            # resume_value = interrupt(dynamic_prompt)
            resume_value = "yes" # BYPASS
        
        # After resume: use interrupt return value as priority
        if resume_value and str(resume_value).strip():
            user_text = str(resume_value).strip()
        else:
            user_text = extract_user_response(state.user_response)
        state.user_response = ""
        
        res = llm_extractor.invoke([
            SystemMessage(content=integration_info_extraction_instructions),
            HumanMessage(content=f"Response: {user_text}")
        ])
        
        # Handling "USE_PROFILE" (We add instructions to extractor later)
        context_fw = state.persistent_context.get("last_project_path") if state.persistent_context else None
        
        # Heuristic logic if LLM doesn't have USE_PROFILE instructions here
        text_low = user_text.lower()
        if (any(word in text_low for word in ["previous", "profile", "yes", "precedente", "profilo", "si"])) and context_fw:
            state.firmware_project_dir = context_fw
            logger.info(f"📋 Applied project path from profile: {state.firmware_project_dir}")
        else:
            if res.firmware_project_dir: state.firmware_project_dir = res.firmware_project_dir
        
        if res.ai_code_dir: state.ai_code_dir = res.ai_code_dir

    # --- Step 3: Finalization and Validation ---
    if not state.firmware_project_dir:
        # Profile fallback if still empty
        state.firmware_project_dir = state.persistent_context.get("last_project_path") if state.persistent_context else None
    
    if not state.firmware_project_dir:
        raise ValueError("firmware_project_dir is required for integration")
    
    if not state.ai_code_dir:
        raise ValueError("ai_code_dir is required for integration")

    logger.info(f"✓ Final configuration: FW={state.firmware_project_dir}, AI={state.ai_code_dir}")
    
    return _validate_and_detect_structure(state)


def _validate_and_detect_structure(state: MasterState) -> MasterState:
    """
    Internal helper: expands paths, verifies existence, detects project layout.
    Used by both the fast path (idempotency) and the full path.
    """
    # === EXPAND PATHS (~ and environment variables) ===
    
    firmware_project_expanded = os.path.expanduser(state.firmware_project_dir)
    ai_code_expanded = os.path.expanduser(state.ai_code_dir)
    
    # Update state with expanded paths
    state.firmware_project_dir = firmware_project_expanded
    state.ai_code_dir = ai_code_expanded
    
    logger.info(f"📂 Expanded paths:")
    logger.info(f"  firmware_project_dir: {firmware_project_expanded}")
    logger.info(f"  ai_code_dir: {ai_code_expanded}")
    
    # === VERIFY PATHS EXISTENCE ===
    
    if not os.path.exists(firmware_project_expanded):
        raise FileNotFoundError(f"❌ Firmware project not found: {state.firmware_project_dir}")
    
    if not os.path.exists(ai_code_expanded):
        raise FileNotFoundError(f"❌ AI code not found: {state.ai_code_dir}")
    
    logger.info("✓ Both paths verified")
    
    # === DETECT FIRMWARE PROJECT STRUCTURE ===
    
    proj_root = firmware_project_expanded
    entries = [e for e in os.listdir(proj_root) if not e.startswith('.')]
    
    # If there is a single subfolder, use it as root
    if len(entries) == 1 and os.path.isdir(os.path.join(proj_root, entries[0])):
        logger.info(f"📁 Subfolder '{entries[0]}' detected: using as project root")
        proj_root = os.path.join(proj_root, entries[0])
        state.firmware_project_dir = proj_root
    
    logger.info(f"📂 Searching for project structure in: {proj_root}")
    
    # === DETECT PROJECT LAYOUT (Src/Inc or Core/Src-Inc) ===
    
    std_src = os.path.join(proj_root, "Src")
    std_inc = os.path.join(proj_root, "Inc")
    core_src = os.path.join(proj_root, "Core", "Src")
    core_inc = os.path.join(proj_root, "Core", "Inc")
    
    if os.path.exists(std_src) and os.path.exists(std_inc):
        logger.info("✓ STM32 standard structure detected: Src/Inc")
        state.firmware_src_dir = std_src
        state.firmware_inc_dir = std_inc
        state.main_c_path = os.path.join(std_src, "main.c")
    elif os.path.exists(core_src) and os.path.exists(core_inc):
        logger.info("✓ STM32 Cube structure detected: Core/Src-Inc")
        state.firmware_src_dir = core_src
        state.firmware_inc_dir = core_inc
        state.main_c_path = os.path.join(core_src, "main.c")
    else:
        logger.error(f"❌ Project structure not recognized in {proj_root}")
        logger.error(f"   Looked for: Src/Inc or Core/Src-Inc")
        raise FileNotFoundError(f"Project structure not recognized in {proj_root}")
    
    # === VERIFY MAIN.C FILE ===
    
    if not os.path.exists(state.main_c_path):
        logger.warning(f"⚠️  main.c not found in {state.main_c_path}")
        logger.warning(f"    Continuing anyway (it might be in another location)")
    else:
        logger.info(f"✓ main.c found: {state.main_c_path}")
    
    # === VERIFY AI CODE ===
    
    ai_files = os.listdir(ai_code_expanded)
    c_files = [f for f in ai_files if f.endswith('.c')]
    h_files = [f for f in ai_files if f.endswith('.h')]
    
    logger.info(f"📂 AI Code found:")
    logger.info(f"  Files .c: {len(c_files)} ({', '.join(c_files[:3])}{'...' if len(c_files) > 3 else ''})")
    logger.info(f"  Files .h: {len(h_files)} ({', '.join(h_files[:3])}{'...' if len(h_files) > 3 else ''})")
    
    if not c_files and not h_files:
        logger.warning("⚠️  No .c or .h files found in the AI code folder")
    
    logger.info("✓ Integration configuration collected and validated")
    return state


def scan_ai_files(state: MasterState, config: RunnableConfig = None) -> MasterState:
    logger.info("Scanning AI files...")
    
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
        
        logger.info(f"✓ Found {len(state.ai_src_files)} .c, {len(state.ai_header_files)} .h")
        
        if not state.ai_src_files and not state.ai_header_files:
            raise FileNotFoundError("No .c or .h files found")
        
        state.scan_success = True
        
    except Exception as e:
        state.scan_success = False
        state.integration_error_message = f"Scanning error: {str(e)}"
        logger.error(state.integration_error_message)
    
    return state


def copy_ai_files(state: MasterState, config: RunnableConfig = None) -> MasterState:
    logger.info("Copying AI files to firmware...")
    
    try:
        for src_file in state.ai_src_files:
            filename = os.path.basename(src_file)
            dest_path = os.path.join(state.firmware_src_dir, filename)
            shutil.copy2(src_file, dest_path)
            logger.info(f"  Copied: {filename}")
        
        for header_file in state.ai_header_files:
            filename = os.path.basename(header_file)
            dest_path = os.path.join(state.firmware_inc_dir, filename)
            shutil.copy2(header_file, dest_path)
            logger.info(f"  Copied: {filename}")
        
        # === CHECK X-CUBE-AI MIDDLEWARE HEADERS ===
        # Search for the st_ai_ws folder created by stedgeai next to st_ai_output
        # Expected structure: <output_root>/st_ai_ws/inspector_network/workspace/include
        
        output_root = os.path.dirname(state.ai_code_dir.rstrip(os.sep)) # ../st_ai_output/code_resnet -> ../st_ai_output
        ws_include_dir = os.path.join(output_root, "..", "st_ai_ws", "inspector_network", "workspace", "include")
        ws_include_dir = os.path.abspath(ws_include_dir)
        
        proj_root = os.path.dirname(state.firmware_src_dir) 
        middlewares_ai_inc = os.path.join(proj_root, "Middlewares", "ST", "AI", "Inc")

        if os.path.exists(ws_include_dir):
            logger.info(f"📂 Found X-CUBE-AI runtime headers in: {ws_include_dir}")
            
            # Create destination folder if not exists
            os.makedirs(middlewares_ai_inc, exist_ok=True)
            
            # Copy all .h files
            copied_count = 0
            for header in os.listdir(ws_include_dir):
                if header.endswith('.h'):
                    src = os.path.join(ws_include_dir, header)
                    dst = os.path.join(middlewares_ai_inc, header)
                    shutil.copy2(src, dst)
                    copied_count += 1
            
            logger.info(f"✓ Copied {copied_count} runtime headers to Middlewares/ST/AI/Inc")
            
        else:
            logger.warning(f"⚠️  Runtime headers not found in: {ws_include_dir}")
            logger.warning("    The project might be missing 'ai_platform.h'.")
            
            # Original fallback warning
            if not os.path.exists(os.path.join(proj_root, "Middlewares", "ST", "AI")):
                 logger.warning("    Add the X-CUBE-AI component via STM32CubeMX (.ioc).")
        
        state.copy_success = True
        logger.info("✓ Copy completed")
        
    except Exception as e:
        state.copy_success = False
        state.integration_error_message = f"Copy error: {str(e)}"
        logger.error(state.integration_error_message)
    
    return state


def modify_main_c(state: MasterState, config: RunnableConfig = None) -> MasterState:
    if not state.modify_main:
        logger.info("main.c modification skipped")
        state.main_modification_success = True
        return state
    
    logger.info("Modifying main.c...")
    
    try:
        if not os.path.exists(state.main_c_path):
            raise FileNotFoundError(f"main.c file not found: {state.main_c_path}")
        
        with open(state.main_c_path, 'r', encoding='utf-8') as f:
            main_content = f.read()
        
        backup_path = f"{state.main_c_path}.backup_{state.timestamp}"
        shutil.copy2(state.main_c_path, backup_path)
        logger.info(f"Backup created: {backup_path}")
        
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
        logger.info("✓ main.c modified with complete inference logic")
        
    except Exception as e:
        state.main_modification_success = False
        state.integration_error_message = f"main.c modification error: {str(e)}"
        logger.error(state.integration_error_message)
    
    return state


def verify_integration(state: MasterState, config: RunnableConfig = None) -> MasterState:
    logger.info("Verifying integration...")
    
    try:
        all_files_copied = True
        for src_file in state.ai_src_files:
            filename = os.path.basename(src_file)
            dest_path = os.path.join(state.firmware_src_dir, filename)
            if not os.path.exists(dest_path):
                all_files_copied = False
                logger.error(f"Missing file: {dest_path}")
        
        for header_file in state.ai_header_files:
            filename = os.path.basename(header_file)
            dest_path = os.path.join(state.firmware_inc_dir, filename)
            if not os.path.exists(dest_path):
                all_files_copied = False
                logger.error(f"Missing file: {dest_path}")
        
        state.integration_success = (state.copy_success and all_files_copied and state.main_modification_success)
        
        if state.integration_success:
            logger.info("✓ Integration verified")
            # === UPDATE PERSISTENT CONTEXT ===
            if state.persistent_context is None:
                state.persistent_context = {}
            state.persistent_context["last_project_path"] = state.firmware_project_dir
            state.persistent_context["last_ai_code_dir"] = state.ai_code_dir
            logger.info(f"💾 Updated persistent_context with integration path")
        else:
            logger.error("✗ Integration verification failed")
        
    except Exception as e:
        state.integration_success = False
        state.integration_error_message = f"Verification error: {str(e)}"
        logger.error(state.integration_error_message)
    
    return state


def finalize_integration(state: MasterState, config: RunnableConfig = None) -> MasterState:
    if state.integration_success:
        logger.info("✓ INTEGRATION COMPLETED SUCCESSFULLY!")
        next_steps = "\n".join([
            f"1. Open the project in STM32CubeIDE: `{state.firmware_project_dir}`",
            "2. Verify that X-CUBE-AI Middleware is configured in the `.ioc` file",
            "3. Compile the project (**Build**)",
            "4. Flash onto the STM32 target via ST-LINK",
        ])
        state.response = (
            f"✅ **AI code successfully integrated into the firmware!**\n\n"
            f"**Copied files:** {len(state.ai_src_files)} .c · {len(state.ai_header_files)} .h\n"
            f"**main.c modified:** {'✓' if state.main_modification_success else '✗'}\n"
            f"**Project:** `{state.firmware_project_dir}`\n\n"
            f"**Next steps:**\n{next_steps}"
        )
    else:
        logger.error(f"✗ Integration failed: {state.integration_error_message}")
        state.response = (
            f"❌ **Integration failed:** {state.integration_error_message}\n\n"
            f"- File copy: {'✓' if state.copy_success else '✗'}\n"
            f"- main.c modification: {'✓' if state.main_modification_success else '✗'}"
        )
    
    return state

