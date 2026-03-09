import os
import subprocess
import shutil
import platform
import re
import json
import logging
from typing import Optional, Literal, List
from datetime import datetime

from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt

from src.assistant.configuration import Configuration
from src.assistant.state import MasterState

# ============================================================================
# LOGGING
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logging.getLogger("langgraph_api.server").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

logger.setLevel(logging.DEBUG)  # Abilita DEBUG logging


class ProjectInfoExtraction(BaseModel):
    """Schema for extracting project info from natural language response"""
    ioc_file_path: Optional[str] = Field(
        default=None,
        description="Path to the .ioc file if specified, otherwise None"
    )
    board_name: Optional[str] = Field(
        default=None,
        description="Name of the STM32 board (e.g., STM32F401VCHx, STM32H743ZI)"
    )
    mcu_series: Optional[str] = Field(  # ✅ NEW
        default=None,
        description="MCU series extracted from board_name (e.g., F4, H7, N6, L4, U5)"
    )
    project_name: Optional[str] = Field(
        default=None,
        description="Name of the project (e.g., MyProject, NeuroControl)"
    )
    toolchain: Optional[str] = Field(
        default=None,
        description="Toolchain to use (e.g., STM32CubeIDE, Keil, IAR)"
    )
    peripheral_config: Optional[List[str]] = Field(
        default_factory=list,
        description="List of pins or peripherals to enable (e.g., ['set_pin PA5 GPIO_Output', 'set_peripheral TIM1'])"
    )

project_info_extraction_instructions = """You are an information extractor for STM32 project configurations.

Analyze the user's response and extract the following fields:

1. **ioc_file_path**: If the user specifies a path to an .ioc file (example: "/path/to/config.ioc", "~/projects/board.ioc")
   → If not specified: null

2. **board_name**: The name of the STM32 board (example: "STM32F401VCHx", "STM32H743ZI", "STM32N657Z0HxQ")
   → If not specified: null

3. **mcu_series**: The MCU series extracted from the board_name
   Common values: "F0", "F1", "F2", "F3", "F4", "F7", "H5", "H7", "L0", "L1", "L4", "L5", "U5", "G0", "G4", "W5", "C0", "N6"
   → If board_name is specified, ALWAYS extract the series!
   → Logic: STM32 + Letter + Digit = series (e.g., STM32F401 → F4, STM32N657 → N6)
   → If it cannot be extracted: null

4. **project_name**: The name of the project (example: "MyProject", "AI_Firmware", "NeuroControl")
   → If not specified: null

5. **toolchain**: The IDE/toolchain to use (example: "STM32CubeIDE", "Keil uVision", "IAR Embedded Workbench")
   → If not specified: null

ALWAYS reply in valid JSON format, even if some fields are null.

**Special Notes:**
- If the user says "use the previous one", "like last time", "yesterday's", or "use profile" -> set `board_name` to "USE_PROFILE".
- Otherwise extract real data.
- If the user mentions specific pins or timers, extract them as CubeMX commands (e.g., "activate pin PA5 as output" -> "set_pin PA5 GPIO_Output", "use timer 1" -> "set_peripheral TIM1").
- The .ioc files are NOT mandatory if the board is specified.

Examples:
- Input: "Create a project for STM32F401 with CubeIDE, name MyApp"
  Output: {"ioc_file_path": null, "board_name": "STM32F401", "mcu_series": "F4", "project_name": "MyApp", "toolchain": "STM32CubeIDE"}

- Input: "Use yesterday's board"
  Output: {"board_name": "USE_PROFILE", "mcu_series": null, "project_name": null, "toolchain": null}

- Input: "I have a config.ioc file in ~/boards/, use that one"
  Output: {"ioc_file_path": "~/boards/config.ioc", "board_name": null, "mcu_series": null, "project_name": null, "toolchain": null}
"""
# ============================================================================
# UTILITIES
# ============================================================================

def extract_mcu_series_from_board(board_name: str) -> Optional[str]:
    """
    Extracts the MCU series from the board name.
    Ex: "STM32F401VCHx" → "F4"
        "STM32N657Z0HxQ" → "N6"
        "STM32H743ZI" → "H7"
    """
    import re
    if not board_name:
        return None
    
    # Pattern robusto: (Opzionale STM32) + (Lettera)(Cifra)
    # Esempi: "STM32H7A3ZI" -> "H7", "F401" -> "F4", "STM32 F4" -> "F4"
    match = re.search(r'(?:STM32)?[\s-]*([A-Z])([0-9])', board_name, re.IGNORECASE)
    if match:
        letter = match.group(1).upper()
        digit = match.group(2)
        return f"{letter}{digit}"
    return None

def get_template_ioc_path(board_name: Optional[str], mcu_series: Optional[str]) -> Optional[str]:
    """
    Looks for a pre-generated .ioc file in the templates/ioc_files folder.
    Priority: 
    1. Exact board name
    2. MCU series (F4, H7, U5, N6)
    """
    template_dir = os.path.join(os.path.dirname(__file__), "templates", "ioc_files")
    if not os.path.exists(template_dir):
        logger.warning(f"⚠️  Template folder not found: {template_dir}")
        return None

    # Try exact board match
    if board_name:
        board_path = os.path.join(template_dir, f"{board_name}.ioc")
        if os.path.exists(board_path):
            logger.info(f"🎯 Template found for board: {board_name}")
            return board_path

    # Try series match
    if mcu_series:
        # Map series to representative file if there is no exact match
        SERIES_MAP = {
            "F4": "STM32F401VCHx.ioc",
            "H7": "STM32H7A3ZITx.ioc",
            "U5": "STM32U585AIIxQ.ioc",
            "N6": "STM32N657X0HxQ.ioc"
        }
        filename = SERIES_MAP.get(mcu_series.upper())
        if filename:
            series_path = os.path.join(template_dir, filename)
            if os.path.exists(series_path):
                logger.info(f"🎯 Template found for series: {mcu_series}")
                return series_path

    logger.warning(f"⚠️  No template found for {board_name} ({mcu_series})")
    return None



# ============================================================================
# WORKFLOW 1: FIRMWARE GENERATION
# ============================================================================

def collect_project_info(state: MasterState, config: RunnableConfig = None) -> MasterState:
    """
    Collects project info from user's natural response.
    The response is parsed by LLM to extract attributes, including mcu_series.
    """
    
    logger.info("📋 Collecting STM32CubeMX project configuration...")
    
    prompt = {
        "instruction": """STM32CubeMX Project Configuration
        
Please specify (in natural language):
- .ioc file path (if available, otherwise not required)
- STM32 board name (e.g., STM32F401VCHx, STM32H743ZI, STM32N657Z0HxQ)
- Project name
- Toolchain (e.g., STM32CubeIDE, Keil, IAR)

Example: "Create project MyApp for STM32F401 with CubeIDE"
        """,
    }
    
    # === IDEMPOTENCY CHECK ===
    # SKIP if board is already set, unless we are explicitly coming from a "change_board" route.
    # IMPORTANT: Validate board_name is a real STM32 board, not a stale word like "AI" from profile.
    is_backtracking = state.route == "change_board"
    _INVALID_BOARD_NAMES = {"AI", "Custom", "Integration", "Customization", "Firmware", "Unknown"}
    _board_is_valid = (
        state.board_name
        and state.board_name not in _INVALID_BOARD_NAMES
        and state.board_name != "STM32F401VCHx"
        and re.search(r'[0-9]', state.board_name)  # Must contain at least one digit
    )
    if _board_is_valid and not state.user_response and not is_backtracking:
        logger.info(f"⏭️  Idempotency: Board '{state.board_name}' already configured. Skipping interrupt.")
        return state

    from src.assistant.utils import extract_user_response, get_llm
    from langchain_core.messages import SystemMessage, HumanMessage
    cfg = Configuration.from_runnable_config(config)
    llm = get_llm(config)
    llm_extractor = llm.with_structured_output(ProjectInfoExtraction)
    
    # --- Step 1: Try using initial message ---
    # SKIP Discovery if we are backtracking to change board (force new input)
    initial_board = None
    if not state.user_response and not is_backtracking:
        res = llm_extractor.invoke([
            SystemMessage(content=project_info_extraction_instructions),
            HumanMessage(content=f"Message: {state.message}")
        ])
        initial_board = res.board_name
        # Save what we found so far
        if res.board_name: state.board_name = res.board_name
        if res.mcu_series: state.mcu_series = res.mcu_series
        if res.project_name: state.project_name = res.project_name
        if res.toolchain: state.toolchain = res.toolchain
        if res.ioc_file_path: state.ioc_file_path = res.ioc_file_path
        if hasattr(res, 'peripheral_config') and res.peripheral_config:
            state.peripheral_config = res.peripheral_config


    # --- Step 2: Verification and Interrupt ---
    # If there is no board in the message or it's generic, ASK.
    if not initial_board or initial_board.lower() == "unknown":
        resume_value = None
        if not state.user_response:
            # Recover board from profile for suggestion
            last_board = state.persistent_context.get("board_name", "None") if state.persistent_context else "None"
            
            # Enrich prompt for the user
            dynamic_prompt = {
                "instruction": prompt["instruction"],
                "suggestion": f" I noticed last time you used: **{last_board}**. Do you want to use the same one, or a new one?"
            }
        if not state.user_response:
            # logger.info("⏸️ Interrupting: Requesting project info with profile suggestion.")
            # resume_value = interrupt(dynamic_prompt)
            logger.info("⏭️  BYPASS: Automatic board selection -> 'STM32H7A3ZI'")
            user_text = "STM32H7A3ZI"
        else:
            # After resume: use interrupt return value as priority
            if resume_value and str(resume_value).strip():
                user_text = str(resume_value).strip()
            else:
                user_text = extract_user_response(state.user_response)
        state.user_response = ""
        
        res = llm_extractor.invoke([
            SystemMessage(content=project_info_extraction_instructions),
            HumanMessage(content=f"Response: {user_text}")
        ])
        
        # Handle "USE_PROFILE"
        context_board = state.persistent_context.get("board_name") if state.persistent_context else None
        if res.board_name == "USE_PROFILE" and context_board:
            state.board_name = context_board
            state.mcu_series = state.persistent_context.get("mcu_series")
            logger.info(f"📋 Applied board from profile: {state.board_name}")
        else:
            if res.board_name: state.board_name = res.board_name
            if res.mcu_series: state.mcu_series = res.mcu_series
        
        if res.project_name: state.project_name = res.project_name
        if res.toolchain: state.toolchain = res.toolchain
        if hasattr(res, 'peripheral_config') and res.peripheral_config:
            state.peripheral_config = res.peripheral_config

    # --- Step 3: Finalization ---
    if not state.board_name:
        state.board_name = "STM32F401VCHx"
        state.mcu_series = "F4"
    
    # Automatic extraction mcu_series if missing
    if not state.mcu_series and state.board_name:
        extracted = extract_mcu_series_from_board(state.board_name)
        if extracted:
            state.mcu_series = extracted
            logger.info(f"📊 MCU series automatically extracted from board_name: {extracted}")
        else:
            # Final fallback to H7
            state.mcu_series = "H7"
            logger.warning(f"⚠️  Unable to extract series, fallback to H7")
        
    state.project_name = state.project_name or "MySTM32Project"
    state.toolchain = state.toolchain or "STM32CubeIDE"
    
    logger.info(f"✓ Final Configuration: {state.board_name} ({state.mcu_series})")
    
    # Synchronize target for AI workflow (avoids discrepancies and idempotency issues)
    if state.mcu_series:
        state.target = f"stm32{state.mcu_series.lower()}"
        state.compression = "high" # Default
        logger.info(f"🎯 Synchronized AI target: {state.target}")

    state.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return state

def search_and_install_stm32_package(state: MasterState, config: RunnableConfig = None) -> MasterState:
    """
    Node that clones the entire STM32 package from GitHub and saves it in ~/STM32Cube/Repository/
    AUTOMATICALLY downloads the latest version available from the repository.
    """
    
    logger.info("🔍 Searching and installing STM32 package from GitHub...")
    
    cfg = Configuration.from_runnable_config(config)
    
    # === 0. EXTRACT MCU_SERIES FROM STATE ===
    
    board_series = state.mcu_series
    
    if not board_series:
        logger.error(f"❌ mcu_series not specified in state!")
        state.package_installation_success = False
        state.package_error_message = "mcu_series unavailable"
        return state
    
    logger.info(f"📊 MCU Series: {board_series}")
    
    # === 1. CHECK IF PACKAGE IS ALREADY INSTALLED ===
    
    stm32_cube_repo = cfg.stm32_repo_path
    os.makedirs(stm32_cube_repo, exist_ok=True)
    
    # Package folders for this series (e.g., STM32Cube_FW_N6_*)
    existing_packages = []
    for folder in os.listdir(stm32_cube_repo):
        if board_series in folder and os.path.isdir(os.path.join(stm32_cube_repo, folder)):
            existing_packages.append(folder)
    
    if existing_packages:
        logger.info(f"✓ Package STM32{board_series} already present!")
        # Sort to find the most recent one (by name)
        existing_packages.sort()
        latest = existing_packages[-1]  # The latest in alphabetical order
        logger.info(f"  Folders found: {len(existing_packages)}")
        for pkg in existing_packages:
            marker = " ← LATEST" if pkg == latest else ""
            logger.info(f"    - {pkg}{marker}")
        
        state.package_installation_success = True
        state.package_installation_path = os.path.join(stm32_cube_repo, latest)
        logger.info(f"✓ Using: {latest}")
        logger.info("✓ Installation skipped (already present)")
        return state
    
    logger.info(f"📥 Package not found, proceeding with download from GitHub...")
    
    # === 2. MAPPA SERIE → GITHUB URL ===
    
    GITHUB_PACKAGES = {
        "F0": "https://github.com/STMicroelectronics/STM32CubeF0.git",
        "F1": "https://github.com/STMicroelectronics/STM32CubeF1.git",
        "F2": "https://github.com/STMicroelectronics/STM32CubeF2.git",
        "F3": "https://github.com/STMicroelectronics/STM32CubeF3.git",
        "F4": "https://github.com/STMicroelectronics/STM32CubeF4.git",
        "F7": "https://github.com/STMicroelectronics/STM32CubeF7.git",
        "H5": "https://github.com/STMicroelectronics/STM32CubeH5.git",
        "H7": "https://github.com/STMicroelectronics/STM32CubeH7.git",
        "L0": "https://github.com/STMicroelectronics/STM32CubeL0.git",
        "L1": "https://github.com/STMicroelectronics/STM32CubeL1.git",
        "L4": "https://github.com/STMicroelectronics/STM32CubeL4.git",
        "L5": "https://github.com/STMicroelectronics/STM32CubeL5.git",
        "U5": "https://github.com/STMicroelectronics/STM32CubeU5.git",
        "G0": "https://github.com/STMicroelectronics/STM32CubeG0.git",
        "G4": "https://github.com/STMicroelectronics/STM32CubeG4.git",
        "W5": "https://github.com/STMicroelectronics/STM32CubeW5.git",
        "C0": "https://github.com/STMicroelectronics/STM32CubeC0.git",
        "N6": "https://github.com/STMicroelectronics/STM32CubeN6.git",
    }
    
    github_url = GITHUB_PACKAGES.get(board_series)
    
    if not github_url:
        logger.error(f"❌ Package non trovato per serie {board_series}")
        state.package_installation_success = False
        state.package_error_message = f"Nessun package GitHub per serie {board_series}"
        return state
    
    logger.info(f"🎯 Repository GitHub: {github_url}")
    
    # === 3. DOWNLOAD LATEST RELEASE TAG FROM GITHUB ===
    
    try:
        logger.info(f"🔎 Searching for latest release tag...")
        
        # Use git ls-remote to get tags without cloning everything
        cmd_tags = ["git", "ls-remote", "--tags", github_url]
        result_tags = subprocess.run(cmd_tags, capture_output=True, text=True, timeout=30)
        
        if result_tags.returncode != 0:
            logger.warning(f"⚠️  Unable to read tags, using main branch")
            latest_version = "main"  # Fallback
        else:
            # Extract tags (filter only "v*" and no "^{}")
            tags = []
            for line in result_tags.stdout.strip().split('\n'):
                if 'refs/tags/' in line:
                    tag = line.split('refs/tags/')[-1].replace('^{}', '')
                    if tag.startswith('v') and '^{}' not in tag:
                        tags.append(tag)
            
            if tags:
                # Sort versions (vX.Y.Z) in descending order
                tags.sort(key=lambda x: [int(p) if p.isdigit() else 0 for p in x[1:].split('.')], reverse=True)
                latest_version = tags[0]
                logger.info(f"✓ Latest release found: {latest_version}")
                logger.info(f"  Available: {len(tags)} version(s)")
                logger.info(f"    Top 3: {', '.join(tags[:3])}")
            else:
                logger.warning(f"⚠️  No tags found, using main branch")
                latest_version = "main"
    
    except Exception as e:
        logger.warning(f"⚠️  Error during tag reading: {e}, using main branch")
        latest_version = "main"
    
    logger.info(f"📥 Version to install: {latest_version}")
    
    # === 4. CLONE FROM GITHUB ===
    
    # Temporary Paths
    temp_clone_path = f"/tmp/STM32Cube{board_series}_{state.thread_id}"
    
    try:
        logger.info(f"📥 Recursive clone in progress (timeout: 10 minutes)...")
        logger.info(f"   Branch: {latest_version}")
        
        cmd_clone = [
            "git", "clone",
            "--recursive",
            "--branch", latest_version,
            "--depth", "1",
            github_url,
            temp_clone_path
        ]
        
        result = subprocess.run(
            cmd_clone,
            capture_output=True,
            text=True,
            timeout=600  # 10 minutes
        )
        
        if result.returncode != 0:
            logger.error(f"❌ Clone failed!")
            logger.error(f"Return code: {result.returncode}")
            logger.error(f"Stderr: {result.stderr}")
            raise RuntimeError(f"Git clone failed: {result.stderr}")
        
        logger.info(f"✓ Repository cloned: {temp_clone_path}")
        
        # === 5. VERIFY STRUCTURE ===
        
        logger.info(f"✅ Verifying repository structure...")
        
        required_dirs = ["Drivers", "Middlewares", "Projects"]
        for dir_name in required_dirs:
            dir_path = os.path.join(temp_clone_path, dir_name)
            if os.path.isdir(dir_path):
                logger.info(f"  ✓ {dir_name}/ present")
            else:
                logger.warning(f"  ⚠️  {dir_name}/ missing (continuing anyway)")
        
        # === 6. EXTRACT VERSION FROM FOLDER OR TAG ===
        
        # Look for Release_Notes.html to extract version
        release_notes_path = os.path.join(temp_clone_path, "Release_Notes.html")
        version_info = latest_version  # Default
        
        if os.path.exists(release_notes_path):
            try:
                with open(release_notes_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    # Look for pattern like "V1.2.0" or "v1.2.0"
                    
                    match = re.search(r'[Vv](\d+\.\d+\.\d+)', content)
                    if match:
                        version_info = f"V{match.group(1)}"
                        logger.info(f"✓ Version extracted from Release_Notes: {version_info}")
            except Exception as e:
                logger.warning(f"⚠️  Cannot read Release_Notes: {e}")
        
        # Convert tag format: v1.2.0 → V1_2_0
        if version_info.startswith('v'):
            version_str = version_info[1:].replace(".", "_")
        else:
            version_str = version_info.replace(".", "_")
        
        final_folder_name = f"STM32Cube_FW_{board_series}_{version_str}"
        
        final_install_path = os.path.join(stm32_cube_repo, final_folder_name)
        
        logger.info(f"📂 Installation folder: {final_folder_name}")
        
        # === 7. MOVE REPOSITORY TO FINAL DESTINATION ===
        
        logger.info(f"📦 Moving repository to final folder...")
        
        # If it already exists, rename the old one
        if os.path.exists(final_install_path):
            logger.warning(f"Folder already present, renaming the old one...")
            old_backup = f"{final_install_path}_backup_{state.timestamp}"
            os.rename(final_install_path, old_backup)
            logger.info(f"  Old folder: {old_backup}")
        
        # Move clone path → final path
        shutil.move(temp_clone_path, final_install_path)
        logger.info(f"✓ Repository moved")
        
        # === 8. VERIFY INSTALLATION ===
        
        logger.info(f"✅ Verifying installation...")
        
        # Check that critical files exist
        critical_files = [
            "Drivers/CMSIS/Device",
            "Drivers/STM32" + board_series + "xx_HAL_Driver",
            "Middlewares",
        ]
        
        files_found = 0
        for critical_path in critical_files:
            full_path = os.path.join(final_install_path, critical_path)
            if os.path.exists(full_path):
                logger.info(f"  ✓ {critical_path}/")
                files_found += 1
            else:
                logger.warning(f"  ⚠️  {critical_path}/ not found")
        
        # === 9. COUNT TOTAL FILES ===
        
        total_files = 0
        total_dirs = 0
        total_size = 0
        
        for root, dirs, files in os.walk(final_install_path):
            total_dirs += len(dirs)
            total_files += len(files)
            for file in files:
                try:
                    file_path = os.path.join(root, file)
                    total_size += os.path.getsize(file_path)
                except OSError:
                    pass
        
        logger.info(f"📊 Installation statistics:")
        logger.info(f"  Directories: {total_dirs}")
        logger.info(f"  Files: {total_files}")
        logger.info(f"  Size: {total_size / 1024 / 1024:.1f} MB")
        logger.info(f"  Version: {version_info}")
        
        state.package_installation_success = True
        state.package_installation_path = final_install_path
        logger.info(f"✓✓✓ Package {board_series} installed successfully! ✓✓✓")
        
    except Exception as e:
        logger.error(f"❌ Error during installation: {str(e)}")
        logger.exception(e)
        state.package_installation_success = False
        state.package_error_message = str(e)
        
        # Cleanup if it fails
        try:
            if os.path.exists(temp_clone_path):
                shutil.rmtree(temp_clone_path)
                logger.info("Cleanup completed (error)")
        except:
            pass
    
    return state


def check_package_installation(state: MasterState) -> Literal["generate_cubemx_script", "finalize_project"]:
    """
    Checks if the package installation was successful.
    If it fails, skips directly to finalize with error.
    """
    if state.package_installation_success:
        logger.info("✓ Package installed, continuing with script generation")
        return "generate_cubemx_script"
    else:
        logger.error(f"❌ Package installation failed: {state.package_error_message}")
        state.firmware_generation_success = False
        state.firmware_error_message = f"Package installation failed: {state.package_error_message}"
        return "finalize_project"


def generate_cubemx_script(state: MasterState, config: RunnableConfig = None) -> MasterState:
    folder = f"{state.project_name}_{state.timestamp}"
    state.firmware_project_path = os.path.join(state.base_dir, folder)

    lines = [f"login {state.st_email} {state.st_password} y"]
    
    # Search for a pre-generated template if the user hasn't provided their own .ioc
    effective_ioc = state.ioc_file_path
    if not effective_ioc:
        effective_ioc = get_template_ioc_path(state.board_name, state.mcu_series)

    if effective_ioc:
        # If we have an .ioc (user or template), we ONLY use config load
        # This bypasses all interactive recommendation pop-ups
        logger.info(f"📂 Using configuration load: {effective_ioc}")
        lines.append(f'config load "{effective_ioc}"')
    else:
        # Extreme fallback: load board (risk of pop-ups)
        logger.warning(f"⚠️  No .ioc available, falling back to generic board load")
        lines.append(f"load {state.board_name}")

    lines += [
        f"project name {state.project_name}",
        f'project toolchain "{state.toolchain}"',
        f"project path {state.firmware_project_path}"
    ]

    # Custom peripheral config injection
    if hasattr(state, 'peripheral_config') and state.peripheral_config:
        logger.info(f"🔧 Injecting {len(state.peripheral_config)} peripheral configuration commands")
        for cmd in state.peripheral_config:
            # Clean and normalize command
            clean_cmd = cmd.strip()
            # We only accept safe/known commands
            if not any(clean_cmd.startswith(prefix) for prefix in ["set_pin", "set_peripheral", "config"]):
                logger.warning(f"⚠️  Peripheral command ignored (invalid format): {clean_cmd}")
                continue
            lines.append(clean_cmd)

    lines += [
        "project generate",
        "exit"
    ]

    state.firmware_script_content = "\n".join(lines)
    state.firmware_script_path = f"/tmp/script_{state.thread_id}.scr"
    with open(state.firmware_script_path, "w") as f:
        f.write(state.firmware_script_content)
    
    logger.info("✓ CubeMX script generated")
    return state


def recover_with_ioc_fallback(state: MasterState) -> bool:
    """
    Attempts recovery by generating a valid .ioc file and modifying the script.
    """
    logger.info("🚑 Attempting Recovery with IOC Fallback...")
    
    # 1. Search for the best template
    fallback_ioc = get_template_ioc_path(state.board_name, state.mcu_series)
    
    if not fallback_ioc:
        logger.error("❌ No template available for fallback!")
        return False
        
    # 2. Update the script to load the template
    lines = [
        f"login {state.st_email} {state.st_password} y",
        f'config load "{fallback_ioc}"',
        f"project name {state.project_name}",
        f'project toolchain "{state.toolchain}"',
        f"project path {state.firmware_project_path}",
        "project generate",
        "exit"
    ]
    
    # Overwrites the existing script
    state.firmware_script_content = "\n".join(lines)
    # state.firmware_script_path is already set, reuse it
    with open(state.firmware_script_path, "w") as f:
        f.write(state.firmware_script_content)
        
    logger.info("✓ CubeMX Script rewritten for Fallback")
    return True


def execute_generation(state: MasterState, config: RunnableConfig = None) -> MasterState:
    """
    Generates the project in a LOCAL TEMPORARY directory (/tmp) 
    and then moves it to the final folder to avoid locking issues on network filesystems.
    """
    import time
    
    # === 1. CREATE LOCAL TEMPORARY DIRECTORY ===
    temp_project_path = f"/tmp/stm32_{state.thread_id}"
    os.makedirs(temp_project_path, exist_ok=True)
    logger.info(f"📂 Temporary directory created: {temp_project_path}")
    
    # === 2. MODIFY THE SCRIPT TO USE THE TEMPORARY PATH ===
    # The script was already created by generate_cubemx_script, we read and modify it
    with open(state.firmware_script_path, "r") as f:
        original_script = f.read()
    
    # Replace final path with temporary path
    temp_script = original_script.replace(
        f"project path {state.firmware_project_path}",
        f"project path {temp_project_path}"
    )
    
    # Write the modified script
    temp_script_path = f"/tmp/script_temp_{state.thread_id}.scr"
    with open(temp_script_path, "w") as f:
        f.write(temp_script)
    
    logger.info(f"✏️  Script modified to use temporary path")
    
    # === 3. EXECUTE CUBEMX ON THE TEMPORARY DIRECTORY ===
    # On macOS (Darwin) we don't use xvfb-run. On Linux we use it if necessary.
    if platform.system() == "Darwin":
        cmd = [state.cubemx_path, "-q", temp_script_path]
    else:
        # Try to use xvfb-run on Linux if available, otherwise direct fallback
        if shutil.which("xvfb-run"):
            cmd = ["xvfb-run", "-a", state.cubemx_path, "-q", temp_script_path]
        else:
            cmd = [state.cubemx_path, "-q", temp_script_path]
    
    try:
        logger.info(f"🚀 Executing CubeMX in temp dir (Attempt 1, timeout: 300s)...")
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if res.returncode != 0:
            logger.warning(f"⚠️  Generation Failed (RC={res.returncode}). Trying Fallback...")
            # FALLBACK: Retry with template .ioc
            if recover_with_ioc_fallback(state):
                # Update the temporary script as well
                with open(state.firmware_script_path, "r") as f:
                    fallback_script = f.read()
                temp_fallback_script = fallback_script.replace(
                    f"project path {state.firmware_project_path}",
                    f"project path {temp_project_path}"
                )
                with open(temp_script_path, "w") as f:
                    f.write(temp_fallback_script)
                
                logger.info(f"🚀 Executing CubeMX (Fallback Attempt, timeout: 600s)...")
                res = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        state.firmware_generation_success = (res.returncode == 0)
        
        if state.firmware_generation_success:
            logger.info("✓ Generation completed in temp dir, waiting for folder creation...")
            time.sleep(2)
            
            # === 4. VERIFY THAT GENERATION IS COMPLETE ===
            for attempt in range(10):
                # We look for Src and Inc in the temporary directory
                src_exists = any(os.path.isdir(os.path.join(dp, "Src")) for dp, dn, filenames in os.walk(temp_project_path))
                inc_exists = any(os.path.isdir(os.path.join(dp, "Inc")) for dp, dn, filenames in os.walk(temp_project_path))
                
                if src_exists and inc_exists:
                    logger.info("✓ Src/ and Inc/ folders verified in temporary directory")
                    break
                
                logger.info(f"Waiting for folders in temp dir... attempt {attempt+1}/10")
                time.sleep(1)
            else:
                logger.warning("⚠️  Folders might not be completely created")
            
            # === 5. MOVE FROM TEMPORARY TO FINAL DIRECTORY ===
            logger.info(f"📦 Moving project from temp dir to final destination...")
            
            # Create base directory if it doesn't exist
            os.makedirs(os.path.dirname(state.firmware_project_path), exist_ok=True)
            
            # If the final destination already exists, remove it
            if os.path.exists(state.firmware_project_path):
                logger.warning(f"⚠️  Destination folder already exists, removing...")
                shutil.rmtree(state.firmware_project_path)
            
            # Move everything from temp to final
            shutil.move(temp_project_path, state.firmware_project_path)
            logger.info(f"✓✓✓ Project successfully moved to: {state.firmware_project_path}")
            
        else:
            state.firmware_error_message = res.stderr or f"Return code {res.returncode}"
            logger.error(f"❌ Generation failed: {state.firmware_error_message}")
    
    except subprocess.TimeoutExpired:
        logger.error("❌ First attempt TIMED OUT. Trying Fallback...")
        # FALLBACK on timeout
        try:
            if recover_with_ioc_fallback(state):
                # Update temporary script with fallback
                with open(state.firmware_script_path, "r") as f:
                    fallback_script = f.read()
                temp_fallback_script = fallback_script.replace(
                    f"project path {state.firmware_project_path}",
                    f"project path {temp_project_path}"
                )
                with open(temp_script_path, "w") as f:
                    f.write(temp_fallback_script)
                
                logger.info(f"🚀 Executing CubeMX (Fallback Attempt)...")
                res = subprocess.run(cmd, capture_output=True, text=True, timeout=150)
                state.firmware_generation_success = (res.returncode == 0)
                
                if state.firmware_generation_success:
                    # Move even in case of successful fallback
                    logger.info("✓ Fallback successful, moving to final destination...")
                    os.makedirs(os.path.dirname(state.firmware_project_path), exist_ok=True)
                    if os.path.exists(state.firmware_project_path):
                        shutil.rmtree(state.firmware_project_path)
                    shutil.move(temp_project_path, state.firmware_project_path)
                    logger.info(f"✓ Project moved after fallback")
                else:
                    state.firmware_error_message = res.stderr or f"Fallback Return code {res.returncode}"
            else:
                state.firmware_generation_success = False
                state.firmware_error_message = "Timeout on first attempt, fallback generation failed"
        except subprocess.TimeoutExpired:
            state.firmware_generation_success = False
            state.firmware_error_message = "Timeout on both attempts (primary + fallback)"
        except Exception as fallback_e:
            state.firmware_generation_success = False
            state.firmware_error_message = f"Fallback error: {str(fallback_e)}"
    
    except Exception as e:
        state.firmware_generation_success = False
        state.firmware_error_message = str(e)
        logger.exception(e)
    
    finally:
        # === 6. CLEANUP TEMPORARY SCRIPTS ===
        try:
            os.remove(state.firmware_script_path)
            logger.info("✓ Cleanup original script")
        except OSError:
            pass
        
        try:
            os.remove(temp_script_path)
            logger.info("✓ Cleanup temporary script")
        except OSError:
            pass
        
        # Cleanup temp dir IF it still exists (error case)
        if os.path.exists(temp_project_path):
            try:
                shutil.rmtree(temp_project_path)
                logger.info("✓ Cleanup temp directory (error)")
            except Exception as cleanup_err:
                logger.warning(f"⚠️  Cannot remove temp dir: {cleanup_err}")
    
    logger.info(f"✓ Firmware generated: {state.firmware_project_path}" if state.firmware_generation_success else f"✗ Firmware failed: {state.firmware_error_message}")
    return state



def finalize_project(state: MasterState, config: RunnableConfig = None) -> MasterState:
    if state.firmware_generation_success:
        print(f"✓ Firmware project generated: {state.firmware_project_path}")
        state.firmware_project_dir = state.firmware_project_path
    else:
        print(f"✗ Firmware error: {state.firmware_error_message}")
    return state

    return state
