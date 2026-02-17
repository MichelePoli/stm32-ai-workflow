import os
import subprocess
import shutil
import platform
import re
import json
import logging
from typing import Optional, Literal
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
    """Schema per estrarre informazioni progetto dalla risposta naturale"""
    ioc_file_path: Optional[str] = Field(
        default=None,
        description="Path al file .ioc se specificato, altrimenti None"
    )
    board_name: Optional[str] = Field(
        default=None,
        description="Nome della board STM32 (es: STM32F401VCHx, STM32H743ZI)"
    )
    mcu_series: Optional[str] = Field(  # ✅ NUOVO
        default=None,
        description="Serie MCU estratta dal board_name (es: F4, H7, N6, L4, U5)"
    )
    project_name: Optional[str] = Field(
        default=None,
        description="Nome del progetto (es: MyProject, NeuroControl)"
    )
    toolchain: Optional[str] = Field(
        default=None,
        description="Toolchain da usare (es: STM32CubeIDE, Keil, IAR)"
    )

project_info_extraction_instructions = """Sei un estrattore di informazioni per la configurazione di progetti STM32.

Analizza la risposta dell'utente e estrai i seguenti campi:

1. **ioc_file_path**: Se l'utente specifica un path a un file .ioc (esempio: "/path/to/config.ioc", "~/projects/board.ioc")
   → Se non specificato: null

2. **board_name**: Il nome della board STM32 (esempio: "STM32F401VCHx", "STM32H743ZI", "STM32N657Z0HxQ")
   → Se non specificato: null

3. **mcu_series**: La serie MCU estratta dal board_name
   Valori comuni: "F0", "F1", "F2", "F3", "F4", "F7", "H5", "H7", "L0", "L1", "L4", "L5", "U5", "G0", "G4", "W5", "C0", "N6"
   → Se il board_name è specificato, estrai SEMPRE la serie!
   → Logica: STM32 + Lettera + Cifra = serie (es: STM32F401 → F4, STM32N657 → N6)
   → Se non puoi estrarre: null

4. **project_name**: Il nome del progetto (esempio: "MyProject", "AI_Firmware", "NeuroControl")
   → Se non specificato: null

5. **toolchain**: L'IDE/toolchain da usare (esempio: "STM32CubeIDE", "Keil uVision", "IAR Embedded Workbench")
   → Se non specificato: null

Rispondi SEMPRE in formato JSON valido, anche se alcuni campi sono null.

**Note Speciali:**
- Se l'utente dice "usa la precedente", "come l'altra volta", "quella di ieri", "usa il profilo" -> imposta `board_name` come "USE_PROFILE".
- Altrimenti estrai i dati reali.

Esempi:
- Input: "Crea un progetto per STM32F401 con CubeIDE, nome MyApp"
  Output: {"ioc_file_path": null, "board_name": "STM32F401", "mcu_series": "F4", "project_name": "MyApp", "toolchain": "STM32CubeIDE"}

- Input: "Usa la board di ieri"
  Output: {"board_name": "USE_PROFILE", "mcu_series": null, "project_name": null, "toolchain": null}

- Input: "Ho un file config.ioc in ~/boards/, usa quello"
  Output: {"ioc_file_path": "~/boards/config.ioc", "board_name": null, "mcu_series": null, "project_name": null, "toolchain": null}
"""
# ============================================================================
# UTILITIES
# ============================================================================

def extract_mcu_series_from_board(board_name: str) -> Optional[str]:
    """
    Estrae la serie MCU dal nome della board.
    Es: "STM32F401VCHx" → "F4"
        "STM32N657Z0HxQ" → "N6"
        "STM32H743ZI" → "H7"
    """
    import re
    if not board_name:
        return None
    
    # Pattern: STM32 + (Lettera)(Cifra) → serie
    match = re.search(r'STM32([A-Z])([0-9])', board_name, re.IGNORECASE)
    if match:
        letter = match.group(1).upper()
        digit = match.group(2)
        return f"{letter}{digit}"
    return None

def get_template_ioc_path(board_name: Optional[str], mcu_series: Optional[str]) -> Optional[str]:
    """
    Cerca un file .ioc pre-generato nella cartella templates/ioc_files.
    Priorità: 
    1. Nome esatto della board
    2. Serie MCU (F4, H7, U5, N6)
    """
    template_dir = os.path.join(os.path.dirname(__file__), "templates", "ioc_files")
    if not os.path.exists(template_dir):
        logger.warning(f"⚠️  Cartella template non trovata: {template_dir}")
        return None

    # Tenta match esatto board
    if board_name:
        board_path = os.path.join(template_dir, f"{board_name}.ioc")
        if os.path.exists(board_path):
            logger.info(f"🎯 Template trovato per board: {board_name}")
            return board_path

    # Tenta match serie
    if mcu_series:
        # Mappa serie a file rappresentativo se non c'è match esatto
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
                logger.info(f"🎯 Template trovato per serie: {mcu_series}")
                return series_path

    logger.warning(f"⚠️  Nessun template trovato per {board_name} ({mcu_series})")
    return None



# ============================================================================
# WORKFLOW 1: FIRMWARE GENERATION
# ============================================================================

def collect_project_info(state: MasterState, config: RunnableConfig = None) -> MasterState:
    """
    Raccoglie info progetto da risposta naturale dell'utente.
    La risposta viene analizzata da LLM per estrarre gli attributi, inclusa mcu_series.
    """
    
    logger.info("📋 Raccolta configurazione progetto STM32CubeMX...")
    
    prompt = {
        "instruction": """Configurazione Progetto STM32CubeMX
        
Per favore specifica (in linguaggio naturale):
- Path file .ioc (se disponibile, altrimenti non necessario)
- Nome della board STM32 (es: STM32F401VCHx, STM32H743ZI, STM32N657Z0HxQ)
- Nome del progetto
- Toolchain (es: STM32CubeIDE, Keil, IAR)

Esempio: "Crea progetto MyApp per STM32F401 con CubeIDE"
        """,
    }
    
    # === ESTRATTORE LLM ===
    # === IDEMPOTENCY CHECK ===
    # SKIP if board is already set, unless we are explicitly coming from a "change_board" route
    is_backtracking = state.route == "change_board"
    if state.board_name and state.board_name != "STM32F401VCHx" and not state.user_response and not is_backtracking:
        logger.info(f"⏭️  Idempotenza: Board '{state.board_name}' già configurata. Salto interrupt.")
        return state

    from src.assistant.utils import extract_user_response, get_llm
    from langchain_core.messages import SystemMessage, HumanMessage
    cfg = Configuration.from_runnable_config(config)
    llm = get_llm(config)
    llm_extractor = llm.with_structured_output(ProjectInfoExtraction)
    
    # --- Passo 1: Prova a usare il messaggio iniziale ---
    # SKIP Discovery if we are backtracking to change board (force new input)
    initial_board = None
    if not state.user_response and not is_backtracking:
        res = llm_extractor.invoke([
            SystemMessage(content=project_info_extraction_instructions),
            HumanMessage(content=f"Messaggio: {state.message}")
        ])
        initial_board = res.board_name
        # Salviamo quello che abbiamo trovato finora
        if res.board_name: state.board_name = res.board_name
        if res.mcu_series: state.mcu_series = res.mcu_series
        if res.project_name: state.project_name = res.project_name
        if res.toolchain: state.toolchain = res.toolchain
        if res.ioc_file_path: state.ioc_file_path = res.ioc_file_path


    # --- Passo 2: Verifica e Interrupt ---
    # Se non c'è una board nel messaggio o è generico, CHIEDI.
    if not initial_board or initial_board.lower() == "unknown":
        resume_value = None
        if not state.user_response:
            # Recupera board dal profilo per suggerimento
            last_board = state.persistent_context.get("board_name", "Nessuna") if state.persistent_context else "Nessuna"
            
            # Arricchiamo il prompt per l'utente
            dynamic_prompt = {
                "instruction": prompt["instruction"],
                "suggestion": f"� Ho visto che l'ultima volta hai usato: **{last_board}**. Vuoi usare la stessa o una nuova?"
            }
        if not state.user_response:
            # logger.info("⏸️ Interrupting: Requesting project info with profile suggestion.")
            # resume_value = interrupt(dynamic_prompt)
            logger.info("⏭️  BYPASS: Selezione automatica board -> 'STM32H7A3ZI'")
            user_text = "STM32H7A3ZI"
        else:
            # Dopo la ripresa: usa interrupt return value come priorità
            if resume_value and str(resume_value).strip():
                user_text = str(resume_value).strip()
            else:
                user_text = extract_user_response(state.user_response)
        state.user_response = ""
        
        res = llm_extractor.invoke([
            SystemMessage(content=project_info_extraction_instructions),
            HumanMessage(content=f"Risposta: {user_text}")
        ])
        
        # Gestione "USE_PROFILE"
        context_board = state.persistent_context.get("board_name") if state.persistent_context else None
        if res.board_name == "USE_PROFILE" and context_board:
            state.board_name = context_board
            state.mcu_series = state.persistent_context.get("mcu_series")
            logger.info(f"📋 Applicata board da profilo: {state.board_name}")
        else:
            if res.board_name: state.board_name = res.board_name
            if res.mcu_series: state.mcu_series = res.mcu_series
        
        if res.project_name: state.project_name = res.project_name
        if res.toolchain: state.toolchain = res.toolchain

    # --- Passo 3: Finalizzazione ---
    if not state.board_name:
        state.board_name = "STM32F401VCHx"
        state.mcu_series = "F4"
    
    # Estrazione automatica mcu_series se mancante
    if not state.mcu_series and state.board_name:
        extracted = extract_mcu_series_from_board(state.board_name)
        if extracted:
            state.mcu_series = extracted
            logger.info(f"📊 MCU series estratta automaticamente da board_name: {extracted}")
        else:
            # Fallback finale a F4
            state.mcu_series = "F4"
            logger.warning(f"⚠️  Impossibile estrarre serie, fallback a F4")
        
    state.project_name = state.project_name or "MySTM32Project"
    state.toolchain = state.toolchain or "STM32CubeIDE"
    
    logger.info(f"✓ Configurazione finale: {state.board_name} ({state.mcu_series})")
    
    # Sincronizza target per workflow AI (evita discrepanze e problemi di idempotenza)
    if state.mcu_series:
        state.target = f"stm32{state.mcu_series.lower()}"
        state.compression = "high" # Default
        logger.info(f"🎯 Sincronizzato target AI: {state.target}")

    state.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return state

def search_and_install_stm32_package(state: MasterState, config: RunnableConfig = None) -> MasterState:
    """
    Nodo che clona l'intero package STM32 da GitHub e lo salva in ~/STM32Cube/Repository/
    Scarica AUTOMATICAMENTE l'ultima versione disponibile dal repository.
    """
    
    logger.info("🔍 Ricerca e installazione package STM32 da GitHub...")
    
    cfg = Configuration.from_runnable_config(config)
    
    # === 0. ESTRAI MCU_SERIES DALLO STATE ===
    
    board_series = state.mcu_series
    
    if not board_series:
        logger.error(f"❌ mcu_series non specificato nello state!")
        state.package_installation_success = False
        state.package_error_message = "mcu_series non disponibile"
        return state
    
    logger.info(f"📊 Serie MCU: {board_series}")
    
    # === 1. VERIFICA SE PACKAGE È GIÀ INSTALLATO ===
    
    stm32_cube_repo = cfg.stm32_repo_path
    os.makedirs(stm32_cube_repo, exist_ok=True)
    
    # Cartelle package per questa serie (es: STM32Cube_FW_N6_*)
    existing_packages = []
    for folder in os.listdir(stm32_cube_repo):
        if board_series in folder and os.path.isdir(os.path.join(stm32_cube_repo, folder)):
            existing_packages.append(folder)
    
    if existing_packages:
        logger.info(f"✓ Package STM32{board_series} già presente!")
        # Ordina per trovare la più recente (per nome)
        existing_packages.sort()
        latest = existing_packages[-1]  # Ultim'ultima in ordine alfabetico
        logger.info(f"  Cartelle trovate: {len(existing_packages)}")
        for pkg in existing_packages:
            marker = " ← LATEST" if pkg == latest else ""
            logger.info(f"    - {pkg}{marker}")
        
        state.package_installation_success = True
        state.package_installation_path = os.path.join(stm32_cube_repo, latest)
        logger.info(f"✓ Uso: {latest}")
        logger.info("✓ Installazione saltata (già presente)")
        return state
    
    logger.info(f"📥 Package non trovato, procedo al download da GitHub...")
    
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
    
    # === 3. SCARICA LATEST RELEASE TAG DA GITHUB ===
    
    try:
        logger.info(f"🔎 Ricerca latest release tag...")
        
        # Usa git ls-remote per ottenere i tag senza clonare tutto
        cmd_tags = ["git", "ls-remote", "--tags", github_url]
        result_tags = subprocess.run(cmd_tags, capture_output=True, text=True, timeout=30)
        
        if result_tags.returncode != 0:
            logger.warning(f"⚠️  Impossibile leggere i tag, uso main branch")
            latest_version = "main"  # Fallback
        else:
            # Estrai i tag (filtro solo "v*" e no "^{}")
            tags = []
            for line in result_tags.stdout.strip().split('\n'):
                if 'refs/tags/' in line:
                    tag = line.split('refs/tags/')[-1].replace('^{}', '')
                    if tag.startswith('v') and '^{}' not in tag:
                        tags.append(tag)
            
            if tags:
                # Ordina versioni (vX.Y.Z) in ordine decrescente
                tags.sort(key=lambda x: [int(p) if p.isdigit() else 0 for p in x[1:].split('.')], reverse=True)
                latest_version = tags[0]
                logger.info(f"✓ Latest release trovato: {latest_version}")
                logger.info(f"  Disponibili: {len(tags)} version(i)")
                logger.info(f"    Top 3: {', '.join(tags[:3])}")
            else:
                logger.warning(f"⚠️  Nessun tag trovato, uso main branch")
                latest_version = "main"
    
    except Exception as e:
        logger.warning(f"⚠️  Errore durante lettura tag: {e}, uso main branch")
        latest_version = "main"
    
    logger.info(f"📥 Versione da installare: {latest_version}")
    
    # === 4. CLONE DA GITHUB ===
    
    # Path temporanei
    temp_clone_path = f"/tmp/STM32Cube{board_series}_{state.timestamp}"
    
    try:
        logger.info(f"📥 Clone ricorsivo in corso (timeout: 10 minuti)...")
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
            timeout=600  # 10 minuti
        )
        
        if result.returncode != 0:
            logger.error(f"❌ Clone fallito!")
            logger.error(f"Return code: {result.returncode}")
            logger.error(f"Stderr: {result.stderr}")
            raise RuntimeError(f"Git clone failed: {result.stderr}")
        
        logger.info(f"✓ Repository clonato: {temp_clone_path}")
        
        # === 5. VERIFICA STRUTTURA ===
        
        logger.info(f"✅ Verifica struttura repository...")
        
        required_dirs = ["Drivers", "Middlewares", "Projects"]
        for dir_name in required_dirs:
            dir_path = os.path.join(temp_clone_path, dir_name)
            if os.path.isdir(dir_path):
                logger.info(f"  ✓ {dir_name}/ presente")
            else:
                logger.warning(f"  ⚠️  {dir_name}/ mancante (continuo comunque)")
        
        # === 6. ESTRAI VERSIONE DAL FOLDER O TAG ===
        
        # Cerca Release_Notes.html per estrarre versione
        release_notes_path = os.path.join(temp_clone_path, "Release_Notes.html")
        version_info = latest_version  # Default
        
        if os.path.exists(release_notes_path):
            try:
                with open(release_notes_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    # Cerca pattern tipo "V1.2.0" o "v1.2.0"
                    
                    match = re.search(r'[Vv](\d+\.\d+\.\d+)', content)
                    if match:
                        version_info = f"V{match.group(1)}"
                        logger.info(f"✓ Versione estratta da Release_Notes: {version_info}")
            except Exception as e:
                logger.warning(f"⚠️  Non posso leggere Release_Notes: {e}")
        
        # Converti tag format: v1.2.0 → V1_2_0
        if version_info.startswith('v'):
            version_str = version_info[1:].replace(".", "_")
        else:
            version_str = version_info.replace(".", "_")
        
        final_folder_name = f"STM32Cube_FW_{board_series}_{version_str}"
        
        final_install_path = os.path.join(stm32_cube_repo, final_folder_name)
        
        logger.info(f"📂 Cartella installazione: {final_folder_name}")
        
        # === 7. SPOSTA IL REPOSITORY NEL POSTO FINALE ===
        
        logger.info(f"📦 Spostamento repository nella cartella finale...")
        
        # Se esiste già, rinomina il vecchio
        if os.path.exists(final_install_path):
            logger.warning(f"Cartella già presente, rinomino il vecchio...")
            old_backup = f"{final_install_path}_backup_{state.timestamp}"
            os.rename(final_install_path, old_backup)
            logger.info(f"  Vecchia cartella: {old_backup}")
        
        # Sposta clone path → final path
        shutil.move(temp_clone_path, final_install_path)
        logger.info(f"✓ Repository spostato")
        
        # === 8. VERIFICA INSTALLAZIONE ===
        
        logger.info(f"✅ Verifica installazione...")
        
        # Controlla che i file critici esistano
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
                logger.warning(f"  ⚠️  {critical_path}/ non trovato")
        
        # === 9. CONTA FILE TOTALI ===
        
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
        
        logger.info(f"📊 Statistiche installazione:")
        logger.info(f"  Directory: {total_dirs}")
        logger.info(f"  File: {total_files}")
        logger.info(f"  Spazio: {total_size / 1024 / 1024:.1f} MB")
        logger.info(f"  Versione: {version_info}")
        
        state.package_installation_success = True
        state.package_installation_path = final_install_path
        logger.info(f"✓✓✓ Package {board_series} installato con successo! ✓✓✓")
        
    except Exception as e:
        logger.error(f"❌ Errore durante installazione: {str(e)}")
        logger.exception(e)
        state.package_installation_success = False
        state.package_error_message = str(e)
        
        # Cleanup se fallisce
        try:
            if os.path.exists(temp_clone_path):
                shutil.rmtree(temp_clone_path)
                logger.info("Cleanup completato (errore)")
        except:
            pass
    
    return state


def check_package_installation(state: MasterState) -> Literal["generate_cubemx_script", "finalize_project"]:
    """
    Controlla se l'installazione del package è andata a buon fine.
    Se fallisce, salta direttamente a finalize con errore.
    """
    if state.package_installation_success:
        logger.info("✓ Package installato, continuo con generazione script")
        return "generate_cubemx_script"
    else:
        logger.error(f"❌ Installazione package fallita: {state.package_error_message}")
        state.firmware_generation_success = False
        state.firmware_error_message = f"Package installation failed: {state.package_error_message}"
        return "finalize_project"


def generate_cubemx_script(state: MasterState, config: RunnableConfig = None) -> MasterState:
    folder = f"{state.project_name}_{state.timestamp}"
    state.firmware_project_path = os.path.join(state.base_dir, folder)

    lines = [f"login {state.st_email} {state.st_password} y"]
    
    # Cerca un template pre-generato se l'utente non ha fornito un suo .ioc
    effective_ioc = state.ioc_file_path
    if not effective_ioc:
        effective_ioc = get_template_ioc_path(state.board_name, state.mcu_series)

    if effective_ioc:
        # Se abbiamo un .ioc (utente o template), usiamo SOLO config load
        # Questo bypassa tutti i pop-up interattivi di raccomandazione
        logger.info(f"📂 Usando caricamento configurazione: {effective_ioc}")
        lines.append(f'config load "{effective_ioc}"')
    else:
        # Fallback estremo: load board (rischio pop-up)
        logger.warning(f"⚠️  Nessun .ioc disponibile, fallback su caricamento board generico")
        lines.append(f"load {state.board_name}")

    lines += [
        f"project name {state.project_name}",
        f'project toolchain "{state.toolchain}"',
        f"project path {state.firmware_project_path}",
        "project generate",
        "exit"
    ]

    state.firmware_script_content = "\n".join(lines)
    state.firmware_script_path = f"/tmp/script_{state.timestamp}.scr"
    with open(state.firmware_script_path, "w") as f:
        f.write(state.firmware_script_content)
    
    logger.info("✓ Script CubeMX generato")
    return state


def recover_with_ioc_fallback(state: MasterState) -> bool:
    """
    Tenta di recuperare generando un file .ioc valido e modificando lo script.
    """
    logger.info("🚑 Attempting Recovery with IOC Fallback...")
    
    # 1. Cerca il template più adatto
    fallback_ioc = get_template_ioc_path(state.board_name, state.mcu_series)
    
    if not fallback_ioc:
        logger.error("❌ Nessun template disponibile per il fallback!")
        return False
        
    # 2. Aggiorna lo script per caricare il template
    lines = [
        f"login {state.st_email} {state.st_password} y",
        f'config load "{fallback_ioc}"',
        f"project name {state.project_name}",
        f'project toolchain "{state.toolchain}"',
        f"project path {state.firmware_project_path}",
        "project generate",
        "exit"
    ]
    
    # Sovrascrive lo script esistente
    state.firmware_script_content = "\n".join(lines)
    # state.firmware_script_path è già settato, lo riusiamo
    with open(state.firmware_script_path, "w") as f:
        f.write(state.firmware_script_content)
        
    logger.info("✓ CubeMX Script rewritten for Fallback")
    return True


def execute_generation(state: MasterState, config: RunnableConfig = None) -> MasterState:
    """
    Genera il progetto in una directory TEMPORANEA LOCALE (/tmp) 
    e poi lo sposta nella cartella finale per evitare problemi di locking su filesystem di rete.
    """
    import time
    
    # === 1. CREA DIRECTORY TEMPORANEA LOCALE ===
    temp_project_path = f"/tmp/stm32_{state.timestamp}"
    os.makedirs(temp_project_path, exist_ok=True)
    logger.info(f"📂 Directory temporanea creata: {temp_project_path}")
    
    # === 2. MODIFICA LO SCRIPT PER USARE IL PATH TEMPORANEO ===
    # Lo script è già stato creato da generate_cubemx_script, lo rileggiamo e modifichiamo
    with open(state.firmware_script_path, "r") as f:
        original_script = f.read()
    
    # Sostituiamo il path finale con quello temporaneo
    temp_script = original_script.replace(
        f"project path {state.firmware_project_path}",
        f"project path {temp_project_path}"
    )
    
    # Scriviamo lo script modificato
    temp_script_path = f"/tmp/script_temp_{state.timestamp}.scr"
    with open(temp_script_path, "w") as f:
        f.write(temp_script)
    
    logger.info(f"✏️  Script modificato per usare path temporaneo")
    
    # === 3. ESEGUI CUBEMX SULLA DIRECTORY TEMPORANEA ===
    # Su macOS (Darwin) non usiamo xvfb-run. Su Linux lo usiamo se necessario.
    if platform.system() == "Darwin":
        cmd = [state.cubemx_path, "-q", temp_script_path]
    else:
        # Tenta di usare xvfb-run su Linux se disponibile, altrimenti fallback diretto
        if shutil.which("xvfb-run"):
            cmd = ["xvfb-run", "-a", state.cubemx_path, "-q", temp_script_path]
        else:
            cmd = [state.cubemx_path, "-q", temp_script_path]
    
    try:
        logger.info(f"🚀 Executing CubeMX in temp dir (Attempt 1, timeout: 300s)...")
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if res.returncode != 0:
            logger.warning(f"⚠️  Generation Failed (RC={res.returncode}). Trying Fallback...")
            # FALLBACK: Riprova con template .ioc
            if recover_with_ioc_fallback(state):
                # Aggiorna anche lo script temporaneo
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
            logger.info("✓ Generazione completata in temp dir, attendo creazione cartelle...")
            time.sleep(2)
            
            # === 4. VERIFICA CHE LA GENERAZIONE SIA COMPLETA ===
            for attempt in range(10):
                # Cerchiamo Src e Inc nella directory temporanea
                src_exists = any(os.path.isdir(os.path.join(dp, "Src")) for dp, dn, filenames in os.walk(temp_project_path))
                inc_exists = any(os.path.isdir(os.path.join(dp, "Inc")) for dp, dn, filenames in os.walk(temp_project_path))
                
                if src_exists and inc_exists:
                    logger.info("✓ Cartelle Src/ e Inc/ verificate nella directory temporanea")
                    break
                
                logger.info(f"Attesa cartelle in temp dir... attempt {attempt+1}/10")
                time.sleep(1)
            else:
                logger.warning("⚠️  Cartelle potrebbero non essere completamente create")
            
            # === 5. SPOSTA DALLA DIRECTORY TEMPORANEA ALLA FINALE ===
            logger.info(f"📦 Spostamento progetto dalla temp dir alla destinazione finale...")
            
            # Crea la directory base se non esiste
            os.makedirs(os.path.dirname(state.firmware_project_path), exist_ok=True)
            
            # Se la destinazione finale esiste già, rimuovila
            if os.path.exists(state.firmware_project_path):
                logger.warning(f"⚠️  Cartella di destinazione già esistente, rimuovo...")
                shutil.rmtree(state.firmware_project_path)
            
            # Sposta tutto dalla temp alla finale
            shutil.move(temp_project_path, state.firmware_project_path)
            logger.info(f"✓✓✓ Progetto spostato con successo in: {state.firmware_project_path}")
            
        else:
            state.firmware_error_message = res.stderr or f"Return code {res.returncode}"
            logger.error(f"❌ Generazione fallita: {state.firmware_error_message}")
    
    except subprocess.TimeoutExpired:
        logger.error("❌ First attempt TIMED OUT. Trying Fallback...")
        # FALLBACK on timeout
        try:
            if recover_with_ioc_fallback(state):
                # Aggiorna script temporaneo con fallback
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
                    # Sposta anche in caso di fallback riuscito
                    logger.info("✓ Fallback riuscito, sposto in destinazione finale...")
                    os.makedirs(os.path.dirname(state.firmware_project_path), exist_ok=True)
                    if os.path.exists(state.firmware_project_path):
                        shutil.rmtree(state.firmware_project_path)
                    shutil.move(temp_project_path, state.firmware_project_path)
                    logger.info(f"✓ Progetto spostato dopo fallback")
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
        # === 6. CLEANUP SCRIPT TEMPORANEI ===
        try:
            os.remove(state.firmware_script_path)
            logger.info("✓ Cleanup script originale")
        except OSError:
            pass
        
        try:
            os.remove(temp_script_path)
            logger.info("✓ Cleanup script temporaneo")
        except OSError:
            pass
        
        # Cleanup temp dir SE ancora esiste (caso di errore)
        if os.path.exists(temp_project_path):
            try:
                shutil.rmtree(temp_project_path)
                logger.info("✓ Cleanup temp directory (errore)")
            except Exception as cleanup_err:
                logger.warning(f"⚠️  Non posso rimuovere temp dir: {cleanup_err}")
    
    logger.info(f"✓ Firmware generato: {state.firmware_project_path}" if state.firmware_generation_success else f"✗ Firmware fallito: {state.firmware_error_message}")
    return state



def finalize_project(state: MasterState, config: RunnableConfig = None) -> MasterState:
    if state.firmware_generation_success:
        print(f"✓ Progetto firmware generato: {state.firmware_project_path}")
        state.firmware_project_dir = state.firmware_project_path
    else:
        print(f"✗ Errore firmware: {state.firmware_error_message}")
    return state

    return state
