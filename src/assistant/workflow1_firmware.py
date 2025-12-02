import os
import subprocess
import shutil
import re
import json
import logging
from typing import Optional, Literal
from datetime import datetime

from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
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

Esempi:
- Input: "Crea un progetto per STM32F401 con CubeIDE, nome MyApp"
  Output: {"ioc_file_path": null, "board_name": "STM32F401", "mcu_series": "F4", "project_name": "MyApp", "toolchain": "STM32CubeIDE"}

- Input: "Ho un file config.ioc in ~/boards/, usa quello"
  Output: {"ioc_file_path": "~/boards/config.ioc", "board_name": null, "mcu_series": null, "project_name": null, "toolchain": null}

- Input: "STM32H743ZI, progetto NeuralNet, Keil"
  Output: {"ioc_file_path": null, "board_name": "STM32H743ZI", "mcu_series": "H7", "project_name": "NeuralNet", "toolchain": "Keil"}

- Input: "Voglio usare STM32N657Z0HxQ per questo"
  Output: {"ioc_file_path": null, "board_name": "STM32N657Z0HxQ", "mcu_series": "N6", "project_name": null, "toolchain": null}
"""


# ============================================================================
# WORKFLOW 1: FIRMWARE GENERATION
# ============================================================================

def collect_project_info(state: MasterState, config: dict) -> MasterState:
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
    
    # L'utente risponde in linguaggio naturale
    user_response = interrupt(prompt)
    
    # Gestisci il caso in cui sia dict o stringa/int
    if isinstance(user_response, dict):
        user_text = user_response.get("response", user_response.get("input", str(user_response)))
    else:
        user_text = str(user_response)
    
    # Default: STM32F401 with CubeIDE
    if not user_text or user_text.strip() == "":
        user_text = "STM32F401VCHx, MyProject, STM32CubeIDE"
    
    logger.info(f"📝 User input RAW: '{user_text}'")
    
    # === USA LLM PER ESTRARRE GLI ATTRIBUTI ===
    
    cfg = Configuration.from_runnable_config(config)
    
    llm = ChatOllama(
        model=cfg.local_llm,
        temperature=0,
        num_ctx=cfg.llm_context_window
    )
    
    # Crea LLM con structured output
    llm_extractor = llm.with_structured_output(ProjectInfoExtraction)
    
    logger.info(f"🤖 Estrazione attributi da LLM...")
    
    extraction_result = llm_extractor.invoke([
        SystemMessage(content=project_info_extraction_instructions),
        HumanMessage(content=f"Risposta utente: {user_text}")
    ])
    
    logger.info(f"✓ Estrazione completata:")
    logger.info(f"  ioc_file_path: {extraction_result.ioc_file_path}")
    logger.info(f"  board_name: {extraction_result.board_name}")
    logger.info(f"  mcu_series: {extraction_result.mcu_series}")  # ✅ NUOVO
    logger.info(f"  project_name: {extraction_result.project_name}")
    logger.info(f"  toolchain: {extraction_result.toolchain}")
    
    # === APPLICA GLI ATTRIBUTI ESTRATTI ALLO STATE ===
    
    # ioc_file_path: Se estratto, usa quello; altrimenti None
    state.ioc_file_path = extraction_result.ioc_file_path or None
    
    # board_name: Se estratto, usa quello; altrimenti usa default o quello nello state
    state.board_name = extraction_result.board_name or state.board_name or "STM32F401VCHx"
    
    # mcu_series: Se estratto, usa quello; altrimenti calcola da board_name o usa ""  # ✅ NUOVO
    state.mcu_series = extraction_result.mcu_series or ""
    
    # project_name: Se estratto, usa quello; altrimenti usa quello nello state
    state.project_name = extraction_result.project_name or state.project_name or "MySTM32Project"
    
    # toolchain: Se estratto, usa quello; altrimenti usa quello nello state
    state.toolchain = extraction_result.toolchain or state.toolchain or "STM32CubeIDE"
    
    logger.info(f"✓ State aggiornato:")
    logger.info(f"  ioc_file_path: {state.ioc_file_path}")
    logger.info(f"  board_name: {state.board_name}")
    logger.info(f"  mcu_series: {state.mcu_series}")  # ✅ NUOVO
    logger.info(f"  project_name: {state.project_name}")
    logger.info(f"  toolchain: {state.toolchain}")
    
    # === VALIDAZIONI ===
    
    if not state.ioc_file_path and not state.board_name:
        logger.warning("⚠️  Nè ioc_file_path nè board_name specificati, uso default")
        state.board_name = "STM32F401VCHx"
        state.mcu_series = "F4"
    
    if state.ioc_file_path and not os.path.exists(state.ioc_file_path):
        raise FileNotFoundError(f"❌ .ioc file non trovato: {state.ioc_file_path}")
    
    state.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info("✓ Configurazione firmware raccolta")
    return state

def search_and_install_stm32_package(state: MasterState, config: dict) -> MasterState:
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
    
    stm32_cube_repo = os.path.expanduser("~/STM32Cube/Repository")
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


def generate_cubemx_script(state: MasterState, config: dict) -> MasterState:
    folder = f"{state.project_name}_{state.timestamp}"
    state.firmware_project_path = os.path.join(state.base_dir, folder)

    lines = [f"login {state.st_email} {state.st_password} y"]
    if state.ioc_file_path:
        if state.board_name:
            lines.append(f"load {state.board_name}")
        lines.append(f'config load "{state.ioc_file_path}"')
    else:
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


def execute_generation(state: MasterState, config: dict) -> MasterState:
    os.makedirs(state.firmware_project_path, exist_ok=True)
    cmd = ["xvfb-run", "-a", state.cubemx_path, "-q", state.firmware_script_path]
    
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        state.firmware_generation_success = (res.returncode == 0)
        if state.firmware_generation_success:
            logger.info("Attendo creazione cartelle...")
            time.sleep(2)
            
            for attempt in range(10):
                src_dir = os.path.join(state.firmware_project_path, "Src")
                inc_dir = os.path.join(state.firmware_project_path, "Inc")
                
                if os.path.exists(src_dir) and os.path.exists(inc_dir):
                    logger.info("✓ Cartelle create con successo")
                    break
                
                logger.info(f"Attesa cartelle... attempt {attempt+1}/10")
                time.sleep(1)
            else:
                logger.warning("⚠ Cartelle potrebbero non essere completamente create")
        else:
            state.firmware_error_message = res.stderr or f"Return code {res.returncode}"
    except Exception as e:
        state.firmware_generation_success = False
        state.firmware_error_message = str(e)
    finally:
        try:
            os.remove(state.firmware_script_path)
        except OSError:
            pass
    
    logger.info(f"✓ Firmware generato: {state.firmware_project_path}" if state.firmware_generation_success else f"✗ Firmware fallito: {state.firmware_error_message}")
    return state


def finalize_project(state: MasterState, config: dict) -> MasterState:
    if state.firmware_generation_success:
        print(f"✓ Progetto firmware generato: {state.firmware_project_path}")
        state.firmware_project_dir = state.firmware_project_path
    else:
        print(f"✗ Errore firmware: {state.firmware_error_message}")
    return state


# ============================================================================
# DECISION NODES - COLLEGA RAMI SEQUENZIALI
# ============================================================================

def decide_continue_to_ai(state: MasterState, config: dict) -> MasterState:
    """
    Nodo di decisione dopo finalize_project.
    La risposta viene analizzata da un LLM. Così anche se l'utente risponde in modo non strutturato, possiamo interpretarla. 
    """
    
    logger.info("📋 Decisione: Continuare verso analisi AI?")
    
    prompt = {
        "instruction": "Il firmware è stato generato con successo! Vuoi continuare con l'analisi del modello AI o terminare qui?",
    }
    
    # L'utente risponde in linguaggio naturale
    user_response = interrupt(prompt)
    
    # DEBUG: Stampa quello che hai ricevuto
    # logger.info(f"🔍 user_response RAW: {user_response}")
    # logger.info(f"🔍 user_response TYPE: {type(user_response)}")
    
    # Gestisci il caso in cui sia dict o stringa/int
    if isinstance(user_response, dict):
        user_text = user_response.get("response", user_response.get("input", str(user_response)))
    else:
        user_text = str(user_response)
    
    # logger.info(f"🔍 user_text ESTRATTO: '{user_text}'")
    
    # === USA LLM PER ANALIZZARE LA RISPOSTA ===
    
    cfg = Configuration.from_runnable_config(config)
    
    llm = ChatOllama(
        model=cfg.local_llm,
        temperature=0,
        num_ctx=cfg.llm_context_window
    )
     
    # Prompt SEMPLIFICATO per l'LLM
    analysis_prompt = f"""Analizza questa risposta e rispondi SOLO con una di queste due parole:
    
Risposta: "{user_text}"

Se l'utente dice SÌ / CONTINUA / PROCEDI → rispondi: CONTINUARE
Se l'utente dice NO / TERMINA / STOP → rispondi: TERMINARE

Risposta:"""
    
    logger.info(f"📝 Analysis prompt:\n{analysis_prompt}")
    
    response = llm.invoke([
        SystemMessage(content="Rispondi SOLO con una parola: CONTINUARE o TERMINARE"),
        HumanMessage(content=analysis_prompt)
    ])
    
    decision_text = response.content.strip().upper()
    
    logger.info(f"🤖 LLM Decision RAW: '{decision_text}'")
    
    # Interpreta la decisione (più tollerante)
    if "CONTINUARE" in decision_text or "CONTINUA" in decision_text or "SÌ" in decision_text:
        logger.info("✓ CONTINUE DETECTED - Going to AI Analysis")
        state.route = "continue_to_ai"
    else:
        logger.info("✓ TERMINATE DETECTED - Going to END")
        state.route = "end_workflow"
    
    logger.info(f"📊 Final state.route: {state.route}")
    
    return state


def decide_continue_to_integration(state: MasterState, config: dict) -> MasterState:
    """
    Nodo di decisione dopo finalize_analysis.
    La risposta viene analizzata da un LLM. Così anche se l'utente risponde in modo non strutturato, possiamo interpretarla.
    """
    
    logger.info("📋 Decisione: Continuare verso integrazione?")
    
    prompt = {
        "instruction": "L'analisi AI è stata completata con successo! Vuoi continuare con l'integrazione del codice AI nel firmware o terminare qui?",
    }
    
    # L'utente risponde in linguaggio naturale
    user_response = interrupt(prompt)
    
    # DEBUG: Stampa quello che hai ricevuto
    # logger.info(f"🔍 user_response RAW: {user_response}")
    # logger.info(f"🔍 user_response TYPE: {type(user_response)}")
    
    # Gestisci il caso in cui sia dict o stringa/int
    if isinstance(user_response, dict):
        user_text = user_response.get("response", user_response.get("input", str(user_response)))
    else:
        user_text = str(user_response)
    
    # logger.info(f"🔍 user_text ESTRATTO: '{user_text}'")
    
    # === USA LLM PER ANALIZZARE LA RISPOSTA ===
    
    cfg = Configuration.from_runnable_config(config)
    
    llm = ChatOllama(
        model=cfg.local_llm,
        temperature=0,
        num_ctx=cfg.llm_context_window
    )
     
    # Prompt SEMPLIFICATO per l'LLM
    analysis_prompt = f"""Analizza questa risposta e rispondi SOLO con una di queste due parole:
    
Risposta: "{user_text}"

Se l'utente dice SÌ / CONTINUA / PROCEDI / INTEGRA → rispondi: CONTINUARE
Se l'utente dice NO / TERMINA / STOP / FINE → rispondi: TERMINARE

Risposta:"""
    
    logger.info(f"📝 Analysis prompt:\n{analysis_prompt}")
    
    response = llm.invoke([
        SystemMessage(content="Rispondi SOLO con una parola: CONTINUARE o TERMINARE"),
        HumanMessage(content=analysis_prompt)
    ])
    
    decision_text = response.content.strip().upper()
    
    logger.info(f"🤖 LLM Decision RAW: '{decision_text}'")
    
    # Interpreta la decisione (più tollerante)
    if "CONTINUARE" in decision_text or "CONTINUA" in decision_text or "SÌ" in decision_text:
        logger.info("✓ CONTINUE DETECTED - Going to Integration")
        state.route = "continue_to_integration"
    else:
        logger.info("✓ TERMINATE DETECTED - Going to END")
        state.route = "end_workflow"
    
    logger.info(f"📊 Final state.route: {state.route}")
    
    return state


def decision_continue_routing(state: MasterState) -> Literal["collect_analysis_info", "collect_integration_info", "end"]:
    """
    Funzione di routing per i nodi di decisione.
    Determina il prossimo nodo in base alla scelta dell'utente.
    """
    
    if state.route == "continue_to_ai":
        logger.info("→ Routing verso: collect_analysis_info")
        return "collect_analysis_info"
    elif state.route == "continue_to_integration":
        logger.info("→ Routing verso: collect_integration_info")
        return "collect_integration_info"
    else:
        logger.info("→ Routing verso: END")
        return "end"

