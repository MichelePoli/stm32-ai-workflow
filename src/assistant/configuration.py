import os
import subprocess
import shutil
import platform
from dataclasses import dataclass, field, fields
from typing import Any, Optional

from langchain_core.runnables import RunnableConfig
from typing_extensions import Annotated


@dataclass(kw_only=True)
class Configuration:
    """Configurazione per il workflow STM32CubeMX con AI."""

    # ============================================================================
    # CREDENZIALI ST (solo da environment variables per sicurezza)
    # ============================================================================
    
    st_email: str = field(
        default_factory=lambda: os.environ.get("ST_EMAIL", "")
    )
    st_password: str = field(
        default_factory=lambda: os.environ.get("ST_PASSWORD", "")
    )

    # ============================================================================
    # PERCORSI STM32
    # ============================================================================
    
    base_dir: str = field(
        default_factory=lambda: os.environ.get(
            "BASE_DIR", 
            os.path.expanduser("~/STM32CubeMX")
        )
    )
    cubemx_path: str = field(
        default_factory=lambda: os.environ.get(
            "CUBEMX_PATH",
            os.path.expanduser("~/STM32CubeMX/STM32CubeMX") if platform.system() != "Darwin" else "/Applications/STMicroelectronics/STM32CubeMX.app/Contents/Resources/STM32CubeMX"
        )
    )
    stm32_repo_path: str = field(
        default_factory=lambda: os.environ.get(
            "STM32_REPO_PATH",
            os.path.expanduser("~/STM32Cube/Repository")
        )
    )
    stedgeai_path: str = field(
        default_factory=lambda: os.environ.get(
            "STEDGEAI_PATH",
            os.path.expanduser("~/stm32-ai_utilities/linux/stedgeai")
        )
    )

    # ============================================================================
    # CONFIGURAZIONE STM32
    # ============================================================================
    
    default_board: str = field(
        default_factory=lambda: os.environ.get("DEFAULT_BOARD", "STM32F401VCHx")
    )
    default_toolchain: str = field(
        default_factory=lambda: os.environ.get("DEFAULT_TOOLCHAIN", "STM32CubeIDE")
    )

    # ============================================================================
    # CONFIGURAZIONE LLM (ROUTING) - ✅ AGGIUNTO
    # ============================================================================
    
    local_llm: str = field(
        default_factory=lambda: os.environ.get("LOCAL_LLM", "mistral")
    )
    llm_temperature: float = field(
        default_factory=lambda: float(os.environ.get("LLM_TEMPERATURE", "0"))
    )
    llm_context_window: int = field(
        default_factory=lambda: int(os.environ.get("LLM_CONTEXT_WINDOW", "4096"))
    )

    # ============================================================================
    # CONFIGURAZIONE AI ANALYSIS - ✅ AGGIUNTO
    # ============================================================================
    
    ai_model_path: str = field(
        default_factory=lambda: os.environ.get(
            "AI_MODEL_PATH",
            "./data/models/default_model.h5"
        )
    )
    ai_output_dir: str = field(
        default_factory=lambda: os.environ.get("AI_OUTPUT_DIR", "./analisiAI")
    )
    ai_target: str = field(
        default_factory=lambda: os.environ.get("AI_TARGET", "stm32f401")
    )
    ai_compression: str = field(
        default_factory=lambda: os.environ.get("AI_COMPRESSION", "high")
    )

    # ============================================================================
    # LOGGING
    # ============================================================================
    
    log_level: str = field(
        default_factory=lambda: os.environ.get("LOG_LEVEL", "INFO")
    )

    # ============================================================================
    # METODI
    # ============================================================================

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """
        Crea un'istanza di Configuration a partire da RunnableConfig.
        
        Priorità dei valori:
        1. RunnableConfig["configurable"]
        2. Environment variables
        3. Defaults dalla dataclass
        """
        runnable_config = (
            config.get("configurable", {}) 
            if config else {}
        )
        
        init_values: dict[str, Any] = {}
        
        for f in fields(cls):
            if not f.init:
                continue
            
            name = f.name
            
            # 1. Controlla RunnableConfig
            if name in runnable_config and runnable_config[name] is not None:
                init_values[name] = runnable_config[name]
            # 2. Controlla Environment Variables
            elif os.environ.get(name.upper()):
                env_val = os.environ.get(name.upper())
                # Converti tipi specifici
                if name.endswith("_temperature"):
                    init_values[name] = float(env_val)
                elif name.endswith("_context_window"):
                    init_values[name] = int(env_val)
                else:
                    init_values[name] = env_val
            # 3. Altrimenti usa il default_factory
        
        return cls(**init_values)

    def validate(self) -> bool:
        """
        Valida la configurazione.
        Ritorna True se ok, False se mancano parametri critici.
        """
        errors = []
        
        # Credenziali ST obbligatorie
        if not self.st_email:
            errors.append("ST_EMAIL non configurata (environment variable)")
        if not self.st_password:
            errors.append("ST_PASSWORD non configurata (environment variable)")
        
        # Path obbligatori
        if not os.path.exists(self.base_dir):
            errors.append(f"BASE_DIR non esiste: {self.base_dir}")
        if not os.path.exists(self.cubemx_path):
            errors.append(f"CUBEMX_PATH non esiste: {self.cubemx_path}")
        
        # LLM configurato
        if not self.local_llm:
            errors.append("LOCAL_LLM non configurato")
        
        # AI Model path
        if not os.path.exists(self.ai_model_path):
            errors.append(f"AI_MODEL_PATH non esiste: {self.ai_model_path}")
        
        if errors:
            for error in errors:
                print(f"❌ {error}")
            return False
        
        return True

    def get_python_path(self, env_name: str) -> str:
        """
        Ritorna il path dell'eseguibile python per un dato ambiente conda.
        Tenta di trovarlo dinamicamente per evitare hardcoded paths.
        """
        # 1. Tenta con 'conda run'
        try:
            cmd = ["conda", "run", "-n", env_name, "which", "python"]
            if platform.system() == "Windows":
                cmd = ["conda", "run", "-n", env_name, "where", "python"]
                
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            path = result.stdout.strip().split('\n')[0]
            if os.path.exists(path):
                return path
        except Exception:
            pass

        # 2. Fallback su percorsi comuni basati sulla piattaforma
        home = os.path.expanduser("~")
        if platform.system() == "Darwin": # macOS
            paths = [
                f"/Library/anaconda3/envs/{env_name}/bin/python",
                f"/opt/anaconda3/envs/{env_name}/bin/python",
                f"{home}/anaconda3/envs/{env_name}/bin/python",
                f"{home}/miniconda3/envs/{env_name}/bin/python",
                f"/usr/local/anaconda3/envs/{env_name}/bin/python"
            ]
        else: # Linux/Generic
            paths = [
                f"{home}/anaconda3/envs/{env_name}/bin/python",
                f"{home}/miniconda3/envs/{env_name}/bin/python",
                f"/opt/conda/envs/{env_name}/bin/python"
            ]

        for p in paths:
            if os.path.exists(p):
                return p

        return f"PYTHON_PATH_NOT_FOUND_{env_name}"

    def summary(self) -> str:
        """Ritorna un summary della configurazione."""
        return f"""
╔════════════════════════════════════════════════════════╗
║        CONFIGURAZIONE LANGGRAPH STM32 + AI             ║
╠════════════════════════════════════════════════════════╣
║ STM32                                                  ║
║   Board:         {self.default_board:<35} ║
║   Toolchain:     {self.default_toolchain:<35} ║
║   Base Dir:      {self.base_dir:<35} ║
║                                                        ║
║ LLM (Routing)                                          ║
║   Model:         {self.local_llm:<35} ║
║   Temperature:   {self.llm_temperature:<35} ║
║   Context:       {self.llm_context_window:<35} ║
║                                                        ║
║ AI Analysis                                            ║
║   Model:         {self.ai_model_path:<35} ║
║   Target:        {self.ai_target:<35} ║
║   Output:        {self.ai_output_dir:<35} ║
║   Compression:   {self.ai_compression:<35} ║
╚════════════════════════════════════════════════════════╝
"""
    
    def __repr__(self) -> str:
        """
        Override __repr__ to redact sensitive fields from logs.
        
        This prevents accidental password/token leakage in debug output.
        """
        safe_dict = {}
        for k, v in self.__dict__.items():
            # Redact fields containing sensitive keywords
            if any(sensitive in k.lower() for sensitive in ['password', 'token', 'key', 'secret', 'api']):
                safe_dict[k] = "***REDACTED***"
            else:
                safe_dict[k] = v
        
        # Format as readable string
        items = ", ".join(f"{k}={repr(v)}" for k, v in list(safe_dict.items())[:5])
        return f"Configuration({items}...)"
