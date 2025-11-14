import os
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
            "/mnt/shared-storage/mrusso/STM32CubeMX"
        )
    )
    cubemx_path: str = field(
        default_factory=lambda: os.environ.get(
            "CUBEMX_PATH",
            "/mnt/shared-storage/mrusso/nuova/STM32CubeMX/STM32CubeMX"
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
            "/mnt/shared-storage/mrusso/resnet_v1_32_32_tfs.h5"
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
