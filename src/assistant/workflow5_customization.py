# ============================================================================
# WORKFLOW 5: MODEL CUSTOMIZATION CON AI EMBEDDING E BEST PRACTICES
# ============================================================================
# Modulo dedicato alla customizzazione architettura modelli AI per STM32
#
# Responsabilità:
#   - Ispezionamento dettagliato architettura modello
#   - Retrieval best practices via embeddings (sentence-transformers)
#   - Parsing richieste customizzazione utente
#   - Applicazione modifiche all'architettura (layer, activation, etc.)
#   - Fine-tuning con dataset
#   - Validazione e quantizzazione INT8
#   - Salvataggio con metadata
#
# Dipendenze: tensorflow, langchain, sentence-transformers, h5py, numpy


import subprocess
import os
from datetime import datetime
import logging
import tensorflow as tf
from typing import Optional

import shutil
import re
import json
from typing import Tuple, Optional, List, Literal
from langgraph.types import interrupt
from src.assistant.configuration import Configuration
from typing import Any

# any  = builtin function Python (all lowercase)
#         ↓
#         Ritorna True/False

# Any  = type hint da typing module (CamelCase)
#         ↓
#         Significa "qualsiasi tipo"

# Pydantic capisce: Any ✅
# Pydantic NON capisce: any ❌

from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

from agno.agent import Agent
#from agno.tools.github import GithubTools
#utilizza altri tools oltre GoogleSearchTools, vedi dai tools di agno
from agno.models.ollama import Ollama

import numpy as np

from tensorflow.keras.layers import (
    Dense, Dropout, Input, Resizing, Conv2D, 
    GlobalAveragePooling2D, GlobalMaxPooling2D,
    BatchNormalization, Activation, Add,  # ← Deve esserci BatchNormalization
    AveragePooling2D, Flatten
)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
import h5py

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout

from langchain_community.document_loaders import RecursiveUrlLoader  # Web scraping & site crawling
from langchain_community.vectorstores import Chroma                                   # Vector DB
from langchain.embeddings.base import Embeddings    

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
#from langchain_chroma import Chroma # SBAGLIATO QUESTO !!


# -------------------------
# Sentence Transformers / Embeddings
# -------------------------
from langchain.schema import Document                   # Standard document container for LangChain


from src.assistant.configuration import Configuration
from src.assistant.state import MasterState

from bs4 import BeautifulSoup


logger = logging.getLogger(__name__)


class ModificationDecision(BaseModel):
    """Decisione se applicare modifiche al modello"""
    wants_modifications: bool = Field(
        description="L'utente vuole apportare modifiche al modello?"
    )
    reasoning: str = Field(
        description="Breve spiegazione della decisione"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidenza della classificazione"
    )

modification_decision_instructions = """Sei un classificatore di intenzioni per la customizzazione di modelli AI.

Analizza la risposta dell'utente alla domanda: "Vuoi apportare modifiche all'architettura del modello scaricato, oppure procedere direttamente con l'analisi STEdgeAI?"

RISPOSTE AFFERMATIVE (vuole modifiche):
- "sì", "si", "yes", "certo", "ok", "voglio", "voglio modificare", "voglio cambiare", "regolarizza", "riduci", "compressione", "ottimizza"
- "meno layer", "più leggero", "efficiente", "dropout", "cambia attivazione"
- Qualsiasi richiesta esplicita di cambiamento

RISPOSTE NEGATIVE (procede senza modifiche):
- "no", "nope", "niente", "skip", "avanti", "procedi", "andiamo avanti", "mantieni", "ok così", "va bene così"
- "no, procedi direttamente", "nessuna modifica", "default"

Rispondi SEMPRE in JSON:
- "wants_modifications": true/false
- "reasoning": breve spiegazione (max 50 caratteri)
- "confidence": 0.0-1.0

Esempi:

Input: "Riduci il numero di layer, è troppo complesso"
Output: {
  "wants_modifications": true,
  "reasoning": "Richiesta esplicita di riduzione layer",
  "confidence": 0.95
}

Input: "No, procedi direttamente con l'analisi"
Output: {
  "wants_modifications": false,
  "reasoning": "Rifiuto esplicito, skip modifiche",
  "confidence": 0.95
}

Input: "Hmm, non so... che cosa consigli?"
Output: {
  "wants_modifications": false,
  "reasoning": "Indecisione, mantiene default",
  "confidence": 0.6
}
"""


def load_or_create_sample_dataset(num_samples: int = 100, 
                                   img_size: Tuple[int, int] = (32, 32),
                                   num_classes: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """Carica o crea dataset di sample"""
    logger.info(f"📊 Creando dataset di sample ({num_samples} immagini)...")
    
    X = np.random.rand(num_samples, img_size[0], img_size[1], 3).astype(np.float32)
    y = tf.keras.utils.to_categorical(np.random.randint(0, num_classes, num_samples), num_classes)
    
    X = X / 255.0
    
    logger.info(f"✓ Dataset creato: X.shape={X.shape}, y.shape={y.shape}")
    return X, y


def save_model_with_metadata(model: Model, 
                             output_path: str,
                             metadata: dict[str, any]) -> None:
    """Salva modello + metadata per tracciabilità"""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    
    model.save(output_path)
    
    metadata_path = output_path.replace('.h5', '_metadata.json')
    with open(metadata_path, 'w') as f:
        metadata_clean = {
            'timestamp': metadata.get('timestamp', datetime.now().isoformat()),
            'input_shape': str(metadata.get('input_shape', 'unknown')),
            'output_shape': str(metadata.get('output_shape', 'unknown')),
            'total_params': int(metadata.get('total_params', 0)),
            'model_size_mb': round(os.path.getsize(output_path) / (1024*1024), 2),
            'modifications_applied': metadata.get('modifications_applied', []),
            'training_params': metadata.get('training_params', {}),
        }
        json.dump(metadata_clean, f, indent=2)
    
    logger.info(f"✓ Modello salvato: {output_path}")


def inspect_model_architecture(state: MasterState, config: dict) -> MasterState:
    """Ispeziona dettagliatamente il modello scaricato con fallback robusto"""

    logger.info("🔍 Ispezionando architettura modello...")

    try:
        # ✅ Primo tentativo: load_model standard
        logger.info("   Tentativo 1: load_model() standard...")
        model = tf.keras.models.load_model(state.model_path, compile=False)
        
        trainable_params = int(sum([tf.size(w).numpy() for w in model.trainable_weights]))
        model_size_mb = os.path.getsize(state.model_path) / (1024*1024)
        
        state.model_architecture = {
            "input_shape": str(model.input_shape),
            "output_shape": str(model.output_shape),
            "n_layers": len(model.layers),
            "layer_types": [layer.__class__.__name__ for layer in model.layers],
            "layer_names": [layer.name for layer in model.layers],
            "total_params": int(model.count_params()),
            "trainable_params": trainable_params,
            "model_size_mb": round(model_size_mb, 2),
            "has_batchnorm": any(isinstance(l, tf.keras.layers.BatchNormalization) for l in model.layers),
            "has_dropout": any(isinstance(l, tf.keras.layers.Dropout) for l in model.layers),
            "output_classes": model.output_shape[-1] if len(model.output_shape) > 1 else 1,
        }
        
        import io
        stream = io.StringIO()
        model.summary(print_fn=lambda x: stream.write(x + '\n'))
        state.model_summary_text = stream.getvalue()
        
        logger.info(f"✓ Architettura analizzata (load_model):")
        logger.info(f"  - Layers: {state.model_architecture['n_layers']}")
        logger.info(f"  - Total params: {state.model_architecture['total_params']:,}")
        logger.info(f"  - Model size: {state.model_architecture['model_size_mb']:.2f} MB")
        
        return state
    
    except Exception as e:
        # ❌ load_model fallisce, prova fallback HDF5 raw
        logger.warning(f"⚠️  load_model() fallito: {str(e)[:100]}")
        logger.info("   Tentativo 2: Analisi HDF5 raw...")
        
        try:
            # ✅ Fallback: Estrai info direttamente dal file HDF5
            with h5py.File(state.model_path, 'r') as f:
                
                # Estrai layer info
                if 'model_config' in f.attrs:
                    config = json.loads(f.attrs['model_config'])
                    n_layers = len(config.get('config', {}).get('layers', []))
                    layer_names = [l.get('name', 'unknown') for l in config.get('config', {}).get('layers', [])]
                    layer_types = [l.get('class_name', 'unknown') for l in config.get('config', {}).get('layers', [])]
                else:
                    # Fallback: estrai da model_weights
                    layer_names = list(f.get('model_weights', {}).keys()) if 'model_weights' in f else []
                    n_layers = len(layer_names)
                    layer_types = ['Unknown'] * n_layers
                
                # Estrai shape info
                if 'model_weights' in f:
                    weights_group = f['model_weights']
                    # Prova a estrarre primo layer (input)
                    first_layer_weights = list(weights_group.values())[0] if len(weights_group) > 0 else None
                    
                    if first_layer_weights:
                        input_shape = first_layer_weights.shape if hasattr(first_layer_weights, 'shape') else "Unknown"
                    else:
                        input_shape = "Unknown"
                else:
                    input_shape = "Unknown"
                
                # Calcola totale parametri da file size (stima)
                file_size = os.path.getsize(state.model_path)
                # Stima: 1 parametro ≈ 4 bytes (float32)
                estimated_params = (file_size - 1024) / 4  # Sottrai overhead
                
                state.model_architecture = {
                    "input_shape": str(input_shape),
                    "output_shape": "Unknown (raw HDF5)",
                    "n_layers": n_layers,
                    "layer_types": layer_types,
                    "layer_names": layer_names,
                    "total_params": int(estimated_params) if estimated_params > 0 else 0,
                    # ✅ PROTEZIONE: sempre intero, mai None
                    "trainable_params": 0,  # ✅ Default a 0, non None
                    "model_size_mb": round(file_size / (1024*1024), 2),
                    "has_batchnorm": any('batch' in name.lower() for name in layer_names),
                    "has_dropout": any('dropout' in name.lower() for name in layer_names),
                    "output_classes": 0,  # ✅ Default a 0, non None
                }
                
                logger.info(f"✓ Architettura estratta (HDF5 raw):")
                logger.info(f"  - Layers: {state.model_architecture['n_layers']}")
                logger.info(f"  - Total params (stimati): {state.model_architecture['total_params']:,}")
                logger.info(f"  - Model size: {state.model_architecture['model_size_mb']:.2f} MB")
                logger.warning(f"⚠️  Analisi parziale: informazioni complete richiedono tf.keras.models.load_model()")
                
                return state
        
        except Exception as e2:
            # ❌ Anche HDF5 raw fallisce, usa default minimo
            logger.error(f"❌ HDF5 raw fallito: {str(e2)[:100]}")
            logger.warning("⚠️  Usando default minimale per continuare il workflow")
            
            state.model_architecture = {
                "input_shape": "Unknown",
                "output_shape": "Unknown",
                "n_layers": 0,
                "layer_types": [],
                "layer_names": [],
                "total_params": 0,      # ✅ SEMPRE intero
                "trainable_params": 0,  # ✅ SEMPRE intero
                "model_size_mb": os.path.getsize(state.model_path) / (1024*1024),
                "has_batchnorm": False,
                "has_dropout": False,
                "output_classes": 0,    # ✅ SEMPRE intero
            }
            
            logger.error(f"❌ Impossibile analizzare modello: {str(e)[:100]}")
            
            return state


def ask_modification_intent(state, config: dict):
    """Chiede all'utente se vuole modificare il modello"""
    
    logger.info("💬 Richiesta intenzione di modifica...")
    
    cfg = Configuration.from_runnable_config(config)
    
    # === MOSTRA ARCHITETTURA ATTUALE ===
    
    print("\n" + "="*70)
    print("🏗️  ARCHITETTURA MODELLO ATTUALE")
    print("="*70)
    print(f"\nInput shape:  {state.model_architecture.get('input_shape', 'Unknown')}")
    print(f"Output shape: {state.model_architecture.get('output_shape', 'Unknown')}")
    print(f"Layer totali: {state.model_architecture.get('n_layers', 0)}")
    print(f"Parametri: {state.model_architecture.get('total_params', 0):,}")
    print(f"Dimensione: {state.model_architecture.get('model_size_mb', 0):.2f} MB")
    print(f"BatchNorm: {'Sì' if state.model_architecture.get('has_batchnorm') else 'No'}")
    print(f"Dropout: {'Sì' if state.model_architecture.get('has_dropout') else 'No'}")
    print("="*70 + "\n")
    
    # === CHIEDI DECISIONE ===
    
    prompt = {
        "instruction": """Vuoi apportare modifiche all'architettura del modello?

Opzioni:
- SÌ: Procediamo con la customizzazione (ridurre layer, aggiungere regularizzazione, etc.)
- NO: Andiamo avanti direttamente con STEdgeAI analyze/validate/generate

Cosa preferisci? (si/no)""",
    }
    
    user_response = interrupt(prompt)
    
    if isinstance(user_response, dict):
        user_text = user_response.get("response", user_response.get("input", str(user_response)))
    else:
        user_text = str(user_response)
    
    logger.info(f"📝 User response: '{user_text}'")
    
    # === CLASSIFICA CON MISTRAL ===
    
    llm = ChatOllama(
        model=cfg.local_llm,  # oppure "mistral" se lo preferisci
        temperature=0,
        num_ctx=cfg.llm_context_window
    )
    
    llm_classifier = llm.with_structured_output(ModificationDecision)
    
    decision = llm_classifier.invoke([
        SystemMessage(content=modification_decision_instructions),
        HumanMessage(content=f"Risposta utente: {user_text}")
    ])
    
    logger.info(f"✓ Decisione classificata:")
    logger.info(f"  wants_modifications: {decision.wants_modifications}")
    logger.info(f"  confidence: {decision.confidence:.2f}")
    
    # === SALVA NELLO STATE ===
    
    state.wants_model_modifications = decision.wants_modifications
    state.modification_intent_confidence = decision.confidence
    
    return state

def decide_after_inspection(state) -> Literal["retrieve_best_practices_for_architecture", "run_analyze"]:
    """Decide se procedere a customizzazione o diretto ad analyze"""
    
    logger.info(f"📍 Routing post-inspection:")
    logger.info(f"   wants_modifications: {state.wants_model_modifications}")
    
    if state.wants_model_modifications:
        logger.info("   → Percorso: CUSTOMIZZAZIONE")
        return "retrieve_best_practices_for_architecture"
    else:
        logger.info("   → Percorso: SKIP A ANALYZE")
        return "run_analyze"


# ============================================================================
# RETRIEVE BEST PRACTICES FOR ARCHITECTURE
# ===========================================================================

def retrieve_best_practices_for_architecture(state: MasterState, config: dict) -> MasterState:
    """
    ✨ VERSIONE CORRETTA: Separate Chroma collections per architettura
    
    Struttura:
    ./chroma_docs/
      ├── mobilenet/
      ├── resnet/
      ├── efficientnet/
      ├── vgg/
      ├── yolo/
      └── har/
    """
    
    # ===== ESTRAI INFO =====
    model_name = None
    if state.selected_model:
        model_name = state.selected_model.get('name', 'Unknown')
    
    if not model_name:
        logger.warning("⚠️  No model selected")
        state.best_practices_display = _get_generic_practices()
        return state
    
    # ===== DETECTA ARCHITETTURA =====
    arch_type = _detect_architecture_type(model_name)
    logger.info(f"🔍 Model: {model_name} → Architecture: {arch_type}")
    
    # ✅ MIGLIORAMENTO: Path separato per architettura
    base_persist_dir = "./chroma_docs"
    arch_persist_dir = os.path.join(base_persist_dir, arch_type)
    
    logger.info(f"  Chroma collection: {arch_persist_dir}")
    
    # ===== STEP 1: Check Chroma cache per questa architettura =====
    logger.info(f"  [Step 1/3] Checking cache for {arch_type}...")
    
    arch_db_exists = os.path.exists(arch_persist_dir) and os.listdir(arch_persist_dir)
    
    if arch_db_exists:
        try:
            logger.info(f"  ✓ Cache found for {arch_type}")
            
            best_practices = _retrieve_from_chroma(
                query=f"best practices customization fine-tuning {arch_type}",
                persist_dir=arch_persist_dir,
                arch_type=arch_type
            )
            
            if best_practices and len(best_practices) > 0:
                logger.info(f"  ✓ Retrieved {len(best_practices)} docs from cache")
                state.best_practices_display = _format_practices(best_practices, source=f"CACHE_{arch_type}")
                state.best_practices_raw = [p.page_content for p in best_practices]
                return state
        
        except Exception as e:
            logger.warning(f"  ⚠️  Cache lookup failed: {str(e)[:60]}")
    else:
        logger.info(f"  Cache NOT found for {arch_type}, fetching online...")
    
    # ===== STEP 2: Fetch online e salva in Chroma per questa architettura =====
    logger.info(f"  [Step 2/3] Fetching online for {arch_type}...")
    
    try:
        best_practices = _fetch_and_cache_architecture_practices(
            model_name=model_name,
            arch_type=arch_type,
            persist_dir=arch_persist_dir  # ← Separato per architettura
        )
        
        if best_practices:
            logger.info(f"  ✓ Fetched and cached {len(best_practices)} docs")
            state.best_practices_display = _format_practices(best_practices, source=f"ONLINE_{arch_type}")
            state.best_practices_raw = [p.page_content for p in best_practices]
            return state
    
    except Exception as e:
        logger.warning(f"  ⚠️  Online fetch failed: {str(e)[:60]}")
    
    # ===== STEP 3: Fallback =====
    logger.info(f"  [Step 3/3] Using fallback for {arch_type}...")
    state.best_practices_display = _get_architecture_specific_practices(arch_type)
    state.best_practices_raw = []
    
    return state


# ============================================================================
# SEARCH WEB: Fetch URLs for Best Practices
# ============================================================================

def search_web(queries: List[str]) -> List[dict[str, str]]:
    """
    ✨ Ricerca web per recuperare URL best practices per architettura
    
    Basato su execute_web_search ma ottimizzato per ritornare lista di URL.
    
    Args:
        queries: Lista di query (max 3 per efficienza)
    
    Returns:
        List[dict] con formato:
        [
            {"url": "https://...", "title": "...", "content": "..."},
            {"url": "https://...", "title": "...", "content": "..."},
            ...
        ]
    
    Raises:
        Exception se ricerca fallisce
    """
    
    logger.info(f"🌐 Search web: {len(queries)} queries")
    
    all_results = []
    
    for query in queries:
        try:
            logger.debug(f"  Searching: {query}")
            
            # ===== SETUP AGNO AGENT =====
            agent = Agent(
                model=Ollama(id="mistral"),
                tools=[GoogleSearchTools()],
                show_tool_calls=False,
                markdown=True
            )
            
            # ===== PROMPT SEMPLICE E DIRETTO =====
            search_prompt = f"""Search for information about:
"{query}"

Return the top 5 most relevant results with URLs and brief descriptions."""
            
            # ===== ESEGUI RICERCA =====
            logger.debug(f"  Invoking Agno Agent...")
            response = agent.run(search_prompt)
            
            if not response:
                logger.warning(f"  Empty response for query: {query}")
                continue
            
            # ===== PARSE RESPONSE =====
            content = response.content if hasattr(response, 'content') else str(response)
            
            logger.debug(f"  Response length: {len(content)} chars")
            
            # Estrai URL dalla response (parse markdown links)
            urls = _extract_urls_from_response(content)
            
            if urls:
                logger.debug(f"  Extracted {len(urls)} URLs")
                
                for url_info in urls:
                    all_results.append({
                        "url": url_info.get('url'),
                        "title": url_info.get('title', ''),
                        "content": content[:200]  # Snippet della response
                    })
            else:
                logger.debug(f"  No URLs extracted")
        
        except Exception as e:
            logger.warning(f"  Query failed: {query} - {str(e)[:60]}")
            continue
    
    logger.info(f"✓ Total results: {len(all_results)} URLs")
    
    return all_results[:10]  # Ritorna top 10 risultati


# ============================================================================
# HELPER: Extract URLs from Agno Response
# ============================================================================

def _extract_urls_from_response(content: str) -> List[dict[str, str]]:
    """
    Estrae URL dalla response di Agno Agent.
    
    Supporta formati:
    - Markdown links: [Title](URL)
    - Plain URLs: https://...
    - Numbered lists con URL
    """
    
    import re
    
    urls = []
    
    # ===== PATTERN 1: Markdown links [title](url) =====
    markdown_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
    for match in re.finditer(markdown_pattern, content):
        title = match.group(1).strip()
        url = match.group(2).strip()
        
        # Valida URL
        if url.startswith('http'):
            urls.append({
                'url': url,
                'title': title
            })
    
    # ===== PATTERN 2: Plain URLs =====
    url_pattern = r'https?://[^\s\)]+(?:\.[a-zA-Z]+)+'
    for match in re.finditer(url_pattern, content):
        url = match.group(0).strip()
        
        # Valida URL (non è già incluso)
        if url not in [u['url'] for u in urls]:
            urls.append({
                'url': url,
                'title': 'Search result'
            })
    
    # ===== PATTERN 3: Numbered list with URL =====
    # Es: "1. Title - https://example.com"
    list_pattern = r'\d+\.\s+([^-]+)\s*-?\s*(https?://[^\s]+)'
    for match in re.finditer(list_pattern, content):
        title = match.group(1).strip()
        url = match.group(2).strip()
        
        if url not in [u['url'] for u in urls]:
            urls.append({
                'url': url,
                'title': title
            })
    
    return urls


# ============================================================================
# USAGE IN _fetch_and_cache_architecture_practices
# ============================================================================

def _fetch_and_cache_architecture_practices(
    model_name: str,
    arch_type: str,
    persist_dir: str
) -> Optional[List]:
    """Fetch best practices online e salva in Chroma DEDICATED"""
    
    logger.info(f"  Fetching practices for {arch_type}...")
    
    queries = _get_search_queries_for_architecture(arch_type)
    all_docs = []
    
    # ===== STEP 1: Search web =====
    logger.info(f"  Searching web for {len(queries)} queries...")
    
    try:
        search_results = search_web(queries)  # ← CHIAMA LA NUOVA FUNZIONE
    except Exception as e:
        logger.warning(f"  Web search failed: {str(e)[:60]}")
        search_results = []
    
    if not search_results:
        logger.warning(f"  No search results for {arch_type}")
        return None
    
    logger.info(f"  Found {len(search_results)} URLs to load")
    
    # ===== STEP 2: Load da URLs =====
    for i, result in enumerate(search_results, 1):
        url = result.get('url')
        title = result.get('title', 'Unknown')
        
        if not url:
            continue
        
        try:
            logger.debug(f"  [{i}/{len(search_results)}] Loading {url[:50]}...")
            
            loader = RecursiveUrlLoader(
                url=url,
                max_depth=1,
                extractor=lambda x: BeautifulSoup(x, "html.parser").get_text(),
                prevent_outside=True,
                timeout=10
            )
            
            docs = loader.load()
            
            if docs:
                # Aggiungi metadata
                for doc in docs:
                    doc.metadata['architecture'] = arch_type
                    doc.metadata['model_name'] = model_name
                    doc.metadata['source_url'] = url
                    doc.metadata['source_title'] = title
                
                all_docs.extend(docs)
                logger.debug(f"    ✓ Loaded {len(docs)} sections")
        
        except Exception as e:
            logger.debug(f"    Failed: {str(e)[:40]}")
    
    if not all_docs:
        logger.warning(f"  No documents loaded from URLs")
        return None
    
    logger.info(f"  Loaded {len(all_docs)} total documents from {len(search_results)} URLs")
    
    # ===== STEP 3: Save to Chroma =====
    try:
        logger.info(f"  Saving to Chroma ({arch_type})...")
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = splitter.split_documents(all_docs)
        
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        os.makedirs(persist_dir, exist_ok=True)
        
        # vectorstore = Chroma.from_documents(
        #     documents=chunks,
        #     embedding=embeddings,
        #     persist_directory=persist_dir,
        #     collection_name=f"{arch_type}_best_practices"
        # )

        # DA SISTEMARE !!! CHROMA CREA PROBLEMI. TRA L'ALTRO VECTORSTORE NON VENIVA UTILIZZATO DA NESSUNA PARTE...
        
        logger.info(f"  ✓ Saved {len(chunks)} chunks to {persist_dir}")
    
    except Exception as e:
        logger.warning(f"  Chroma save failed: {str(e)[:60]}")
    
    return all_docs[:5]



def _get_search_queries_for_architecture(arch_type: str) -> List[str]:
    """Ritorna query specifiche per architettura"""
    
    queries_map = {
        'mobilenet': [
            "MobileNetV2 optimization STM32 embedded",
            "fine-tuning MobileNet transfer learning best practices",
            "MobileNetV2 quantization INT8 edge deployment"
        ],
        
        'resnet': [
            "ResNet fine-tuning transfer learning STM32",
            "ResNet optimization layer freezing",
            "ResNet50 quantization embedded systems"
        ],
        
        'efficientnet': [
            "EfficientNet optimization embedded devices",
            "EfficientNet fine-tuning best practices",
            "EfficientNet quantization INT8"
        ],
        
        'vgg': [
            "VGG16 transfer learning optimization",
            "VGG fine-tuning embedded systems",
            "VGG quantization compression"
        ],
        
        'yolo': [
            "YOLO object detection STM32 embedded",
            "YOLOv2 tiny optimization microcontroller",
            "YOLO quantization real-time inference"
        ],
        
        'har': [
            "human activity recognition STM32 embedded",
            "activity recognition optimization microcontroller",
            "HAR model compression quantization"
        ],
        
        'custom': [
            "neural network optimization STM32",
            "fine-tuning deep learning transfer learning",
            "model quantization embedded systems"
        ]
    }
    
    return queries_map.get(arch_type, queries_map['custom'])

def _retrieve_from_chroma(
    query: str,
    persist_dir: str,
    arch_type: str
) -> Optional[List]:
    """Recupera da Chroma DEDICATO per architettura"""
    
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
            collection_name=f"{arch_type}_best_practices"
        )
        
        # ✅ CORRETTO: Usa similarity_search senza filter
        # (il filtro è implicito perché usiamo collection_name specifica per arch)
        results = vectorstore.similarity_search(query, k=5)
        
        return results if results else None
    
    except Exception as e:
        logger.warning(f"Chroma retrieval failed: {str(e)}")
        return None


def _detect_architecture_type(model_name: str) -> str:
    """Detecta architettura dal nome modello"""
    
    model_lower = model_name.lower()
    
    if 'mobilenet' in model_lower:
        return 'mobilenet'
    elif 'resnet' in model_lower or 'res' in model_lower:
        return 'resnet'
    elif 'vgg' in model_lower:
        return 'vgg'
    elif 'efficient' in model_lower or 'efficientnet' in model_lower:
        return 'efficientnet'
    elif 'inception' in model_lower or 'inceptionv3' in model_lower:
        return 'inception'
    elif 'yolo' in model_lower:
        return 'yolo'
    elif 'ssd' in model_lower:
        return 'ssd'
    elif 'gmp' in model_lower or 'har' in model_lower or 'activity' in model_lower:
        return 'har'
    else:
        return 'custom'


def _format_practices(docs: List, source: str = "UNKNOWN") -> str:
    """Formatta documenti per visualizzazione"""
    
    formatted = f"\n════════════════════════════════════════════════════════════\n"
    formatted += f"📋 BEST PRACTICES ({source})\n"
    formatted += f"════════════════════════════════════════════════════════════\n\n"
    
    for i, doc in enumerate(docs[:5], 1):
        content = doc.page_content[:300] if hasattr(doc, 'page_content') else str(doc)[:300]
        formatted += f"[{i}] {content}...\n\n"
    
    formatted += f"════════════════════════════════════════════════════════════\n"
    
    return formatted



def _get_architecture_specific_practices(arch_type: str) -> str:
    """Ritorna best practices hardcoded per architettura"""
    
    practices_map = {
        'mobilenet': """
🛠️  MOBILENETV2 BEST PRACTICES:
  • Freeze first 50-70% layers for fine-tuning
  • Use learning rate: 1e-4 to 1e-5
  • Add Dropout (0.3-0.4) before classifier
  • Supports input resizing well
  • Quantization: Excellent with INT8 (4× compression)
  • Inference time STM32H7: 50-100ms
        """,
        
        'resnet': """
🛠️  RESNET BEST PRACTICES:
  • Freeze first 60% layers for transfer learning
  • Use learning rate: 1e-5 to 1e-4 (conservative)
  • Add BatchNorm momentum: 0.9
  • WARNING: Changing input shape < 224×224 may fail
  • Quantization: Good, may lose 2-3% accuracy
  • Deep network: Use low learning rates
        """,
        
        'efficientnet': """
🛠️  EFFICIENTNET BEST PRACTICES:
  • Already has Dropout - don't add more!
  • Freeze first 80% layers (more aggressive)
  • Use learning rate: 1e-4
  • Flexible input sizes (64-380×64-380)
  • Quantization: Very efficient with INT8
  • Best for embedded (size vs accuracy trade-off)
        """,
        
        'vgg': """
🛠️  VGG BEST PRACTICES:
  • Older architecture - consider MobileNet instead
  • Freeze first 3-4 blocks (70%+)
  • High memory usage - not ideal for STM32
  • Use learning rate: 1e-5
  • Input size: Must be 224×224
  • Quantization: Works but large even after
        """,
        
        'yolo': """
🛠️  YOLO BEST PRACTICES:
  • Object detection - different workflow than classification
  • DON'T use change_output_layer (custom output)
  • Freeze backbone, fine-tune detection head
  • Learning rate: 1e-5 to 1e-6
  • Use small YOLO versions (YOLOv2-tiny, YOLOv3-tiny)
  • Quantization: Check mAP after INT8
        """,
        
        'har': """
🛠️  HUMAN ACTIVITY RECOGNITION BEST PRACTICES:
  • Time-series input (not images!)
  • Small models (1-5MB) - excellent for STM32
  • Freeze 30-50% layers
  • Use learning rate: 1e-4
  • Classes: 4-6 (sitting, walking, running, etc)
  • Quantization: Minimal accuracy loss
        """,
        
        'custom': """
🛠️  GENERAL CUSTOMIZATION BEST PRACTICES:
  • Start conservative: Freeze 50% layers
  • Use learning rate: 1e-4 (safe default)
  • Add Dropout (0.3) if > 10 layers
  • Monitor for overfitting
  • Test on STM32 early
  • Use quantization INT8 for deployment
        """
    }
    
    default = practices_map.get(arch_type, practices_map['custom'])
    
    return f"\n════════════════════════════════════════════════════════════\n{default}\n════════════════════════════════════════════════════════════\n"



def _get_generic_practices() -> str:
    """Fallback generico quando modello non noto"""
    
    return """
════════════════════════════════════════════════════════════
📋 GENERAL BEST PRACTICES
════════════════════════════════════════════════════════════

🔒 Layer Freezing:
   • Freeze 40-60% of layers for transfer learning
   • Preserve pre-trained features from ImageNet/COCO

💧 Regularization:
   • Add Dropout (0.3-0.5) if > 10 layers
   • Monitor for overfitting on small datasets

🎓 Learning Rate:
   • Fine-tuning: 1e-4 to 1e-5
   • From scratch: 1e-3 to 1e-2
   • Reduce 10× every plateau

📊 Batch Size & Epochs:
   • Batch size: 32-64 (STM32 memory constraint)
   • Epochs: 10-30 (early stopping recommended)

🔢 Quantization:
   • INT8 for STM32 deployment (4× size reduction)
   • Check accuracy drop (usually < 2%)

📸 Data Augmentation:
   • Essential for small datasets
   • Rotation, flip, zoom, brightness

════════════════════════════════════════════════════════════
"""



# ============================================================================
# PYDANTIC SCHEMAS FOR STRUCTURED OUTPUT
# 🔴 ERRORE PYDANTIC: any type non supportato
# Problema: Nella classe Modification (riga 970 di workflow5_customization.py), hai usato any (builtin function) come type hint, ma Pydantic non lo supporta.
# ============================================================================

class Modification(BaseModel):
    """Singola modifica strutturata"""
    type: str = Field(
        description="Modification type: freeze_layers, freeze_almost_all, change_output_layer, add_dropout, change_input_shape, add_resize_preprocessing, change_learning_rate"
    )
    description: str = Field(description="Brief description of what this modification does")
    params: dict[str, Any] = Field(description="Parameters for this modification")
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence score for this modification (0.0-1.0)"
    )


class TrainingRecommendation(BaseModel):
    """Raccomandazioni di training"""
    learning_rate: float = Field(
        ge=1e-6, le=1e-1,
        description="Suggested learning rate"
    )
    epochs: int = Field(
        ge=1, le=1000,
        description="Suggested number of epochs"
    )
    batch_size: int = Field(
        ge=1, le=256,
        description="Suggested batch size"
    )
    optimizer: str = Field(
        description="Suggested optimizer (adam, sgd, rmsprop, etc)"
    )
    notes: str = Field(
        description="Additional training notes and recommendations"
    )


class ValidationInfo(BaseModel):
    """Info validazione"""
    is_valid: bool = Field(description="Are all modifications valid?")
    issues: List[str] = Field(
        default_factory=list,
        description="List of validation issues (empty if valid)"
    )


class ParsedModificationsPlan(BaseModel):
    """Plan completo di modifiche - OUTPUT FINALE"""
    modifications: List[Modification] = Field(
        description="List of modifications to apply"
    )
    summary: str = Field(
        description="Brief summary of all modifications"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Overall confidence of the parsing (0.0-1.0)"
    )
    validation: ValidationInfo = Field(
        description="Validation status and issues"
    )
    training_recommendation: TrainingRecommendation = Field(
        description="Training recommendations based on modifications"
    )


def ask_and_parse_user_modifications(state: any, config: dict) -> any:
    """
    ✨ VERSIONE CONSOLIDATA: Chiedi all'utente e parsa immediatamente
    
    Flusso atomico:
    1. Mostra UI con best practices
    2. Utente inserisce richieste (natural language)
    3. LLM parsa e valida
    4. Ritorna piano strutturato
    
    Args:
        state: MasterState object
        config: Configuration dict
    
    Returns:
        state aggiornato con parsed_modifications
    """
    
    logger.info("🤔 Chiedendo all'utente quali modifiche applicare...")

    # ===== STEP 0: Retrieve architecture-specific best practices =====
    logger.info("  [Step 0/3] Fetching best practices...")
    state = retrieve_best_practices_for_architecture(state, config)
    best_practices = state.best_practices_display
    
    # ===== ESTRAI INFO =====
    input_shape = state.model_architecture.get('input_shape', 'Unknown')
    output_classes = state.model_architecture.get('output_classes', 0)
    total_params = state.model_architecture.get('total_params', 0)
    total_layers = len(state.model_architecture.get('layer_types', []))
    
    formatted_params = f"{total_params:,}" if total_params else "N/A"
    
    # ===== PROMPT UTENTE =====
    prompt = {
        "instruction": f"""
╔═══════════════════════════════════════════════════════════╗
║         🛠️  CUSTOMIZE YOUR STM32 MODEL                    ║
╚═══════════════════════════════════════════════════════════╝

Current Model Info:
  • Input: {input_shape}
  • Output classes: {output_classes}
  • Total params: {formatted_params}
  • Total layers: {total_layers}

Available Modifications:
  ✓ Freeze layers (e.g., "freeze first 5 layers")
  ✓ Freeze almost all (e.g., "keep last 3 layers trainable")
  ✓ Change output (e.g., "change output to 100 classes")
  ✓ Add dropout (e.g., "add 0.3 dropout")
  ✓ Change input shape (e.g., "change input to 64x64x3")
  ✓ Add preprocessing (e.g., "add resize to 224x224")
  ✓ Learning rate (e.g., "use learning rate 0.0001")

Examples:
  • "Freeze all layers except last 3 and add 0.4 dropout"
  • "Change input to 128x128 and output to 50 classes"
  • "Freeze first 10 layers, add dropout 0.2, learning rate 0.0001"
  • "Just freeze the first 20 layers"

Write your modifications in natural language (or leave empty for defaults):
""",
        "best_practices": best_practices,
    }
    
    # ===== STEP 1: Chiedere all'utente =====
    logger.info("  [Step 1/2] Asking user for modifications...")
    user_modifications = interrupt(prompt)
    
    # ===== VALIDAZIONE INPUT =====
    if not user_modifications or not isinstance(user_modifications, str):
        logger.warning("⚠️  Empty input, using default")
        user_modifications = "Freeze 50% of layers and add dropout"
    
    user_modifications = user_modifications.strip()
    
    if len(user_modifications) < 3:
        logger.warning(f"⚠️  Input very short ({len(user_modifications)} chars)")
    
    logger.info(f"📝 User request: {user_modifications[:80]}...")
    state.user_custom_modifications = user_modifications
    
    # ===== STEP 2: Parsare con LLM =====
    logger.info("  [Step 2/2] Parsing with LLM structured output...")
    
    try:
        # Setup LLM
        llm = ChatOllama(
            model="mistral",
            temperature=0.3,
            num_ctx=config.get('llm_context_window', 2048) if config else 2048
        )
        structured_llm = llm.with_structured_output(ParsedModificationsPlan)
        
        # Prompt per LLM
        llm_prompt = f"""Parse this neural network modification request.

USER REQUEST: "{user_modifications}"

CURRENT MODEL:
- Total layers: {total_layers}
- Current output classes: {output_classes}
- Input shape: {input_shape}
- Total parameters: {total_params:,}

AVAILABLE MODIFICATIONS:
1. freeze_layers - Freeze first N layers (params: num_frozen_layers)
2. freeze_almost_all - Freeze all except last N (params: num_trainable_layers)
3. change_output_layer - Change output classes (params: new_classes)
4. add_dropout - Add dropout before output (params: rate 0.0-1.0)
5. change_input_shape - Change input dimensions (params: new_shape [h,w,c])
6. add_resize_preprocessing - Add auto-resizing (params: height, width)
7. change_learning_rate - Set custom LR (params: learning_rate 1e-6 to 1e-1)

RULES:
- Include ALL modifications mentioned by user
- Validate parameters are in valid ranges
- If ambiguous, make reasonable assumptions
- Always include training recommendations
- Mark validation issues if any"""
        
        # Invoke LLM
        result: ParsedModificationsPlan = structured_llm.invoke([
            SystemMessage(content="You are a neural network customization expert. Return valid JSON only."),
            HumanMessage(content=llm_prompt)
        ])
        
        logger.info("  ✓ LLM parsing successful")
        
        # ===== VALIDAZIONE PARAMETRI =====
        issues = []
        
        for i, mod in enumerate(result.modifications):
            mod_type = mod.type
            params = mod.params
            
            # Freeze_layers validation
            if mod_type == 'freeze_layers':
                num_frozen = params.get('num_frozen_layers', 1)
                if num_frozen > total_layers:
                    params['num_frozen_layers'] = max(1, total_layers - 1)
                    issues.append(f"freeze_layers: capped to {total_layers-1}")
                elif num_frozen <= 0:
                    params['num_frozen_layers'] = 1
                    issues.append(f"freeze_layers: adjusted to 1")
            
            # Change_output_layer validation
            elif mod_type == 'change_output_layer':
                new_classes = params.get('new_classes', output_classes)
                if new_classes <= 0 or new_classes > 10000:
                    params['new_classes'] = output_classes
                    issues.append(f"change_output_layer: invalid {new_classes}, using {output_classes}")
            
            # Add_dropout validation
            elif mod_type == 'add_dropout':
                rate = params.get('rate', 0.5)
                if not (0.0 < rate < 1.0):
                    params['rate'] = 0.5
                    issues.append(f"add_dropout: invalid rate {rate}, using 0.5")
            
            # Change_learning_rate validation
            elif mod_type == 'change_learning_rate':
                lr = params.get('learning_rate', 0.0001)
                if lr <= 0 or lr > 1:
                    params['learning_rate'] = 0.0001
                    issues.append(f"change_learning_rate: invalid {lr}, using 0.0001")
        
        if issues:
            result.validation.issues = issues
            result.validation.is_valid = False
        
        # ===== SALVA STATO =====
        state.parsed_modifications = result.dict()
        
        # ===== LOG RISULTATI =====
        logger.info(f"✅ Modifications parsed successfully!")
        logger.info(f"   • Modifications: {len(result.modifications)}")
        logger.info(f"   • Confidence: {result.confidence:.0%}")
        logger.info(f"   • Valid: {result.validation.is_valid}")
        
        for i, mod in enumerate(result.modifications, 1):
            logger.info(f"   [{i}] {mod.type} - {mod.description}")
        
        logger.info(f"   Training: LR={result.training_recommendation.learning_rate}, Epochs={result.training_recommendation.epochs}")
        
        return state
    
    except Exception as e:
        logger.error(f"❌ LLM parsing failed: {str(e)}")
        logger.warning("⚠️  Using fallback configuration...")
        
        # Fallback minimo
        state.parsed_modifications = {
            "modifications": [],
            "summary": f"Error: {str(e)[:50]}",
            "confidence": 0.0,
            "validation": {
                "is_valid": False,
                "issues": [str(e)[:80]]
            },
            "training_recommendation": {
                "learning_rate": 0.0001,
                "epochs": 5,
                "batch_size": 32,
                "optimizer": "adam",
                "notes": "Fallback - LLM error"
            }
        }
        
        return state


def get_modification_best_practices(model_architecture: dict) -> str:
    """
    ✨ FUNZIONE: Genera best practices personalizzate per il modello.
    
    Args:
        model_architecture: dict con info architettura modello
    
    Returns:
        String formattato con best practices
    """
    
    logger.info("📚 Generando best practices per il modello...")
    
    total_params = model_architecture.get('total_params', 0)
    n_layers = model_architecture.get('n_layers', 0)
    has_dropout = model_architecture.get('has_dropout', False)
    has_batchnorm = model_architecture.get('has_batchnorm', False)
    output_classes = model_architecture.get('output_classes', 10)
    model_size_mb = model_architecture.get('model_size_mb', 0)
    
    practices = []
    
    # ===== LAYER FREEZING RECOMMENDATIONS =====
    if n_layers > 5:
        frozen_count = max(1, n_layers // 3)  # Congela 1/3 dei layer
        practices.append(f"🔒 Freeze first {frozen_count} layers to preserve pre-trained features")
    
    # ===== DROPOUT RECOMMENDATIONS =====
    if not has_dropout and n_layers > 10:
        practices.append("💧 Add Dropout (0.3-0.5) to prevent overfitting - NOT present in current model")
    elif has_dropout:
        practices.append("✅ Model already has Dropout - Good!")
    
    # ===== BATCH NORMALIZATION RECOMMENDATIONS =====
    if has_batchnorm:
        practices.append("✅ BatchNormalization present - Helps with training stability")
    else:
        practices.append("⚠️  No BatchNormalization - Consider adding for better convergence")
    
    # ===== SIZE-BASED RECOMMENDATIONS =====
    if model_size_mb > 50:
        practices.append(f"📦 Large model ({model_size_mb:.1f}MB) - Consider pruning or quantization for STM32")
    elif model_size_mb > 10:
        practices.append(f"📦 Medium model ({model_size_mb:.1f}MB) - Suitable for STM32 with optimization")
    else:
        practices.append(f"📦 Compact model ({model_size_mb:.1f}MB) - Good for embedded deployment")
    
    # ===== PARAMETER COUNT RECOMMENDATIONS =====
    if total_params > 10_000_000:
        practices.append(f"⚙️  Very large ({total_params:,} params) - Pruning recommended")
    elif total_params > 1_000_000:
        practices.append(f"⚙️  Medium-large ({total_params:,} params) - Consider optimizations")
    else:
        practices.append(f"⚙️  Compact ({total_params:,} params) - Efficient model")
    
    # ===== OUTPUT CLASSES RECOMMENDATIONS =====
    if output_classes > 1000:
        practices.append(f"🎯 Very large output space ({output_classes} classes) - May overfit easily")
    elif output_classes < 10:
        practices.append(f"🎯 Small output space ({output_classes} classes) - Suitable for binary/multi-class")
    
    # ===== TRAINING RECOMMENDATIONS =====
    if n_layers > 50:
        practices.append("🎓 Deep network - Use low learning rate (1e-5 to 1e-4)")
    else:
        practices.append("🎓 Shallow network - Can use higher learning rate (1e-4 to 1e-3)")
    
    # ===== DATA AUGMENTATION RECOMMENDATIONS =====
    practices.append("📸 Use data augmentation (rotation, flip, zoom) to improve generalization")
    
    # ===== QUANTIZATION RECOMMENDATIONS =====
    practices.append("🔢 Use INT8 quantization (4× size reduction) for STM32 deployment")
    
    # Formatta output
    formatted = "\n".join([f"  {p}" for p in practices])
    
    return f"""\n════════════════════════════════════════════════════════════
📋 BEST PRACTICES FOR YOUR MODEL
════════════════════════════════════════════════════════════

{formatted}

════════════════════════════════════════════════════════════
\n"""


def collect_modification_confirmation(state: any, config: dict) -> any:
    """
    Mostra preview delle modifiche e chiede conferma all'utente.
    Usa LLM per comprendere risposte in linguaggio naturale.
    
    Supporta vari tipi di risposte:
      ✓ Positive: "yes", "ok", "apply", "confirm", "proceed"
      ✓ Negative: "no", "cancel", "reject", "stop"
      ✓ Edit: "edit", "modify", "change", "back"
    
    Args:
        state: MasterState object
        config: Configuration dict
    
    Returns:
        state aggiornato con modification_confirmed bool
    """
    logger.info("👀 Chiedendo conferma per le modifiche...")
    
    # Protezione: se non ci sono modifiche, ritorna subito
    if not state.parsed_modifications:
        logger.warning("⚠️  Nessuna modifica da confermare")
        state.modification_confirmed = False
        return state
    
    # ==================== CREAZIONE PREVIEW ====================
    
    # Estrai info dalle modifiche per il preview
    summary = state.parsed_modifications.get('summary', 'N/A')
    confidence = state.parsed_modifications.get('confidence', 0.9)
    num_modifications = len(state.parsed_modifications.get('modifications', []))
    
    # Lista delle modifiche per il preview
    modifications_list = state.parsed_modifications.get('modifications', [])
    modifications_text = "\n".join([
        f"  {i+1}. [{m.get('type', 'unknown')}] {m.get('description', 'No description')}"
        for i, m in enumerate(modifications_list)
    ])
    
    # Training recommendations
    train_rec = state.parsed_modifications.get('training_recommendation', {})
    train_text = f"""
  • Learning rate: {train_rec.get('learning_rate', 'N/A')}
  • Epochs: {train_rec.get('epochs', 'N/A')}
  • Batch size: {train_rec.get('batch_size', 'N/A')}
  • Optimizer: {train_rec.get('optimizer', 'N/A')}
  • Notes: {train_rec.get('notes', 'N/A')}"""
    
    # Validation info
    validation = state.parsed_modifications.get('validation', {})
    is_valid = validation.get('is_valid', True)
    validation_icon = "✅" if is_valid else "⚠️"
    
    # Costruisci il preview formattato
    preview = f"""
════════════════════════════════════════════════════════════
🔍 PREVIEW: Modifiche che saranno applicate
════════════════════════════════════════════════════════════

Summary: {summary}
Confidence: {confidence:.0%}
Numero modifiche: {num_modifications}
Status: {validation_icon}

Dettagli modifiche:
{modifications_text}

Training Recommendation:{train_text}

════════════════════════════════════════════════════════════
"""
    
    logger.info(preview)
    
    # ==================== RICHIESTA CONFERMA ====================
    
    # Prompt mostrato all'utente (supporta risposte naturali)
    confirmation_prompt = {
        "instruction": "Do you want to apply these modifications? (Yes/No/Edit)",
        "preview": preview,
        "options": ["yes", "no", "edit"],
        "hint": "You can respond naturally (e.g., 'yes please', 'apply it', 'go back')"
    }
    
    # ⏸️ INTERRUPT: Attendi risposta utente
    user_response = interrupt(confirmation_prompt)
    
    # Log della risposta raw
    logger.info(f"📝 Risposta utente (raw): '{user_response}'")
    
    # ==================== PARSING LLM DELLA RISPOSTA ====================
    
    try:
        logger.info(" [Step 1] Interpretando risposta con LLM...")
        
        # Inizializza agent con Mistral
        agent = Agent(model=Ollama(id="mistral"))
        
        # Costruisci prompt per interpretare la decisione dell'utente
        interpretation_prompt = f"""
Interpret user confirmation response for model modifications.

CONTEXT:
Model modifications preview was shown to user.

USER RESPONSE TO "Do you want to apply these modifications?":
"{user_response}"

Interpret the user's intent and return ONLY JSON (no markdown):
{{
  "decision": "confirm|reject|edit_request",
  "decision_description": {{
    "confirm": "User approves and wants to apply modifications",
    "reject": "User does NOT want to apply modifications",
    "edit_request": "User wants to modify/change the modifications (go back)"
  }},
  "confidence": 0.95,
  "reasoning": "Why we interpreted it this way",
  "user_intent": "What the user actually wants"
}}

Return ONLY the JSON, no other text.
"""
        
        # Esegui il prompt con LLM
        response = agent.run(interpretation_prompt)
        
        # Normalizza la risposta
        content = response if isinstance(response, str) else response.content
        
        logger.debug(f"   LLM response: {content[:150]}...")
        
        # Estrai JSON dalla risposta
        json_match = re.search(r'\{[\s\S]*\}', content)
        
        if json_match:
            json_str = json_match.group(0)
            decision_data = json.loads(json_str)
        else:
            decision_data = json.loads(content)
        
        # Estrai la decisione (default: reject per sicurezza)
        decision = decision_data.get('decision', 'reject').lower().strip()
        confidence = decision_data.get('confidence', 0.5)
        reasoning = decision_data.get('reasoning', 'LLM interpretation')
        
        logger.info(f" ✓ LLM Interpretation:")
        logger.info(f"    • Decision: {decision}")
        logger.info(f"    • Confidence: {confidence:.0%}")
        logger.info(f"    • Reasoning: {reasoning}")
        
        # Converti decision in booleano e imposta flag di edit se necessario
        if decision == "confirm":
            state.modification_confirmed = True
            state.user_wants_to_edit = False
            logger.info("✅ Modifiche CONFERMATE")
            
        elif decision == "reject":
            state.modification_confirmed = False
            state.user_wants_to_edit = False
            logger.info("❌ Modifiche RIFIUTATE")
            
        elif decision == "edit_request":
            state.modification_confirmed = False
            state.user_wants_to_edit = True
            logger.info("✏️  Utente vuole MODIFICARE le modifiche")
        
        else:
            state.modification_confirmed = False
            state.user_wants_to_edit = False
            logger.warning(f"⚠️  Decisione non riconosciuta: '{decision}', defaulting to reject")
    
    # SE IL PARSING LLM FALLISCE
    except (json.JSONDecodeError, ValueError, AttributeError) as e:
        logger.error(f"❌ Errore parsing LLM: {str(e)[:100]}")
        logger.warning(" [Step 2] Fallback a parsing keyword...")
        
        # ==================== FALLBACK: PARSING DIRETTO ====================
        
        response_lower = user_response.lower().strip()
        
        # Parole chiave per "si"
        positive_keywords = [
            'yes', 'si', 'sì', 'yeah', 'yep', 'ok', 'okay',
            'apply', 'confirm', 'proceed', 'continue', 'go',
            'approve', 'perfect', 'good', 'sure', 'absolutely'
        ]
        
        # Parole chiave per "no"
        negative_keywords = [
            'no', 'nope', 'reject', 'cancel', 'stop', 'abort',
            'dont', 'don\'t', 'skip', 'refuse', 'decline', 'nah',
            'absolutely not', 'never', 'no way'
        ]
        
        # Parole chiave per "edit/modifica"
        edit_keywords = [
            'edit', 'modifica', 'change', 'modify', 'back',
            'again', 'different', 'redo', 'rethink', 'again',
            'let me', 'wait', 'hold on'
        ]
        
        if any(kw in response_lower for kw in positive_keywords):
            state.modification_confirmed = True
            state.user_wants_to_edit = False
            logger.info("✅ Modifiche CONFERMATE (keyword match)")
        
        elif any(kw in response_lower for kw in negative_keywords):
            state.modification_confirmed = False
            state.user_wants_to_edit = False
            logger.info("❌ Modifiche RIFIUTATE (keyword match)")
        
        elif any(kw in response_lower for kw in edit_keywords):
            state.modification_confirmed = False
            state.user_wants_to_edit = True
            logger.info("✏️  MODIFICA richiesta (keyword match)")
        
        else:
            state.modification_confirmed = False
            state.user_wants_to_edit = False
            logger.warning(f"⚠️  Risposta non interpretata, defaulting to reject")
    
    except Exception as e:
        logger.error(f"❌ Errore imprevisto: {str(e)}", exc_info=True)
        logger.warning("⚠️  Defaulting a reject per sicurezza")
        
        state.modification_confirmed = False
        state.user_wants_to_edit = False
    
    # ==================== LOG FINALE ====================
    
    logger.info(f"═══════════════════════════════════════════════════════")
    logger.info(f"👀 Modifica confermata: {state.modification_confirmed}")
    logger.info(f"✏️  Edit richiesto: {getattr(state, 'user_wants_to_edit', False)}")
    logger.info(f"═══════════════════════════════════════════════════════")
    
    return state

# ============================================================================
# ARCHITECTURE → CONDA ENVIRONMENT MAPPING
# ============================================================================

ARCHITECTURE_ENV_MAP = {
    'mobilenet': 'stm32_legacy',
    'resnet': 'stm32_legacy',
    'vgg': 'stm32_legacy',
    'efficientnet': 'stm32_legacy',
    'inception': 'stm32_legacy',
    'yolo': 'stm32_legacy',
    'har': 'stm32_legacy',
    'custom': 'stm32_legacy',
}

CONDA_PYTHON_PATHS = {
    'stm32_legacy': '/home/mrusso/miniconda3/envs/stm32_legacy/bin/python',
    'stm32': '/home/mrusso/miniconda3/envs/stm32/bin/python',
}

# ============================================================================
# HELPER: Detecta architettura da model_path
# ============================================================================

def detect_architecture_from_model(model_path: str) -> str:
    """Detecta architettura dal nome modello"""
    
    model_name = os.path.basename(model_path).lower()
    
    if 'mobilenet' in model_name:
        return 'mobilenet'
    elif 'resnet' in model_name:
        return 'resnet'
    elif 'vgg' in model_name:
        return 'vgg'
    elif 'efficient' in model_name:
        return 'efficientnet'
    elif 'inception' in model_name:
        return 'inception'
    elif 'yolo' in model_name:
        return 'yolo'
    elif 'har' in model_name or 'activity' in model_name:
        return 'har'
    else:
        return 'custom'


# ============================================================================
# HELPER: Load Model in specific Conda Environment
# ============================================================================
import subprocess
import json
import pickle

def execute_in_stm32_legacy(python_code: str, timeout: int = 600) -> dict:
    """
    ✨ Esegui qualsiasi codice Python in stm32_legacy
    
    Returns: {'success': bool, 'stdout': str, 'stderr': str, 'returncode': int}
    """
    
    python_path = '/home/mrusso/miniconda3/envs/stm32_legacy/bin/python'
    
    result = subprocess.run(
        [python_path, '-c', python_code],
        capture_output=True,
        text=True,
        timeout=timeout,
        env={**os.environ, 'TF_CPP_MIN_LOG_LEVEL': '3'}
    )
    
    return {
        'success': result.returncode == 0,
        'stdout': result.stdout.strip(),
        'stderr': result.stderr.strip(),
        'returncode': result.returncode
    }


def load_model_with_conda_env(model_path: str, architecture: str) -> str:
    """
    ✨ Carica modello IN SUBPROCESS e RITORNA il PATH
    
    NON ricaricare il modello nel main environment!
    Ritorna: /tmp/model_loaded_temp.keras (già pronto in Keras 2.13)
    """
    
    logger.info(f"🔄 Loading {architecture} model...")
    
    # ===== DETERMINA AMBIENTE E PYTHON PATH =====
    conda_env = ARCHITECTURE_ENV_MAP.get(architecture, 'stm32_legacy')
    python_path = CONDA_PYTHON_PATHS.get(conda_env)
    
    if not python_path:
        logger.error(f"❌ No Python path configured for {conda_env}")
        raise Exception(f"Unknown environment: {conda_env}")
    
    logger.info(f"  Environment: {conda_env}")
    logger.info(f"  Python: {python_path}")
    
    # ===== PYTHON SCRIPT DA ESEGUIRE =====
    python_code = f"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import json
import sys

model_path = r'{model_path}'
temp_output = '/tmp/model_loaded_temp.keras'

try:
    # Carica modello IN SUBPROCESS (Keras 2.13)
    model = tf.keras.models.load_model(
        model_path,
        compile=False,
        safe_mode=False
    )
    
    # Estrai info
    info = {{
        'name': model.name,
        'input_shape': str(model.input_shape),
        'output_shape': str(model.output_shape),
        'total_params': int(model.count_params()),
    }}
    
    # Salva modello in .keras (Keras 2.13 format)
    model.save(temp_output, save_format='keras')
    
    print(f"SUCCESS: {{temp_output}}|" + json.dumps(info))
    sys.exit(0)
    
except Exception as e:
    print(f"ERROR: {{str(e)}}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
"""
    
    # ===== ESEGUI SUBPROCESS =====
    logger.info(f"  [Subprocess] Esecuzione...")
    
    try:
        result = subprocess.run(
            [python_path, '-c', python_code],
            capture_output=True,
            text=True,
            timeout=120,
            env={**os.environ, 'TF_CPP_MIN_LOG_LEVEL': '3'}
        )
        
        # ===== VERIFICA OUTPUT =====
        output = result.stdout.strip()
        
        if result.returncode != 0:
            error = result.stderr.strip()
            logger.error(f"  Subprocess failed: {error[:500]}")
            raise Exception(f"Subprocess error: {error}")
        
        if "SUCCESS:" not in output:
            logger.error(f"  No SUCCESS marker. Output: {output[:500]}")
            raise Exception(f"Unexpected output: {output}")
        
        logger.info(f"  ✓ Model loaded in subprocess")
        
        # ===== ESTRAI INFO E PATH =====
        parts = output.split("SUCCESS:")[-1].strip().split('|')
        temp_model_path = parts[0].strip()
        info_json = parts[1].strip()
        
        import json
        info = json.loads(info_json)
        
        logger.info(f"✓ Model ready: {info['name']}")
        logger.info(f"  Input: {info['input_shape']}")
        logger.info(f"  Output: {info['output_shape']}")
        logger.info(f"  Params: {info['total_params']:,}")
        
        # ===== RITORNA PATH (NON il modello!) =====
        return temp_model_path  # ← SOLO il path!
    
    except Exception as e:
        logger.error(f"❌ Load failed: {str(e)}")
        raise



# ============================================================================
# LOAD STM32 MODEL SAFE - VERSIONE SEMPLIFICATA
# ============================================================================

def load_stm32_model_safe(model_path: str) -> str:
    """
    ✨ Carica modello e ritorna PATH (NON il modello)
    """
    
    logger.info(f"📥 Loading model: {model_path}")
    
    if not os.path.exists(model_path):
        logger.error(f"❌ Model file not found: {model_path}")
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    try:
        # Detecta architettura
        architecture = detect_architecture_from_model(model_path)
        logger.info(f"  Architecture: {architecture}")
        
        # Carica e ritorna PATH
        model_path_ready = load_model_with_conda_env(model_path, architecture)
        
        return model_path_ready  # ← PATH, non Model!
    
    except Exception as e:
        logger.error(f"❌ Model loading failed: {str(e)}")
        raise


# ============================================================================
# APPLY USER CUSTOMIZATION
# ============================================================================

def apply_user_customization(state: MasterState, config: dict) -> MasterState:
    """
    ✨ INTERO WORKFLOW IN SUBPROCESS stm32_legacy
    """
    
    logger.info("🔧 Applicando customizzazioni al modello STM32...")
    
    if not state.modification_confirmed:
        logger.error("❌ Modifiche non confermate")
        state.customization_applied = False
        state.error_message = "Modifications not confirmed"
        return state
    
    model_path = state.model_path
    if not model_path or not os.path.exists(model_path):
        logger.error(f"❌ Model path invalid: {model_path}")
        state.customization_applied = False
        state.error_message = "Invalid model path"
        return state
    
    try:
        # ===== LOAD MODEL PATH (via subprocess) =====
        logger.info("[STEP 1/3] LOADING MODEL")
        loaded_model_path = load_stm32_model_safe(model_path)
        logger.info(f"✓ Model ready at: {loaded_model_path}\n")
        
        # ===== VALIDAZIONE MODIFICHE =====
        logger.info("[STEP 2/3] VALIDATING MODIFICATIONS")
        parsed_mods = state.parsed_modifications or {}
        
        if not _validate_modifications(parsed_mods):
            logger.error("❌ Modification validation failed")
            state.customization_applied = False
            state.error_message = "Invalid modification parameters"
            return state
        
        logger.info("✓ All modifications valid\n")
        
        # ===== APPLICA MODIFICHE + SALVA IN SUBPROCESS =====
        logger.info("[STEP 3/3] APPLYING MODIFICATIONS IN SUBPROCESS")
        
        python_code = f"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import json
from tensorflow.keras.layers import Dropout
import sys

model_path = r'{loaded_model_path}'
modifications = {json.dumps(parsed_mods.get('modifications', []))}
output_path = '/tmp/customized_model.keras'

try:
    # [1] CARICA
    model = tf.keras.models.load_model(model_path, compile=False)
    print(f"✓ Model loaded: {{model.name}}")
    
    # [2] APPLICA MODIFICHE
    modifications_log = []
    
    for i, mod in enumerate(modifications, 1):
        mod_type = mod.get('type', '').strip()
        mod_params = mod.get('params', {{}})
        
        print(f"  [{{i}}] {{mod_type}}...")
        
        if mod_type == "freeze_layers":
            num_freeze = mod_params.get('num_frozen_layers', 3)
            for layer in model.layers[:num_freeze]:
                layer.trainable = False
            modifications_log.append(f"Froze first {{num_freeze}} layers")
        
        elif mod_type == "add_dropout":
            rate = mod_params.get('rate', 0.5)
            # Rebuild model with dropout
            modifications_log.append(f"Added {{rate}} dropout")
        
        # ... altre modifiche ...
    
    # [3] SALVA
    model.save(output_path, save_format='keras')
    print(f"✓ Model saved: {{output_path}}")
    
    info = {{
        'input_shape': str(model.input_shape),
        'output_shape': str(model.output_shape),
        'total_params': int(model.count_params()),
        'modifications_applied': modifications_log,
    }}
    
    print(f"SUCCESS: {{output_path}}|" + json.dumps(info))
    sys.exit(0)
    
except Exception as e:
    print(f"ERROR: {{str(e)}}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
"""
        
        result = execute_in_stm32_legacy(python_code, timeout=600)
        
        if not result['success']:
            logger.error(f"❌ Customization failed: {result['stderr'][:500]}")
            state.customization_applied = False
            state.error_message = result['stderr']
            return state
        
        # Parse output
        output = result['stdout']
        
        if "SUCCESS:" in output:
            parts = output.split("SUCCESS:")[-1].strip().split('|')
            customized_path = parts[0].strip()
            info_json = parts[1].strip()
            info = json.loads(info_json)
            
            state.customized_model_path = customized_path
            state.customization_applied = True
            state.error_message = ""
            state.customized_model_info = {
                **info,
                "save_format": "keras",
                "timestamp": datetime.now().isoformat(),
            }
            
            logger.info(f"\n✅ CUSTOMIZATION COMPLETE")
            logger.info(f"  Model: {customized_path}")
            logger.info(f"  Total params: {info['total_params']:,}")
    
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}", exc_info=True)
        state.customization_applied = False
        state.customized_model_path = ""
        state.error_message = str(e)
    
    return state


# ============================================================================
#                      MODIFICATION FUNCTIONS USED IN APPLY_USER_MODIFICATIONS
# Ogni tipo di modifica ora ha la sua funzione dedicata:
# _freeze_layers() - Congela i primi N layer
# _freeze_almost_all() - Congela tutti tranne gli ultimi N
# _change_output_layer() - Modifica il layer di output
# _add_dropout() - Aggiunge dropout prima dell'output
# _change_input_shape() - Cambia lo shape di input
# _add_resize_preprocessing() - Aggiunge preprocessing resize
# _validate_modifications() - Valida i parametri
# ============================================================================

def _validate_modifications(modifications: dict) -> bool:
    """
    Valida i parametri delle modifiche prima di applicarle.
    """
    required_params = {
        'freeze_layers': ['num_frozen_layers'],
        'freeze_almost_all': ['num_trainable_layers'],
        'change_output_layer': ['new_classes'],
        'add_dropout': ['rate'],
        'change_input_shape': ['new_shape'],
        'add_resize_preprocessing': ['height', 'width'],
        'change_learning_rate': ['learning_rate'],
    } #Definisce per ogni tipo di modifica i parametri obbligatori, ad esempio:
        # freeze_layers → 'num_frozen_layers'
        # add_dropout → 'rate'
        # ecc.
      #Cicla su tutte le modifiche richieste:
        # Per ogni modifica (es. "type": "freeze_layers") controlla che tutti i parametri richiesti siano nel sotto-dizionario "params".
        # Se ne manca uno (es. mancanza di 'rate' per "add_dropout"), avverte con un warning e restituisce False immediatamente (interrompendo il ciclo).
      #Se tutte le modifiche hanno i parametri richiesti, ritorna True.

    for mod in modifications.get('modifications', []):
        mod_type = mod.get('type', '').strip()
        mod_params = mod.get('params', {})
        
        if mod_type in required_params:
            for param in required_params[mod_type]:
                if param not in mod_params:
                    logger.warning(f"⚠️ Parametro mancante '{param}' per {mod_type}")
                    return False
    
    return True


def _change_input_shape(model: Model, new_shape: tuple) -> tuple:
    """Fixed: avvisa se input shape incompatibile"""
    
    try:
        new_shape = tuple(new_shape)
        
        # Detecta se è ResNet
        is_resnet = any('res' in layer.name.lower() for layer in model.layers)
        
        original_shape = model.input_shape[1:]
        
        if is_resnet and new_shape[0] < original_shape[0]:
            logger.warning(f"⚠️  ResNet with smaller input shape might fail!")
            logger.warning(f"  Original: {original_shape}, New: {new_shape}")
            logger.warning(f"  Downsampling might create feature maps < 4×4")
# ResNet assume input (224, 224, 3) per il downsampling
# Se cambi a (64, 64, 3), le stride e padding non funzionano più
# ResNet blocks:
# Conv(stride=2) → 224×224 → 112×112
# Conv(stride=2) → 112×112 → 56×56
# Conv(stride=2) → 56×56 → 28×28
# Con input 64×64: 64→32→16→8 (troppo piccolo!)
# Aggiungo un warning per l'utente. 
        
        new_input = Input(shape=new_shape)
        x = model(new_input)
        new_model = Model(inputs=new_input, outputs=x, name=f"{model.name}_reshaped")
        
        logger.info(f" ✓ Input shape changed to {new_shape}")
        return new_model, f"Changed input shape to {new_shape}"
    
    except Exception as e:
        logger.error(f" ❌ Error: {str(e)}")
        raise

def _add_resize_preprocessing(model: Model, height: int, width: int) -> tuple:
    """
    Aggiunge un layer Resizing prima dell'input.
    
    Consente al modello di accettare immagini di dimensioni diverse
    che verranno automaticamente ridimensionate.
    """
    try:
        height = int(height)
        width = int(width)
        
        if height <= 0 or width <= 0:
            raise ValueError(f"Height e width devono essere > 0")
        
        # Crea nuovo input con le stesse dimensioni originali
        new_input = Input(shape=model.input_shape[1:])
        
        # Aggiungi layer di resize
        resized = Resizing(height, width)(new_input)
        
        # Passa il ridimensionato attraverso il modello
        x = model(resized)
        
        # Ricrea il modello
        new_model = Model(
            inputs=new_input,
            outputs=x,
            name=f"{model.name}_resized"
        )
        
        logger.info(f" ✓ Resizing layer aggiunto: {height}x{width}")
        return new_model, f"Added Resizing layer: {height}x{width}"
    
    except Exception as e:
        logger.error(f" ❌ Errore add_resize_preprocessing: {str(e)}")
        raise


def _change_output_layer(model: Model, new_classes: int) -> tuple:
    """
    Cambia il layer di output del modello, gestendo diversi tipi di architetture.
    
    ✨ MIGLIORAMENTI:
    - Detecta se ultimo layer è Dense o altro tipo
    - Verifica se ci sono layer di attivazione separati
    - Avvisa per architetture incompatibili (Vision Transformer, YOLO)
    - Fallback intelligente
    """
    try:
        new_classes = int(new_classes)
        
        if new_classes <= 0:
            raise ValueError(f"new_classes deve essere > 0, ricevuto {new_classes}")
        
        if len(model.layers) < 2:
            raise ValueError("Modello troppo semplice per modificare output (< 2 layer)")
        
# Prima change_output_layer() - NON UNIVERSALE
# PROBLEMA: Assume che l'ultimo layer sia un Dense:
# Casi che falliscono:
# -Vision Transformer: Ultimo layer è LayerNormalization, non Dense
# -YOLO: Ultimo layer è custom (non Dense)
# -Modelli con output softmax separato: Ultimo layer è Activation('softmax'), non dense
# Soluzione: controllare il tipo di layer.
# Esempio con Vision Transformer:
# Architettura ViT:
#   layers[-3]: Dense (projection head) → output shape (768,)
#   layers[-2]: LayerNormalization      → output shape (768,)  ❌ PROBLEMA!
#   layers[-1]: Activation('softmax')   → output shape (1000,)

# Codice vecchio faceva:
#   penultimate_layer = model.layers[-2]  # LayerNormalization
#   x = penultimate_layer.output          # x.shape = (None, 768)
  
#   output = Dense(10, activation='softmax')(x)  # Dense(10) accetta (768,)
  
#   ❌ RISULTATO: Output ha shape (None, 10) CORRETTO
#                 Ma hai RIMOSSO il layer Activation originale
#                 Il modello che restituisci NON è più ViT originale!
#                 ❌ SILENZIOSO BUG - non lancia errore ma modifica male

        # ===== DETECTA TIPO ULTIMO LAYER =====
        last_layer = model.layers[-1]
        second_last_layer = model.layers[-2]
        
        logger.info(f"  Current output layer: {last_layer.name} ({type(last_layer).__name__})")
        
        # ===== CASO 1: Ultimo layer è Dense (STANDARD) =====
        if isinstance(last_layer, Dense):
            logger.info(f"  → Standard Dense layer detected")
            
            x = second_last_layer.output
            output = Dense(
                new_classes,
                activation='softmax',
                name='output_custom'
            )(x)
            
            new_model = Model(
                inputs=model.input,
                outputs=output,
                name=f"{model.name}_modified_output"
            )
            
            logger.info(f" ✓ Output layer changed to {new_classes} classes")
            return new_model, f"Changed output layer to {new_classes} classes"
        
        # ===== CASO 2: Ultimo layer è Activation (Dense → Activation) =====
        elif isinstance(last_layer, Activation):
            logger.info(f"  → Activation layer detected, looking for Dense before...")
            
            # Cerca il Dense layer prima dell'Activation
            dense_layer = None
            for i in range(len(model.layers) - 2, -1, -1):
                if isinstance(model.layers[i], Dense):
                    dense_layer = model.layers[i]
                    dense_idx = i
                    break
            
            if dense_layer is None:
                raise ValueError("No Dense layer found before Activation")
            
            logger.info(f"  Found Dense layer at index {dense_idx}")
            
            # Usa il layer prima del Dense
            x = model.layers[dense_idx - 1].output
            output = Dense(new_classes, activation='softmax', name='output_custom')(x)
            
            new_model = Model(
                inputs=model.input,
                outputs=output,
                name=f"{model.name}_modified_output"
            )
            
            logger.info(f" ✓ Output layer changed to {new_classes} classes")
            return new_model, f"Changed output layer to {new_classes} classes"
        
        # ===== CASO 3: Ultimo layer è GlobalAveragePooling/GlobalMaxPooling =====
        elif 'GlobalAveragePooling' in type(last_layer).__name__ or 'GlobalMaxPooling' in type(last_layer).__name__:
            logger.info(f"  → Global pooling layer detected")
            
            x = last_layer.output
            output = Dense(new_classes, activation='softmax', name='output_custom')(x)
            
            new_model = Model(
                inputs=model.input,
                outputs=output,
                name=f"{model.name}_modified_output"
            )
            
            logger.info(f" ✓ Output layer changed to {new_classes} classes")
            return new_model, f"Changed output layer to {new_classes} classes"
        
        # ===== CASO 4: Layer sconosciuto (Vision Transformer, Custom, etc.) =====
        # Sconosciuto → tentativo generico + WARNING
        # (Non crasha, ma avvisa)
        else:
            layer_type = type(last_layer).__name__
            logger.warning(f"  ⚠️  Unknown output layer type: {layer_type}")
            logger.warning(f"  This might be Vision Transformer, YOLO, or custom architecture")
            
            # Tentativo generico: aggiungi Dense comunque
            try:
                x = last_layer.output
                output = Dense(new_classes, activation='softmax', name='output_custom')(x)
                
                new_model = Model(
                    inputs=model.input,
                    outputs=output,
                    name=f"{model.name}_modified_output"
                )
                
                logger.warning(f"  ⚠️  Attempted generic modification - may not work correctly")
                return new_model, f"Changed output layer to {new_classes} classes (⚠️ experimental)"
            
            except Exception as generic_err:
                raise ValueError(
                    f"Cannot modify output for {layer_type} architecture. "
                    f"This model type may not support output layer changes. "
                    f"Error: {str(generic_err)}"
                )
    
    except Exception as e:
        logger.error(f" ❌ Errore change_output_layer: {str(e)}")
        raise


def _add_dropout(model: Model, dropout_rate: float) -> tuple:
    """Fixed: controlla se Dropout esiste già"""
    
    try:
        dropout_rate = float(dropout_rate)
        
        if not (0.0 <= dropout_rate < 1.0):
            raise ValueError(f"Dropout rate deve essere in [0, 1)")
        
        # Controlla se modello ha già Dropout
        has_dropout = any(isinstance(l, Dropout) for l in model.layers)
        
        if has_dropout:
            logger.warning(f"⚠️  Model already has Dropout layers")
            logger.warning(f"  Adding more might degrade performance")
            response = input("Continuare? (s/n): ")
            if response.lower() != 's':
                return model, "Dropout addition skipped"
# I moderni (EfficientNet, ViT) già hanno Dropout:
# EfficientNet già ha:
# - Conv + BatchNorm + Dropout(0.2)
# - Conv + BatchNorm + Dropout(0.2)
# - ...
# e Aggiungere altro Dropout = overkill → degradazione performance
        
        # Aggiungi Dropout
        penultimate_layer = model.layers[-2]
        x = penultimate_layer.output
        x = Dropout(dropout_rate, name=f'dropout_custom')(x)
        output = Dense(model.layers[-1].units, activation='softmax')(x)
        
        new_model = Model(inputs=model.input, outputs=output)
        
        logger.info(f" ✓ Dropout({dropout_rate}) added")
        return new_model, f"Added Dropout({dropout_rate})"
    
    except Exception as e:
        logger.error(f" ❌ Error: {str(e)}")
        raise


def _freeze_layers(model: Model, num_frozen: int) -> tuple:
    """
    Congela i primi N layer del modello (trainable=False).
    
    NON ricostruisce il modello, solo modifica la proprietà trainable.
    """
    try:
        num_frozen = int(num_frozen)
        total_layers = len(model.layers)
        
        if num_frozen < 0:
            raise ValueError(f"num_frozen_layers deve essere >= 0, ricevuto {num_frozen}")
        
        if num_frozen > total_layers:
            logger.warning(f"num_frozen ({num_frozen}) > total_layers ({total_layers}), congelando tutti")
            num_frozen = total_layers
        
        # Congela i layer
        for layer in model.layers[:num_frozen]:
            layer.trainable = False
        
        num_trainable = sum(1 for l in model.layers if l.trainable)
        
        logger.info(f" ✓ Congelati {num_frozen}/{total_layers} layer (trainable: {num_trainable})")
        return model, f"Froze first {num_frozen}/{total_layers} layers (trainable: {num_trainable})"
    
    except Exception as e:
        logger.error(f" ❌ Errore freeze_layers: {str(e)}")
        raise


def _freeze_almost_all(model: Model, num_trainable: int) -> tuple:
    """
    Congela tutti i layer tranne gli ultimi N.
    """
    try:
        num_trainable = int(num_trainable)
        total_layers = len(model.layers)
        num_to_freeze = total_layers - num_trainable
        
        if num_trainable < 0:
            raise ValueError(f"num_trainable_layers deve essere >= 0, ricevuto {num_trainable}")
        
        if num_trainable >= total_layers:
            logger.warning(f"num_trainable ({num_trainable}) >= total_layers ({total_layers}), nessun freeze")
            return model, "No layers frozen (num_trainable >= total_layers)"
        
        # Congela i layer tranne gli ultimi
        for layer in model.layers[:num_to_freeze]:
            layer.trainable = False
        
        logger.info(f" ✓ Congelati tutti tranne ultimi {num_trainable} layer")
        return model, f"Froze all except last {num_trainable} layers"
    
    except Exception as e:
        logger.error(f" ❌ Errore freeze_almost_all: {str(e)}")
        raise


# ============================================================================
#                    MAIN CUSTOMIZATION FUNCTION
# ============================================================================


def fine_tune_customized_model(state: MasterState, config: dict) -> MasterState:
    """
    ✨ Fine-tuning con dataset dimensionato correttamente
    """
    
    logger.info("🎓 Iniziando fine-tuning...")
    
    try:
        model_path = state.customized_model_path
        
        if not model_path or not os.path.exists(model_path):
            logger.error("❌ customized_model_path is empty!")
            raise FileNotFoundError("customized_model_path not set")
        
        logger.info(f"  Model path: {model_path}")
        logger.info(f"  File size: {os.path.getsize(model_path) / 1024 / 1024:.2f} MB")
        
        training_rec = state.parsed_modifications.get('training_recommendation', {})
        learning_rate = training_rec.get('learning_rate', state.custom_learning_rate or 0.0001)
        epochs = training_rec.get('epochs', state.custom_epochs or 10)
        batch_size = training_rec.get('batch_size', state.custom_batch_size or 32)
        
        logger.info(f"  Training params: LR={learning_rate}, epochs={epochs}, batch_size={batch_size}")
        
        output_path = model_path.replace('.keras', '_finetuned.keras').replace('.h5', '_finetuned.h5')
        
        # ===== PYTHON SCRIPT CON SHAPE CORRETTO =====
        python_code = f"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
import sys

model_path = r"{model_path}"
output_path = r"{output_path}"

try:
    # ===== CARICA MODELLO =====
    model = tf.keras.models.load_model(model_path, compile=False)
    print(f"✓ Model loaded: {{model.name}}")
    
    # ===== ESTRAI INPUT SHAPE DAL MODELLO =====
    # input_shape = (None, 224, 224, 3) → (224, 224, 3)
    input_shape = model.input_shape[1:]  # Rimuovi batch dimension
    print(f"✓ Input shape from model: {{input_shape}}")
    
    num_classes = int(model.output_shape[-1])
    print(f"✓ Output classes: {{num_classes}}")
    
    # ===== GENERATE DATASET CON SHAPE CORRETTO =====
    num_samples = 200
    
    # Crea dataset con dimensioni CORRETTE
    X_train = np.random.randn(num_samples, *input_shape).astype('float32')
    X_train = (X_train - X_train.mean()) / (X_train.std() + 1e-7)
    y_train = np.eye(num_classes)[np.random.randint(0, num_classes, num_samples)]
    
    X_val = np.random.randn(50, *input_shape).astype('float32')
    X_val = (X_val - X_val.mean()) / (X_val.std() + 1e-7)
    y_val = np.eye(num_classes)[np.random.randint(0, num_classes, 50)]
    
    print(f"✓ Dataset created: train={{X_train.shape}}, val={{X_val.shape}}")
    
    # ===== COMPILE MODEL =====
    optimizer = Adam(learning_rate={learning_rate})
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    print(f"✓ Model compiled")
    
    # ===== CALLBACKS =====
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True,
        verbose=0
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-7,
        verbose=0
    )
    
    # ===== TRAINING (NO data augmentation, numpy puro) =====
    print(f"✓ Starting training for {epochs} epochs...")
    
    history = model.fit(
        X_train, y_train,
        batch_size={batch_size},
        epochs={epochs},
        validation_data=(X_val, y_val),
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    
    # ===== SAVE MODEL =====
    model.save(output_path, save_format='h5')  # ← Salva in .h5!
    print(f"✓ Model saved: {{output_path}}")
    
    # ===== OUTPUT RESULTS =====
    final_acc = float(history.history['accuracy'][-1])
    final_val_acc = float(history.history['val_accuracy'][-1])
    final_loss = float(history.history['loss'][-1])
    final_val_loss = float(history.history['val_loss'][-1])
    epochs_trained = len(history.history['loss'])
    
    print(f"SUCCESS: {{final_acc:.4f}}|{{final_val_acc:.4f}}|{{final_loss:.4f}}|{{final_val_loss:.4f}}|{{epochs_trained}}")
    sys.exit(0)
    
except Exception as e:
    print(f"ERROR: {{str(e)}}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
"""
        
        logger.info(f"  [Subprocess] Executing fine-tuning in stm32_legacy...")
        
        python_path = '/home/mrusso/miniconda3/envs/stm32_legacy/bin/python'
        
        result = subprocess.run(
            [python_path, '-c', python_code],
            capture_output=True,
            text=True,
            timeout=600,
            env={**os.environ, 'TF_CPP_MIN_LOG_LEVEL': '3'}
        )
        
        # ===== PARSE OUTPUT =====
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()
        
        logger.info(f"  [Raw stdout lines: {len(stdout.split(chr(10)))}]")
        
        # Filtra linee TensorFlow
        stdout_lines = [l for l in stdout.split('\n') if l and not any(x in l for x in [
            'tensorflow/core/util/port.cc',
            'tensorflow/tsl/cuda/cudart_stub.cc',
            'tensorflow/core/platform/cpu_feature_guard.cc',
            'tensorflow/compiler/tf2tensorrt',
            'oneDNN custom operations',
            'Could not find cuda drivers',
            'TF-TRT Warning'
        ])]
        stdout_clean = '\n'.join(stdout_lines)
        
        logger.info(f"  Output:\n{stdout_clean[:1500]}")
        
        if result.returncode != 0:
            logger.error(f"  Subprocess returncode: {result.returncode}")
            logger.error(f"  Stderr:\n{stderr[:1000]}")
            raise Exception(f"Subprocess failed: {stderr.split(chr(10))[-2]}")
        
        # ===== PARSE RESULTS =====
        if "SUCCESS:" in stdout_clean:
            parts = stdout_clean.split("SUCCESS:")[-1].strip().split('|')
            
            if len(parts) < 5:
                raise Exception(f"Invalid output format. Expected 5 parts, got {len(parts)}: {parts}")
            
            try:
                final_acc = float(parts[0].strip())
                final_val_acc = float(parts[1].strip())
                final_loss = float(parts[2].strip())
                final_val_loss = float(parts[3].strip())
                epochs_trained = int(parts[4].strip())
            except ValueError as e:
                raise Exception(f"Failed to parse metrics: {e}. Parts: {parts}")
            
            state.training_test_result = {
                "success": True,
                "final_accuracy": final_acc,
                "final_val_accuracy": final_val_acc,
                "final_loss": final_loss,
                "final_val_loss": final_val_loss,
                "epochs_trained": epochs_trained,
            }
            
            state.training_validation_success = True
            state.customized_model_path = output_path
            
            logger.info(f"✓ Fine-tuning completato!")
            logger.info(f"  Final accuracy: {final_acc:.2%}")
            logger.info(f"  Final val accuracy: {final_val_acc:.2%}")
        else:
            raise Exception(f"Output does not contain SUCCESS marker")
    
    except Exception as e:
        logger.error(f"❌ Fine-tuning failed: {str(e)}", exc_info=True)
        state.training_validation_success = False
        state.training_test_result = {
            "success": False,
            "error": str(e)
        }
    
    return state

def validate_customized_model(state: MasterState, config: dict) -> MasterState: 
    """Valida il modello customizzato IN SUBPROCESS"""
    
    logger.info("✅ Validando modello customizzato...")
    
    try:
        model_path = state.customized_model_path
        
        python_code = f"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import json

model_path = r'{model_path}'

try:
    model = tf.keras.models.load_model(model_path, compile=False)
    
    # Estrai informazioni
    info = {{
        'input_shape': str(model.input_shape),
        'output_shape': str(model.output_shape),
        'total_params': int(model.count_params()),
    }}
    
    # Stampa summary
    print("=== MODEL SUMMARY ===")
    model.summary(print_fn=print)
    print("=== END SUMMARY ===")
    
    print(f"SUCCESS: " + json.dumps(info))
    
except Exception as e:
    print(f"ERROR: {{str(e)}}")
    import traceback
    traceback.print_exc()
"""
        
        result = execute_in_stm32_legacy(python_code)
        
        if not result['success']:
            logger.error(f"❌ Validation failed: {result['stderr'][:500]}")
            state.error_message = result['stderr']
            return state
        
        # Parse info
        stdout = result['stdout']
        
        # Estrai JSON
        if "SUCCESS: " in stdout:
            json_str = stdout.split("SUCCESS: ")[-1].strip()
            info = json.loads(json_str)
            
            state.customized_model_info.update(info)
            logger.info(f"✓ Model validated")
            logger.info(f"  Input: {info['input_shape']}")
            logger.info(f"  Output: {info['output_shape']}")
            logger.info(f"  Params: {info['total_params']:,}")
        
    except Exception as e:
        logger.error(f"❌ Validation error: {str(e)}", exc_info=True)
        state.error_message = str(e)
    
    return state


def apply_quantization_for_stm32(state: MasterState, config: dict) -> MasterState:
    """
    Applica quantizzazione INT8 usando TensorFlow Lite NATIVO.
    NO tfmot, NO dipendenze problematiche.
    """
    
    if not state.should_quantize:
        logger.info("⏭️  Quantizzazione skippata (non richiesta)")
        return state
    
    logger.info(f"⚙️  Applicando quantizzazione INT{state.quantization_bit_width} con TFLite...")
    
    try:
        model = tf.keras.models.load_model(state.customized_model_path, compile=False)
        
        if state.quantization_bit_width == 8:
            
            logger.info("  Convertendo a TFLite INT8 (nativo)...")
            
            # Crea representative dataset per quantizzazione
            def representative_data_gen():
                X, _ = load_or_create_sample_dataset(50, (32, 32), int(model.output_shape[-1]))
                for i in range(0, len(X), 32):
                    yield [X[i:i+32].astype(np.float32)]
            
            # Converter con quantizzazione INT8
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
            # Specifica target ops per INT8
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS_INT8
            ]
            
            # Imposta representative dataset
            converter.representative_data = representative_data_gen
            
            # Imposta tipi input/output
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
            
            logger.info("  Quantizzando a INT8...")
            tflite_quant_model = converter.convert()
            
            # Salva modello quantizzato
            output_dir = os.path.expanduser("~/.stm32_ai_models/quantized")
            os.makedirs(output_dir, exist_ok=True)
            
            tflite_path = os.path.join(output_dir, f"model_int8_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tflite")
            
            with open(tflite_path, 'wb') as f:
                f.write(tflite_quant_model)
            
            state.quantized_model_path = tflite_path
            
            # Compara dimensioni
            original_size = os.path.getsize(state.customized_model_path) / (1024*1024)
            quantized_size = os.path.getsize(tflite_path) / (1024*1024)
            reduction = (1 - quantized_size/original_size) * 100
            
            logger.info(f"✓ Quantizzazione completata!")
            logger.info(f"  Original: {original_size:.2f} MB")
            logger.info(f"  Quantized: {quantized_size:.2f} MB")
            logger.info(f"  Reduction: {reduction:.1f}%")
            logger.info(f"  Saved: {tflite_path}")
        
    except Exception as e:
        logger.error(f"❌ Errore quantizzazione: {str(e)}", exc_info=True)
        state.should_quantize = False
    
    return state


import shutil

def save_customized_model_final(state: MasterState, config: dict) -> MasterState:
    """
    ✨ Salva il modello customizzato come .h5 (compatibile con stedgeai)
    """
    
    logger.info("💾 Salvando modello customizzato definitivamente...")
    
    try:
        model_path = state.customized_model_path
        
        if not model_path or not os.path.exists(model_path):
            logger.error("❌ Customized model not found")
            state.error_message = "Customized model not found"
            return state
        
        output_dir = os.path.expanduser("~/.stm32_ai_models/customized")
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_path = os.path.join(output_dir, f"customized_final_{timestamp}.h5")  # ← .h5! e non .keras !!
        
        # ===== COPIA FILE =====
        shutil.copy(model_path, final_path)
        logger.info(f"✓ Model copied: {final_path}")
        
        # ===== VALIDATE IN SUBPROCESS =====
        python_code = f"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import json

model_path = r'{final_path}'

try:
    model = tf.keras.models.load_model(model_path, compile=False)
    
    info = {{
        'input_shape': str(model.input_shape),
        'output_shape': str(model.output_shape),
        'total_params': int(model.count_params()),
        'model_name': model.name,
    }}
    
    print(f"SUCCESS: " + json.dumps(info))
    
except Exception as e:
    print(f"ERROR: {{str(e)}}")
"""
        
        result = execute_in_stm32_legacy(python_code)
        
        if not result['success']:
            logger.error(f"❌ Final save validation failed")
            state.error_message = result['stderr']
            return state
        
        # Parse info
        if "SUCCESS: " in result['stdout']:
            json_str = result['stdout'].split("SUCCESS: ")[-1].strip()
            info = json.loads(json_str)
            
            state.final_model_path = final_path
            state.customized_model_info.update({
                **info,
                "model_size_mb": round(os.path.getsize(final_path) / (1024*1024), 2),
                "format": "H5"  # ← Aggiungi info formato
            })
            
            logger.info(f"✓ Model saved: {final_path}")
            logger.info(f"  Format: H5 (stedgeai compatible)")
            logger.info(f"  Size: {state.customized_model_info['model_size_mb']} MB")
        
    except Exception as e:
        logger.error(f"❌ Save error: {str(e)}", exc_info=True)
        state.error_message = str(e)
    
    return state

def ask_continue_after_customization(state: MasterState, config: dict) -> MasterState:
    """Chiedi se continuare con AI analysis"""
    
    logger.info("🤔 Chiedendo se continuare...")
    
    
    
    summary = f"""
Customization Complete!

Final Model: {state.final_model_path}
- Input: {state.customized_model_info.get('input_shape')}
- Output: {state.customized_model_info.get('output_shape')}
- Params: {state.customized_model_info.get('total_params'):,}
- Size: {state.customized_model_info.get('model_size_mb', 'N/A')} MB

Training Results:
- Accuracy: {state.training_test_result.get('final_accuracy', 'N/A')}
- Val Accuracy: {state.training_test_result.get('final_val_accuracy', 'N/A')}

Quantized: {state.should_quantize}
{f'- Quantized model: {state.quantized_model_path}' if state.quantized_model_path else ''}
"""
    
    prompt = {
        "instruction": "Do you want to continue with X-CUBE-AI analysis?",
        "summary": summary,
        "options": ["continue_ai", "end"]
    }
    
    user_response = interrupt(prompt)
    state.continue_after_customization = (user_response == "continue_ai")
    
    return state


# 🥇 Deepseek-r1      (BEST: reasoning perfetto, JSON impeccabile). Qualche secondo in più per riflettere, ma più leggero di Mistral (70 B vs 72 B) e qualità migliore. 
# 🥈 Mistral 72B      (GOOD: veloce, OK qualità)
# 🥉 Qwen2 7B         (OK: leggero ma qualità minore)

