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

import os
import json
import logging
import numpy as np
from typing import Literal, Optional, Tuple, Dict, List, Any
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Dense, Dropout, Input, Resizing, Conv2D, 
    GlobalAveragePooling2D, GlobalMaxPooling2D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import h5py

from langchain_ollama import ChatOllama
from langchain.embeddings.base import Embeddings
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.types import interrupt
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from langchain.schema import Document  

from src.assistant.configuration import Configuration
from src.assistant.state import MasterState


logger = logging.getLogger(__name__)


# ============================================================================
# WORKFLOW 5: MODELS CUSTOMIZATION
# ============================================================================

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


def retrieve_best_practices_info(state: MasterState, config: dict) -> MasterState:
    """
    Recupera best practices da:
    1. Chroma database (se esiste e non è vuoto)
    2. Online fetch (se Chroma è vuoto o assente)
    3. Default hardcoded (fallback finale)
    
    Segue pattern: Check Existence → Build if Missing → Fallback to Defaults
    """
    
    persist_dir = "./chroma_customization_docs"
    
    print("\n" + "="*80)
    print("BEST PRACTICES RETRIEVAL")
    print("="*80 + "\n")
    
    logger.info("📚 Recuperando best practices...")
    
    # ✅ Check if vectorstore already exists and is not empty
    db_exists = os.path.exists(persist_dir) and os.listdir(persist_dir)
    
    if db_exists:
        print(f"✓ Vectorstore exists at {persist_dir}")
        print(f"✅ Loading from existing database (skipping rebuild)\n")
        
        try:
            best_practices = _retrieve_from_chroma(state, persist_dir)
            
            if best_practices:
                state.best_practices_display = _format_practices(best_practices, source="DATABASE")
                state.best_practices_raw = [p.page_content for p in best_practices]
                logger.info("✓ Best practices recuperate da database")
                print("="*80 + "\n")
                return state
            else:
                print("⚠️  Database vuoto, fallback online...\n")
        
        except Exception as e:
            print(f"⚠️  Errore nel caricamento da Chroma: {e}")
            print(f"   Fallback online...\n")
    else:
        print(f"❌ Vectorstore NOT found at {persist_dir}")
        print(f"📥 Fetching best practices from online sources...\n")
    
    # ============================================================
    # FETCH DA ONLINE (se DB assente o vuoto)
    # ============================================================
    
    try:
        best_practices = _fetch_best_practices_online(state, persist_dir)
        
        if best_practices:
            state.best_practices_display = _format_practices(best_practices, source="ONLINE")
            state.best_practices_raw = [p.page_content for p in best_practices]
            logger.info("✓ Best practices recuperate online")
            print("="*80 + "\n")
            return state
        else:
            raise Exception("Online fetch returned empty results")
    
    except Exception as e:
        logger.warning(f"⚠️  Online fetch fallito: {str(e)}")
        print(f"⚠️  Online fetch failed: {e}")
        print(f"   Loading DEFAULT hardcoded practices...\n")
    
    # ============================================================
    # FALLBACK: DEFAULT HARDCODED
    # ============================================================
    
    state.best_practices_display = _get_default_practices()
    state.best_practices_raw = []
    logger.warning("Using fallback default best practices")
    
    print("="*80 + "\n")
    return state


# ============================================================
# HELPER FUNCTIONS for retrieve_best_practices_info
# ============================================================

def _retrieve_from_chroma(state: MasterState, persist_dir: str) -> Optional[List[Document]]:
    """Carica best practices da Chroma database"""
    
    print("🔧 Loading embedding model (all-MiniLM-L6-v2)...")
    
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    print("   ✓ Embedding model loaded\n")
    
    print("🗄️  Loading Chroma vectorstore...")
    vectorstore = Chroma(
        embedding_function=embedding_model,
        collection_name="model-customization-practices",
        persist_directory=persist_dir
    )
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    query = f"Best practices for fine-tuning {state.selected_model.get('name', 'ResNet')} for {state.custom_use_case}"
    print(f"📝 Query: {query}\n")
    
    practices = retriever.invoke(query)
    
    if practices:
        print(f"   ✓ Retrieved {len(practices)} best practices from database\n")
    else:
        print(f"   ⚠️  No practices found in database\n")
    
    return practices if practices else None


def _fetch_best_practices_online(state: MasterState, persist_dir: str) -> Optional[List[Document]]:
    """
    Scarica best practices online da fonti rilevanti:
    - Keras/TensorFlow docs
    - PyTorch documentation
    - Research papers summary
    
    Salva i risultati in Chroma per cache persistente
    """
    
    print("-" * 80)
    print("FETCHING BEST PRACTICES FROM ONLINE SOURCES")
    print("-" * 80 + "\n")
    
    model_name = state.selected_model.get('name', 'ResNet')
    use_case = state.custom_use_case
    
    all_docs = []
    
    # ========== SOURCE 1: PyImageSearch (Transfer Learning & Fine-tuning) ==========
    try:
        print("📚 Source 1: Fetching PyImageSearch documentation...")
        loader = RecursiveUrlLoader(
            url="https://pyimagesearch.com/2020/04/27/fine-tuning-resnet-with-keras-tensorflow-and-deep-learning/",
            max_depth=2,
            extractor=lambda x: BeautifulSoup(x, "html.parser").get_text()
        )
        docs = loader.load()
        print(f"   ✓ Loaded {len(docs)} PyImageSearch pages\n")
        all_docs.extend(docs)
    except Exception as e:
        print(f"   ⚠️  PyImageSearch fetch failed: {e}\n")
    
    # ========== SOURCE 2: TensorFlow Official Docs ==========
    try:
        print("📚 Source 2: Fetching TensorFlow transfer learning guide...")
        loader = RecursiveUrlLoader(
            url="https://www.tensorflow.org/tutorials/images/transfer_learning",
            max_depth=3,
            extractor=lambda x: BeautifulSoup(x, "html.parser").get_text()
        )
        docs = loader.load()
        print(f"   ✓ Loaded {len(docs)} TensorFlow pages\n")
        all_docs.extend(docs)
    except Exception as e:
        print(f"   ⚠️  TensorFlow fetch failed: {e}\n")
    
    # ========== SOURCE 3: Keras Model Customization ==========
    try:
        print("📚 Source 3: Fetching Keras model customization guide...")
        loader = RecursiveUrlLoader(
            url="https://keras.io/guides/functional_api/",
            max_depth=2,
            extractor=lambda x: BeautifulSoup(x, "html.parser").get_text()
        )
        docs = loader.load()
        print(f"   ✓ Loaded {len(docs)} Keras pages\n")
        all_docs.extend(docs)
    except Exception as e:
        print(f"   ⚠️  Keras fetch failed: {e}\n")
    
    # ========== SOURCE 4: STM32 AI Model Optimization (Model Zoo) ==========
    try:
        print("📚 Source 4: Fetching STM32 AI optimization guide...")
        loader = RecursiveUrlLoader(
            url="https://github.com/STMicroelectronics/stm32ai-modelzoo/wiki",
            max_depth=2,
            extractor=lambda x: BeautifulSoup(x, "html.parser").get_text()
        )
        docs = loader.load()
        print(f"   ✓ Loaded {len(docs)} STM32 AI pages\n")
        all_docs.extend(docs)
    except Exception as e:
        print(f"   ⚠️  STM32 AI fetch failed: {e}\n")
    
    # ========== FALLBACK: Curated Best Practices Document ==========
    if not all_docs:
        print("⚠️  All online sources failed, using curated best practices...\n")
        curated_doc = Document(
            page_content="""# CURATED BEST PRACTICES FOR MODEL FINE-TUNING

## Transfer Learning Best Practices
1. Freeze early layers (maintain pre-trained feature extractors)
2. Fine-tune only the last 2-3 layers with low learning rate
3. Use progressively lower learning rate (1e-5 to 1e-3 range)
4. Monitor validation accuracy carefully

## Hyperparameter Tuning
- Learning Rate: Start at 1e-4, decrease if overfitting
- Batch Size: 32-64 for most datasets, 128 for large datasets
- Epochs: 10-20 for transfer learning (avoid overfitting)
- Optimizer: Adam (adaptive) or SGD with momentum

## Data Augmentation
- Random rotation: ±15 degrees
- Random flip: horizontal (50% probability)
- Random zoom: 10-20% range
- Brightness/contrast adjustment: ±20%

## Quantization for STM32 Deployment
- INT8 quantization: 4× memory reduction with minimal accuracy loss
- Quantization-aware training (QAT) for better post-quantization accuracy
- Test on representative data before deployment

## Validation & Testing
- Use stratified k-fold cross-validation
- Monitor loss and accuracy separately
- Test on real-world data before final deployment
- Keep test set untouched during development

## Model Architecture Modifications
- Remove final classification layer
- Add new dense layers matching your classes
- Use GlobalAveragePooling2D before final layers
- Consider dropout (0.2-0.5) for regularization
""",
            metadata={"source": "CURATED_BEST_PRACTICES"}
        )
        all_docs = [curated_doc]
    
    if not all_docs:
        print("❌ No documents fetched from any source\n")
        return None
    
    print(f"✂️  Splitting {len(all_docs)} documents (chunk_size=512, overlap=50)...")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )
    
    doc_splits = []
    for doc in all_docs:
        splits = splitter.split_documents([doc])
        doc_splits.extend(splits)
    
    print(f"   ✓ Created {len(doc_splits)} document chunks\n")
    
    # ========== EMBEDDING & VECTORSTORE CREATION ==========
    
    print("🔧 Loading embedding model (all-MiniLM-L6-v2)...")
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    print("   ✓ Embedding model loaded\n")
    
    print(f"🗄️  Creating Chroma vectorstore (collection: model-customization-practices)...")
    
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        embedding=embedding_model,
        collection_name="model-customization-practices",
        persist_directory=persist_dir,
        collection_metadata={"hnsw:space": "cosine"}
    )
    vectorstore.persist()
    
    print(f"   ✓ Vectorstore persisted at '{persist_dir}'")
    print(f"   ✓ Total documents indexed: {len(doc_splits)}\n")
    
    # ========== RETRIEVE RELEVANT PRACTICES ==========
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    query = f"Best practices for fine-tuning {model_name} for {use_case}"
    print(f"📝 Retrieving relevant practices...")
    print(f"   Query: {query}\n")
    
    practices = retriever.invoke(query)
    
    print(f"   ✓ Retrieved {len(practices)} best practices\n")
    
    return practices if practices else None


def _format_practices(practices: List[Document], source: str = "UNKNOWN") -> str:
    """Formatta le best practices per display"""
    
    formatted = f"""
════════════════════════════════════════════════════════════
📋 BEST PRACTICES & SUGGESTIONS [{source}]
════════════════════════════════════════════════════════════

"""
    
    for i, practice in enumerate(practices, 1):
        # Prendi primi 300 caratteri per evitare testo troppo lungo
        content = practice.page_content[:300]
        if len(practice.page_content) > 300:
            content += "..."
        
        formatted += f"{i}. {content}\n\n"
    
    formatted += """════════════════════════════════════════════════════════════
"""
    
    return formatted

def _get_default_practices() -> str:
    """Ritorna le best practices hardcoded di default"""
    
    return """
════════════════════════════════════════════════════════════
📋 BEST PRACTICES (DEFAULT FALLBACK)
════════════════════════════════════════════════════════════
1. Use low learning rate (1e-5 to 1e-3) for transfer learning
2. Freeze early layers, fine-tune only last layers
3. Batch size: 32-64, Epochs: 10-20 (avoid overfitting)
4. Use data augmentation (rotation, flip, zoom)
5. Monitor loss and accuracy trends separately
6. Apply INT8 quantization for STM32 deployment (4× size reduction)
7. Test on real data before final deployment
8. Use stratified k-fold cross-validation
════════════════════════════════════════════════════════════
"""

# 🥇 Deepseek-r1      (BEST: reasoning perfetto, JSON impeccabile). Qualche secondo in più per riflettere, ma più leggero di Mistral (70 B vs 72 B) e qualità migliore. 
# 🥈 Mistral 72B      (GOOD: veloce, OK qualità)
# 🥉 Qwen2 7B         (OK: leggero ma qualità minore)

############# Fine helper functions

def ask_user_for_custom_modifications(state: any, config: dict) -> any:
    """
    Mostra best practices e chiede all'utente quali modifiche applicare.
    
    Supporta:
      ✓ Freeze layers (first N)
      ✓ Freeze almost all (keep last N trainable)
      ✓ Change output classes
      ✓ Add dropout
      ✓ Change input shape
      ✓ Add resize preprocessing
      ✓ Change learning rate
    
    Args:
        state: MasterState object
        config: Configuration dict
    
    Returns:
        state aggiornato con user_custom_modifications
    """
    logger.info("🤔 Chiedendo all'utente quali modifiche applicare...")
    
    # ✅ PROTEZIONE: Estrai i valori in modo safe
    input_shape = state.model_architecture.get('input_shape', 'Unknown')
    output_classes = state.model_architecture.get('output_classes', 0)
    total_params = state.model_architecture.get('total_params', 0)
    num_layers = len(state.model_architecture.get('layer_types', []))
    
    # ✅ PROTEZIONE: Formatta total_params in modo safe
    formatted_params = f"{total_params:,}" if total_params else "N/A"
    
    # Genera best practices
    best_practices = get_modification_best_practices(state.model_architecture)
    
    # Prompt mostrato all'utente
    prompt = {
        "instruction": f"""
╔═══════════════════════════════════════════════════════════╗
║         🛠️  CUSTOMIZE YOUR STM32 MODEL                    ║
╚═══════════════════════════════════════════════════════════╝

Current Model Info:
  • Input: {input_shape}
  • Output classes: {output_classes}
  • Total params: {formatted_params}
  • Layers: {num_layers}

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
  • "Add resize preprocessing to 224x224 and change output to 1000"

Write your modifications in natural language:
""",
        "best_practices": best_practices,
    }
    
    # ⏸️ INTERRUPT: Attendi risposta utente
    user_modifications = interrupt(prompt)
    
    # Salva la richiesta nello state
    state.user_custom_modifications = user_modifications
    
    logger.info(f"📝 User request: {user_modifications[:100]}...")
    
    return state


# ============================================================================
#                    PARSE USER MODIFICATIONS
# ============================================================================

def parse_user_modifications(state: any, config: dict) -> any:
    """
    Usa LLM per interpretare richieste utente in linguaggio naturale
    e convertirle in operazioni strutturate JSON.
    
    Supporta tutte le modifiche disponibili:
      - freeze_layers: Congela i primi N layer
      - freeze_almost_all: Congela tutti tranne gli ultimi N
      - change_output_layer: Modifica il numero di classi di output
      - add_dropout: Aggiunge dropout prima dell'output
      - change_input_shape: Cambia la dimensione di input
      - add_resize_preprocessing: Aggiunge ridimensionamento automatico
      - change_learning_rate: Imposta un learning rate custom
    
    Args:
        state: MasterState object
        config: Configuration dict
    
    Returns:
        state aggiornato con parsed_modifications
    """
    logger.info("🔍 Interpretando modifiche dell'utente con LLM...")
    
    # Inizializza l'agente LLM con mistral (veloce e affidabile)
    agent = Agent(model=Ollama(id="mistral"))
    
    logger.info(f"📝 User request: {state.user_custom_modifications}")
    
    # Costruisce il prompt che spiega all'LLM cosa deve fare
    prompt = f"""
Parse user request to customize neural network model.

# RICHIESTA DELL'UTENTE
USER REQUEST: "{state.user_custom_modifications}"

# INFO ATTUALI DEL MODELLO
CURRENT MODEL:
  - Input shape: {state.model_architecture['input_shape']}
  - Output classes: {state.model_architecture['output_classes']}
  - Total params: {state.model_architecture['total_params']}
  - Number of layers: {len(state.model_architecture.get('layer_types', []))}

# SUPPORTED MODIFICATION TYPES
Available modifications:
  1. freeze_layers: Freeze first N layers (keep other trainable)
     params: {{"num_frozen_layers": <int>}}
  
  2. freeze_almost_all: Freeze all except last N layers
     params: {{"num_trainable_layers": <int>}}
  
  3. change_output_layer: Change number of output classes
     params: {{"new_classes": <int>}}
  
  4. add_dropout: Add dropout layer before output (0.0-1.0)
     params: {{"rate": <float>}}
  
  5. change_input_shape: Change input dimensions
     params: {{"new_shape": [height, width, channels]}}
  
  6. add_resize_preprocessing: Auto-resize input to target size
     params: {{"height": <int>, "width": <int>}}
  
  7. change_learning_rate: Set custom learning rate for fine-tuning
     params: {{"learning_rate": <float>}}

# INSTRUCTIONS
- Return ONLY valid JSON (no markdown, no extra text)
- Include confidence score (0.0-1.0) for each modification
- For ambiguous requests, ask for clarification or make reasonable assumptions
- Ensure all numeric values are valid (e.g., freeze_layers < total_layers)
- Include training recommendations based on modifications

# OUTPUT FORMAT (STRICT JSON)
{{
  "modifications": [
    {{
      "type": "<modification_type>",
      "description": "Brief description of what this does",
      "params": {{...}},
      "confidence": 0.95
    }}
  ],
  "summary": "Brief summary of all modifications",
  "confidence": 0.92,
  "validation": {{
    "is_valid": true,
    "issues": []
  }},
  "training_recommendation": {{
    "learning_rate": 0.0001,
    "epochs": 15,
    "batch_size": 32,
    "optimizer": "adam",
    "notes": "Use lower learning rate for fine-tuning"
  }}
}}
"""
    
    # PROVA A ESEGUIRE
    try:
        logger.info(" [Step 1] Eseguendo LLM parsing...")
        
        # Esegui il prompt con l'LLM
        response = agent.run(prompt)
        
        # Se la response è stringa semplice, usala direttamente; altrimenti estrai .content
        content = response if isinstance(response, str) else response.content
        
        logger.debug(f"   LLM response length: {len(content)} chars")
        
        # ESTRAZIONE JSON ROBUSTA: usa regex per trovare JSON anche se c'è testo intorno
        json_match = re.search(r'\{[\s\S]*\}', content)
        
        if json_match:
            json_str = json_match.group(0)
            modifications_plan = json.loads(json_str)
            logger.info(" ✓ JSON estratto con regex")
        else:
            modifications_plan = json.loads(content)
            logger.info(" ✓ JSON parsato direttamente")
        
        # Validazione base della struttura
        if 'modifications' not in modifications_plan:
            raise ValueError("Missing 'modifications' key in response")
        
        if not isinstance(modifications_plan['modifications'], list):
            raise ValueError("'modifications' must be a list")
        
        # Salva il piano interpretato nello state
        state.parsed_modifications = modifications_plan
        
        # Log di successo
        num_mods = len(modifications_plan.get('modifications', []))
        confidence = modifications_plan.get('confidence', 0.0)
        summary = modifications_plan.get('summary', 'N/A')
        
        logger.info(f"✅ Modifiche interpretate con successo!")
        logger.info(f"   • Numero operazioni: {num_mods}")
        logger.info(f"   • Confidenza LLM: {confidence:.0%}")
        logger.info(f"   • Summary: {summary}")
        
        # Mostra dettagli di ogni modifica
        for i, mod in enumerate(modifications_plan.get('modifications', []), 1):
            mod_type = mod.get('type', 'unknown')
            mod_desc = mod.get('description', 'N/A')
            mod_conf = mod.get('confidence', 0.0)
            logger.info(f"   [{i}] {mod_type} (confidence: {mod_conf:.0%})")
            logger.info(f"        → {mod_desc}")
    
    # SE FALLISCE IL PARSING JSON
    except (json.JSONDecodeError, ValueError, AttributeError) as e:
        logger.error(f"❌ Errore parsing LLM: {str(e)[:100]}")
        logger.warning("⚠️  Usando fallback generico...")
        
        # FALLBACK: Crea un piano GENERICO di default
        state.parsed_modifications = {
            "modifications": [
                {
                    "type": "freeze_almost_all",
                    "description": "Freeze all layers except the last 3 for fine-tuning",
                    "params": {
                        "num_trainable_layers": 3
                    },
                    "confidence": 0.5
                },
                {
                    "type": "add_dropout",
                    "description": "Add dropout (0.3) before output to prevent overfitting",
                    "params": {
                        "rate": 0.3
                    },
                    "confidence": 0.5
                },
                {
                    "type": "change_output_layer",
                    "description": "Ensure output matches your dataset classes",
                    "params": {
                        "new_classes": state.model_architecture.get('output_classes', 10)
                    },
                    "confidence": 0.6
                }
            ],
            "summary": "Basic fine-tuning setup (fallback - please review)",
            "confidence": 0.5,
            "validation": {
                "is_valid": False,
                "issues": ["LLM parsing failed, using default configuration"]
            },
            "training_recommendation": {
                "learning_rate": 0.0001,
                "epochs": 10,
                "batch_size": 32,
                "optimizer": "adam",
                "notes": "FALLBACK CONFIG - Please review and adjust as needed"
            }
        }
        
        logger.warning(f"⚠️  Using default configuration (3 modifications)")
    
    except Exception as e:
        logger.error(f"❌ Errore imprevisto: {str(e)}", exc_info=True)
        logger.warning("⚠️  Using minimal fallback...")
        
        state.parsed_modifications = {
            "modifications": [],
            "summary": f"Error during parsing: {str(e)[:50]}",
            "confidence": 0.0,
            "validation": {
                "is_valid": False,
                "issues": [f"Critical error: {str(e)[:80]}"]
            },
            "training_recommendation": {
                "learning_rate": 0.0001,
                "epochs": 5,
                "batch_size": 32,
                "optimizer": "adam",
                "notes": "ERROR - No modifications parsed"
            }
        }
    
    logger.info(f"📊 Parsed modifications: {len(state.parsed_modifications.get('modifications', []))} operations")
    
    return state


# ============================================================================
#                  COLLECT MODIFICATION CONFIRMATION
# ============================================================================

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
#                          LOAD MODEL FUNCTIONS
# ============================================================================

def load_stm32_model_safe(model_path: str) -> Optional[tf.keras.Model]:
    """
    Carica modello STM32 con fallback multi-step per problemi DepthwiseConv2D.
    
    Problema: file .h5 STM32 salvati con Keras < 3.x contengono parametro 'groups=1'
    che la versione nuova non riconosce.
    
    Strategy:
    1. Standard load_model() con safe_mode=False
    2. Se fallisce: fix manual della config rimovendo 'groups'
    3. Se ancora fallisce: carica solo pesi e ricrea architettura
    
    Args:
        model_path: Path al file .h5 STM32
    
    Returns:
        tf.keras.Model se successo, None se tutti i metodi falliscono
    """
    logger.info(f"📥 Caricando modello STM32 da {model_path}")
    
    if not os.path.exists(model_path):
        logger.error(f"❌ File non trovato: {model_path}")
        return None
    
    # =============== STEP 1: Try standard load with safe_mode=False ===============
    logger.info(" [Step 1/3] Tentativo standard (safe_mode=False)...")
    try:
        model = load_model(
            model_path,
            compile=False,
            safe_mode=False
        )
        logger.info(" ✓ SUCCESSO: Modello caricato direttamente")
        return model
    
    except Exception as e:
        error_msg = str(e)
        if "groups" not in error_msg and "DepthwiseConv2D" not in error_msg:
            logger.error(f" ❌ Errore non-DepthwiseConv2D: {error_msg[:80]}")
            raise
        logger.warning(f" ⚠️ DepthwiseConv2D groups issue detected")
    
    # =============== STEP 2: Fix config - Rimuovi 'groups' manualmente ===============
    logger.info(" [Step 2/3] Tentativo: Fix config DepthwiseConv2D...")
    try:
        with h5py.File(model_path, 'r') as f:
            # Estrai configurazione modello
            if 'model_config' not in f.attrs:
                raise ValueError("model_config attribute not found in HDF5")
            
            model_config_str = f.attrs['model_config']
            if isinstance(model_config_str, bytes):
                model_config_str = model_config_str.decode('utf-8')
            
            model_config = json.loads(model_config_str)
            
            # Rimuovi ricorsivamente 'groups' da tutta la config
            def remove_groups_recursive(obj):
                if isinstance(obj, dict):
                    obj.pop('groups', None)
                    for v in obj.values():
                        remove_groups_recursive(v)
                elif isinstance(obj, list):
                    for item in obj:
                        remove_groups_recursive(item)
            
            remove_groups_recursive(model_config)
            model_config_str_fixed = json.dumps(model_config)
            
            # Ricrea modello da config pulita
            model = model_from_json(model_config_str_fixed)
            
            # Carica pesi
            if 'model_weights' in f:
                logger.info(" Caricando pesi da HDF5...")
                success_count = 0
                
                for layer in model.layers:
                    if layer.name in f['model_weights']:
                        try:
                            weight_group = f['model_weights'][layer.name]
                            layer_weights = [
                                weight_group[key][:] 
                                for key in sorted(weight_group.keys())
                            ]
                            
                            if layer_weights:
                                layer.set_weights(layer_weights)
                                success_count += 1
                        
                        except Exception as w_err:
                            logger.debug(f" Skipped layer '{layer.name}': {str(w_err)[:50]}")
                
                logger.info(f" ✓ Pesi caricati: {success_count}/{len(model.layers)} layer")
        
        logger.info(" ✓ SUCCESSO: Modello ricostruito da config fissa")
        return model
    
    except Exception as e:
        logger.warning(f" ⚠️ Step 2 fallito: {str(e)[:80]}")
    
    # =============== STEP 3: Custom objects fallback ===============
    logger.info(" [Step 3/3] Tentativo: custom_objects vuoto...")
    try:
        model = load_model(
            model_path,
            compile=False,
            custom_objects={}
        )
        logger.info(" ✓ SUCCESSO: Caricato con custom_objects vuoto")
        return model
    
    except Exception as e:
        logger.error(f" ❌ Step 3 fallito: {str(e)[:80]}")
    
    # =============== FAILURE ===============
    logger.error("❌ FALLITO: Impossibile caricare il modello con nessun metodo")
    return None


# ============================================================================
#                      MODIFICATION FUNCTIONS
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
    }
    
    for mod in modifications.get('modifications', []):
        mod_type = mod.get('type', '').strip()
        mod_params = mod.get('params', {})
        
        if mod_type in required_params:
            for param in required_params[mod_type]:
                if param not in mod_params:
                    logger.warning(f"⚠️ Parametro mancante '{param}' per {mod_type}")
                    return False
    
    return True


def _change_input_shape(model: Model, new_shape: tuple, i: int) -> tuple:
    """
    Cambia lo shape dell'input layer del modello.
    
    Args:
        model: Modello da modificare
        new_shape: Nuova shape (es. (64, 64, 3))
        i: Indice della modifica
    
    Returns:
        Tuple (modello_modificato, descrizione)
    """
    try:
        if not isinstance(new_shape, (list, tuple)):
            raise ValueError(f"new_shape deve essere list o tuple, ricevuto {type(new_shape)}")
        
        # Crea nuovo input layer
        new_input = Input(shape=new_shape)
        
        # Passa attraverso il modello originale
        x = model(new_input)
        
        # Ricrea il modello
        new_model = Model(
            inputs=new_input,
            outputs=x,
            name=f"{model.name}_input_reshaped"
        )
        
        logger.info(f" ✓ Input shape cambiato a {new_shape}")
        return new_model, f"Changed input shape to {new_shape}"
    
    except Exception as e:
        logger.error(f" ❌ Errore change_input_shape: {str(e)}")
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
    Cambia il layer di output del modello.
    
    IMPORTANTE: Ricostruisce completamente il percorso dal penultimo layer.
    """
    try:
        new_classes = int(new_classes)
        
        if new_classes <= 0:
            raise ValueError(f"new_classes deve essere > 0, ricevuto {new_classes}")
        
        if len(model.layers) < 2:
            raise ValueError("Modello troppo semplice per modificare output (< 2 layer)")
        
        # Accedi all'output del penultimo layer
        penultimate_layer = model.layers[-2]
        x = penultimate_layer.output
        
        # Crea nuovo layer di output
        output = Dense(
            new_classes,
            activation='softmax',
            name='output_custom'
        )(x)
        
        # Ricrea il modello
        new_model = Model(
            inputs=model.input,
            outputs=output,
            name=f"{model.name}_modified_output"
        )
        
        logger.info(f" ✓ Output layer cambiato a {new_classes} classi")
        return new_model, f"Changed output layer to {new_classes} classes"
    
    except Exception as e:
        logger.error(f" ❌ Errore change_output_layer: {str(e)}")
        raise


def _add_dropout(model: Model, dropout_rate: float) -> tuple:
    """
    Aggiunge un layer Dropout prima dell'output finale.
    
    Ricrea correttamente la catena dei layer senza duplicazione.
    """
    try:
        dropout_rate = float(dropout_rate)
        
        if not (0.0 <= dropout_rate < 1.0):
            raise ValueError(f"Dropout rate deve essere in [0, 1), ricevuto {dropout_rate}")
        
        if len(model.layers) < 2:
            raise ValueError("Modello troppo semplice per aggiungere dropout")
        
        # Prendi l'uscita del penultimo layer
        penultimate_layer = model.layers[-2]
        x = penultimate_layer.output
        
        # Aggiungi Dropout
        x = Dropout(dropout_rate, name=f'dropout_custom')(x)
        
        # Ricrea il layer di output (non riapplicare quello vecchio!)
        output_layer = model.layers[-1]
        output = Dense(
            output_layer.units,
            activation=output_layer.activation,
            name='output_after_dropout'
        )(x)
        
        # Ricrea il modello
        new_model = Model(
            inputs=model.input,
            outputs=output,
            name=f"{model.name}_with_dropout"
        )
        
        logger.info(f" ✓ Dropout({dropout_rate}) aggiunto prima output")
        return new_model, f"Added Dropout({dropout_rate}) before output"
    
    except Exception as e:
        logger.error(f" ❌ Errore add_dropout: {str(e)}")
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

# Ogni tipo di modifica ora ha la sua funzione dedicata:
# _freeze_layers() - Congela i primi N layer
# _freeze_almost_all() - Congela tutti tranne gli ultimi N
# _change_output_layer() - Modifica il layer di output
# _add_dropout() - Aggiunge dropout prima dell'output
# _change_input_shape() - Cambia lo shape di input
# _add_resize_preprocessing() - Aggiunge preprocessing resize
# _validate_modifications() - Valida i parametri

# ============================================================================
#                    MAIN CUSTOMIZATION FUNCTION
# ============================================================================

def apply_user_customization(state: dict, config: dict) -> dict:
    """
    Applica modifiche utente al modello STM32.
    
    Workflow:
    1. Valida input
    2. Carica modello con load_stm32_model_safe()
    3. Valida modifiche
    4. Applica modifiche in sequenza
    5. Salva modello customizzato
    6. Popola state con metadata
    
    Args:
        state: dict con parsed_modifications, model_path, modification_confirmed, etc.
        config: Configuration globale
    
    Returns:
        state aggiornato con customized_model_path, customization_applied, etc.
    """
    logger.info("🔧 Applicando customizzazioni al modello STM32...")
    
    # ===== VALIDAZIONE INIZIALE =====
    if not state.get('modification_confirmed', False):
        logger.error("❌ Modifiche non confermate")
        state['customization_applied'] = False
        state['error_message'] = "Modifications not confirmed"
        return state
    
    model_path = state.get('model_path')
    if not model_path or not os.path.exists(model_path):
        logger.error(f"❌ Model path non valido: {model_path}")
        state['customization_applied'] = False
        state['error_message'] = "Invalid model path"
        return state
    
    # ===== CARICA MODELLO =====
    logger.info(" [Pre-check] Tentando di caricare il modello...")
    base_model = load_stm32_model_safe(model_path)
    
    if base_model is None:
        logger.error("❌ FALLITO: Impossibile caricare il modello")
        state['customization_applied'] = False
        state['error_message'] = "Model loading failed"
        return state
    
    logger.info(f" ✓ Pre-check passed: {base_model.name}")
    logger.info(f" Input: {base_model.input_shape} | Output: {base_model.output_shape}")
    
    current_model = base_model
    modifications_log = []
    
    try:
        # ===== VALIDAZIONE MODIFICHE =====
        parsed_mods = state.get('parsed_modifications', {})
        if not _validate_modifications(parsed_mods):
            logger.warning("⚠️ Alcuni parametri mancano, proseguendo comunque...")
        
        # ===== APPLICA MODIFICHE =====
        for i, mod in enumerate(parsed_mods.get('modifications', []), 1):
            mod_type = mod.get('type', '').strip()
            mod_params = mod.get('params', {})
            
            logger.info(f" [{i}] Applicando modifica: {mod_type}")
            
            try:
                # --- FREEZE LAYERS ---
                if mod_type == "freeze_layers":
                    num_frozen = int(mod_params.get('num_frozen_layers', 3))
                    current_model, desc = _freeze_layers(current_model, num_frozen)
                    modifications_log.append(desc)
                
                # --- FREEZE ALMOST ALL ---
                elif mod_type == "freeze_almost_all":
                    num_trainable = int(mod_params.get('num_trainable_layers', 3))
                    current_model, desc = _freeze_almost_all(current_model, num_trainable)
                    modifications_log.append(desc)
                
                # --- CHANGE OUTPUT LAYER ---
                elif mod_type == "change_output_layer":
                    new_classes = int(mod_params.get('new_classes', 10))
                    current_model, desc = _change_output_layer(current_model, new_classes)
                    modifications_log.append(desc)
                
                # --- ADD DROPOUT ---
                elif mod_type == "add_dropout":
                    dropout_rate = float(mod_params.get('rate', 0.5))
                    current_model, desc = _add_dropout(current_model, dropout_rate)
                    modifications_log.append(desc)
                
                # --- CHANGE INPUT SHAPE ---
                elif mod_type == "change_input_shape":
                    new_shape = mod_params.get('new_shape')
                    if isinstance(new_shape, str):
                        new_shape = eval(new_shape)  # Es: "(64, 64, 3)"
                    current_model, desc = _change_input_shape(current_model, new_shape, i)
                    modifications_log.append(desc)
                
                # --- ADD RESIZE PREPROCESSING ---
                elif mod_type == "add_resize_preprocessing":
                    height = int(mod_params.get('height', 64))
                    width = int(mod_params.get('width', 64))
                    current_model, desc = _add_resize_preprocessing(current_model, height, width)
                    modifications_log.append(desc)
                
                # --- CHANGE LEARNING RATE ---
                elif mod_type == "change_learning_rate":
                    lr = float(mod_params.get('learning_rate', 0.0001))
                    if lr <= 0:
                        raise ValueError(f"Learning rate deve essere > 0, ricevuto {lr}")
                    state['custom_learning_rate'] = lr
                    modifications_log.append(f"Custom learning rate: {lr}")
                    logger.info(f" ✓ Learning rate impostato a {lr}")
                
                # --- UNKNOWN TYPE ---
                else:
                    logger.warning(f" ⚠️ Tipo modifica sconosciuto: {mod_type}")
                    modifications_log.append(f"SKIPPED: unknown type '{mod_type}'")
            
            except Exception as mod_err:
                logger.error(f" ❌ Errore applicando {mod_type}: {str(mod_err)}")
                modifications_log.append(f"FAILED: {mod_type} - {str(mod_err)}")
                # Continua con la prossima modifica
        
        # ===== SALVA MODELLO CUSTOMIZZATO =====
        cache_dir = os.path.expanduser("~/.stm32_ai_models/temp")
        os.makedirs(cache_dir, exist_ok=True)
        
        # Usa .keras per evitare problemi legacy di .h5
        temp_filename = f"customized_{datetime.now().strftime('%Y%m%d_%H%M%S')}.keras"
        temp_path = os.path.join(cache_dir, temp_filename)
        
        saved_format = 'keras'
        try:
            current_model.save(temp_path, save_format='keras')
            logger.info(f"✓ Modello salvato (.keras): {temp_path}")
        
        except Exception as save_err:
            logger.warning(f"⚠️ .keras save failed: {str(save_err)[:80]}")
            
            # Fallback a .h5
            temp_path = temp_path.replace('.keras', '.h5')
            saved_format = 'h5'
            try:
                current_model.save(temp_path, save_format='h5')
                logger.info(f"✓ Modello salvato (.h5): {temp_path}")
            
            except Exception as h5_err:
                logger.error(f"❌ Entrambi i formati hanno fallito")
                raise RuntimeError(f"Save failed: keras={str(save_err)}, h5={str(h5_err)}")
        
        # ===== POPOLA STATE =====
        total_params = current_model.count_params()
        trainable_params = sum(
            tf.keras.backend.count_params(w) 
            for w in current_model.trainable_weights
        )
        
        state['customized_model_path'] = temp_path
        state['customization_applied'] = True
        state['error_message'] = None
        state['customized_model_info'] = {
            "input_shape": str(current_model.input_shape),
            "output_shape": str(current_model.output_shape),
            "total_params": int(total_params),
            "trainable_params": int(trainable_params),
            "frozen_params": int(total_params - trainable_params),
            "modifications_applied": modifications_log,
            "num_modifications": len(modifications_log),
            "save_format": saved_format,
            "timestamp": datetime.now().isoformat(),
            "original_model": state.get('selected_model', {}).get('name', 'unknown')
        }
        
        logger.info(f"✅ Customizzazioni completate!")
        logger.info(f" - Modifiche applicate: {len(modifications_log)}")
        logger.info(f" - Parametri totali: {total_params:,}")
        logger.info(f" - Parametri trainable: {trainable_params:,}")
        logger.info(f" - Salvato in: {temp_path}")
    
    except Exception as e:
        logger.error(f"❌ Errore generale: {str(e)}", exc_info=True)
        state['customization_applied'] = False
        state['customized_model_path'] = None
        state['customized_model_info'] = None
        state['error_message'] = str(e)
    
    return state


# Problema - Dal log:
# "Unrecognized keyword arguments passed to DepthwiseConv2D: {'groups': 1}"
# Succede perché:
# -Il modello STM32 usa DepthwiseConv2D con parametro groups=1
# -La tua versione Keras 3 NON riconosce groups in DepthwiseConv2D
# -Keras di default (safe_mode=True) è RIGIDO e blocca tutto ciò che non capisce
# -Con safe_mode=False Keras è PIÙ PERMISSIVO e prova a caricare comunque. Consente deserializzazione di DepthwiseConv2D con groups anche se sconosciuto e permette layer personalizzati di STM32.

# con safe_mode=False dice : OK: è STM32 ufficiale, non malevolo
# però non va lo stesso...

def fine_tune_customized_model(state: MasterState, config: dict) -> MasterState:
    """Esegue il fine-tuning del modello customizzato"""
    
    logger.info("🎓 Iniziando fine-tuning...")
    
    try:
        model = tf.keras.models.load_model(state.customized_model_path, compile=False)
        
        training_rec = state.parsed_modifications.get('training_recommendation', {})
        learning_rate = training_rec.get('learning_rate', state.custom_learning_rate or 0.0001)
        epochs = training_rec.get('epochs', state.custom_epochs or 10)
        batch_size = training_rec.get('batch_size', state.custom_batch_size or 32)
        
        logger.info(f"  Training params: LR={learning_rate}, epochs={epochs}, batch_size={batch_size}")
        
        X_train, y_train = load_or_create_sample_dataset(
            num_samples=200,
            img_size=(32, 32),
            num_classes=int(model.output_shape[-1])
        )
        
        X_val, y_val = load_or_create_sample_dataset(
            num_samples=50,
            img_size=(32, 32),
            num_classes=int(model.output_shape[-1])
        )
        
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        train_datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            zoom_range=0.1
        )
        
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1
        )
        
        logger.info(f"  Allenando per {epochs} epoche...")
        
        history = model.fit(
            train_datagen.flow(X_train, y_train, batch_size=batch_size),
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )
        
        state.training_test_result = {
            "success": True,
            "final_accuracy": float(history.history['accuracy'][-1]),
            "final_val_accuracy": float(history.history['val_accuracy'][-1]),
            "final_loss": float(history.history['loss'][-1]),
            "final_val_loss": float(history.history['val_loss'][-1]),
            "epochs_trained": len(history.history['loss']),
            "history": {k: [float(v) for v in vals] for k, vals in history.history.items()}
        }
        
        state.training_validation_success = True
        
        logger.info(f"✓ Fine-tuning completato!")
        logger.info(f"  Final accuracy: {state.training_test_result['final_accuracy']:.2%}")
        logger.info(f"  Final val accuracy: {state.training_test_result['final_val_accuracy']:.2%}")
        
        state.customized_model_path = state.customized_model_path.replace('_temp.h5', '_finetuned.h5')
        model.save(state.customized_model_path)
        
    except Exception as e:
        logger.error(f"❌ Errore fine-tuning: {str(e)}", exc_info=True)
        state.training_validation_success = False
        state.training_test_result = {
            "success": False,
            "error": str(e)
        }
    
    return state


def validate_customized_model(state: MasterState, config: dict) -> MasterState:
    """Valida il modello customizzato"""
    
    logger.info("✅ Validando modello customizzato...")
    
    try:
        model = tf.keras.models.load_model(state.customized_model_path, compile=False)
        
        logger.info("\n" + "="*60)
        logger.info("MODEL ARCHITECTURE AFTER CUSTOMIZATION")
        logger.info("="*60)
        model.summary(print_fn=logger.info)
        logger.info("="*60 + "\n")
        
        state.customized_model_info.update({
            "input_shape": str(model.input_shape),
            "output_shape": str(model.output_shape),
            "total_params": int(model.count_params()),
            "model_size_mb": round(os.path.getsize(state.customized_model_path) / (1024*1024), 2)
        })
        
        logger.info("✓ Modello validato con successo")
        
    except Exception as e:
        logger.error(f"❌ Errore validazione: {str(e)}")
    
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


def save_customized_model_final(state: MasterState, config: dict) -> MasterState:
    """Salva il modello customizzato definitivamente"""
    
    logger.info("💾 Salvando modello customizzato definitivamente...")
    
    try:
        output_dir = os.path.expanduser("~/.stm32_ai_models/customized")
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_path = os.path.join(output_dir, f"customized_final_{timestamp}.h5")
        
        shutil.copy(state.customized_model_path, final_path)
        
        state.final_model_path = final_path
        
        metadata = {
            **state.customized_model_info,
            "training_results": state.training_test_result,
            "quantization_applied": state.should_quantize,
            "quantized_model_path": state.quantized_model_path or None,
            "user_request": state.user_custom_modifications,
            "modifications_parsed": state.parsed_modifications.get('summary', ''),
        }
        
        save_model_with_metadata(final_path, final_path, metadata)
        
        logger.info(f"✓ Modello salvato: {final_path}")
        
    except Exception as e:
        logger.error(f"❌ Errore salvataggio: {str(e)}")
    
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

