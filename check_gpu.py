import sys
import os
import tensorflow as tf

# --- FIX AUTOMATICO CONDA ---
# Aggiunge le librerie di questo environment al path del sistema
conda_lib_path = os.path.join(sys.prefix, 'lib')
if os.path.exists(conda_lib_path):
    # Aggiorna env var per il processo corrente
    # Nota: per TensorFlow a volte serve impostarlo PRIMA di importare tf,
    # ma proviamo a farlo dinamicamente qui.
    current_ld = os.environ.get('LD_LIBRARY_PATH', '')
    if conda_lib_path not in current_ld:
        os.environ['LD_LIBRARY_PATH'] = f"{conda_lib_path}:{current_ld}"
        print(f"🔧 Autoconfigurazione: Aggiunto {conda_lib_path} a LD_LIBRARY_PATH")
        
        # Trucco: Riavvia lo script con le nuove variabili d'ambiente se necessario
        # (TensorFlow legge le variabili all'avvio, modificarle dopo import tf spesso non basta)
        if 'RESTARTED_WITH_LD' not in os.environ:
             print("🔄 Riavvio script per applicare le modifiche...")
             os.environ['RESTARTED_WITH_LD'] = 'true'
             try:
                 os.execv(sys.executable, [sys.executable] + sys.argv)
             except Exception as e:
                 print(f"⚠️ Fallito riavvio automatico: {e}")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Meno log di C++
print("----------------------------------------------------------------")
print(f"Python Executable: {sys.executable}")
print(f"TensorFlow Version: {tf.__version__}")
print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'Not Set')}")
try:
    build_info = tf.sysconfig.get_build_info()
    print(f"TF Build Info: cuda_version={build_info.get('cuda_version', '?')} cudnn_version={build_info.get('cudnn_version', '?')}")
except Exception:
    print("TF Build Info: N/A")
print("----------------------------------------------------------------")

gpus = tf.config.list_physical_devices('GPU')

if gpus:
    print(f"✅ SUCCESSO! Trovate {len(gpus)} GPU:")
    for i, gpu in enumerate(gpus):
        print(f"  [{i}] {gpu.device_type}: {gpu.name}")
        
    try:
        # Test configurazione memoria
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("\n🔧 Memory Growth impostata correttamente.")
    except RuntimeError as e:
        print(f"\n⚠️ Errore impostazione memoria: {e}")
else:
    print("❌ ATTENZIONE: Nessuna GPU rilevata.")
    print("Il codice girerà su CPU (molto più lento).")
    print("\nPossibili cause:")
    print("1. Driver NVIDIA non installati o versione errata.")
    print("2. CUDA/cuDNN non compatibili con questa versione di TensorFlow.")
    print("3. Variabile ambiente CUDA_VISIBLE_DEVICES nascosta.")

print("----------------------------------------------------------------")
