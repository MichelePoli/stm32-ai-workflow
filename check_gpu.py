import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Meno log di C++

print("----------------------------------------------------------------")
print(f"TensorFlow Version: {tf.__version__}")
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
