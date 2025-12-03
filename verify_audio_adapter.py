import os
import numpy as np
import tensorflow as tf
import shutil
from src.assistant.workflow7_dataset import process_speech_commands, audio_to_spectrogram

def create_dummy_wav(filename, duration_sec=1.0, sample_rate=16000):
    """Crea un file WAV dummy (sinusoide)"""
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec))
    audio = np.sin(2 * np.pi * 440 * t) # 440Hz sine
    audio = audio.astype(np.float32)
    audio = np.expand_dims(audio, axis=-1)
    
    # Encode to WAV
    wav_encoded = tf.audio.encode_wav(audio, sample_rate)
    tf.io.write_file(filename, wav_encoded)

def test_audio_adapter():
    base_dir = "test_audio_adapter_output"
    extract_dir = os.path.join(base_dir, "extracted")
    output_dir = os.path.join(base_dir, "output")
    
    # Cleanup
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Setup dummy structure: extracted/speech_commands_v0.02/yes/file1.wav
    root_data = os.path.join(extract_dir, "speech_commands_v0.02")
    classes = ["yes", "no", "up"]
    
    print(f"🛠️ Creating dummy dataset in {root_data}...")
    
    for cls in classes:
        cls_dir = os.path.join(root_data, cls)
        os.makedirs(cls_dir, exist_ok=True)
        # Create 5 dummy wavs per class
        for i in range(5):
            create_dummy_wav(os.path.join(cls_dir, f"sample_{i}.wav"))
            
    # 2. Run Processing
    print("🚀 Running process_speech_commands...")
    process_speech_commands(extract_dir, output_dir, target_shape=(32, 32))
    
    # 3. Verify Output
    x_path = os.path.join(output_dir, "x_train.npy")
    y_path = os.path.join(output_dir, "y_train.npy")
    classes_path = os.path.join(output_dir, "classes.json")
    
    if os.path.exists(x_path) and os.path.exists(y_path) and os.path.exists(classes_path):
        X = np.load(x_path)
        y = np.load(y_path)
        print(f"✅ Output files found!")
        print(f"  X shape: {X.shape} (Expected: (15, 32, 32, 1))")
        print(f"  y shape: {y.shape} (Expected: (15,))")
        
        if X.shape == (15, 32, 32, 1):
            print("✅ Shape verification PASSED")
        else:
            print("❌ Shape verification FAILED")
            
        # Check normalization
        print(f"  X range: [{X.min():.2f}, {X.max():.2f}]")
        if X.min() >= 0 and X.max() <= 1.0:
             print("✅ Normalization verification PASSED")
        else:
             print("❌ Normalization verification FAILED")
             
    else:
        print("❌ Output files MISSING")

if __name__ == "__main__":
    test_audio_adapter()
