
import sys
from unittest.mock import MagicMock

# MOCK langgraph to avoid ImportErrors
sys.modules["langgraph"] = MagicMock()
sys.modules["langgraph.types"] = MagicMock()
sys.modules["langgraph.graph"] = MagicMock()

# Now we can import our modules
from src.assistant.state import MasterState
from src.assistant.workflow7_dataset import download_dataset
import os
import shutil

def test_download():
    print("🧪 Testing Dataset Download Logic...")
    
    # Setup State
    state = MasterState()
    state.base_dir = os.getcwd()
    state.real_dataset_name = "mnist" # Small dataset for testing
    state.dataset_source = "real"
    
    # Clean previous run
    target_dir = os.path.join(state.base_dir, "data", "real_datasets", "mnist")
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
        
    # Run Download
    try:
        print(f"  ⬇️  Downloading {state.real_dataset_name}...")
        state = download_dataset(state, {})
        
        # Verify
        if os.path.exists(state.real_dataset_path):
            print(f"  ✓ Dataset dir created: {state.real_dataset_path}")
            files = os.listdir(state.real_dataset_path)
            print(f"  ✓ Files found: {files}")
            
            if "x_train.npy" in files:
                print("  ✓ SUCCESS: x_train.npy found!")
            else:
                print("  ❌ FAILURE: x_train.npy missing")
        else:
            print("  ❌ FAILURE: Directory not created")
            
    except Exception as e:
        print(f"  ❌ EXCEPTION: {e}")

if __name__ == "__main__":
    test_download()
