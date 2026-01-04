
import nni
import os
import sys
import json
import subprocess
from nni.experiment import Experiment

# FORCE LOAD CUDA LIBRARIES (Fix for Conda Envs)
conda_lib_path = os.path.join(sys.prefix, 'lib')
if os.path.exists(conda_lib_path):
    os.environ['LD_LIBRARY_PATH'] = f"{conda_lib_path}:{os.environ.get('LD_LIBRARY_PATH', '')}"
    print(f"[NNI] 🔧 Added Conda lib to LD_LIBRARY_PATH: {conda_lib_path}")

# Get absolute path to current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Define search space
search_space = {
    'learning_rate': {'_type': 'choice', '_value': [0.001, 0.0005, 0.0001]},
    'batch_size': {'_type': 'choice', '_value': [16, 32, 64]},
    'optimizer': {'_type': 'choice', '_value': ['Adam']},
    'freeze_backbone': {'_type': 'choice', '_value': [True, False]},
} # tolto False da freeze_backbone per testing più veloce. dopo rimetti. 

# Create experiment
experiment = Experiment('local')
# CRITICAL: Use same Python interpreter for trials
experiment.config.trial_command = f'{sys.executable} trial.py'
experiment.config.trial_code_directory = current_dir  # Use absolute path
experiment.config.search_space = search_space
experiment.config.tuner.name = 'Random'
experiment.config.tuner.class_args = {'optimize_mode': 'maximize'}
experiment.config.max_trial_number = 8 # REDUCED TO 2 FOR TESTING, usually 8 
experiment.config.trial_concurrency = 2 # REDUCED TO 1 TO SAVE RAM
experiment.config.training_service.use_active_gpu = True # Enable GPU Usage

# Run with error handling
try:
    print(f"[NNI] Starting experiment in {current_dir}")
    
    # --- PORT HUNTING ---
    import socket
    def find_free_port(start_port=8080, max_attempts=20):
        for port in range(start_port, start_port + max_attempts):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex(('localhost', port)) != 0:
                    return port
        return start_port

    target_port = find_free_port(8080)
    print(f"[NNI] Web UI will be available at http://localhost:{target_port}")
    
    experiment.run(port=target_port, wait_completion=True)
    print("[NNI] Experiment completed successfully")
    
    # --- AUTO-RETRAIN BEST MODEL ---
    print("\n[NNI] 🏆 Exporting best trial...")
    
    # Robust way to get best trial (export_top_trial might be missing)
    trials = experiment.list_trial_jobs()
    valid_trials = [t for t in trials if t.status == 'SUCCEEDED' and t.finalMetricData]
    
    if not valid_trials:
        raise Exception("No successful trials found.")
        
    # Sort by metric (assuming 'maximize' mode -> higher is better)
    # finalMetricData is a list of MetricData, we take the last one (or index 0 if only one)
    # data is usually a string, convert to float
    best_trial = max(valid_trials, key=lambda t: float(t.finalMetricData[0].data))
    
    # DEBUG: Inspect structure
    print(f"   • Raw hyperParameters type: {type(best_trial.hyperParameters)}")
    print(f"   • Raw hyperParameters value: {best_trial.hyperParameters}")
    sys.stdout.flush()
    # le print di debug appariranno subito.


    # Handle hyperParameters structure
    hp = best_trial.hyperParameters
    
    # 1. If list, take the latest one
    if isinstance(hp, list):
         latest_hp = hp[-1]
    else:
         latest_hp = hp
         
    # 2. Extract 'parameters' dict (handle Object vs Dict)
    if hasattr(latest_hp, 'parameters'):
        best_params = latest_hp.parameters
    elif isinstance(latest_hp, dict):
        best_params = latest_hp['parameters']
    else:
        # Fallback: assume latest_hp IS the params dict (very old NNI?)
        print(f"[NNI] Warning: Unknown HP structure. Type: {type(latest_hp)}")
        best_params = latest_hp
    print(f"   • Best Params: {best_params}")
    
    print("\n[NNI] 💾 Retraining best model for saving...")
    
    # Prepare environment for retrain
    env = os.environ.copy()
    env['RETRAIN_MODE'] = 'true'
    env['NNI_PARAMS'] = json.dumps(best_params)
    
    # Re-run trial.py with best params
    subprocess.run(
        [sys.executable, 'trial.py'],
        cwd=current_dir,
        env=env,
        check=True
    )
    
    print(f"[NNI] ✅ Best model saved to: {os.path.join(current_dir, 'best_model.h5')}")

except Exception as e:
    print(f"[NNI] Error during experiment: {e}")
    import traceback
    traceback.print_exc()

finally:
    # Stop experiment and exit
    print("\n[NNI] 🛑 Experiment flow finished. Stopping NNI...")
    experiment.stop()
