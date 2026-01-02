
import nni
import os
import os
import sys
import json
import subprocess
from nni.experiment import Experiment

# FORCE LOAD CUDA LIBRARIES (Fix for Conda Envs)
conda_lib_path = os.path.join(sys.prefix, 'lib')
if os.path.exists(conda_lib_path):
    os.environ['LD_LIBRARY_PATH'] = f"{conda_lib_path}:"
    print(f"[NNI] 🔧 Added Conda lib to LD_LIBRARY_PATH: {conda_lib_path}")

# Get absolute path to current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Define search space
search_space = {
    'learning_rate': {'_type': 'choice', '_value': [0.001, 0.0001, 0.00001]},
    'batch_size': {'_type': 'choice', '_value': [16, 32, 64]},
    'optimizer': {'_type': 'choice', '_value': ['Adam', 'SGD']},
}

# Create experiment
experiment = Experiment('local')
# CRITICAL: Use same Python interpreter for trials
experiment.config.trial_command = f'{sys.executable} trial.py'
experiment.config.trial_code_directory = current_dir  # Use absolute path
experiment.config.search_space = search_space
experiment.config.tuner.name = 'TPE'
experiment.config.tuner.class_args = {'optimize_mode': 'maximize'}
experiment.config.max_trial_number = 6
experiment.config.trial_concurrency = 1 # Reduced to 1 to prevent GPU OOM
experiment.config.training_service.use_active_gpu = True # Enable GPU Usage

# Run with error handling
try:
    print(f"[NNI] Starting experiment in {current_dir}")
    print(f"[NNI] Web UI will be available at http://localhost:8080")
    experiment.run(port=8080, wait_completion=True)
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
    
    best_params = best_trial.hyperParameters['parameters']
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
    # Keep server alive for debugging
    print("\n[NNI] 🛑 Experiment flow finished.")
    print("[NNI] Press Enter to stop the NNI Web UI and exit...")
    input()
