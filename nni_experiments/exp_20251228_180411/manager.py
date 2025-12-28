
import nni
import os
import sys
import json
import subprocess
from nni.experiment import Experiment

# Get absolute path to current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Define search space
search_space = {
    'learning_rate': {'_type': 'choice', '_value': [0.001, 0.0001, 0.00001]},
    'batch_size': {'_type': 'choice', '_value': [16, 32]},
    'optimizer': {'_type': 'choice', '_value': ['Adam', 'SGD']},
    'freeze_mode': {'_type': 'choice', '_value': ['freeze_base', 'train_all']},
}

# Create experiment
experiment = Experiment('local')
# Use the same Python interpreter for trials
experiment.config.trial_command = f'{sys.executable} trial.py'
experiment.config.trial_code_directory = current_dir  # Absolute path
experiment.config.search_space = search_space
experiment.config.tuner.name = 'TPE'
experiment.config.tuner.class_args = {'optimize_mode': 'maximize'}
experiment.config.max_trial_number = 6
experiment.config.trial_concurrency = 3  # Run 3 trials concurrently

# Run with error handling
try:
    print(f"[NNI] Starting experiment in {current_dir}")
    print(f"[NNI] Web UI will be available at http://localhost:8080")
    experiment.run(port=8080, wait_completion=True)
    print("[NNI] Experiment completed successfully")

    # --- AUTO-RETRAIN BEST MODEL ---
    print("\n[NNI] 🏆 Exporting best trial...")
    best_trial = experiment.export_top_trial(top_k=1)[0]
    best_params = best_trial.parameter
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
    experiment.stop()
