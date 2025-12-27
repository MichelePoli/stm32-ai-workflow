
import nni
import os
import sys
from nni.experiment import Experiment

# Get absolute path to current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Define search space
search_space = {
    'learning_rate': {'_type': 'choice', '_value': [0.001, 0.0001, 0.00001]},
    'batch_size': {'_type': 'choice', '_value': [16, 32, 64]},
}

# Create experiment
experiment = Experiment('local')
# CRITICAL: Use same Python interpreter for trials
experiment.config.trial_command = f'{sys.executable} trial.py'
experiment.config.trial_code_directory = current_dir  # Use absolute path
experiment.config.search_space = search_space
experiment.config.tuner.name = 'TPE'
experiment.config.tuner.class_args = {'optimize_mode': 'maximize'}
experiment.config.max_trial_number = 10
experiment.config.trial_concurrency = 1

# Run with error handling
try:
    print(f"[NNI] Starting experiment in {current_dir}")
    print(f"[NNI] Web UI will be available at http://localhost:8080")
    experiment.run(port=8080, wait_completion=True)
    print("[NNI] Experiment completed successfully")
except Exception as e:
    print(f"[NNI] Error during experiment: {e}")
    import traceback
    traceback.print_exc()
finally:
    experiment.stop()
