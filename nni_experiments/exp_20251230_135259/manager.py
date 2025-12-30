
import nni
import os
import sys
import json
import subprocess
from nni.experiment import Experiment

current_dir = os.path.dirname(os.path.abspath(__file__))

search_space = {
    'learning_rate': {'_type': 'choice', '_value': [0.001, 0.0001, 0.00001]},
    'batch_size': {'_type': 'choice', '_value': [16, 32, 64]},
    'optimizer': {'_type': 'choice', '_value': ['Adam', 'SGD']},
    'freeze_mode': {'_type': 'choice', '_value': ['none', 'partial', 'full']},
}

experiment = Experiment('local')
experiment.config.trial_command = f'{sys.executable} trial.py'
experiment.config.trial_code_directory = current_dir
experiment.config.search_space = search_space
experiment.config.tuner.name = 'TPE'
experiment.config.tuner.class_args = {'optimize_mode': 'maximize'}
experiment.config.max_trial_number = 12
experiment.config.trial_concurrency = 3

try:
    print(f"[NNI] Starting experiment in {current_dir}")
    print("[NNI] Web UI will be available at http://localhost:8080")
    print("\n[NNI] Searching over the following hyper‑parameter space:")
    for k, v in search_space.items():
        print(f"  - {k}: {v['_value']}")

    print("\n[NNI] Beginning hyper‑parameter search...")
    model = None  # Placeholder to keep the variable local to the script
    experiment.run()
except Exception as e:
    print(f"[NNI] Unexpected error: {e}")
finally:
    if experiment._current_trial is not None:
        print("[NNI] Ending current trial")
        experiment._current_trial.stop()

    # Determine the best hyper‑parameters from the completed trials
    if experiment.completed_trials:
        best_trial = experiment.get_best_trial()
        print("\n[NNI] Hyper‑parameter search completed.")
        print(f"[NNI] Best trial: #{best_trial.number} with parameters:")
        print(f"  - learning_rate: {best_trial.hyperparameters.get('learning_rate')}")
        print(f"  - batch_size: {best_trial.hyperparameters.get('batch_size')}")
        print(f"  - optimizer: {best_trial.hyperparameters.get('optimizer')}")
        print(f"  - freeze_mode: {best_trial.hyperparameters.get('freeze_mode')}")
        print(f"[NNI] Best validation accuracy: {best_trial.final_metric}")
    else:
        print("[NNI] No trials were completed.")
