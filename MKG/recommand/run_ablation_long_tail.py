import os
import argparse
import sys
import torch

from train import main as original_train_main
from train import Config, USE_CHEM_FINGERPRINT, USE_DISEASE_TEXT, USE_CROSS_MODAL

import evaluate_long_tail

def run_ablation(ablation_type):
    # Hack train.py flags globally
    import train
    
    if ablation_type == 'no_chem':
        train.USE_CROSS_MODAL = False
        train.USE_CHEM_FINGERPRINT = False
        ckpt_name = 'best_model_no_chem.pt'
        print(">>> Ablation: Removed all Chemical Modality Features.")
    elif ablation_type == 'no_text':
        train.USE_DISEASE_TEXT = False
        ckpt_name = 'best_model_no_text.pt'
        print(">>> Ablation: Removed Disease Text Modality.")
    elif ablation_type == 'no_modal':
        train.USE_CROSS_MODAL = False
        train.USE_CHEM_FINGERPRINT = False
        train.USE_DISEASE_TEXT = False
        ckpt_name = 'best_model_no_modal.pt'
        print(">>> Ablation: Removed ALL Modalities (Structure only).")
    else:
        print("Unknown ablation type")
        return
    
    # Temporarily change the model save path so we don't overwrite the original model
    train.Config.MODEL_SAVE_PATH = os.path.join(train.Config.CURRENT_DIR, 'checkpoints', ckpt_name)
    # We monkey-patch the original script's standard saving location (we'll just replace the checkpoint file naming in config)
    # Actually wait, train.py saves to `os.path.join(Config.MODEL_SAVE_PATH, 'best_model.pt')`.
    # Let's override `Config.MODEL_SAVE_PATH` to save inside a new dir.
    train.Config.MODEL_SAVE_PATH = os.path.join(train.Config.CURRENT_DIR, 'checkpoints', f'ablation_{ablation_type}')
    os.makedirs(train.Config.MODEL_SAVE_PATH, exist_ok=True)
    
    print(f"\n==============================================")
    print(f"Starting Training for Ablation: {ablation_type}")
    print(f"==============================================\n")
    
    # Run the original train main loop
    train.main()
    
    print(f"\n==============================================")
    print(f"Training completed. Evaluating Long-Tail...")
    print(f"==============================================\n")
    
    # Now run the evaluation matching the ablation config
    import evaluate_long_tail
    evaluate_long_tail.USE_CROSS_MODAL = train.USE_CROSS_MODAL
    evaluate_long_tail.USE_CHEM_FINGERPRINT = train.USE_CHEM_FINGERPRINT
    evaluate_long_tail.USE_DISEASE_TEXT = train.USE_DISEASE_TEXT
    evaluate_long_tail.USE_CHEM_DENSE = True # Model ALWAYS expects attr tensor config logic
    
    # Override the loading path in evaluate
    evaluate_long_tail.Config.MODEL_SAVE_PATH = train.Config.MODEL_SAVE_PATH
    
    evaluate_long_tail.main()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, choices=['no_chem', 'no_text', 'no_modal'], required=True)
    args = parser.parse_args()
    run_ablation(args.type)
