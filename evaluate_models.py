#!/usr/bin/env python

import argparse
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pickle
import tensorflow as tf

from kgcnn.data.transform.scaler.force import EnergyForceExtensiveLabelScaler
from kgcnn.utils.devices import set_devices_gpu

# Import model-specific modules
try:
    from force_hdnnp2nd import load_data as load_data_hdnnp2nd, evaluate_model as evaluate_model_hdnnp2nd, BASE_MODEL_CONFIG as hdnnp2nd_base_config
    from force_hdnnp4th import load_data as load_data_hdnnp4th, evaluate_model as evaluate_model_hdnnp4th, BASE_MODEL_CONFIG as hdnnp4th_base_config
    from force_schnet import load_data as load_data_schnet, evaluate_model as evaluate_model_schnet, BASE_MODEL_CONFIG as schnet_base_config
    from force_painn import load_data as load_data_painn, evaluate_model as evaluate_model_painn, BASE_MODEL_CONFIG as painn_base_config
except ImportError as e:
    print(f"Warning: Could not import model modules: {e}")

EVAL_BATCH_SIZE = 32  # Batch size for evaluation

BASE_MODEL_CONFIGS = {
    'hdnnp2nd': hdnnp2nd_base_config,
    'hdnnp4th': hdnnp4th_base_config,
    'schnet': schnet_base_config,
    'painn': painn_base_config,
}

# Model type mappings
MODEL_LOADERS = {
    'hdnnp2nd': load_data_hdnnp2nd,
    'hdnnp4th': load_data_hdnnp4th,
    'schnet': load_data_schnet,
    'painn': load_data_painn,
}

MODEL_EVALUATORS = {
    'hdnnp2nd': evaluate_model_hdnnp2nd,
    'hdnnp4th': evaluate_model_hdnnp4th,
    'schnet': evaluate_model_schnet, 
    'painn': evaluate_model_painn,
}

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate multiple trained models"
    )
    parser.add_argument(
        "-m", "--models",
        nargs="+",
        required=True,
        help="Paths to model directories or files"
    )
    parser.add_argument(
        "-t", "--type",
        choices=['painn', 'schnet', 'hdnnp2nd', 'hdnnp4th'],
        required=False,
        default='hdnnp4th',
        help="Type of models to evaluate"
    )
    parser.add_argument(
        "-d", "--data_directory",
        required=True,
        help="Path to dataset directory containing the .kgcnn.pickle file"
    )
    parser.add_argument(
        "--dataset_name",
        required=True,
        help="Prefix of the dataset"
    ) 
    parser.add_argument(
        "--scaler",
        default=None,
        help="Path to scaler file"
    )
    parser.add_argument(
        "--indices",
        default=None,
        help="Path to indices file"
    )
    parser.add_argument(
        "--output_dir",
        default=os.getcwd(),
        help="Output directory for evaluation results"
    )
    parser.add_argument(
        "-g", "--gpuid",
        type=int,
        default=None,
        help="GPU ID to use for evaluation"
    )
    
    return parser.parse_args()

def load_indices(indices_path: Optional[str]) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Load training indices."""
    if indices_path is None:
        return None
    if not os.path.exists(indices_path):
        raise FileNotFoundError(f"Indices file not found at {indices_path}")
    
    with open(indices_path, 'rb') as f:
        indices: Dict[str, List[np.ndarray,]|np.ndarray] = pickle.load(f)

    assert isinstance(indices, dict), "Indices file must contain a dictionary"
    assert all(key in indices for key in ['train', 'val', 'test']), "Indices dictionary must contain 'train', 'val', and 'test' keys"
    
    test_indices = indices['test']

    return (np.array([]), np.array([]), test_indices) # Return only test indices for evaluation

def load_scaler(scaler_path: Optional[str]) -> EnergyForceExtensiveLabelScaler:
    """Load scaler based on model type."""
    if scaler_path is None:
        return None
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found at {scaler_path}")
    
    scaler = EnergyForceExtensiveLabelScaler()
    scaler.load(scaler_path)
    return scaler

def evaluate_single_model(
    model_path: str,
    model_type: str,
    dataset,
    indices: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    scaler,
    output_dir: str,
    model_index: int
) -> None:
    """Evaluate a single model."""
    print(f"\n{'='*60}")
    print(f"Evaluating model {model_index + 1}: {model_path}")
    print(f"{'='*60}")
    
    # Load the model
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"Successfully loaded model from {model_path}")
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return
    
    # Create model config for evaluation
    model_config = BASE_MODEL_CONFIGS.get(model_type, {}).copy()
    if not model_config:
        print(f"Error: No base model config found for model type '{model_type}'")
        return
    
    # Create train config
    train_config = {
        "energy_batch_size": EVAL_BATCH_SIZE,
        "use_wandb": False,
        "use_scaler": scaler is not None
    }

    # Change to output directory for evaluation
    original_cwd = os.getcwd()
    os.chdir(output_dir)
    
    # Get the appropriate evaluator
    evaluator = MODEL_EVALUATORS[model_type]
    
    # Run evaluation
    evaluator(
        dataset=dataset,
        model_energy_force=model,
        indices=indices if indices else (np.array([]), np.array([]), np.arange(len(dataset))),
        model_config=model_config,
        train_config=train_config,
        scaler=scaler,
        model_index=model_index
    )
    
    print(f"Evaluation completed for model {model_index + 1}")
    
    os.chdir(original_cwd)

def main():
    """Main function."""
    args = parse_arguments()
    
    print("="*80)
    print("MULTI-MODEL EVALUATION SCRIPT")
    print("="*80)
    print(f"Model type: {args.type}")
    print(f"Number of models: {len(args.models)}")
    print(f"Data directory: {args.data_directory}")
    print(f"Dataset name: {args.dataset_name}")
    print(f"Scaler path: {args.scaler}")
    print(f"Indices path: {args.indices}")
    print(f"Output directory: {args.output_dir}")
    
    # Setup GPU
    set_devices_gpu(args.gpuid)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load dataset
    print(f"\nLoading dataset {args.dataset_name} from {args.data_directory}...")
    loader = MODEL_LOADERS.get(args.type)
    if loader is None:
        print(f"Error: No data loader found for model type '{args.type}'")
        sys.exit(1)
    
    # Create a minimal config for data loading
    config = {"data_directory": args.data_directory, "dataset_name": args.dataset_name}
    dataset = loader(config)
    print(f"Successfully loaded dataset with {len(dataset)} data points")

    # Load indices
    indices = load_indices(args.indices)
    
    # Load scaler
    scaler = load_scaler(args.scaler)
    if scaler:
        scaler.transform_dataset(dataset) # Evaluate includes inverse transform
    
    # Evaluate each model
    for i, model_path in enumerate(args.models):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path does not exist: {model_path}")
            
        evaluate_single_model(
            model_path=model_path,
            model_type=args.type,
            dataset=dataset,
            indices=indices,
            scaler=scaler,
            output_dir=args.output_dir,
            model_index=i
        )
    
    print(f"\n{'='*80}")
    print("EVALUATION COMPLETED")
    print(f"{'='*80}")
    print(f"Results saved in: {os.path.abspath(args.output_dir)}")

if __name__ == "__main__":
    main()