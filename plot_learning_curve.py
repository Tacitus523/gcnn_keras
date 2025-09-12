#!/usr/bin/env python3
"""
Standalone script to plot training curves from pickled histories.

Usage:
    python learning_curve.py --histories histories.pkl --dataset_name "MyDataset" --model_name "HDNNP"
"""

import argparse
import pickle
import os
from typing import List, Any

from kgcnn.utils.plots import plot_train_test_loss

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot training curves from pickled histories",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--histories", "-H",
        type=str,
        required=True,
        help="Path to pickled histories file (e.g., histories.pkl)"
    )
    
    parser.add_argument(
        "--dataset_name", "-d",
        type=str,
        default="Dataset",
        help="Name of the dataset for plot labels"
    )
    
    parser.add_argument(
        "--model_name", "-m",
        type=str,
        default="Model",
        help="Name of the model for plot labels"
    )
    
    parser.add_argument(
        "--data_unit", "-u",
        type=str,
        default="eV",
        help="Unit of the loss data"
    )
    
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        default="",
        help="Directory to save the plot (default: current directory)"
    )
    
    parser.add_argument(
        "--file_name", "-f",
        type=str,
        default="loss.png",
        help="Output filename for the plot"
    )
    
    parser.add_argument(
        "--show_plot", "-s",
        action="store_true",
        help="Show the plot interactively (default: False)"
    )
    
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=[16, 9],
        help="Figure size as width height (default: 10 6)"
    )
    
    parser.add_argument(
        "--dpi",
        type=float,
        default=300,
        help="DPI for saved figure"
    )
    
    return parser.parse_args()


def load_pickled_histories(filepath: str) -> List[Any]:
    """Load pickled histories from file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Histories file not found: {filepath}")
    
    with open(filepath, "rb") as f:
        histories = pickle.load(f)
    
    print(f"Loaded {len(histories)} training histories from {filepath}")
    return histories


def main():
    """Main function to create learning curve plots."""
    args = parse_arguments()
    
    # Load histories
    try:
        histories = load_pickled_histories(args.histories)
    except Exception as e:
        print(f"Error loading histories: {e}")
        return 1

    # Create output directory if it doesn't exist
    if args.output_dir and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created output directory: {args.output_dir}")

    # Plot training curves
    try:
        fig = plot_train_test_loss(
            histories=histories,
            data_unit=args.data_unit,
            model_name=args.model_name,
            dataset_name=args.dataset_name,
            filepath=args.output_dir,
            file_name=args.file_name,
            figsize=args.figsize,
            dpi=args.dpi,
            show_fig=args.show_plot
        )
        
        output_path = os.path.join(args.output_dir, f"{args.model_name}_{args.dataset_name}_{args.file_name}")
        print(f"Training curve plot saved to: {output_path}")
        
    except Exception as e:
        print(f"Error creating plot: {e}")
        return 1
    
    print("Learning curve plot completed successfully!")
    return 0


if __name__ == "__main__":
    exit(main())
