#!/usr/bin/env python
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --job-name=plot
#SBATCH --output=plot.out
#SBATCH --error=plot.out

import argparse
import json
from typing import Dict, List, Optional, Union
from ase.atoms import Atoms
from ase.io import read
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error

# Default geometry file
GEOMS: str = "model_geoms.extxyz"
DATA_SOURCES_FILE: Optional[str] = None  # File containing the data source of each entry

# Keywords for extracting data
REF_ENERGY_KEY: Optional[str] = "ref_energy"
REF_FORCES_KEY: Optional[str] = "ref_forces"
REF_CHARGES_KEY: Optional[str] = "ref_charges"
PRED_ENERGY_KEY: Optional[str] = "pred_energy"
PRED_FORCES_KEY: Optional[str] = "pred_forces"
PRED_CHARGES_KEY: Optional[str] = "pred_charges"

# Units for plotting
ENERGY_UNIT: str = "eV"
FORCES_UNIT: str = r"$\frac{eV}{\AA}$"
CHARGES_UNIT: str = "e"

DPI = 100

# Conversion factors
H_to_eV: float = 27.2114
angstrom_to_bohr: float = 1.88973
bohr_to_angstrom: float = 1 / angstrom_to_bohr
H_B_to_ev_angstrom: float = H_to_eV / bohr_to_angstrom

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plotting script for model data")
    parser.add_argument(
        "-g",
        "--geoms",
        type=str,
        default=GEOMS,
        help="Path to a geometry file containing both reference and predicted data."
    )
    parser.add_argument(
        "-s",
        "--sources",
        type=str,
        default=DATA_SOURCES_FILE,
        help="Path to the data sources file, default: %s" % DATA_SOURCES_FILE,
    )
    parser.add_argument(
        "--plot-charges",
        action="store_true",
        default=False,
        help="Enable charge plotting (default: False). If not specified, charges will not be plotted even if available."
    )
    return parser.parse_args()

def extract_data(
    mols: List[Atoms],
    energy_keyword: Optional[str] = None,
    forces_keyword: Optional[str] = None,
    charges_keyword: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    ref_energy: List[float] = []
    ref_forces: List[float] = []
    ref_charges: List[float] = []
    ref_elements: List[str] = []
    for m in mols:
        if charges_keyword is not None and charges_keyword in m.arrays:
            ref_charges.extend(m.arrays[charges_keyword].flatten())
        if energy_keyword is not None:
            if energy_keyword == "energy":
                ref_energy.append(m.get_potential_energy())
            else:
                ref_energy.append(m.info[energy_keyword])
        if forces_keyword is not None:
            if forces_keyword == "forces":
                ref_forces.extend(m.get_forces().flatten())
            else:
                ref_forces.extend(m.arrays[forces_keyword].flatten())
        ref_elements.extend(m.get_chemical_symbols())
    result = {}
    result["energy"] = np.array(ref_energy) # eV #* H_to_eV # Convert Hartree to eV
    result["forces"] = np.array(ref_forces) # eV/Å #* H_B_to_ev_angstrom # Convert Hartree/Bohr to eV/Å
    result["charges"] = np.array(ref_charges) if ref_charges else None
    result["n_atoms"] = np.array([len(m) for m in mols])
    result["elements"] = np.array(ref_elements)
    return result

def create_metrics_collection(
    ref_data: Dict[str, np.ndarray],
    pred_data: Dict[str, np.ndarray],
    ) -> Dict[str, Dict[str, float]]:
    """
    Create metrics collection similar to the evaluate function in train.py
    
    Args:
        ref_data: Dictionary containing reference data
        pred_data: Dictionary containing predicted data
        
    Returns:
        Dictionary with metrics for train/test/val splits
    """
    
    metrics_collection: Dict[str, Dict[str, float]] = {"train": {}, "test": {}, "val": {}}
    
    # For now, put all metrics in "test" since we don't have train/val split info
    metrics = {}

    # Energy metrics
    if "energy" in ref_data and "energy" in pred_data:
        delta_e = ref_data["energy"] - pred_data["energy"]
        metrics["mae_e"] = mean_absolute_error(ref_data["energy"], pred_data["energy"])
        metrics["rmse_e"] = root_mean_squared_error(ref_data["energy"], pred_data["energy"])
        metrics["r2_e"] = r2_score(ref_data["energy"], pred_data["energy"])
        metrics["q95_e"] = np.percentile(np.abs(delta_e), 95)

        delta_e_per_atom = delta_e / ref_data["n_atoms"]
        metrics["mae_e_per_atom"] = np.mean(np.abs(delta_e_per_atom))
        metrics["rmse_e_per_atom"] = np.sqrt(np.mean(delta_e_per_atom**2))

    # Forces metrics
    if "forces" in ref_data and "forces" in pred_data:
        delta_f = ref_data["forces"] - pred_data["forces"]
        metrics["mae_f"] = mean_absolute_error(ref_data["forces"], pred_data["forces"])
        metrics["rmse_f"] = root_mean_squared_error(ref_data["forces"], pred_data["forces"])
        metrics["r2_f"] = r2_score(ref_data["forces"], pred_data["forces"])
        
        # Relative metrics
        metrics["q95_f"] = np.percentile(np.abs(delta_f), 95)

        f_norm = np.linalg.norm(ref_data["forces"])
        if f_norm > 0:
            metrics["rel_mae_f"] = np.mean(np.abs(delta_f)) / (np.mean(np.abs(ref_data["forces"])) + 1e-8) * 100
            metrics["rel_rmse_f"] = np.sqrt(np.mean(delta_f**2)) / (np.sqrt(np.mean(ref_data["forces"]**2)) + 1e-8) * 100

    # Charges metrics
    if "charges" in ref_data and "charges" in pred_data:
        delta_q = ref_data["charges"] - pred_data["charges"]
        metrics["mae_q"] = mean_absolute_error(ref_data["charges"], pred_data["charges"])
        metrics["rmse_q"] = root_mean_squared_error(ref_data["charges"], pred_data["charges"])
        metrics["r2_q"] = r2_score(ref_data["charges"], pred_data["charges"])
        metrics["q95_q"] = np.percentile(np.abs(delta_q), 95)
    
    # Store in test category (could be split into train/val/test if that info is available)
    metrics_collection["test"] = metrics
    
    # Save metrics to JSON file
    with open("evaluation_metrics.json", "w") as f:
        json.dump(metrics_collection, f, indent=2)
    print(f"\nMetrics saved to: mace_evaluation_metrics.json")

    return metrics_collection

def plot_data(
    ref_data: Dict[str, np.ndarray],
    pred_data: Dict[str, np.ndarray],
    key: str,
    sources: Optional[np.ndarray],
    x_label: str,
    y_label: str,
    unit: str,
    filename: str,
) -> None:
    """
    Create a scatter plot comparing reference and model values.
    
    Args:
        ref_data: Dictionary containing reference data
        pred_data: Dictionary containing predicted data
        key: Key to extract the specific data from dictionaries
        sources: Optional data sources for coloring
        x_label: Label for x-axis
        y_label: Label for y-axis
        unit: Unit for the data
        filename: Output filename for the plot
    """
    df: pd.DataFrame = create_dataframe(ref_data, pred_data, key, sources, x_label, y_label)
    rmse: float = root_mean_squared_error(df[x_label], df[y_label])
    r2: float = r2_score(df[x_label], df[y_label])

    print(f"RMSE for {key}: {rmse:.2f} {unit}")
    print(f"R² for {key}: {r2:.4f}")

    sns.set_context("talk")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=df,
        x=x_label,
        y=y_label,
        hue="source" if sources is not None else None,
        palette="tab10" if sources is not None else None,
        alpha=0.6,
        edgecolor=None,
        s=20,
    )
    plt.plot(ref_data[key], ref_data[key], color="black", label="_Identity Line")
    plt.xlabel(f"{x_label} ({unit})")
    plt.ylabel(f"{y_label} ({unit})")
    plt.text(
        0.70,
        0.25,
        f"RMSE: {rmse:.2f} {unit}\nR²: {r2:.4f}",
        transform=plt.gca().transAxes,
        fontsize=15,
        verticalalignment="top",
        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'),
    )
    # Improve legend if available
    if "source" in df.columns:
        plt.legend(title=None, loc="upper left", fontsize="small")
        for legend_handle in ax.get_legend().legend_handles:
            legend_handle.set_alpha(1)
    plt.tight_layout()
    plt.savefig(filename, dpi=DPI)
    plt.close()

def create_dataframe(
    ref_data: Dict[str, np.ndarray],
    pred_data: Dict[str, np.ndarray],
    key: str,
    sources: Optional[np.ndarray],
    x_label: str,
    y_label: str,
) -> pd.DataFrame:
    """
    Create a DataFrame from reference and predicted data.
    
    Args:
        ref_data: Dictionary containing reference data
        pred_data: Dictionary containing predicted data
        key: Key to extract the specific data from dictionaries
        sources: Optional data sources
        x_label: Label for x-axis
        y_label: Label for y-axis
    Returns:
        DataFrame containing the reference and predicted data
    """
    df = pd.DataFrame(
        {
            x_label: ref_data[key],
            y_label: pred_data[key],
        }
    )
    if sources is not None:
        assert len(ref_data[key]) % len(sources) == 0, "Number of sources does not match the number of data points"
        repetitions = len(ref_data[key]) // len(sources)
        df["source"] = np.repeat(sources, repetitions)
    return df

def main() -> None:
    args: argparse.Namespace = parse_args()
    molecules: List[Atoms] = read(args.geoms, format="extxyz", index=":")
    ref_data: Dict[str, np.ndarray] = extract_data(
        molecules, REF_ENERGY_KEY, REF_FORCES_KEY, REF_CHARGES_KEY
    )
    model_data: Dict[str, np.ndarray] = extract_data(
        molecules, PRED_ENERGY_KEY, PRED_FORCES_KEY, PRED_CHARGES_KEY
    )
    assert len(ref_data["energy"]) == len(molecules), "Number of reference data does not match the number of configurations"
    assert len(model_data["energy"]) == len(molecules), "Number of model data does not match the number of configurations"

    if args.sources is not None:
        with open(args.sources, "r") as f:
            sources: np.ndarray = np.array([line.strip() for line in f.readlines()])
        assert len(sources) == len(molecules), f"Number of sources does not match the number of configurations: {len(sources)} != {len(molecules)}"
    else:
        sources = None

    for name, data in zip(["Ref", "Model"], [ref_data, model_data]):
        for key, value in data.items():
            # Skip non-numeric data
            if isinstance(value, np.ndarray) and value.dtype in (np.float32, np.float64, np.int32, np.int64):
                print(f"{name} {key}: {value.shape} Min Max: {np.min(value): .1f} {np.max(value): .1f}")

    plot_data(
        ref_data,
        model_data,
        "energy",
        sources,
        "Ref Energy",
        "Model Energy",
        ENERGY_UNIT,
        "model_energy.png"
    )

    plot_data(
        ref_data,
        model_data,
        "forces",
        sources if sources is not None else ref_data["elements"],
        "Ref Forces",
        "Model Forces",
        FORCES_UNIT,
        "model_forces.png"
    )

    # Only plot charges if explicitly enabled and data is available
    if args.plot_charges and "charges" in ref_data and "charges" in model_data:
        plot_data(
            ref_data,
            model_data,
            "charges",
            sources if sources is not None else ref_data["elements"],
            "Ref Charges",
            "Model Charges",
            CHARGES_UNIT,
            "model_charges.png",
        )

if __name__ == "__main__":
    main()