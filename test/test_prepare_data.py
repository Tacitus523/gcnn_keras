#!/usr/bin/env python3
"""Test cases for prepare_data.py functionality."""

import pytest
import numpy as np
import os
import tempfile
import shutil
from unittest.mock import patch, MagicMock
import pickle
from typing import Dict, Any

# Import the functions we want to test
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from prepare_data import (
    prepare_kgcnn_dataset, get_properties_from_extxyz, 
    get_energies, get_forces, get_charges, get_esps, get_esp_grads,
    read_forces_file, read_irregular_file, prepare_config
)


class TestPrepareData:
    """Test class for prepare_data.py functionality."""
    
    @pytest.fixture(autouse=True)
    def setup_test_environment(self):
        """Set up temporary directories and test files for each test."""
        self.test_dir = tempfile.mkdtemp()
        self.assets_dir = os.path.join(os.path.dirname(__file__), "assets")
        
        # Create test target folder
        self.target_folder = os.path.join(self.test_dir, "kgcnn_inputs_test")
        os.makedirs(self.target_folder, exist_ok=True)
        
        yield
        
        # Cleanup
        shutil.rmtree(self.test_dir)

    def test_extxyz_reading(self):
        """Test reading data from extxyz format files."""
        config_data = {
            "DATA_FOLDER": self.assets_dir,
            "EXTXYZ_FILE": "test_molecules.extxyz",
            "TARGET_FOLDER": self.target_folder,
            "PREFIX": "TestMolecules"
        }
        config = prepare_config(config_data)
        
        # Test reading properties from extxyz
        charges, total_charges, energies, forces, esps, esp_grads = get_properties_from_extxyz(config)
        
        # Verify that we get reasonable data
        assert charges is not None, "Should read charges from extxyz"
        assert total_charges is not None, "Should read total charges from extxyz"
        assert energies is not None, "Should read energies from extxyz"
        assert forces is not None, "Should read forces from extxyz"
        
        # Check data types and shapes
        assert len(charges) > 0, "Should have charge data for molecules"
        assert len(forces) > 0, "Should have force data for molecules"

    def test_individual_file_reading(self):
        """Test reading data from individual format files."""
        config_data = {
            "DATA_FOLDER": self.assets_dir,
            "GEOMETRY_FILE": "test_molecules.xyz",
            "ENERGY_FILE": "test_energies.txt",
            "FORCE_FILE": "test_forces.xyz",
            "CHARGE_FILE": "test_charges.txt",
            "ESP_FILE": "test_esp.txt",
            "ESP_GRAD_FILE": "test_esp_grad.xyz",
            "TARGET_FOLDER": self.target_folder,
            "PREFIX": "TestMolecules"
        }
        config = prepare_config(config_data)
        
        # Test individual file reading functions
        energies = get_energies(config)
        charges, total_charges = get_charges(config)
        forces = get_forces(config)
        esps = get_esps(config)
        esp_grads = get_esp_grads(config)
        
        # Verify data was read correctly
        assert energies is not None, "Should read energies from file"
        assert charges is not None, "Should read charges from file"
        assert total_charges is not None, "Should calculate total charges"
        assert forces is not None, "Should read forces from file"
        assert esps is not None, "Should read ESP values from file"
        assert esp_grads is not None, "Should read ESP gradients from file"
        
        # Check data consistency
        assert len(charges) == len(energies), "Number of charge arrays should match energies"
        assert len(forces) == len(energies), "Number of force arrays should match energies"
        assert len(total_charges) == len(energies), "Number of total charges should match energies"

    def test_file_reading_utilities(self):
        """Test utility functions for reading different file formats."""
        # Test read_irregular_file
        test_file = os.path.join(self.assets_dir, "test_charges.txt")
        charges = read_irregular_file(test_file, conversion_factor=1.0)
        
        assert isinstance(charges, list), "Should return list of arrays"
        assert len(charges) > 0, "Should read some charge data"
        assert all(isinstance(charge_array, np.ndarray) for charge_array in charges), "Each item should be numpy array"
        
        # Test read_forces_file
        forces_file = os.path.join(self.assets_dir, "test_forces.xyz")
        forces = read_forces_file(forces_file)
        
        assert isinstance(forces, list), "Should return list of force arrays"
        assert len(forces) > 0, "Should read some force data"
        assert all(isinstance(force_array, np.ndarray) for force_array in forces), "Each item should be numpy array"
        assert all(force_array.shape[1] == 3 for force_array in forces), "Forces should be 3D vectors"

    def test_error_handling(self):
        """Test error handling for missing files and invalid configurations."""
        config_data = {
            "DATA_FOLDER": self.assets_dir,
            "ENERGY_FILE": "nonexistent_file.txt",
            "GEOMETRY_FILE": "nonexistent_geom.xyz",
            "EXTXYZ_FILE": "nonexistent.extxyz",
            "TARGET_FOLDER": self.target_folder,
            "PREFIX": "TestMolecules"
        }
        config = prepare_config(config_data)
        
        # Test missing energy file
        with pytest.raises(FileNotFoundError):
            get_energies(config)
        
        # Test missing geometry file
        with pytest.raises(FileNotFoundError):
            prepare_kgcnn_dataset(config)

        # Test missing extxyz file
        with pytest.raises(FileNotFoundError):
            get_properties_from_extxyz(config)
        
        # Test config with no geometry source
        invalid_config_data = {
            "DATA_FOLDER": self.assets_dir,
            "TARGET_FOLDER": self.target_folder,
            "PREFIX": "TestMolecules"
        }
        invalid_config = prepare_config(invalid_config_data)
        
        with pytest.raises(ValueError):
            prepare_kgcnn_dataset(invalid_config)

    def test_prepare_kgcnn_dataset(self):
        """Test the main dataset preparation function."""
        config_data = {
            "DATA_FOLDER": self.assets_dir,
            "GEOMETRY_FILE": "test_molecules.xyz",
            "ENERGY_FILE": "test_energies.txt",
            "FORCE_FILE": "test_forces.xyz",
            "CHARGE_FILE": "test_charges.txt",
            "ESP_FILE": "test_esp.txt",
            "ESP_GRAD_FILE": "test_esp_grad.xyz",
            "TARGET_FOLDER": self.target_folder,
            "PREFIX": "TestMolecules"
        }
        config = prepare_config(config_data)
        
        # Run dataset preparation
        prepare_kgcnn_dataset(config)
        
        # Check that output files are created
        expected_files = [
            f"{config['PREFIX']}.csv",
            f"{config['PREFIX']}.sdf",
            f"{config['PREFIX']}.kgcnn.pickle"
        ]
        
        for filename in expected_files:
            file_path = os.path.join(self.target_folder, filename)
            assert os.path.isfile(file_path), f"Expected output file {filename} to be created."

    def test_individual_dataset_vs_extxyz_equivalence(self):
        """Test that datasets prepared from individual files and extxyz are equivalent."""
        config_individual_data = {
            "DATA_FOLDER": self.assets_dir,
            "GEOMETRY_FILE": "test_molecules.xyz",
            "ENERGY_FILE": "test_energies.txt",
            "FORCE_FILE": "test_forces.xyz",
            "CHARGE_FILE": "test_charges.txt",
            "ESP_FILE": "test_esp.txt",
            "ESP_GRAD_FILE": "test_esp_grad.xyz",
            "TARGET_FOLDER": self.target_folder,
            "PREFIX": "TestMoleculesIndividual"
        }
        config_individual = prepare_config(config_individual_data)
        
        config_extxyz_data = {
            "DATA_FOLDER": self.assets_dir,
            "EXTXYZ_FILE": "test_molecules.extxyz",
            "TARGET_FOLDER": self.target_folder,
            "PREFIX": "TestMoleculesExtxyz"
        }
        config_extxyz = prepare_config(config_extxyz_data)
        
        # Prepare datasets
        prepare_kgcnn_dataset(config_individual)
        prepare_kgcnn_dataset(config_extxyz)
        
        # Load the generated pickle files and compare contents
        with open(os.path.join(self.target_folder, f"{config_individual['PREFIX']}.kgcnn.pickle"), 'rb') as f:
            data_individual = pickle.load(f)
        
        with open(os.path.join(self.target_folder, f"{config_extxyz['PREFIX']}.kgcnn.pickle"), 'rb') as f:
            data_extxyz = pickle.load(f)
        
        # Compare keys and lengths of datasets
        for data_point_individual, data_point_extxyz in zip(data_individual, data_extxyz):
            assert data_point_individual.keys() == data_point_extxyz.keys(), "Datasets should have the same keys"
            for key in data_point_individual.keys():
                assert type(data_point_individual[key]) == type(data_point_extxyz[key]), f"Type of {key} should match between datasets"
                if isinstance(data_point_individual[key], np.ndarray):
                    assert data_point_individual[key].shape == data_point_extxyz[key].shape, f"Shape of {key} should match between datasets"
                else:
                    assert data_point_individual[key] == data_point_extxyz[key], f"Value of {key} should match between datasets"
            for key in data_point_individual.keys():
                if isinstance(data_point_individual[key], np.ndarray) and data_point_individual[key].dtype in [np.float32, np.float64]:
                    assert np.allclose(data_point_individual[key], data_point_extxyz[key], atol=1e-4), f"Data for {key} should match between datasets: {data_point_individual[key]} vs {data_point_extxyz[key]}"

    def test_no_data_scenario(self):
        """Test scenario where no data files are provided."""
        config_data = {
            "DATA_FOLDER": self.assets_dir,
            "TARGET_FOLDER": self.target_folder,
            "PREFIX": "TestMolecules"
        }
        config = prepare_config(config_data)
        
        with pytest.raises(ValueError):
            prepare_kgcnn_dataset(config)

    def test_prepare_config_function(self):
        """Test the prepare_config function itself."""
        # Test with minimal config
        config_data = {
            "TARGET_FOLDER": self.target_folder,
            "PREFIX": "TestConfig"
        }
        config = prepare_config(config_data)
        
        # Check that default values are set
        assert config["CUTOFF"] == 10.0, "Should set default cutoff"
        assert config["MAX_NEIGHBORS"] == 25, "Should set default max neighbors"
        assert config["GEOMETRY_FILE"] is None, "Should set default geometry file to None"
        assert config["EXTXYZ_ENERGY_KEY"] == "ref_energy", "Should set default extxyz energy key"
        
        # Test with custom values
        config_data_custom = {
            "TARGET_FOLDER": self.target_folder,
            "PREFIX": "TestConfigCustom",
            "CUTOFF": 5.0,
            "MAX_NEIGHBORS": 50,
            "GEOMETRY_FILE": "custom.xyz"
        }
        config_custom = prepare_config(config_data_custom)
        
        assert config_custom["CUTOFF"] == 5.0, "Should preserve custom cutoff"
        assert config_custom["MAX_NEIGHBORS"] == 50, "Should preserve custom max neighbors"
        assert config_custom["GEOMETRY_FILE"] == "custom.xyz", "Should preserve custom geometry file"

if __name__ == "__main__":
    pytest.main([__file__])
