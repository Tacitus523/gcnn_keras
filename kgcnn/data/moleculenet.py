import os
import numpy as np
import rdkit
import rdkit.Chem
import rdkit.Chem as Chem

from kgcnn.data.base import DownloadDataset, MemoryGeometricGraphDataset
from kgcnn.mol.molgraph import MolecularGraph, OneHotEncoder
from kgcnn.utils.data import load_json_file


class MuleculeNetDataset(DownloadDataset, MemoryGeometricGraphDataset):

    mol_filename = "mol.json"

    def __init__(self, reload=False, verbose=1):

        DownloadDataset.__init__(self, reload=reload, verbose=verbose)
        MemoryGeometricGraphDataset.__init__(self, verbose=verbose)

        if self.fits_in_memory:
            self.read_in_memory(verbose=verbose)

    @classmethod
    def _smiles_to_mol_list(cls, smiles, addHs=True, sanitize=True, embed_molecule=True, verbose=1):
        if len(smiles) == 0:
            print("Error:kgcnn: Can not translate smiles, received empty list for %s." % cls.dataset_name)
        if verbose > 0:
            print("INFO:kcnn: Generating molecules and store %s to disk..." % cls.mol_filename, end='', flush=True)
        molecule_list = []
        for i, sm in enumerate(smiles):
            mg = MolecularGraph()
            mg.MolFromSmiles(sm, addHs=addHs, sanitize=sanitize)
            if embed_molecule:
                mg.EmbedMolecule()
            molecule_list.append(rdkit.Chem.MolToMolBlock(mg.mol))
        if verbose > 0:
            print("done")
        return molecule_list

    def read_in_memory(self, verbose=1):
        """Load mol_filename into memory.

        Args:
            verbose (int): Print progress or info for processing where 0=silent. Default is 1.
        """
        mol_path = os.path.join(self.data_main_dir, self.data_directory, self.mol_filename)
        if not os.path.exists(mol_path):
            raise FileNotFoundError("ERROR:kgcnn: Can not load molecules for dataset %s" % self.dataset_name)

        mols = load_json_file(mol_path)
        atoms = []
        coords = []
        number = []
        for x in mols:
            mg = MolecularGraph().MolFromMolBlock(x, removeHs=False, sanitize=True)
            atoms.append(mg.GetSymbol())
            coords.append(mg.GetPositions())
            number.append(mg.GetAtomicNum())
        self.node_symbol = atoms
        self.node_coordinates = coords
        self.node_number = number
        self.graph_size = [len(x) for x in atoms]

        return self

    def set_attributes(self,
                       nodes=None,
                       edges=None,
                       graph=None,
                       encoder=None,
                       removeHs=True,
                       verbose=1):

        # We have to reload the dataset here to start fresh
        self.read_in_memory(verbose=verbose)

        mol_path = os.path.join(self.data_main_dir, self.data_directory, self.mol_filename)
        if not os.path.exists(mol_path):
            raise FileNotFoundError("ERROR:kgcnn: Can not load molecules for dataset %s" % self.dataset_name)

        if verbose > 0:
            print("INFO:kgcnn: Making attributes...", end='', flush=True)

        mols = load_json_file(mol_path)

        # Choose default values here:
        if nodes is None:
            nodes = ['Symbol', 'TotalDegree', 'FormalCharge', 'NumRadicalElectrons', 'Hybridization',
                  'IsAromatic', 'IsInRing', 'TotalNumHs', 'CIPCode', "ChiralityPossible", "ChiralTag"]
        if edges is None:
            edges = ['BondType', 'Stereo', 'IsAromatic', 'IsConjugated', 'IsInRing', "Stereo"]
        if graph is None:
            graph = ['ExactMolWt', 'NumAtoms']
        if encoder is None:
            encoder = {
                "Symbol": OneHotEncoder(['B', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'As', 'Se', 'Br', 'Te', 'I', 'At']),
                "Hybridization": OneHotEncoder([Chem.rdchem.HybridizationType.SP,
                                                Chem.rdchem.HybridizationType.SP2,
                                                Chem.rdchem.HybridizationType.SP3,
                                                Chem.rdchem.HybridizationType.SP3D,
                                                Chem.rdchem.HybridizationType.SP3D2]),
                "TotalDegree": OneHotEncoder([0, 1, 2, 3, 4, 5], add_others=False),
                "TotalNumHs": OneHotEncoder([0, 1, 2, 3, 4], add_others=False),
                "BondType": OneHotEncoder([Chem.rdchem.BondType.SINGLE,
                                           Chem.rdchem.BondType.DOUBLE,
                                           Chem.rdchem.BondType.TRIPLE,
                                           Chem.rdchem.BondType.AROMATIC], add_others=False),
                "Stereo": OneHotEncoder([Chem.rdchem.BondStereo.STEREONONE,
                                         Chem.rdchem.BondStereo.STEREOANY,
                                         Chem.rdchem.BondStereo.STEREOZ,
                                         Chem.rdchem.BondStereo.STEREOE], add_others=False),
                "CIPCode": OneHotEncoder(['R', 'S'], add_others=False)}

        # Reset all attributes
        graph_attributes = []
        node_attributes = []
        edge_attributes = []
        edge_indices = []
        node_coordinates = []
        node_symbol = []
        node_number = []

        for i, sm in enumerate(mols):
            mg = MolecularGraph().MolFromMolBlock(sm, removeHs=removeHs, sanitize=True)
            mg.set_attributes(nodes=nodes, edges=edges, graph=graph, encoder=encoder)
            node_attributes.append(np.array(mg.node_attributes, dtype="float32"))
            edge_attributes.append(np.array(mg.edge_attributes, dtype="float32"))
            edge_indices.append(np.array(mg.edge_indices, dtype="int64"))
            graph_attributes.append(np.array(mg.graph_attributes, dtype="float32"))
            node_symbol.append(mg.node_symbol)
            node_coordinates.append(np.array(mg.node_coordinates, dtype="float32"))
            node_number.append(mg.node_number)

        self.graph_size = [len(x) for x in node_attributes]
        self.graph_attributes = graph_attributes
        self.node_attributes = node_attributes
        self.edge_attributes = edge_attributes
        self.edge_indices = edge_indices
        self.node_coordinates = node_coordinates
        self.node_symbol = node_symbol
        self.node_number = node_number

        if verbose > 0:
            print("done")
            for key, value in encoder.items():
                print("INFO:kgcnn: OneHotEncoder", key, "found", value.found_values)

        return self