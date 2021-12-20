import os
import pickle
import numpy as np
import json

from sklearn.preprocessing import StandardScaler
from kgcnn.data.qm import QMDataset
from kgcnn.data.download import DownloadDataset
from kgcnn.mol.methods import ExtensiveMolecularScaler
from kgcnn.mol.io import write_list_to_xyz_file


class QM9Dataset(QMDataset, DownloadDataset):
    """Store and process QM9 dataset."""

    download_info = {
        "dataset_name": "QM9",
        "data_directory_name": "qm9",
        # https://ndownloader.figshare.com/files/3195398
        "download_url": "https://ndownloader.figshare.com/files/3195389",
        "download_file_name": 'dsgdb9nsd.xyz.tar.bz2',
        "unpack_tar": True,
        "unpack_zip": False,
        "unpack_directory_name": 'dsgdb9nsd.xyz',
    }

    def __init__(self, reload: bool = False, verbose: int = 1):
        """Initialize QM9 dataset.

        Args:
            reload (bool): Whether to reload the data and make new dataset. Default is False.
            verbose (int): Print progress or info for processing where 0=silent. Default is 1.
        """
        QMDataset.__init__(self, verbose=verbose)
        DownloadDataset.__init__(self, **self.download_info, reload=reload, verbose=verbose)

        self.target_names = ['A', 'B', 'C', 'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'U0', 'U', 'H',
                             'G', 'Cv']
        self.target_units = ["GHz", "GHz", "GHz", "D", r"a_0^3", "eV", "eV", "eV", r"a_0^2", "eV", "eV", "eV", "eV",
                             "eV", r"cal/mol K"]
        self.target_unit_conversion = np.array(
            [[1.0, 1.0, 1.0, 1.0, 1.0, 27.2114, 27.2114, 27.2114, 1.0, 27.2114, 27.2114, 27.2114,
              27.2114, 27.2114, 1.0]]
        )  # Pick always same units for training
        self.dataset_name = "QM9"
        self.require_prepare_data = True
        self.fits_in_memory = True
        self.verbose = verbose
        self.data_directory = os.path.join(self.data_main_dir, self.data_directory_name)
        self.file_name = "qm9.xyz"

        if self.require_prepare_data:
            self.prepare_data(overwrite=reload)

        if self.fits_in_memory:
            self.read_in_memory()

    def prepare_data(self, overwrite: bool = False, xyz_column_name: str = None, make_sdf: bool = True):
        """Process data by loading all single xyz-files and store all pickled information to file.
        The single files are deleted afterwards, requires to re-extract the tar-file for overwrite.

        Args:
            overwrite (bool): Whether to redo the processing, requires un-zip of the data again. Defaults to False.
            xyz_column_name (str): Not used.
            make_sdf (bool): Whether to make SDF file.
        """
        path = self.data_directory

        dataset_size = 133885

        if (os.path.exists(os.path.join(path, "qm9.pickle")) or os.path.exists(
                os.path.join(path, "qm9.json"))) and not overwrite:
            self.info("Single molecules already pickled.")
        else:
            if not os.path.exists(os.path.join(path, 'dsgdb9nsd.xyz')):
                self.error("Can not find extracted dsgdb9nsd.xyz directory. Run reload dataset again.")
                return
            qm9 = []
            # Read individual files
            self.info("Reading dsgdb9nsd files ...")
            for i in range(1, dataset_size + 1):
                mol = []
                file = "dsgdb9nsd_" + "{:06d}".format(i) + ".xyz"
                open_file = open(os.path.join(path, "dsgdb9nsd.xyz", file), "r")
                lines = open_file.readlines()
                mol.append(int(lines[0]))
                labels = lines[1].strip().split(' ')[1].split('\t')
                if int(labels[0]) != i:
                    self.warning("Index for QM9 not matching xyz-file.")
                labels = [lines[1].strip().split(' ')[0].strip()] + [int(labels[0])] + [float(x) for x in labels[1:]]
                mol.append(labels)
                cords = []
                for j in range(int(lines[0])):
                    atom_info = lines[2 + j].strip().split('\t')
                    cords.append([atom_info[0]] + [float(x.replace('*^', 'e')) for x in atom_info[1:]])
                mol.append(cords)
                freqs = lines[int(lines[0]) + 2].strip().split('\t')
                freqs = [float(x) for x in freqs]
                mol.append(freqs)
                smiles = lines[int(lines[0]) + 3].strip().split('\t')
                mol.append(smiles)
                inchis = lines[int(lines[0]) + 4].strip().split('\t')
                mol.append(inchis)
                open_file.close()
                qm9.append(mol)

            # Save pickle data
            self.info("Saving qm9.json ...")
            with open(os.path.join(path, "qm9.json"), 'w') as f:
                json.dump(qm9, f)

            # Remove file after reading
            self.info("Cleaning up extracted files...")
            for i in range(1, dataset_size + 1):
                file = "dsgdb9nsd_" + "{:06d}".format(i) + ".xyz"
                file = os.path.join(path, "dsgdb9nsd.xyz", file)
                os.remove(file)

        if os.path.exists(os.path.join(path, self.file_name)) and not overwrite:
            self.info("Single xyz-file %s for molecules already created." % self.file_name)
        else:
            self.info("Reading dataset...", end='', flush=True)
            if os.path.exists(os.path.join(path, "qm9.pickle")):
                with open(os.path.join(path, "qm9.pickle"), 'rb') as f:
                    qm9 = pickle.load(f)
            elif os.path.exists(os.path.join(path, "qm9.json")):
                with open(os.path.join(path, "qm9.json"), 'rb') as f:
                    qm9 = json.load(f)
            else:
                raise FileNotFoundError("Can not find pickled QM9 dataset.")

            # Try extract bond-info and save mol-file.
            self.info("Writing single xyz-file ...")
            pos = [[y[1:] for y in x[2]] for x in qm9]
            atoms = [[y[0] for y in x[2]] for x in qm9]
            atoms_pos = [[x, y] for x, y in zip(atoms, pos)]
            write_list_to_xyz_file(os.path.join(path, "qm9.xyz"), atoms_pos)

        super(QM9Dataset, self).prepare_data(overwrite=overwrite)
        return self

    def read_in_memory(self, label_column_name: str = None):
        """Load the pickled QM9 data into memory and already split into items.

        Args:
            label_column_name(str): Not used.

        Returns:
            self
        """
        path = self.data_directory

        self.info("Reading dataset ...")
        if os.path.exists(os.path.join(path, "qm9.pickle")):
            with open(os.path.join(path, "qm9.pickle"), 'rb') as f:
                qm9 = pickle.load(f)
        elif os.path.exists(os.path.join(path, "qm9.json")):
            with open(os.path.join(path, "qm9.json"), 'rb') as f:
                qm9 = json.load(f)
        else:
            raise FileNotFoundError("Can not find pickled QM9 dataset.")

        # labels
        # self.length = 133885
        labels = np.array([x[1][1:] if len(x[1]) == 17 else x[1] for x in qm9])  # Remove 'gdb' tag here

        # Atoms as nodes
        atoms = [[y[0] for y in x[2]] for x in qm9]
        atom_dict = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9}
        zval = [[atom_dict[y] for y in x] for x in atoms]
        outzval = [np.array(x, dtype="int") for x in zval]
        nodes = outzval

        # Mean molecular weight mmw
        massdict = {'H': 1.0079, 'C': 12.0107, 'N': 14.0067, 'O': 15.9994, 'F': 18.9984}
        mass = [[massdict[y] for y in x] for x in atoms]
        mmw = np.array([[np.mean(x), len(x)] for x in mass])

        # Coordinates
        coord = [[[y[1], y[2], y[3]] for y in x[2]] for x in qm9]
        coord = [np.array(x) for x in coord]

        self.graph_number = [x for x in labels[:, 0]]
        self.node_coordinates = coord
        self.graph_labels = [x for x in labels[:, 1:]]
        self.node_symbol = atoms
        self.node_number = nodes
        self.graph_attributes = [x for x in mmw]

        # Try to read mol information
        self.read_in_memory_sdf()
        return self


class QM9GraphLabelScaler:
    """A standard scaler that scales all QM9 targets. For now, the main difference is that intensive and extensive
    properties are scaled differently. In principle, also dipole, polarizability or rotational constants
    could to be standardized differently."""

    def __init__(self, intensive_scaler=None, extensive_scaler=None):
        if intensive_scaler is None:
            intensive_scaler = {}
        if extensive_scaler is None:
            extensive_scaler = {}

        self.intensive_scaler = StandardScaler(**intensive_scaler)
        self.extensive_scaler = ExtensiveMolecularScaler(**extensive_scaler)

        self.scale_ = None

    def fit_transform(self, node_number, graph_labels):
        r"""Fit and transform all target labels for QM9.

        Args:
            node_number (list): List of atomic numbers for each molecule. E.g. `[np.array([6,1,1,1]), ...]`.
            graph_labels (np.ndarray): Array of QM9 labels of shape `(N, 15)`.

        Returns:
            np.ndarray: Transformed labels of shape `(N, 15)`.
        """
        self.fit(node_number, graph_labels)
        return self.transform(node_number, graph_labels)

    def transform(self, node_number, graph_labels):
        r"""Transform all target labels for QM9. Requires :obj:`fit()` called previously.

        Args:
            node_number (list): List of atomic numbers for each molecule. E.g. `[np.array([6,1,1,1]), ...]`.
            graph_labels (np.ndarray): Array of QM9 unscaled labels of shape `(N, 15)`.

        Returns:
            np.ndarray: Transformed labels of shape `(N, 15)`.
        """
        self._check_input(node_number, graph_labels)

        intensive_labels = graph_labels[:, :9]
        extensive_labels = graph_labels[:, 9:]

        trafo_intensive = self.intensive_scaler.transform(intensive_labels)
        trafo_extensive = self.extensive_scaler.transform(node_number, extensive_labels)

        out_labels = np.concatenate([trafo_intensive, trafo_extensive], axis=-1)
        return out_labels

    def fit(self, node_number, graph_labels):
        r"""Fit scaling of QM9 graph labels or targets.

        Args:
            node_number (list): List of atomic numbers for each molecule. E.g. `[np.array([6,1,1,1]), ...]`.
            graph_labels (np.ndarray): Array of QM9 labels of shape `(N, 15)`.

        Returns:
            self
        """
        self._check_input(node_number, graph_labels)

        # Note: Rotational Constants and r2 as well as dipole moment and polarizability
        # should be treated separately.
        intensive_labels = graph_labels[:, :9]
        extensive_labels = graph_labels[:, 9:]

        self.intensive_scaler.fit(intensive_labels)
        self.extensive_scaler.fit(node_number, extensive_labels)
        # print(self.intensive_scaler.scale_, self.extensive_scaler.scale_)
        self.scale_ = np.concatenate([self.intensive_scaler.scale_, self.extensive_scaler.scale_[0]], axis=0)
        return self

    def inverse_transform(self, node_number, graph_labels):
        r"""Back-transform all target labels for QM9.

        Args:
            node_number (list): List of atomic numbers for each molecule. E.g. `[np.array([6,1,1,1]), ...]`.
            graph_labels (np.ndarray): Array of QM9 scaled labels of shape `(N, 15)`.

        Returns:
            np.ndarray: Back-transformed labels of shape `(N, 15)`.
        """
        self._check_input(node_number, graph_labels)

        intensive_labels = graph_labels[:, :9]
        extensive_labels = graph_labels[:, 9:]

        inverse_trafo_intensive = self.intensive_scaler.inverse_transform(intensive_labels)
        inverse_trafo_extensive = self.extensive_scaler.inverse_transform(node_number, extensive_labels)

        out_labels = np.concatenate([inverse_trafo_intensive, inverse_trafo_extensive], axis=-1)
        return out_labels

    @staticmethod
    def padd(selected_targets, target_indices):
        r"""Padding a set of specific targets defined by `target_indices` to the full QM9 target dimension of 15.

        Args:
            selected_targets (np.ndarray): A reduced selection of QM9 target `(n_samples, n_targets)` where
                `n_targets` <= 15.
            target_indices (np.ndarray): Indices of specific targets of shape `(n_targets, )`.

        Returns:
            np.ndarray: Array of QM9 labels of shape `(N, 15)`.
        """
        labels = np.zeros((len(selected_targets), 15))
        labels[:, target_indices] = selected_targets
        return labels

    @staticmethod
    def _check_input(node_number, graph_labels):
        assert len(node_number) == len(graph_labels), "`QM9GraphLabelScaler` needs same length input."
        assert graph_labels.shape[-1] == 15, "`QM9GraphLabelScaler` got wrong targets."

# dataset = QM9Dataset()
# scaler = QM9GraphLabelScaler()
# trafo_labels = scaler.fit_transform(dataset.node_number, dataset.graph_labels)
# rev_labels = scaler.inverse_transform(dataset.node_number, tafo_labels
# print(np.amax(np.abs(dataset.graph_labels-rev_labels)))