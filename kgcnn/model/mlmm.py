import tensorflow as tf
from typing import Union
from kgcnn.layers.conv.hdnnp_conv import ElectrostaticQMMMEnergyPointCharge, ElectrostaticQMMMForcePointCharge

ks = tf.keras

@ks.utils.register_keras_serializable(package='kgcnn', name='MLMMEnergyForceModel')
class MLMMEnergyForceModel(ks.models.Model):
    r"""Machine Learning/Molecular Mechanics model that adjusts energy and forces from any charge, energy and force
        predicting model by taking the ESP and its gradients with respect to coordinates into account.

    .. code-block:: python

        import tensorflow as tf
        from kgcnn.model.force import EnergyForceModel
        from kgcnn.model.mlmm import MLMMEnergyForceModel

        model_energy_config = {
            "name": "HDNNP4th",
            "inputs": [{"shape": (None,), "name": "node_number", "dtype": "int64", "ragged": True},
                    {"shape": (None, 3), "name": "node_coordinates", "dtype": "float32", "ragged": True},
                    {"shape": (None, 2), "name": "range_indices", "dtype": "int64", "ragged": True},
                    {"shape": (None, 3), "name": "angle_indices_nodes", "dtype": "int64", "ragged": True},
                    {"shape": (1,), "name": "total_charge", "dtype": "float32", "ragged": False},
                    {"shape": (None,), "name": "esp", "dtype": "float32", "ragged": True},
                    {"shape": (None, 3), "name": "esp_gradient", "dtype": "float32", "ragged": True}],
            "g2_kwargs": {"eta": eta_array, "rs": Rs_array, "rc": cutoff_rad, "elements": elemental_mapping},
            "g4_kwargs": {"eta": eta_ang_array, "zeta": zeta_array, "lamda": lambd_array, "rc": cutoff_ang
                        , "elements": elemental_mapping, "multiplicity": 2.0},
            "normalize_kwargs": {},
            "mlp_charge_kwargs": {"units": [15, 1],
                                "num_relations": 96,
                                "activation": ["tanh", "linear"]},
            "mlp_local_kwargs": {"units": [35, 35, 1],
                                "num_relations": 96,
                                "activation": ["tanh", "tanh", "linear"]},
            "cent_kwargs": {},
            "electrostatic_kwargs": {"name": "electrostatic_layer"},
            "qmmm_kwargs": {"name": "qmmm_layer"},
            "node_pooling_args": {"pooling_method": "sum"},
            "verbose": 10,
            "output_embedding": "charge+qm_energy", "output_to_tensor": True,
            "use_output_mlp": False,
            "output_mlp": {"use_bias": [True, True], "units": [64, 1],
                        "activation": ["swish", "linear"]}
        }

        force_model = EnergyForceModel(
            model_energy = model_energy_config,
            coordinate_input=1,
            energy_output = 1,
            output_as_dict=False,
            output_to_tensor=False,
            output_squeeze_states=True,
            is_physical_force=False
        )

        qmmm_model = MLMMEnergyForceModel(
            model_force = force_model,
            esp_input = 5,
            esp_grad_input = 6,
            charge_energy_force_output = [0,1,2]
        )

    """

    def __init__(self,
                 model_force: ks.models.Model = None,
                 esp_input: Union[int, str] = 5,
                 esp_grad_input: Union[int, str] = 6,
                 charge_energy_force_output: list = [0,1,2],
                 qmmm_energy_kwargs: dict = None,
                 qmmm_force_kwargs: dict = None,
                 **kwargs):
        r"""Initialize :obj:`MLMMEnergyForceModel` with sub-sub-model for force prediction.

        This wrapper model was designed for models that predict charge, energy and force from geometric
        information.

        Args:
            model_force (ks.models.Model): Keras model for charge, energy and force prediction.
            esp_input (str, int): Index or key where to find esp tensor in model input.
            esp_grad_input (str, int): Index or key where to find esp gradient tensor in model input.
            charge_energy_force_outputs (list): List of indices where to find charges, energy and forces in
                model output.
            qmmm_energy_kwargs (dict): Additional arguments for qmmm_energy_layer.
            qmmm_force_kwargs (dict): Additional arguments for qmmm_force_layer.
        """
        super(MLMMEnergyForceModel, self).__init__(self, **kwargs)
        if model_force is None:
            raise ValueError("Require valid model in `model_force` for QM prediction.")
        if qmmm_energy_kwargs is None:
            qmmm_energy_kwargs = {}
        if qmmm_force_kwargs is None:
            qmmm_force_kwargs = {}

        self.model_force = model_force


        # Additional parameters of io and behavior of this class.
        self.esp_input = esp_input
        self.esp_grad_input = esp_grad_input
        self.charge_energy_force_output = charge_energy_force_output

        # Layers.
        self.qmmm_energy_layer = ElectrostaticQMMMEnergyPointCharge(**qmmm_energy_kwargs)
        self.qmmm_force_layer = ElectrostaticQMMMForcePointCharge(**qmmm_force_kwargs)

    def call(self, inputs, **kwargs):
        """Forward pass that applies qmmm electrostatic interaction to energy and force predictions.

        Args:
            inputs (list, dict): Must be list of (tensor) input for energy model.
                Index or key to find coordinates must be provided.

        Returns:
           list: Force model outputs
        """
        esp_input = inputs[self.esp_input]
        esp_grad_input = inputs[self.esp_grad_input]

        qm_outputs = self.model_force(inputs)
        qm_charge = qm_outputs[self.charge_energy_force_output[0]]
        qm_energy = qm_outputs[self.charge_energy_force_output[1]] 
        qm_force = qm_outputs[self.charge_energy_force_output[2]]

        qmmm_energy = self.qmmm_energy_layer([qm_charge, esp_input])
        qmmm_force = self.qmmm_force_layer([qm_charge, esp_grad_input])

        total_energy = qm_energy + qmmm_energy
        total_force = qm_force + qmmm_force

        total_outputs = [qm_output for qm_output in qm_outputs]
        total_outputs[self.charge_energy_force_output[0]] = qm_charge.to_tensor()
        total_outputs[self.charge_energy_force_output[1]] = total_energy
        total_outputs[self.charge_energy_force_output[2]] = total_force.to_tensor()
        return total_outputs
        

    def get_config(self):
        """Get config."""
        # Keras model does not provide config from base class.
        # conf = super(MLMMEnergyForceModel, self).get_config()
        conf = {}
        # Serialize class if _model_energy is not dict.

        model_force = ks.utils.serialize_keras_object(self.model_force)
        conf.update({
            "model_force": model_force,
            "esp_input": self.esp_input,
            "esp_grad_input": self.esp_grad_input,
            "charge_energy_force_output": self.charge_energy_force_output
        })
        return conf