import tensorflow as tf
from typing import Union
from kgcnn.layers.casting import ChangeTensorType
from kgcnn.layers.modules import ExpandDims
from kgcnn.model.utils import get_model_class

ks = tf.keras


@ks.utils.register_keras_serializable(package='kgcnn', name='EnergyForceModel')
class EnergyForceModel(ks.models.Model):
    r"""Force model that generates forces from any energy predicting model by taking the derivative with respect to
    the input coordinates.

    For now the model has to cast to dense tensor for using :obj:`batch_jacobian` , however, this will likely support
    ragged tensors in the future.

    .. code-block:: python

        import tensorflow as tf
        from kgcnn.model.force import EnergyForceModel
        model = EnergyForceModel(
            model_energy= {
                "module_name": "kgcnn.literature.Schnet",
                "class_name": "make_model",
                "config": {
                    "name": "SchnetEnergy",
                    "inputs": [
                        {"shape": [None], "name": "z", "dtype": "float32", "ragged": True},
                        {"shape": [None, 3], "name": "R", "dtype": "float32", "ragged": True},
                        {"shape": [None, 2], "name": "range_indices", "dtype": "int64", "ragged": True}
                    ],
                    "input_embedding": {
                        "node": {"input_dim": 95, "output_dim": 128}
                    },
                    "last_mlp": {"use_bias": [True, True, True], "units": [128, 64, 1],
                                 "activation": ['kgcnn>shifted_softplus', 'kgcnn>shifted_softplus', 'linear']},
                    "interaction_args": {
                        "units": 128, "use_bias": True, "activation": "kgcnn>shifted_softplus", "cfconv_pool": "sum"
                    },
                    "node_pooling_args": {"pooling_method": "sum"},
                    "depth": 6,
                    "gauss_args": {"bins": 25, "distance": 5, "offset": 0.0, "sigma": 0.4}, "verbose": 10,
                    "output_embedding": "graph",
                    "use_output_mlp": False,
                    "output_mlp": None
                }
            },
            coordinate_input=1,
            output_as_dict=True,
            output_to_tensor=False,
            output_squeeze_states=True,
            is_physical_force=True
        )

    """

    def __init__(self,
                 model_energy=None,
                 coordinate_input: Union[int, str] = 1,
                 esp_input: Union[int, str] = None,
                 esp_grad_input: Union[int, str] = None,
                 energy_output: Union[int, str] = 1,
                 output_as_dict: bool = True,
                 ragged_validate: bool = False,
                 output_to_tensor: bool = True,
                 output_squeeze_states: bool = False,
                 nested_model_config: bool = True,
                 is_physical_force: bool = True,
                 **kwargs):
        r"""Initialize :obj:`EnergyForceModel` with sub-model for energy prediction.

        This wrapper model was designed for models in `kgcnn.literature` that predict energy from geometric
        information and precalculated esp-values.

        .. note::

            The (serialized) energy model is inferred by `module_name` , `class_name` , `config` within :obj:`kgcnn` ,
            but you can also pass any model directly to `model_energy` and
            set `nested_model_config` to `False` .

        Args:
            model_energy (dict): Keras model for energy prediction. Can also be a serialization dict.
            coordinate_input (str, int): Index or key where to find coordinate tensor in model input.
            esp_input (str, int): Index or key where to find esp tensor in model input. Optional
            esp_grad_input (str, int): Index or key where to find esp gradient tensor in model input. Optional
            energy_output (str, int): Index or key where to find coordinate tensor in model input.
            output_as_dict (bool): Whether to return energy and force as list or as dict. Default is True.
            ragged_validate (bool): Whether to validate ragged tensor creation. Default is False.
            output_to_tensor (bool): Whether to cast the output to tensor or keep ragged output. Default is True
            output_squeeze_states (bool): Whether to squeeze states, which can be done for one energy value to remove
                an axis of one.
            nested_model_config (bool): Whether `config` has model config of the energy model. Default is True.
            is_physical_force (bool): Whether gradient of force, which is the negative gradient, is to be returned.
        """
        super(EnergyForceModel, self).__init__(self, **kwargs)
        if model_energy is None:
            raise ValueError("Require valid model in `model_energy` for force prediction.")

        # Input for model_energy.
        self._model_energy = model_energy

        if isinstance(model_energy, ks.models.Model):
            # Ignoring module_name and class_name.
            self.energy_model = model_energy
        elif isinstance(model_energy, dict):
            if "module_name" not in model_energy:
                self.energy_model = ks.utils.deserialize_keras_object(model_energy)
            else:
                self.energy_model_class = get_model_class(model_energy["module_name"], model_energy["class_name"])
                self.energy_model = self.energy_model_class(**model_energy["config"])
        else:
            raise TypeError("Input `model_energy` must be dict or `ks.models.Model` .")

        if output_as_dict is True and energy_output is not 0:
            print("Kgcnn warning: energy-model returns more than just energy, setting output_as_dict as False")
            output_as_dict = False

        # Additional parameters of io and behavior of this class.
        self.ragged_validate = ragged_validate
        self.coordinate_input = coordinate_input
        self.esp_input = esp_input
        self.esp_grad_input = esp_grad_input
        self.energy_output = energy_output
        self.output_as_dict = output_as_dict
        self.output_to_tensor = output_to_tensor
        self.output_squeeze_states = output_squeeze_states
        self.is_physical_force = is_physical_force
        self.nested_model_config = nested_model_config

        # Layers.
        self.cast_coordinates = ChangeTensorType(input_tensor_type="ragged", output_tensor_type="mask")
        self.cast_esp = ChangeTensorType(input_tensor_type="ragged", output_tensor_type="mask")
        self.layer_exp_dims = ExpandDims(axis=-1)

    def call(self, inputs, **kwargs):
        """Forward pass that wraps energy model in gradient tape.

        Args:
            inputs (list, dict): Must be list of (tensor) input for energy model.
                Index or key to find coordinates must be provided.
            training (bool): Whether model is in training, passed to energy model. Default is False.

        Returns:
            dict, list: Model output plus force or derivative.
        """
        x = inputs[self.coordinate_input]
        inputs_energy = [i for i in inputs]
        # x is ragged tensor of shape (batch, [N], 3) with cartesian coordinates.
        # `batch_jacobian` does not yet support ragged tensor input.
        # Cast to masked tensor for coordinates only.
        x_pad, x_mask = self.cast_coordinates(x, **kwargs)  # (batch, N, 3), (batch, N, 3)
        if self.esp_input is not None and self.esp_grad_input is not None:
            esp = inputs[self.esp_input] # (batch, N)
            desp_dr = inputs[self.esp_grad_input] # (batch, N, 3)
            desp_dr_pad, desp_dr_mask = self.cast_coordinates(desp_dr, **kwargs)  # (batch, N, 3), (batch, N, 3)
            desp_dr_pad = self.layer_exp_dims(desp_dr_pad) # (batch, N, 3, states)
            esp_pad, esp_mask = self.cast_esp(esp, **kwargs) # (batch, N), (batch, N)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x_pad)
            # Temporary solution for casting.
            # Cast back to ragged tensor for model input.
            x_pad_to_ragged = self._cast_coordinates_pad_to_ragged(x_pad, x_mask, self.ragged_validate)
            inputs_energy[self.coordinate_input] = x_pad_to_ragged
            if self.esp_input is not None and self.esp_grad_input is not None:
                tape.watch(esp_pad)
                esp_pad_to_ragged = self._cast_esp_pad_to_ragged(esp_pad, esp_mask, self.ragged_validate)
                inputs_energy[self.esp_input] = esp_pad_to_ragged
            # Predict energy.
            # Energy must be tensor of shape (batch, states)
            outputs = self.energy_model(inputs_energy, **kwargs)
            if isinstance(outputs, list):
                eng = outputs[self.energy_output]
            else:
                eng = outputs
        de_dr = tape.batch_jacobian(eng, x_pad) # (batch, states)x(batch, N, 3) = (batch, states, N, 3)
        de_dr = tf.transpose(de_dr, perm=[0, 2, 3, 1]) # (batch, N, 3, states)

        if self.esp_input is not None and self.esp_grad_input is not None:
            de_desp = tape.batch_jacobian(eng, esp_pad)
            de_desp = ExpandDims(axis=-1)(de_desp)
            de_desp = tf.transpose(de_desp, perm=[0, 2, 3, 1])
            de_dr = de_dr + de_desp*desp_dr_pad

        if self.is_physical_force:
            de_dr = -de_dr
        if self.output_squeeze_states:
            de_dr = tf.squeeze(de_dr, axis=-1)
        if not self.output_to_tensor:
            de_dr = self._cast_coordinates_pad_to_ragged(de_dr, x_mask, self.ragged_validate)
        elif isinstance(outputs, list):
            for index, output in enumerate(outputs):
                if isinstance(output, tf.RaggedTensor):
                    outputs[index] = output.to_tensor()
        if self.output_as_dict:
            return {"energy": eng, "force": de_dr}
        else:
            if isinstance(outputs, list):
                outputs.append(de_dr)
                return outputs
            return outputs, de_dr

    # Temporary solution.
    @staticmethod
    @tf.function
    def _cast_coordinates_pad_to_ragged(x_pad, x_mask, validate):
        # Assume that float mask is the same for all coordinates.
        x_mask_number = tf.cast(x_mask[:, :, 0], dtype="bool")  # (batch, N)
        x_values = x_pad[x_mask_number]
        x_row_length = tf.reduce_sum(tf.cast(x_mask_number, dtype="int64"), axis=-1)
        return tf.RaggedTensor.from_row_lengths(x_values, x_row_length, validate=validate)

    @staticmethod
    @tf.function
    def _cast_esp_pad_to_ragged(esp_pad, esp_mask, validate):
        esp_mask_number = tf.cast(esp_mask, dtype="bool")  # (batch, N)
        esp_values = esp_pad[esp_mask_number]
        esp_row_length = tf.reduce_sum(tf.cast(esp_mask_number, dtype="int64"), axis=-1)
        return tf.RaggedTensor.from_row_lengths(esp_values, esp_row_length, validate=validate)

    def get_config(self):
        """Get config."""
        # Keras model does not provide config from base class.
        # conf = super(EnergyForceModel, self).get_config()
        conf = {}
        # Serialize class if _model_energy is not dict.
        if isinstance(self._model_energy, dict):
            model_energy = self._model_energy
        else:
            model_energy = ks.utils.serialize_keras_object(self._model_energy)
        conf.update({
            "model_energy": model_energy,
            "coordinate_input": self.coordinate_input,
            "esp_input": self.esp_input,
            "esp_grad_input": self.esp_grad_input,
            "output_as_dict": self.output_as_dict,
            "ragged_validate": self.ragged_validate,
            "output_to_tensor": self.output_to_tensor,
            "output_squeeze_states": self.output_squeeze_states,
            "nested_model_config": self.nested_model_config
        })
        return conf
