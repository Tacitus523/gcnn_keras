import tensorflow as tf
import tensorflow.keras as ks

from kgcnn.layers.keras import Dense


# import tensorflow.keras.backend as ksb


class MLP(ks.layers.Layer):
    """
    Multilayer perceptron that consist of N dense keras layers.
        
    Args:
        units (list): Size of hidden layers for each layer.
        use_bias (list, optional): Use bias for hidden layers. Defaults to True.
        activation (list, optional): Activity identifier. Defaults to None.
        activity_regularizer (list, optional): Activity regularizer identifier. Defaults to None.
        kernel_regularizer (list, optional): Kernel regularizer identifier. Defaults to None.
        bias_regularizer (list, optional): Bias regularizer identifier. Defaults to None.
        **kwargs 
    """

    def __init__(self,
                 units,
                 use_bias=True,
                 activation=None,
                 activity_regularizer=None,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_constraint=None,
                 bias_constraint=None,
                 ragged_validate=False,
                 input_tensor_type="ragged",
                 **kwargs):
        """Init MLP as for dense."""
        super(MLP, self).__init__(**kwargs)
        self._supports_ragged_inputs = True
        self.ragged_validate = ragged_validate
        self.input_tensor_type = input_tensor_type
        # Make to one element list
        if isinstance(units, int):
            units = [units]

        if not isinstance(use_bias, list) and not isinstance(use_bias, tuple):
            use_bias = [use_bias for _ in units]
        if not isinstance(activation, list) and not isinstance(activation, tuple):
            activation = [activation for _ in units]
        if not isinstance(kernel_regularizer, list) and not isinstance(kernel_regularizer, tuple):
            kernel_regularizer = [kernel_regularizer for _ in units]
        if not isinstance(bias_regularizer, list) and not isinstance(bias_regularizer, tuple):
            bias_regularizer = [bias_regularizer for _ in units]
        if not isinstance(activity_regularizer, list) and not isinstance(activity_regularizer, tuple):
            activity_regularizer = [activity_regularizer for _ in units]
        if not isinstance(kernel_initializer, list) and not isinstance(kernel_initializer, tuple):
            kernel_initializer = [kernel_initializer for _ in units]
        if not isinstance(bias_initializer, list) and not isinstance(bias_initializer, tuple):
            bias_initializer = [bias_initializer for _ in units]
        if not isinstance(kernel_constraint, list) and not isinstance(kernel_constraint, tuple):
            kernel_constraint = [kernel_constraint for _ in units]
        if not isinstance(bias_constraint, list) and not isinstance(bias_constraint, tuple):
            bias_constraint = [bias_constraint for _ in units]

        for x in [activation, kernel_regularizer, bias_regularizer, activity_regularizer, kernel_initializer,
                  bias_initializer, kernel_constraint, bias_constraint, use_bias]:
            if len(x) != len(units):
                raise ValueError("Error: Provide matching list of units", units, "and", x, "or simply a single value.")

        # Serialized props
        self.mlp_units = list(units)
        self.mlp_use_bias = list(use_bias)
        self.mlp_activation = list([tf.keras.activations.get(x) for x in activation])
        self.mlp_kernel_regularizer = list([tf.keras.regularizers.get(x) for x in kernel_regularizer])
        self.mlp_bias_regularizer = list([tf.keras.regularizers.get(x) for x in bias_regularizer])
        self.mlp_activity_regularizer = list([tf.keras.regularizers.get(x) for x in activity_regularizer])
        self.mlp_kernel_initializer = list([tf.keras.initializers.get(x) for x in kernel_initializer])
        self.mlp_bias_initializer = list([tf.keras.initializers.get(x) for x in bias_initializer])
        self.mlp_kernel_constraint = list([tf.keras.constraints.get(x) for x in kernel_constraint])
        self.mlp_bias_constraint = list([tf.keras.constraints.get(x) for x in bias_constraint])

        self.mlp_dense_list = [Dense(
            units=self.mlp_units[i],
            use_bias=self.mlp_use_bias[i],
            name=self.name + '_dense_' + str(i),
            activation=self.mlp_activation[i],
            activity_regularizer=self.mlp_activity_regularizer[i],
            kernel_regularizer=self.mlp_kernel_regularizer[i],
            bias_regularizer=self.mlp_bias_regularizer[i],
            kernel_initializer=self.mlp_kernel_initializer[i],
            bias_initializer=self.mlp_bias_initializer[i],
            kernel_constraint=self.mlp_kernel_constraint[i],
            bias_constraint=self.mlp_bias_constraint[i],
            ragged_validate=self.ragged_validate,
            input_tensor_type=self.input_tensor_type
        ) for i in range(len(self.mlp_units))]

    def build(self, input_shape):
        """Build layer."""
        super(MLP, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.
        
        Args:
            inputs: Input tensor.

        Returns:
            out: MLP pass.
        
        """
        x = inputs
        for i in range(len(self.mlp_units)):
            x = self.mlp_dense_list[i](x, **kwargs)
        out = x
        return out

    def get_config(self):
        """Update config."""
        config = super(MLP, self).get_config()
        config.update({"units": self.mlp_units,
                       'use_bias': self.mlp_use_bias,
                       'activation': [tf.keras.activations.serialize(x) for x in self.mlp_activation],
                       'activity_regularizer': [tf.keras.regularizers.serialize(x) for x in
                                                self.mlp_activity_regularizer],
                       'kernel_regularizer': [tf.keras.regularizers.serialize(x) for x in self.mlp_kernel_regularizer],
                       'bias_regularizer': [tf.keras.regularizers.serialize(x) for x in self.mlp_bias_regularizer],
                       "kernel_initializer": [tf.keras.initializers.serialize(x) for x in self.mlp_kernel_initializer],
                       "bias_initializer": [tf.keras.initializers.serialize(x) for x in self.mlp_bias_initializer],
                       "kernel_constraint": [tf.keras.constraints.serialize(x) for x in self.mlp_kernel_constraint],
                       "bias_constraint": [tf.keras.constraints.serialize(x) for x in self.mlp_bias_constraint],
                       })
        config.update({"ragged_validate": self.ragged_validate,
                       "input_tensor_type": self.input_tensor_type})
        return config