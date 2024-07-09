import tensorflow as tf
from tensorflow.keras.activations import relu, tanh, elu, selu # type: ignore

# Define a custom Swish activation function, Tensorflow one has problems with saving custom gradients
def swish(x):
    return x * tf.sigmoid(x)

# Define Leaky ReLU as a custom activation function
def leaky_relu(x):
    return tf.keras.activations.relu(x, alpha=0.2)

# Wrapper function to select activation dynamically
def custom_activation(x, activation):
    if activation == 'swish':
        return swish(x)
    elif activation == 'leaky_relu':
        return leaky_relu(x)
    elif activation == 'relu':
        return relu(x)
    elif activation == 'tanh':
        return tanh(x)
    elif activation == 'elu':
        return elu(x)
    elif activation == 'selu':
        return selu(x)
    else:
        raise ValueError(f"Unsupported activation: {activation}")