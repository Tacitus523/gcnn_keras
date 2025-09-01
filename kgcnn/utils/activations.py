import tensorflow as tf
from tensorflow.keras.activations import relu, tanh, elu, selu # type: ignore
from kgcnn.ops.activ import shifted_softplus, softplus2, leaky_softplus, leaky_relu, swish

# # Define a custom Swish activation function, Tensorflow one has problems with saving custom gradients
# def swish(x):
#     return x * tf.sigmoid(x)

# # Define Leaky ReLU as a custom activation function
# def leaky_relu(x):
#     return tf.keras.activations.relu(x, alpha=0.2)

# Wrapper function to select activation dynamically
def custom_activation(x, activation):
    if activation == 'swish':
        return swish(beta=1.0)(x)
    elif activation == 'leaky_relu':
        return leaky_relu(alpha=0.05)(x)
    elif activation == 'relu':
        return relu(x)
    elif activation == 'tanh':
        return tanh(x)
    elif activation == 'elu':
        return elu(x)
    elif activation == 'selu':
        return selu(x)
    elif activation == 'shifted_softplus':
        return shifted_softplus(x)
    elif activation == 'softplus2':
        return softplus2(x)
    elif activation == 'leaky_softplus':
        return leaky_softplus(alpha=0.05)(x)
    else:
        raise ValueError(f"Unsupported activation: {activation}")