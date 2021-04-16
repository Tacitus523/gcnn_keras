import tensorflow as tf
import tensorflow.keras as ks
import tensorflow.keras.backend as ksb

from kgcnn.utils.partition import _change_partition_type, _change_edge_tensor_indexing_by_row_partition


class GatherNodes(ks.layers.Layer):
    """
    Gather nodes by edge_indices. Indexlist must match flatten nodes.
    
    If graphs edge_indices were in 'sample' mode, the edge_indices must be corrected for disjoint graphs.
    
    Args:
        node_indexing (str): Indices refering to 'sample' or to the continous 'batch'.
                             For disjoint representation 'batch' is default.
        partition_type (str): Partition tensor type to assign nodes/edges to batch. Default is "row_length".
        **kwargs
    """

    def __init__(self, node_indexing='batch', partition_type="row_length", concat_nodes=True, **kwargs):
        """Initialize layer."""
        super(GatherNodes, self).__init__(**kwargs)
        self.node_indexing = node_indexing
        self.partition_type = partition_type
        self.concat_nodes = concat_nodes

    def build(self, input_shape):
        """Build layer."""
        super(GatherNodes, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs (list): of [node, node_partition, edge_index, edge_partition]

            - node (tf.tensor): Flatten node feature tensor of shape (batch*None,F)
            - node_partition (tf.tensor): Row partition for nodes. This can be either row_length, value_rowids,
              row_splits. Yields the assignment of nodes to each graph in batch.
              Default is row_length of shape (batch,)
            - edge_index (tf.tensor): Flatten edge indices of shape (batch*None,2)
            - edge_partition (tf.tensor): Row partition for edge. This can be either row_length, value_rowids,
              row_splits. Yields the assignment of edges to each graph in batch.
              Default is row_length of shape (batch,)
            
        Returns:
            features (tf.tensor): Gathered node features of (ingoing,outgoing) nodes.        
            Output shape is (batch*None,F+F).  
        """
        node, node_part, edge_index, edge_part = inputs

        indexlist = _change_edge_tensor_indexing_by_row_partition(edge_index, node_part, edge_part,
                                                                  partition_type_node=self.partition_type,
                                                                  partition_type_edge=self.partition_type,
                                                                  to_indexing='batch',
                                                                  from_indexing=self.node_indexing)

        out = tf.gather(node, indexlist, axis=0)

        if self.concat_nodes:
            out_shape = tf.shape(out)
            out = tf.reshape(out, (out_shape[0],-1))

        return out

    def get_config(self):
        """Update config."""
        config = super(GatherNodes, self).get_config()
        config.update({"node_indexing": self.node_indexing,
                       "partition_type": self.partition_type,
                       "concat_nodes": self.concat_nodes})
        return config


class GatherNodesOutgoing(ks.layers.Layer):
    """
    Gather nodes by edge edge_indices. Indexlist must match flatten nodes.
    
    If graphs edge_indices were in 'sample' mode, the edge_indices must be corrected for disjoint graphs.
    For outgoing nodes, layer uses only index[1].
    
    Args:
        node_indexing (str): Indices refering to 'sample' or to the continous 'batch'.
                             For disjoint representation 'batch' is default.
        partition_type (str): Partition tensor type to assign nodes/edges to batch. Default is "row_length".
        **kwargs
    """

    def __init__(self, node_indexing='batch', partition_type="row_length", **kwargs):
        """Initialize layer."""
        super(GatherNodesOutgoing, self).__init__(**kwargs)
        self.node_indexing = node_indexing
        self.partition_type = partition_type

    def build(self, input_shape):
        """Build layer."""
        super(GatherNodesOutgoing, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs (list): of [node, node_partition, edge_index, edge_partition]

            - node (tf.tensor): Flatten node feature tensor of shape (batch*None,F)
            - node_partition (tf.tensor): Row partition for nodes. This can be either row_length, value_rowids,
              row_splits. Yields the assignment of nodes to each graph in batch.
              Default is row_length of shape (batch,)
            - edge_index (tf.tensor): Flatten edge indices of shape (batch*None,2)
              For ingoing gather nodes according to index[1]
            - edge_partition (tf.tensor): Row partition for edge. This can be either row_length, value_rowids,
              row_splits. Yields the assignment of edges to each graph in batch.
              Default is row_length of shape (batch,)
        
        Returns:
            features (tf.tensor): A list of gathered outgoing node features from edge_indices.
            Output shape is (batch*None,F).
        
        """
        node, node_part, edge_index, edge_part = inputs
        # node,edge_index= inputs
        indexlist = _change_edge_tensor_indexing_by_row_partition(edge_index, node_part, edge_part,
                                                                  partition_type_node=self.partition_type,
                                                                  partition_type_edge=self.partition_type,
                                                                  to_indexing='batch',
                                                                  from_indexing=self.node_indexing)

        out = tf.gather(node, indexlist[:, 1], axis=0)
        return out

    def get_config(self):
        """Update config."""
        config = super(GatherNodesOutgoing, self).get_config()
        config.update({"node_indexing": self.node_indexing,
                       "partition_type": self.partition_type})
        return config


class GatherNodesIngoing(ks.layers.Layer):
    """
    Gather nodes by edge edge_indices. Indexlist must match flatten nodes.
    
    If graphs edge_indices were in 'sample' mode, the edge_indices must be corrected for disjoint graphs.
    For ingoing nodes, layer uses only index[0].
    
    Args:
        node_indexing (str): Indices refering to 'sample' or to the continous 'batch'.
                             For disjoint representation 'batch' is default.
        partition_type (str): Partition tensor type to assign nodes/edges to batch. Default is "row_length".
        **kwargs
    """

    def __init__(self, node_indexing='batch', partition_type="row_length", **kwargs):
        """Initialize layer."""
        super(GatherNodesIngoing, self).__init__(**kwargs)
        self.node_indexing = node_indexing
        self.partition_type = partition_type

    def build(self, input_shape):
        """Build layer."""
        super(GatherNodesIngoing, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs (list): of [node, node_partition, edge_index, edge_partition]

            - node (tf.tensor): Flatten node feature tensor of shape (batch*None,F)
            - node_partition (tf.tensor): Row partition for nodes. This can be either row_length, value_rowids,
              row_splits. Yields the assignment of nodes to each graph in batch.
              Default is row_length of shape (batch,)
            - edge_index (tf.tensor): Flatten edge_indices of shape (batch*None,2)
              For ingoing gather nodes according to index[0]
            - edge_partition (tf.tensor): Row partition for edge. This can be either row_length, value_rowids,
              row_splits. Yields the assignment of edges to each graph in batch.
              Default is row_length of shape (batch,)
    
        Returns:
            features (tf.tensor): adj_matrix list of gathered ingoing node features from edge_indices.
            Output shape is (batch*None,F).
        """
        node, node_part, edge_index, edge_part = inputs
        # node,edge_index= inputs
        indexlist = _change_edge_tensor_indexing_by_row_partition(edge_index, node_part, edge_part,
                                                                  partition_type_node=self.partition_type,
                                                                  partition_type_edge=self.partition_type,
                                                                  to_indexing='batch',
                                                                  from_indexing=self.node_indexing)

        out = tf.gather(node, indexlist[:, 0], axis=0)
        return out

    def get_config(self):
        """Update config."""
        config = super(GatherNodesIngoing, self).get_config()
        config.update({"node_indexing": self.node_indexing,
                       "partition_type": self.partition_type})
        return config


class GatherState(ks.layers.Layer):
    """
    Layer to repeat environment or global state for node or edge lists. The node or edge lists are flattened.
    
    To repeat the correct environment for each sample, a tensor with the target length/partition is required.

    Args:
        partition_type (str): Partition tensor type to assign nodes/edges to batch. Default is "row_length".
        **kwargs
    """

    def __init__(self, partition_type="row_length", **kwargs):
        """Initialize layer."""
        super(GatherState, self).__init__(**kwargs)
        self.partition_type = partition_type

    def build(self, input_shape):
        """Build layer."""
        super(GatherState, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs (list): of [environment, target_length]

            - environment (tf.tensor): List of graph specific feature tensor of shape (batch*None,F)
            - target_partition (tf.tensor): Assignment of nodes or edges to each graph in batch.
              Default is row_length of shape (batch,).

        Returns:
            features (tf.tensor): A tensor with repeated single state for each graph.
            Output shape is (batch*N,F).
        """
        env, target_part = inputs

        target_len = _change_partition_type(target_part, self.partition_type, "row_length")

        out = tf.repeat(env, target_len, axis=0)
        return out

    def get_config(self):
        """Update config."""
        config = super(GatherState, self).get_config()
        config.update({"partition_type": self.partition_type})
        return config
