hyper = {
    "MEGAN": {
        "explanation": {
            "channels": 2,
            "gt_suffix": None,
        },
        "model": {
            "class_name": "make_model",
            "module_name": "kgcnn.literature.MEGAN",
            "config": {
                'name': "MEGAN",
                'units': [5, 5, 5],
                'importance_units': [],
                'final_units': [3, 1],
                "final_activation": "linear",
                'dropout_rate': 0.0,
                'importance_factor': 1.0,
                'importance_multiplier': 3.0,
                'sparsity_factor': 3.0,
                'final_dropout_rate': 0.00,
                'importance_channels': 2,
                'regression_reference': 0,
                'regression_limits': [-3, +3],
                'return_importances': False,
                'use_edge_features': True,
                'inputs': [{'shape': (None, 3), 'name': "node_attributes", 'dtype': 'float32', 'ragged': True},
                           {'shape': (None, 1), 'name': "edge_attributes", 'dtype': 'float32', 'ragged': True},
                           {'shape': (None, 2), 'name': "edge_indices", 'dtype': 'int64', 'ragged': True}],
            }
        },
        "training": {
            "fit": {
                "batch_size": 64,
                "epochs": 100,
                "validation_freq": 1,
                "verbose": 2,
                "callbacks": []
            },
            "compile": {
                "optimizer": {"class_name": "Adam", "config": {"lr": 1e-02}},
                "loss": "mean_squared_error",
                "metrics": ["mean_squared_error"],
            },
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": 42, "shuffle": True}},
        },
        "data": {
            "dataset": {
                "class_name": "VgdRbMotifsDataset",
                "module_name": "kgcnn.data.datasets.VgdRbMotifsDataset",
                "config": {},
                "methods": []
            },
            "data_unit": ""
        },
        "info": {
            "postfix": "",
            "postfix_file": "",
            "kgcnn_version": "2.2.0"
        }
    },
}