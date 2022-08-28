# Summary of Benchmark Training

Note that these are the results for models within `kgcnn`, and that training is not always done with optimal hyperparameter, when comparing with literature.

## CoraLuDataset

Cora Dataset after Lu et al. (2003) of 2708 publications and 1433 sparse node attributes and 7 node classes.

| model | kgcnn | epochs | Accuracy | 
| :---: | :---: | :---: | :---: | 
| GAT | 2.1.0 | 250 | 0.8645 &pm; 0.0129  | 
| GATv2 | 2.1.0 | 250 | 0.8464 &pm; 0.0097  | 
| GCN | 2.1.0 | 300 | 0.7969 &pm; 0.0144  | 

## CoraDataset

Cora Dataset of 19793 publications and 8710 sparse node attributes and 70 node classes.

| model | kgcnn | epochs | Accuracy | 
| :---: | :---: | :---: | :---: | 
| GCN | 2.1.0 | 300 | 0.6150 &pm; 0.0121  | 

## ESOLDataset

ESOL (MoleculeNet) consists of 1128 compounds as smiles and their corresponding water solubility in log10(mol/L). Here we use a random split.

| model | kgcnn | epochs | MAE [log mol/L] | RMSE [log mol/L] | 
| :---: | :---: | :---: | :---: | :---: | 
| AttentiveFP | 2.1.0 | 200 | 0.4272 &pm; 0.0297  | 0.6002 &pm; 0.0268  | 
| CMPNN | 2.1.0 | 300 | 0.4596 &pm; 0.0257  | 0.6561 &pm; 0.0295  | 
| DimeNetPP | 2.1.0 | 872 | 0.4572 &pm; 0.0304  | 0.6377 &pm; 0.0501  | 
| DMPNN | 2.1.0 | 300 | 0.4463 &pm; 0.0357  | 0.6442 &pm; 0.0429  | 
| GAT | 2.1.0 | 500 | 0.4699 &pm; 0.0435  | 0.6711 &pm; 0.0745  | 
| GATv2 | 2.1.0 | 500 | 0.4628 &pm; 0.0432  | 0.6615 &pm; 0.0565  | 
| GCN | 2.1.0 | 800 | 0.5639 &pm; 0.0102  | 0.7995 &pm; 0.0324  | 
| GIN | 2.1.0 | 300 | 0.5107 &pm; 0.0395  | 0.7241 &pm; 0.0441  | 
| GIN.make_model_edge | 2.1.0 | 300 | 0.4761 &pm; 0.0259  | 0.6733 &pm; 0.0407  | 
| GraphSAGE | 2.1.0 | 500 | 0.4654 &pm; 0.0377  | 0.6556 &pm; 0.0697  | 
| HamNet | 2.1.0 | 400 | 0.5492 &pm; 0.0509  | 0.7645 &pm; 0.0676  | 
| INorp | 2.1.0 | 500 | 0.4828 &pm; 0.0201  | 0.6748 &pm; 0.0350  | 
| Megnet | 2.1.0 | 800 | 0.5597 &pm; 0.0314  | 0.7972 &pm; 0.0439  | 
| NMPN | 2.1.0 | 800 | 0.5706 &pm; 0.0497  | 0.8144 &pm; 0.0710  | 
| PAiNN | 2.1.0 | 250 | 0.4182 &pm; 0.0198  | 0.5961 &pm; 0.0344  | 
