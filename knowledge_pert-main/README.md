# Prior knowledge to CPA through graph embeddings


#### GraphAutoencoder Training

To train a model, select the architecture and hyperparameters modifying `config/net_config/your_file.yaml`. Important 
also to change the `adversarial` parameter in the model initialization. `adversarial=False` is just an autoencoder 
without any adversarial component.

In the config file `change_drugs`, if True, create a new dictionary with homogenized drug names. In GSM is not necessary
since the dictionary is manually created. `perturbation` expects to receive `gene` or `drug` depending on the kind of
perturbation in the dataset to analyze. If `drug`, it will parse all drug names and try to harmonize them. The keys are 
simply the name of the observations that carry the information asked. 
`GSM` dataset works with `change_drugs=False` and `perturbation=drug`. `Sciplex` works with `change_drugs=False` and 
`perturbation=drug`. In `Norman` dataset, `perturbation=gene` since all perturbations are `gene KO.

The architecture can also be defined in the config files. The system is `GNN-NN(Enc)-NN(Dec)-GNN`. In the transition
between GNN and NN we take all the nodes corresponding to a certain graph and concat them to build a tensor representing
the entire graph. We then encode this tensor. The tensor is later reconstructed. 

The input is a graph with 3 node features `ID, value, PertFlag`, but the output are just 2 node features `mean, var` since
we don't care about reconstructing the discrete values. 

#### Data

Download data from https://drive.google.com/drive/folders/1VXKOAgPrUEzNJGLve2ErekjJDyCZFp5N?usp=sharing (Dataset folder
with Norman, Kang and GSM dataset). 
Data preprocessed specifically for this project.

----------------------------------

Models are trained from .yaml files. 

#### Knowledge Graph generation

Run `python data_handling/process_data.py` to generate the initial Knowledge Graph with OmniPath and Drug Gene Database 
interactions. The data will be stored in `omnipath_triples`. The data is already generated. In case that the data don't 
exist at the time of training a model, the data will be generated then.

#### KGE model

Write a custom .yaml file `config/your_file.yaml` and execute ``python Train_KG.py --config=config/your_file.yaml``. 
The results are stored in `TB-KGE` with the same name specified on config file. The model is stored in 
`trained_models/KG_models`. Models are trained with PyKEEN library. Early Stopping focus on  Adjusted Mean Rank Index 
with frequency 5 and patience 10.

#### GNN graph generation

As we use KG embeddings as node features, per each KG model, new data must be generated since the node features change.
To do so, just run ``python GNN_Data_Generation.py --kge_model=trained_models/KG_models/your_model.plk``. 
The results will be stored in `GNN Data`. `GNN Data/processed` contains the PyG data objects. `GNN Data/raw` contains 
Pickle files just with the triples involved in each subgraph.


#### GAE model

Write a custom .yaml file `config/your_file.yaml` and execute ``python Train_GNN.py --config=config/your_file.yaml``. 
The results are stored in `TB-GNN` with the same name specified on config file. The model is stored in 
`trained_models/GNN_models`. Currently, just GAE working, VGAE doesn't work yet.

#### BASELINE

The baseline of the project is just running the KG Embedding model, take the nodes that are involved in the subgraph 
centered around a drug and mean pool its embedding to obtain an entire subgraph embedding. 
``python BASELINE.py --config=config/your_file.yaml`` runs the pipeline and store the embeddings in ``TB-KGE``.

#### MAIN WORKFLOW

The main workflow is build a graph with KGE as node features and operate with GNN to obtain a subgraph embedding. 
The GNN will be mainly an Encoder than can be fully trained as GAE, pre-trained as GAE and fine-tuned jointly with CPA
or fully jointly trained with CPA.

![image](Main%20Workflow.png)

----------------------------------

#### GSM Graph Data

To download the GSM graph data: https://drive.google.com/file/d/1WbKVmdgU3uKyTvDC0vux0WU0O6ifNyXb/view?usp=sharing
2.3 GB zip file
Add link to download new GSM anndata from GDrive
