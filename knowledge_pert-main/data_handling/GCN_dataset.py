import os.path as osp
import json
import scanpy as sc
import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder

import torch
from torch_geometric.data import Dataset, InMemoryDataset

from .graph_data import drug_subgraph, create_graph_object, cell_graph, create_cell_graph, handle_features, new_create_cell_graph
from data_handling.process_data import harmonize_names

class Dataset2:  # FACEBOOK RESEARCH CPA IMPLEMENTATION WITH ON THE FLY GRAPH GENERATION
    def __init__(self, dict_in):

        data = sc.read(dict_in['data_path'])
        #self.adata = sc.read(dict_in['data_path'])

        if len(data.obs['control'].unique()) == 1:
            data.obs['control'] = [1 if x == 'control_1.0' else 0 for x in data.obs.drug_dose_name.values]

        self.perturbation_key = dict_in['perturbation_key']
        self.dose_key = dict_in['dose_key']
        self.cell_type_key = dict_in['cell_type_key']
        self.split_key = dict_in['split_key']
        self.drugs_names = np.array(data.obs[self.perturbation_key].values)
        self.condition = data.obs[self.perturbation_key]
        
        self.cell_types_interest = dict_in['cell_type_interest']
        self.combination = dict_in['combination']
        self.fc_selected = dict_in['fc_genes']

        # get unique drugs
        drugs_names_unique = set()
        for d in self.drugs_names:
            [drugs_names_unique.add(i) for i in d.split("+")]
        self.drugs_names_unique = np.array(list(drugs_names_unique))

        self.dict_in = dict_in
        self.dict_in, data = self._setup(data) # setup data with name transformations and removing unknown drugs

        if isinstance(data.X, (np.ndarray)):
            self.genes = torch.Tensor(data.X)
        else:
            self.genes = torch.Tensor(data.X.A)
        self.doses = data.obs[self.dose_key]
        #self.condition = data.obs[self.perturbation_key]
        self.condition = self.dict_in['condition']

        self.var_names = data.var_names
        self.pert_categories = np.array(data.obs['cov_drug_dose_name'].values)

        self.de_genes = data.uns['rank_genes_groups_cov']
        self.ctrl = data.obs['control'].values
        self.ctrl_name = list(np.unique(data[data.obs['control'] == 1].obs[self.perturbation_key]))

        # self.drugs_names = np.array(self.adata.obs[self.perturbation_key].values)
        self.dose_names = np.array(data.obs[self.dose_key].values)

        # get unique drugs
        drugs_names_unique = set()
        for d in self.drugs_names_official:
            [drugs_names_unique.add(i) for i in d.split("+")]
        self.drugs_names_unique_official = np.array(list(drugs_names_unique))

        # save encoder for a comparison with Mo's model
        # later we need to remove this part
        encoder_drug = OneHotEncoder(sparse=False)
        encoder_drug.fit(self.drugs_names_unique_official.reshape(-1, 1))

        self.atomic_drugs_dict = dict(zip(self.drugs_names_unique_official, encoder_drug.transform(
            self.drugs_names_unique_official.reshape(-1, 1))))

        # get drug combinations
        drugs = []
        drugs_wo_dose = []
        for i, comb in enumerate(self.drugs_names_official):
            drugs_combos = encoder_drug.transform(
                np.array(comb.split("+")).reshape(-1, 1))
            dose_combos = str(data.obs[self.dose_key].values[i]).split("+")
            for j, d in enumerate(dose_combos):
                if j == 0:
                    drug_ohe = float(d) * drugs_combos[j]
                    drug_dose = drugs_combos[j]
                else:
                    drug_ohe += float(d) * drugs_combos[j]
                    drug_dose += drugs_combos[j]
            drugs.append(drug_ohe)
            drugs_wo_dose.append(drug_dose)
        self.drugs = torch.Tensor(drugs)
        self.drugs_wo_dose = torch.Tensor(drugs_wo_dose)
        self.drugs_wo_dose = torch.squeeze(self.drugs_wo_dose)

        self.cell_types_names = np.array(data.obs[self.cell_type_key].values)
        self.cell_types_names_unique = np.unique(self.cell_types_names)

        encoder_ct = OneHotEncoder(sparse=False)
        encoder_ct.fit(self.cell_types_names_unique.reshape(-1, 1))

        self.atomic_сovars_dict = dict(zip(list(self.cell_types_names_unique), encoder_ct.transform(
            self.cell_types_names_unique.reshape(-1, 1))))

        self.cell_types = torch.Tensor(encoder_ct.transform(
            self.cell_types_names.reshape(-1, 1))).float()

        self.num_cell_types = len(self.cell_types_names_unique)
        self.num_genes = data.X.shape[1]
        self.num_drugs = len(self.drugs_names_unique_official)

        self.indices = {
            "all": list(range(data.X.shape[0])),
            "control": np.where(data.obs['control'] == 1)[0].tolist(),
            "treated": np.where(data.obs['control'] != 1)[0].tolist(),
            "train": np.where(data.obs[self.split_key] == 'train')[0].tolist(),
            "test": np.where(data.obs[self.split_key] == 'test')[0].tolist(),
            "ood": np.where(data.obs[self.split_key] == 'ood')[0].tolist()
        }
        
        if self.cell_types_interest != '':
            for cell in self.cell_types_interest:
                self.indices[cell] = np.where(data.obs['cell_type'] == cell)[0].tolist()
                

        atomic_ohe = encoder_drug.transform(
            self.drugs_names_unique_official.reshape(-1, 1))

        self.drug_dict = {}
        for idrug, drug in enumerate(self.drugs_names_unique_official):
            i = np.where(atomic_ohe[idrug] == 1)[0][0]
            self.drug_dict[i] = drug

        self.cell_subgraph = cell_graph(self.dict_in, self.var_names, self.de_genes)

        self.nodes_names = self.cell_subgraph['nodes_id'].keys()
        self.nodes_index = list(self.cell_subgraph['genes_index'].values())
        self.nodes_genes_index = list(self.cell_subgraph['nodes_genes_index'].values())
        self.gene_ordering_index = self.cell_subgraph['gene_ordering_index']
        
        if self.combination and len(self.fc_selected)==0:
            self.fc_genes_index = set(range(0,5000,1)).difference(set(self.nodes_index))
        elif self.combination and self.fc_selected == 'all':
            self.fc_genes_index = set(range(0,5000,1)) # this is equivalent to running CPA and add the knowledge module
        elif self.combination and len(self.fc_selected) != 0:
            self.fc_genes_index = []
            for gene in self.fc_selected:
                self.fc_genes_index.append(list(self.cell_subgraph['var_names']).index(gene))               
        
        if self.combination:
            self.fc_genes_names = self.cell_subgraph['var_names'][list(self.fc_genes_index)]
            self.fc_genes = self.genes[:, list(self.fc_genes_index)]
            self.graph_genes = self.genes[:, list(self.nodes_index)]
            self.combination_order = dict() # dictionary in which the keys are the table index and the value the output network index
            i = 0
            for index in self.nodes_index:
                self.combination_order[index] = i
                i = i + 1
            for index in self.fc_genes_index:
                self.combination_order[index] = i
                i = i + 1

        self.node_features = handle_features(dict(
            genes=self.genes,
            doses=self.doses,
            condition=self.condition,
            nodes_id=self.cell_subgraph['nodes_id'],
            genes_index=self.cell_subgraph['genes_index'],
            gene_ordering_index=self.gene_ordering_index,
            official_names_drugs=self.cell_subgraph['official_names_drugs'],
        ))

    def _setup(self, data):

        with open(self.dict_in['entities_path']) as json_file:
            entities = json.load(json_file)

        if self.dict_in['perturbation'] == 'drug':
            harmonize_dict = dict(
                drugs_path=self.dict_in['drugs_path'],
                drugs=self.drugs_names_unique,
            )
            new_names = harmonize_names(harmonize_dict)

            drug_official_names = self.dict_in['drugs_official']

            stable_drug_names = []
            for drug in self.condition:
                if drug in drug_official_names.keys():
                    stable_drug_names.append(drug_official_names[drug])
                if drug in ['control', 'ctrl', 'vehicle']:
                    stable_drug_names.append(drug)

            stable_drug_names_set = set(stable_drug_names)

            # Remove drugs that do not appear on the prior knowledge network
            known_drugs = []
            for i, row in self.condition.iteritems():
                if row in drug_official_names:
                    if drug_official_names[row] in stable_drug_names_set:
                        known_drugs.append(i)
                if row in ['control', 'ctrl', 'vehicle']:  # let controls pass as known drug
                    known_drugs.append(i)

            prunned_data = data[known_drugs, :]

            df = pd.DataFrame(stable_drug_names, index=prunned_data.obs['condition'].index, columns=['drug_official'])

            # create new observation with the official names of each drug
            prunned_data.obs['drug_official'] = df

            self.drugs_names_official = prunned_data.obs['drug_official']
            self.dict_in['condition'] = prunned_data.obs['drug_official']

            return dict(
                self.dict_in,
                entities_dict=entities,
                harmonized_drugs=new_names['harmonized_drugs'],
                official_names_drugs=drug_official_names,
            ), prunned_data

        else:

            self.drugs_names_official = self.drugs_names
            self.dict_in['condition'] = self.condition
            drug_official_names = None

            return dict(
                self.dict_in,
                entities_dict=entities,
                official_names_drugs=drug_official_names,
            ), data

    def subset(self, split, condition="all"):
        idx = list(set(self.indices[split]) & set(self.indices[condition]))
        return SubDataset(self, idx)

    def __getitem__(self, i):

        if torch.is_tensor(i):
            i = i.tolist()

        input_dict = dict(
            self.dict_in,
            subgraph=self.cell_subgraph['subgraph'],
            nodes_id=self.cell_subgraph['nodes_id'],
            output_edges=self.cell_subgraph['output_edges'],
            input_edges=self.cell_subgraph['input_edges'],
            gene_expression=self.genes[i, :],
            dose=self.doses[i],
            condition=self.condition[i],
            var_names=self.var_names,
            genes_index=self.cell_subgraph['genes_index'],
            gene_ordering_index=self.gene_ordering_index,
        )

        #graph = create_cell_graph(input_dict)

        input_dict = dict(
            node_features=self.node_features[i, :, :],
            output_edges=self.cell_subgraph['output_edges'],
            input_edges=self.cell_subgraph['input_edges'],
        )

        graph = new_create_cell_graph(input_dict)

        if self.combination:
            return graph, self.fc_genes[i], self.drugs_wo_dose[i], self.cell_types[i]

        return graph, self.drugs_wo_dose[i], self.cell_types[i]

    def __len__(self):
        return len(self.genes)


class SubDataset:
    """
    Subsets a `Dataset` by selecting the examples given by `indices`.
    """

    def __init__(self, dataset, indices):
        self.perturbation_key = dataset.perturbation_key
        self.dose_key = dataset.dose_key
        self.covars_key = dataset.cell_type_key

        self.perts_dict = dataset.atomic_drugs_dict
        self.covars_dict = dataset.atomic_сovars_dict

        self.genes = dataset.genes[indices]
        self.drugs = dataset.drugs[indices]
        self.drugs_wo_dose = dataset.drugs_wo_dose[indices]
        self.cell_types = dataset.cell_types[indices]
        self.doses = dataset.doses[indices]
        self.condition = dataset.condition[indices]
        
        self.combination = dataset.combination

        self.drugs_names = dataset.drugs_names[indices]
        self.pert_categories = dataset.pert_categories[indices]
        self.cell_types_names = dataset.cell_types_names[indices]

        self.var_names = dataset.var_names
        self.de_genes = dataset.de_genes
        self.ctrl_name = dataset.ctrl_name[0]

        self.num_cell_types = dataset.num_cell_types
        self.num_genes = dataset.num_genes
        self.num_drugs = dataset.num_drugs

        self.output_edges = dataset.cell_subgraph['output_edges']
        self.input_edges = dataset.cell_subgraph['input_edges']
        self.symbols = dataset.cell_subgraph['var_names']
        self.de_symbols = dataset.cell_subgraph['de_names']
        self.nodes_id = dataset.cell_subgraph['nodes_id']
        self.genes_index = dataset.cell_subgraph['genes_index']

        self.dict_in = dataset.dict_in

        self.nodes_names = dataset.nodes_names
        self.nodes_index = dataset.nodes_index
        self.nodes_genes_index = dataset.nodes_genes_index
        self.gene_ordering_index = dataset.gene_ordering_index

        self.node_features = dataset.node_features[indices, :, :]
        if dataset.combination:
            self.fc_genes = dataset.fc_genes[indices, :]
            self.fc_genes_names = dataset.fc_genes_names 
            self.graph_genes = dataset.graph_genes[indices, :]
            self.combination_order = dataset.combination_order
        
    def __getitem__(self, i):
        if torch.is_tensor(i):
            i = i.tolist()

        input_dict = dict(
            self.dict_in,
            nodes_id=self.nodes_id,
            output_edges=self.output_edges,
            input_edges=self.input_edges,
            gene_expression=self.genes[i, :],
            dose=self.doses[i],
            condition=self.condition[i],
            var_names=self.symbols,
            genes_index=self.genes_index,
            gene_ordering_index=self.gene_ordering_index,
        )

        #graph = create_cell_graph(input_dict)

        input_dict = dict(
            node_features=self.node_features[i, :, :],
            output_edges=self.output_edges,
            input_edges=self.input_edges,
        )

        graph = new_create_cell_graph(input_dict)
        
        if self.combination:
            return graph, self.fc_genes[i], self.drugs_wo_dose[i], self.cell_types[i]

        return graph, self.drugs_wo_dose[i], self.cell_types[i]

    def __len__(self):
        return len(self.genes)


def load_dataset_splits(
        input_dict,
        return_dataset=False):
    dataset = Dataset2(input_dict)

    splits = {
        "training": dataset.subset("train", "all"),
        "training_control": dataset.subset("train", "control"),
        "training_treated": dataset.subset("train", "treated"),
        "test": dataset.subset("test", "all"),
        "test_control": dataset.subset("test", "control"),
        "test_treated": dataset.subset("test", "treated"),
        "ood": dataset.subset("ood", "all")
    }
    
    for cell in dataset.cell_types_interest:
        splits["ood_control_" + str(cell)]=dataset.subset(cell, "control")
        splits["ood_treated_" + str(cell)]=dataset.subset(cell, "treated")

    if return_dataset:
        return splits, dataset
    else:
        return splits
