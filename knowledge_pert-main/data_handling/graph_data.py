import pandas as pd
import numpy as np
import mygene
import os.path as osp
import random

from pykeen.triples import TriplesFactory

import torch
from torch_geometric.data import Data


def drug_subgraph(drug, input_dict):

    """
    Generate subgraphs of a certain drug containing drug-gene and gene-gene relations
    :param drug: drug of interest
    :param input_dict: contains path to the triples
    :return: dict with induced_subgraph in Pandas, Triples Factory and embeddings of all nodes
    """

    # Create a subgraph with drug-gene relations and all gene-gene interactions between genes affected by same drug
    path = input_dict['triples_path']
    triples = np.loadtxt(path, dtype=str, delimiter='\t')  # Load triples
    tf = TriplesFactory.from_labeled_triples(triples)
    triples = pd.DataFrame(triples)
    aux = triples[triples[0].isin([drug])]
    induced_subgraph = pd.concat([aux, triples[triples[0].isin(aux[2]) & triples[2].isin(aux[2])]])

    return dict(
        subgraph=induced_subgraph,
        embeddings=input_dict['entity_embeddings'],
        TriplesFactory=tf
    )


def create_graph_object(input_dict):

    """
    Creates a Pytorch Geometric Data object from a dataframe with triples
    :param input_dict: that contains the dataframe, the embeddings and the path to the TriplesFactory to obtain the IDs
    :return: PG Data
    """

    triples_df = input_dict['subgraph']
    entity_embeddings = input_dict['embeddings']
    tf = input_dict['TriplesFactory']

    nodes_id = {}

    output_edge = []
    input_edge = []

    # Fill the dictionary with the IDs of all nodes and create input and output edges
    for index, row in triples_df.iterrows():
 #       if row[0] == 'BMS-345541':
 #           print("bingo")
        if len(nodes_id) == 0:
            nodes_id[row[0]] = 0
        if row[0] not in nodes_id.keys():
            nodes_id[row[0]] = max(nodes_id.values()) + 1
        if row[2] not in nodes_id.keys():
            nodes_id[row[2]] = max(nodes_id.values()) + 1

        output_edge.append(nodes_id[row[0]])
        input_edge.append(nodes_id[row[2]])

    node_features = [[]] * len(nodes_id)

    # Use as node features the embeddings of the Knowledge Graph
    for i in range(0, len(nodes_id)):
        entity = list(nodes_id.keys())[i]
        node_features[i] = (entity_embeddings()[tf.entity_to_id[entity]].detach().numpy().tolist())

    edge_index = torch.tensor([output_edge,
                               input_edge], dtype=torch.long)
    x = torch.tensor(node_features)
    data = Data(x=x, edge_index=edge_index)

    return data


def cell_graph(input_dict, var_names, de_names):

    """
    First try to harmonize the names of the drugs to maintain all drugs even thought small name variations.
    It also maintains just the triples between elements (genes & drugs) in the experiment
    :param input_dict: with adata, triples path and harmonized drugs names
    :return: same dictionary + new subgraph
    """

    triples_path = input_dict['triples_path']
    coexpression = input_dict['coexpression_network']
    condition = input_dict['condition']
    entities_dict = input_dict['entities_dict']
    official_names_drug = input_dict['official_names_drugs']

    triples = np.loadtxt(triples_path, dtype=str, delimiter='\t')  # Load triples

    if input_dict['perturbation'] == 'drug':
        # Create a list with all the names of the drugs in the experiment
        flat_list = []
        real_drugs = list(condition.unique())
        for i in ['vehicle', 'control']:
            if i in real_drugs:
                real_drugs.remove(i)  # vehicle is how control condition is denoted

        for sublist in [input_dict['harmonized_drugs'][x] for x in real_drugs]:  # take all names of the real drugs
            for item in sublist:                                                 # input all names in a flatten set
                flat_list.append(item)
        set_list = set(flat_list)
        if 'nan' in set_list:
            set_list.remove('nan')      # remove nan if there's nan

    indices = []

    # To gene symbols if there are gene IDS
    all_de_genes = []
    for value in de_names.values():
        all_de_genes = all_de_genes + list(value)

    de_gene_symbol_dict = {}
    if var_names[0].startswith('ENSG0'):
        mg = mygene.MyGeneInfo()
        ens = var_names
        ginfo = mg.querymany(ens, scopes='ensembl.gene')
        symbols = []
        for i in range(0, len(ginfo)):
            if 'symbol' in ginfo[i].keys():
                symbols.append(ginfo[i]['symbol'])
                for gene in all_de_genes:
                    if gene == ginfo[i]['query']:
                        de_gene_symbol_dict[gene] = ginfo[i]['symbol']

        de_symbols = {}
        for key in de_names.keys():
            for gene in de_names[key]:
                if key not in de_symbols.keys() and gene in de_gene_symbol_dict.keys():
                    de_symbols[key] = []
                    de_symbols[key].append(de_gene_symbol_dict[gene])
                elif key in de_symbols.keys() and gene in de_gene_symbol_dict.keys():
                    de_symbols[key].append(de_gene_symbol_dict[gene])
    else:
        symbols = var_names
        de_symbols = de_names
        
        
    # Add triples from co-expression network
    if coexpression is not None:
        coexpression_network = pd.read_csv(coexpression)
        
        triples_df = pd.DataFrame(triples)
        
        co2 = pd.DataFrame(columns=[0, 1, 2])
        co2[0] = coexpression_network['TF']
        co2[2] = coexpression_network['target']
        co2[1] = 'unknown'

        new_triples = pd.concat([triples_df, co2], axis=0)
        new_triples = new_triples.to_numpy()
        triples = new_triples

    # Maintain just the triples between genes and drugs in the anndata
    for i in range(0, len(triples)):
        if triples[i, 0] in symbols and triples[i, 2] in symbols:
            indices.append(i)
        # If perturbation is drug, then there exists set_list, if not, no
        if input_dict['perturbation'] == 'drug':
            if triples[i, 0].lower() in set_list and triples[i, 2] in symbols:
                indices.append(i)

    triples = triples[indices, :]
    triples = pd.DataFrame(triples)

    if input_dict['perturbation'] == 'drug': 
        harmonized_drugs_dataset = dict()  # dictionary of harmonized_drugs that appear in this dataset
        for key, value in official_names_drug.items():
            if value not in ['control', 'ctrl', 'vehicle']:
                harmonized_drugs_dataset[value] = input_dict['harmonized_drugs'][value]
    else:
        harmonized_drugs_dataset = None
    

    nodes_id = {}

    output_edge = []
    input_edge = []

    # Fill the dictionary with the IDs of all nodes and create input and output edges
    for index, row in triples.iterrows():
        origin = row[0]
        destination = row[2]

        if row[0] not in entities_dict['Gene']:
            origin = row[0].lower()
            for k, v in harmonized_drugs_dataset.items():
                if origin in v:
                    origin = k
        if row[2] not in entities_dict['Gene']:
            destination = row[2].lower()
            for k, v in harmonized_drugs_dataset.items():
                if destination in v:
                    destination = k

        if len(nodes_id) == 0:
            nodes_id[origin] = 0
        if origin not in nodes_id.keys():
            nodes_id[origin] = max(nodes_id.values()) + 1
        if destination not in nodes_id.keys():
            nodes_id[destination] = max(nodes_id.values()) + 1

        output_edge.append(nodes_id[origin])
        input_edge.append(nodes_id[destination])

    if input_dict['random'] is not '':
        random_out = random.choices(output_edge, k=round(len(output_edge)*input_dict['random']))
        random_in = random.choices(input_edge, k=round(len(input_edge)*input_dict['random']))
        
        output_edge = output_edge + random_out
        input_edge = input_edge + random_in

    genes_index = {}
    nodes_genes_index = {}
    gene_ordering_index = set()
    for i in range(0, len(nodes_id)):
        entity = list(nodes_id.keys())[i]
        if entity in entities_dict['Gene']:
            genes_index[entity] = list(symbols).index(entity)
            nodes_genes_index[entity] = i
            gene_ordering_index.add(i)

    return dict(
        input_dict,
        subgraph=triples,
        nodes_id=nodes_id,
        output_edges=output_edge,
        input_edges=input_edge,
        var_names=symbols,
        de_names=de_symbols,
        genes_index=genes_index,
        nodes_genes_index=nodes_genes_index,
        gene_ordering_index=gene_ordering_index,
        official_names_drugs=harmonized_drugs_dataset,
    )


def create_cell_graph(input_dict):

    """
    Creates a graph object using gene-drug connections. There's a graph per cell.
    The drugs have as node features its doses and the genes the expression.
    :param input_dict:
    :return:
    """

    features = input_dict['gene_expression']
    entities_dict = input_dict['entities_dict']
    condition = input_dict['condition']
    dose = input_dict['dose']
    var_names = input_dict['var_names']
    nodes_id = input_dict['nodes_id']
    output_edge = input_dict['output_edges']
    input_edge = input_dict['input_edges']
    genes_index = input_dict['genes_index']
    gene_ordering_index = input_dict['gene_ordering_index']
    

    node_features = []

    # Use as node features the gene expression data if the node is a gene
    for entity in nodes_id:
        if entity in genes_index:  # if it's a Gene, then ID 0, gene expression and PertFlag
            # hashing a set is average O(1)
            index = genes_index[entity]
            feature = [float(0.0), float(features[index]),
                       float(1.0) if entity in condition.split('+') else float(0.0)]
            node_features.append(feature)
        else:  # if it's a Drug, then ID 1, Dose and PertFlag
            
            feature = [float(1.0), float(dose) if entity in condition.split('+') else float(0.0),
                       float(1.0) if entity in condition.split('+') else float(0.0)]
            node_features.append(feature)
    # last feature is a flag indicating if that gene or drug is the perturbation

    edge_index = torch.tensor([output_edge,
                               input_edge], dtype=torch.long)
    x = torch.tensor(node_features, dtype=torch.float)
    data = Data(x=x, edge_index=edge_index)

    return data


def handle_features(input_dict):

    genes = input_dict['genes']
    doses = input_dict['doses']
    condition = input_dict['condition']
    nodes_id = input_dict['nodes_id']
    genes_index = input_dict['genes_index']
    official_names_drug = input_dict['official_names_drugs']

    # Create perturbation dictionary
    pert_dict = {}
    j = 0
    for pert in condition:
        pert = pert.split('+')
        for element in pert:
            if element not in pert_dict.keys():
                pert_dict[element] = []
                pert_dict[element].append(j)
            else:
                pert_dict[element].append(j)

        j = j + 1

    # Initialize to 0 all node features
    node_features = torch.zeros([genes.shape[0], len(nodes_id), 3], dtype=torch.float)

    i = 0
    for entity in nodes_id:
        if entity in genes_index:  # O(1) hashing a dictionary, if it's a gene
            index = genes_index[entity]
            node_features[:, i, 1] = genes[:, index]
            node_features[:, i, 0] = torch.ones([genes.shape[0], ])
            if entity in pert_dict:  # if perturbation
                node_features[list(pert_dict[entity]), i, 2] = torch.ones(size=node_features[list(pert_dict[entity]), i, 2].shape, dtype=torch.float)
        elif entity in official_names_drug.keys():  # if it's not a gene it's a drug
            if entity in pert_dict:  # if it's perturbation on PertFlag and write dose, otherwise all 0
                node_features[list(pert_dict[entity]), i, 2] = torch.ones(size=node_features[list(pert_dict[entity]), i, 2].shape, dtype=torch.float)
                node_features[list(pert_dict[entity]), i, 1] = node_features[list(pert_dict[entity]), i, 2]*torch.tensor(doses[list(pert_dict[entity])].values.astype('float'), dtype=node_features[list(pert_dict[entity]), i, 1].dtype)
        i = i + 1
    # return the matrix with all node_features
    return node_features


def new_create_cell_graph(input_dict):

    node_features = input_dict['node_features']
    output_edge = input_dict['output_edges']
    input_edge = input_dict['input_edges']

    edge_index = torch.tensor([output_edge,
                               input_edge], dtype=torch.long)
    data = Data(x=node_features, edge_index=edge_index)

    return data