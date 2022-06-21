from data_handling.GCN_dataset import GeneExpressionGraph
from utils.kge_train_utils import *


import scanpy as sc

if __name__ == '__main__':

    standard_path = os.path.join('omnipath_triples/', 'triples_unfiltered')
    entities_path = os.path.join('omnipath_triples/', 'entities.json')  # Avoid generating data twice and wasting time

    drugs_path = '/home/aletl/Documents/knowldege_graph-/omnipath_triples/drug_names.json'
    data_path = '/home/aletl/Documents/datasets/GSM_new_allgenes.h5ad'
    adata = sc.read(data_path)

    # Ad hoc dictionary with equivalences of names, there's no other way of doing it
    drugs_official_name = {'Nutlin': 'nutlin-3',
                           'Dex': 'dexamethasone',
                           'Vehicle': 'vehicle',
                           'BMS': 'bms-345541',
                           'SAHA': 'saha',
                           }

    data_dict = dict(
        drugs_official=drugs_official_name,
        adata=adata,
        drugs=adata.obs['condition'].unique(),
        condition=adata.obs['condition'],
        dose=adata.obs['dose'],
        perturbation='drug',
        split=adata.obs['split'],
        compose=False,
        triples_path=standard_path,
        entities_path=entities_path,
        drugs_path=drugs_path,
    )

    GeneExpressionGraph(root='CellGraph_Data/', dict_in=data_dict, transform=None)
