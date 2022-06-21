import argparse

from data_handling.process_data import *
from data_handling.GCN_dataset import DrugGeneDataset
from utils.kge_train_utils import *

"""
Reads a trained KG and creates the subgraphs of drugs of interests to read train the GNN model
"""

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--kge_model", help="KGE Model Path")

    args = parser.parse_args()

    kge_model = torch.load(args.kge_model)  # KGE trained model to generate the embeddings that will be node features
    entity_embeddings = kge_model.entity_representations[0]

    standard_path = os.path.join('omnipath_triples/', 'triples')
    entities_path = os.path.join('omnipath_triples/', 'entities.json')  # Avoid generating data twice and wasting time
    if not os.path.isfile(standard_path):  # If entities dict don't exist yet
        # Process the data and generate files for both PyKeen and Neo4j
        processed = omnipath_to_triples()
        omnipath_to_neo4j(processed)
        training_dict = dict(
            processed,
            entity_embeddings=entity_embeddings,
        )
    else:  # If entities dictionary exists on the standard path, just pass it
        training_dict = dict(
            triples_path=standard_path,
            entities_path=entities_path,
            entity_embeddings=entity_embeddings,
        )

    DrugGeneDataset(root='GNN_Data/', dict_in=training_dict, transform=None)




