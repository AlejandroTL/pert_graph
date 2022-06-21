import argparse
import yaml

from data_handling.process_data import *
from utils.kge_train_utils import *

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", help="path of YAML config file")
    args = parser.parse_args()

    with open(args.config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Parse the input parameters
    selection_dict = dict(
        model=config['parameters']['KnowledgeGraph']['model'],
        loss=config['parameters']['KnowledgeGraph']['loss'],
        batch_size=config['parameters']['KnowledgeGraph']['batch_size'],
        epochs=config['parameters']['KnowledgeGraph']['epochs'],
        sampler=config['parameters']['KnowledgeGraph']['sampler'],
        layers=config['parameters']['KnowledgeGraph']['layers'],
        embedding=config['parameters']['KnowledgeGraph']['z_dim'],
        name=config['name']
    )

    # Avoid generating data twice and wasting time
    standard_path = os.path.join('omnipath_triples/', 'triples')
    if not os.path.isfile(standard_path):  # If triples don't exist yet
        # Process the data and generate files for both PyKeen and Neo4j
        processed = omnipath_to_triples()
        omnipath_to_neo4j(processed)
        training_dict = dict(
            selection_dict,
            triples_path=processed['triples_path']
        )
    else:  # If triples exists on the standard path, just pass it
        training_dict = dict(
            selection_dict,
            triples_path=standard_path
        )

    # Train
    selection_train(training_dict)
