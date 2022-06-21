import argparse
import yaml

from data_handling.process_data import *
from data_handling.graph_data import *
from utils.kge_train_utils import *
from utils.utils import *


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", help="path of YAML config file")
    args = parser.parse_args()

    with open(args.config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # If model previously trained
    if config['parameters']['KnowledgeGraph']['prev'] is not None:
        kge_model = torch.load(config['parameters']['KnowledgeGraph']['prev'])  # KGE trained model
        entity_embeddings = kge_model.entity_representations[0]
    else:  # If not previously train model, train one
        # Input parameters
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

        # Prepare data
        standard_path = os.path.join('omnipath_triples/', 'triples')  # Avoid generating data twice and wasting time
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

        # Load recently trained model
        kge_model = torch.load(os.path.join('trained_models/KG_models', '{}.plk'.format(config['name'])))  # KGE trained model
        entity_embeddings = kge_model.entity_representations[0]

    # The KG is already trained, now we iterate over the drugs to generate all subgraph embeddings

    # If previous data doesn't exist, generate them
    if config['parameters']['KnowledgeGraph']['prev_data'] is None:
        graph_dict = dict(
            triples_path=os.path.join('omnipath_triples/', 'triples'),
            entity_embeddings=entity_embeddings
        )
        entities_path = os.path.join('omnipath_triples/', 'entities.json')
        with open(entities_path) as json_file:
            entities = json.load(json_file)
        subgraph_embeddings = []
        for drug in entities['Drug']:  # create subgraph per each drug
            induced_drug_subgraph = drug_subgraph(drug, graph_dict)
            subgraph_embedding = mean_pool(induced_drug_subgraph)
            subgraph_embeddings.append(subgraph_embedding['graph_embedding'])

    else:
        path = osp.join('omnipath_triples','triples')
        triples = np.loadtxt(path, dtype=str, delimiter='\t')  # Load triples
        tf = TriplesFactory.from_labeled_triples(triples)
        subgraph_embeddings = []
        for file in os.listdir(config['parameters']['KnowledgeGraph']['prev_data']):
            induced_drug_subgraph = pd.read_pickle(osp.join(config['parameters']['KnowledgeGraph']['prev_data'], file))
            induced_subgraph = dict(
                subgraph=induced_drug_subgraph,
                embeddings=entity_embeddings,
                TriplesFactory=tf
            )
            subgraph_embedding = mean_pool(induced_subgraph)
            subgraph_embeddings.append(subgraph_embedding['graph_embedding'])

        entire_subgraph_embeddings = torch.stack(subgraph_embeddings, 0)
        writer = SummaryWriter(os.path.join('TB-KGE', config['name']))
        writer.add_embedding(entire_subgraph_embeddings, tag='Subgraph Embedding')

