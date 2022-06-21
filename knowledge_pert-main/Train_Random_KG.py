import argparse

from data_handling.process_data import *
from utils.kge_train_utils import *

"""
Train n random models and store the results on TB
"""

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--n", help="random models to train", default=5)

    args = parser.parse_args()

    n = int(args.n)

    standard_path = os.path.join('omnipath_triples/', 'triples')  # Avoid generating data twice and wasting time
    if not os.path.isfile(standard_path):  # If triples don't exist yet
        # Process the data and generate files for both PyKeen and Neo4j
        processed = omnipath_to_triples()
        omnipath_to_neo4j(processed)
    else:  # If triples exists on the standard path, just pass it
        processed = dict(
            triples_path=standard_path
        )

    # Train
    random_train(n, processed)

