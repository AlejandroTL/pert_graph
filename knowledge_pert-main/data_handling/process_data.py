import omnipath as op
import numpy as np
import pandas as pd
import os
import argparse
from pykeen.triples import TriplesFactory
import json
import math
import scanpy as sc

from difflib import SequenceMatcher


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


def official_names_drugs(input_dict):

    adata = sc.read(input_dict['data_path'])

    with open(input_dict['drug_path']) as json_file:
        drug_names = json.load(json_file)

    values = []
    for key, value in drug_names.items():
        values = values + [key] + value

    values = set(values)

    drugs_dict = {}
    for i in adata.obs['condition'].unique():
        if i.lower() in values:
            drugs_dict[i] = i.lower()

    aux_path = os.path.basename(os.path.normpath(input_dict['data_path'])).split('.')[0]
    entities_path = "omnipath_triples/official_drug_names/" + aux_path + ".json"
    # create json object from dictionary
    entities_json = json.dumps(drugs_dict)
    f = open(entities_path, "w")
    f.write(entities_json)
    f.close()


def harmonize_names(input_dict):

    """
    Harmonize the names of the drugs of the anndata
    :param input_dict:
    :return:
    """

    drug_names_path = input_dict['drugs_path']  # to know which are the drugs
    drug_to_harmonize = input_dict['drugs']
    with open(drug_names_path) as json_file:
        drug_names = json.load(json_file)

    for key in drug_to_harmonize:
        for drug in list(drug_names):
            if drug in drug_names.keys() and key in drug_names.keys():
                if key != drug:
                    if similar(key, drug) > 0.9:
                        print(key, drug, similar(key, drug))
                        drug_names[key].extend(drug_names[drug])
                        drug_names.pop(drug, None)

    return dict(
        input_dict,
        harmonized_drugs=drug_names
    )


def omnipath_to_triples(filter=True):

    """
    Take the OmniPath transcriptional network and process it to build a Knowledge Graph

    Parameters
    -----
    No one yet, maybe paths later

    Return
    -----
    Dictionary with two elements
        df: Cleaned dataframe :class: pd.DataFrame
        path_triples: path
    """

    # Build a dictionary to differentiate drugs and genes
    entities = {'Gene': [], 'Drug': []}

    interactions = op.interactions.AllInteractions.get(dorothea_levels=["A", "B", "C"], organism='human',
                                                       genesymbols=True)

    features = ['source', 'target', 'source_genesymbol', 'target_genesymbol', 'is_stimulation', 'is_inhibition']
    kgg = interactions[features]

    # Filter all unknown relationships, maintain just stimulation and inhibition
    kgg = kgg[(kgg['is_stimulation'] == True) | (kgg['is_inhibition'] == True)]

    # Create pd.DataFrame of triples s,p,o
    KG = pd.DataFrame(columns=['subject', 'predicate', 'object'])
    i = 0
    for index, row in kgg.iterrows():
        if row['is_stimulation']:
            KG.loc[i, :] = [row['source_genesymbol'], 'stimulates', row['target_genesymbol']]
        else:
            KG.loc[i, :] = [row['source_genesymbol'], 'inhibits', row['target_genesymbol']]
        if row['source_genesymbol'] not in entities['Gene']:
            entities['Gene'].append(row['source_genesymbol'])
        if row['target_genesymbol'] not in entities['Gene']:
            entities['Gene'].append(row['target_genesymbol'])
        i += 1

    # Now add drug-gene interactions
    drug_target = pd.read_csv('https://www.dgidb.org/data/monthly_tsvs/2022-Feb/interactions.tsv', sep='\t')
    drug_target = drug_target[drug_target['gene_name'].notna()] # clean gene_name NANs

    # Some drugs have different names, build a dictionary to make a reference
    drug_references = {}

    # Create dictionary to simplify complex relationships in just two groups
    # the interactions will be just stimulates and inhibits
    drug_gene_dict = {'inhibits': ['inhibitor', 'blocker', 'antagonist', 'antibody', 'inverse agonist',
                                   'negative modulator', 'antisense oligonucleotide', 'antagonist,allosteric modulator',
                                   'blocker, inhibitor', 'inhibitory allosteric modulator', 'supressor',
                                   'antagonist,inhibitor'],
                      'stimulates': ['agonist', 'vaccine', 'partial agonist', 'inducer',
                                     'agonist,allosteric modulator']}

    for index, row in drug_target.iterrows():
        if row['interaction_types'] in drug_gene_dict['stimulates']:
            KG.loc[i, :] = [row['drug_claim_primary_name'], 'stimulates', row['gene_name']]
        if row['interaction_types'] in drug_gene_dict['inhibits']:
            KG.loc[i, :] = [row['drug_claim_primary_name'], 'inhibits', row['gene_name']]
        if (filter is False) and (row['interaction_types'] != row['interaction_types']):
            KG.loc[i, :] = [row['drug_claim_primary_name'], 'unknown', row['gene_name']]
        if row['drug_claim_primary_name'] not in entities['Drug']:
            entities['Drug'].append(row['drug_claim_primary_name'])
        if row['gene_name'] not in entities['Gene']:
            entities['Gene'].append(row['gene_name'])

        if str(row['drug_claim_primary_name']).lower() not in drug_references.items():
            drug_references[str(row['drug_claim_primary_name']).lower()] = []
        drug_references[str(row['drug_claim_primary_name']).lower()].append(str(row['drug_claim_primary_name']).lower())
        drug_references[str(row['drug_claim_primary_name']).lower()].append(str(row['drug_claim_name']).lower())
        drug_references[str(row['drug_claim_primary_name']).lower()].append(str(row['drug_name']).lower())
        i += 1
        
    # Add PPI data
    df = pd.read_csv('dataset_PPI.csv', header=None, sep=' ')
    df = df[[3,4]]
    
    for index, row in df.iterrows():
        KG.loc[i, :] = [row[3], 'unknown', row[4]]
        if row[3] not in entities['Gene']:
            entities['Gene'].append(row[3])
        if row[4] not in entities['Gene']:
            entities['Gene'].append(row[4])
        i += 1
        
    # Store the data
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'omnipath_triples'))
    if not os.path.exists(path):
        os.makedirs(path)
    if filter is True:
        triples_path = os.path.join(path, 'triples')
    else:
        triples_path = os.path.join(path, 'triples_unfiltered_all')
    np.savetxt(triples_path, KG, fmt='%s', delimiter='\t')

    # Store also entities dict to read it without running all this function again

    entities_path = os.path.join(path, 'entities.json')
    # create json object from dictionary
    entities_json = json.dumps(entities)
    f = open(entities_path, "w")
    f.write(entities_json)
    f.close()

    # Store also the drug references dict to read it without running all this function again

    drug_names_path = os.path.join(path, 'drug_names.json')
    # create json object from dictionary
    drug_names_json = json.dumps(drug_references)
    f = open(drug_names_path, "w")
    f.write(drug_names_json)
    f.close()


    # Ablation study requires data in three separated folders

    triples = np.loadtxt(triples_path, dtype=str, delimiter='\t')  # Load triples
    tf = TriplesFactory.from_labeled_triples(triples)
    ablation_path = os.path.join(path, 'ablation/')
    if not os.path.exists(ablation_path):
        os.makedirs(ablation_path)
    ratios = [0.7, 0.15, 0.15]
    training, testing, validation = tf.split(ratios, random_state=42)
    np.savetxt(os.path.join(ablation_path, 'training'), training.mapped_triples, fmt='%s', delimiter='\t')
    np.savetxt(os.path.join(ablation_path, 'testing'), testing.mapped_triples, fmt='%s', delimiter='\t')
    np.savetxt(os.path.join(ablation_path, 'validation'), validation.mapped_triples, fmt='%s', delimiter='\t')

    return dict(
        df=KG,
        triples_path=triples_path,
        entities_path=entities_path
    )


def omnipath_to_neo4j(input_dict):

    """
    Take cleaned dataframe and build files for Neo4J

    Parameters
    -----
    Cleaned dataframe :class: pd.DataFrame

    Return
    -----
    outputs two files (for this Dorothea OmniPath data)
    """
    file = input_dict['df']
    entities_path = input_dict['entities_path']
    with open(entities_path) as json_file:
        entities = json.load(json_file)

    # Split the df in two, one for drug-gene and other for gene-gene
    drugs_genes, genes_genes = [x for _, x in file.groupby(file['subject'].isin(entities['Gene']))]

    path = os.path.join(os.path.dirname(__file__), '..', 'omnipath_triples')
    if not os.path.exists(path):
        os.makedirs(path)

    path_inhibits_drug = os.path.join(path, 'Neo4j_triples_inhibits_drug.csv')
    path_stimulates_drug = os.path.join(path, 'Neo4j_triples_stimulates_drug.csv')
    path_inhibits_gene = os.path.join(path, 'Neo4j_triples_inhibits_gene.csv')
    path_stimulates_gene = os.path.join(path, 'Neo4j_triples_stimulates_gene.csv')

    drugs_genes[drugs_genes['predicate'] == 'inhibits'].to_csv(path_inhibits_drug)
    drugs_genes[drugs_genes['predicate'] == 'stimulates'].to_csv(path_stimulates_drug)
    genes_genes[genes_genes['predicate'] == 'inhibits'].to_csv(path_inhibits_gene)
    genes_genes[genes_genes['predicate'] == 'stimulates'].to_csv(path_stimulates_gene)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--f", action="store_true")
    parser.set_defaults(f=False)

    args = parser.parse_args()
    
    print(args.f)

    processed = omnipath_to_triples(filter=args.f)
    omnipath_to_neo4j(processed)
