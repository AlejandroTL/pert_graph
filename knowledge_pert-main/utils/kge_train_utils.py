import os
import random
import numpy as np
from typing import List

import pykeen.nn
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory

import torch
from torch.utils.tensorboard import SummaryWriter


def random_choices():
    """
    Random choice of hyperparameters

    Return
    -----
    dictionary with options :class: dict
    """

    model = random.choice(['TransE', 'ComplEx', 'DistMult', 'RGCN', 'RotatE'])
    loss = random.choice(['softplus', 'marginranking', 'mse'])
    batch_size = random.choice([128, 256, 512, None])
    epochs = random.choice([50, 100, 150, 200, 250])
    embedding = random.choice([64, 128, 256])
    if model == 'RGCN':
        sampler = "schlichtkrull"
        layers = random.choice([2, 3])
    else:
        sampler = None
        layers = None

    return dict(
        model=model,
        loss=loss,
        batch_size=batch_size,
        epochs=epochs,
        sampler=sampler,
        layers=layers,
        embedding=embedding
    )


def random_train(n, input_dict):

    """
    Train n number of models with different random choices
    stores the result on tb-logs

    :param n: number of models to train
    :return:
    """

    path = input_dict['triples_path']
    triples = np.loadtxt(path, dtype=str, delimiter='\t')  # Load triples
    tf = TriplesFactory.from_labeled_triples(triples)
    ratios = [0.7, 0.15, 0.15]
    training, testing, validation = tf.split(ratios, random_state=42)  # split in train and test

    unique_params = []  # select hyperparameters randomnly

    for _ in range(0, n+1):
        ini_dict = random_choices()
        while list(ini_dict.values()) in unique_params:  # don't repeat hyperparameters combinations
            ini_dict = random_choices()
        unique_params.append(list(ini_dict.values()))

        hyper_dict = dict(
            ini_dict,
            train=training,
            test=testing,
            validation=validation
        )

        train_routine(hyper_dict)


def selection_train(input_dict):

    """
    Train selected model with selected hyperparameters and loss function
    :param input_dict:
    :return:
    """

    path = input_dict['triples_path']
    triples = np.loadtxt(path, dtype=str, delimiter='\t')  # Load triples
    tf = TriplesFactory.from_labeled_triples(triples)
    ratios = [0.7, 0.15, 0.15]
    training, testing, validation = tf.split(ratios, random_state=42)  # split in train and test

    hyper_dict = dict(
        input_dict,
        train=training,
        test=testing,
        validation=validation
    )

    train_routine(hyper_dict)


def train_routine(hyper_dict):

    """
    Training routine with pipeline

    :param hyper_dict: dictionary with input data and hyperparameters
    :return: output in tb-logs
    """
    training = hyper_dict['train']
    testing = hyper_dict['test']
    validation = hyper_dict['validation']
    model = hyper_dict['model']
    loss = hyper_dict['loss']
    batch_size = hyper_dict['batch_size']
    epochs = hyper_dict['epochs']
    sampler = hyper_dict['sampler']
    layers = hyper_dict['layers']  # not used at this moment, hyperparam just for RGCN
    embedding = hyper_dict['embedding']
    name = hyper_dict['name']

    logdir = os.path.join('TB-KGE', name)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Define device

    results = pipeline(
        training=training,
        testing=testing,
        validation=validation,
        model=model,
        model_kwargs=dict(
            embedding_dim=embedding
        ),
        loss=loss,
        training_loop='sLCWA',
        negative_sampler='basic',
        result_tracker='tensorboard',
        result_tracker_kwargs=dict(
            experiment_path=logdir,
        ),
        training_kwargs=dict(
            num_epochs=epochs,
            batch_size=batch_size,
            sampler=sampler),
        stopper='early',
        stopper_kwargs=dict(frequency=5,
                            patience=10,
                            relative_delta=0.05,
                            metric='adjusted_mean_rank_index'),
        random_seed=42,
        device=device
    )

    path = 'trained_models/KG_models'
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(results.model, os.path.join(path, '{}.plk'.format(name)))
    # Store loss plot in TensorBoard, I didn't find a better way (yet)
    writer = SummaryWriter(logdir)
    for epoch in range(len(results.losses)):
        writer.add_scalar('{}/Test'.format(loss), results.losses[epoch], epoch)

    # Store entity embeddings in Tensorboard too, predicate embeddings are dropped
    model = results.model

    # RepresentationsModule is a base class of nn.Embeddings
    entity_representation_modules: List['pykeen.nn.RepresentationModule'] = model.entity_representations

    # nn.Embedding inherits from nn.Module
    entity_embeddings: pykeen.nn.Embedding = entity_representation_modules[0]

    # so forward() is needed
    entity_embedding_tensor: torch.FloatTensor = entity_embeddings()

    writer.add_embedding(entity_embedding_tensor)

