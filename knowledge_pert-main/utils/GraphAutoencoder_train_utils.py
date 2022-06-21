from enum import auto
import time
from collections import defaultdict
from tqdm import tqdm

from utils.utils import *
from models.net import *

from torch_geometric.loader import DataLoader

from sklearn.metrics import r2_score, balanced_accuracy_score, make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

criterion_mse = nn.MSELoss(reduction='sum')
criterion_bce = nn.CrossEntropyLoss(reduction='sum')


def pjson(s):
    """
    Prints a string in JSON format and flushes stdout
    """
    print(json.dumps(s), flush=True)


class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=5, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True


def train(model, train_loader, optimizer, device):

    """
    Simple train function
    :param model: model to train
    :param train_loader:  dataloader
    :param optimizer: optimizer
    :param device: cpu or cuda
    :return: loss
    """

    model.train()
    loss_t = torch.Tensor([0]).to(device)
    for data in train_loader:
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        batch = data.batch.to(device)
        optimizer.zero_grad()
        encoded_node_features, z1, graph_embedding_decoded, decoded_node_features = model(x, edge_index, batch)
        loss = criterion_mse(decoded_node_features, x)
        loss_t = loss
        print("Batch loss: ", loss_t)
        loss_t.backward()
        optimizer.step()

    return float(loss_t)


@torch.no_grad()
def test(model, test_loader, device):

    """
    Simple validation function
    :param model: model to train
    :param test_loader: dataloader
    :param device: cuda or cpu
    :return: test loss
    """

    model.eval()
    loss_t = torch.Tensor([0]).to(device)
    for data, drug, cov in test_loader:
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        batch = data.batch.to(device)
        a, b, c, d = model(x, edge_index, batch)
        loss = criterion_mse(d[:, 1:3], x[:, 1])
        loss_bce = criterion_bce(d[:, 0], x[:, 0])
        loss_bce2 = criterion_bce(d[:, -1], x[:, -1])
        loss_t = loss + loss_bce + loss_bce2

    return float(loss_t)


def train_routine(model, config, datasets, return_model):

    """
    Entire training routine with callbacks and loggings
    :param model: Model to train, defined in the main script
    :param config: dictionary with the training parameters
    :param datasets: datasets to train, previoulsy splitted
    :param return_model
    :return:
    """

    train_loader = DataLoader(datasets["training"],
                              batch_size=config['training']['GNN']['batch_size'],
                              num_workers=4,
                              pin_memory=True,
                              drop_last=True
                              )
    
    genes_index_all = []
    for i in range(0, config['training']['GNN']['batch_size']):
        genes_index_all = genes_index_all + [element + i*model.n_nodes for element in model.genes_index]

    if model.adversarial is False:
        test_loader = DataLoader(datasets["test"],
                                 batch_size=config['training']['GNN']['batch_size'],
                                 num_workers=4,
                                 pin_memory=True
                                 )

    device = model.device

    args = config['training']['GNN']
    pjson({"training_args": args})
    pjson({"autoencoder_params": model.hparams})

    start_time = time.time()
    for epoch in range(config['training']['GNN']['epochs']+1):
        print("Epoch: ", epoch)
        # At each epoch, initialize the losses at 0
        epoch_training_stats = defaultdict(float)
        epoch_testing_stats = defaultdict(float)

        for data, drug, cov in tqdm(train_loader):
            x = data.x.to(device)
            edge_index = data.edge_index.to(device)
            batch = data.batch.to(device)
            drug = drug.to(device)
            cov = cov.to(device)
            minibatch_training_stats = model.update(x, edge_index, batch, drug, cov, genes_index_all, epoch)
            
            del x, edge_index, batch, drug, cov
            torch.cuda.empty_cache()

            for key, val in minibatch_training_stats.items():
                epoch_training_stats[key] += val

        for key, val in epoch_training_stats.items():
            epoch_training_stats[key] = val
            if not (key in model.history.keys()):
                model.history[key] = []
            model.history[key].append(val)
        model.history['epoch'].append(epoch)
        
        if model.adversarial is False:        
            for data, drug, cov in tqdm(test_loader):
                x = data.x.to(device)
                edge_index = data.edge_index.to(device)
                batch = data.batch.to(device)
                drug = drug.to(device)
                cov = cov.to(device)
                minibatch_testing_stats = model.update(x, edge_index, batch, drug, cov, genes_index_all, epoch)
                
                del x, edge_index, batch, drug, cov
                torch.cuda.empty_cache()

                for key, val in minibatch_testing_stats.items():
                    epoch_testing_stats[key] += val
                    
            for key, val in epoch_testing_stats.items():
                epoch_testing_stats[key] = val
                if not (key+"_test" in model.history.keys()):
                    model.history[key+"_test"] = []
                model.history[key+"_test"].append(val)

        ellapsed_minutes = (time.time() - start_time) / 60
        model.history['elapsed_time_min'] = ellapsed_minutes

            # decay learning rate if necessary
            # also check stopping condition: patience ran out OR
            # time ran out OR max epochs achieved
        stop = ellapsed_minutes > args["max_minutes"] or \
               (epoch == args["epochs"] - 1)

        if (epoch % args["checkpoint_freq"]) == 0 or stop:
            if args["evaluation"]:
                evaluation_stats = evaluate(model, datasets)
                for key, val in evaluation_stats.items():
                    if not (key in model.history.keys()):
                        model.history[key] = []
                    model.history[key].append(val)
                model.history['stats_epoch'].append(epoch)

                pjson({
                    "epoch": epoch,
                    "training_stats": epoch_training_stats,
                    "evaluation_stats": evaluation_stats,
                    "ellapsed_minutes": ellapsed_minutes
                })
            else:
                evaluation_stats = {
            "training": [0, 0, 0, 0],
            "test": [0, 0, 0, 0],
            "ood": [0, 0, 0, 0],
            "optimal for perturbations": [0],
            "covariate disentanglement": [0],
            "optimal for covariates": [0],
            }
                model.train()
                for key, val in evaluation_stats.items():
                    if not (key in model.history.keys()):
                        model.history[key] = []
                    model.history[key].append(val)
                model.history['stats_epoch'].append(epoch)

            if not os.path.exists(args["save_dir"]):
                os.makedirs(args["save_dir"])
            #torch.save(
            #    (model.state_dict(), args, model.history),
            #    os.path.join(
            #        args["save_dir"],
            #        "model={}_seed={}_epoch={}.pt".format(args["name"], args["seed"], epoch)))

            pjson({"model_saved": "model={}_seed={}_epoch={}.pt\n".format(
                args["name"], args["seed"], epoch)})
            stop = stop or model.early_stopping(
                np.mean(evaluation_stats["test"]))
            if stop:
                pjson({"early_stop": epoch})
                break

    if return_model:
        return model


def evaluate_disentanglement(autoencoder, dataset, nonlinear=False):
    """
    Given a ComPert model, this function measures the correlation between
    its latent space and 1) a dataset's drug vectors 2) a datasets covariate
    vectors.
    """

    test_dataloader = DataLoader(dataset,
                                 batch_size=128, num_workers=4)

    latent_basal_total = torch.empty((0,autoencoder.hparams['architecture']['GNN']["z_dim"])).to(autoencoder.device) 
    
    for genes_control in test_dataloader: # cannot load all the data because of CUDA OOM

        encode_outputs = autoencoder.encode_graph(
            genes_control[0].x.to(autoencoder.device),
            genes_control[0].edge_index.to(autoencoder.device),
            genes_control[0].batch.to(autoencoder.device))
        
        latent_basal = encode_outputs['latent_basal']
        latent_basal_total = torch.cat((latent_basal_total, latent_basal), dim=0)

    latent_basal = latent_basal_total.detach().cpu().numpy()

    if nonlinear:
        clf = KNeighborsClassifier(
            n_neighbors=int(np.sqrt(len(latent_basal))))
    else:
        clf = LogisticRegression(solver="liblinear",
                                 multi_class="auto",
                                 max_iter=10000)

    pert_scores = cross_val_score(
        clf,
        StandardScaler().fit_transform(latent_basal), dataset.drugs_names,
        scoring=make_scorer(balanced_accuracy_score), cv=5, n_jobs=-1)

    if len(np.unique(dataset.cell_types_names)) > 1:
        cov_scores = cross_val_score(
            clf,
            StandardScaler().fit_transform(latent_basal), dataset.cell_types_names,
            scoring=make_scorer(balanced_accuracy_score), cv=5, n_jobs=-1)
        return np.mean(pert_scores), np.mean(cov_scores)
    else:
        return np.mean(pert_scores), 0


def complex_evaluate_disentanglement(autoencoder, dataset, nonlinear=False):
    """
    Given a ComPert model, this function measures the correlation between
    its latent space and 1) a dataset's drug vectors 2) a datasets covariate
    vectors.
    """

    test_dataloader = DataLoader(dataset,
                                 batch_size=128, num_workers=4)

    latent_basal_total = torch.empty((0,autoencoder.hparams['architecture']['Global']["z_dim"])).to(autoencoder.device) 
    
    for genes_control in test_dataloader: # cannot load all the data because of CUDA OOM
        
        inputs = dict(
                    graph_x=genes_control[0].x.to(autoencoder.device),
                    graph_edge=genes_control[0].edge_index.to(autoencoder.device),
                    graph_batch=genes_control[0].batch.to(autoencoder.device),
                    fc_genes=genes_control[1].to(autoencoder.device),
                    fc_drugs=genes_control[2].to(autoencoder.device),
                    fc_covariates=genes_control[3].to(autoencoder.device)
                )

        encode_outputs = autoencoder.encode(
            inputs
            )
        
        aggregation_outputs = autoencoder.emb_aggregation(encode_outputs)
        latent_basal = aggregation_outputs['total_basal']
        
        latent_basal_total = torch.cat((latent_basal_total, latent_basal), dim=0)

    latent_basal = latent_basal_total.detach().cpu().numpy()

    if nonlinear:
        clf = KNeighborsClassifier(
            n_neighbors=int(np.sqrt(len(latent_basal))))
    else:
        clf = LogisticRegression(solver="liblinear",
                                 multi_class="auto",
                                 max_iter=10000)

    pert_scores = cross_val_score(
        clf,
        StandardScaler().fit_transform(latent_basal), dataset.drugs_names,
        scoring=make_scorer(balanced_accuracy_score), cv=5, n_jobs=-1)

    if len(np.unique(dataset.cell_types_names)) > 1:
        cov_scores = cross_val_score(
            clf,
            StandardScaler().fit_transform(latent_basal), dataset.cell_types_names,
            scoring=make_scorer(balanced_accuracy_score), cv=5, n_jobs=-1)
        return np.mean(pert_scores), np.mean(cov_scores)
    else:
        return np.mean(pert_scores), 0


def evaluate_r2(autoencoder, dataset, dataset_control):
    """
    Measures different quality metrics about an ComPert `autoencoder`, when
    tasked to translate some `genes_control` into each of the drug/cell_type
    combinations described in `dataset`.
    Considered metrics are R2 score about means and variances for all genes, as
    well as R2 score about means and variances about differentially expressed
    (_de) genes.
    """

    test_control_dataloader = DataLoader(dataset_control,
                                         batch_size=128, num_workers=4)

    mean_score, var_score, mean_score_de, var_score_de = [], [], [], []

    num = 128
    num_nodes = len(dataset_control.nodes_id)
    
    # Create dictionary with the index of the genes in the table, just once
    index_table = dict()
    for gen in dataset.symbols:
        index_table[gen] = list(dataset.symbols).index(gen)

    for pert_category in tqdm(np.unique(dataset.pert_categories)):
        # pert_category category contains: 'celltype_perturbation_dose' info
        de_idx_one_graph = []  # find index of DE genes in the graph
        de_idx_table = []  # find index of DE in the table
        if pert_category not in dataset.de_symbols.keys():
            continue  # if pert category is not a pert category
        for de_gen in np.array(dataset.de_symbols[pert_category]):
            if de_gen in set(dataset.nodes_names):  # access to set in constant time
                de_idx_one_graph.append(dataset.nodes_id[de_gen])
                de_idx_table.append(index_table[de_gen])

        idx = np.where(dataset.pert_categories == pert_category)[0]

        if len(idx) > 30:
            emb_drugs = dataset.drugs[idx][0].view(
                1, -1).repeat(num, 1).clone()
            emb_cts = dataset.cell_types[idx][0].view(
                1, -1).repeat(num, 1).clone()
            
            emb_drugs = emb_drugs.to(autoencoder.device)
            emb_cts = emb_cts.to(autoencoder.device)
            
            nodes_predict_total = torch.empty((0,2)).to(autoencoder.device)
            for genes_control in test_control_dataloader:
                x = genes_control[0].x.to(autoencoder.device)
                edge_index = genes_control[0].edge_index.to(autoencoder.device)
                batch = genes_control[0].batch.to(autoencoder.device)

                if len(torch.unique(batch)) < 128: # last batch has no 128 examples, so dimension must change
                    emb_drugs = emb_drugs[:len(torch.unique(batch)),:]
                    emb_cts = emb_cts[:len(torch.unique(batch)),:]

                predict_outputs = autoencoder.predict(
                    x,
                    edge_index,
                    batch,
                    emb_drugs,
                    emb_cts
                )
                
                nodes_predict = predict_outputs['nodes_reconstructed']
                
                if autoencoder.adversarial:
                    nodes_predict_total = torch.cat((nodes_predict_total, nodes_predict), dim=0)
                else:
                    nodes_predict_total = torch.cat((nodes_predict_total, nodes_predict[0]), dim=0)
                    
                del x, edge_index, batch
                torch.cuda.empty_cache()
                
            del emb_drugs, emb_cts
            torch.cuda.empty_cache()
                
            nodes_predict = nodes_predict_total

            mean_predict = nodes_predict[:, 0]  # node feature with the mean
            var_predict = nodes_predict[:, 1]   # node feature with the var

            # we split the calculated nodes in the different graphs imputed
            mean_predict = mean_predict.reshape(-1, 1)
            auxiliar_mean = torch.split(mean_predict, num_nodes)
            mean_predict = torch.cat(auxiliar_mean, dim=1)
            # now we have a tensor of n_nodes x n_graphs
            var_predict = var_predict.reshape(-1, 1)
            auxiliar_var = torch.split(var_predict, num_nodes)
            var_predict = torch.cat(auxiliar_var, dim=1)

            # Just evaluate reconstruction on the genes nodes, the drug nodes we don't care
            #mean_predict = mean_predict[dataset.nodes_genes_index, :]
            #var_predict = var_predict[dataset.nodes_genes_index, :]

            # estimate metrics only for reasonably-sized drug/cell-type combos
            y_true = dataset.genes[idx, :].numpy()
            y_true_genes = y_true[:, dataset.nodes_index]  # take just the genes that are nodes of the graph

            # true means and variances
            yt_m = y_true.mean(axis=0)
            yt_v = y_true.var(axis=0)
            # predicted means and variances
            yp_m = mean_predict.mean(1).detach().cpu()
            yp_v = var_predict.mean(1).detach().cpu()

            mean_score.append(r2_score(yt_m[dataset.nodes_index], yp_m[dataset.nodes_genes_index]))
            var_score.append(r2_score(yt_v[dataset.nodes_index], yp_v[dataset.nodes_genes_index]))

            if len(de_idx_one_graph) > 0:
                
                mean_score_de.append(r2_score(yt_m[de_idx_table], yp_m[de_idx_one_graph]))
                var_score_de.append(r2_score(yt_v[de_idx_table], yp_v[de_idx_one_graph]))
            
                # Same process for DE genes
                # here we don't use just the genes nodes because we will filter the DE genes later
                # true means and variances
                #yt_m_de = y_true.mean(axis=0)
                #yt_v_de = y_true.var(axis=0)
                # predicted means and variances
                #mean_predict_de = nodes_predict[:, 0]  # node feature with the mean
                #var_predict_de = nodes_predict[:, 1]   # node feature with the var

                #mean_predict_de = mean_predict_de.reshape(-1, 1)
                #auxiliar_mean = torch.split(mean_predict_de, num_nodes)
                #mean_predict_de = torch.cat(auxiliar_mean, dim=1)
                #var_predict_de = var_predict_de.reshape(-1, 1)
                #auxiliar_var = torch.split(var_predict_de, num_nodes)
                #var_predict_de = torch.cat(auxiliar_var, dim=1)

                #mean_predict_de = mean_predict_de[de_idx_one_graph, :]
                #var_predict_de = var_predict_de[de_idx_one_graph, :]

                #yp_m_de = mean_predict_de.mean(1).detach().cpu()
                #yp_v_de = var_predict_de.mean(1).detach().cpu()

                #mean_score_de.append(r2_score(yt_m_de[de_idx_table], yp_m_de))
                #var_score_de.append(r2_score(yt_v_de[de_idx_table], yp_v_de))

    return [np.mean(s) if len(s) else -1
            for s in [mean_score, mean_score_de, var_score, var_score_de]]


def evaluate_r2_table(autoencoder, dataset, dataset_control):
    """
    R2 score evaluation function when the output is not a graph
    """
    
    test_control_dataloader = DataLoader(dataset_control,
                                         batch_size=128, num_workers=4)

    mean_score, var_score, mean_score_de, var_score_de = [], [], [], []

    num = 128
    num_nodes = len(dataset_control.nodes_id)
    
    # Create dictionary with the index of the genes in the table, just once
    index_table = dict()
    for gen in dataset.symbols:
        index_table[gen] = list(dataset.symbols).index(gen)

    for pert_category in tqdm(np.unique(dataset.pert_categories)):
        # pert_category category contains: 'celltype_perturbation_dose' info
        de_idx_one_graph = []  # find index of DE genes in the graph
        de_idx_table = []  # find index of DE in the table
        if pert_category not in dataset.de_symbols.keys():
            continue  # if pert category is not a pert category
        for de_gen in np.array(dataset.de_symbols[pert_category]):
            if de_gen in set(dataset.nodes_names):  # access to set in constant time
                de_idx_one_graph.append(dataset.nodes_id[de_gen])
                de_idx_table.append(index_table[de_gen])

        idx = np.where(dataset.pert_categories == pert_category)[0]

        if len(idx) > 30:
            emb_drugs = dataset.drugs[idx][0].view(
                1, -1).repeat(num, 1).clone()
            emb_cts = dataset.cell_types[idx][0].view(
                1, -1).repeat(num, 1).clone()
            
            emb_drugs = emb_drugs.to(autoencoder.device)
            emb_cts = emb_cts.to(autoencoder.device)
            
            mean_predict_total = torch.empty((0,1)).to(autoencoder.device)
            var_predict_total = torch.empty((0,1)).to(autoencoder.device)
            for genes_control in test_control_dataloader:
                x = genes_control[0].x.to(autoencoder.device)
                edge_index = genes_control[0].edge_index.to(autoencoder.device)
                batch = genes_control[0].batch.to(autoencoder.device)

                if len(torch.unique(batch)) < 128: # last batch has no 128 examples, so dimension must change
                    emb_drugs = emb_drugs[:len(torch.unique(batch)),:]
                    emb_cts = emb_cts[:len(torch.unique(batch)),:]

                predict_outputs = autoencoder.predict(
                    x,
                    edge_index,
                    batch,
                    emb_drugs,
                    emb_cts
                )
                
                nodes_predict = predict_outputs['nodes_reconstructed']
                
                mean_predict = nodes_predict[:, 0]  # node feature with the mean
                var_predict = nodes_predict[:, 1]   # node feature with the var
                
                if autoencoder.adversarial:
                    mean_predict_total = torch.cat((mean_predict_total, mean_predict.view(-1,1)), dim=0)
                    var_predict_total = torch.cat((var_predict_total, var_predict.view(-1,1)), dim=0)
                #else:
                #    nodes_predict_total = torch.cat((nodes_predict_total, nodes_predict[0]), dim=0)
                    
                del x, edge_index, batch
                torch.cuda.empty_cache()
                
            del emb_drugs, emb_cts
            torch.cuda.empty_cache()
                
            mean_predict = mean_predict_total
            var_predict = var_predict_total
            
            mean_predict = mean_predict.reshape(-1, 1)
            auxiliar_mean = torch.split(mean_predict, num_nodes)
            mean_predict = torch.cat(auxiliar_mean, dim=1)
            var_predict = var_predict.reshape(-1, 1)
            auxiliar_var = torch.split(var_predict, num_nodes)
            var_predict = torch.cat(auxiliar_var, dim=1)
            
            yp_m = mean_predict.mean(1).detach().cpu()
            yp_v = var_predict.mean(1).detach().cpu()

            y_true = dataset.genes[idx, :].numpy()
            y_true_genes = y_true[:, dataset.nodes_index] 

            # true means and variances
            yt_m = y_true.mean(axis=0)
            yt_v = y_true.var(axis=0)

            mean_score.append(r2_score(yt_m[dataset.nodes_index], yp_m[dataset.nodes_genes_index]))
            var_score.append(r2_score(yt_v[dataset.nodes_index], yp_v[dataset.nodes_genes_index]))

            mean_score_de.append(r2_score(yt_m[de_idx_table], yp_m[de_idx_one_graph]))
            var_score_de.append(r2_score(yt_v[de_idx_table], yp_v[de_idx_one_graph]))

    return [
        np.mean(s) if len(s) else -1
        for s in [mean_score, mean_score_de, var_score, var_score_de]
    ]

def complex_evaluate_r2_table(autoencoder, dataset, dataset_control):
    """
    R2 score evaluation function when the output is not a graph
    """
    
    test_control_dataloader = DataLoader(dataset_control,
                                         batch_size=128, num_workers=4)

    mean_score, var_score, mean_score_de, var_score_de = [], [], [], []

    num = 128
    num_nodes = len(dataset_control.nodes_id)
    
    # Create dictionary with the index of the genes in the table, just once
    index_table = dict()
    for gen in dataset.symbols:
        index_table[gen] = list(dataset.symbols).index(gen)

    for pert_category in tqdm(np.unique(dataset.pert_categories)):
        # pert_category category contains: 'celltype_perturbation_dose' info
        de_idx_one_graph = []  # find index of DE genes in the graph
        de_idx_table = []  # find index of DE in the table
        fc_de_index = [] # find index of DE genes in output
        combination_index = []
        if pert_category not in dataset.de_symbols.keys():
            continue  # if pert category is not a pert category
        for de_gen in np.array(dataset.de_symbols[pert_category]):
            de_idx_table.append(index_table[de_gen])
            combination_index.append(dataset.combination_order[index_table[de_gen]])
            if de_gen in set(dataset.nodes_names):  # access to set in constant time
                de_idx_one_graph.append(dataset.nodes_id[de_gen])
            else:
                fc_de_index.append(index_table[de_gen])

        idx = np.where(dataset.pert_categories == pert_category)[0]

        if len(idx) > 30:
            emb_drugs = dataset.drugs[idx][0].view(
                1, -1).repeat(num, 1).clone()
            emb_cts = dataset.cell_types[idx][0].view(
                1, -1).repeat(num, 1).clone()
            
            emb_drugs = emb_drugs.to(autoencoder.device)
            emb_cts = emb_cts.to(autoencoder.device)
            
            mean_predict_total = torch.empty((0,5000)).to(autoencoder.device)
            var_predict_total = torch.empty((0,5000)).to(autoencoder.device)
            for genes_control in test_control_dataloader:
                x = genes_control[0].x.to(autoencoder.device)
                edge_index = genes_control[0].edge_index.to(autoencoder.device)
                batch = genes_control[0].batch.to(autoencoder.device)
                fc_genes = genes_control[1].to(autoencoder.device)

                if len(torch.unique(batch)) < 128: # last batch has no 128 examples, so dimension must change
                    emb_drugs = emb_drugs[:len(torch.unique(batch)),:]
                    emb_cts = emb_cts[:len(torch.unique(batch)),:]

                inputs = dict(
                    graph_x=x,
                    graph_edge=edge_index,
                    graph_batch=batch,
                    fc_genes=fc_genes,
                    fc_drugs=emb_drugs,
                    fc_covariates=emb_cts
                )

                predict_outputs = autoencoder.predict(
                    inputs
                )
                
                nodes_predict = predict_outputs['genes_reconstructed']
                
                dim = nodes_predict.size(1) // 2
                
                mean_predict = nodes_predict[:, :dim]  # node feature with the mean
                var_predict = nodes_predict[:, dim:]   # node feature with the var
                
                if autoencoder.adversarial:
                    mean_predict_total = torch.cat((mean_predict_total, mean_predict), dim=0)
                    var_predict_total = torch.cat((var_predict_total, var_predict), dim=0)
                #else:
                #    nodes_predict_total = torch.cat((nodes_predict_total, nodes_predict[0]), dim=0)
                    
                del x, edge_index, batch, fc_genes
                torch.cuda.empty_cache()
                
            del emb_drugs, emb_cts
            torch.cuda.empty_cache()
                
            mean_predict = mean_predict_total
            var_predict = var_predict_total
            
            yp_m = mean_predict.mean(0).detach().cpu()
            yp_v = var_predict.mean(0).detach().cpu()

            y_true = torch.concat((dataset.graph_genes[idx,:], dataset.fc_genes[idx,:]), dim=1)
            if autoencoder.flavour == 'plugin':
                y_true = dataset.fc_genes[idx,:].numpy()

            # true means and variances
            yt_m = y_true.mean(axis=0)
            yt_v = y_true.var(axis=0)

            mean_score.append(r2_score(yt_m, yp_m))
            var_score.append(r2_score(yt_v, yp_v))
            
            de_idx = de_idx_one_graph + fc_de_index
            if autoencoder.flavour == 'plugin':
                de_idx = fc_de_index

            mean_score_de.append(r2_score(yt_m[de_idx], yp_m[de_idx]))
            var_score_de.append(r2_score(yt_v[de_idx], yp_v[de_idx]))

    return [
        np.mean(s) if len(s) else -1
        for s in [mean_score, mean_score_de, var_score, var_score_de]
    ]


def evaluate(autoencoder, datasets):
    """
    Measure quality metrics using `evaluate()` on the training, test, and
    out-of-distributiion (ood) splits.
    """

    autoencoder.eval()
    with torch.no_grad():
        if autoencoder.hparams['architecture']['GNN']['gnn_decoder_layers'] is not False:
            stats_test = evaluate_r2(
                autoencoder,
                datasets["test_treated"],
                datasets["test_control"])
        else:
            stats_test = evaluate_r2_table(
                autoencoder,
                datasets["test_treated"],
                datasets["test_control"])

        if autoencoder.adversarial:
            stats_disent_pert, stats_disent_cov = evaluate_disentanglement(
                autoencoder, datasets["test"])
        else:
            stats_disent_pert, stats_disent_cov = 0, 0

        if autoencoder.hparams['architecture']['GNN']['gnn_decoder_layers'] is not False:
            evaluation_stats = {
                "training": evaluate_r2(
                    autoencoder,
                    datasets["training_treated"],
                    datasets["training_control"]),
                "test": stats_test,
                "ood": evaluate_r2(
                    autoencoder,
                    datasets["ood"],
                    datasets["test_control"]),
                "perturbation disentanglement": stats_disent_pert,
                "optimal for perturbations": 1/datasets['test'].num_drugs,
                "covariate disentanglement": stats_disent_cov,
                "optimal for covariates": 1/datasets['test'].num_cell_types,
            }
        else:
            evaluation_stats = {
                "training": evaluate_r2_table(
                    autoencoder,
                    datasets["training_treated"],
                    datasets["training_control"]),
                "test": stats_test,
                "ood": evaluate_r2_table(
                    autoencoder,
                    datasets["ood"],
                    datasets["test_control"]),
                "perturbation disentanglement": stats_disent_pert,
                "optimal for perturbations": 1/datasets['test'].num_drugs,
                "covariate disentanglement": stats_disent_cov,
                "optimal for covariates": 1/datasets['test'].num_cell_types,
            }
    autoencoder.train()
    return evaluation_stats


def complex_evaluate(autoencoder, datasets):
    """
    Measure quality metrics using `evaluate()` on the training, test, and
    out-of-distributiion (ood) splits.
    """

    autoencoder.eval()
    with torch.no_grad():
        stats_test = complex_evaluate_r2_table(
            autoencoder,
            datasets["test_treated"],
            datasets["test_control"])

        if autoencoder.adversarial:
            stats_disent_pert, stats_disent_cov = complex_evaluate_disentanglement(
                autoencoder, datasets["test"])
        else:
            stats_disent_pert, stats_disent_cov = 0, 0

        evaluation_stats = {
            "training": complex_evaluate_r2_table(
                autoencoder,
                datasets["training_treated"],
                datasets["training_control"]),
            "test": stats_test,
            "ood": complex_evaluate_r2_table(
                autoencoder,
                datasets["ood"],
                datasets["test_control"]),
            "perturbation disentanglement": stats_disent_pert,
            "optimal for perturbations": 1/datasets['test'].num_drugs,
            "covariate disentanglement": stats_disent_cov,
            "optimal for covariates": 1/datasets['test'].num_cell_types,
            }
    autoencoder.train()
    return evaluation_stats


def complex_train_routine(model, config, datasets, return_model):

    """
    Entire training routine with callbacks and loggings
    :param model: Model to train, defined in the main script
    :param config: dictionary with the training parameters
    :param datasets: datasets to train, previoulsy splitted
    :param return_model
    :return:
    """

    train_loader = DataLoader(datasets["training"],
                              batch_size=config['training']['Global']['batch_size'],
                              num_workers=4,
                              pin_memory=True,
                              drop_last=True
                              )
    
    genes_index_all = []
    for i in range(0, config['training']['Global']['batch_size']):
        genes_index_all = genes_index_all + [element + i*model.n_nodes for element in model.genes_index]

    if model.adversarial is False:
        test_loader = DataLoader(datasets["test"],
                                 batch_size=config['training']['Global']['batch_size'],
                                 num_workers=4,
                                 pin_memory=True
                                 )

    device = model.device

    args = config['training']['Global']
    pjson({"training_args": args})
    pjson({"autoencoder_params": model.hparams})

    start_time = time.time()
    for epoch in range(config['training']['Global']['epochs']+1):
        print("Epoch: ", epoch)
        # At each epoch, initialize the losses at 0
        epoch_training_stats = defaultdict(float)
        epoch_testing_stats = defaultdict(float)

        for graph, fc_genes, drug, cov in tqdm(train_loader):
            x = graph.x.to(device)
            edge_index = graph.edge_index.to(device)
            batch = graph.batch.to(device)
            fc_genes = fc_genes.to(device)
            drug = drug.to(device)
            cov = cov.to(device)
            
            minibatch_dict = dict(
                graph_x=x,
                graph_edge=edge_index,
                graph_batch=batch,
                fc_genes=fc_genes,
                fc_drugs=drug,
                fc_covariates=cov
            )
            
            minibatch_training_stats = model.update(minibatch_dict, genes_index_all, epoch)
            
            del x, edge_index, batch, drug, cov
            torch.cuda.empty_cache()

            for key, val in minibatch_training_stats.items():
                epoch_training_stats[key] += val

        for key, val in epoch_training_stats.items():
            epoch_training_stats[key] = val
            if not (key in model.history.keys()):
                model.history[key] = []
            model.history[key].append(val)
        model.history['epoch'].append(epoch)
        
        if model.adversarial is False:        
            for data, drug, cov in tqdm(test_loader):
                x = data.x.to(device)
                edge_index = data.edge_index.to(device)
                batch = data.batch.to(device)
                drug = drug.to(device)
                cov = cov.to(device)
                minibatch_testing_stats = model.update(x, edge_index, batch, drug, cov, genes_index_all, epoch)
                
                del x, edge_index, batch, drug, cov
                torch.cuda.empty_cache()

                for key, val in minibatch_testing_stats.items():
                    epoch_testing_stats[key] += val
                    
            for key, val in epoch_testing_stats.items():
                epoch_testing_stats[key] = val
                if not (key+"_test" in model.history.keys()):
                    model.history[key+"_test"] = []
                model.history[key+"_test"].append(val)

        ellapsed_minutes = (time.time() - start_time) / 60
        model.history['elapsed_time_min'] = ellapsed_minutes

            # decay learning rate if necessary
            # also check stopping condition: patience ran out OR
            # time ran out OR max epochs achieved
        stop = ellapsed_minutes > args["max_minutes"] or \
               (epoch == args["epochs"] - 1)

        if (epoch % args["checkpoint_freq"]) == 0 or stop:
            if args["evaluation"]:
                evaluation_stats = complex_evaluate(model, datasets)
                for key, val in evaluation_stats.items():
                    if not (key in model.history.keys()):
                        model.history[key] = []
                    model.history[key].append(val)
                model.history['stats_epoch'].append(epoch)

                pjson({
                    "epoch": epoch,
                    "training_stats": epoch_training_stats,
                    "evaluation_stats": evaluation_stats,
                    "ellapsed_minutes": ellapsed_minutes
                })
            else:
                evaluation_stats = {
            "training": [0, 0, 0, 0],
            "test": [0, 0, 0, 0],
            "ood": [0, 0, 0, 0],
            "optimal for perturbations": [0],
            "covariate disentanglement": [0],
            "optimal for covariates": [0],
            }
                model.train()
                for key, val in evaluation_stats.items():
                    if not (key in model.history.keys()):
                        model.history[key] = []
                    model.history[key].append(val)
                model.history['stats_epoch'].append(epoch)

            if not os.path.exists(args["save_dir"]):
                os.makedirs(args["save_dir"])
            #torch.save(
            #    (model.state_dict(), args, model.history),
            #    os.path.join(
            #        args["save_dir"],
            #        "model={}_seed={}_epoch={}.pt".format(args["name"], args["seed"], epoch)))

            pjson({"model_saved": "model={}_seed={}_epoch={}.pt\n".format(
                args["name"], args["seed"], epoch)})
            stop = stop or model.early_stopping(
                np.mean(evaluation_stats["test"]))
            if stop:
                pjson({"early_stop": epoch})
                break

    if return_model:
        return model