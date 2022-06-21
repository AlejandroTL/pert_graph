import torch
import torch.nn as nn
import numpy as np
import json
import yaml

from models.VAE_modules import Decoder, Encoder
from models.GNN_modules import GNN_network
from torch.distributions import Normal
from torch_geometric.utils import to_dense_batch, add_self_loops, negative_sampling, remove_self_loops
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.kl import kl_divergence
from torch_geometric.nn import InnerProductDecoder

criterion_mse = nn.MSELoss(reduction='sum')
criterion_bce = nn.CrossEntropyLoss(reduction='sum')
criterion_gll = nn.GaussianNLLLoss(reduction='mean')


class MLP(torch.nn.Module):
    """
    A multilayer perceptron with ReLU activations and optional BatchNorm.
    """

    def __init__(self, sizes, batch_norm=True, last_layer_act="linear", dropout=0.3):
        super(MLP, self).__init__()
        layers = []
        for s in range(len(sizes) - 1):
            layers += [
                torch.nn.Linear(sizes[s], sizes[s + 1]),
                torch.nn.BatchNorm1d(sizes[s + 1])
                if batch_norm and s < len(sizes) - 2 else None,
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout)
            ]

        layers = [l for l in layers if l is not None][:-1]
        self.activation = last_layer_act
        if self.activation == "linear":
            pass
        elif self.activation == "ReLU":
            self.relu = torch.nn.ReLU()
        else:
            raise ValueError("last_layer_act must be one of 'linear' or 'ReLU'")

        self.network = torch.nn.Sequential(*layers)

    def forward(self, x):
        if self.activation == "ReLU":
           x = self.network(x)
           dim = x.size(1) // 2
           return torch.cat((self.relu(x[:, :dim]), x[:, dim:]), dim=1)
        return self.network(x)


class GaussianLoss(torch.nn.Module):
    """
    Gaussian log-likelihood loss. It assumes targets `y` with n rows and d
    columns, but estimates `yhat` with n rows and 2d columns. The columns 0:d
    of `yhat` contain estimated means, the columns d:2*d of `yhat` contain
    estimated variances. This module assumes that the estimated variances are
    positive---for numerical stability, it is recommended that the minimum
    estimated variance is greater than a small number (1e-3).
    """

    def __init__(self):
        super(GaussianLoss, self).__init__()

    def forward(self, yhat, y):
        dim = yhat.size(1) // 2
        mean = yhat[:, :dim]
        variance = yhat[:, dim:]

        term1 = variance.log().div(2)
        term2 = (y - mean).pow(2).div(variance.mul(2))

        return (term1 + term2).mean()


class GraphAutoencoder(nn.Module):

    def __init__(self,
                 n_nodes,
                 num_drugs,
                 num_cell_types,
                 genes_index_set,
                 doser_type='mlp',
                 adversarial=False,
                 device="cpu",
                 hparams="",
                 seed=0,
                 variational=False):

        self.n_nodes = n_nodes
        self.n_drugs = num_drugs
        self.n_genes = n_nodes - num_drugs
        self.n_cell_types = num_cell_types
        
        self.genes_index = list(genes_index_set)
            
        self.adversarial = adversarial
        self.device = device
        
        self.variational = variational
        self.beta = 0

        super().__init__()
        self.loss_autoencoder = nn.GaussianNLLLoss()
        self.loss_adversary_drugs = torch.nn.BCEWithLogitsLoss()
        self.loss_adversary_cell_types = torch.nn.CrossEntropyLoss()
        self.iteration = 0

        self.set_hparams_(seed, hparams)
        
        self.warmup_kl = self.hparams['training']['GNN']['warmup_kl']
        self.max_epoch_kl = self.hparams['training']['GNN']['max_epoch_kl']
        self.max_beta = self.hparams['training']['GNN']['max_beta']        
        
        self.matrix_loss_weight = self.hparams['training']['GNN']['matrix_loss_weight']
        
        
        if self.hparams['architecture']['GNN']['loss'] == 'mse':
            self.loss_autoencoder = nn.MSELoss()

        factor0 = self.hparams['architecture']['GNN']['node_embeddings_dim']-1 if self.hparams['architecture']['GNN']['node_embeddings'] else 0

        # set models
        self.graph_encoder = GNN_network(
            x_dim=self.hparams['architecture']['GNN']["x_dim"] + factor0,
            layer_sizes=self.hparams['architecture']['GNN']["gnn_encoder_layers"],
            dropout_rate=self.hparams['architecture']['GNN']["gnn_encoder_dropout"],
            Conv=self.hparams['architecture']['GNN']["Conv"],
            k=self.hparams['architecture']['GNN']["k"],
        )

        self.fc_encoder = Encoder(
            x_dim=self.n_nodes * self.hparams['architecture']['GNN']["gnn_encoder_layers"][-1],
            layer_sizes=self.hparams['architecture']['GNN']["vae_encoder_layers"],
            z_dimension=self.hparams['architecture']['GNN']["z_dim"],
            dropout_rate=self.hparams['architecture']['GNN']["vae_encoder_dropout"],
            batch_norm=self.hparams['architecture']['GNN']["vae_encoder_batch_norm"]
        )

        factor = 2 if self.hparams['architecture']['GNN']["gnn_decoder_layers"] is False else 1  # if no GNN, we generate mean and var of each gene
        
        self.fc_decoder = Decoder(
            z_dimension=self.hparams['architecture']['GNN']["z_dim"],
            layer_sizes=self.hparams['architecture']['GNN']["vae_decoder_layers"],
            x_dim=self.n_nodes * self.hparams['architecture']['GNN']["gnn_encoder_layers"][-1] * factor,
            dropout_rate=self.hparams['architecture']['GNN']["vae_decoder_dropout"],
            batch_norm=self.hparams['architecture']['GNN']["vae_encoder_batch_norm"]
        )

        if self.hparams['architecture']['GNN']["gnn_decoder_layers"] is not False:
            self.graph_decoder = GNN_network(
                x_dim=1,
                layer_sizes=self.hparams['architecture']['GNN']["gnn_decoder_layers"],
                dropout_rate=self.hparams['architecture']['GNN']["gnn_decoder_dropout"],
                Conv=self.hparams['architecture']['GNN']["Conv"],
                k=self.hparams['architecture']['GNN']["k"],
            )
            
        if self.hparams['architecture']['GNN']['matrix_reconstruction']:
            self.matrix_decoder = InnerProductDecoder()
            
        if self.adversarial is True:
            self.adversary_drugs = MLP(
                [self.hparams['architecture']['GNN']["z_dim"]] +
                [self.hparams['architecture']['GNN']["adversary_width"]] *
                self.hparams['architecture']['GNN']["adversary_depth"] +
                [self.n_drugs], dropout=self.hparams['architecture']['GNN']["adversary_dropout"]
            )
            print("Discriminator Drugs: ", self.adversary_drugs)

            self.adversary_cell_types = MLP(
                [self.hparams['architecture']['GNN']["z_dim"]] +
                [self.hparams['architecture']['GNN']["adversary_width"]] *
                self.hparams['architecture']['GNN']["adversary_depth"] +
                [self.n_cell_types], dropout=self.hparams['architecture']['GNN']["adversary_dropout"]
            )
            print("Discriminator Covs: ", self.adversary_cell_types)

            # set dosers
            self.doser_type = doser_type
            if doser_type == 'mlp':
                self.dosers = torch.nn.ModuleList()
                for _ in range(self.n_drugs):
                    self.dosers.append(
                        MLP([1] +
                            [self.hparams['architecture']['GNN']["dosers_width"]] *
                            self.hparams['architecture']['GNN']["dosers_depth"] +
                            [1],
                            batch_norm=False))
            #    print("Dosers: ", self.dosers)


            # Drug and covariate (cell_type) embeddings
            self.drug_embeddings = torch.nn.Embedding(
                self.n_drugs, self.hparams['architecture']['GNN']["z_dim"])
            self.cell_type_embeddings = torch.nn.Embedding(
                self.n_cell_types, self.hparams['architecture']['GNN']["z_dim"])
            
            if self.hparams['architecture']['GNN']['node_embeddings']:
                self.node_embeddings = torch.nn.Embedding(
                    self.n_nodes, self.hparams['architecture']['GNN']['node_embeddings_dim']
                )

        self.to(self.device)
        
        get_params = lambda model, cond: list(model.parameters()) if cond else []

        self.optimizer_autoencoder = torch.optim.Adam(
            list(self.graph_encoder.parameters()) +
            list(self.fc_encoder.parameters()) +
            list(self.fc_decoder.parameters()) +
            get_params(self.graph_decoder, self.hparams['architecture']['GNN']["gnn_decoder_layers"] is not False) if self.hparams['architecture']['GNN']["gnn_decoder_layers"] is not False else [] +
            get_params(self.drug_embeddings, self.hparams['data']['perturbation'] == 'drug') +
            get_params(self.node_embeddings, self.hparams['architecture']['GNN']['node_embeddings'] is not False) if self.hparams['architecture']['GNN']['node_embeddings'] else [] +
            list(self.cell_type_embeddings.parameters()),
            lr=float(self.hparams['training']['GNN']["autoencoder_lr"]),
            weight_decay=float(self.hparams['training']['GNN']["autoencoder_wd"])
        )

        if self.adversarial is True:
            self.optimizer_adversaries = torch.optim.Adam(
                list(self.adversary_drugs.parameters()) +
                list(self.adversary_cell_types.parameters()),
                lr=self.hparams['training']['GNN']["adversary_lr"],
                weight_decay=self.hparams['training']['GNN']["adversary_wd"])

            self.optimizer_dosers = torch.optim.Adam(
                self.dosers.parameters(),
                lr=self.hparams['training']['GNN']["dosers_lr"],
                weight_decay=self.hparams['training']['GNN']["dosers_wd"])

        # learning rate schedulers
        self.scheduler_autoencoder = torch.optim.lr_scheduler.StepLR(
            self.optimizer_autoencoder, step_size=self.hparams['training']['GNN']["step_size_lr"])

        if self.adversarial is True:
            self.scheduler_adversary = torch.optim.lr_scheduler.StepLR(
                self.optimizer_adversaries, step_size=self.hparams['training']['GNN']["step_size_lr"])

            self.scheduler_dosers = torch.optim.lr_scheduler.StepLR(
                self.optimizer_dosers, step_size=self.hparams['training']['GNN']["step_size_lr"])

        self.history = {'epoch': [], 'stats_epoch': []}

        #early-stopping
        self.patience = self.hparams['training']['GNN']['patience']
        self.best_score = -1e3
        self.patience_trials = 0

    def set_hparams_(self, seed, hparams):
        """
        Hyper-parameters set by config file
        """

        with open(hparams) as f:
            self.hparams = yaml.load(f, Loader=yaml.FullLoader)

        return self.hparams

    def sampling(self, mu, log_var):
        """Samples from standard Normal distribution and applies re-parametrization trick.
           It is actually sampling from latent space distributions with N(mu, var), computed by encoder.
           Parameters
           ----------
           mu: torch.Tensor
                Torch Tensor of Means.
           log_var: torch.Tensor
                Torch Tensor of log. variances.
           Returns
           -------
           Torch Tensor of sampled data.
        """
        var = torch.exp(log_var) + 1e-4
        return Normal(mu, var.sqrt()).rsample()
    
    def beta_callback(self, current_epoch):
        
        beta = -1
        if current_epoch > self.warmup_kl:
            beta = (self.max_beta/self.max_epoch_kl)*(current_epoch - self.warmup_kl)
        if current_epoch >= self.max_epoch_kl:
            beta = self.max_beta
            
        return max(beta, 0)
    
    def augment_node_features(self, x):
        
        gex = x[:,1].reshape(-1, 1)
        
        augmented = (gex.T.view(-1, self.node_embeddings.weight.size(0),1) * self.node_embeddings.weight).view(-1,self.node_embeddings.weight.size(0),self.node_embeddings.weight.size(1))
        
        augmented = augmented.view(-1, self.node_embeddings.weight.size(1))
        
        augmented = torch.cat((x[:,0].view(-1,1), augmented, x[:,2].view(-1,1)),dim=1)
        
        return augmented
        
    def forward(self, x, edge_index, batch):

        """
        Entire forward pass
        :param x:
        :param edge_index:
        :param batch:
        :return:
        """

        encoded_node_features = self.graph_encoder(x, edge_index)  # GNN
        # Transform to dense batch to concat the node features of each graph
        encoded_node_features, mask = to_dense_batch(encoded_node_features, batch, batch_size=len(torch.unique(batch)))
        # Change the dimensionality of the data to [batch, node*features]
        encoded_node_features = encoded_node_features.view([len(torch.unique(batch)),
                                                            encoded_node_features.shape[1] *
                                                            encoded_node_features.shape[2]])
        z_mean, z_var = self.fc_encoder(encoded_node_features)  # VAE Encoder
        #z1 = self.sampling(z_mean, z_var)  # sampling the latent space
        graph_embedding_decoded = self.fc_decoder(z_mean)
        #graph_embedding_decoded = self.fc_decoder(z1)  # decode the latent space
        # Change the dimensionality of the data to [1, node*features]
        graph_embedding_decoded = graph_embedding_decoded.view([1,
                                                                graph_embedding_decoded.shape[0] *
                                                                graph_embedding_decoded.shape[1]])
        decoded_node_features = self.graph_decoder(graph_embedding_decoded.T, edge_index)  # GNN again

        return encoded_node_features, z_mean, graph_embedding_decoded, decoded_node_features

    def encode_graph(self, x, edge_index, batch):

        """
        Encoding step
        :param x:
        :param edge_index:
        :param batch:
        :return:
        """

        if self.hparams['architecture']['GNN']['node_embeddings']:
            x = self.augment_node_features(x)

        encoded_node_features_original = self.graph_encoder(x, edge_index)
        encoded_node_features, mask = to_dense_batch(encoded_node_features_original, batch, batch_size=len(torch.unique(batch)))
        encoded_node_features = encoded_node_features.view([len(torch.unique(batch)),
                                                            encoded_node_features.shape[1] *
                                                            encoded_node_features.shape[2]])
        z_mean, z_var = self.fc_encoder(encoded_node_features)
        
        if self.variational:
         
            z1 = self.sampling(z_mean, z_var)
            return dict(
                latent_basal=z1,
                z_mean=z_mean,
                z_var=z_var,
                encoded_nodes=encoded_node_features_original
                )
        
        return dict(
            latent_basal=z_mean,
            encoded_nodes=encoded_node_features_original
        )

    def decode_to_graph(self, x, edge_index, batch):

        """
        Decoding step
        :param x:
        :param edge_index:
        :param batch:
        :return:
        """

        graph_embedding_decoded = self.fc_decoder(x)
        graph_embedding_decoded_view = graph_embedding_decoded.view([1,
                                                                graph_embedding_decoded.shape[0] *
                                                                graph_embedding_decoded.shape[1]])
        graph_embedding_decoded_t = graph_embedding_decoded_view.T
        
        if self.hparams['architecture']['GNN']["gnn_decoder_layers"] is not False:
            decoded_node_features = self.graph_decoder(graph_embedding_decoded_t, edge_index)
            return decoded_node_features
        
        return graph_embedding_decoded_t

    def compute_drug_embeddings_(self, drugs):
        """
        Compute sum of drug embeddings, each of them multiplied by its
        dose-response curve.
        """

        if self.doser_type == 'mlp':
            doses = []
            for d in range(drugs.size(1)):
                this_drug = drugs[:, d].view(-1, 1)
                doses.append(self.dosers[d](this_drug).sigmoid() * this_drug.gt(0))
            return torch.cat(doses, 1) @ self.drug_embeddings.weight
        else:
            return self.dosers(drugs) @ self.drug_embeddings.weight

    def predict(self, x, edge_index, batch, drugs, cell_types, return_latent_basal=False):

        # Encode the graph
        encode_outputs = self.encode_graph(x, edge_index, batch)
        latent_basal = encode_outputs['latent_basal']
        encoded_node_features_graph = encode_outputs['encoded_nodes']
        basal_distribution = None
        if self.variational:
            basal_mean = encode_outputs['z_mean']
            basal_var = encode_outputs['z_var']
            var = torch.exp(basal_var) + 1e-4
            basal_distribution = Normal(basal_mean, var.sqrt())

        if self.adversarial is False:
            gene_reconstructions = self.decode_to_graph(latent_basal, edge_index, batch)
            
            dim = gene_reconstructions.size(0) // 2
            means = gene_reconstructions[:, :dim]
            variances = gene_reconstructions[:, dim:].exp().add(1).log().add(1e-3)

            gene_reconstructions = torch.cat((means, variances), dim=1)

            return dict(
                nodes_reconstructed=gene_reconstructions
                )
        
        # Build the latent treated
        drug_emb = self.compute_drug_embeddings_(drugs)
        cell_emb = self.cell_type_embeddings(cell_types.argmax(1))

        latent_treated = latent_basal + drug_emb + cell_emb
        gene_reconstructions = self.decode_to_graph(latent_treated, edge_index, batch)

        # convert variance estimates to a positive value in [1e-3, \infty)
        dim = gene_reconstructions.size(1) // 2
        means = gene_reconstructions[:, :dim]
        variances = gene_reconstructions[:, dim:].exp().add(1).log().add(1e-3)
        if self.hparams['architecture']['GNN']['gnn_decoder_layers'] is False:
            dim = gene_reconstructions.size(0) // 2
            means = gene_reconstructions[:dim, :]
            variances = gene_reconstructions[dim:, :].exp().add(1).log().add(1e-3)

        nodes_reconstructed = torch.cat((means, variances), dim=1)

        if return_latent_basal is True:
            return dict(
                nodes_reconstructed=nodes_reconstructed,
                latent_basal=latent_basal,
                basal_distribution=basal_distribution,
                encoded_node_features_graph=encoded_node_features_graph
            )

        return dict(
            nodes_reconstructed=nodes_reconstructed
        )
    
    def full_forward(self, x, edge_index, batch, drugs, cell_types):
        
        encode_outputs = self.encode_graph(x, edge_index, batch)
        latent_basal = encode_outputs['latent_basal']
        basal_distribution = None
        if self.variational:
            basal_mean = encode_outputs['z_mean']
            basal_var = encode_outputs['z_var']
            var = torch.exp(basal_var) + 1e-4
            basal_distribution = Normal(basal_mean, var.sqrt())
        drug_emb = self.compute_drug_embeddings_(drugs)
        cell_emb = self.cell_type_embeddings(cell_types.argmax(1))
        
        latent_treated = latent_basal + drug_emb + cell_emb
        gene_reconstructions = self.decode_to_graph(latent_treated, edge_index, batch)
        
        # convert variance estimates to a positive value in [1e-3, \infty)
        dim = gene_reconstructions.size(1) // 2
        means = gene_reconstructions[:, :dim]
        variances = gene_reconstructions[:, dim:].exp().add(1).log().add(1e-3)
        if self.hparams['architecture']['GNN']['gnn_decoder_layers'] is False:
            dim = gene_reconstructions.size(0) // 2
            means = gene_reconstructions[:dim, :]
            variances = gene_reconstructions[dim:, :].exp().add(1).log().add(1e-3)

        nodes_reconstructed = torch.cat((means, variances), dim=1)

        return nodes_reconstructed, latent_basal, latent_treated, drugs, drug_emb, cell_emb

    def early_stopping(self, score):
        """
        Decays the learning rate, and possibly early-stops training.
        """
        self.scheduler_autoencoder.step()
        
        if self.adversarial:
            self.scheduler_adversary.step()
            self.scheduler_dosers.step()

        if score > self.best_score:
            self.best_score = score
            self.patience_trials = 0
        else:
            self.patience_trials += 1

        return self.patience_trials > self.patience

    def update(self, node_features, edge_index, batch, drugs=None, cell_types=None, genes_index_all=[], epoch=0):

        """
        Update step during training
        :param node_features:
        :param edge_index:
        :param batch:
        :param drugs:
        :param cell_types:
        :return:
        """

        #self.optimizer_autoencoder.zero_grad()
        predict_outputs = self.predict(node_features,
                                       edge_index,
                                       batch,
                                       drugs,
                                       cell_types,
                                       return_latent_basal=True)
        
        reconstructed_node_features = predict_outputs['nodes_reconstructed']
        latent_basal = predict_outputs['latent_basal']
        encoded_node_features_graph = predict_outputs['encoded_node_features_graph']

        # get index of genes across all graphs in minibatch
        # genes_index_all = []
        # batch_size = len(torch.unique(batch))
        # for i in range(0, batch_size):
        #    genes_index_all = genes_index_all + [element + i*self.n_nodes for element in self.genes_index]
        
        # reconstruct just the genes
        node_features = node_features[genes_index_all, :]  
        #reconstructed_node_features = reconstructed_node_features[genes_index_all, :]
        
        # take mean and variance
        dim = reconstructed_node_features.size(1) // 2
        
        x = node_features[:, 1]
        mean = reconstructed_node_features[:, :dim]
        variance = reconstructed_node_features[:, dim:]
        
        mean = mean[genes_index_all, :].flatten()
        variance = variance[genes_index_all, :].flatten()
        
        # compute the KL divergence if variational
        kl_loss = torch.tensor([0.0], device=self.device)
        if self.variational:
            basal_distribution = predict_outputs["basal_distribution"]
            dist_pz = Normal(
                torch.zeros_like(basal_distribution.loc), torch.ones_like(basal_distribution.scale)
            )
            kl_loss = kl_divergence(basal_distribution, dist_pz).sum(-1)
            self.beta = self.beta_callback(epoch)
        
        # compute reconstruction loss
        if self.hparams['architecture']['GNN']['loss'] == 'mse':
            reconstruction_loss = self.loss_autoencoder(x, mean)
        else:
            reconstruction_loss = self.loss_autoencoder(x, mean, variance)

        # If adversarial mode and not in warmup
        if self.adversarial is True:
            
            adversary_drugs_loss = torch.tensor([0.0], device=self.device)
            
            # If more there are some drugs, try to find it on the latent space
            if self.n_drugs > 0:
                adversary_drugs_predictions = self.adversary_drugs(  # predict drugs
                    latent_basal)
                adversary_drugs_loss = self.loss_adversary_drugs(  #  BCE loss
                    adversary_drugs_predictions, drugs.gt(0).float())

            # Same with cells
            adversary_cell_types_loss = torch.tensor([0.0], device=self.device)   
            if self.n_cell_types > 0:
                adversary_cell_types_predictions = self.adversary_cell_types(
                    latent_basal)
                adversary_cell_types_loss = self.loss_adversary_cell_types(
                    adversary_cell_types_predictions, cell_types.argmax(1))

            # two place-holders for when adversary is not executed
            adversary_drugs_penalty = torch.Tensor([0])
            adversary_cell_types_penalty = torch.Tensor([0])


            # We train the adversarial networks "adversary_steps" times more than the AE
            if self.iteration % self.hparams['training']['GNN']["adversary_steps"] and epoch >= self.hparams['training']['GNN']['epochs_warmup']:
                
                # Compute the gradients with respect to the basal
                adversary_drugs_penalty = torch.autograd.grad(
                    adversary_drugs_predictions.sum(),
                    latent_basal,
                    create_graph=True)[0].pow(2).mean()

                adversary_cell_types_penalty = torch.autograd.grad(
                    adversary_cell_types_predictions.sum(),
                    latent_basal,
                    create_graph=True)[0].pow(2).mean()

                self.optimizer_adversaries.zero_grad()
                (adversary_drugs_loss +
                 self.hparams['training']['GNN']["penalty_adversary"] *
                 adversary_drugs_penalty +
                 adversary_cell_types_loss +
                 self.hparams['training']['GNN']["penalty_adversary"] *
                 adversary_cell_types_penalty).backward()
                self.optimizer_adversaries.step()  # just step the adversarial networks
            else:                
                self.optimizer_autoencoder.zero_grad()
                self.optimizer_dosers.zero_grad()
                (reconstruction_loss + self.beta*kl_loss.mean() -
                self.hparams['training']['GNN']["reg_adversary"] *
                 adversary_drugs_loss -
                 self.hparams['training']['GNN']["reg_adversary"] *
                 adversary_cell_types_loss).backward()
                self.optimizer_autoencoder.step()  # step the autoencoder
                self.optimizer_dosers.step()  # and the dosers

            self.iteration += 1

            return {
                "loss_reconstruction": reconstruction_loss.item(),
                "loss_kl": kl_loss.mean().item(),
                "loss_adv_drugs": adversary_drugs_loss.item(),
                "loss_adv_cell_types": adversary_cell_types_loss.item(),
                "penalty_adv_drugs": adversary_drugs_penalty.item(),
                "penalty_adv_cell_types": adversary_cell_types_penalty.item()
            }

        else:
            self.optimizer_autoencoder.zero_grad()
            reconstruction_loss.backward()
            self.optimizer_autoencoder.step()

        return {
            "loss_reconstruction": reconstruction_loss.item(),
            "kl_loss": kl_loss.mean().item(),
            "loss_adv_drugs": 0,
            "loss_adv_cell_types": 0,
            "penalty_adv_drugs": 0,
            "penalty_adv_cell_types": 0
        }



class CPA(torch.nn.Module):
    """
    Our main module, the CPA autoencoder
    """

    def __init__(
        self,
        num_genes,
        num_drugs,
        num_covariates,
        device="cuda",
        seed=0,
        patience=5,
        doser_type="mlp",
        variational=False,
        decoder_activation="linear",
        hparams="",
    ):
        super(CPA, self).__init__()
        # set generic attributes
        self.num_genes = num_genes
        self.num_drugs = num_drugs
        self.num_covariates = num_covariates
        self.variational = variational
        self.device = device
        self.seed = seed
        self.loss_ae = nn.GaussianNLLLoss()
        # early-stopping
        self.patience = patience
        self.best_score = -1e3
        self.patience_trials = 0

        # set hyperparameters
        self.set_hparams_(seed, hparams)
        
        if self.hparams['architecture']['FC']['loss'] == 'mse':
            self.loss_autoencoder = nn.MSELoss()

        # set models
        factor0 = 2 if self.variational else 1
        
        self.encoder = MLP(
            [num_genes]
            + [self.hparams['architecture']['FC']["autoencoder_width"]] * self.hparams['architecture']['FC']["autoencoder_depth"]
            + [self.hparams['architecture']['FC']["z_dim"]*factor0]
        )
        print("FC encoder: ", self.encoder)

        self.decoder = MLP(
            [self.hparams['architecture']['FC']["z_dim"]]
            + [self.hparams['architecture']['FC']["autoencoder_width"]] * self.hparams['architecture']['FC']["autoencoder_depth"]
            + [num_genes * 2],
            last_layer_act=decoder_activation,
        )
        print("FC decoder: ", self.decoder)

        self.adversary_drugs = MLP(
            [self.hparams['architecture']['FC']["z_dim"]]
            + [self.hparams['architecture']['FC']["adversary_width"]] * self.hparams['architecture']['FC']["adversary_depth"]
            + [num_drugs]
        )

        self.loss_adversary_drugs = torch.nn.BCEWithLogitsLoss()
        self.doser_type = doser_type
        if doser_type == "mlp":
            self.dosers = torch.nn.ModuleList()
            for _ in range(num_drugs):
                self.dosers.append(
                    MLP(
                        [1]
                        + [self.hparams['architecture']['FC']["dosers_width"]] * self.hparams['architecture']['FC']["dosers_depth"]
                        + [1],
                        batch_norm=False,
                    )
                )
        #else:
        #    self.dosers = GeneralizedSigmoid(num_drugs, self.device, nonlin=doser_type)

        self.num_covariates = [self.num_covariates]
        if self.num_covariates == [0]:
            pass
        else:
            assert 0 not in self.num_covariates
            self.adversary_covariates = []
            self.loss_adversary_covariates = []
            self.covariates_embeddings = (
                []
            )  # TODO: Continue with checking that dict assignment is possible via covaraites names and if dict are possible to use in optimisation
            for num_covariate in self.num_covariates:
                self.adversary_covariates.append(
                    MLP(
                        [self.hparams['architecture']['FC']["z_dim"]]
                        + [self.hparams['architecture']['FC']["adversary_width"]]
                        * self.hparams['architecture']['FC']["adversary_depth"]
                        + [num_covariate]
                    )
                )
                self.loss_adversary_covariates.append(torch.nn.CrossEntropyLoss())
                self.covariates_embeddings.append(
                    torch.nn.Embedding(num_covariate, self.hparams['architecture']['FC']["z_dim"])
                )
            self.covariates_embeddings = torch.nn.Sequential(
                *self.covariates_embeddings
            )

        self.drug_embeddings = torch.nn.Embedding(self.num_drugs, self.hparams['architecture']['FC']["z_dim"])

        self.iteration = 0

        self.to(self.device)

        # optimizers
        has_drugs = self.num_drugs > 0
        has_covariates = self.num_covariates[0] > 0
        get_params = lambda model, cond: list(model.parameters()) if cond else []
        _parameters = (
            get_params(self.encoder, True)
            + get_params(self.decoder, True)
            + get_params(self.drug_embeddings, has_drugs)
        )
        for emb in self.covariates_embeddings:
            _parameters.extend(get_params(emb, has_covariates))

        self.optimizer_autoencoder = torch.optim.Adam(
            _parameters,
            lr=self.hparams['training']['FC']["autoencoder_lr"],
            weight_decay=self.hparams['training']['FC']["autoencoder_wd"],
        )

        _parameters = get_params(self.adversary_drugs, has_drugs)
        for adv in self.adversary_covariates:
            _parameters.extend(get_params(adv, has_covariates))

        self.optimizer_adversaries = torch.optim.Adam(
            _parameters,
            lr=self.hparams['training']['FC']["adversary_lr"],
            weight_decay=self.hparams['training']['FC']["adversary_wd"],
        )

        if has_drugs:
            self.optimizer_dosers = torch.optim.Adam(
                self.dosers.parameters(),
                lr=self.hparams['training']['FC']["dosers_lr"],
                weight_decay=self.hparams['training']['FC']["dosers_wd"],
            )

        # learning rate schedulers
        self.scheduler_autoencoder = torch.optim.lr_scheduler.StepLR(
            self.optimizer_autoencoder, step_size=self.hparams['training']['FC']["step_size_lr"]
        )

        self.scheduler_adversary = torch.optim.lr_scheduler.StepLR(
            self.optimizer_adversaries, step_size=self.hparams['training']['FC']["step_size_lr"]
        )

        if has_drugs:
            self.scheduler_dosers = torch.optim.lr_scheduler.StepLR(
                self.optimizer_dosers, step_size=self.hparams['training']['FC']["step_size_lr"]
            )

        self.history = {"epoch": [], "stats_epoch": []}

    def set_hparams_(self, seed, hparams):
        """
        Hyper-parameters set by config file
        """

        with open(hparams) as f:
            self.hparams = yaml.load(f, Loader=yaml.FullLoader)

        return self.hparams

    def move_inputs_(self, genes, drugs, covariates):
        """
        Move minibatch tensors to CPU/GPU.
        """
        if genes.device.type != self.device:
            genes = genes.to(self.device)
            if drugs is not None:
                drugs = drugs.to(self.device)
            if covariates is not None:
                covariates = [cov.to(self.device) for cov in covariates]
        return (genes, drugs, covariates)

    def compute_drug_embeddings_(self, drugs):
        """
        Compute sum of drug embeddings, each of them multiplied by its
        dose-response curve.
        """

        if self.doser_type == "mlp":
            doses = []
            for d in range(drugs.size(1)):
                this_drug = drugs[:, d].view(-1, 1)
                doses.append(self.dosers[d](this_drug).sigmoid() * this_drug.gt(0))
            return torch.cat(doses, 1) @ self.drug_embeddings.weight
        else:
            return self.dosers(drugs) @ self.drug_embeddings.weight
        
    def sampling(self, mu, log_var):
        """Samples from standard Normal distribution and applies re-parametrization trick.
           It is actually sampling from latent space distributions with N(mu, var), computed by encoder.
           Parameters
           ----------
           mu: torch.Tensor
                Torch Tensor of Means.
           log_var: torch.Tensor
                Torch Tensor of log. variances.
           Returns
           -------
           Torch Tensor of sampled data.
        """
        var = torch.exp(log_var) + 1e-4
        return Normal(mu, var.sqrt()).rsample()

    def predict(
        self, 
        genes, 
        drugs, 
        covariates, 
        return_latent_basal=False,
        return_latent_treated=False,
    ):
        """
        Predict "what would have the gene expression `genes` been, had the
        cells in `genes` with cell types `cell_types` been treated with
        combination of drugs `drugs`.
        """

        genes, drugs, covariates = self.move_inputs_(genes, drugs, covariates)
        if self.loss_ae == 'nb':
            genes = torch.log1p(genes)

        latent_basal = self.encoder(genes)
        
        if self.variational:
            dim = latent_basal.size(1) // 2
            latent_mean = latent_basal[:, :dim]
            latent_var = latent_basal[:, :dim]
            
            latent_basal = self.sampling(latent_mean, latent_var)
        

        latent_treated = latent_basal

        if self.num_drugs > 0:
            latent_treated = latent_treated + self.compute_drug_embeddings_(drugs)
        if self.num_covariates[0] > 0:
            for i, emb in enumerate(self.covariates_embeddings):
                emb = emb.to(self.device)
                latent_treated = latent_treated + emb(
                    covariates[i].argmax(0)
                )  #argmax because OHE

        gene_reconstructions = self.decoder(latent_treated)
#        if self.loss_ae == 'gauss':
            # convert variance estimates to a positive value in [1e-3, \infty)
        dim = gene_reconstructions.size(1) // 2
        gene_means = gene_reconstructions[:, :dim]
        gene_vars = F.softplus(gene_reconstructions[:, dim:]).add(1e-3)
            #gene_vars = gene_reconstructions[:, dim:].exp().add(1).log().add(1e-3)

        if self.loss_ae == 'nb':
            gene_means = F.softplus(gene_means).add(1e-3)
            #gene_reconstructions[:, :dim] = torch.clamp(gene_reconstructions[:, :dim], min=1e-4, max=1e4)
            #gene_reconstructions[:, dim:] = torch.clamp(gene_reconstructions[:, dim:], min=1e-4, max=1e4)
        gene_reconstructions = torch.cat([gene_means, gene_vars], dim=1)
                
        if return_latent_basal:
            if return_latent_treated:
                return gene_reconstructions, latent_basal, latent_treated
            else:
                return gene_reconstructions, latent_basal
        if return_latent_treated:
            return gene_reconstructions, latent_treated
        return gene_reconstructions

    def early_stopping(self, score):
        """
        Decays the learning rate, and possibly early-stops training.
        """
        self.scheduler_autoencoder.step()
        self.scheduler_adversary.step()
        self.scheduler_dosers.step()

        if score > self.best_score:
            self.best_score = score
            self.patience_trials = 0
        else:
            self.patience_trials += 1

        return self.patience_trials > self.patience

    def update(self, genes, drugs, covariates):
        """
        Update CPA's parameters given a minibatch of genes, drugs, and
        cell types.
        """
        genes, drugs, covariates = self.move_inputs_(genes, drugs, covariates)
        gene_reconstructions, latent_basal = self.predict(
            genes,
            drugs,
            covariates,
            return_latent_basal=True,
        )

        dim = gene_reconstructions.size(1) // 2
        gene_means = gene_reconstructions[:, :dim]
        gene_vars = gene_reconstructions[:, dim:]
        reconstruction_loss = self.loss_autoencoder(gene_means, genes, gene_vars)
        adversary_drugs_loss = torch.tensor([0.0], device=self.device)
        if self.num_drugs > 0:
            adversary_drugs_predictions = self.adversary_drugs(latent_basal)
            adversary_drugs_loss = self.loss_adversary_drugs(
                adversary_drugs_predictions, drugs.gt(0).float()
            )

        adversary_covariates_loss = torch.tensor(
            [0.0], device=self.device
        )
        if self.num_covariates[0] > 0:
            adversary_covariate_predictions = []
            for i, adv in enumerate(self.adversary_covariates):
                adv = adv.to(self.device)
                adversary_covariate_predictions.append(adv(latent_basal))
                adversary_covariates_loss += self.loss_adversary_covariates[i](
                    adversary_covariate_predictions[-1], covariates[i].argmax(1)
                )

        # two place-holders for when adversary is not executed
        adversary_drugs_penalty = torch.tensor([0.0], device=self.device)
        adversary_covariates_penalty = torch.tensor([0.0], device=self.device)

        if self.iteration % self.hparams['training']['FC']["adversary_steps"]:

            def compute_gradients(output, input):
                grads = torch.autograd.grad(output, input, create_graph=True)
                grads = grads[0].pow(2).mean()
                return grads

            if self.num_drugs > 0:
                adversary_drugs_penalty = compute_gradients(
                    adversary_drugs_predictions.sum(), latent_basal
                )

            if self.num_covariates[0] > 0:
                adversary_covariates_penalty = torch.tensor([0.0], device=self.device)
                for pred in adversary_covariate_predictions:
                    adversary_covariates_penalty += compute_gradients(
                        pred.sum(), latent_basal
                    )  # TODO: Adding up tensor sum, is that right?

            self.optimizer_adversaries.zero_grad()
            (
                adversary_drugs_loss
                + self.hparams['training']['FC']["penalty_adversary"] * adversary_drugs_penalty
                + adversary_covariates_loss
                + self.hparams['training']['FC']["penalty_adversary"] * adversary_covariates_penalty
            ).backward()
            self.optimizer_adversaries.step()
        else:
            self.optimizer_autoencoder.zero_grad()
            if self.num_drugs > 0:
                self.optimizer_dosers.zero_grad()
            (
                reconstruction_loss
                - self.hparams['training']['FC']["reg_adversary"] * adversary_drugs_loss
                - self.hparams['training']['FC']["reg_adversary"] * adversary_covariates_loss
            ).backward()
            self.optimizer_autoencoder.step()
            if self.num_drugs > 0:
                self.optimizer_dosers.step()
        self.iteration += 1

        return {
            "loss_reconstruction": reconstruction_loss.item(),
            "loss_adv_drugs": adversary_drugs_loss.item(),
            "loss_adv_covariates": adversary_covariates_loss.item(),
            "penalty_adv_drugs": adversary_drugs_penalty.item(),
            "penalty_adv_covariates": adversary_covariates_penalty.item(),
        }

    @classmethod
    def defaults(self):
        """
        Returns the list of default hyper-parameters for CPA
        """

        return self.set_hparams_(self, "")
    
    
    
class KnowledgeCPA(torch.nn.Module):
            
    def __init__(self,
                 n_nodes,
                 fc_genes,
                 num_drugs,
                 num_cell_types,
                 genes_index_set,
                 doser_type='mlp',
                 aggregation='mlp',
                 adversarial=False,
                 device="cpu",
                 hparams="",
                 seed=0,
                 variational=False,
                 flavour='combine'):

        self.n_nodes = n_nodes
        self.n_drugs = num_drugs
        self.flavour = flavour
        self.n_genes = n_nodes - num_drugs + 1
        self.n_cell_types = num_cell_types
        self.fc_genes = fc_genes
        
        self.total_genes = self.fc_genes + self.n_genes
        
        if self.flavour == 'plugin':
            self.total_genes = self.fc_genes
        
        self.genes_index = list(genes_index_set) # index of the genes in the graphs of the batches
        self.aggregation = aggregation  # aggregation option, mlp, sum, concat...
        if self.aggregation == 'sum':
            self.knowledge_weight = self.hparams['architecture']['Global']['knowledge_weight']
            
        self.adversarial = adversarial # adversarial option, bool
        self.device = device
        
        self.variational = variational
        self.beta = 0

        super().__init__()
        self.loss_autoencoder = nn.GaussianNLLLoss()
        self.loss_adversary_drugs = torch.nn.BCEWithLogitsLoss()
        self.loss_adversary_cell_types = torch.nn.CrossEntropyLoss()
        self.iteration = 0

        self.set_hparams_(seed, hparams)
        
        # Beta-VAE hyperparams
        self.warmup_kl = self.hparams['training']['Global']['warmup_kl']
        self.max_epoch_kl = self.hparams['training']['Global']['max_epoch_kl']
        self.max_beta = self.hparams['training']['Global']['max_beta']        
        
        # reconstruction matrix loss weight
        self.matrix_loss_weight = self.hparams['training']['Global']['matrix_loss_weight']
        
        if self.hparams['architecture']['Global']['loss'] == 'mse':
            self.loss_autoencoder = nn.MSELoss()
            
        knowledge_AE = GraphAutoencoder(n_nodes=self.n_nodes ,
                                        num_drugs=self.n_drugs,
                                        num_cell_types=self.n_cell_types,
                                        genes_index_set=genes_index_set,
                                        doser_type=doser_type,
                                        adversarial=False, # must be False
                                        device="cpu",
                                        hparams=hparams,
                                        seed=0,
                                        variational=variational)
        
        data_AE = CPA(num_genes=self.fc_genes,
                  num_drugs=self.n_drugs,
                  num_covariates=self.n_cell_types,
                  doser_type="mlp",
                  hparams=hparams,
                  #adversarial=False,  # must be False
                  device="cpu",
                  seed=0,
                  patience=5,
                  decoder_activation="linear",)
 
        self.graph_encoder = knowledge_AE.graph_encoder
        print("Global Graph encoder: ", self.graph_encoder)
        self.graph_fc_encoder = knowledge_AE.fc_encoder
        print("Global Graph FC encoder: ", self.graph_fc_encoder)
        self.fc_encoder = data_AE.encoder
        print("Global FC encoder: ", self.fc_encoder) 
        
        self.decoder = MLP(
            [self.hparams['architecture']['Global']["z_dim"]]
            + self.hparams['architecture']['Global']["decoder_layers"] #[self.hparams['architecture']['Global']["decoder_width"]] * self.hparams['architecture']['Global']["decoder_depth"]
            + [self.total_genes * 2],
            last_layer_act="linear",
        )
        print("Global decoder: ", self.decoder)
        
        if self.aggregation in ['linear','mlp']:
            factor_var = 2 if self.hparams['architecture']['Global']["variational"] else 1  # if variational, needed to generate both mean and std
            activation = self.aggregation if self.aggregation == 'linear' else 'ReLU'
            self.aggregator = MLP(
                [self.hparams['architecture']['FC']["z_dim"] + self.hparams['architecture']['GNN']["z_dim"]] +
                [self.hparams['architecture']['Global']["aggregator_width"]] *
                self.hparams['architecture']['Global']["aggregator_depth"] +
                [self.hparams['architecture']['Global']["z_dim"]*factor_var], 
                last_layer_act=activation,
                dropout=self.hparams['architecture']['Global']["aggregator_dropout"]
            )
            print("Aggregator: ", self.aggregator)
        
        if self.adversarial is True:
            self.adversary_drugs = MLP(
                [self.hparams['architecture']['Global']["z_dim"]] +
                [self.hparams['architecture']['Global']["adversary_width"]] *
                self.hparams['architecture']['Global']["adversary_depth"] +
                [self.n_drugs], dropout=self.hparams['architecture']['Global']["adversary_dropout"]
            )
            print("Discriminator Drugs: ", self.adversary_drugs)

            self.adversary_cell_types = MLP(
                [self.hparams['architecture']['Global']["z_dim"]] +
                [self.hparams['architecture']['Global']["adversary_width"]] *
                self.hparams['architecture']['Global']["adversary_depth"] +
                [self.n_cell_types], dropout=self.hparams['architecture']['Global']["adversary_dropout"]
            )
            print("Discriminator Covs: ", self.adversary_cell_types)

            # set dosers
            self.doser_type = doser_type
            if doser_type == 'mlp':
                self.dosers = torch.nn.ModuleList()
                for _ in range(self.n_drugs):
                    self.dosers.append(
                        MLP([1] +
                            [self.hparams['architecture']['Global']["dosers_width"]] *
                            self.hparams['architecture']['Global']["dosers_depth"] +
                            [1],
                            batch_norm=False))
            #    print("Dosers: ", self.dosers)


            # Drug and covariate (cell_type) embeddings
            self.drug_embeddings = torch.nn.Embedding(
                self.n_drugs, self.hparams['architecture']['Global']["z_dim"])
            self.cell_type_embeddings = torch.nn.Embedding(
                self.n_cell_types, self.hparams['architecture']['Global']["z_dim"])
            
            if self.hparams['architecture']['Global']['node_embeddings']:
                self.node_embeddings = torch.nn.Embedding(
                    self.n_nodes, self.hparams['architecture']['Global']['node_embeddings_dim']
                )

        self.to(self.device)
        
        get_params = lambda model, cond: list(model.parameters()) if cond else []

        self.optimizer_knowledge_autoencoder = torch.optim.Adam(
            list(self.graph_encoder.parameters()) +
            list(self.graph_fc_encoder.parameters()) +
            list(self.fc_encoder.parameters()) +
            list(self.decoder.parameters()) +
            list(self.aggregator.parameters()) if self.aggregation in ['linear','mlp'] else [] +
            get_params(self.drug_embeddings, self.hparams['data']['perturbation'] == 'drug') +
            list(self.cell_type_embeddings.parameters()),
            lr=float(self.hparams['training']['Global']["autoencoder_lr"]),
            weight_decay=float(self.hparams['training']['Global']["autoencoder_wd"])
        )

        if self.adversarial is True:
            self.optimizer_adversaries = torch.optim.Adam(
                list(self.adversary_drugs.parameters()) +
                list(self.adversary_cell_types.parameters()),
                lr=self.hparams['training']['Global']["adversary_lr"],
                weight_decay=self.hparams['training']['Global']["adversary_wd"])

            self.optimizer_dosers = torch.optim.Adam(
                self.dosers.parameters(),
                lr=self.hparams['training']['Global']["dosers_lr"],
                weight_decay=self.hparams['training']['Global']["dosers_wd"])

        # learning rate schedulers
        self.scheduler_autoencoder = torch.optim.lr_scheduler.StepLR(
            self.optimizer_knowledge_autoencoder, step_size=self.hparams['training']['Global']["step_size_lr"])

        if self.adversarial is True:
            self.scheduler_adversary = torch.optim.lr_scheduler.StepLR(
                self.optimizer_adversaries, step_size=self.hparams['training']['Global']["step_size_lr"])

            self.scheduler_dosers = torch.optim.lr_scheduler.StepLR(
                self.optimizer_dosers, step_size=self.hparams['training']['Global']["step_size_lr"])

        self.history = {'epoch': [], 'stats_epoch': []}

        #early-stopping
        self.patience = self.hparams['training']['Global']['patience']
        self.best_score = -1e3
        self.patience_trials = 0
        

    def set_hparams_(self, seed, hparams):
        """
        Hyper-parameters set by config file
        """

        with open(hparams) as f:
            self.hparams = yaml.load(f, Loader=yaml.FullLoader)

        return self.hparams
    
    def beta_callback(self, current_epoch):
        
        beta = -1
        if current_epoch > self.warmup_kl and self.max_epoch_kl > 0:
            beta = (self.max_beta/self.max_epoch_kl)*(current_epoch - self.warmup_kl)
        if current_epoch >= self.max_epoch_kl:
            beta = self.max_beta
            
        return max(beta, 0)
    
    def sampling(self, mu, log_var):
        """Samples from standard Normal distribution and applies re-parametrization trick.
           It is actually sampling from latent space distributions with N(mu, var), computed by encoder.
           Parameters
           ----------
           mu: torch.Tensor
                Torch Tensor of Means.
           log_var: torch.Tensor
                Torch Tensor of log. variances.
           Returns
           -------
           Torch Tensor of sampled data.
        """
        var = torch.exp(log_var) + 1e-4
        return Normal(mu, var.sqrt()).rsample()
    
    def early_stopping(self, score):
        """
        Decays the learning rate, and possibly early-stops training.
        """
        self.scheduler_autoencoder.step()
        
        if self.adversarial:
            self.scheduler_adversary.step()
            self.scheduler_dosers.step()

        if score > self.best_score:
            self.best_score = score
            self.patience_trials = 0
        else:
            self.patience_trials += 1

        return self.patience_trials > self.patience

    def compute_drug_embeddings_(self, drugs):
        """
        Compute sum of drug embeddings, each of them multiplied by its
        dose-response curve.
        """

        if self.doser_type == 'mlp':
            doses = []
            for d in range(drugs.size(1)):
                this_drug = drugs[:, d].view(-1, 1)
                doses.append(self.dosers[d](this_drug).sigmoid() * this_drug.gt(0))
            return torch.cat(doses, 1) @ self.drug_embeddings.weight
        else:
            return self.dosers(drugs) @ self.drug_embeddings.weight
        
    def encode(self, inputs):
        
        x_graph = inputs['graph_x']
        edge_graph = inputs['graph_edge']
        batch_graph = inputs['graph_batch']
        fc_genes = inputs['fc_genes']
        
        encoded_node_features_original = self.graph_encoder(x_graph, edge_graph)
        encoded_node_features, mask = to_dense_batch(encoded_node_features_original, batch_graph, batch_size=len(torch.unique(batch_graph)))
        encoded_node_features = encoded_node_features.view([len(torch.unique(batch_graph)),
                                                            encoded_node_features.shape[1] *
                                                            encoded_node_features.shape[2]])
        knowledge_basal, _ = self.graph_fc_encoder(encoded_node_features)
        
        data_basal = self.fc_encoder(fc_genes)
        
        return dict(
            knowledge_basal=knowledge_basal,
            data_basal=data_basal
        )
        
    def emb_aggregation(self, embedding):
        
        knowledge_basal = embedding['knowledge_basal']
        data_basal = embedding['data_basal']
        
        total_basal = torch.concat((knowledge_basal, data_basal), dim=1) # default option
        
        if self.aggregation in ['linear', 'mlp']:
            total_basal = self.aggregator(total_basal)
            
        elif self.aggregation == 'sum':
            total_basal = self.knowledge_weight*knowledge_basal + data_basal
        
        if self.variational:
            
            if self.aggregation not in ['linear', 'mlp', 'sum']:
                dim = knowledge_basal.size(1) // 2
                
                mean_basal = torch.concat((knowledge_basal[:,:dim], data_basal[:,:dim]), dim=1)
                var_basal = torch.concat((knowledge_basal[:, dim:], data_basal[:, dim:]), dim=1) 
                
                total_basal = torch.concat((mean_basal, var_basal), dim=1) 
            
            dim = total_basal.size(1) // 2
            
            z_mean = total_basal[:, :dim]
            z_var = total_basal[:, dim:]

            total_basal = self.sampling(z_mean, z_var)
            
            return dict(
                total_basal=total_basal,
                z_mean=z_mean,
                z_var=z_var,
            )

        return dict(
            total_basal=total_basal
        )
        
    def decode(self, inputs):
        
        drugs = inputs['fc_drugs']
        covariates = inputs['fc_covariates']
        latent_basal = inputs['total_basal']
        
        latent_treated = latent_basal
        
        if self.n_drugs > 0:
             latent_treated = latent_treated + self.compute_drug_embeddings_(drugs)
        if self.n_cell_types > 0:
            cell_emb = self.cell_type_embeddings(covariates.argmax(1))
            latent_treated = latent_treated + cell_emb

        gene_reconstructions = self.decoder(latent_treated)

        return dict(
            latent_treated = latent_treated,
            gene_reconstructions=gene_reconstructions
        )
        
    def predict(self, inputs, return_latent_basal=False):
        
        encode_outputs = self.encode(inputs)
        aggregation_outputs = self.emb_aggregation(encode_outputs)
        
        total_basal = aggregation_outputs['total_basal']
        
        basal_distribution = None
        if self.variational:
            basal_mean = aggregation_outputs['z_mean']
            basal_var = aggregation_outputs['z_var']
            var = torch.exp(basal_var) + 1e-4
            basal_distribution = Normal(basal_mean, var.sqrt())
            
        if self.adversarial is False:
            gene_reconstructions = self.decode(total_basal)
            
            dim = gene_reconstructions.size(0) // 2
            means = gene_reconstructions[:, :dim]
            variances = gene_reconstructions[:, dim:].exp().add(1).log().add(1e-3)

            gene_reconstructions = torch.cat((means, variances), dim=1)

            return dict(
                nodes_reconstructed=gene_reconstructions,
                basal_distribution=basal_distribution
                )
        
        # Build the latent treated
        decode_inputs = dict(
            inputs,
            total_basal=total_basal,
            basal_distribution=basal_distribution
        )

        gene_reconstructions = self.decode(decode_inputs)
        
        gene_reconstructions = gene_reconstructions['gene_reconstructions']

        # convert variance estimates to a positive value in [1e-3, \infty)
        dim = gene_reconstructions.size(1) // 2
        means = gene_reconstructions[:, :dim]
        variances = gene_reconstructions[:, dim:].exp().add(1).log().add(1e-3)

        gene_reconstructions = torch.cat((means, variances), dim=1)

        if return_latent_basal is True:
            return dict(
                genes_reconstructed=gene_reconstructions,
                latent_basal=decode_inputs['total_basal'],
                basal_distribution=basal_distribution,
            )

        return dict(
            genes_reconstructed=gene_reconstructions,
            basal_distribution=basal_distribution
        )
        
        
    def full_forward(self, inputs):
        
        encode_outputs = self.encode(inputs)
        aggregation_outputs = self.emb_aggregation(encode_outputs)
        
        total_basal = aggregation_outputs['total_basal']
        
        basal_distribution = None
        if self.variational:
            basal_mean = aggregation_outputs['z_mean']
            basal_var = aggregation_outputs['z_var']
            var = torch.exp(basal_var) + 1e-4
            basal_distribution = Normal(basal_mean, var.sqrt())
        
        # Build the latent treated
        decode_inputs = dict(
            inputs,
            total_basal=total_basal,
            basal_distribution=basal_distribution
        )

        decode_outputs = self.decode(decode_inputs)
        
        latent_treated = decode_outputs['latent_treated']
        gene_reconstructions = decode_outputs['gene_reconstructions']

        # convert variance estimates to a positive value in [1e-3, \infty)
        dim = gene_reconstructions.size(1) // 2
        means = gene_reconstructions[:, :dim]
        variances = gene_reconstructions[:, dim:].exp().add(1).log().add(1e-3)

        gene_reconstructions = torch.cat((means, variances), dim=1)

        return dict(
            genes_reconstructed=gene_reconstructions,
            latent_basal=decode_inputs['total_basal'],
            latent_treated=latent_treated,
            basal_distribution=basal_distribution
        )
        
        
    def update(self, inputs, genes_index_all=[], epoch=0):

        """
        Update step during training
        :param node_features:
        :param edge_index:
        :param batch:
        :param drugs:
        :param cell_types:
        :return:
        """

        #self.optimizer_autoencoder.zero_grad()
        predict_outputs = self.predict(inputs,
                                       return_latent_basal=True)
        
        reconstructed_genes = predict_outputs['genes_reconstructed']
        latent_basal = predict_outputs['latent_basal']
        
        # take mean and variance
        dim = reconstructed_genes.size(1) // 2
        
        x = inputs['graph_x'][:, 1]
        fc_x = inputs['fc_genes']
        x = x[genes_index_all]
        
        x = x.view(fc_x.size(0),-1)
        
        total_x = torch.concat((x, fc_x), dim=1)
        
        if self.flavour == 'plugin':
            total_x = fc_x           
        
        mean = reconstructed_genes[:, :dim]
        variance = reconstructed_genes[:, dim:]
        
        mean = mean.view(fc_x.size(0),-1)
        variance = variance.view(fc_x.size(0),-1)

        #mean = mean[genes_index_all, :].flatten()
        #variance = variance[genes_index_all, :].flatten()
        
        # compute the KL divergence if variational
        kl_loss = torch.tensor([0.0], device=self.device)
        if self.variational:
            basal_distribution = predict_outputs["basal_distribution"]
            dist_pz = Normal(
                torch.zeros_like(basal_distribution.loc), torch.ones_like(basal_distribution.scale)
            )
            kl_loss = kl_divergence(basal_distribution, dist_pz).sum(-1)
            self.beta = self.beta_callback(epoch)
        
        # compute reconstruction loss
        if self.hparams['architecture']['Global']['loss'] == 'mse':
            reconstruction_loss = self.loss_autoencoder(total_x, mean)
        else:
            reconstruction_loss = self.loss_autoencoder(total_x, mean, variance)

        # If adversarial mode and not in warmup
        if self.adversarial is True:
            
            adversary_drugs_loss = torch.tensor([0.0], device=self.device)
            
            # If more there are some drugs, try to find it on the latent space
            if self.n_drugs > 0:
                adversary_drugs_predictions = self.adversary_drugs(  # predict drugs
                    latent_basal)
                adversary_drugs_loss = self.loss_adversary_drugs(  #  BCE loss
                    adversary_drugs_predictions, inputs['fc_drugs'].gt(0).float())

            # Same with cells
            adversary_cell_types_loss = torch.tensor([0.0], device=self.device)   
            if self.n_cell_types > 0:
                adversary_cell_types_predictions = self.adversary_cell_types(
                    latent_basal)
                adversary_cell_types_loss = self.loss_adversary_cell_types(
                    adversary_cell_types_predictions, inputs['fc_covariates'].argmax(1))

            # two place-holders for when adversary is not executed
            adversary_drugs_penalty = torch.Tensor([0])
            adversary_cell_types_penalty = torch.Tensor([0])


            # We train the adversarial networks "adversary_steps" times more than the AE
            if self.iteration % self.hparams['training']['Global']["adversary_steps"] and epoch >= self.hparams['training']['Global']['epochs_warmup']:
                
                # Compute the gradients with respect to the basal
                adversary_drugs_penalty = torch.autograd.grad(
                    adversary_drugs_predictions.sum(),
                    latent_basal,
                    create_graph=True)[0].pow(2).mean()

                adversary_cell_types_penalty = torch.autograd.grad(
                    adversary_cell_types_predictions.sum(),
                    latent_basal,
                    create_graph=True)[0].pow(2).mean()

                self.optimizer_adversaries.zero_grad()
                (adversary_drugs_loss +
                 self.hparams['training']['Global']["penalty_adversary"] *
                 adversary_drugs_penalty +
                 adversary_cell_types_loss +
                 self.hparams['training']['Global']["penalty_adversary"] *
                 adversary_cell_types_penalty).backward()
                self.optimizer_adversaries.step()  # just step the adversarial networks
            else:                
                self.optimizer_knowledge_autoencoder.zero_grad()
                self.optimizer_dosers.zero_grad()
                (reconstruction_loss + self.beta*kl_loss.mean() -
                self.hparams['training']['Global']["reg_adversary"] *
                 adversary_drugs_loss -
                 self.hparams['training']['Global']["reg_adversary"] *
                 adversary_cell_types_loss).backward()
                #print(self.CPA.encoder.network[0].weight.grad)
                self.optimizer_knowledge_autoencoder.step()  # step the autoencoder
                self.optimizer_dosers.step()  # and the dosers

            self.iteration += 1

            return {
                "loss_reconstruction": reconstruction_loss.item(),
                "loss_kl": kl_loss.mean().item(),
                "loss_adv_drugs": adversary_drugs_loss.item(),
                "loss_adv_cell_types": adversary_cell_types_loss.item(),
                "penalty_adv_drugs": adversary_drugs_penalty.item(),
                "penalty_adv_cell_types": adversary_cell_types_penalty.item()
            }

        else:
            self.optimizer_knowledge_autoencoder.zero_grad()
            reconstruction_loss.backward()
            self.optimizer_knowledge_autoencoder.step()

        return {
            "loss_reconstruction": reconstruction_loss.item(),
            "kl_loss": kl_loss.mean().item(),
            "loss_adv_drugs": 0,
            "loss_adv_cell_types": 0,
            "penalty_adv_drugs": 0,
            "penalty_adv_cell_types": 0
        }