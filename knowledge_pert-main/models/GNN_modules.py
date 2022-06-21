import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, Sequential, ChebConv, GATConv

from torch_scatter import scatter_mean


class GNN_Encoder(nn.Module):

    """
    Constructs the encoder network of GAE. This class implements the
    encoder part of Variational GAE. The decoder is a InnerProduct Decoder.
    Parameters
    ----------
    x_dim: Int
        Input dim
    layer_sizes: List
        Hidden layer sizes.
    z_dimension: integer
        number of latent space dimensions.
    dropout_rate: float
        dropout rate
    Cheb: str
        Use ChebConv or GCNConv
    k: int
        paramter of ChebConv
    """

    def __init__(self, x_dim: int, layer_sizes: list, z_dimension: int, dropout_rate: float, Conv: str = 'GCN', k: int = 2):
        super().__init__()

        module = [] # Create first module as list
        if len(layer_sizes) > 0:
            if Conv == 'Cheb':
                module.append((ChebConv(x_dim, layer_sizes[0], K=k, bias=False), 'x, edge_index -> x'))
            elif Conv == 'GAT':
                module.append((GATConv(in_size, layer_sizes[0], heads=k, bias=False), 'x, edge_index -> x'))
            else:
                module.append((GCNConv(x_dim, layer_sizes[0], bias=False), 'x, edge_index -> x'))
            module.append(nn.LeakyReLU(negative_slope=0.3))
            module.append(nn.Dropout(p=dropout_rate))
            for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
                if Conv == 'Cheb':
                    module.append((ChebConv(in_size, out_size, K=k, bias=False), 'x, edge_index -> x'))
                elif Conv == 'GAT':
                    module.append((GATConv(in_size, out_size, heads=k, bias=False), 'x, edge_index -> x'))
                else:
                    module.append((GCNConv(in_size, out_size, bias=False), 'x, edge_index -> x'))
                module.append(nn.LeakyReLU(negative_slope=0.3))
                module.append(nn.Dropout(p=dropout_rate))

        self.GNN = Sequential('x, edge_index', module)  # create now sequential container

        if Conv == 'Cheb':
            self.mean_encoder = ChebConv(layer_sizes[-1], z_dimension, K=k)
            self.log_var_encoder = ChebConv(layer_sizes[-1], z_dimension, K=k)
        if Conv == 'GAT':
            self.mean_encoder = GATConv(layer_sizes[-1], z_dimension, K=k)
            self.log_var_encoder = GATConv(layer_sizes[-1], z_dimension, K=k)
        else:
            self.mean_encoder = GCNConv(layer_sizes[-1], z_dimension)
            self.log_var_encoder = GCNConv(layer_sizes[-1], z_dimension)

        print(self)

    def forward(self, x, edge_index, graph_index):
        x = self.GNN(x, edge_index).relu()
        mean = self.mean_encoder(x, edge_index)
        logvar = self.log_var_encoder(x, edge_index)
        if graph_index is not None:
            mean_pool = scatter_mean(mean, graph_index, dim=0)  # mean pooling
        else:
            mean_pool = torch.mean(mean, dim=0)
        return mean, logvar, mean_pool


class GNN_network(nn.Module):

    """
    General GNN
    Parameters
    ----------
    x_dim: Int
        Input dim
    layer_sizes: List
        Hidden layer sizes.
    dropout_rate: float
        dropout rate
    """

    def __init__(self, x_dim: int, layer_sizes: list, dropout_rate: float, Conv: str = 'GCN', k: int = 2):
        super().__init__()

        module = [] # Create first module as list
        if len(layer_sizes) > 0:
            if Conv == 'Cheb':
                module.append((ChebConv(x_dim, layer_sizes[0], K=k, bias=False), 'x, edge_index -> x'))
            elif Conv == 'GAT':
                module.append((GATConv(x_dim, layer_sizes[0], heads=k, bias=False), 'x, edge_index -> x'))
            else:
                module.append((GCNConv(x_dim, layer_sizes[0], bias=False), 'x, edge_index -> x'))
            module.append(nn.LeakyReLU(negative_slope=0.3))
            module.append(nn.Dropout(p=dropout_rate))
            for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
                if Conv == 'Cheb':
                    module.append((ChebConv(in_size, out_size, K=k, bias=False), 'x, edge_index -> x'))
                elif Conv == 'GAT':
                    if i == (len(layer_sizes)-2):
                        module.append((GATConv(in_size*k, out_size, heads=1, bias=False), 'x, edge_index -> x'))
                    else:
                        module.append((GATConv(in_size*k, out_size, heads=k, bias=False), 'x, edge_index -> x'))
                else:
                    module.append((GCNConv(in_size, out_size, bias=False), 'x, edge_index -> x'))
                module.append(nn.LeakyReLU(negative_slope=0.3))
                module.append(nn.Dropout(p=dropout_rate))

        self.GNN = Sequential('x, edge_index', module)  # create now sequential container

        print(self)

    def forward(self, x, edge_index):
        x = self.GNN(x, edge_index).relu()
        return x

