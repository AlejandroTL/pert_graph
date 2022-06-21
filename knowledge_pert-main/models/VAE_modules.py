import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    Constructs the encoder network of VAE. This class implements the
    encoder part of Variational Auto-encoder.
    Parameters
    ----------
    x_dim: Int
        Input dim
    layer_sizes: List
        Layer sizes, included input size.
    z_dimension: integer
        number of latent space dimensions.
    dropout_rate: float
        dropout rate
    batch_norm: bool
        If True, use batch norm
    """

    def __init__(self, x_dim: int, layer_sizes: list, z_dimension: int, dropout_rate: float, batch_norm: bool):
        super().__init__()

        # encoder architecture
        self.FC = None
        last_layer = x_dim
        if len(layer_sizes) > 0:
            print("Encoder Architecture:")
            self.FC = nn.Sequential()
            self.FC.add_module(name="FirstL", module=nn.Linear(x_dim, layer_sizes[0], bias=False))
            print("\tFirst Layer in, out:", x_dim, layer_sizes[0])
            for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
                print("\tHidden Layer", i, "in/out:", in_size, out_size)
                self.FC.add_module(name="L{:d}".format(i), module=nn.Linear(in_size, out_size, bias=False))
                if batch_norm is True:
                    self.FC.add_module("N{:d}".format(i), module=nn.BatchNorm1d(out_size))
                self.FC.add_module(name="A{:d}".format(i), module=nn.LeakyReLU(negative_slope=0.3))
                self.FC.add_module(name="D{:d}".format(i), module=nn.Dropout(p=dropout_rate))
            last_layer = layer_sizes[-1]

        print("\tMean/Var Layer in/out:", last_layer, z_dimension)
        self.mean_encoder = nn.Linear(last_layer, z_dimension)
        self.log_var_encoder = nn.Linear(last_layer, z_dimension)

    def forward(self, x: torch.Tensor):
        if self.FC is not None:
            x = self.FC(x)

        mean = self.mean_encoder(x)
        log_var = self.log_var_encoder(x)
        return mean, log_var


class Decoder(nn.Module):
    """
            Constructs the decoder of VAE. This class implements the
            decoder part of Variational Auto-encoder. Decodes data from latent space to data space.
            # Parameters
               z_dimension: integer
               number of latent space dimensions.
               layer_sizes: List
               Hidden layer sizes.
               x_dimension: integer
               number of gene expression space dimensions.
               dropout_rate: float
               dropout rate
        """
    def __init__(self, z_dimension: int, layer_sizes: list, x_dim: int,  dropout_rate: float, batch_norm: bool):
        super().__init__()

        layer_sizes = layer_sizes
        # decoder architecture
        print("Decoder Architecture:")

        if len(layer_sizes) > 0:
            # Create first Decoder layer
            self.FirstL = nn.Sequential()
            print("\tFirst Layer in, out", z_dimension, layer_sizes[0])
            self.FirstL.add_module(name="L0", module=nn.Linear(z_dimension, layer_sizes[0], bias=False))
            if batch_norm is True:
                self.FirstL.add_module("N0", module=nn.BatchNorm1d(layer_sizes[0]))
            self.FirstL.add_module(name="A0", module=nn.LeakyReLU(negative_slope=0.3))
            self.FirstL.add_module(name="D0", module=nn.Dropout(p=dropout_rate))

            # Create all Decoder hidden layers
            if len(layer_sizes) > 1:
                self.HiddenL = nn.Sequential()
                for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
                    print("\tHidden Layer", i+1, "in/out:", in_size, out_size)
                    self.HiddenL.add_module(name="L{:d}".format(i+1), module=nn.Linear(in_size, out_size, bias=False))
                    if batch_norm is True:
                        self.HiddenL.add_module("N{:d}".format(i+1), module=nn.BatchNorm1d(out_size, affine=True))
                    self.HiddenL.add_module(name="A{:d}".format(i+1), module=nn.LeakyReLU(negative_slope=0.3))
                    self.HiddenL.add_module(name="D{:d}".format(i+1), module=nn.Dropout(p=dropout_rate))
            else:
                self.HiddenL = None

            # Create Output Layers
            print("\tOutput Layer in/out: ", layer_sizes[-1], x_dim, "\n")
            self.recon_decoder = nn.Sequential(nn.Linear(layer_sizes[-1], x_dim))

        else:

            self.FirstL = None
            self.HiddenL = None
            print("\tLinear Decoder in/out: ", z_dimension, x_dim, "\n")
            self.recon_decoder = nn.Sequential(nn.Linear(z_dimension, x_dim))

    def forward(self, z: torch.Tensor):

        if self.FirstL is not None:
            dec_latent = self.FirstL(z)
        else:
            dec_latent = z

        # Compute Hidden Output
        if self.HiddenL is not None:
            x = self.HiddenL(dec_latent)
        else:
            x = dec_latent

        # Compute Decoder Output
        recon_x = self.recon_decoder(x)
        return recon_x
