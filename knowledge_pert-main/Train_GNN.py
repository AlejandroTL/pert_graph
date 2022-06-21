import argparse
import yaml
from torch_geometric.nn import GAE, VGAE
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models.GNN_modules import *
from utils.gcn_train_utils import *
from utils.utils import *

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", help="path of YAML config file")
    args = parser.parse_args()

    tb_path = 'TB-GNN'
    if not os.path.exists(tb_path):
        os.makedirs(tb_path)

    with open(args.config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # CREATE DATALOADERS
    dataloader_dict = dict(
        data_path=config['parameters']['GNN']['data_path'],
        batch_size=config['parameters']['GNN']['batch_size']
    )

    dataloaders = dataloaders_setup(dataloader_dict)

    # DEVICE, OPTIMIZER AND MODEL
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Define device
    in_channels, out_channels = next(iter(dataloaders['train_dataloader'])).num_features, config['parameters']['GNN']['z_dim']  # in/out dims

    #model = GAE(GCNEncoder(in_channels, out_channels))  # Graph Autoencoder
    if config['parameters']['GNN']['model'] == 'GAE':
        model = GAE(GNN_Encoder(in_channels,
                                config['parameters']['GNN']['layers'],
                                config['parameters']['GNN']['z_dim'],
                                config['parameters']['GNN']['dropout_rate']
                                ))

    else:
        model = VGAE(GNN_Encoder(in_channels,
                                 config['parameters']['GNN']['layers'],
                                 config['parameters']['GNN']['z_dim'],
                                 config['parameters']['GNN']['dropout_rate']
                                 ))
    model = model.to(device)  # model to device
    optimizer = torch.optim.Adam(model.parameters(), lr=config['parameters']['GNN']['lr'])  # set optimizer

    # CALLBACKS
    if config['parameters']['GNN']['lr_scheduler']:
        scheduler = ReduceLROnPlateau(optimizer, factor=config['parameters']['GNN']['lr_factor'],
                                      patience=config['parameters']['GNN']['lr_patience'])
    else:
        scheduler = None

    if config['parameters']['GNN']['early_stopping']:
        early_stopping = EarlyStopping(patience=config['parameters']['GNN']['es_patience'])
    else:
        early_stopping = None

    # TRAIN ROUTINE
    train_routine(epochs=config['parameters']['GNN']['epochs'],
                  model=model,
                  train_loader=dataloaders['train_dataloader'],
                  test_loader=dataloaders['test_dataloader'],
                  optimizer=optimizer,
                  lr_scheduler=scheduler,
                  early_stopping=early_stopping,
                  device=device,
                  tb_dir=osp.join(tb_path, config['name']))
