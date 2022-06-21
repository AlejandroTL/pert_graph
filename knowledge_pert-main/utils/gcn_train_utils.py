import os
from torch.utils.tensorboard import SummaryWriter

import torch


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
    loss = torch.Tensor([0]).to(device)
    for data in train_loader:
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        batch = data.batch.to(device)
        optimizer.zero_grad()
        z, logvar, z_mp = model.encode(x, edge_index, batch)
        loss = model.recon_loss(z, edge_index)
        if model.__class__.__name__ == 'VGAE':
            z, logvar, z_mp = model.encode(x, edge_index, batch)
            loss = model.recon_loss(z, edge_index)
            loss = loss + (1 / data.num_nodes) * model.kl_loss(z, logvar)
        loss.backward()
        optimizer.step()
    return float(loss)


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
    loss = torch.Tensor([0]).to(device)
    for data in test_loader:
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        batch = data.batch.to(device)
        z, logvar, z_mp = model.encode(x, edge_index, batch)
        loss = model.recon_loss(z, edge_index)

    return float(loss)


def train_routine(epochs, model, train_loader, test_loader, optimizer, lr_scheduler, early_stopping, device, tb_dir):

    """
    Entire training routine with callbacks and loggings
    :param epochs:
    :param model:
    :param train_loader:
    :param test_loader:
    :param optimizer:
    :param lr_scheduler:
    :param early_stopping:
    :param device:
    :param tb_dir:
    :return:
    """

    writer = SummaryWriter(tb_dir)

    for epoch in range(epochs+1):
        # At each epoch, initialize the losses at 0
        train_running_loss = torch.Tensor([0]).to(device)
        test_running_loss = torch.Tensor([0]).to(device)

        recon_loss_train = train(model, train_loader, optimizer, device)
        recon_loss_test = test(model, test_loader, device)

        train_running_loss += recon_loss_train
        test_running_loss += recon_loss_test

        writer.add_scalar('training loss', train_running_loss, epoch)
        writer.add_scalar('test loss', test_running_loss, epoch)

        if lr_scheduler is not None:
            lr_scheduler.step(test_running_loss)
        if early_stopping is not None:
            early_stopping(test_running_loss)
            if early_stopping.early_stop:
                print("Early stopped at epoch ", epoch)
                break

    path = 'trained_models/GNN_models'
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(model, os.path.join(path, os.path.split(tb_dir)[1]))
    print("Training finished")







