import argparse

from utils.GraphAutoencoder_train_utils import *
from utils.utils import *

import umap
import umap.plot

import matplotlib.pyplot as plt

from compert.plotting import ComPertHistory

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
        data_path=config['data']['data_path'],
        entities_path=config['data']['entities_path'],
        drugs_path=config['data']['drugs_path'],
        triples_path=config['data']['triples_path'],
        coexpression_network=config['data']['coexpression_path'],
        batch_size=config['training']['GNN']['batch_size'],
        change_drugs=config['data']['change_drugs'],
        perturbation=config['data']['perturbation'],
        condition_key=config['data']['condition_key'],
        perturbation_key=config['data']['condition_key'],
        split_key=config['data']['split_key'],
        dose_key=config['data']['dose_key'],
        cell_type_key=config['data']['cell_type_key'],
        dataloader_splits=False,
        combination=False,
        fc_genes=['SCYL3'],
        random=config['data']['random']
    )

    datasets = prepare_data(dataloader_dict)

    train_dataloader = DataLoader(datasets["training"],
                                  batch_size=1
                                  )

    # DEVICE AND MODEL
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Define device

    model = GraphAutoencoder(
        n_nodes=int(next(iter(train_dataloader))[0].num_nodes),
        num_drugs=datasets['training'].drugs.shape[1],
        num_cell_types=len(datasets['training'].covars_dict),
        genes_index_set=datasets['training'].gene_ordering_index,
        adversarial=config['architecture']['GNN']['adversarial'],
        variational=config['architecture']['GNN']['variational'],
        device=device,
        hparams=args.config_path,
        seed=0)
    print("Model set")

    # TRAIN ROUTINE
    model_trained = train_routine(model=model,
                                  config=config,
                                  datasets=datasets,
                                  return_model=True)
    
    # PLOT EMBEDDINGS
    train_dataloader = DataLoader(datasets["test"],
                              batch_size=64,
                              num_workers=0
                              )

    latent_basal_total = torch.empty(size=[0, model_trained.fc_encoder.mean_encoder.out_features]).to(device)
    nodes_reconstructed_total = torch.empty(size=[0, 2]).to(device)
    latent_treated_total = torch.empty(size=[0, model_trained.fc_encoder.mean_encoder.out_features]).to(device)
    drug_emb_total = torch.empty(size=[0, model_trained.fc_encoder.mean_encoder.out_features]).to(device)
    drug_total = torch.empty(size=[0, model_trained.drug_embeddings.num_embeddings]).to(device)

    model.eval()
    for graph, drug, cov in train_dataloader:
        x = graph.x.to(device)
        edge_index = graph.edge_index.to(device)
        batch = graph.batch.to(device)
        drug = drug.to(device)
        cov = cov.to(device)
        nodes_reconstructed, latent_basal, latent_treated, drugs, drug_emb, cell_emb = model_trained.full_forward(x, edge_index, batch, drug, cov)
        
        latent_basal_total = torch.cat([latent_basal_total, latent_basal], 0)
        nodes_reconstructed_total = torch.cat([nodes_reconstructed_total, nodes_reconstructed], 0)
        latent_treated_total = torch.cat([latent_treated_total, latent_treated], 0)
        drug_emb_total = torch.cat([drug_emb_total, drug_emb], 0)
        drug_total = torch.cat([drug_total, drugs], 0)


    if not os.path.exists(config['training']['GNN']["save_dir"]):
                os.makedirs(config['training']['GNN']["save_dir"])
    # PLOT LOSSES
    pretty_history = ComPertHistory(model_trained.history)
    
    pretty_history.plot_losses()
    plt.savefig(os.path.join(config['training']['GNN']["save_dir"], 'Reconstruction and Adv Loss.png'))

    pretty_history.plot_metrics(epoch_min=0)
    plt.savefig(os.path.join(config['training']['GNN']["save_dir"], 'R2 and disentanglement.png'))
    
    # LATENT BASAL
    mapper = umap.UMAP().fit(latent_basal_total.cpu().detach().numpy())
    umap.plot.points(mapper, labels=datasets["test"].drugs_names[:latent_basal_total.shape[0]])
    plt.title('Latent Basal')
    plt.savefig(os.path.join(config['training']['GNN']["save_dir"], "Latent_basal.png"))
    
    # LATENT TREATED
    mapper = umap.UMAP().fit(latent_treated_total.cpu().detach().numpy())
    umap.plot.points(mapper, labels=datasets["test"].drugs_names[:latent_treated_total.shape[0]])
    plt.title('Latent Treated')
    plt.savefig(os.path.join(config['training']['GNN']["save_dir"], "Latent_treated.png"))
    
    # DRUG EMBEDDINGS
    mapper = umap.UMAP().fit(model.drug_embeddings.weight.cpu().detach().numpy())
    umap.plot.points(mapper, labels=np.array(['BMS', 'Dex', 'Nutlin', 'SAHA', 'Control']))#, labels=datasets["test"].pert_categories[:latent_treated_total.shape[0]])
    plt.title('Drug embeddings')
    plt.savefig(os.path.join(config['training']['GNN']["save_dir"], "Drug_embeddings.png"))
    
    # LATENT BASAL
    mapper = umap.UMAP().fit(drug_emb_total.cpu().detach().numpy())
    umap.plot.points(mapper, labels=datasets["test"].pert_categories[:latent_basal_total.shape[0]])
    plt.title('Drug embeddings * f(dose)')
    plt.savefig(os.path.join(config['training']['GNN']["save_dir"], "Drug_embeddings_dose.png"))

