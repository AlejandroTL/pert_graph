import argparse

from utils.GraphAutoencoder_train_utils import *
from utils.utils import *

import umap
import umap.plot
import anndata

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
        batch_size=config['training']['Global']['batch_size'],
        change_drugs=config['data']['change_drugs'],
        perturbation=config['data']['perturbation'],
        condition_key=config['data']['condition_key'],
        perturbation_key=config['data']['condition_key'],
        split_key=config['data']['split_key'],
        dose_key=config['data']['dose_key'],
        cell_type_key=config['data']['cell_type_key'],
        dataloader_splits=False,
        combination=config['data']['combination'],
        fc_genes=config['data']['fc_genes'],
        cell_type_interest=config['data']['cell_type_interest'],
        random=config['data']['random']
    )

    datasets = prepare_data(dataloader_dict)

    train_dataloader = DataLoader(datasets["training"],
                                  batch_size=1
                                  )

    # DEVICE AND MODEL
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Define device

    model = KnowledgeCPA(
        n_nodes=int(next(iter(train_dataloader))[0].num_nodes),
        fc_genes=datasets['training'].fc_genes.size(1),
        num_drugs=datasets['training'].drugs.shape[1],
        num_cell_types=len(datasets['training'].covars_dict),
        genes_index_set=datasets['training'].gene_ordering_index,
        adversarial=config['architecture']['Global']['adversarial'],
        variational=config['architecture']['Global']['variational'],
        aggregation=config['architecture']['Global']['aggregation'],
        flavour=config['architecture']['Global']['flavour'],
        device=device,
        hparams=args.config_path,
        seed=0)
    print("Model set")

    # TRAIN ROUTINE
    model_trained = complex_train_routine(model=model,
                                          config=config,
                                          datasets=datasets,
                                          return_model=True)
    
    
    # GENERATE NEW DATASET WITH THE CELL TYPE OF INTEREST
    # PREDICT RESPONSE IN THE CELL OF INTEREST
    
  #  num = datasets["ood_control_"+ str('CD4 T')].genes.size(0)
  #  drug = "stimulated"
  #  idx = np.where(datasets['training'].drugs_names == drug)[0]
  #  emb_drugs = datasets['training'].drugs[idx][0].view(
  #              1, -1).repeat(num, 1).clone()
  #  emb_cts = datasets["ood_control_"+ str('CD4 T')].cell_types[0][0].view(
  #              1, -1).repeat(num, 1).clone()
    
  #  ood_dataloader = DataLoader(datasets["ood_control_"+ str('CD4 T')],
  #                          batch_size=len(datasets["ood_control_"+ str('CD4 T')]),
  #                          num_workers=0
  #                          )
    
  #  model.eval()
  #  for graph, fc_genes, drug, cov in ood_dataloader:
  #      x = graph.x.to(device)
  #      edge_index = graph.edge_index.to(device)
  #      batch = graph.batch.to(device)
  #      fc_genes = fc_genes.to(device)
        
  #      predict_inputs = dict(
  #                  graph_x=x,
  #                  graph_edge=edge_index,
  #                  graph_batch=batch,
  #                  fc_genes=fc_genes,
  #                  fc_drugs=emb_drugs.to(device),
  #                  fc_covariates=emb_cts.to(device)
  #              )
        
  #      predict_outputs = model_trained.predict(predict_inputs)
        
  #  predicted = predict_outputs['genes_reconstructed']
  #  pred_adata = anndata.AnnData(predicted, obs={"condition": ["pred"] * len(predicted)},
  #                               var={"var_names": datasets["training"].symbols})
    
    path_to_save = config['training']['Global']["save_dir"]
    os.makedirs(path_to_save, exist_ok=True)
    sc.settings.figdir = path_to_save
    
    cell_type_analysis(adata_path=config['data']['data_path'],
                       datasets=datasets,
                       model=model,
                       cell_type='CD4 T',
                       drug='stimulated',
                       path_to_save=path_to_save,
                       figure="a",
                       n_out=7)
    
    # PLOTS
    
    test_dataloader = DataLoader(datasets["test"],
                              batch_size=64,
                              num_workers=0
                              )

    latent_basal_total = torch.empty(size=[0, model_trained.graph_fc_encoder.mean_encoder.out_features]).to(device)
    #cells_reconstructed_total = torch.empty(size=[0, model.drug_embeddings.weight.size(0)]).to(device)
    latent_treated_total = torch.empty(size=[0, model_trained.graph_fc_encoder.mean_encoder.out_features]).to(device)

    model.eval()
    for graph, fc_genes, drug, cov in test_dataloader:
        x = graph.x.to(device)
        edge_index = graph.edge_index.to(device)
        batch = graph.batch.to(device)
        fc_genes = fc_genes.to(device)
        drug = drug.to(device)
        cov = cov.to(device)
        
        predict_inputs = dict(
                    graph_x=x,
                    graph_edge=edge_index,
                    graph_batch=batch,
                    fc_genes=fc_genes,
                    fc_drugs=drug,
                    fc_covariates=cov
                )
        
        ff_outputs = model_trained.full_forward(predict_inputs)
        latent_basal = ff_outputs['latent_basal']
        latent_treated = ff_outputs['latent_treated']
        genes_reconstructed = ff_outputs['genes_reconstructed']
        
        latent_basal_total = torch.cat([latent_basal_total, latent_basal], 0)
        #cells_reconstructed_total = torch.cat([cells_reconstructed_total, genes_reconstructed], 0)
        latent_treated_total = torch.cat([latent_treated_total, latent_treated], 0)
        
        del ff_outputs, latent_basal, latent_treated, genes_reconstructed, edge_index, batch, fc_genes, drug, cov, graph
        torch.cuda.empty_cache()

    if not os.path.exists(config['training']['Global']["save_dir"]):
                os.makedirs(config['training']['Global']["save_dir"])
                
    # PLOT LOSSES
    pretty_history = ComPertHistory(model_trained.history)
    
    pretty_history.plot_losses()
    plt.savefig(os.path.join(config['training']['Global']["save_dir"], 'Reconstruction and Adv Loss.png'))

    pretty_history.plot_metrics(epoch_min=0)
    plt.savefig(os.path.join(config['training']['Global']["save_dir"], 'R2 and disentanglement.png'))
    
    # LATENT BASAL
    mapper = umap.UMAP().fit(latent_basal_total.cpu().detach().numpy())
    umap.plot.points(mapper, labels=datasets["test"].drugs_names[:latent_basal_total.shape[0]])
    plt.title('Latent Basal')
    plt.savefig(os.path.join(config['training']['Global']["save_dir"], "Latent_basal.png"))
    
    # LATENT TREATED
    mapper = umap.UMAP().fit(latent_treated_total.cpu().detach().numpy())
    umap.plot.points(mapper, labels=datasets["test"].drugs_names[:latent_treated_total.shape[0]])
    plt.title('Latent Treated')
    plt.savefig(os.path.join(config['training']['Global']["save_dir"], "Latent_treated.png"))
    
    # DRUG EMBEDDINGS
    mapper = umap.UMAP().fit(model.drug_embeddings.weight.cpu().detach().numpy())
    umap.plot.points(mapper, labels=np.array(['BMS', 'Dex', 'Nutlin', 'SAHA', 'Control']))#, labels=datasets["test"].pert_categories[:latent_treated_total.shape[0]])
    plt.title('Drug embeddings')
    plt.savefig(os.path.join(config['training']['Global']["save_dir"], "Drug_embeddings.png"))

