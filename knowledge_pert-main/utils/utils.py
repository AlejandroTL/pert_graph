import torch
import random
import math
import numpy as np
import scanpy as sc
import json
import anndata
import matplotlib
import numpy

import os
import os.path as osp
from scipy import stats, sparse
from adjustText import adjust_text
from matplotlib import pyplot
import pandas as pd

from torch_geometric.loader import DataLoader
from data_handling.GCN_dataset import Dataset2, load_dataset_splits
from data_handling.process_data import official_names_drugs
#from cpa.data import Dataset, load_dataset_splits


def mean_pool(input_dict):

    induced_subgraph = input_dict['subgraph']
    embeddings = input_dict['embeddings']
    tf = input_dict['TriplesFactory']

    # Get all entities involved in the graph
    entities = set(induced_subgraph[0]).union(set(induced_subgraph[2]))

    # Get all the KG ids of the entities
    ids = []
    for i in set(induced_subgraph[0]).union(set(induced_subgraph[2])):
        ids.append(tf.entity_to_id[i])

    # Just take mean of KGE of all nodes
    graph_embedding = torch.mean(embeddings()[ids], 0)

    return dict(
        graph_embedding=graph_embedding
    )


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def prepare_data(input_dict):

    data_path = input_dict['data_path']

    random.seed(42)  # set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    g = torch.Generator()
    g.manual_seed(0)

    if input_dict['change_drugs'] is True:
        # Generate automatically the official name of the drugs for the specific dataset
        ond = dict(
            data_path=data_path,
            drug_path=input_dict['drugs_path']
        )
        official_names_drugs(ond)

    official_paths = os.path.basename(os.path.normpath(data_path)).split('.')[0]
    official_paths = os.path.join('omnipath_triples/official_drug_names/', official_paths + ".json")

    if input_dict['perturbation'] == 'drug':
        # Ad hoc dictionary with equivalences of names, there's no other way of doing it
        with open(official_paths) as json_file:
            drugs_official_name = json.load(json_file)
    else:
        drugs_official_name = None

    dataset_dict = dict(
        input_dict,
        drugs_official=drugs_official_name,
    )

    datasets = load_dataset_splits(dataset_dict)

    return datasets

def reg_mean_plot(adata, condition_key, axis_keys, labels, path_to_save="./reg_mean.pdf", gene_list=None, top_100_genes=None,
                  show=False,
                  legend=True, title=None,
                  x_coeff=0.30, y_coeff=0.8, fontsize=14, **kwargs):
    """
        Plots mean matching figure for a set of specific genes.
        # Parameters
            adata: `~anndata.AnnData`
                Annotated Data Matrix.
            condition_key: basestring
                Condition state to be used.
            axis_keys: dict
                dictionary of axes labels.
            path_to_save: basestring
                path to save the plot.
            gene_list: list
                list of gene names to be plotted.
            show: bool
                if `True`: will show to the plot after saving it.
        # Example
        ```python
        import anndata
        import scgen
        import scanpy as sc
        train = sc.read("./tests/data/train.h5ad", backup_url="https://goo.gl/33HtVh")
        network = scgen.VAEArith(x_dimension=train.shape[1], model_path="../models/test")
        network.train(train_data=train, n_epochs=0)
        unperturbed_data = train[((train.obs["cell_type"] == "CD4T") & (train.obs["condition"] == "control"))]
        condition = {"ctrl": "control", "stim": "stimulated"}
        pred, delta = network.predict(adata=train, adata_to_predict=unperturbed_data, conditions=condition)
        pred_adata = anndata.AnnData(pred, obs={"condition": ["pred"] * len(pred)}, var={"var_names": train.var_names})
        CD4T = train[train.obs["cell_type"] == "CD4T"]
        all_adata = CD4T.concatenate(pred_adata)
        scgen.plotting.reg_mean_plot(all_adata, condition_key="condition", axis_keys={"x": "control", "y": "pred", "y1": "stimulated"},
                                     gene_list=["ISG15", "CD3D"], path_to_save="tests/reg_mean.pdf", show=False)
        network.sess.close()
        ```
    """
    import seaborn as sns
    sns.set()
    sns.set(color_codes=True)
    if sparse.issparse(adata.X):
        adata.X = adata.X.A
    diff_genes = top_100_genes
    stim = adata[adata.obs[condition_key] == axis_keys["y"]]
    ctrl = adata[adata.obs[condition_key] == axis_keys["x"]]
    if diff_genes is not None:
        if hasattr(diff_genes, "tolist"):
            diff_genes = diff_genes.tolist()
        adata_diff = adata[:, diff_genes]
        stim_diff = adata_diff[adata_diff.obs[condition_key] == axis_keys["y"]]
        ctrl_diff = adata_diff[adata_diff.obs[condition_key] == axis_keys["x"]]
        x_diff = numpy.average(ctrl_diff.X, axis=0)
        y_diff = numpy.average(stim_diff.X, axis=0)
        m, b, r_value_diff, p_value_diff, std_err_diff = stats.linregress(x_diff, y_diff)
        print(r_value_diff ** 2)
    if "y1" in axis_keys.keys():
        real_stim = adata[adata.obs[condition_key] == axis_keys["y1"]]
    x = numpy.average(ctrl.X, axis=0)
    y = numpy.average(stim.X, axis=0)
    m, b, r_value, p_value, std_err = stats.linregress(x, y)
    print(r_value ** 2)
    df = pd.DataFrame({axis_keys["x"]: x, axis_keys["y"]: y})
    ax = sns.regplot(x=axis_keys["x"], y=axis_keys["y"], data=df, scatter_kws={'rasterized': True})
    ax.tick_params(labelsize=fontsize)
    if "range" in kwargs:
        start, stop, step = kwargs.get("range")
        ax.set_xticks(numpy.arange(start, stop, step))
        ax.set_yticks(numpy.arange(start, stop, step))
    # _p1 = pyplot.scatter(x, y, marker=".", label=f"{axis_keys['x']}-{axis_keys['y']}")
    # pyplot.plot(x, m * x + b, "-", color="green")
    ax.set_xlabel(labels["x"], fontsize=fontsize)
    ax.set_ylabel(labels["y"], fontsize=fontsize)
    # if "y1" in axis_keys.keys():
        # y1 = numpy.average(real_stim.X, axis=0)
        # _p2 = pyplot.scatter(x, y1, marker="*", c="red", alpha=.5, label=f"{axis_keys['x']}-{axis_keys['y1']}")
    if gene_list is not None:
        texts = []
        for i in gene_list:
            j = adata.var_names.tolist().index(i)
            x_bar = x[j]
            y_bar = y[j]
            texts.append(pyplot.text(x_bar, y_bar , i, fontsize=11, color ="black"))
            pyplot.plot(x_bar, y_bar, 'o', color="red", markersize=5)
            # if "y1" in axis_keys.keys():
                # y1_bar = y1[j]
                # pyplot.text(x_bar, y1_bar, i, fontsize=11, color="black")
    if gene_list is not None:
        adjust_text(texts, x=x, y=y, arrowprops=dict(arrowstyle="->", color='grey', lw=0.5), force_points=(0.0, 0.0))
    if legend:
        pyplot.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    if title is None:
        pyplot.title(f"", fontsize=fontsize)
    else:
        pyplot.title(title, fontsize=fontsize)
    ax.text(max(x) - max(x) * x_coeff, max(y) - y_coeff * max(y), r'$\mathrm{R^2_{\mathrm{\mathsf{all\ genes}}}}$= ' + f"{r_value ** 2:.2f}", fontsize=kwargs.get("textsize", fontsize))
    if diff_genes is not None:
        ax.text(max(x) - max(x) * x_coeff, max(y) - (y_coeff+0.15) * max(y), r'$\mathrm{R^2_{\mathrm{\mathsf{top\ 100\ DEGs}}}}$= ' + f"{r_value_diff ** 2:.2f}", fontsize=kwargs.get("textsize", fontsize))
    pyplot.savefig(f"{path_to_save}", bbox_inches='tight', dpi=100)
    if show:
        pyplot.show()
    pyplot.close()

def cell_type_analysis(adata_path, datasets, model, cell_type, drug, path_to_save, figure, n_out):
    
    adata =sc.read(adata_path)
    if sparse.issparse(adata.X):
        adata.X = adata.X.A
    
    num = 128#datasets["ood_control_"+ cell_type].genes.size(0)
    idx = np.where(datasets['training'].drugs_names == drug)[0]
    emb_drugs = datasets['training'].drugs[idx][0].view(
                1, -1).repeat(num, 1).clone()
    emb_cts = datasets["ood_control_"+ cell_type].cell_types[0][0].view(
                1, -1).repeat(num, 1).clone()
    
    ood_dataloader = DataLoader(datasets["ood_control_"+ cell_type],
                            batch_size=128,
                            num_workers=0
                            )
    
    model.eval()
    predicted_total = torch.empty((0,10000)).to(model.device)
    for graph, fc_genes, drug, cov in ood_dataloader:
        
        if len(torch.unique(graph.batch)) < 128: # last batch has no 128 examples, so dimension must change
                    emb_drugs = emb_drugs[:len(torch.unique(graph.batch)),:]
                    emb_cts = emb_cts[:len(torch.unique(graph.batch)),:]
        
        predict_inputs = dict(
                    graph_x=graph.x.to(model.device),
                    graph_edge=graph.edge_index.to(model.device),
                    graph_batch=graph.batch.to(model.device),
                    fc_genes=fc_genes.to(model.device),
                    fc_drugs=emb_drugs.to(model.device),
                    fc_covariates=emb_cts.to(model.device)
                )
        
        predict_outputs = model.predict(predict_inputs)
        predicted = predict_outputs['genes_reconstructed'] 
        
        predicted_total = torch.cat((predicted_total, predicted), dim=0)
        
        del predicted, graph, fc_genes, drug, cov
        torch.cuda.empty_cache()
         
    predicted = predicted_total   
    dim = predicted.size(1) // 2
    means = predicted[:, :dim].cpu().detach().numpy()
    
    cell_type_data = adata[adata.obs["cell_type"] == cell_type]
    cell_type_data.uns['log1p']["base"] = None
    if model.flavour == 'plugin':
        pred_adata = anndata.AnnData(means, obs={"condition": ["pred"] * len(means)},
                                    var={"var_names": cell_type_data.var_names})
    else:
        pred_adata = anndata.AnnData(means, obs={"condition": ["pred"] * len(means)},
                                 var={"var_names": list(datasets["training"].genes_index)+list(datasets["training"].fc_genes_names)})
    
    all_adata = cell_type_data.concatenate(pred_adata)
    print(all_adata.obs.groupby(['condition']).size())
    sc.tl.rank_genes_groups(cell_type_data, groupby="condition", n_genes=100, method="wilcoxon")
    diff_genes = cell_type_data.uns["rank_genes_groups"]["names"]["stimulated"]
    reg_mean_plot(all_adata, 
                labels={"x": "pred", "y":"stim"},
                condition_key="condition",
                axis_keys={"x": "pred", "y": "stimulated"},
                gene_list=diff_genes[:5],
                top_100_genes=diff_genes,
                legend=False,
                fontsize=20,
                textsize=14, 
                range=[0, 6, 1],
                x_coeff=0.35,
                show=True,
                path_to_save=os.path.join(path_to_save, f"SupplFig7{figure}_{cell_type}_reg_mean.pdf"))
    
    matplotlib.rc('ytick', labelsize=18)
    matplotlib.rc('xtick', labelsize=18)
    sc.pl.violin(all_adata, keys="ISG15", groupby="condition",
                 save=f"_ISG15.pdf",
                 color = "#ee0ef0",
                 show=True)
    ctrl = cell_type_data[cell_type_data.obs['condition'] == 'control']
    stim = cell_type_data[cell_type_data.obs['condition'] == 'stimulated']
    print(f"Control: {ctrl.shape[0]}")
    print(f"Prediction: {pred_adata.shape[0]}")
    print(f"Stimulated: {stim.shape[0]}")
    os.rename(src=os.path.join(path_to_save, f"violin_ISG15.pdf"), 
              dst=os.path.join(path_to_save, f"SupplFig7{figure}_violin_ISG15_{cell_type}_{n_out}.pdf"))