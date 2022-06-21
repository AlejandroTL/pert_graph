import anndata
import numpy as np
import pandas as pd

#from scvi.data import setup_anndata
import cpa
import scanpy as sc

from sklearn.metrics import r2_score
import numpy as np

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def generate_synth():
    n_cells = 2000
    n_genes = 500
    X = np.random.randint(low=0, high=1000, size=(n_cells, n_genes))
    obs = pd.DataFrame(
        dict(
            c0=np.random.randn(n_cells),
            c1=np.random.randn(n_cells),
            drug_name=np.array(["HESPERADIN", "SNS-032", "AQP10"])[np.random.randint(3, size=n_cells)],
            dose_val=np.array([0.1, 0.05, 0.5, 0.25, 0.75])[np.random.randint(5, size=n_cells)],
            covar_1=np.array(["v1", "v2"])[np.random.randint(2, size=n_cells)],
            covar_2=np.random.randint(10, size=n_cells),
            control=np.random.randint(10, size=n_cells),
            split=np.array(["train", "test", "ood"])[np.random.randint(3, size=n_cells)],
        )
    )
    obs.loc[:, "covar_1"] = obs.loc[:, "covar_1"].astype("category")
    obs.loc[:, "covar_2"] = obs.loc[:, "covar_2"].astype("category")
    obs.loc[:, "control"] = obs.loc[:, "control"].astype("category")

    dataset = anndata.AnnData(
        X=X,
        obs=obs,
    )

    cpa.CPA.setup_anndata(
        dataset,
        drug_key="drug_name",
        dose_key='dose_val',
        categorical_covariate_keys=["covar_1", "covar_2"],
        control_key='control'
    )

    return dict(dataset=dataset)


def test_cpa():
    data = generate_synth()
    dataset = data["dataset"]
    graph = dict(
        graph='KnowledgeGraph',
        kge_model='trained_models/KG_models/RotatE-MeanPool.plk',
        kge_data='omnipath_triples/triples',
        kge_drugs='omnipath_triples/drug_names.json',
        option='Concatenation'
     )
    model = cpa.CPA(adata=dataset,
                    n_latent=128,
                    loss_ae='gauss',
                    doser_type='logsigm',
                    split_key='split',
                    )
    model.train(max_epochs=3, plan_kwargs=dict(autoencoder_lr=1e-4))
    model.predict(batch_size=1024)


#test_cpa()

# TEST WITH GSM DATA
sc.settings.set_figure_params(dpi=100)

data_path = '/home/aletl/Downloads/datasets/GSM_new.h5ad'
adata = sc.read(data_path)

cpa.CPA.setup_anndata(adata,
                      drug_key='condition',
                      dose_key='dose_val',
                      categorical_covariate_keys=['cell_type'],
                      control_key='control',
                     )

ae_hparams = {'autoencoder_depth': 4,
              'autoencoder_width': 512,
              'adversary_depth': 3,
              'adversary_width': 256,
              'dosers_depth': 3,
              'dosers_width': 64,
              'use_batch_norm': True,
              'use_layer_norm': False,
              'output_activation': 'linear',
              'dropout_rate': 0.0,
              'variational': False,
              'seed': 60,
              }

trainer_params = {
    'n_epochs_warmup': 0,
    'adversary_lr': 0.0006158304832265454,
    'adversary_wd': 3.546249921082396e-06,
    'adversary_steps': 5,
    'autoencoder_lr': 0.002563090275772759,
    'autoencoder_wd': 2.8299682410882683e-05,
    'dosers_lr': 0.0028643381083830787,
    'dosers_wd': 7.850495446598981e-07,
    'penalty_adversary': 6.20968938643343,
    'reg_adversary': 1.323092865499999,
    'kl_weight': 0.00000,
    'step_size_lr': 45,
}

graph = dict(
    graph='KnowledgeGraph',
    kge_model='trained_models/KG_models/RotatE-MeanPool.plk',
    kge_data='omnipath_triples/triples',
    kge_drugs='omnipath_triples/drug_names.json',
    option='Concatenation'
)

model = cpa.CPA(adata=adata,
                n_latent=256,
                loss_ae='gauss',
                doser_type='logsigm',
                split_key='split',
                **ae_hparams,
               )


model.train(max_epochs=2,
            use_gpu=False,
            batch_size=64,
            early_stopping=True,
            plan_kwargs=trainer_params,
            early_stopping_patience=15,
            check_val_every_n_epoch=20,
            save_path=None,
           )

