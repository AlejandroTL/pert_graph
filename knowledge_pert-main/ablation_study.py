from pykeen.ablation import ablation_pipeline
from datetime import datetime
import os

# All parameters
ablation_path = os.path.join('omnipath_triples/', 'ablation/')
metadata = dict(title="Ablation Study.")

models = ["ComplEx", "TransE", "RGCN", "RotatE"]  # models to test
datasets = [
   {
       "training": os.path.join(ablation_path, 'training'),
       "validation": os.path.join(ablation_path, 'validation'),
       "testing": os.path.join(ablation_path, 'testing')
   }
]
losses = ["BCEAfterSigmoidLoss", "softplus", "marginranking"]  # Losses to test
training_loops = ["lcwa"]  # Locally complete KG
optimizers = ["adam"]
create_inverse_triples = [True, False]
stopper = "early"
stopper_kwargs = {
   "frequency": 5,
   "patience": 15,
   "relative_delta": 0.002,
   "metric": "adjusted_mean_rank_index",
}

# Define HPO ranges

model_to_model_kwargs_ranges = {  # Just optimize embedding dim
   "ComplEx": {
       "embedding_dim": {
           "type": "int",
           "low": 4,
           "high": 6,
           "scale": "power_two"
       }
   },
   "TransE": {
       "embedding_dim": {
           "type": "int",
           "low": 4,
           "high": 6,
           "scale": "power_two"
       }
   },
   "RGCN": {
       "embedding_dim": {
           "type": "int",
           "low": 4,
           "high": 6,
           "scale": "power_two"
       }
    },
   "RotatE": {
       "embedding_dim": {
           "type": "int",
           "low": 4,
           "high": 6,
           "scale": "power_two"
       }
   }
}

model_to_training_loop_to_training_kwargs_ranges = {
   "ComplEx": {
       "lcwa": {
           "num_epochs": {
               "type": "int",
               "low": 50,
               "high": 200,
               "step": 50
           },
           "label_smoothing": {
               "type": "float",
               "low": 0.001,
               "high": 1.0,
               "scale": "log"
           },
           "batch_size": {
               "type": "int",
               "low": 7,
               "high": 9,
               "scale": "power_two"
           }
       }
   },
   "TransE": {
       "lcwa": {
           "num_epochs": {
               "type": "int",
               "low": 50,
               "high": 200,
               "step": 50
           },
           "label_smoothing": {
               "type": "float",
               "low": 0.001,
               "high": 1.0,
               "scale": "log"
           },
           "batch_size": {
               "type": "int",
               "low": 7,
               "high": 9,
               "scale": "power_two"
           }
       }
   },
   "RGCN": {
       "lcwa": {
           "num_epochs": {
               "type": "int",
               "low": 50,
               "high": 200,
               "step": 50
           },
           "label_smoothing": {
               "type": "float",
               "low": 0.001,
               "high": 1.0,
               "scale": "log"
           },
           "batch_size": {
               "type": "int",
               "low": 7,
               "high": 9,
               "scale": "power_two"
           }
       }
   },
   "RotatE": {
       "lcwa": {
           "num_epochs": {
               "type": "int",
               "low": 50,
               "high": 200,
               "step": 50
           },
           "label_smoothing": {
               "type": "float",
               "low": 0.001,
               "high": 1.0,
               "scale": "log"
           },
           "batch_size": {
               "type": "int",
               "low": 7,
               "high": 9,
               "scale": "power_two"
           }
       }
   }
}


model_to_optimizer_to_optimizer_kwargs_ranges = {
   "ComplEx": {
       "adam": {
           "lr": {
               "type": "float",
               "low": 0.001,
               "high": 0.1,
               "scale": "log"
           }
       }
   },
   "TransE": {
       "adam": {
           "lr": {
               "type": "float",
               "low": 0.001,
               "high": 0.1,
               "scale": "log"
           }
       }
   },
   "RGCN": {
       "adam": {
           "lr": {
               "type": "float",
               "low": 0.001,
               "high": 0.1,
               "scale": "log"
           }
       }
   },
    "RotatE": {
           "adam": {
               "lr": {
                   "type": "float",
                   "low": 0.001,
                   "high": 0.1,
                   "scale": "log"
               }
           }
       },
}

# Run ablation experiment
if __name__ == '__main__':

    ablation_pipeline(
       models=models,
       datasets=datasets,
       losses=losses,
       training_loops=training_loops,
       optimizers=optimizers,
       model_to_model_kwargs_ranges=model_to_model_kwargs_ranges,
       model_to_optimizer_to_optimizer_kwargs_ranges=model_to_optimizer_to_optimizer_kwargs_ranges,
       directory="ablation_results/ablation_{:%Y_%m_%d}".format(datetime.now()),
       best_replicates=5,
       n_trials=2,
       timeout=300,
       metric="hits@10",
       direction="maximize",
       sampler="random",
       pruner="nop",
    )
