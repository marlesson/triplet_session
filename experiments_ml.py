import luigi
from mars_gym.simulation.training import SupervisedModelTraining
from train import TripletTraining, TripletPredTraining
import numpy as np

# PYTHONPATH="."  luigi  --module experiments_ml ExperimentRuns --local-scheduler
class NARMModelExperimentRuns(luigi.WrapperTask):
  '''
  https://luigi.readthedocs.io/en/stable/luigi_patterns.html
  '''
  
  seed: int = luigi.IntParameter(default=42)
  
  evaluate: bool = luigi.BoolParameter(default=False)
  
  experiments: int = luigi.IntParameter(default=1)

  def requires(self):
    random_state = np.random.RandomState(self.seed)

    tasks = []

    _hidden_size  = [10, 50, 100, 200]

    _n_layers  = [1, 2, 4]

    _hist_size = [5, 10, 20, 30]

    _weight_decay = [0, 1e-5, 1e-6, 1e-4]

    _dropout   = [0, 0.2, 0.4, 0.6]

    n_factors = 100

    obs       = ""

    for i in range(self.experiments): 
      hidden_size   = int(random_state.choice(_hidden_size))
      n_layers      = int(random_state.choice(_n_layers))
      dropout       = float(random_state.choice(_dropout))
      weight_decay  = float(random_state.choice(_weight_decay))
      hist_size      = int(random_state.choice(_hist_size))


      job = SupervisedModelTraining(
            project="mercado_livre.config.mercado_livre_rnn",
            recommender_module_class="model.NARMModel",
            recommender_extra_params={
              "n_factors": n_factors, 
              "hidden_size": hidden_size,
              "n_layers": n_layers,
              "dropout": dropout,
              "from_index_mapping": False,
              "path_item_embedding": False, 
              "freeze_embedding": False},
            data_frames_preparation_extra_params={
              "sample_days": 30, 
              "history_window": hist_size, 
              "column_stratification": "SessionID"},
            optimizer_params={
              "weight_decay": weight_decay
            },
            test_size=0.1,
            val_size=0.1,
            test_split_type= "random",
            dataset_split_method="column",
            learning_rate=0.001,
            metrics=["loss"],
            batch_size=512,
            loss_function="ce",
            epochs=100,
            run_evaluate=True,
            sample_size_eval=2000
          )      

      yield job

class MLTransformerModelExperimentRuns(luigi.WrapperTask):
  '''
  https://luigi.readthedocs.io/en/stable/luigi_patterns.html
  '''
  
  seed: int = luigi.IntParameter(default=42)
  
  evaluate: bool = luigi.BoolParameter(default=False)
  
  experiments: int = luigi.IntParameter(default=1)

  def requires(self):
    random_state = np.random.RandomState(self.seed)

    tasks = []

    _n_hid  = [10, 50, 100, 200]

    _n_layers  = [1, 2, 4]

    _n_head = [1, 2, 4]

    _num_filters = [10, 50, 100, 200]

    _hist_size = [5, 10, 20, 30]

    _weight_decay = [0, 1e-5, 1e-6, 1e-4]

    _dropout   = [0, 0.2, 0.4, 0.6]

    n_factors = 100

    obs       = ""

    for i in range(self.experiments): 
      n_hid   = int(random_state.choice(_n_hid))
      n_layers      = int(random_state.choice(_n_layers))
      n_head      = int(random_state.choice(_n_head))
      num_filters      = int(random_state.choice(_num_filters))
      hist_size      = int(random_state.choice(_hist_size))

      dropout       = float(random_state.choice(_dropout))
      weight_decay  = float(random_state.choice(_weight_decay))
      
      job = SupervisedModelTraining(
            project="mercado_livre.config.mercado_livre_transformer",
            recommender_module_class="model.MLTransformerModel",
            recommender_extra_params={
              "n_factors": n_factors, 
              "n_hid": n_hid,
              "n_head": n_head,
              "n_layers": n_layers,
              "num_filters": num_filters,
              "dropout": dropout,
              "hist_size": hist_size,
              "from_index_mapping": False,
              "path_item_embedding": False, 
              "freeze_embedding": False},
            data_frames_preparation_extra_params={
              "sample_days": 30, 
              "history_window": hist_size, 
              "column_stratification": "SessionID",
              "filter_only_buy": True},
            optimizer_params={
              "weight_decay": weight_decay
            },
            test_size=0.1,
            val_size=0.1,
            test_split_type= "random",
            dataset_split_method="column",
            learning_rate=0.001,
            metrics=["loss"],
            batch_size=512,
            loss_function="ce",
            epochs=1000,
            run_evaluate=True,
            run_evaluate_extra_params=" ",
            sample_size_eval=2000
          )      

      yield job


if __name__ == '__main__':
  print("..........")
  MLTransformerModelExperimentRuns().run()  
  NARMModelExperimentRuns().run()
