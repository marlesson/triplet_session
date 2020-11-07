import luigi
from mars_gym.simulation.training import SupervisedModelTraining
from train import TripletTraining, TripletPredTraining
import numpy as np

# PYTHONPATH="."  luigi  --module experiments_ml ExperimentRuns --local-scheduler
class ExperimentRuns(luigi.WrapperTask):
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

    _weight_decay = [0, 1e-5, 1e-6, 1e-4]

    _dropout   = [0, 0.2, 0.4, 0.6]

    n_factors = 100

    obs       = ""

    for i in range(self.experiments): 
      hidden_size   = int(random_state.choice(_hidden_size))
      n_layers      = int(random_state.choice(_n_layers))
      dropout       = float(random_state.choice(_dropout))
      weight_decay  = float(random_state.choice(_weight_decay))
      
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
              "history_window": 10, 
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


if __name__ == '__main__':
  print("..........")
  ExperimentRuns().run()