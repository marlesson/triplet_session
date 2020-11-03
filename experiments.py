import luigi
from mars_gym.simulation.training import SupervisedModelTraining
from train import TripletTraining, TripletPredTraining
import numpy as np

# PYTHONPATH="."  luigi  --module experiments ExperimentRuns --local-scheduler
class ExperimentRuns(luigi.WrapperTask):
  '''
  https://luigi.readthedocs.io/en/stable/luigi_patterns.html
  '''
  sample_days = luigi.IntParameter(default=8)
  
  seed: int = luigi.IntParameter(default=42)
  
  evaluate: bool = luigi.BoolParameter(default=False)
  
  evaluate: int = luigi.BoolParameter(default=False)
  
  experiments: int = luigi.IntParameter(default=1)

  project: str = luigi.Parameter(default="diginetica.config.diginetica")
#diginetica.config.diginetica_interaction'
  def requires(self):
    random_state = np.random.RandomState(self.seed)

    tasks = []

    project = self.project 
    
    sample_days = self.sample_days

    # Sesssion Truncate Size | ++
    _max_itens_per_session = [5, 10, 20, 30]
    
    # Minimum Interaction between Item A and Item B | +-
    _min_itens_interactions = [1, 2, 4, 8]
    
    # Dist máxima between Item A and Item B | ++
    _max_relative_pos = [1, 2, 4, 8, 16, 24]
    
    # Deep of Positive interaction for Item A | 
    _pos_max_deep = [0, 1, 2, 3]

    # Filter only one interaction por position
    _filter_first_interaction = [False]

    _negative_random = [0, 0.05, 0.1, 0.2]

    _l2_reg = [0, 1e-5, 1e-6, 1e-4]

    _weight_decay = [0, 1e-5, 1e-6, 1e-4]

    _dropout   = [0, 0.2, 0.4]

    _c_bias    = [0, 100]

    n_factors = 100
    
    epochs    = 300

    learning_rate = 1e-4

    obs       = ""

    for i in range(self.experiments): 
      pos_max_deep      = int(random_state.choice(_pos_max_deep))
      max_relative_pos  = int(random_state.choice(_max_relative_pos))
      filter_first_interaction =  bool(random_state.choice(_filter_first_interaction))
      min_itens_interactions = int(random_state.choice(_min_itens_interactions))
      max_itens_per_session = int(random_state.choice(_max_itens_per_session))
      negative_random = float(random_state.choice(_negative_random))

      l2_reg = float(random_state.choice(_l2_reg))
      weight_decay = float(random_state.choice(_weight_decay))
      dropout = float(random_state.choice(_dropout))

      c_bias = int(random_state.choice(_c_bias))

      #print(pos_max_deep, max_relative_pos, filter_first_interaction)
      
      job_triplet = TripletTraining(
        project=project+"_triplet",
        recommender_module_class="model.TripletNet",
        recommender_extra_params={
          "n_factors": n_factors, 
          "use_normalize": True, 
          "negative_random": negative_random, 
          "dropout": dropout},
        data_frames_preparation_extra_params={
          "sample_days": sample_days, 
          "column_stratification": "SessionIDX",
          "max_itens_per_session": max_itens_per_session,
          "min_itens_interactions": min_itens_interactions,
          "max_relative_pos": max_relative_pos,
          "pos_max_deep": pos_max_deep,
          "filter_first_interaction": filter_first_interaction},
        loss_function_params={
          "triplet_loss": "bpr_triplet",
          "swap": True,
          "l2_reg": l2_reg,
          "reduction": "mean",
          "c": c_bias
        },
        optimizer_params={
          "weight_decay": weight_decay
        },
        learning_rate=learning_rate,
        early_stopping_min_delta=0.0001,
        early_stopping_patience=20,        
        test_split_type= "time",
        dataset_split_method="column",
        metrics=["loss","triplet_dist", "triplet_acc"],
        save_item_embedding_tsv=True,
        batch_size=128,
        generator_workers=10,
        epochs=epochs,
        observation=obs
      )

      job_triplet_pre = TripletPredTraining(
        project=self.project+"_interaction",
        data_frames_preparation_extra_params={
          "sample_days": sample_days, 
          "history_window": 10,
          "column_stratification": "SessionID"},
        path_item_embedding="/media/workspace/triplet_session/output/models/TripletTraining/results/{}/item_embeddings.npy".format(job_triplet.task_id),
        from_index_mapping="/media/workspace/triplet_session/output/models/TripletTraining/results/{}/index_mapping.pkl".format(job_triplet.task_id),         
        test_split_type= "time",
        dataset_split_method="column",
        run_evaluate=True,
        sample_size_eval=5000,
        observation=obs
        )    

      if self.evaluate:
        yield job_triplet_pre
      else:
        yield job_triplet


if __name__ == '__main__':
  print("..........")
  ExperimentRuns().run()