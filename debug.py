# mars-gym run supervised \
# --project globo.config.globo_interaction_with_negative_sample \
# --recommender-module-class model.SASRec \
# --recommender-extra-params '{
#   "n_factors": 100, 
#   "num_blocks": 2, 
#   "num_heads": 1, 
#   "dropout": 0.5, 
#   "hist_size": 10,
#   "from_index_mapping": false,
#   "path_item_embedding": false, 
#   "freeze_embedding": false}' \
# --data-frames-preparation-extra-params '{
#   "sample_days": 8, 
#   "history_window": 10, 
#   "column_stratification": "SessionID"}' \
# --early-stopping-min-delta 0.0001 \
# --test-split-type time \
# --dataset-split-method column \
# --learning-rate 0.001 \
# --metrics='["loss"]' \
# --generator-workers 10  \
# --batch-size 512 \
# --loss-function bce \
# --epochs 100 \
# --run-evaluate  \
# --sample-size-eval 5000

import luigi
from mars_gym.simulation.training import SupervisedModelTraining
from train import TripletTraining, TripletPredTraining

if __name__ == '__main__':
  # job = SupervisedModelTraining(
  #   project="globo.config.globo_interaction_with_negative_sample",
  #   recommender_module_class="model.SASRec",
  #   recommender_extra_params={
  #     "n_factors": 100, 
  #     "num_blocks": 2, 
  #     "num_heads": 1, 
  #     "dropout": 0.5, 
  #     "hist_size": 10,
  #     "from_index_mapping": False,
  #     "path_item_embedding": False, 
  #     "freeze_embedding": False},
  #   data_frames_preparation_extra_params={
  #     "sample_days": 8, 
  #     "history_window": 10, 
  #     "column_stratification": "SessionID"},
  #   test_split_type= "time",
  #   dataset_split_method="column",
  #   metrics=["loss"],
  #   loss_function="bce_logists", 
  #   epochs=1,
  #   run_evaluate=True,
  #   sample_size_eval=5000
  # )

  # job = SupervisedModelTraining(
  #   project="globo.config.globo_interaction_with_negative_sample",
  #   recommender_module_class="model.Caser",
  #   recommender_extra_params={
  #     "n_factors": 100, 
  #     "p_L": 10, 
  #     "p_d": 50, 
  #     "p_nh": 16,
  #     "p_nv": 4,
  #     "dropout": 0.2, 
  #     "hist_size": 10,
  #     "from_index_mapping": False,
  #     "path_item_embedding": False, 
  #     "freeze_embedding": False},
  #   data_frames_preparation_extra_params={
  #     "sample_days": 8, 
  #     "history_window": 10, 
  #     "column_stratification": "SessionID"},
  #   test_split_type= "time",
  #   dataset_split_method="column",
  #   metrics=["loss"],
  #   loss_function="bce", 
  #   epochs=1,
  #   run_evaluate=True,
  #   sample_size_eval=5000
  # )
  # job.run()



  # PYTHONPATH="."  luigi  \
  # --module train TripletTraining  \
  # --project globo.config.globo_triplet  \
  # --recommender-module-class model.TripletNet  \
  # --recommender-extra-params '{"n_factors": 100, "use_normalize": true, "negative_random": 0.05, "dropout": 0.2}'  \
  # --data-frames-preparation-extra-params '{"sample_days": 4, "column_stratification": "SessionIDX", 
  # "max_itens_per_session": 20, "min_itens_interactions": 2, "max_relative_pos": 2}' \
  # --loss-function-params '{"triplet_loss": "bpr_triplet", "swap": true, "l2_reg": 1e-6, "reduction": "mean", "c": 100}'  \
  # --optimizer-params '{"weight_decay": 1e-5}' \
  # --optimizer adam \
  # --learning-rate 1e-4 \
  # --early-stopping-min-delta 0.0001  \
  # --early-stopping-patience 20  \
  # --test-split-type time  \
  # --dataset-split-method column  \
  # --metrics='["loss","triplet_dist", "triplet_acc"]'  \
  # --save-item-embedding-tsv  \
  # --local-scheduler  \
  # --batch-size 128  \
  # --generator-workers 10  \
  # --epochs 100  \
  # --obs ""

  # job = TripletTraining(
  #   project="globo.config.globo_triplet",
  #   recommender_module_class="model.TripletNet",
  #   recommender_extra_params={
  #     "n_factors": 100, 
  #     "use_normalize": True, 
  #     "negative_random": 0.05, 
  #     "dropout": 0.2},
  #   data_frames_preparation_extra_params={
  #     "sample_days": 4, 
  #     "column_stratification": "SessionIDX",
  #     "max_itens_per_session": 20,
  #     "min_itens_interactions": 2,
  #     "max_relative_pos": 2},
  #   loss_function_params={
  #     "triplet_loss": "bpr_triplet",
  #     "swap": True,
  #     "l2_reg": 1e-6,
  #     "reduction": "mean",
  #     "c": 100
  #   },
  #   optimizer_params={
  #     "weight_decay": 1e-5
  #   },
  #   test_split_type= "time",
  #   dataset_split_method="column",
  #   metrics=["loss","triplet_dist", "triplet_acc"],
  #   epochs=1,
  #   save_item_embedding_tsv=True,
  #   sample_size_eval=5000
  # )

  # PYTHONPATH="."  luigi  \
  # --module train CoOccurrenceTraining  \
  # --project globo.config.globo_interaction \
  # --local-scheduler  \
  # --data-frames-preparation-extra-params '{"sample_days": 4, "history_window": 10, "column_stratification": "SessionID"}' \
  # --test-split-type time \
  # --dataset-split-method column \
  # --run-evaluate  \
  # --sample-size-eval 5000

  job = TripletPredTraining(
    project="globo.config.globo_interaction",
    data_frames_preparation_extra_params={
      "sample_days": 4, 
      "column_stratification": "SessionID",
      "history_window": 10},
    test_split_type= "time",
    dataset_split_method="column",    
    sample_size_eval=5000,
    path_item_embedding="/media/workspace/triplet_session/output/models/TripletTraining/results/TripletTraining____mars_gym_model_b____e18c3fa0be/item_embeddings.npy",
    from_index_mapping="/media/workspace/triplet_session/output/models/TripletTraining/results/TripletTraining____mars_gym_model_b____e18c3fa0be/index_mapping.pkl"
  )


  # mars-gym run supervised \
  # --project globo.config.globo_interaction_with_negative_sample \
  # --recommender-module-class model.SASRec \
  # --recommender-extra-params '{
  #   "n_factors": 100, 
  #   "num_blocks": 2, 
  #   "num_heads": 1, 
  #   "dropout": 0.5, 
  #   "hist_size": 10,
  #   "from_index_mapping": false,
  #   "path_item_embedding": false, 
  #   "freeze_embedding": false}' \
  # --data-frames-preparation-extra-params '{
  #   "sample_days": 4, 
  #   "history_window": 10, 
  #   "column_stratification": "SessionID"}' \
  # --early-stopping-min-delta 0.0001 \
  # --test-split-type time \
  # --dataset-split-method column \
  # --learning-rate 0.001 \
  # --metrics='["loss"]' \
  # --generator-workers 10  \
  # --batch-size 128 \
  # --loss-function bce \
  # --epochs 100 \
  # --run-evaluate  \
    

  # job = SupervisedModelTraining(
  #   project="globo.config.globo_interaction_with_negative_sample",
  #   recommender_module_class="model.SASRec",
  #   recommender_extra_params={
  #     "n_factors": 100, 
  #     "num_blocks": 2, 
  #     "num_heads": 1, 
  #     "dropout": 0.5,
  #     "hist_size": 10,
  #     "from_index_mapping": False,
  #     "path_item_embedding": False, 
  #     "freeze_embedding": False},
  #   data_frames_preparation_extra_params={
  #     "sample_days": 4, 
  #     "history_window": 10, 
  #     "column_stratification": "SessionID"},
  #   test_split_type= "time",
  #   dataset_split_method="column",
  #   metrics=["loss"],
  #   loss_function="bce", 
  #   epochs=2,
  #   run_evaluate=True,
  #   sample_size_eval=5000
  # )
      
  job.run()  