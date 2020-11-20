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


  # job = TripletTraining(
  #   project="diginetica.config.diginetica_triplet",
  #   recommender_module_class="model.TripletNet",
  #   recommender_extra_params={
  #     "n_factors": 100, 
  #     "use_normalize": True, 
  #     "negative_random": 0.05, 
  #     "dropout": 0.2},
  #   data_frames_preparation_extra_params={
  #     "sample_days": 8, 
  #     "column_stratification": "SessionIDX",
  #     "max_itens_per_session": 20,
  #     "min_itens_interactions": 2,
  #     "max_relative_pos": 16,
  #     "pos_max_deep": 0},
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
  #   epochs=300,
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

  # job = TripletPredTraining(
  #   project="diginetica.config.diginetica_interaction",
  #   data_frames_preparation_extra_params={
  #     "sample_days": 8, 
  #     "column_stratification": "SessionID",
  #     "history_window": 10},
  #   test_split_type= "time",
  #   dataset_split_method="column",    
  #   sample_size_eval=5000,
  #   run_evaluate=True,
  #   path_item_embedding="/media/workspace/triplet_session/output/models/TripletTraining/results/TripletTraining____mars_gym_model_b____29a21f2c1e/item_embeddings.npy",
  #   from_index_mapping="/media/workspace/triplet_session/output/models/TripletTraining/results/TripletTraining____mars_gym_model_b____29a21f2c1e/index_mapping.pkl"
  # )


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
  #     "from_index_mapping": "/media/workspace/triplet_session/output/models/TripletTraining/results/TripletTraining____mars_gym_model_b____a0fb4ce70e/index_mapping.pkl",
  #     "path_item_embedding": "/media/workspace/triplet_session/output/models/TripletTraining/results/TripletTraining____mars_gym_model_b____a0fb4ce70e/item_embeddings.npy", 
  #     "freeze_embedding": False},
  #   data_frames_preparation_extra_params={
  #     "sample_days": 4, 
  #     "history_window": 10, 
  #     "column_stratification": "SessionID"},
  #   test_split_type= "time",
  #   dataset_split_method="column",
  #   metrics=["loss"],
  #   loss_function="bce_logists", 
  #   epochs=2,
  #   run_evaluate=True,
  #   sample_size_eval=5000
  # )

     
  # job = SupervisedModelTraining(
  #   project="globo.config.globo_interaction_with_negative_sample",
  #   recommender_module_class="model.TransformerModel",
  #   recommender_extra_params={
  #     "n_factors": 100, 
  #     "dropout": 0.5,
  #     "hist_size": 10,
  #     "from_index_mapping": "/media/workspace/triplet_session/output/models/TripletTraining/results/TripletTraining____mars_gym_model_b____a0fb4ce70e/index_mapping.pkl",
  #     "path_item_embedding": "/media/workspace/triplet_session/output/models/TripletTraining/results/TripletTraining____mars_gym_model_b____a0fb4ce70e/item_embeddings.npy", 
  #     "freeze_embedding": False},
  #   data_frames_preparation_extra_params={
  #     "sample_days": 4, 
  #     "history_window": 10, 
  #     "column_stratification": "SessionID"},
  #   test_split_type= "time",
  #   dataset_split_method="column",
  #   metrics=["loss"],
  #   epochs=2,
  #   run_evaluate=True,
  #   sample_size_eval=5000
  # # )      
  
  # job = SupervisedModelTraining(
  #   project="diginetica.config.diginetica_rnn",
  #   recommender_module_class="model.MLSASRec",
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
  #   loss_function="ce", 
  #   epochs=2,
  #   run_evaluate=True,
  #   sample_size_eval=5000
  # )


  # job = SupervisedModelTraining(
  #   project="mercado_livre.config.mercado_livre_rnn",
  #   recommender_module_class="model.MLCaser",
  #   recommender_extra_params={
  #     "n_factors": 100, 
  #     "p_L": 30, 
  #     "p_d": 50, 
  #     "p_nh": 16,
  #     "p_nv": 4,
  #     "dropout": 0.2, 
  #     "hist_size": 30,
  #     "from_index_mapping": False,
  #     "path_item_embedding": False, 
  #     "freeze_embedding": False},
  #   data_frames_preparation_extra_params={
  #     "sample_days": 30, 
  #     "history_window": 30, 
  #     "column_stratification": "SessionID",
  #     "filter_only_buy": True},
  #   test_size= 0.1,
  #   val_size= 0.1,      
  #   test_split_type= "random",
  #   dataset_split_method="column",
  #   metrics=["loss"],
  #   loss_function="bce", 
  #   epochs=2,
  #   run_evaluate=True,
  #   sample_size_eval=5000
  # )


     
  # job = SupervisedModelTraining(
  #   project="mercado_livre.config.mercado_livre_transformer",
  #   recommender_module_class="model.MLTransformerModel",
  #   recommender_extra_params={
  #     "n_factors": 100, 
  #     "n_hid": 100,
  #     "n_head": 1,
  #     "n_layers": 1,
  #     "num_filters": 50,
  #     "dropout": 0.2,
  #     "hist_size": 30,
  #     "from_index_mapping": False,
  #     "path_item_embedding": False, 
  #     "freeze_embedding": False},
  #   data_frames_preparation_extra_params={
  #     "sample_days": 30, 
  #     "history_window": 30, 
  #     "column_stratification": "SessionID",
  #     "filter_only_buy": True},
  #   test_size= 0.1,
  #   val_size= 0.1,      
  #   test_split_type= "random",
  #   dataset_split_method="column",
  #   metrics=["loss"],
  #   epochs=2,
  #   batch_size=2,
  #   loss_function="ce",
  #   run_evaluate=True,
  #   sample_size_eval=5000
  # )      
  

     
  job = SupervisedModelTraining(
    project="mercado_livre.config.mercado_livre_narm",
    recommender_module_class="model.MLNARMModel",
    recommender_extra_params={
      "n_factors": 100, 
      "hidden_size": 100,
      "n_layers": 1,
      "dense_size": 19,
      "dropout": 0.5,
      "from_index_mapping": False,
      "path_item_embedding": "/media/workspace/triplet_session/output/mercado_livre/assets/mercadolivre-100d.bin", 
      "freeze_embedding": False},
    data_frames_preparation_extra_params={
      "sample_days": 30, 
      "history_window": 20, 
      "column_stratification": "SessionID",
      "normalize_dense_features": "min_max",
      "min_interactions": 2,
      "filter_only_buy": True},
    test_size= 0.1,
    val_size= 0.1,      
    test_split_type= "random",
    dataset_split_method="column",
    metrics=["loss"],
    epochs=2,
    batch_size=2,
    loss_function="ce",
    run_evaluate=True,
    sample_size_eval=5000
  )      
  

  job.run()
