########################################
# Triplet
#########################################

# TripletTraining____mars_gym_model_b____6d2fc7db1b

PYTHONPATH="."  luigi  \
--module train TripletTraining  \
--project globo.config.globo_triplet  \
--recommender-module-class model.TripletNet  \
--recommender-extra-params '{"n_factors": 100, "use_normalize": true, "negative_random": 0.05, "dropout": 0.2}'  \
--data-frames-preparation-extra-params '{"sample_days": 8, "column_stratification": "SessionIDX", 
"max_itens_per_session": 15, "min_itens_interactions": 3, "max_relative_pos": 5}' \
--loss-function-params '{"triplet_loss": "bpr_triplet",  "swap": true, "l2_reg": 1e-6, "reduction": "mean", "c": 100}'  \
--optimizer-params '{"weight_decay": 1e-5}' \
--optimizer adam \
--learning-rate 1e-4 \
--early-stopping-min-delta 0.0001  \
--early-stopping-patience 10  \
--test-split-type time  \
--dataset-split-method column  \
--metrics='["loss","triplet_dist", "triplet_acc"]'  \
--save-item-embedding-tsv  \
--local-scheduler  \
--batch-size 512  \
--generator-workers 10  \
--epochs 100  \
--obs ""

#mars-gym run evaluation --model-task-id SupervisedModelTraining____mars_gym_model_b____012b6fbf48 --only-new-interactions --only-exist-items

########################################
# Random
#########################################

PYTHONPATH="."  luigi  \
--module train RandomTraining  \
--project globo.config.globo_interaction \
--local-scheduler  \
--data-frames-preparation-extra-params '{"sample_days": 8, "history_window": 10, "column_stratification": "SessionID"}' \
--test-split-type time \
--dataset-split-method column \
--run-evaluate  \
--sample-size-eval 5000


########################################
# Most Popular
#########################################

PYTHONPATH="."  luigi  \
--module train MostPopularTraining  \
--project globo.config.globo_interaction \
--local-scheduler  \
--data-frames-preparation-extra-params '{"sample_days": 8, "history_window": 10, "column_stratification": "SessionID"}' \
--test-split-type time \
--dataset-split-method column \
--run-evaluate  \
--sample-size-eval 5000


########################################
# Co-ocurrence
#########################################

PYTHONPATH="."  luigi  \
--module train CoOccurrenceTraining  \
--project globo.config.globo_interaction \
--local-scheduler  \
--data-frames-preparation-extra-params '{"sample_days": 8, "history_window": 10, "column_stratification": "SessionID"}' \
--test-split-type time \
--dataset-split-method column \
--run-evaluate  \
--sample-size-eval 5000


########################################
# KNN
#########################################

PYTHONPATH="."  luigi  \
--module train IKNNTraining  \
--project globo.config.globo_interaction \
--local-scheduler  \
--data-frames-preparation-extra-params '{"sample_days": 8, "history_window": 10, "column_stratification": "SessionID"}' \
--test-split-type time \
--dataset-split-method column \
--run-evaluate  \
--sample-size-eval 5000

########################################
# Dot-Product Sim
#########################################
# Map
mars-gym run supervised \
--project globo.config.globo_interaction_with_negative_sample \
--recommender-module-class model.DotModel \
--recommender-extra-params '{"n_factors": 100, "dropout": 0.1, "hist_size": 10, 
  "from_index_mapping": false,
  "path_item_embedding": false, 
  "freeze_embedding": false}' \
--data-frames-preparation-extra-params '{"sample_days": 8, "history_window": 10, "column_stratification": "SessionID"}' \
--early-stopping-min-delta 0.0001 \
--negative-proportion 0.8 \
--test-split-type time \
--dataset-split-method column \
--learning-rate 0.001 \
--metrics='["loss", "acc"]' \
--generator-workers 10  \
--batch-size 512 \
--epochs 100 \
--run-evaluate  \
--sample-size-eval 5000 

# trainable
mars-gym run supervised \
--project globo.config.globo_interaction_with_negative_sample \
--recommender-module-class model.DotModel \
--recommender-extra-params '{"n_factors": 100, "dropout": 0.1, "hist_size": 10, 
  "from_index_mapping": "/media/workspace/triplet_session/output/models/TripletTraining/results/TripletTraining____mars_gym_model_b____6d2fc7db1b/index_mapping.pkl",
  "path_item_embedding": "/media/workspace/triplet_session/output/models/TripletTraining/results/TripletTraining____mars_gym_model_b____6d2fc7db1b/item_embeddings.npy", 
  "freeze_embedding": false}' \
--data-frames-preparation-extra-params '{"sample_days": 8, "history_window": 10, "column_stratification": "SessionID"}' \
--early-stopping-min-delta 0.0001 \
--negative-proportion 0.8 \
--test-split-type time \
--dataset-split-method column \
--learning-rate 0.001 \
--metrics='["loss", "acc"]' \
--generator-workers 10  \
--batch-size 512 \
--epochs 100 \
--run-evaluate  \
--sample-size-eval 5000 --obs "exp 4"


# ########################################
# # MF-BPR
# #########################################

# Only Map
mars-gym run supervised \
--project globo.config.globo_mf_bpr \
--recommender-module-class model.MatrixFactorizationModel \
--recommender-extra-params '{"n_factors": 100, "dropout": 0.2, "hist_size": 10, "weight_decay": 1e-3, 
  "from_index_mapping": false,
  "path_item_embedding": false, 
  "freeze_embedding": false}' \
--data-frames-preparation-extra-params '{"sample_days": 8, "history_window": 10, "column_stratification": "SessionID"}' \
--early-stopping-min-delta 0.0001 \
--test-split-type time \
--dataset-split-method column \
--learning-rate 0.001 \
--optimizer-params '{"weight_decay": 1e-5}' \
--metrics='["loss"]' \
--generator-workers 10  \
--batch-size 512 \
--epochs 100 \
--loss-function dummy \
--obs "" \
--run-evaluate  \
--sample-size-eval 5000 

# Trainable Embs
mars-gym run supervised \
--project globo.config.globo_mf_bpr \
--recommender-module-class model.MatrixFactorizationModel \
--recommender-extra-params '{"n_factors": 100, "dropout": 0.2, "hist_size": 10, "weight_decay": 1e-3, 
  "from_index_mapping": "/media/workspace/triplet_session/output/models/TripletTraining/results/TripletTraining____mars_gym_model_b____6d2fc7db1b/index_mapping.pkl",
  "path_item_embedding": "/media/workspace/triplet_session/output/models/TripletTraining/results/TripletTraining____mars_gym_model_b____6d2fc7db1b/item_embeddings.npy", 
  "freeze_embedding": false}' \
--data-frames-preparation-extra-params '{"sample_days": 8, "history_window": 10, "column_stratification": "SessionID"}' \
--early-stopping-min-delta 0.0001 \
--test-split-type time \
--dataset-split-method column \
--learning-rate 0.001 \
--optimizer-params '{"weight_decay": 1e-5}' \
--metrics='["loss"]' \
--generator-workers 10  \
--batch-size 512 \
--epochs 100 \
--loss-function dummy \
--obs "" \
--run-evaluate  \
--sample-size-eval 5000 --obs "exp 4"

# ########################################
# # NARMModel
# #########################################


mars-gym run supervised \
--project globo.config.globo_rnn \
--recommender-module-class model.NARMModel \
--recommender-extra-params '{"n_factors": 100, "hidden_size": 100, "n_layers": 1, "dropout": 0.25, 
  "from_index_mapping": false,
  "path_item_embedding": false, 
  "freeze_embedding": false}' \
--data-frames-preparation-extra-params '{"sample_days": 8, "history_window": 10, "column_stratification": "SessionID"}' \
--early-stopping-min-delta 0.0001 \
--test-split-type time \
--dataset-split-method column \
--learning-rate 0.001 \
--metrics='["loss"]' \
--generator-workers 10  \
--batch-size 512 \
--loss-function ce \
--epochs 100 \
--run-evaluate  \
--sample-size-eval 5000 

mars-gym run supervised \
--project globo.config.globo_rnn \
--recommender-module-class model.NARMModel \
--recommender-extra-params '{"n_factors": 100, "hidden_size": 100, "n_layers": 1, "dropout": 0.25, 
  "from_index_mapping": "/media/workspace/triplet_session/output/models/TripletTraining/results/TripletTraining____mars_gym_model_b____6d2fc7db1b/index_mapping.pkl",
  "path_item_embedding": "/media/workspace/triplet_session/output/models/TripletTraining/results/TripletTraining____mars_gym_model_b____6d2fc7db1b/item_embeddings.npy", 
  "freeze_embedding": false}' \
--data-frames-preparation-extra-params '{"sample_days": 8, "history_window": 10, "column_stratification": "SessionID"}' \
--early-stopping-min-delta 0.0001 \
--test-split-type time \
--dataset-split-method column \
--learning-rate 0.001 \
--metrics='["loss"]' \
--generator-workers 10  \
--batch-size 512 \
--loss-function ce \
--epochs 100 \
--run-evaluate  \
--sample-size-eval 5000 --obs "exp 4"




# ########################################
# # GRURecModel
# #########################################


mars-gym run supervised \
--project globo.config.globo_rnn \
--recommender-module-class model.GRURecModel \
--recommender-extra-params '{"n_factors": 100, "hidden_size": 100, "n_layers": 1, "dropout": 0.2, 
  "from_index_mapping": false,
  "path_item_embedding": false, 
  "freeze_embedding": false}' \
--data-frames-preparation-extra-params '{"sample_days": 8, "history_window": 10, "column_stratification": "SessionID"}' \
--early-stopping-min-delta 0.0001 \
--test-split-type time \
--dataset-split-method column \
--learning-rate 0.001 \
--metrics='["loss"]' \
--generator-workers 10  \
--batch-size 512 \
--loss-function ce \
--epochs 100 \
--obs "dropout" \
--run-evaluate  \
--sample-size-eval 5000 



mars-gym run supervised \
--project globo.config.globo_rnn \
--recommender-module-class model.GRURecModel \
--recommender-extra-params '{"n_factors": 100, "hidden_size": 100, "n_layers": 1, "dropout": 0.2, 
  "from_index_mapping": "/media/workspace/triplet_session/output/models/TripletTraining/results/TripletTraining____mars_gym_model_b____6d2fc7db1b/index_mapping.pkl",
  "path_item_embedding": "/media/workspace/triplet_session/output/models/TripletTraining/results/TripletTraining____mars_gym_model_b____6d2fc7db1b/item_embeddings.npy", 
  "freeze_embedding": false}' \
--data-frames-preparation-extra-params '{"sample_days": 8, "history_window": 10, "column_stratification": "SessionID"}' \
--early-stopping-min-delta 0.0001 \
--test-split-type time \
--dataset-split-method column \
--learning-rate 0.001 \
--metrics='["loss"]' \
--generator-workers 10  \
--batch-size 512 \
--loss-function ce \
--epochs 100 \
--obs "dropout" \
--run-evaluate  \
--sample-size-eval 5000 --obs "exp 4"


