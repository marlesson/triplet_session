########################################
# Triplet
#########################################

# TripletTraining____mars_gym_model_b____e18c3fa0be

PYTHONPATH="."  luigi  \
--module train TripletTraining  \
--project globo.config.globo_triplet  \
--recommender-module-class model.TripletNet  \
--recommender-extra-params '{"n_factors": 100, "use_normalize": true, "negative_random": 0.05, "dropout": 0.2}'  \
--data-frames-preparation-extra-params '{"sample_days": 4, "column_stratification": "SessionIDX", 
"max_itens_per_session": 20, "min_itens_interactions": 2, "max_relative_pos": 2, "filter_first_interaction": true}' \
--loss-function-params '{"triplet_loss": "bpr_triplet", "swap": true, "l2_reg": 1e-6, "reduction": "mean", "c": 100}'  \
--optimizer-params '{"weight_decay": 1e-5}' \
--optimizer adam \
--learning-rate 1e-4 \
--early-stopping-min-delta 0.0001  \
--early-stopping-patience 20  \
--test-split-type time  \
--dataset-split-method column  \
--metrics='["loss","triplet_dist", "triplet_acc"]'  \
--save-item-embedding-tsv  \
--local-scheduler  \
--batch-size 128  \
--generator-workers 10  \
--epochs 100  \
--obs ""


# PYTHONPATH="."  luigi  \
# --module train TripletTraining  \
# --project globo.config.globo_triplet  \
# --recommender-module-class model.TripletNet  \
# --recommender-extra-params '{"n_factors": 100, "use_normalize": true, "negative_random": 0.05, "dropout": 0.2}'  \
# --data-frames-preparation-extra-params '{"sample_days": 4, "column_stratification": "SessionIDX", 
# "max_itens_per_session": 20, "min_itens_interactions": 2, "max_relative_pos": 3}' \
# --loss-function-params '{"triplet_loss": "triplet_margin",  "swap": true, "l2_reg": 1e-6, "reduction": "mean", "c": 100}'  \
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
# --epochs 200  \
# --obs ""


#mars-gym run evaluation --model-task-id SupervisedModelTraining____mars_gym_model_b____f946f98b54 --only-new-interactions --only-exist-items

########################################
# Random
#########################################

PYTHONPATH="."  luigi  \
--module train RandomTraining  \
--project globo.config.globo_interaction \
--local-scheduler  \
--data-frames-preparation-extra-params '{"sample_days": 4, "history_window": 10, "column_stratification": "SessionID"}' \
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
--data-frames-preparation-extra-params '{"sample_days": 4, "history_window": 10, "column_stratification": "SessionID"}' \
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
--data-frames-preparation-extra-params '{"sample_days": 4, "history_window": 10, "column_stratification": "SessionID"}' \
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
--data-frames-preparation-extra-params '{"sample_days": 4, "history_window": 10, "column_stratification": "SessionID"}' \
--test-split-type time \
--dataset-split-method column \
--run-evaluate  \
--sample-size-eval 5000


########################################
# Dot 
#########################################


PYTHONPATH="."  luigi  \
--module train TripletPredTraining  \
--project globo.config.globo_interaction \
--local-scheduler  \
--data-frames-preparation-extra-params '{"sample_days": 4, "history_window": 10, "column_stratification": "SessionID"}' \
--path-item-embedding "/media/workspace/triplet_session/output/models/TripletTraining/results/TripletTraining____mars_gym_model_b____792208ac10/item_embeddings.npy" \
--from-index-mapping "/media/workspace/triplet_session/output/models/TripletTraining/results/TripletTraining____mars_gym_model_b____792208ac10/index_mapping.pkl" \
--test-split-type time \
--dataset-split-method column \
--run-evaluate  \
--sample-size-eval 5000 --obs ""


########################################
# Dot-Product Sim
#########################################
# Map
mars-gym run supervised \
--project globo.config.globo_interaction_with_negative_sample \
--recommender-module-class model.DotModel \
--recommender-extra-params '{
  "n_factors": 100, 
  "dropout": 0.1, 
  "hist_size": 10, 
  "from_index_mapping": false,
  "path_item_embedding": false, 
  "freeze_embedding": false}' \
--data-frames-preparation-extra-params '{"sample_days": 4, "history_window": 10, "column_stratification": "SessionID"}' \
--early-stopping-min-delta 0.0001 \
--negative-proportion 0.8 \
--test-split-type time \
--dataset-split-method column \
--learning-rate 0.001 \
--metrics='["loss", "acc"]' \
--generator-workers 10  \
--batch-size 128 \
--epochs 100 \
--run-evaluate  \
--sample-size-eval 5000 --obs "exp 1"

# trainable
mars-gym run supervised \
--project globo.config.globo_interaction_with_negative_sample \
--recommender-module-class model.DotModel \
--recommender-extra-params '{
  "n_factors": 100, 
  "dropout": 0.1, 
  "hist_size": 10, 
  "from_index_mapping": "/media/workspace/triplet_session/output/models/TripletTraining/results/TripletTraining____mars_gym_model_b____792208ac10/index_mapping.pkl",
  "path_item_embedding": "/media/workspace/triplet_session/output/models/TripletTraining/results/TripletTraining____mars_gym_model_b____792208ac10/item_embeddings.npy", 
  "freeze_embedding": false}' \
--data-frames-preparation-extra-params '{"sample_days": 4, "history_window": 10, "column_stratification": "SessionID"}' \
--early-stopping-min-delta 0.0001 \
--negative-proportion 0.8 \
--test-split-type time \
--dataset-split-method column \
--learning-rate 0.001 \
--metrics='["loss", "acc"]' \
--generator-workers 10  \
--batch-size 128 \
--epochs 100 \
--run-evaluate  \
--sample-size-eval 5000 --obs "exp 2 - pos"


########################################
# Attention Transformer
#########################################
# Map
mars-gym run supervised \
--project globo.config.globo_interaction_with_negative_sample \
--recommender-module-class model.TransformerModel \
--recommender-extra-params '{
  "n_factors": 100, 
  "dropout": 0.1, 
  "hist_size": 10, 
  "from_index_mapping": false,
  "path_item_embedding": false, 
  "freeze_embedding": false}' \
--data-frames-preparation-extra-params '{"sample_days": 4, "history_window": 10, "column_stratification": "SessionID"}' \
--early-stopping-min-delta 0.0001 \
--negative-proportion 0.8 \
--test-split-type time \
--dataset-split-method column \
--learning-rate 0.001 \
--metrics='["loss", "acc"]' \
--generator-workers 10  \
--batch-size 128 \
--epochs 100 \
--run-evaluate  \
--sample-size-eval 5000 --obs "exp 1"

# trainable
mars-gym run supervised \
--project globo.config.globo_interaction_with_negative_sample \
--recommender-module-class model.TransformerModel \
--recommender-extra-params '{
  "n_factors": 100, 
  "dropout": 0.1, 
  "hist_size": 10, 
  "from_index_mapping": "/media/workspace/triplet_session/output/models/TripletTraining/results/TripletTraining____mars_gym_model_b____792208ac10/index_mapping.pkl",
  "path_item_embedding": "/media/workspace/triplet_session/output/models/TripletTraining/results/TripletTraining____mars_gym_model_b____792208ac10/item_embeddings.npy", 
  "freeze_embedding": false}' \
--data-frames-preparation-extra-params '{"sample_days": 4, "history_window": 10, "column_stratification": "SessionID"}' \
--early-stopping-min-delta 0.0001 \
--negative-proportion 0.8 \
--test-split-type time \
--dataset-split-method column \
--learning-rate 0.001 \
--metrics='["loss", "acc"]' \
--generator-workers 10  \
--batch-size 128 \
--epochs 100 \
--run-evaluate  \
--sample-size-eval 5000 --obs "exp 2 - pos"


# ########################################
# # MF-BPR
# #########################################

# Only Map
mars-gym run supervised \
--project globo.config.globo_mf_bpr \
--recommender-module-class model.MatrixFactorizationModel \
--recommender-extra-params '{
  "n_factors": 100, 
  "dropout": 0.2, 
  "hist_size": 10, 
  "weight_decay": 1e-3, 
  "from_index_mapping": false,
  "path_item_embedding": false, 
  "freeze_embedding": false}' \
--data-frames-preparation-extra-params '{"sample_days": 4, "history_window": 10, "column_stratification": "SessionID"}' \
--early-stopping-min-delta 0.0001 \
--test-split-type time \
--dataset-split-method column \
--learning-rate 0.001 \
--optimizer-params '{"weight_decay": 1e-5}' \
--metrics='["loss"]' \
--generator-workers 10  \
--batch-size 128 \
--epochs 100 \
--loss-function dummy \
--obs "" \
--run-evaluate  \
--sample-size-eval 5000 

# Trainable Embs
mars-gym run supervised \
--project globo.config.globo_mf_bpr \
--recommender-module-class model.MatrixFactorizationModel \
--recommender-extra-params '{
  "n_factors": 100, 
  "dropout": 0.2, 
  "hist_size": 10, 
  "weight_decay": 1e-3, 
  "from_index_mapping": "/media/workspace/triplet_session/output/models/TripletTraining/results/TripletTraining____mars_gym_model_b____e18c3fa0be/index_mapping.pkl",
  "path_item_embedding": "/media/workspace/triplet_session/output/models/TripletTraining/results/TripletTraining____mars_gym_model_b____e18c3fa0be/item_embeddings.npy", 
  "freeze_embedding": false}' \
--data-frames-preparation-extra-params '{"sample_days": 4, "history_window": 10, "column_stratification": "SessionID"}' \
--early-stopping-min-delta 0.0001 \
--test-split-type time \
--dataset-split-method column \
--learning-rate 0.001 \
--optimizer-params '{"weight_decay": 1e-5}' \
--metrics='["loss"]' \
--generator-workers 10  \
--batch-size 128 \
--epochs 100 \
--loss-function dummy \
--obs "" \
--run-evaluate  \
--sample-size-eval 5000 --obs "exp 1"

# ########################################
# # NARMModel
# #########################################


mars-gym run supervised \
--project globo.config.globo_rnn \
--recommender-module-class model.NARMModel \
--recommender-extra-params '{
  "n_factors": 100, 
  "hidden_size": 100, 
  "n_layers": 1, 
  "dropout": 0.25, 
  "from_index_mapping": false,
  "path_item_embedding": false, 
  "freeze_embedding": false}' \
--data-frames-preparation-extra-params '{"sample_days": 4, "history_window": 10, "column_stratification": "SessionID"}' \
--early-stopping-min-delta 0.0001 \
--test-split-type time \
--dataset-split-method column \
--learning-rate 0.001 \
--metrics='["loss"]' \
--generator-workers 10  \
--batch-size 128 \
--loss-function ce \
--epochs 100 \
--run-evaluate  \
--sample-size-eval 5000

mars-gym run supervised \
--project globo.config.globo_rnn \
--recommender-module-class model.NARMModel \
--recommender-extra-params '{
  "n_factors": 100, 
  "hidden_size": 100, 
  "n_layers": 1, 
  "dropout": 0.25, 
  "from_index_mapping": "/media/workspace/triplet_session/output/models/TripletTraining/results/TripletTraining____mars_gym_model_b____e18c3fa0be/index_mapping.pkl",
  "path_item_embedding": "/media/workspace/triplet_session/output/models/TripletTraining/results/TripletTraining____mars_gym_model_b____e18c3fa0be/item_embeddings.npy", 
  "freeze_embedding": false
  }' \
--data-frames-preparation-extra-params '{"sample_days": 4, "history_window": 10, "column_stratification": "SessionID"}' \
--early-stopping-min-delta 0.0001 \
--test-split-type time \
--dataset-split-method column \
--learning-rate 0.001 \
--metrics='["loss"]' \
--generator-workers 10  \
--batch-size 128 \
--loss-function ce \
--epochs 100 \
--run-evaluate  \
--sample-size-eval 5000 --obs "exp 1"


########################################
# Caser
#########################################

mars-gym run supervised \
--project globo.config.globo_interaction_with_negative_sample \
--recommender-module-class model.Caser \
--recommender-extra-params '{
  "n_factors": 100, 
  "p_L": 10, 
  "p_d": 50, 
  "p_nh": 16,
  "p_nv": 4,  
  "dropout": 0.1, 
  "hist_size": 10, 
  "from_index_mapping": false,
  "path_item_embedding": false, 
  "freeze_embedding": false}' \
--data-frames-preparation-extra-params '{"sample_days": 4, "history_window": 10, "column_stratification": "SessionID"}' \
--early-stopping-min-delta 0.0001 \
--negative-proportion 0.8 \
--test-split-type time \
--dataset-split-method column \
--learning-rate 0.001 \
--metrics='["loss", "acc"]' \
--generator-workers 10  \
--batch-size 128 \
--epochs 100 \
--run-evaluate  \
--sample-size-eval 5000 

mars-gym run supervised \
--project globo.config.globo_interaction_with_negative_sample \
--recommender-module-class model.Caser \
--recommender-extra-params '{
  "n_factors": 100, 
  "p_L": 10, 
  "p_d": 50, 
  "p_nh": 16,
  "p_nv": 4,  
  "dropout": 0.1, 
  "hist_size": 10, 
  "from_index_mapping": "/media/workspace/triplet_session/output/models/TripletTraining/results/TripletTraining____mars_gym_model_b____e18c3fa0be/index_mapping.pkl",
  "path_item_embedding": "/media/workspace/triplet_session/output/models/TripletTraining/results/TripletTraining____mars_gym_model_b____e18c3fa0be/item_embeddings.npy", 
  "freeze_embedding": false}' \
--data-frames-preparation-extra-params '{"sample_days": 4, "history_window": 10, "column_stratification": "SessionID"}' \
--early-stopping-min-delta 0.0001 \
--negative-proportion 0.8 \
--test-split-type time \
--dataset-split-method column \
--learning-rate 0.001 \
--metrics='["loss", "acc"]' \
--generator-workers 10  \
--batch-size 128 \
--epochs 100 \
--run-evaluate  \
--sample-size-eval 5000 --obs "exp 1"

######################################
# SASRec
######################################

mars-gym run supervised \
--project globo.config.globo_interaction_with_negative_sample \
--recommender-module-class model.SASRec \
--recommender-extra-params '{
  "n_factors": 100, 
  "num_blocks": 2, 
  "num_heads": 1, 
  "dropout": 0.5, 
  "hist_size": 10,
  "from_index_mapping": false,
  "path_item_embedding": false, 
  "freeze_embedding": false}' \
--data-frames-preparation-extra-params '{
  "sample_days": 4, 
  "history_window": 10, 
  "column_stratification": "SessionID"}' \
--early-stopping-min-delta 0.0001 \
--negative-proportion 0.8 \
--test-split-type time \
--dataset-split-method column \
--learning-rate 0.001 \
--metrics='["loss"]' \
--generator-workers 10  \
--batch-size 128 \
--loss-function bce_logists \
--epochs 100 \
--run-evaluate  \
--sample-size-eval 5000

mars-gym run supervised \
--project globo.config.globo_interaction_with_negative_sample \
--recommender-module-class model.SASRec \
--recommender-extra-params '{
  "n_factors": 100, 
  "num_blocks": 2, 
  "num_heads": 1, 
  "dropout": 0.5, 
  "hist_size": 10,
  "from_index_mapping": "/media/workspace/triplet_session/output/models/TripletTraining/results/TripletTraining____mars_gym_model_b____e18c3fa0be/index_mapping.pkl",
  "path_item_embedding": "/media/workspace/triplet_session/output/models/TripletTraining/results/TripletTraining____mars_gym_model_b____e18c3fa0be/item_embeddings.npy", 
  "freeze_embedding": false}' \
--data-frames-preparation-extra-params '{
  "sample_days": 4, 
  "history_window": 10, 
  "column_stratification": "SessionID"}' \
--early-stopping-min-delta 0.0001 \
--negative-proportion 0.8 \
--test-split-type time \
--dataset-split-method column \
--learning-rate 0.001 \
--metrics='["loss"]' \
--generator-workers 10  \
--batch-size 128 \
--loss-function bce_logists \
--epochs 100 \
--run-evaluate  \
--sample-size-eval 5000 --obs "exp 1"

# ########################################
# # GRURecModel
# #########################################


# mars-gym run supervised \
# --project globo.config.globo_rnn \
# --recommender-module-class model.GRURecModel \
# --recommender-extra-params '{"n_factors": 100, "hidden_size": 100, "n_layers": 1, "dropout": 0.2, 
#   "from_index_mapping": false,
#   "path_item_embedding": false, 
#   "freeze_embedding": false}' \
# --data-frames-preparation-extra-params '{"sample_days": 4, "history_window": 10, "column_stratification": "SessionID"}' \
# --early-stopping-min-delta 0.0001 \
# --test-split-type time \
# --dataset-split-method column \
# --learning-rate 0.001 \
# --metrics='["loss"]' \
# --generator-workers 10  \
# --batch-size 128 \
# --loss-function ce \
# --epochs 100 \
# --obs "dropout" \
# --run-evaluate  \
# --sample-size-eval 5000 



# mars-gym run supervised \
# --project globo.config.globo_rnn \
# --recommender-module-class model.GRURecModel \
# --recommender-extra-params '{"n_factors": 100, "hidden_size": 100, "n_layers": 1, "dropout": 0.2, 
#   "from_index_mapping": "/media/workspace/triplet_session/output/models/TripletTraining/results/TripletTraining____mars_gym_model_b____e18c3fa0be/index_mapping.pkl",
#   "path_item_embedding": "/media/workspace/triplet_session/output/models/TripletTraining/results/TripletTraining____mars_gym_model_b____e18c3fa0be/item_embeddings.npy", 
#   "freeze_embedding": false}' \
# --data-frames-preparation-extra-params '{"sample_days": 4, "history_window": 10, "column_stratification": "SessionID"}' \
# --early-stopping-min-delta 0.0001 \
# --test-split-type time \
# --dataset-split-method column \
# --learning-rate 0.001 \
# --metrics='["loss"]' \
# --generator-workers 10  \
# --batch-size 128 \
# --loss-function ce \
# --epochs 100 \
# --obs "dropout" \
# --run-evaluate  \
# --sample-size-eval 5000 --obs "exp 1"


