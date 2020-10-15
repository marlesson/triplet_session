########################################
# Triplet
#########################################

# TripletTraining____mars_gym_model_b____70594e2606

PYTHONPATH="."  luigi  \
--module train TripletTraining  \
--project globo.config.globo_triplet  \
--recommender-module-class model.TripletNet  \
--recommender-extra-params '{"n_factors": 100, "use_normalize": true, "negative_random": 0.05, "dropout": 0.2}'  \
--data-frames-preparation-extra-params '{"sample_days": 8, "column_stratification": "SessionID", "max_itens_per_session": 15, 
  "min_itens_interactions": 3, "max_relative_pos": 5}' \
--loss-function-params '{"triplet_loss": "bpr_triplet",  "swap": true, "l2_reg": 1e-6, "reduction": "mean", "c": 100}'  \
--optimizer-params '{"weight_decay": 1e-5}' \
--optimizer adam \
--learning-rate 1e-3  \
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



########################################
# Random
#########################################

PYTHONPATH="."  luigi  \
--module train RandomTraining  \
--project globo.config.globo_interaction \
--local-scheduler  \
--data-frames-preparation-extra-params '{"sample_days": 8, "history_window": 10, "column_stratification": "SessionID"}' \
--load-index-mapping-path '/media/workspace/triplet_session/output/models/TripletTraining/results/TripletTraining____mars_gym_model_b____70594e2606' \
--test-split-type time \
--dataset-split-method column \
--run-evaluate  \
--sample-size-eval 1000


########################################
# Most Popular
#########################################

PYTHONPATH="."  luigi  \
--module train MostPopularTraining  \
--project globo.config.globo_interaction \
--local-scheduler  \
--data-frames-preparation-extra-params '{"sample_days": 8, "history_window": 10, "column_stratification": "SessionID"}' \
--load-index-mapping-path '/media/workspace/triplet_session/output/models/TripletTraining/results/TripletTraining____mars_gym_model_b____70594e2606' \
--test-split-type time \
--dataset-split-method column \
--run-evaluate  \
--sample-size-eval 1000


########################################
# Co-ocurrence
#########################################

PYTHONPATH="."  luigi  \
--module train CoOccurrenceTraining  \
--project globo.config.globo_interaction \
--local-scheduler  \
--data-frames-preparation-extra-params '{"sample_days": 8, "history_window": 10, "column_stratification": "SessionID"}' \
--load-index-mapping-path '/media/workspace/triplet_session/output/models/TripletTraining/results/TripletTraining____mars_gym_model_b____70594e2606' \
--test-split-type time \
--dataset-split-method column \
--run-evaluate  \
--sample-size-eval 1000


########################################
# KNN
#########################################

PYTHONPATH="."  luigi  \
--module train IKNNTraining  \
--project globo.config.globo_interaction \
--local-scheduler  \
--data-frames-preparation-extra-params '{"sample_days": 8, "history_window": 10, "column_stratification": "SessionID"}' \
--load-index-mapping-path '/media/workspace/triplet_session/output/models/TripletTraining/results/TripletTraining____mars_gym_model_b____70594e2606' \
--test-split-type time \
--dataset-split-method column \
--run-evaluate  \
--sample-size-eval 1000


########################################
# Dot-Product Sim
#########################################

mars-gym run supervised \
--project globo.config.globo_interaction_with_negative_sample \
--recommender-module-class model.DotModel \
--recommender-extra-params '{"n_factors": 100, "dropout": 0.1, "hist_size": 10, "path_item_embedding": "/media/workspace/triplet_session/output/models/TripletTraining/results/TripletTraining____mars_gym_model_b____70594e2606/item_embeddings.npy", "freeze_embedding": false}' \
--data-frames-preparation-extra-params '{"sample_days": 8, "history_window": 10, "column_stratification": "SessionID"}' \
--load-index-mapping-path '/media/workspace/triplet_session/output/models/TripletTraining/results/TripletTraining____mars_gym_model_b____70594e2606' \
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
--sample-size-eval 1000 

# {'count': 341,
#  'coverage_at_20': 0.4802,
#  'coverage_at_5': 0.13019999999999998,
#  'mean_average_precision': 0.436767529588564,
#  'model_task': 'SupervisedModelTraining____mars_gym_model_b____b680d8d4d1',
#  'mrr_at_10': 0.4172159381836801,
#  'mrr_at_5': 0.4105816226783969,
#  'ndcg_at_20': 0.54013286392734,
#  'ndcg_at_5': 0.4873838095432938,
#  'precision_at_1': 0.33724340175953077}




# ########################################
# # MF-BPR
# #########################################

# TripletTraining____mars_gym_model_b____70594e2606
mars-gym run supervised \
--project globo.config.globo_mf_bpr \
--recommender-module-class model.MatrixFactorizationModel \
--recommender-extra-params '{"n_factors": 100, "dropout": 0.2, "hist_size": 10, "path_item_embedding": false, "freeze_embedding": false, "weight_decay": 1e-3}' \
--data-frames-preparation-extra-params '{"sample_days": 8, "history_window": 10, "column_stratification": "SessionID"}' \
--load-index-mapping-path '/media/workspace/triplet_session/output/models/TripletTraining/results/TripletTraining____mars_gym_model_b____70594e2606' \
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
--sample-size-eval 1000 

mars-gym run supervised \
--project globo.config.globo_mf_bpr \
--recommender-module-class model.MatrixFactorizationModel \
--recommender-extra-params '{"n_factors": 100, "dropout": 0.2, "hist_size": 10, "path_item_embedding": "/media/workspace/triplet_session/output/models/TripletTraining/results/TripletTraining____mars_gym_model_b____70594e2606/item_embeddings.npy", "freeze_embedding": false, "weight_decay": 1e-3}' \
--data-frames-preparation-extra-params '{"sample_days": 8, "history_window": 10, "column_stratification": "SessionID"}' \
--load-index-mapping-path '/media/workspace/triplet_session/output/models/TripletTraining/results/TripletTraining____mars_gym_model_b____70594e2606' \
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
--sample-size-eval 1000 

mars-gym run supervised \
--project globo.config.globo_mf_bpr \
--recommender-module-class model.MatrixFactorizationModel \
--recommender-extra-params '{"n_factors": 100, "dropout": 0.2, "hist_size": 10, "path_item_embedding": "/media/workspace/triplet_session/output/models/TripletTraining/results/TripletTraining____mars_gym_model_b____70594e2606/item_embeddings.npy", "freeze_embedding": true, "weight_decay": 1e-3}' \
--data-frames-preparation-extra-params '{"sample_days": 8, "history_window": 10, "column_stratification": "SessionID"}' \
--load-index-mapping-path '/media/workspace/triplet_session/output/models/TripletTraining/results/TripletTraining____mars_gym_model_b____70594e2606' \
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
--sample-size-eval 1000 

mars-gym run supervised \
--project globo.config.globo_mf_bpr \
--recommender-module-class model.MatrixFactorizationModel \
--recommender-extra-params '{"n_factors": 100, "dropout": 0.2, "hist_size": 10, "path_item_embedding": "/media/workspace/triplet_session/output/models/TripletTraining/results/TripletTraining____mars_gym_model_b____70594e2606/item_embeddings.npy", "freeze_embedding": false, "weight_decay": 1e-3}' \
--data-frames-preparation-extra-params '{"sample_days": 8, "history_window": 10, "column_stratification": "SessionID"}' \
--load-index-mapping-path '/media/workspace/triplet_session/output/models/TripletTraining/results/TripletTraining____mars_gym_model_b____70594e2606' \
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
--sample-size-eval 1000 

# mars-gym run supervised \
# --project globo.config.globo_mf_bpr \
# --recommender-module-class model.MatrixFactorizationModel \
# --recommender-extra-params '{"n_factors": 100, "dropout": 0.2, "hist_size": 10, "path_item_embedding": false, "freeze_embedding": false, "weight_decay": 1e-3}' \
# --data-frames-preparation-extra-params '{"sample_days": 8, "history_window": 10, "column_stratification": "SessionID"}' \
# --early-stopping-min-delta 0.0001 \
# --test-split-type time \
# --dataset-split-method column \
# --learning-rate 0.001 \
# --optimizer-params '{"weight_decay": 1e-5}' \
# --metrics='["loss"]' \
# --generator-workers 10  \
# --batch-size 512 \
# --epochs 100 \
# --loss-function dummy \
# --obs "" \
# --run-evaluate  \
# --sample-size-eval 1000 

# # {'count': 624,
# #  'coverage_at_20': 0.3751,
# #  'coverage_at_5': 0.1231,
# #  'mean_average_precision': 0.27400054289786596,
# #  'model_task': 'SupervisedModelTraining____mars_gym_model_b____4395c4261a',
# #  'mrr_at_10': 0.2583759411884412,
# #  'mrr_at_5': 0.25072115384615384,
# #  'ndcg_at_20': 0.33299310379843633,
# #  'ndcg_at_5': 0.29772432500608675,
# #  'precision_at_1': 0.20512820512820512}


# ########################################
# # NARMModel
# #########################################


mars-gym run supervised \
--project globo.config.globo_rnn \
--recommender-module-class model.NARMModel \
--recommender-extra-params '{"n_factors": 100, "hidden_size": 100, "n_layers": 1, "dropout": 0.25, "path_item_embedding": false, "freeze_embedding": false}' \
--data-frames-preparation-extra-params '{"sample_days": 8, "history_window": 10, "column_stratification": "SessionID"}' \
--load-index-mapping-path '/media/workspace/triplet_session/output/models/TripletTraining/results/TripletTraining____mars_gym_model_b____70594e2606' \
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
--sample-size-eval 1000 

mars-gym run supervised \
--project globo.config.globo_rnn \
--recommender-module-class model.NARMModel \
--recommender-extra-params '{"n_factors": 100, "hidden_size": 100, "n_layers": 1, "dropout": 0.25, "path_item_embedding": "/media/workspace/triplet_session/output/models/TripletTraining/results/TripletTraining____mars_gym_model_b____70594e2606/item_embeddings.npy", "freeze_embedding": false}' \
--data-frames-preparation-extra-params '{"sample_days": 8, "history_window": 10, "column_stratification": "SessionID"}' \
--load-index-mapping-path '/media/workspace/triplet_session/output/models/TripletTraining/results/TripletTraining____mars_gym_model_b____70594e2606' \
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
--sample-size-eval 1000 

# # {'count': 624,
# #  'coverage_at_20': 0.6014,
# #  'coverage_at_5': 0.2145,
# #  'mean_average_precision': 0.2780381041028684,
# #  'model_task': 'SupervisedModelTraining____mars_gym_model_b____7ba204d941',
# #  'mrr_at_10': 0.2614494301994302,
# #  'mrr_at_5': 0.25006677350427353,
# #  'ndcg_at_20': 0.35253372780410885,
# #  'ndcg_at_5': 0.3015511655987205,
# #  'precision_at_1': 0.20032051282051283}

# ########################################
# # GRURecModel
# #########################################


# mars-gym run supervised \
# --project globo.config.globo_rnn \
# --recommender-module-class model.GRURecModel \
# --recommender-extra-params '{"n_factors": 100, "hidden_size": 100, "n_layers": 1, "path_item_embedding": false, "freeze_embedding": false, "dropout": 0.2}' \
# --data-frames-preparation-extra-params '{"sample_days": 8, "history_window": 10, "column_stratification": "SessionID"}' \
# --early-stopping-min-delta 0.0001 \
# --test-split-type time \
# --dataset-split-method column \
# --learning-rate 0.001 \
# --metrics='["loss"]' \
# --generator-workers 10  \
# --batch-size 512 \
# --loss-function ce \
# --epochs 100 \
# --obs "dropout" \
# --run-evaluate  \
# --sample-size-eval 1000 


# # {'count': 624,
# #  'coverage_at_20': 0.5761,
# #  'coverage_at_5': 0.2052,
# #  'mean_average_precision': 0.21551375556592595,
# #  'model_task': 'SupervisedModelTraining____mars_gym_model_b____96f3c83609',
# #  'mrr_at_10': 0.1970053673178673,
# #  'mrr_at_5': 0.1924145299145299,
# #  'ndcg_at_20': 0.25740852196261316,
# #  'ndcg_at_5': 0.22571755201364452,
# #  'precision_at_1': 0.16185897435897437}

