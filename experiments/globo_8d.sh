

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
--sample-size-eval 1000

# {'count': 624,
#  'coverage_at_20': 0.8301999999999999,
#  'coverage_at_5': 0.36090000000000005,
#  'mean_average_precision': 0.05025802965049458,
#  'model_task': 'RandomTraining____mars_gym_model_b____51cf88caa7',
#  'mrr_at_10': 0.02749287749287749,
#  'mrr_at_5': 0.020219017094017095,
#  'ndcg_at_20': 0.0708347774846018,
#  'ndcg_at_5': 0.028254135539337463,
#  'precision_at_1': 0.011217948717948718}



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
--sample-size-eval 1000

# {'count': 624,
#  'coverage_at_20': 0.2575,
#  'coverage_at_5': 0.0827,
#  'mean_average_precision': 0.1547536900530248,
#  'model_task': 'MostPopularTraining____mars_gym_model_b____1663bf504c',
#  'mrr_at_10': 0.1393842846967847,
#  'mrr_at_5': 0.13243856837606838,
#  'ndcg_at_20': 0.21428037854953688,
#  'ndcg_at_5': 0.18158104842540837,
#  'precision_at_1': 0.08333333333333333}



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
--sample-size-eval 1000

# {'count': 624,
#  'coverage_at_20': 0.8273,
#  'coverage_at_5': 0.33640000000000003,
#  'mean_average_precision': 0.1918626948292674,
#  'model_task': 'CoOccurrenceTraining____mars_gym_model_b____1663bf504c',
#  'mrr_at_10': 0.17233605514855513,
#  'mrr_at_5': 0.16615918803418803,
#  'ndcg_at_20': 0.22722668143787517,
#  'ndcg_at_5': 0.1914110754823819,
#  'precision_at_1': 0.14262820512820512}


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
--sample-size-eval 1000


# {'count': 624,
#  'coverage_at_20': 0.8228,
#  'coverage_at_5': 0.3325,
#  'mean_average_precision': 0.21066209425153942,
#  'model_task': 'IKNNTraining____mars_gym_model_b____1663bf504c',
#  'mrr_at_10': 0.19211627492877495,
#  'mrr_at_5': 0.18453525641025642,
#  'ndcg_at_20': 0.2577381496794921,
#  'ndcg_at_5': 0.21782864878276087,
#  'precision_at_1': 0.15224358974358973}


########################################
# MF-BPR
#########################################


mars-gym run supervised \
--project globo.config.globo_mf_bpr \
--recommender-module-class model.MatrixFactorizationModel \
--recommender-extra-params '{"n_factors": 100, "dropout": 0.2, "hist_size": 10, "path_item_embedding": false, "freeze_embedding": false, "weight_decay": 1e-3}' \
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
--sample-size-eval 1000 

# {'count': 624,
#  'coverage_at_20': 0.3751,
#  'coverage_at_5': 0.1231,
#  'mean_average_precision': 0.27400054289786596,
#  'model_task': 'SupervisedModelTraining____mars_gym_model_b____4395c4261a',
#  'mrr_at_10': 0.2583759411884412,
#  'mrr_at_5': 0.25072115384615384,
#  'ndcg_at_20': 0.33299310379843633,
#  'ndcg_at_5': 0.29772432500608675,
#  'precision_at_1': 0.20512820512820512}


########################################
# Dot-Product Sim
#########################################


mars-gym run supervised \
--project globo.config.globo_interaction_with_negative_sample \
--recommender-module-class model.DotModel \
--recommender-extra-params '{"n_factors": 100, "dropout": 0.1, "hist_size": 10, "path_item_embedding": false, "freeze_embedding": false}' \
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
--sample-size-eval 1000 


# {'count': 624,
#  'coverage_at_20': 0.6970000000000001,
#  'coverage_at_5': 0.2639,
#  'mean_average_precision': 0.2732920388803836,
#  'model_task': 'SupervisedModelTraining____mars_gym_model_b____0541bb99ef',
#  'mrr_at_10': 0.25181814713064715,
#  'mrr_at_5': 0.24246794871794874,
#  'ndcg_at_20': 0.35786870846248087,
#  'ndcg_at_5': 0.2952730388784639,
#  'precision_at_1': 0.1907051282051282}


########################################
# NARMModel
#########################################


mars-gym run supervised \
--project globo.config.globo_rnn \
--recommender-module-class model.NARMModel \
--recommender-extra-params '{"n_factors": 100, "hidden_size": 100, "n_layers": 1}' \
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
--sample-size-eval 1000 

# {'count': 624,
#  'coverage_at_20': 0.6014,
#  'coverage_at_5': 0.2145,
#  'mean_average_precision': 0.2780381041028684,
#  'model_task': 'SupervisedModelTraining____mars_gym_model_b____7ba204d941',
#  'mrr_at_10': 0.2614494301994302,
#  'mrr_at_5': 0.25006677350427353,
#  'ndcg_at_20': 0.35253372780410885,
#  'ndcg_at_5': 0.3015511655987205,
#  'precision_at_1': 0.20032051282051283}

########################################
# GRURecModel
#########################################


mars-gym run supervised \
--project globo.config.globo_rnn \
--recommender-module-class model.GRURecModel \
--recommender-extra-params '{"n_factors": 100, "hidden_size": 100, "n_layers": 1, "path_item_embedding": false, "freeze_embedding": false, "dropout": 0.2}' \
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
--sample-size-eval 1000 


# {'count': 624,
#  'coverage_at_20': 0.5761,
#  'coverage_at_5': 0.2052,
#  'mean_average_precision': 0.21551375556592595,
#  'model_task': 'SupervisedModelTraining____mars_gym_model_b____96f3c83609',
#  'mrr_at_10': 0.1970053673178673,
#  'mrr_at_5': 0.1924145299145299,
#  'ndcg_at_20': 0.25740852196261316,
#  'ndcg_at_5': 0.22571755201364452,
#  'precision_at_1': 0.16185897435897437}

