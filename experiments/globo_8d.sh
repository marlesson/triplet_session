

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

# {'count': 3140,
#  'coverage_at_20': 0.9997,
#  'coverage_at_5': 0.8973,
#  'mean_average_precision': 0.04956445079880917,
#  'model_task': 'RandomTraining____mars_gym_model_b____4d25b6c1fa',
#  'mrr_at_10': 0.02727940804772015,
#  'mrr_at_5': 0.02115711252653928,
#  'ndcg_at_20': 0.0716252387046003,
#  'ndcg_at_5': 0.0319673384445724,
#  'precision_at_1': 0.009872611464968152}




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

# {'count': 3140,
#  'coverage_at_20': 0.2908,
#  'coverage_at_5': 0.1069,
#  'mean_average_precision': 0.14607446419585235,
#  'model_task': 'MostPopularTraining____mars_gym_model_b____4d25b6c1fa',
#  'mrr_at_10': 0.129721084824588,
#  'mrr_at_5': 0.12270169851380043,
#  'ndcg_at_20': 0.2017552152544706,
#  'ndcg_at_5': 0.167890932324348,
#  'precision_at_1': 0.07802547770700637}



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

# {'count': 3140,
#  'coverage_at_20': 1.0,
#  'coverage_at_5': 0.8608,
#  'mean_average_precision': 0.17688935767416883,
#  'model_task': 'CoOccurrenceTraining____mars_gym_model_b____4d25b6c1fa',
#  'mrr_at_10': 0.15791982610453947,
#  'mrr_at_5': 0.1517356687898089,
#  'ndcg_at_20': 0.2154645681823592,
#  'ndcg_at_5': 0.17876004329149262,
#  'precision_at_1': 0.1251592356687898}



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


# {'count': 3140,
#  'coverage_at_20': 1.0,
#  'coverage_at_5': 0.8519,
#  'mean_average_precision': 0.20018792684926118,
#  'model_task': 'IKNNTraining____mars_gym_model_b____4d25b6c1fa',
#  'mrr_at_10': 0.18190615205742594,
#  'mrr_at_5': 0.17466029723991508,
#  'ndcg_at_20': 0.2438981891778377,
#  'ndcg_at_5': 0.20551416290190583,
#  'precision_at_1': 0.14426751592356687}


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
--sample-size-eval 5000 


# {'count': 3140,
#  'coverage_at_20': 0.5489999999999999,
#  'coverage_at_5': 0.20420000000000002,
#  'mean_average_precision': 0.24943206609185653,
#  'model_task': 'SupervisedModelTraining____mars_gym_model_b____7abc238ce8',
#  'mrr_at_10': 0.23403447578606815,
#  'mrr_at_5': 0.2249177282377919,
#  'ndcg_at_20': 0.3189276987495823,
#  'ndcg_at_5': 0.2796111615513426,
#  'precision_at_1': 0.17070063694267515}


########################################
# Dot-Product Sim
#########################################


mars-gym run supervised \
--project globo.config.globo_interaction_with_negative_sample \
--recommender-module-class model.DotModel \
--recommender-extra-params '{"n_factors": 100, "dropout": 0.2, "hist_size": 10, "path_item_embedding": false, "freeze_embedding": false}' \
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

# {'count': 3140,
#  'coverage_at_20': 0.878,
#  'coverage_at_5': 0.4551,
#  'mean_average_precision': 0.2536406373848965,
#  'model_task': 'SupervisedModelTraining____mars_gym_model_b____d4544a2777',
#  'mrr_at_10': 0.23262941057527048,
#  'mrr_at_5': 0.2210615711252654,
#  'ndcg_at_20': 0.34558646551752953,
#  'ndcg_at_5': 0.2798350465399603,
#  'precision_at_1': 0.1627388535031847}



########################################
# NARMModel
#########################################


mars-gym run supervised \
--project globo.config.globo_rnn \
--recommender-module-class model.NARMModel \
--recommender-extra-params '{"n_factors": 100, "hidden_size": 100, "n_layers": 1, "dropout": 0.2, "path_item_embedding": false, "freeze_embedding": false}' \
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

# {'count': 3140,
#  'coverage_at_20': 0.8432,
#  'coverage_at_5': 0.40509999999999996,
#  'mean_average_precision': 0.276903398336647,
#  'model_task': 'SupervisedModelTraining____mars_gym_model_b____27db17fa2a',
#  'mrr_at_10': 0.25658041401273884,
#  'mrr_at_5': 0.2446151804670913,
#  'ndcg_at_10': 0.3271655474100198,
#  'ndcg_at_15': 0.3446515015366286,
#  'ndcg_at_20': 0.3636136669130627,
#  'ndcg_at_5': 0.296092413307472,
#  'ndcg_at_50': 0.4024698513142655,
#  'precision_at_1': 0.19394904458598727}

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
--sample-size-eval 5000


# {'count': 3140,
#  'coverage_at_20': 0.3345,
#  'coverage_at_5': 0.11960000000000001,
#  'mean_average_precision': 0.1470935267989046,
#  'model_task': 'SupervisedModelTraining____mars_gym_model_b____ad46089cb3',
#  'mrr_at_10': 0.13068951572136286,
#  'mrr_at_5': 0.12372611464968153,
#  'ndcg_at_10': 0.18726076682925935,
#  'ndcg_at_15': 0.19628621111695826,
#  'ndcg_at_20': 0.20292961265135057,
#  'ndcg_at_5': 0.1694465973940016,
#  'ndcg_at_50': 0.23144895561080633,
#  'precision_at_1': 0.07834394904458598}
