########################################
# Most Popular
#########################################

PYTHONPATH="."  luigi  \
--module train MostPopularTraining  \
--project mercado_livre.config.mercado_livre_interaction \
--local-scheduler  \
--data-frames-preparation-extra-params '{
  "sample_days": 30, 
  "history_window": 30, 
  "column_stratification": "SessionID",
  "filter_only_buy": true}' \
--test-size 0.1 \
--val-size 0.1 \
--test-split-type random \
--dataset-split-method column \
--run-evaluate  \
--run-evaluate-extra-params " " \
--sample-size-eval 2000 --obs "Most Popular"

# {'count': 1923,
#  'coverage_at_20': 0.22829999999999998,
#  'coverage_at_5': 0.0676,
#  'mean_average_precision': 0.23182236457860583,
#  'model_task': 'MostPopularTraining____mars_gym_model_b____2229d09323',
#  'mrr_at_10': 0.21124152902624044,
#  'mrr_at_5': 0.19507713641879007,
#  'ndcg_at_10': 0.30719757287049576,
#  'ndcg_at_15': 0.33082660452268003,
#  'ndcg_at_20': 0.3484842142844164,
#  'ndcg_at_5': 0.26532690285191335,
#  'ndcg_at_50': 0.3931065800157984,
#  'precision_at_1': 0.12428497139885596}


# ########################################
# # MLNARMModel
# #########################################


mars-gym run supervised \
--project mercado_livre.config.mercado_livre_rnn \
--recommender-module-class model.MLNARMModel \
--recommender-extra-params '{
  "n_factors": 100, 
  "hidden_size": 100, 
  "n_layers": 1, 
  "dropout": 0.5, 
  "from_index_mapping": false,
  "path_item_embedding": false, 
  "freeze_embedding": false}' \
--data-frames-preparation-extra-params '{
  "sample_days": 30, 
  "history_window": 30, 
  "column_stratification": "SessionID",
  "filter_only_buy": true}' \
--test-size 0.1 \
--val-size 0.1 \
--early-stopping-min-delta 0.0001 \
--test-split-type random \
--dataset-split-method column \
--learning-rate 0.001 \
--metrics='["loss"]' \
--generator-workers 10  \
--batch-size 512 \
--loss-function ce \
--epochs 100 \
--run-evaluate  \
--run-evaluate-extra-params " " \
--sample-size-eval 2000 --obs ""

#SupervisedModelTraining____mars_gym_model_b____453fd11ae0

mars-gym evaluate supervised --model-task-id SupervisedModelTraining____mars_gym_model_b____453fd11ae0 

# {'count': 1923,
#  'coverage_at_20': 0.41490000000000005,
#  'coverage_at_5': 0.1361,
#  'mean_average_precision': 0.48195540613550464,
#  'model_task': 'SupervisedModelTraining____mars_gym_model_b____453fd11ae0',
#  'mrr_at_10': 0.46894423395983453,
#  'mrr_at_5': 0.4577049748656613,
#  'ndcg_at_10': 0.5295488768349669,
#  'ndcg_at_15': 0.5436897351847644,
#  'ndcg_at_20': 0.5518291126438414,
#  'ndcg_at_5': 0.5004710066212015,
#  'ndcg_at_50': 0.5849445217401592,
#  'precision_at_1': 0.41445657826313054}




mars-gym run supervised \
--project mercado_livre.config.mercado_livre_rnn \
--recommender-module-class model.MLNARMModel \
--recommender-extra-params '{
  "n_factors": 100, 
  "hidden_size": 100, 
  "n_layers": 1, 
  "dropout": 0.5, 
  "from_index_mapping": false,
  "path_item_embedding": false, 
  "freeze_embedding": false}' \
--data-frames-preparation-extra-params '{
  "sample_days": 30, 
  "history_window": 30, 
  "column_stratification": "SessionID",
  "filter_only_buy": false}' \
--test-size 0.1 \
--val-size 0.1 \
--early-stopping-min-delta 0.0001 \
--test-split-type random \
--dataset-split-method column \
--learning-rate 0.001 \
--metrics='["loss"]' \
--generator-workers 10  \
--batch-size 512 \
--loss-function ce \
--epochs 100 \
--run-evaluate  \
--run-evaluate-extra-params " " \
--sample-size-eval 2000 --obs ""




PYTHONPATH="." luigi --module mercado_livre.evaluation MLEvaluationTask \
--model-task-class "mars_gym.simulation.training.SupervisedModelTraining" \
--model-task-id SupervisedModelTraining____mars_gym_model_b____c549ab3480 \
--batch-size 1000 \
--local-scheduler


######################################
# MLSASRec
######################################

mars-gym run supervised \
--project mercado_livre.config.mercado_livre_rnn \
--recommender-module-class model.MLSASRec \
--recommender-extra-params '{
  "n_factors": 100, 
  "num_blocks": 2, 
  "num_heads": 1, 
  "dropout": 0.5, 
  "hist_size": 30,
  "from_index_mapping": false,
  "path_item_embedding": false, 
  "freeze_embedding": false}' \
--data-frames-preparation-extra-params '{
  "sample_days": 30, 
  "history_window": 30, 
  "column_stratification": "SessionID",
  "filter_only_buy": true}' \
--test-size 0.1 \
--val-size 0.1 \
--early-stopping-min-delta 0.0001 \
--test-split-type random \
--dataset-split-method column \
--learning-rate 0.001 \
--metrics='["loss"]' \
--generator-workers 10  \
--batch-size 512 \
--loss-function ce \
--epochs 100 \
--run-evaluate  \
--run-evaluate-extra-params " " \
--sample-size-eval 2000

# {'count': 1923,
#  'coverage_at_20': 0.3826,
#  'coverage_at_5': 0.1195,
#  'mean_average_precision': 0.36709362226217207,
#  'model_task': 'SupervisedModelTraining____mars_gym_model_b____cc252e8603',
#  'mrr_at_10': 0.35052373523512365,
#  'mrr_at_5': 0.3370168140058935,
#  'ndcg_at_10': 0.42212608363308307,
#  'ndcg_at_15': 0.44019740245708744,
#  'ndcg_at_20': 0.4526532388391623,
#  'ndcg_at_5': 0.38735167885377786,
#  'ndcg_at_50': 0.4910920986474575,
#  'precision_at_1': 0.2860114404576183}



########################################
# TransformerModel
#########################################

mars-gym run supervised \
--project mercado_livre.config.mercado_livre_transformer \
--recommender-module-class model.MLTransformerModel \
--recommender-extra-params '{
  "n_factors": 100, 
  "n_hid": 50,
  "n_head": 2,
  "n_layers": 1,
  "num_filters": 100,
  "dropout": 0.2, 
  "hist_size": 30, 
  "from_index_mapping": false,
  "path_item_embedding": false, 
  "freeze_embedding": false}' \
--data-frames-preparation-extra-params '{
  "sample_days": 30, 
  "history_window": 30, 
  "column_stratification": "SessionID",
  "filter_only_buy": true}' \
--test-size 0.1 \
--val-size 0.1 \
--early-stopping-min-delta 0.001 \
--optimizer-params '{"weight_decay": 1e-6}' \
--optimizer radam \
--test-split-type random \
--dataset-split-method column \
--learning-rate 0.001 \
--metrics='["loss"]' \
--generator-workers 10  \
--batch-size 512 \
--loss-function ce \
--epochs 1000 \
--run-evaluate  \
--run-evaluate-extra-params " " \
--sample-size-eval 2000 --obs "1"

# {'count': 1923,
#  'coverage_at_20': 0.4539,
#  'coverage_at_5': 0.1474,
#  'mean_average_precision': 0.3824224852186435,
#  'model_task': 'SupervisedModelTraining____mars_gym_model_b____711adb687a',
#  'mrr_at_10': 0.36722919710439217,
#  'mrr_at_5': 0.35355347547235216,
#  'ndcg_at_10': 0.44172683255404005,
#  'ndcg_at_15': 0.4570438995872158,
#  'ndcg_at_20': 0.4685455451193424,
#  'ndcg_at_5': 0.4063762186764765,
#  'ndcg_at_50': 0.5060554806757377,
#  'precision_at_1': 0.3000520020800832}



########################################
# Caser
#########################################

mars-gym run supervised \
--project mercado_livre.config.mercado_livre_rnn \
--recommender-module-class model.MLCaser \
--recommender-extra-params '{
  "n_factors": 100, 
  "p_L": 30, 
  "p_d": 50, 
  "p_nh": 4,
  "p_nv": 4,  
  "dropout": 0.5, 
  "hist_size": 30, 
  "from_index_mapping": false,
  "path_item_embedding": false, 
  "freeze_embedding": false}' \
--data-frames-preparation-extra-params '{
  "sample_days": 30, 
  "history_window": 30, 
  "column_stratification": "SessionID",
  "filter_only_buy": true}'  \
--test-size 0.1 \
--val-size 0.1 \
--early-stopping-min-delta 0.0001 \
--test-split-type random \
--dataset-split-method column \
--learning-rate 0.001 \
--metrics='["loss"]' \
--generator-workers 10  \
--batch-size 512 \
--loss-function ce \
--epochs 100 \
--run-evaluate  \
--run-evaluate-extra-params " " \
--sample-size-eval 2000 



mars-gym run supervised \
--project mercado_livre.config.mercado_livre_rnn \
--recommender-module-class model.MLCaser \
--recommender-extra-params '{
  "n_factors": 100, 
  "p_L": 30, 
  "p_d": 50, 
  "p_nh": 4,
  "p_nv": 4,  
  "dropout": 0.5, 
  "hist_size": 30, 
  "from_index_mapping": false,
  "path_item_embedding": false, 
  "freeze_embedding": false}' \
--data-frames-preparation-extra-params '{
  "sample_days": 30, 
  "history_window": 30, 
  "column_stratification": "SessionID",
  "filter_only_buy": false}'  \
--test-size 0.1 \
--val-size 0.1 \
--early-stopping-min-delta 0.0001 \
--test-split-type random \
--dataset-split-method column \
--learning-rate 0.001 \
--metrics='["loss"]' \
--generator-workers 10  \
--batch-size 512 \
--loss-function ce \
--epochs 100 \
--run-evaluate  \
--run-evaluate-extra-params " " \
--sample-size-eval 2000 



PYTHONPATH="." luigi --module mercado_livre.evaluation MLEvaluationTask \
--model-task-class "mars_gym.simulation.training.SupervisedModelTraining" \
--model-task-id SupervisedModelTraining____mars_gym_model_b____1017b578ca \
--batch-size 1000 \
--local-scheduler