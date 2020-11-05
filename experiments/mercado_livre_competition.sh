########################################
# Most Popular
#########################################

PYTHONPATH="."  luigi  \
--module train MostPopularTraining  \
--project mercado_livre.config.mercado_livre_interaction \
--local-scheduler  \
--data-frames-preparation-extra-params '{"sample_days": 30, "history_window": 10, "column_stratification": "SessionID"}' \
--test-size 0.1 \
--val-size 0.1 \
--test-split-type random \
--dataset-split-method column \
--run-evaluate  \
--sample-size-eval 2000 --obs "Most Popular"

# {'count': 1803,
#  'coverage_at_20': 0.19079999999999997,
#  'coverage_at_5': 0.053200000000000004,
#  'mean_average_precision': 0.21955095513303172,
#  'model_task': 'MostPopularTraining____mars_gym_model_b____9cbbc03ad2',
#  'mrr_at_10': 0.2013850372835398,
#  'mrr_at_5': 0.18455352190793123,
#  'ndcg_at_10': 0.2943913596800349,
#  'ndcg_at_15': 0.3143512186292841,
#  'ndcg_at_20': 0.3271172511823409,
#  'ndcg_at_5': 0.25102567183253394,
#  'ndcg_at_50': 0.366171951012553,
#  'precision_at_1': 0.11758180809761509}

# ########################################
# # NARMModel
# #########################################


mars-gym run supervised \
--project mercado_livre.config.mercado_livre_rnn \
--recommender-module-class model.NARMModel \
--recommender-extra-params '{
  "n_factors": 100, 
  "hidden_size": 100, 
  "n_layers": 1, 
  "dropout": 0.25, 
  "from_index_mapping": false,
  "path_item_embedding": false, 
  "freeze_embedding": false}' \
--data-frames-preparation-extra-params '{"sample_days": 30, "history_window": 10, "column_stratification": "SessionID"}' \
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
--sample-size-eval 2000 --obs ""


PYTHONPATH="." luigi --module mercado_livre.evaluation MLEvaluationTask \
--model-task-class "mars_gym.simulation.training.SupervisedModelTraining" \
--model-task-id SupervisedModelTraining____mars_gym_model_b____b92d68b8b7 \
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
  "hist_size": 10,
  "from_index_mapping": false,
  "path_item_embedding": false, 
  "freeze_embedding": false}' \
--data-frames-preparation-extra-params '{
  "sample_days": 30, 
  "history_window": 10, 
  "column_stratification": "SessionID"}' \
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
--sample-size-eval 2000



PYTHONPATH="." luigi --module mercado_livre.evaluation MLEvaluationTask \
--model-task-class "mars_gym.simulation.training.SupervisedModelTraining" \
--model-task-id SupervisedModelTraining____mars_gym_model_b____a1e3012119 \
--batch-size 1000 \
--local-scheduler


