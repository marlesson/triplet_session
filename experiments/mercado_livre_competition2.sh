# {'count': 1857,
#  'coverage_at_20': 0.2321,
#  'coverage_at_5': 0.06570000000000001,
#  'mean_average_precision': 0.6073892904621753,
#  'model_task': 'SupervisedModelTraining____mars_gym_model_b____dbab7e0e22',
#  'mrr_at_10': 0.6033216230308314,
#  'mrr_at_5': 0.5954227248249866,
#  'ndcg_at_10': 0.6866547375767993,
#  'ndcg_at_15': 0.690386212956009,
#  'ndcg_at_20': 0.6912965797947,
#  'ndcg_at_5': 0.666648573879492,
#  'ndcg_at_50': 0.6916513442353528,
#  'precision_at_1': 0.5228863758750674}



mars-gym run supervised \
--project mercado_livre.config.mercado_livre_narm \
--recommender-module-class model.MLNARMModel \
--recommender-extra-params '{
  "n_factors": 100, 
  "hidden_size": 100, 
  "dense_size": 19,
  "n_layers": 1, 
  "dropout": 0.5, 
  "from_index_mapping": false,
  "path_item_embedding": "/media/workspace/triplet_session/output/mercado_livre/assets/mercadolivre-100d.bin", 
  "freeze_embedding": true}' \
--data-frames-preparation-extra-params '{
  "sample_days": 30, 
  "history_window": 30, 
  "column_stratification": "SessionID",
  "normalize_dense_features": "min_max",
  "min_interactions": 2,
  "filter_only_buy": true}' \
--test-size 0.1 \
--val-size 0.1 \
--early-stopping-min-delta 0.0001 \
--test-split-type random \
--dataset-split-method column \
--learning-rate 0.001 \
--optimizer-params '{"weight_decay": 1e-6}' \
--metrics='["loss"]' \
--generator-workers 0  \
--batch-size 512 \
--loss-function ce \
--epochs 1000 \
--run-evaluate  \
--run-evaluate-extra-params " " \
--sample-size-eval 2000 --obs "best1"
#SupervisedModelTraining____mars_gym_model_b____32067edf5d

PYTHONPATH="." luigi --module mercado_livre.evaluation MLEvaluationTask \
--model-task-class "mars_gym.simulation.training.SupervisedModelTraining" \
--model-task-id SupervisedModelTraining____mars_gym_model_b____dbab7e0e22 \
--batch-size 1000 \
--local-scheduler

## All


mars-gym run supervised \
--project mercado_livre.config.mercado_livre_narm \
--recommender-module-class model.MLNARMModel \
--recommender-extra-params '{
  "n_factors": 100, 
  "hidden_size": 100, 
  "dense_size": 19,
  "n_layers": 1, 
  "dropout": 0.5, 
  "from_index_mapping": false,
  "path_item_embedding": "/media/workspace/triplet_session/output/mercado_livre/assets/mercadolivre-100d.bin", 
  "freeze_embedding": true}' \
--data-frames-preparation-extra-params '{
  "sample_days": 30, 
  "history_window": 30, 
  "column_stratification": "SessionID",
  "normalize_dense_features": "min_max",
  "min_interactions": 2,
  "filter_only_buy": false}' \
--test-size 0.1 \
--val-size 0.1 \
--early-stopping-min-delta 0.0001 \
--test-split-type random \
--dataset-split-method column \
--learning-rate 0.001 \
--optimizer-params '{"weight_decay": 1e-6}' \
--metrics='["loss"]' \
--generator-workers 0  \
--batch-size 512 \
--loss-function ce \
--epochs 1000 \
--run-evaluate  \
--run-evaluate-extra-params " " \
--sample-size-eval 2000 --obs ""