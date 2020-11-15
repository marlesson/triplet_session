# {'count': 1870,
#  'coverage_at_20': 0.4022,
#  'coverage_at_5': 0.1257,
#  'mean_average_precision': 0.5443134518114011,
#  'model_task': 'SupervisedModelTraining____mars_gym_model_b____1f6210ddd3',
#  'mrr_at_10': 0.5331156098803157,
#  'mrr_at_5': 0.5214795008912656,
#  'ndcg_at_10': 0.6042521766040143,
#  'ndcg_at_15': 0.6173788813539623,
#  'ndcg_at_20': 0.6271319979086323,
#  'ndcg_at_5': 0.5739731766883653,
#  'ndcg_at_50': 0.6520301006193968,
#  'precision_at_1': 0.46844919786096256}


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
  "path_item_embedding": false, 
  "freeze_embedding": false}' \
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
--metrics='["loss"]' \
--generator-workers 0  \
--batch-size 512 \
--loss-function ce \
--epochs 100 \
--run-evaluate  \
--run-evaluate-extra-params " " \
--sample-size-eval 2000 --obs ""

PYTHONPATH="." luigi --module mercado_livre.evaluation MLEvaluationTask \
--model-task-class "mars_gym.simulation.training.SupervisedModelTraining" \
--model-task-id SupervisedModelTraining____mars_gym_model_b____1f6210ddd3 \
--batch-size 1000 \
--history-window 30 \
--local-scheduler


### Completo


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
  "freeze_embedding": false}' \
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
--metrics='["loss"]' \
--generator-workers 0  \
--batch-size 512 \
--loss-function ce \
--epochs 100 \
--run-evaluate  \
--run-evaluate-extra-params " " \
--sample-size-eval 2000 --obs ""