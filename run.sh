# bin/bash!

mars-gym run supervised --project mercado_livre.config.mercado_livre_rnn --recommender-module-class model.MLNARMModel --recommender-extra-params '{
  "n_factors": 100, 
  "hidden_size": 100, 
  "n_layers": 1, 
  "dropout": 0.5, 
  "from_index_mapping": false,
  "path_item_embedding": false, 
  "freeze_embedding": false}' --data-frames-preparation-extra-params '{
  "sample_days": 30, 
  "history_window": 5, 
  "column_stratification": "SessionID",
  "filter_only_buy": true}' --test-size 0.1 --val-size 0.1 --early-stopping-min-delta 0.0001 --test-split-type random --dataset-split-method column --learning-rate 0.001 --metrics='["loss"]' --generator-workers 10  --batch-size 512 --loss-function ce --epochs 100 --run-evaluate  --run-evaluate-extra-params " " --sample-size-eval 2000 --obs "new 2"

# {'count': 1923,
#  'coverage_at_20': 0.3925,
#  'coverage_at_5': 0.1293,
#  'mean_average_precision': 0.4285678360815418,
#  'model_task': 'SupervisedModelTraining____mars_gym_model_b____304d67b780',
#  'mrr_at_10': 0.4136781979215676,
#  'mrr_at_5': 0.40210608424336974,
#  'ndcg_at_10': 0.47625529468500755,
#  'ndcg_at_15': 0.4897556049895294,
#  'ndcg_at_20': 0.5022945267745338,
#  'ndcg_at_5': 0.4464809399569543,
#  'ndcg_at_50': 0.540975032014036,
#  'precision_at_1': 0.35725429017160687}


PYTHONPATH="." luigi --module mercado_livre.evaluation MLEvaluationTask \
--model-task-class "mars_gym.simulation.training.SupervisedModelTraining" \
--model-task-id SupervisedModelTraining____mars_gym_model_b____c549ab3480 \
--batch-size 1000 \
--history-window 5 \
--local-scheduler

################################################

mars-gym run supervised --project mercado_livre.config.mercado_livre_rnn --recommender-module-class model.MLNARMModel --recommender-extra-params '{
  "n_factors": 100, 
  "hidden_size": 100, 
  "n_layers": 1, 
  "dropout": 0.5, 
  "from_index_mapping": false,
  "path_item_embedding": false, 
  "freeze_embedding": false}' --data-frames-preparation-extra-params '{
  "sample_days": 30, 
  "history_window": 10, 
  "column_stratification": "SessionID",
  "filter_only_buy": true}' --test-size 0.1 --val-size 0.1 --early-stopping-min-delta 0.0001 --test-split-type random --dataset-split-method column --learning-rate 0.001 --metrics='["loss"]' --generator-workers 10  --batch-size 512 --loss-function ce --epochs 100 --run-evaluate  --run-evaluate-extra-params " " --sample-size-eval 2000 --obs "new 2"

# {'count': 1923,
#  'coverage_at_20': 0.3918,
#  'coverage_at_5': 0.1281,
#  'mean_average_precision': 0.4450604673943399,
#  'model_task': 'SupervisedModelTraining____mars_gym_model_b____fa73f6858b',
#  'mrr_at_10': 0.4292125812016608,
#  'mrr_at_5': 0.41854740856300915,
#  'ndcg_at_10': 0.48626038084566237,
#  'ndcg_at_15': 0.5058643647427757,
#  'ndcg_at_20': 0.5164921462364205,
#  'ndcg_at_5': 0.4585492720954563,
#  'ndcg_at_50': 0.5523687957518786,
#  'precision_at_1': 0.3780551222048882}


mars-gym run supervised --project mercado_livre.config.mercado_livre_rnn --recommender-module-class model.MLNARMModel --recommender-extra-params '{
  "n_factors": 100, 
  "hidden_size": 100, 
  "n_layers": 1, 
  "dropout": 0.5, 
  "from_index_mapping": false,
  "path_item_embedding": false, 
  "freeze_embedding": false}' --data-frames-preparation-extra-params '{
  "sample_days": 30, 
  "history_window": 20, 
  "column_stratification": "SessionID",
  "filter_only_buy": true}' --test-size 0.1 --val-size 0.1 --early-stopping-min-delta 0.0001 --test-split-type random --dataset-split-method column --learning-rate 0.001 --metrics='["loss"]' --generator-workers 10  --batch-size 512 --loss-function ce --epochs 100 --run-evaluate  --run-evaluate-extra-params " " --sample-size-eval 2000 --obs "new 2"  


# {'count': 1936,
#  'coverage_at_20': 0.40299999999999997,
#  'coverage_at_5': 0.13470000000000001,
#  'mean_average_precision': 0.46373775106428594,
#  'model_task': 'SupervisedModelTraining____mars_gym_model_b____d22f4e4ca5',
#  'mrr_at_10': 0.4498669733044733,
#  'mrr_at_5': 0.4393336776859504,
#  'ndcg_at_10': 0.5077335195863266,
#  'ndcg_at_15': 0.521225111134686,
#  'ndcg_at_20': 0.5331235994987719,
#  'ndcg_at_5': 0.4806424448246689,
#  'ndcg_at_50': 0.5661211595423538,
#  'precision_at_1': 0.3972107438016529}


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


