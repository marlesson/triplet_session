
## Original

mars-gym run supervised \
--project mercado_livre.config.mercado_livre_rnn \
--recommender-module-class model.NARMModel \
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
  "history_window": 20, 
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
--generator-workers 10  \
--batch-size 512 \
--loss-function ce \
--epochs 100 \
--run-evaluate  \
--run-evaluate-extra-params " " \
--sample-size-eval 2000 --obs ""


# {'count': 1875,
#  'coverage_at_20': 0.31579999999999997,
#  'coverage_at_5': 0.0973,
#  'mean_average_precision': 0.4055599880554155,
#  'model_task': 'SupervisedModelTraining____mars_gym_model_b____c92ef1aefd',
#  'mrr_at_10': 0.38897439153439156,
#  'mrr_at_5': 0.3744088888888889,
#  'ndcg_at_10': 0.4490304673285157,
#  'ndcg_at_15': 0.4686972383219778,
#  'ndcg_at_20': 0.4822887261161155,
#  'ndcg_at_5': 0.41097623803352096,
#  'ndcg_at_50': 0.5196535461214186,
#  'precision_at_1': 0.3376}


PYTHONPATH="." luigi --module mercado_livre.evaluation EvaluationSubmission \
--model-task-class "mars_gym.simulation.training.SupervisedModelTraining" \
--model-task-id SupervisedModelTraining____mars_gym_model_b____c92ef1aefd \
--normalize-file-path "4956728137_std_scaler.pkl" \
--history-window 20 \
--batch-size 1000 \
--local-scheduler \
--file "/media/workspace/triplet_session/output/mercado_livre/dataset/test_0.10_test=random_42_SessionInteractionDataFrame_____SessionID_4956728137.csv"

# {'count': 1000,
#  'mean_average_precision': 0.19109563492063492,
#  'model_task': 'SupervisedModelTraining____mars_gym_model_b____c92ef1aefd',
#  'mrr_at_10': 0.19109563492063492,
#  'mrr_at_5': 0.18885,
#  'ndcg_at_10': 0.2203986655840672,
#  'ndcg_at_15': 0.2203986655840672,
#  'ndcg_at_20': 0.2203986655840672,
#  'ndcg_at_5': 0.21457760819564894,
#  'ndcg_at_50': 0.2203986655840672,
#  'precision_at_1': 0.163}

###   ML NARM


mars-gym run supervised \
--project mercado_livre.config.mercado_livre_narm \
--recommender-module-class model.MLNARMModel \
--recommender-extra-params '{
  "n_factors": 100, 
  "hidden_size": 200, 
  "dense_size": 19,
  "n_layers": 1, 
  "dropout": 0.2, 
  "from_index_mapping": false,
  "path_item_embedding": "/media/workspace/triplet_session/output/mercado_livre/assets/mercadolivre-100d.bin", 
  "freeze_embedding": true}' \
--data-frames-preparation-extra-params '{
  "sample_days": 60, 
  "history_window": 20, 
  "column_stratification": "SessionID",
  "normalize_dense_features": "min_max",
  "min_interactions": 2,
  "filter_only_buy": true}' \
--test-size 0.0 \
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
--obs ""

SupervisedModelTraining____mars_gym_model_b____cac288a509

PYTHONPATH="." luigi --module mercado_livre.evaluation EvaluationSubmission \
--model-task-class "mars_gym.simulation.training.SupervisedModelTraining" \
--model-task-id SupervisedModelTraining____mars_gym_model_b____cac288a509 \
--normalize-file-path "1e51172d1f_std_scaler.pkl" \
--history-window 20 \
--batch-size 1000 \
--percent-limit 1.0 \
--local-scheduler \
--local  

# Normal

# {'count': 1000,
#  'mean_average_precision': 0.24455793650793653,
#  'model_task': 'SupervisedModelTraining____mars_gym_model_b____cac288a509',
#  'mrr_at_10': 0.24455793650793653,
#  'mrr_at_5': 0.24168333333333333,
#  'ndcg_at_10': 0.27953293207830093,
#  'ndcg_at_15': 0.27953293207830093,
#  'ndcg_at_20': 0.27953293207830093,
#  'ndcg_at_5': 0.27204028852173046,
#  'ndcg_at_50': 0.27953293207830093,
#  'precision_at_1': 0.211}


# Pos Process

{'count': 1000,
 'mean_average_precision': 0.2282940476190476,
 'model_task': 'SupervisedModelTraining____mars_gym_model_b____5d5ef06c6a',
 'mrr_at_10': 0.2282940476190476,
 'mrr_at_5': 0.22606666666666667,
 'ndcg_at_10': 0.2566032130172329,
 'ndcg_at_15': 0.2566032130172329,
 'ndcg_at_20': 0.2566032130172329,
 'ndcg_at_5': 0.2506333309779429,
 'ndcg_at_50': 0.2566032130172329,
 'precision_at_1': 0.201}


#### Other

PYTHONPATH="."  luigi  \
--module train MercadoLivreTraining  \
--project mercado_livre.config.mercado_livre_narm_custom \
--local-scheduler  \
--recommender-module-class model.MLNARMModel \
--recommender-extra-params '{
  "n_factors": 100, 
  "hidden_size": 200, 
  "dense_size": 19,
  "n_layers": 1, 
  "dropout": 0.2, 
  "from_index_mapping": false,
  "path_item_embedding": "/media/workspace/triplet_session/output/mercado_livre/assets/mercadolivre-100d.bin", 
  "freeze_embedding": true}' \
--data-frames-preparation-extra-params '{
  "sample_days": 60, 
  "history_window": 20, 
  "column_stratification": "SessionID",
  "normalize_dense_features": "min_max",
  "min_interactions": 2,
  "filter_only_buy": true}' \
--test-size 0.0 \
--val-size 0.1 \
--early-stopping-min-delta 0.0001 \
--test-split-type random \
--dataset-split-method column \
--learning-rate 0.001 \
--metrics='["loss"]' \
--generator-workers 10  \
--batch-size 512 \
--loss-function custom_ce \
--loss-function-params '{"c": 1}'  \
--epochs 100 \
--obs ""


PYTHONPATH="." luigi --module mercado_livre.evaluation EvaluationSubmission \
--model-task-class "train.MercadoLivreTraining" \
--model-task-id MercadoLivreTraining____mars_gym_model_b____d338444271 \
--normalize-file-path "7c7b77b344_std_scaler.pkl" \
--history-window 20 \
--batch-size 1000 \
--local-scheduler \
--local 