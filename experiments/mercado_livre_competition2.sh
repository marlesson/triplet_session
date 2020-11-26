
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
  "dropout": 0.5, 
  "from_index_mapping": false,
  "path_item_embedding": "/media/workspace/triplet_session/output/mercado_livre/assets/mercadolivre-100d.bin", 
  "freeze_embedding": true}' \
--data-frames-preparation-extra-params '{
  "sample_days": 360, 
  "history_window": 20, 
  "column_stratification": "SessionID",
  "normalize_dense_features": "min_max",
  "min_interactions": 2,
  "filter_only_buy": true,
  "sample_view": 10000}' \
--optimizer adam \
--optimizer-params '{"weight_decay": 0}' \
--test-size 0.0 \
--val-size 0.1 \
--early-stopping-min-delta 0.0001 \
--test-split-type random \
--dataset-split-method column \
--learning-rate 0.001 \
--metrics='["loss"]' \
--generator-workers 5  \
--batch-size 512 \
--loss-function ce \
--epochs 100 \
--obs ""

PYTHONPATH="." luigi --module mercado_livre.evaluation EvaluationSubmission \
--model-task-class "mars_gym.simulation.training.SupervisedModelTraining" \
--model-task-id SupervisedModelTraining____mars_gym_model_b____eb56e503cd \
--normalize-file-path "bb573d7539_std_scaler.pkl" \
--history-window 20 \
--batch-size 1000 \
--percent-limit 1 \
--local-scheduler \
--local  

# {'count': 1000,
#  'mean_average_precision': 0.2394373015873016,
#  'model_task': 'SupervisedModelTraining____mars_gym_model_b____3fcfe47a17',
#  'mrr_at_10': 0.2394373015873016,
#  'mrr_at_5': 0.23443333333333333,
#  'ndcg_at_10': 0.2859531007691741,
#  'ndcg_at_15': 0.2859531007691741,
#  'ndcg_at_20': 0.2859531007691741,
#  'ndcg_at_5': 0.2734792498335262,
#  'ndcg_at_50': 0.2859531007691741,
#  'ndcg_ml': 0.24913278525801766,
#  'percent_limit': 1.0,
#  'precision_at_1': 0.195}

{'count': 1000,
 'mean_average_precision': 0.22002619047619046,
 'model_task': 'SupervisedModelTraining____mars_gym_model_b____d91b3ee0df',
 'mrr_at_10': 0.22002619047619046,
 'mrr_at_5': 0.21621666666666664,
 'ndcg_at_10': 0.2608505589556474,
 'ndcg_at_15': 0.2608505589556474,
 'ndcg_at_20': 0.2608505589556474,
 'ndcg_at_5': 0.25094851204300833,
 'ndcg_at_50': 0.2608505589556474,
 'ndcg_ml': 0.22531822556200137,
 'percent_limit': 1.0,
 'precision_at_1': 0.181}


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
  "dropout": 0.5, 
  "from_index_mapping": false,
  "path_item_embedding": "/media/workspace/triplet_session/output/mercado_livre/assets/mercadolivre-100d.bin", 
  "freeze_embedding": true}' \
--data-frames-preparation-extra-params '{
  "sample_days": 360, 
  "history_window": 20, 
  "column_stratification": "SessionID",
  "normalize_dense_features": "min_max",
  "min_interactions": 2,
  "filter_only_buy": true,
  "sample_view": 10000}' \
--optimizer-params '{"weight_decay": 1e-5}' \
--test-size 0.0 \
--val-size 0.1 \
--early-stopping-min-delta 0.0001 \
--test-split-type random \
--dataset-split-method column \
--learning-rate 0.0001 \
--metrics='["loss"]' \
--generator-workers 10  \
--batch-size 512 \
--loss-function custom_ce \
--loss-function-params '{"c": 1}'  \
--epochs 100 \
--obs ""


PYTHONPATH="." luigi --module mercado_livre.evaluation EvaluationSubmission \
--model-task-class "train.MercadoLivreTraining" \
--model-task-id MercadoLivreTraining____mars_gym_model_b____7f1da3af0f \
--normalize-file-path "bb573d7539_std_scaler.pkl" \
--history-window 20 \
--batch-size 1000 \
--percent-limit 1 \
--local-scheduler \
--local 

# {'count': 1000,
#  'mean_average_precision': 0.13661785714285712,
#  'model_task': 'MercadoLivreTraining____mars_gym_model_b____7f1da3af0f',
#  'mrr_at_10': 0.13661785714285712,
#  'mrr_at_5': 0.13558333333333333,
#  'ndcg_at_10': 0.1456271790359318,
#  'ndcg_at_15': 0.1456271790359318,
#  'ndcg_at_20': 0.1456271790359318,
#  'ndcg_at_5': 0.14291650827500021,
#  'ndcg_at_50': 0.1456271790359318,
#  'ndcg_ml': 0.1233714427025859,
#  'percent_limit': 1.0,
#  'precision_at_1': 0.128}

PYTHONPATH="." luigi --module mercado_livre.evaluation EvaluationSubmission \
--model-task-class "train.MercadoLivreTraining" \
--model-task-id MercadoLivreTraining____mars_gym_model_b____7f1da3af0f \
--normalize-file-path "bb573d7539_std_scaler.pkl" \
--history-window 20 \
--batch-size 1000 \
--percent-limit 1 \
--local-scheduler 