
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
--sample-size-eval 5 --obs ""


PYTHONPATH="." luigi --module mercado_livre.evaluation EvaluationSubmission \
--model-task-class "mars_gym.simulation.training.SupervisedModelTraining" \
--model-task-id SupervisedModelTraining____mars_gym_model_b____56c475908e \
--normalize-file-path "4956728137_std_scaler.pkl" \
--history-window 20 \
--batch-size 1000 \
--local-scheduler \
--file "/media/workspace/triplet_session/output/mercado_livre/dataset/test_0.10_test=random_42_SessionInteractionDataFrame_____SessionID_4956728137.csv"

# {'count': 1000,
#  'mean_average_precision': 0.2090595238095238,
#  'model_task': 'SupervisedModelTraining____mars_gym_model_b____beec81f53a',
#  'mrr_at_10': 0.2090595238095238,
#  'mrr_at_5': 0.20761666666666667,
#  'ndcg_at_10': 0.23962853236489454,
#  'ndcg_at_15': 0.23962853236489454,
#  'ndcg_at_20': 0.23962853236489454,
#  'ndcg_at_5': 0.23570265441986954,
#  'ndcg_at_50': 0.23962853236489454,
#  'precision_at_1': 0.179}

Com a mascara

# {'count': 1000,
#  'mean_average_precision': 0.2081757936507936,
#  'model_task': 'SupervisedModelTraining____mars_gym_model_b____56c475908e',
#  'mrr_at_10': 0.2081757936507936,
#  'mrr_at_5': 0.20645,
#  'ndcg_at_10': 0.23868854316208246,
#  'ndcg_at_15': 0.23868854316208246,
#  'ndcg_at_20': 0.23868854316208246,
#  'ndcg_at_5': 0.23422637343415534,
#  'ndcg_at_50': 0.23868854316208246,
#  'precision_at_1': 0.178}


#### Other


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
  "hist_size": 20, 
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
--sample-size-eval 5 --obs ""


PYTHONPATH="." luigi --module mercado_livre.evaluation EvaluationSubmission \
--model-task-class "mars_gym.simulation.training.SupervisedModelTraining" \
--model-task-id SupervisedModelTraining____mars_gym_model_b____bcc72a076e \
--normalize-file-path "4956728137_std_scaler.pkl" \
--history-window 20 \
--batch-size 1000 \
--local-scheduler \
--file "/media/workspace/triplet_session/output/mercado_livre/dataset/test_0.10_test=random_42_SessionInteractionDataFrame_____SessionID_4956728137.csv"



mars-gym run supervised \
--project mercado_livre.config.mercado_livre_rnn \
--recommender-module-class model.MLCaser \
--recommender-extra-params '{
  "n_factors": 100, 
  "p_L": 20, 
  "p_d": 50, 
  "p_nh": 4,
  "p_nv": 4,  
  "dropout": 0.5, 
  "hist_size": 20, 
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
--sample-size-eval 5 --obs ""


PYTHONPATH="." luigi --module mercado_livre.evaluation EvaluationSubmission \
--model-task-class "mars_gym.simulation.training.SupervisedModelTraining" \
--model-task-id SupervisedModelTraining____mars_gym_model_b____bcc72a076e \
--normalize-file-path "4956728137_std_scaler.pkl" \
--history-window 20 \
--batch-size 1000 \
--local-scheduler \
--file "/media/workspace/triplet_session/output/mercado_livre/dataset/test_0.10_test=random_42_SessionInteractionDataFrame_____SessionID_4956728137.csv"
