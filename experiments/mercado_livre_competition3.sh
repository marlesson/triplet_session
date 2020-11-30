
#--optimizer-params '{"weight_decay": 1e-05}' \
# Original
# {'count': 1000,
#  'mean_average_precision': 0.2118043650793651,
#  'model_task': 'SupervisedModelTraining____mars_gym_model_b____ec63b04d80',
#  'mrr_at_10': 0.2118043650793651,
#  'mrr_at_5': 0.20828333333333332,
#  'ndcg_at_10': 0.25565647755200616,
#  'ndcg_at_15': 0.25565647755200616,
#  'ndcg_at_20': 0.25565647755200616,
#  'ndcg_at_5': 0.24664288170915558,
#  'ndcg_at_50': 0.25565647755200616,
#  'ndcg_ml': 0.22116540461666073,
#  'percent_limit': 1.0,
#  'precision_at_1': 0.169}


########################################
# Most Popular
#########################################

PYTHONPATH="."  luigi  \
--module train MostPopularTraining  \
--project mercado_livre.config.mercado_livre_interaction \
--local-scheduler  \
--data-frames-preparation-extra-params '{
  "sample_days": 60, 
  "history_window": 20, 
  "column_stratification": "SessionID",
  "normalize_dense_features": "min_max",
  "min_interactions": 5,
  "filter_only_buy": true,
  "sample_view": 10000}' \
--test-size 0.0 \
--val-size 0.1 \
--test-split-type random \
--dataset-split-method column \
--obs "Most Popular"

#########################

mars-gym run supervised \
--project mercado_livre.config.mercado_livre_narm \
--recommender-module-class model.MLNARMModel2 \
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
  "sample_days": 60, 
  "history_window": 20, 
  "column_stratification": "SessionID",
  "normalize_dense_features": "min_max",
  "min_interactions": 5,
  "filter_only_buy": true,
  "sample_view": 10000}' \
--optimizer radam \
--optimizer-params '{"weight_decay": 1e-05}' \
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
SupervisedModelTraining____mars_gym_model_b____dfeb3adddb

PYTHONPATH="." luigi --module mercado_livre.evaluation EvaluationSubmission \
--model-task-class "mars_gym.simulation.training.SupervisedModelTraining" \
--model-task-id SupervisedModelTraining____mars_gym_model_b____06e74a0146 \
--normalize-file-path "5623558488_std_scaler.pkl" \
--history-window 20 \
--batch-size 1000 \
--percent-limit 1 \
--local-scheduler \
--local  

{'count': 1000,
 'mean_average_precision': 0.2677611111111111,
 'model_task': 'SupervisedModelTraining____mars_gym_model_b____06e74a0146',
 'mrr_at_10': 0.2677611111111111,
 'mrr_at_5': 0.26526666666666665,
 'ndcg_at_10': 0.30712078947113514,
 'ndcg_at_15': 0.30712078947113514,
 'ndcg_at_20': 0.30712078947113514,
 'ndcg_at_5': 0.3006351645788738,
 'ndcg_at_50': 0.30712078947113514,
 'ndcg_ml': 0.2636584813154937,
 'percent_limit': 1.0,
 'precision_at_1': 0.229}



#--------------------
mars-gym run supervised --project mercado_livre.config.mercado_livre_narm --recommender-module-class model.MLNARMModel2 --recommender-extra-params '{
  "n_factors": 100, 
  "hidden_size": 200, 
  "dense_size": 19,
  "n_layers": 1, 
  "dropout": 0.2, 
  "history_window": 20, 
  "history_word_window": 3,
  "from_index_mapping": false,
  "path_item_embedding": "/media/workspace/triplet_session/output/mercado_livre/assets/mercadolivre-100d.bin", 
  "freeze_embedding": true}' --data-frames-preparation-extra-params '{
  "sample_days": 60, 
  "history_window": 20, 
  "column_stratification": "SessionID",
  "normalize_dense_features": "min_max",
  "min_interactions": 5,
  "filter_only_buy": true,
  "sample_view": 10000}' --optimizer adam --optimizer-params '{"weight_decay": 1e-4}' --test-size 0.0 --val-size 0.1 --early-stopping-min-delta 0.0001 --test-split-type random --dataset-split-method column --learning-rate 0.001 --metrics='["loss"]' --generator-workers 0  --batch-size 512 --loss-function ce --epochs 1000 --obs "1"




PYTHONPATH="." luigi --module mercado_livre.evaluation EvaluationSubmission \
--model-task-class "train.MercadoLivreTraining" \
--model-task-id MercadoLivreTraining____mars_gym_model_b____ac1489a46d \
--normalize-file-path "f0bb52b478_std_scaler.pkl" \
--history-window 20 \
--batch-size 1000 \
--percent-limit 1 \
--local-scheduler \
--local  