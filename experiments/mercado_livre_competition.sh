########################################
# Most Popular
#########################################

PYTHONPATH="."  luigi  \
--module train MostPopularTraining  \
--project mercado_livre.config.mercado_livre_interaction \
--local-scheduler  \
--data-frames-preparation-extra-params '{
  "sample_days": 30, 
  "history_window": 10, 
  "column_stratification": "SessionID",
  "filter_only_buy": true}' \
--test-size 0.1 \
--val-size 0.1 \
--test-split-type random \
--dataset-split-method column \
--run-evaluate  \
--run-evaluate-extra-params " " \
--sample-size-eval 2000 --obs "Most Popular"

# {'count': 1883,
#  'coverage_at_20': 0.23199999999999998,
#  'coverage_at_5': 0.0705,
#  'mean_average_precision': 0.23818963464092105,
#  'model_task': 'MostPopularTraining____mars_gym_model_b____22a43850b3',
#  'mrr_at_10': 0.2179047703344039,
#  'mrr_at_5': 0.2012126040007081,
#  'ndcg_at_10': 0.31761205471922455,
#  'ndcg_at_15': 0.34209003255821546,
#  'ndcg_at_20': 0.35841711179965924,
#  'ndcg_at_5': 0.2746144632597502,
#  'ndcg_at_50': 0.4028361962948689,
#  'precision_at_1': 0.12639405204460966}


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
  "dropout": 0.5, 
  "from_index_mapping": false,
  "path_item_embedding": false, 
  "freeze_embedding": false}' \
--data-frames-preparation-extra-params '{
  "sample_days": 30, 
  "history_window": 10, 
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
--run-evaluate-extra-params "" \
--sample-size-eval 2000 --obs "filter only buy2"

#SupervisedModelTraining____mars_gym_model_b____c549ab3480

PYTHONPATH="." luigi --module mercado_livre.evaluation MLEvaluationTask \
--model-task-class "mars_gym.simulation.training.SupervisedModelTraining" \
--model-task-id SupervisedModelTraining____mars_gym_model_b____c549ab3480 \
--batch-size 1000 \
--local-scheduler

mars-gym evaluate supervised --model-task-id SupervisedModelTraining____mars_gym_model_b____c549ab3480 

# {'count': 1883,
#  'coverage_at_20': 0.44310000000000005,
#  'coverage_at_5': 0.15,
#  'mean_average_precision': 0.5056065270486231,
#  'model_task': 'SupervisedModelTraining____mars_gym_model_b____c549ab3480',
#  'mrr_at_10': 0.4928396513500072,
#  'mrr_at_5': 0.48252788104089217,
#  'ndcg_at_10': 0.5477472675988672,
#  'ndcg_at_15': 0.5623398645857777,
#  'ndcg_at_20': 0.5715215652154559,
#  'ndcg_at_5': 0.521200769656147,
#  'ndcg_at_50': 0.6016170613387436,
#  'precision_at_1': 0.443441317047265}



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
  "history_window": 10, 
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

# {'count': 1575,
#  'coverage_at_20': 0.293,
#  'coverage_at_5': 0.0834,
#  'mean_average_precision': 0.5706972444672092,
#  'model_task': 'SupervisedModelTraining____mars_gym_model_b____31de201200',
#  'mrr_at_10': 0.559378936759889,
#  'mrr_at_5': 0.5508677248677248,
#  'ndcg_at_10': 0.6209522660691101,
#  'ndcg_at_15': 0.6357390850957264,
#  'ndcg_at_20': 0.6441403552044158,
#  'ndcg_at_5': 0.5989437601845358,
#  'ndcg_at_50': 0.6678620211665856,
#  'precision_at_1': 0.5022222222222222}


mars-gym evaluate supervised --model-task-id SupervisedModelTraining____mars_gym_model_b____31de201200 

# {'count': 1996,
#  'coverage_at_20': 0.3539,
#  'coverage_at_5': 0.1032,
#  'mean_average_precision': 0.6538278717965795,
#  'model_task': 'SupervisedModelTraining____mars_gym_model_b____31de201200',
#  'mrr_at_10': 0.644817810223622,
#  'mrr_at_5': 0.6378841015364061,
#  'ndcg_at_10': 0.696523562786302,
#  'ndcg_at_15': 0.7083197541649245,
#  'ndcg_at_20': 0.7150715833048564,
#  'ndcg_at_5': 0.67861785632544,
#  'ndcg_at_50': 0.7338909585438341,
#  'precision_at_1': 0.5966933867735471}



PYTHONPATH="." luigi --module mercado_livre.evaluation MLEvaluationTask \
--model-task-class "mars_gym.simulation.training.SupervisedModelTraining" \
--model-task-id SupervisedModelTraining____mars_gym_model_b____31de201200 \
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
#SupervisedModelTraining____mars_gym_model_b____65ecad12c6

# {'count': 1883,
#  'coverage_at_20': 0.4168,
#  'coverage_at_5': 0.1335,
#  'mean_average_precision': 0.41581036701559637,
#  'model_task': 'SupervisedModelTraining____mars_gym_model_b____65ecad12c6',
#  'mrr_at_10': 0.40033971457232215,
#  'mrr_at_5': 0.38983005841741897,
#  'ndcg_at_10': 0.46969889861629416,
#  'ndcg_at_15': 0.48525699192261906,
#  'ndcg_at_20': 0.4982449659162299,
#  'ndcg_at_5': 0.44225677114874595,
#  'ndcg_at_50': 0.5362738055808602,
#  'precision_at_1': 0.33669676048858205}



########################################
# TransformerModel
#########################################


mars-gym run supervised \
--project mercado_livre.config.mercado_livre_transformer \
--recommender-module-class model.MLTransformerModel \
--recommender-extra-params '{
  "n_factors": 100, 
  "n_hid": 100,
  "n_head": 1,
  "n_layers": 1,
  "num_filters": 50,
  "dropout": 0.2, 
  "hist_size": 10, 
  "from_index_mapping": false,
  "path_item_embedding": false, 
  "freeze_embedding": false}' \
--data-frames-preparation-extra-params '{
  "sample_days": 30, 
  "history_window": 10, 
  "column_stratification": "SessionID",
  "filter_only_buy": true}' \
--test-size 0.1 \
--val-size 0.1 \
--early-stopping-min-delta 0.001 \
--optimizer-params '{"weight_decay": 0}' \
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
--sample-size-eval 2000 --obs "2"


# {'count': 1883,
#  'coverage_at_20': 0.4706,
#  'coverage_at_5': 0.1576,
#  'mean_average_precision': 0.44769524680979,
#  'model_task': 'SupervisedModelTraining____mars_gym_model_b____2c3439f2a6',
#  'mrr_at_10': 0.4332279628084195,
#  'mrr_at_5': 0.4229509647725261,
#  'ndcg_at_10': 0.49404530238244665,
#  'ndcg_at_15': 0.5106850728725741,
#  'ndcg_at_20': 0.5199737572164871,
#  'ndcg_at_5': 0.4674522099380267,
#  'ndcg_at_50': 0.5533498850711043,
#  'precision_at_1': 0.37812002124269783}



mars-gym run supervised \
--project mercado_livre.config.mercado_livre_transformer \
--recommender-module-class model.MLTransformerModel \
--recommender-extra-params '{
  "n_factors": 100, 
  "n_hid": 100,
  "n_head": 1,
  "n_layers": 1,
  "num_filters": 50,
  "dropout": 0.2, 
  "hist_size": 10, 
  "from_index_mapping": false,
  "path_item_embedding": false, 
  "freeze_embedding": false}' \
--data-frames-preparation-extra-params '{
  "sample_days": 30, 
  "history_window": 20, 
  "column_stratification": "SessionID",
  "filter_only_buy": true}' \
--test-size 0.1 \
--val-size 0.1 \
--early-stopping-min-delta 0.001 \
--optimizer-params '{"weight_decay": 0}' \
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
--sample-size-eval 2000 --obs "2"

# {'count': 1883,
#  'coverage_at_20': 0.467,
#  'coverage_at_5': 0.1555,
#  'mean_average_precision': 0.4516306393730758,
#  'model_task': 'SupervisedModelTraining____mars_gym_model_b____e9af33328c',
#  'mrr_at_10': 0.43810134958568303,
#  'mrr_at_5': 0.4268543104974332,
#  'ndcg_at_10': 0.506382372445821,
#  'ndcg_at_15': 0.5201844239794605,
#  'ndcg_at_20': 0.529385319483825,
#  'ndcg_at_5': 0.47698445125351163,
#  'ndcg_at_50': 0.5632105667438765,
#  'precision_at_1': 0.37599575146043546}


#SupervisedModelTraining____mars_gym_model_b____c0474b920f


########################################
# Caser
#########################################

mars-gym run supervised \
--project mercado_livre.config.mercado_livre_rnn \
--recommender-module-class model.MLCaser \
--recommender-extra-params '{
  "n_factors": 100, 
  "p_L": 10, 
  "p_d": 50, 
  "p_nh": 16,
  "p_nv": 4,  
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
--run-evaluate-extra-params "" \
--sample-size-eval 2000 




