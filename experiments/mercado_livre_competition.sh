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
--run-evaluate-extra-params "" \
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
--project mercado_livre.config.mercado_livre_rnn \
--recommender-module-class model.MLTransformerModel \
--recommender-extra-params '{
  "n_factors": 100, 
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
--run-evaluate-extra-params "" \
--sample-size-eval 2000 

# {'count': 1882,
#  'coverage_at_20': 0.48450000000000004,
#  'coverage_at_5': 0.1603,
#  'mean_average_precision': 0.5053555457210229,
#  'model_task': 'SupervisedModelTraining____mars_gym_model_b____306a5ddd1f',
#  'mrr_at_10': 0.49227286911930906,
#  'mrr_at_5': 0.48406836698547645,
#  'ndcg_at_10': 0.5402550606326615,
#  'ndcg_at_15': 0.5534483440550252,
#  'ndcg_at_20': 0.5630706485015277,
#  'ndcg_at_5': 0.519070674721603,
#  'ndcg_at_50': 0.5949914093233198,
#  'precision_at_1': 0.44845908607863977}

mars-gym run supervised \
--project mercado_livre.config.mercado_livre_rnn \
--recommender-module-class model.MLTransformerModel \
--recommender-extra-params '{
  "n_factors": 100, 
  "n_hid": 512,
  "n_head": 1,
  "n_layers": 1,
  "num_filters": 100,
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
--sample-size-eval 2000 --obs "mlp"


# {'count': 1883,
#  'coverage_at_20': 0.4788,
#  'coverage_at_5': 0.1573,
#  'mean_average_precision': 0.43518744957703154,
#  'model_task': 'SupervisedModelTraining____mars_gym_model_b____9ffcec86a0',
#  'mrr_at_10': 0.4200903657621661,
#  'mrr_at_5': 0.41111701186050637,
#  'ndcg_at_10': 0.48020474007566283,
#  'ndcg_at_15': 0.4962838585404742,
#  'ndcg_at_20': 0.5084917518689729,
#  'ndcg_at_5': 0.45675447650246953,
#  'ndcg_at_50': 0.5425673893012359,
#  'precision_at_1': 0.36484333510355815}


mars-gym run supervised \
--project mercado_livre.config.mercado_livre_rnn \
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
--learning-rate 0.0001 \
--metrics='["loss"]' \
--generator-workers 10  \
--batch-size 512 \
--loss-function ce \
--epochs 1000 \
--run-evaluate  \
--run-evaluate-extra-params " " \
--sample-size-eval 2000 --obs "new arch"

# {'count': 1883,
#  'coverage_at_20': 0.4797,
#  'coverage_at_5': 0.1574,
#  'mean_average_precision': 0.4707748267640072,
#  'model_task': 'SupervisedModelTraining____mars_gym_model_b____c953202fbc',
#  'mrr_at_10': 0.457786038826931,
#  'mrr_at_5': 0.44685785094707026,
#  'ndcg_at_10': 0.519818560552211,
#  'ndcg_at_15': 0.5349965744557524,
#  'ndcg_at_20': 0.5432159574846857,
#  'ndcg_at_5': 0.49174701801058607,
#  'ndcg_at_50': 0.5722090707894935,
#  'precision_at_1': 0.40148698884758366}

mars-gym run supervised \
--project mercado_livre.config.mercado_livre_rnn \
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
--learning-rate 0.0001 \
--metrics='["loss"]' \
--generator-workers 10  \
--batch-size 512 \
--loss-function ce \
--epochs 1000 \
--run-evaluate  \
--run-evaluate-extra-params " " \
--sample-size-eval 2000 --obs "mask"


# {'count': 1883,
#  'coverage_at_20': 0.4966,
#  'coverage_at_5': 0.16269999999999998,
#  'mean_average_precision': 0.4691947921570609,
#  'model_task': 'SupervisedModelTraining____mars_gym_model_b____198e88b707',
#  'mrr_at_10': 0.4567666843689149,
#  'mrr_at_5': 0.44750398300584177,
#  'ndcg_at_10': 0.5189567661889394,
#  'ndcg_at_15': 0.5310781048088629,
#  'ndcg_at_20': 0.5383535217571818,
#  'ndcg_at_5': 0.4947676863660591,
#  'ndcg_at_50': 0.571909545989453,
#  'precision_at_1': 0.3993627190653213}


mars-gym run supervised \
--project mercado_livre.config.mercado_livre_rnn \
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
--sample-size-eval 2000 --obs "mask normalize lr"


# {'count': 1883,
#  'coverage_at_20': 0.4708,
#  'coverage_at_5': 0.1586,
#  'mean_average_precision': 0.4382248740305504,
#  'model_task': 'SupervisedModelTraining____mars_gym_model_b____584e8d543c',
#  'mrr_at_10': 0.42454774970706993,
#  'mrr_at_5': 0.4139936271906532,
#  'ndcg_at_10': 0.4931919918951091,
#  'ndcg_at_15': 0.5060911900218513,
#  'ndcg_at_20': 0.5166518753844432,
#  'ndcg_at_5': 0.4658760037606546,
#  'ndcg_at_50': 0.5504965929748469,
#  'precision_at_1': 0.3616569304301646}


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
--sample-size-eval 2000 --obs ""


# {'count': 1883,
#  'coverage_at_20': 0.473,
#  'coverage_at_5': 0.1578,
#  'mean_average_precision': 0.45022495692334086,
#  'model_task': 'SupervisedModelTraining____mars_gym_model_b____eaf3d118fe',
#  'mrr_at_10': 0.4362841295130196,
#  'mrr_at_5': 0.42518144804390157,
#  'ndcg_at_10': 0.4994611826286401,
#  'ndcg_at_15': 0.5152059941126835,
#  'ndcg_at_20': 0.5253698554018364,
#  'ndcg_at_5': 0.4706150679039585,
#  'ndcg_at_50': 0.5566638654561603,
#  'precision_at_1': 0.379182156133829}

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




