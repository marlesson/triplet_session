PYTHONPATH="."  luigi  \
--module yoochoose.train TripletTraining  \
--project yoochoose.config.triplet_yoochoose  \
--recommender-module-class yoochoose.model.TripletNet  \
--recommender-extra-params '{"n_factors": 50, "use_normalize": true, "negative_random": 0.05}'  \
--data-frames-preparation-extra-params '{"column_stratification": "SessionID", "sample_limit": 1375000, "min_itens_interactions": 10, "max_relative_pos": 5 }'  \
--loss-function-params '{"swap": true, "margin": 1, "balance": 1}'  \
--optimizer-params '{"weight_decay": 1e-5}' \ #1e-5
--optimizer adam \
--learning-rate 1e-3  \
--early-stopping-min-delta 0.0001  \
--early-stopping-patience 35  \
--test-size 0.01  \
--test-split-type time  \
--dataset-split-method column  \
--metrics='["loss","triplet_dist", "triplet_mse", "triplet_acc"]'  \
--save-item-embedding-tsv  \
--local-scheduler  \
--batch-size 128  \
--generator-workers 10  \
--epochs 35  \
--obs ""

PYTHONPATH="."  luigi  \
--module yoochoose.train TripletTraining  \
--project yoochoose.config.triplet_yoochoose  \
--recommender-module-class yoochoose.model.TripletNet  \
--recommender-extra-params '{"n_factors": 50, "use_normalize": true, "negative_random": 0.05, "dropout": 0.2}'  \
--data-frames-preparation-extra-params '{"column_stratification": "SessionID", "sample_limit": 1375000, "min_itens_interactions": 10, "max_relative_pos": 5 }'  \
--loss-function-params '{"swap": true, "margin": 1, "balance": 1}'  \
--optimizer adam \
--learning-rate 1e-3  \
--early-stopping-min-delta 0.0001  \
--early-stopping-patience 35  \
--test-size 0.01  \
--test-split-type time  \
--dataset-split-method column  \
--metrics='["loss","triplet_dist", "triplet_mse", "triplet_acc"]'  \
--save-item-embedding-tsv  \
--local-scheduler  \
--batch-size 128  \
--generator-workers 10  \
--epochs 35  \
--obs ""



PYTHONPATH="."  luigi  \
--module yoochoose.train TripletTraining  \
--project yoochoose.config.triplet_yoochoose  \
--recommender-module-class yoochoose.model.TripletNet  \
--recommender-extra-params '{"n_factors": 50, "use_normalize": true, "negative_random": 0.05, "dropout": 0.3}'  \
--data-frames-preparation-extra-params '{"column_stratification": "SessionID", "sample_limit": 1375000, "min_itens_interactions": 10, "max_relative_pos": 5 }'  \
--loss-function-params '{"swap": true, "margin": 1, "balance": 1}'  \
--optimizer adam \
--learning-rate 1e-3  \
--early-stopping-min-delta 0.0001  \
--early-stopping-patience 10  \
--test-size 0.01  \
--test-split-type time  \
--dataset-split-method column  \
--metrics='["loss","triplet_dist", "triplet_mse", "triplet_acc"]'  \
--save-item-embedding-tsv  \
--local-scheduler  \
--batch-size 128  \
--generator-workers 10  \
--epochs 35  \
--obs ""


# No Embs

mars-gym run supervised \
--project yoochoose.config.sample_yoochoose_with_negative_sample \
--recommender-module-class yoochoose.model.DotModel \
--recommender-extra-params '{"n_factors": 50, "path_item_embedding": false}' \
--data-frames-preparation-extra-params '{"sample_limit": 1375000, "column_stratification": "SessionID"}' \
--load-index-mapping-path '/media/workspace/triplet_session/output/models/TripletTraining/results/TripletTraining____mars_gym_model_b____671ffec623' \
--early-stopping-min-delta 0.0001 \
--negative-proportion 0.5 \
--test-size 0.01 \
--test-split-type time \
--dataset-split-method column \
--learning-rate 0.001 \
--metrics='["loss", "acc"]' \
--sample-size-eval 1000 \
--batch-size 200 \
--epochs 30

mars-gym evaluate supervised --model-task-id SupervisedModelTraining____mars_gym_model_b____37878c6196 --only-new-interactions --local-schedule
#'ndcg_at_5': 0.21532346581878775,



# TripletTraining____mars_gym_model_b____671ffec623 (best)
mars-gym run supervised \
--project yoochoose.config.sample_yoochoose_with_negative_sample \
--recommender-module-class yoochoose.model.DotModel \
--recommender-extra-params '{"n_factors": 50, "path_item_embedding": "/media/workspace/triplet_session/output/models/TripletTraining/results/TripletTraining____mars_gym_model_b____671ffec623/item_embeddings.npy"}' \
--data-frames-preparation-extra-params '{"sample_limit": 1375000, "column_stratification": "SessionID"}' \
--load-index-mapping-path '/media/workspace/triplet_session/output/models/TripletTraining/results/TripletTraining____mars_gym_model_b____671ffec623' \
--early-stopping-min-delta 0.0001 \
--negative-proportion 0.5 \
--test-size 0.01 \
--test-split-type time \
--dataset-split-method column \
--learning-rate 0.001 \
--metrics='["loss", "acc"]' \
--sample-size-eval 1000 \
--batch-size 200 \
--epochs 30

mars-gym evaluate supervised --model-task-id SupervisedModelTraining____mars_gym_model_b____a8e3acfad1 --only-new-interactions --local-schedule
#'ndcg_at_5': 0.3249505370072184,


# TripletTraining____mars_gym_model_b____a49e066d5b
mars-gym run supervised \
--project yoochoose.config.sample_yoochoose_with_negative_sample \
--recommender-module-class yoochoose.model.DotModel \
--recommender-extra-params '{"n_factors": 50, "path_item_embedding": "/media/workspace/triplet_session/output/models/TripletTraining/results/TripletTraining____mars_gym_model_b____a49e066d5b/item_embeddings.npy"}' \
--data-frames-preparation-extra-params '{"sample_limit": 1375000, "column_stratification": "SessionID"}' \
--load-index-mapping-path '/media/workspace/triplet_session/output/models/TripletTraining/results/TripletTraining____mars_gym_model_b____a49e066d5b' \
--early-stopping-min-delta 0.0001 \
--negative-proportion 0.5 \
--test-size 0.01 \
--test-split-type time \
--dataset-split-method column \
--learning-rate 0.001 \
--metrics='["loss", "acc"]' \
--sample-size-eval 1000 \
--batch-size 200 \
--epochs 30

mars-gym evaluate supervised --model-task-id SupervisedModelTraining____mars_gym_model_b____b6250eeca6 --only-new-interactions --local-schedule

#'ndcg_at_5': 0.2451469316375755,


# TripletTraining____mars_gym_model_b____1b251536a4
mars-gym run supervised \
--project yoochoose.config.sample_yoochoose_with_negative_sample \
--recommender-module-class yoochoose.model.DotModel \
--recommender-extra-params '{"n_factors": 50, "path_item_embedding": "/media/workspace/triplet_session/output/models/TripletTraining/results/TripletTraining____mars_gym_model_b____1b251536a4/item_embeddings.npy"}' \
--data-frames-preparation-extra-params '{"sample_limit": 1375000, "column_stratification": "SessionID"}' \
--load-index-mapping-path '/media/workspace/triplet_session/output/models/TripletTraining/results/TripletTraining____mars_gym_model_b____1b251536a4' \
--early-stopping-min-delta 0.0001 \
--negative-proportion 0.5 \
--test-size 0.01 \
--test-split-type time \
--dataset-split-method column \
--learning-rate 0.001 \
--metrics='["loss", "acc"]' \
--sample-size-eval 1000 \
--batch-size 200 \
--epochs 30

mars-gym evaluate supervised --model-task-id SupervisedModelTraining____mars_gym_model_b____8b4083e71c --only-new-interactions --local-schedule
#'ndcg_at_5': 0.21396239651436127,



#TripletTraining____mars_gym_model_b____0cbaaed152
mars-gym run supervised \
--project yoochoose.config.sample_yoochoose_with_negative_sample \
--recommender-module-class yoochoose.model.DotModel \
--recommender-extra-params '{"n_factors": 50, "path_item_embedding": "/media/workspace/triplet_session/output/models/TripletTraining/results/TripletTraining____mars_gym_model_b____0cbaaed152/item_embeddings.npy"}' \
--data-frames-preparation-extra-params '{"sample_limit": 1375000, "column_stratification": "SessionID"}' \
--load-index-mapping-path '/media/workspace/triplet_session/output/models/TripletTraining/results/TripletTraining____mars_gym_model_b____0cbaaed152' \
--early-stopping-min-delta 0.0001 \
--negative-proportion 0.5 \
--test-size 0.01 \
--test-split-type time \
--dataset-split-method column \
--learning-rate 0.001 \
--metrics='["loss", "acc"]' \
--sample-size-eval 1000 \
--batch-size 200 \
--epochs 30

mars-gym evaluate supervised --model-task-id SupervisedModelTraining____mars_gym_model_b____3a4d60be45 --only-new-interactions --local-schedule
#'ndcg_at_5': 0.12660904434021653



#TripletTraining____mars_gym_model_b____27f6ba016b
mars-gym run supervised \
--project yoochoose.config.sample_yoochoose_with_negative_sample \
--recommender-module-class yoochoose.model.DotModel \
--recommender-extra-params '{"n_factors": 50, "path_item_embedding": "/media/workspace/triplet_session/output/models/TripletTraining/results/TripletTraining____mars_gym_model_b____27f6ba016b/item_embeddings.npy"}' \
--data-frames-preparation-extra-params '{"sample_limit": 1375000, "column_stratification": "SessionID"}' \
--load-index-mapping-path '/media/workspace/triplet_session/output/models/TripletTraining/results/TripletTraining____mars_gym_model_b____27f6ba016b' \
--early-stopping-min-delta 0.0001 \
--negative-proportion 0.5 \
--test-size 0.01 \
--test-split-type time \
--dataset-split-method column \
--learning-rate 0.001 \
--metrics='["loss", "acc"]' \
--sample-size-eval 1000 \
--batch-size 200 \
--run-evaluate \
--epochs 30

mars-gym evaluate supervised --model-task-id SupervisedModelTraining____mars_gym_model_b____3a4d60be45 --only-new-interactions --local-schedule
#'ndcg_at_5': 0.273239974093788,

#TripletTraining____mars_gym_model_b____fc382e82e0

mars-gym run supervised \
--project yoochoose.config.sample_yoochoose_with_negative_sample \
--recommender-module-class yoochoose.model.DotModel \
--recommender-extra-params '{"n_factors": 50, "path_item_embedding": "/media/workspace/triplet_session/output/models/TripletTraining/results/TripletTraining____mars_gym_model_b____fc382e82e0/item_embeddings.npy"}' \
--data-frames-preparation-extra-params '{"sample_limit": 1375000, "column_stratification": "SessionID"}' \
--load-index-mapping-path '/media/workspace/triplet_session/output/models/TripletTraining/results/TripletTraining____mars_gym_model_b____fc382e82e0' \
--early-stopping-min-delta 0.0001 \
--negative-proportion 0.5 \
--test-size 0.01 \
--test-split-type time \
--dataset-split-method column \
--learning-rate 0.001 \
--metrics='["loss", "acc"]' \
--sample-size-eval 1000 \
--batch-size 200 \
--run-evaluate \
--epochs 30



# TripletTraining____mars_gym_model_b____9cc68fd7b5
mars-gym run supervised \
--project yoochoose.config.sample_yoochoose_with_negative_sample \
--recommender-module-class yoochoose.model.DotModel \
--recommender-extra-params '{"n_factors": 50, "path_item_embedding": "/media/workspace/triplet_session/output/models/TripletTraining/results/TripletTraining____mars_gym_model_b____9cc68fd7b5/item_embeddings.npy"}' \
--data-frames-preparation-extra-params '{"sample_limit": 1375000, "column_stratification": "SessionID"}' \
--load-index-mapping-path '/media/workspace/triplet_session/output/models/TripletTraining/results/TripletTraining____mars_gym_model_b____9cc68fd7b5' \
--early-stopping-min-delta 0.0001 \
--negative-proportion 0.5 \
--test-size 0.01 \
--test-split-type time \
--dataset-split-method column \
--learning-rate 0.001 \
--metrics='["loss", "acc"]' \
--sample-size-eval 1000 \
--batch-size 200 \
--run-evaluate \
--epochs 30
#'ndcg_at_5': 0.2611469316375756,


# TripletTraining____mars_gym_model_b____ab4259f5a5
mars-gym run supervised \
--project yoochoose.config.sample_yoochoose_with_negative_sample \
--recommender-module-class yoochoose.model.DotModel \
--recommender-extra-params '{"n_factors": 50, "path_item_embedding": "/media/workspace/triplet_session/output/models/TripletTraining/results/TripletTraining____mars_gym_model_b____ab4259f5a5/item_embeddings.npy"}' \
--data-frames-preparation-extra-params '{"sample_limit": 1375000, "column_stratification": "SessionID"}' \
--load-index-mapping-path '/media/workspace/triplet_session/output/models/TripletTraining/results/TripletTraining____mars_gym_model_b____ab4259f5a5' \
--early-stopping-min-delta 0.0001 \
--negative-proportion 0.5 \
--test-size 0.01 \
--test-split-type time \
--dataset-split-method column \
--learning-rate 0.001 \
--metrics='["loss", "acc"]' \
--sample-size-eval 1000 \
--batch-size 200 \
--run-evaluate \
--epochs 30

# 'ndcg_at_5': 0.2512162550795022,


mars-gym run supervised \
--project yoochoose.config.sample_yoochoose_with_negative_sample  \
--recommender-module-class yoochoose.model.DotModel  \
--recommender-extra-params '{"n_factors": 50, "path_item_embedding": "/media/workspace/triplet_session/output/models/TripletTraining/results/TripletTraining____mars_gym_model_b____ab4259f5a5/item_embeddings.npy"}'  \
--data-frames-preparation-extra-params '{"sample_limit": 1375000, "column_stratification": "SessionID"}'  \
--load-index-mapping-path '/media/workspace/triplet_session/output/models/TripletTraining/results/TripletTraining____mars_gym_model_b____ab4259f5a5'  \
--early-stopping-min-delta 0.0001  \
--negative-proportion 0.5  \
--test-size 0.01  \
--test-split-type time  \
--dataset-split-method column  \
--learning-rate 0.001  \
--metrics='["loss", "acc"]'  \
--sample-size-eval 1000  \
--batch-size 200  \
--run-evaluate  \
--epochs 30  \
--obs "freeze" 
#  'ndcg_at_5': 0.5254703974563634,

mars-gym run supervised \
--project yoochoose.config.sample_yoochoose_with_negative_sample  \
--recommender-module-class yoochoose.model.DotModel  \
--recommender-extra-params '{"n_factors": 50, "path_item_embedding": false}'  \
--data-frames-preparation-extra-params '{"sample_limit": 1375000, "column_stratification": "SessionID"}'  \
--load-index-mapping-path '/media/workspace/triplet_session/output/models/TripletTraining/results/TripletTraining____mars_gym_model_b____ab4259f5a5'  \
--early-stopping-min-delta 0.0001  \
--negative-proportion 0.5  \
--test-size 0.01  \
--test-split-type time  \
--dataset-split-method column  \
--learning-rate 0.001  \
--metrics='["loss", "acc"]'  \
--sample-size-eval 1000  \
--batch-size 200  \
--run-evaluate  \
--epochs 30  \
--obs "freeze" 

#  'ndcg_at_5': 0.21532346581878775,
