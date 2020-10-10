mars-gym run supervised \
--project globo.config.sample_globo_with_negative_sample \
--recommender-module-class model.DotModel \
--recommender-extra-params '{"n_factors": 100, "path_item_embedding": false, "freeze_embedding": false}' \
--data-frames-preparation-extra-params '{"sample_days": 8, "column_stratification": "SessionID"}' \
--early-stopping-min-delta 0.0001 \
--negative-proportion 0.5 \
--test-size 0.01 \
--test-split-type time \
--dataset-split-method column \
--learning-rate 0.001 \
--metrics='["loss", "acc"]' \
--generator-workers 10  \
--run-evaluate  \
--sample-size-eval 1000 \
--batch-size 128 \
--epochs 30



PYTHONPATH="."  luigi  \
--module train TripletTraining  \
--project globo.config.triplet_globo  \
--recommender-module-class model.TripletNet  \
--recommender-extra-params '{"n_factors": 100, "use_normalize": true, "negative_random": 0, "dropout": 0.0}'  \
--data-frames-preparation-extra-params '{"column_stratification": "SessionID", "sample_days": 8, "min_itens_interactions": 10, "max_relative_pos": 5 }'  \
--loss-function-params '{"swap": true, "margin": 1}'  \
--optimizer adam \
--learning-rate 1e-3  \
--early-stopping-min-delta 0.0001  \
--early-stopping-patience 10  \
--test-size 0.01  \
--test-split-type time  \
--dataset-split-method column  \
--metrics='["loss","triplet_dist", "triplet_acc"]'  \
--save-item-embedding-tsv  \
--local-scheduler  \
--batch-size 128  \
--generator-workers 10  \
--epochs 35  \
--obs ""








mars-gym run supervised \
--project globo.config.sample_globo_with_negative_sample \
--recommender-module-class model.DotModel \
--recommender-extra-params '{"n_factors": 100, "path_item_embedding": false, "freeze_embedding": false}' \
--data-frames-preparation-extra-params '{"sample_days": 8, "column_stratification": "SessionID"}' \
--load-index-mapping-path '/media/workspace/triplet_session/output/models/TripletTraining/results/TripletTraining____mars_gym_model_b____eafd46d19b' \
--early-stopping-min-delta 0.0001 \
--negative-proportion 0.5 \
--test-size 0.01 \
--test-split-type time \
--dataset-split-method column \
--learning-rate 0.001 \
--metrics='["loss", "acc"]' \
--generator-workers 10  \
--run-evaluate  \
--sample-size-eval 1000 \
--batch-size 128 \
--epochs 30

mars-gym run supervised \
--project globo.config.sample_globo_with_negative_sample \
--recommender-module-class model.DotModel \
--recommender-extra-params '{"n_factors": 100, "freeze_embedding": false, "path_item_embedding": "/media/workspace/triplet_session/output/models/TripletTraining/results/TripletTraining____mars_gym_model_b____eafd46d19b/item_embeddings.npy"}' \
--data-frames-preparation-extra-params '{"sample_days": 16, "column_stratification": "SessionID"}' \
--load-index-mapping-path '/media/workspace/triplet_session/output/models/TripletTraining/results/TripletTraining____mars_gym_model_b____eafd46d19b' \
--early-stopping-min-delta 0.0001 \
--negative-proportion 0.5 \
--test-size 0.01 \
--test-split-type time \
--dataset-split-method column \
--learning-rate 0.001 \
--metrics='["loss", "acc"]' \
--generator-workers 10  \
--run-evaluate  \
--sample-size-eval 1000 \
--batch-size 128 \
--epochs 30

mars-gym run supervised \
--project globo.config.sample_globo_with_negative_sample \
--recommender-module-class model.DotModel \
--recommender-extra-params '{"n_factors": 100,  "freeze_embedding": true, "path_item_embedding": "/media/workspace/triplet_session/output/models/TripletTraining/results/TripletTraining____mars_gym_model_b____eafd46d19b/item_embeddings.npy"}' \
--data-frames-preparation-extra-params '{"sample_days": 16, "column_stratification": "SessionID"}' \
--load-index-mapping-path '/media/workspace/triplet_session/output/models/TripletTraining/results/TripletTraining____mars_gym_model_b____eafd46d19b' \
--early-stopping-min-delta 0.0001 \
--negative-proportion 0.5 \
--test-size 0.01 \
--test-split-type time \
--dataset-split-method column \
--learning-rate 0.001 \
--metrics='["loss", "acc"]' \
--generator-workers 10  \
--run-evaluate  \
--sample-size-eval 1000 \
--batch-size 128 \
--epochs 30


PYTHONPATH="."  luigi  \
--module globo.train TripletTraining  \
--project globo.config.triplet_globo  \
--recommender-module-class globo.model.TripletNet  \
--recommender-extra-params '{"n_factors": 100, "use_normalize": true, "negative_random": 0, "dropout": 0.2}'  \
--data-frames-preparation-extra-params '{"column_stratification": "SessionID", "sample_days": 8, "min_itens_interactions": 10, "max_relative_pos": 5 }'  \
--loss-function-params '{"swap": true, "margin": 1}'  \
--optimizer adam \
--learning-rate 1e-3  \
--early-stopping-min-delta 0.0001  \
--early-stopping-patience 10  \
--test-size 0.01  \
--test-split-type time  \
--dataset-split-method column  \
--metrics='["loss","triplet_dist", "triplet_acc"]'  \
--save-item-embedding-tsv  \
--local-scheduler  \
--batch-size 128  \
--generator-workers 10  \
--epochs 35  \
--obs ""


PYTHONPATH="."  luigi  \
--module globo.train TripletTraining  \
--project globo.config.triplet_globo  \
--recommender-module-class globo.model.TripletNet  \
--recommender-extra-params '{"n_factors": 100, "use_normalize": true, "negative_random": 0, "dropout": 0}'  \
--data-frames-preparation-extra-params '{"column_stratification": "SessionID", "sample_days": 8, "min_itens_interactions": 10, "max_relative_pos": 5 }'  \
--loss-function-params '{"swap": true, "margin": 1}'  \
--optimizer adam \
--learning-rate 1e-3  \
--early-stopping-min-delta 0.0001  \
--early-stopping-patience 10  \
--test-size 0.01  \
--test-split-type time  \
--dataset-split-method column  \
--metrics='["loss","triplet_dist", "triplet_acc"]'  \
--save-item-embedding-tsv  \
--local-scheduler  \
--batch-size 128  \
--generator-workers 10  \
--epochs 35  \
--obs ""



PYTHONPATH="."  luigi  \
--module globo.train TripletTraining  \
--project globo.config.triplet_globo  \
--recommender-module-class globo.model.TripletNet  \
--recommender-extra-params '{"n_factors": 100, "use_normalize": true, "negative_random": 0, "dropout": 0}'  \
--data-frames-preparation-extra-params '{"column_stratification": "SessionID", "sample_days": 8, "min_itens_interactions": 2, "max_relative_pos": 5 }'  \
--loss-function-params '{"swap": true, "margin": 1}'  \
--optimizer adam \
--learning-rate 1e-3  \
--early-stopping-min-delta 0.0001  \
--early-stopping-patience 10  \
--test-size 0.01  \
--test-split-type time  \
--dataset-split-method column  \
--metrics='["loss","triplet_dist", "triplet_acc"]'  \
--save-item-embedding-tsv  \
--local-scheduler  \
--batch-size 128  \
--generator-workers 10  \
--epochs 35  \
--obs ""

###########3 Experiment


PYTHONPATH="."  luigi  \
--module train TripletTraining  \
--project globo.config.triplet_globo  \
--recommender-module-class model.TripletNet  \
--recommender-extra-params '{"n_factors": 100, "use_normalize": true, "negative_random": 0, "dropout": 0.5}'  \
--data-frames-preparation-extra-params '{"column_stratification": "SessionID", "sample_days": 2, "min_itens_interactions": 2, "max_relative_pos": 5 }'  \
--loss-function-params '{"swap": true, "margin": 1, "p": 1}'  \
--optimizer adam \
--learning-rate 1e-3  \
--early-stopping-min-delta 0.0001  \
--early-stopping-patience 10  \
--test-size 0.01  \
--test-split-type time  \
--dataset-split-method column  \
--metrics='["loss","triplet_dist", "triplet_acc"]'  \
--save-item-embedding-tsv  \
--local-scheduler  \
--batch-size 128  \
--generator-workers 10  \
--epochs 35  \
--obs ""

mars-gym evaluate supervised --model-task-id SupervisedModelTraining____mars_gym_model_b____e153cc7f7a --only-new-interactions 

