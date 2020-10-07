# Triple Session

## Triplet

PYTHONPATH="."  luigi \
--module yoochoose.train TripletTraining \
--project yoochoose.config.triplet_yoochoose \
--recommender-module-class yoochoose.model.TripletNet \
--recommender-extra-params '{"n_factors": 100, "use_normalize": true, "negative_random": 0.05}' \
--data-frames-preparation-extra-params '{"sample_limit": 1375000, "min_itens_interactions": 5, "max_relative_pos": 5 }' \
--loss-function-params '{"swap": true}' \
--early-stopping-min-delta 0.00001 \
--early-stopping-patience 20 \
--test-size 0.01 \
--test-split-type time \
--dataset-split-method holdout \
--learning-rate 0.001 \
--metrics='["loss", "triplet_acc"]' \
--save-item-embedding-tsv \
--local-scheduler \
--batch-size 200 \
--generator-workers 10 \
--epochs 100

## Supervised

mars-gym run supervised \
--project yoochoose.config.sample_yoochoose_with_negative_sample \
--recommender-module-class yoochoose.model.SimpleLinearModel \
--recommender-extra-params '{"n_factors": 100, "path_item_embedding": false}' \
--data-frames-preparation-extra-params '{"sample_limit": 1375000}' \
--early-stopping-min-delta 0.0001 \
--negative-proportion 0.5 \
--test-size 0.01 \
--test-split-type time \
--dataset-split-method holdout \
--learning-rate 0.001 \
--metrics='["loss", "acc"]' \
--sample-size-eval 1000 \
--batch-size 200 \
--epochs 30

#####  Without Embs

mars-gym run supervised \
--project yoochoose.config.sample_yoochoose_with_negative_sample \
--recommender-module-class yoochoose.model.SimpleLinearModel \
--recommender-extra-params '{"n_factors": 100, "path_item_embedding": false}' \
--data-frames-preparation-extra-params '{"sample_limit": 1375000}' \
--load-index-mapping-path '/media/workspace/triplet_session/output/models/TripletTraining/results/TripletTraining____mars_gym_model_b____a5870b51c2' \
--early-stopping-min-delta 0.0001 \
--negative-proportion 0.5 \
--test-size 0.01 \
--test-split-type time \
--dataset-split-method holdout \
--learning-rate 0.001 \
--metrics='["loss", "acc"]' \
--sample-size-eval 1000 \
--batch-size 200 \
--epochs 30

mars-gym evaluate supervised --model-task-id SupervisedModelTraining____mars_gym_model_b____9c099cc463 \
--only-new-interactions --local-schedule

mars-gym run supervised \
--project yoochoose.config.sample_yoochoose_with_negative_sample \
--recommender-module-class yoochoose.model.SimpleLinearModel \
--recommender-extra-params '{"n_factors": 100, "path_item_embedding": "/media/workspace/triplet_session/output/models/TripletTraining/results/TripletTraining____mars_gym_model_b____a5870b51c2/item_embeddings.npy"}' \
--data-frames-preparation-extra-params '{"sample_limit": 1375000}' \
--load-index-mapping-path '/media/workspace/triplet_session/output/models/TripletTraining/results/TripletTraining____mars_gym_model_b____a5870b51c2' \
--early-stopping-min-delta 0.0001 \
--negative-proportion 0.5 \
--test-size 0.01 \
--test-split-type time \
--dataset-split-method holdout \
--learning-rate 0.001 \
--metrics='["loss", "acc"]' \
--sample-size-eval 1000 \
--batch-size 200 \
--epochs 30

mars-gym evaluate supervised --model-task-id SupervisedModelTraining____mars_gym_model_b____eee9a60a06 \
--only-new-interactions --local-schedule

SupervisedModelTraining____mars_gym_model_b____71d4bc483b

##########################################################################################

PYTHONPATH="."  luigi \
--module yoochoose.train TripletTraining \
--project yoochoose.config.triplet_yoochoose \
--recommender-module-class yoochoose.model.TripletNet \
--recommender-extra-params '{"n_factors": 50, "use_normalize": true, "negative_random": 0.05}' \
--data-frames-preparation-extra-params '{"sample_limit": 1375000, "min_itens_interactions": 10, "max_relative_pos": 5 }' \
--loss-function-params '{"swap": true, "margin": 100}' \
--early-stopping-min-delta 0.00001 \
--early-stopping-patience 20 \
--test-size 0.01 \
--test-split-type time \
--dataset-split-method holdout \
--learning-rate 0.001 \
--metrics='["loss", "triplet_acc"]' \
--save-item-embedding-tsv \
--local-scheduler \
--batch-size 200 \
--generator-workers 10 \
--epochs 50


PYTHONPATH="."  luigi \
--module yoochoose.train TripletTraining \
--project yoochoose.config.triplet_yoochoose \
--recommender-module-class yoochoose.model.TripletNet \
--recommender-extra-params '{"n_factors": 50, "use_normalize": true, "negative_random": 0.05}' \
--data-frames-preparation-extra-params '{"sample_limit": 1375000, "min_itens_interactions": 10, "max_relative_pos": 5 }' \
--loss-function-params '{"swap": true, "triplet_loss": "bpr_triplet"}' \
--early-stopping-min-delta 0.00001 \
--early-stopping-patience 20 \
--test-size 0.01 \
--test-split-type time \
--dataset-split-method holdout \
--learning-rate 0.001 \
--metrics='["loss", "triplet_acc"]' \
--save-item-embedding-tsv \
--local-scheduler \
--batch-size 200 \
--generator-workers 10 \
--epochs 50

mars-gym evaluate supervised --model-task-id SupervisedModelTraining____mars_gym_model_b____71d4bc483b \
--only-new-interactions --local-schedule

mars-gym evaluate supervised --model-task-id SupervisedModelTraining____mars_gym_model_b____d7145b62e4 \
--only-new-interactions --local-schedule

mars-gym evaluate supervised --model-task-id SupervisedModelTraining____mars_gym_model_b____cb3d453c3d \
--only-new-interactions --local-schedule

mars-gym evaluate supervised --model-task-id SupervisedModelTraining____mars_gym_model_b____eee9a60a06 \
--only-new-interactions --local-schedule

mars-gym evaluate supervised --model-task-id SupervisedModelTraining____mars_gym_model_b____9dcc4f4a86 \
--only-new-interactions --local-schedule


PYTHONPATH="."  luigi \
--module yoochoose.train SupervisedTraining \
--project yoochoose.config.sample_yoochoose_with_negative_sample \
--recommender-module-class yoochoose.model.SimpleLinearModel \
--recommender-extra-params '{"n_factors": 100, "path_item_embedding": false}' \
--early-stopping-min-delta 0.0001 \
--negative-proportion 0.5 \
--data-frames-preparation-extra-params '{"sample_limit": 1375000}' \
--test-size 0.01 \
--learning-rate 0.0001 \
--metrics='["loss", "acc"]' \
--sample-size-eval 1000 \
--batch-size 200 \
--epochs 50 \
--local-scheduler

PYTHONPATH="."  luigi \
--module mars_gym.evaluation.task EvaluateTestSetPredictions \
--model-task-class yoochoose.train.SupervisedTraining \
--model-task-id SupervisedTraining____mars_gym_model_b____ee693214ca \
--only-new-interactions --local-schedule


PYTHONPATH="."  luigi \
--module yoochoose.train SupervisedTraining \
--project yoochoose.config.sample_yoochoose_with_negative_sample \
--recommender-module-class yoochoose.model.SimpleLinearModel \
--recommender-extra-params '{"n_factors": 100, "path_item_embedding": false}' \
--early-stopping-min-delta 0.0001 \
--negative-proportion 0.5 \
--data-frames-preparation-extra-params '{"sample_limit": 1375000}' \
--load-index-mapping-path '/media/workspace/triplet_session/output/models/TripletTraining/results/TripletTraining____mars_gym_model_b____1e1c519ae5' \
--test-size 0.01 \
--learning-rate 0.0001 \
--metrics='["loss", "acc"]' \
--sample-size-eval 1000 \
--batch-size 200 \
--epochs 50 \
--local-scheduler

PYTHONPATH="."  luigi \
--module mars_gym.evaluation.task EvaluateTestSetPredictions \
--model-task-class yoochoose.train.SupervisedTraining \
--model-task-id SupervisedTraining____mars_gym_model_b____ee693214ca \
--only-new-interactions --local-schedule


PYTHONPATH="."  luigi \
--module yoochoose.train SupervisedTraining \
--project yoochoose.config.sample_yoochoose_with_negative_sample \
--recommender-module-class yoochoose.model.SimpleLinearModel \
--recommender-extra-params '{"n_factors": 100, "path_item_embedding": "/media/workspace/triplet_session/output/models/TripletTraining/results/TripletTraining____mars_gym_model_b____1e1c519ae5/item_embeddings.npy"}' \
--early-stopping-min-delta 0.0001 --negative-proportion 0.5 \
--data-frames-preparation-extra-params '{"sample_limit": 1375000}' \
--load-index-mapping-path '/media/workspace/triplet_session/output/models/TripletTraining/results/TripletTraining____mars_gym_model_b____1e1c519ae5' \
--test-size 0.01 \
--learning-rate 0.0001 \
--metrics='["loss", "acc"]' \
--sample-size-eval 1000 \
--batch-size 200 \
--epochs 50 \
--local-scheduler

PYTHONPATH="."  luigi \
--module mars_gym.evaluation.task EvaluateTestSetPredictions \
--model-task-class yoochoose.train.SupervisedTraining \
--model-task-id SupervisedTraining____mars_gym_model_b____f5ac8f04e4 \
--only-new-interactions --local-schedule

PYTHONPATH="."  luigi \
--module mars_gym.evaluation.task EvaluateTestSetPredictions \
--model-task-class yoochoose.train.SupervisedTraining \
--model-task-id SupervisedTraining____mars_gym_model_b____0e8fc4018f \
--only-new-interactions --local-schedule

