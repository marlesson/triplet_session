# Triple Session


## Supervised

mars-gym run supervised \
--project yoochoose.config.sample_yoochoose_with_negative_sample \
--recommender-module-class yoochoose.model.SimpleLinearModel \
--recommender-extra-params '{"n_factors": 100}' \
--early-stopping-min-delta 0.0001 --negative-proportion 0.5 \
--data-frames-preparation-extra-params '{"sample_limit": 1375000}' \
--test-size 0.01 \
--learning-rate 0.001 \
--metrics='["loss", "acc"]' \
--batch-size 500 \
--epochs 30


mars-gym evaluate supervised --model-task-id SupervisedModelTraining____mars_gym_model_b____52b94db5d5 --only-new-interactions

####

PYTHONPATH="."  luigi \
--module yoochoose.train SupervisedTraining \
--project yoochoose.config.sample_yoochoose_with_negative_sample \
--recommender-module-class yoochoose.model.SimpleLinearModel \
--recommender-extra-params '{"n_factors": 100, "path_item_embedding": "/media/workspace/triplet_session/output/models/TripletTraining/results/TripletTraining____mars_gym_model_b____cdb03e12ab/item_embeddings.npy"}' \
--early-stopping-min-delta 0.0001 --negative-proportion 0.5 \
--data-frames-preparation-extra-params '{"sample_limit": 1375000}' \
--load-index-mapping-path '/media/workspace/triplet_session/output/models/TripletTraining/results/TripletTraining____mars_gym_model_b____cdb03e12ab' \
--test-size 0.01 \
--learning-rate 0.001 \
--metrics='["loss", "acc"]' \
--sample-size-eval 1000 \
--batch-size 500 \
--epochs 2 \
--local-scheduler



## Triplet

PYTHONPATH="."  luigi \
--module yoochoose.train TripletTraining \
--project yoochoose.config.triplet_yoochoose \
--recommender-module-class yoochoose.model.TripletNet \
--recommender-extra-params '{"n_factors": 100, "use_normalize": true, "negative_random": 0.05}' \
--early-stopping-min-delta 0.00001 \
--early-stopping-patience 20 \
--data-frames-preparation-extra-params '{"sample_limit": 1375000}' \
--test-size 0.01 \
--learning-rate 0.001 \
--metrics='["loss", "triplet_acc"]' \
--batch-size 500 \
--epochs 1 \
--save-item-embedding-tsv \
--local-scheduler