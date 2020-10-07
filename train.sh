# bin/bash!

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
--load-index-mapping-path '/media/workspace/triplet_session/output/models/TripletTraining/results/TripletTraining____mars_gym_model_b____3255ff25a5' \
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


mars-gym run supervised \
--project yoochoose.config.sample_yoochoose_with_negative_sample \
--recommender-module-class yoochoose.model.SimpleLinearModel \
--recommender-extra-params '{"n_factors": 100, "path_item_embedding": "/media/workspace/triplet_session/output/models/TripletTraining/results/TripletTraining____mars_gym_model_b____3255ff25a5/item_embeddings.npy"}' \
--data-frames-preparation-extra-params '{"sample_limit": 1375000}' \
--load-index-mapping-path '/media/workspace/triplet_session/output/models/TripletTraining/results/TripletTraining____mars_gym_model_b____3255ff25a5' \
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