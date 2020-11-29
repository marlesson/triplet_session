
## Submission

### Treinar 5 k_fold models + 1 completo

```bash
mars-gym run supervised \
--project mercado_livre.config.mercado_livre_narm \
--recommender-module-class model.MLNARMModel2 \
--recommender-extra-params '{
  "n_factors": 100, 
  "hidden_size": 200, 
  "dense_size": 19,
  "n_layers": 1, 
  "dropout": 0.2, 
  "history_window": 20, 
  "history_word_window": 3,
  "from_index_mapping": false,
  "path_item_embedding": "~/triplet_session/output/mercado_livre/assets/mercadolivre-100d.bin", 
  "freeze_embedding": true}' \
--data-frames-preparation-extra-params '{
  "sample_days": 60, 
  "history_window": 20, 
  "column_stratification": "SessionID",
  "normalize_dense_features": "min_max",
  "min_interactions": 5,
  "filter_only_buy": true,
  "sample_view": 300000}' \
--optimizer adam \
--optimizer-params '{"weight_decay": 1e-4}' \
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
--epochs 1000 \
--obs "All"
```

```bash
mars-gym run supervised \
--project mercado_livre.config.mercado_livre_narm \
--recommender-module-class model.MLNARMModel2 \
--recommender-extra-params '{
  "n_factors": 100, 
  "hidden_size": 200, 
  "dense_size": 19,
  "n_layers": 1, 
  "dropout": 0.2, 
  "history_window": 20, 
  "history_word_window": 3,
  "from_index_mapping": false,
  "path_item_embedding": "~/triplet_session/output/mercado_livre/assets/mercadolivre-100d.bin", 
  "freeze_embedding": true}' \
--data-frames-preparation-extra-params '{
  "sample_days": 60, 
  "history_window": 20, 
  "column_stratification": "SessionID",
  "normalize_dense_features": "min_max",
  "min_interactions": 5,
  "filter_only_buy": true,
  "sample_view": 300000}' \
--optimizer adam \
--optimizer-params '{"weight_decay": 1e-4}' \
--test-size 0.0 \
--val-size 0.1 \
--early-stopping-min-delta 0.0001 \
--test-split-type random \
--dataset-split-method k_fold \
--n-splits 5 \
--learning-rate 0.001 \
--metrics='["loss"]' \
--generator-workers 5  \
--batch-size 512 \
--loss-function ce \
--epochs 1000 \
--split-index 0 --obs "kfold@0"
```


### Gerar submissão com 100 posições na reclist

Dá um total de 6 arquivos diferentes, ficar atento ao UID do modelo e o UID do normalize file (usar o normalize file do treino total)

```bash
PYTHONPATH="." luigi --module mercado_livre.evaluation EvaluationSubmission \
--local-scheduler \
--model-task-class "mars_gym.simulation.training.SupervisedModelTraining" \
--model-task-id SupervisedModelTraining____mars_gym_model_b____c179ab54fa \
--normalize-file-path "7cef5bca66_std_scaler.pkl" \
--history-window 20 \
--batch-size 1000 \
--percent-limit 1 \
--submission-size 100 \
--model-eval "model" \
--local
```

### Unificar o ensamble com notebook "Rank Ensamble"

Gera 4 arquivos devido as estratégias instant_runoff, borda, dowdall, average_rank
Usar o notebook "Rank Ensamble.ipynb"


### Otimiza o arquivo final com o pos-processamento do dominio, limitando em 10 recomendações. 

Usa o notebook "Submission Pos Process.ipynb"