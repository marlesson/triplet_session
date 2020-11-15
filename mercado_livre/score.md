
# Fatos

* De 30 dias para 6 meses houve melhora no modelo
* Usar apenas os daos de interacao buy reduz muito o score no ML
* Embeddings normalizados ajudam a melhorar a curva de loss mas reduzem um pouco a performance. A mascara tambem reduziu um pouco a peformance


# Historico


# SupervisedModelTraining____mars_gym_model_b____2ff0b99c0d_32c1432eff_sub

Apenas buy

-> 'ML': 0.23492	

{
    "model_task": "SupervisedModelTraining____mars_gym_model_b____2ff0b99c0d",
    "count": 1870,
    "mean_average_precision": 0.5127539920130239,
    "precision_at_1": 0.4443850267379679,
    "mrr_at_5": 0.4903565062388592,
    "mrr_at_10": 0.5004483914778032,
    "ndcg_at_5": 0.5358896178223215,
    "ndcg_at_10": 0.5621532686566242,
    "ndcg_at_15": 0.5764143386237068,
    "ndcg_at_20": 0.584676081311685,
    "ndcg_at_50": 0.6140391120052096,
    "coverage_at_5": 0.12029999999999999,
    "coverage_at_20": 0.3846
}

# SupervisedModelTraining____mars_gym_model_b____1f6210ddd3_4e69a83073_sub

Apenas buy and uso do history domain 

-> ML: 0.23956

 {'count': 1870,
  'coverage_at_20': 0.4022,
  'coverage_at_5': 0.1257,
  'mean_average_precision': 0.5443134518114011,
  'model_task': 'SupervisedModelTraining____mars_gym_model_b____1f6210ddd3',
  'mrr_at_10': 0.5331156098803157,
  'mrr_at_5': 0.5214795008912656,
  'ndcg_at_10': 0.6042521766040143,
  'ndcg_at_15': 0.6173788813539623,
  'ndcg_at_20': 0.6271319979086323,
  'ndcg_at_5': 0.5739731766883653,
  'ndcg_at_50': 0.6520301006193968,
  'precision_at_1': 0.46844919786096256}