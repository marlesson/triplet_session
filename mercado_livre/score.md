
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


# SupervisedModelTraining____mars_gym_model_b____3bc7f59896

Apenas Buy e com uso da CNN com WordEmb, aparentemente ficou bom na validação mas não refletiu no score. 

-> ML: 0.2031

{'count': 1857,
 'coverage_at_20': 0.4061,
 'coverage_at_5': 0.1276,
 'mean_average_precision': 0.5660747253273697,
 'model_task': 'SupervisedModelTraining____mars_gym_model_b____3bc7f59896',
 'mrr_at_10': 0.5553912266755563,
 'mrr_at_5': 0.5437443905941483,
 'ndcg_at_10': 0.6287461556465984,
 'ndcg_at_15': 0.6417145415780894,
 'ndcg_at_20': 0.6499002693440401,
 'ndcg_at_5': 0.5988114726222036,
 'ndcg_at_50': 0.6748772862328072,
 'precision_at_1': 0.4878836833602585}


# SupervisedModelTraining____mars_gym_model_b____c96004c977

Apenas Buy e sem a CNN, 

-> ML: 0.19766

{
    "model_task": "SupervisedModelTraining____mars_gym_model_b____c96004c977",
    "count": 1857,
    "mean_average_precision": 0.5551107849679379,
    "precision_at_1": 0.4771136241249327,
    "mrr_at_5": 0.5325076287919583,
    "mrr_at_10": 0.5435300578676991,
    "ndcg_at_5": 0.5872703080903975,
    "ndcg_at_10": 0.6159378120004624,
    "ndcg_at_15": 0.6317353551524754,
    "ndcg_at_20": 0.640325819491581,
    "ndcg_at_50": 0.6645987814806446,
    "coverage_at_5": 0.12689999999999999,
    "coverage_at_20": 0.4008
}

# SupervisedModelTraining____mars_gym_model_b____dbab7e0e22

MOdelo com MLP e variaveis de last item, last module e NLP. Usou apenas 
Buy e performou bem no test, mas no ML foi uma merda

{
    "model_task": "SupervisedModelTraining____mars_gym_model_b____dbab7e0e22",
    "count": 1857,
    "mean_average_precision": 0.6073892904621753,
    "precision_at_1": 0.5228863758750674,
    "mrr_at_5": 0.5954227248249866,
    "mrr_at_10": 0.6033216230308314,
    "ndcg_at_5": 0.666648573879492,
    "ndcg_at_10": 0.6866547375767993,
    "ndcg_at_15": 0.690386212956009,
    "ndcg_at_20": 0.6912965797947,
    "ndcg_at_50": 0.6916513442353528,
    "coverage_at_5": 0.06570000000000001,
    "coverage_at_20": 0.2321
}

-> ML: 0.198032

# SupervisedModelTraining____mars_gym_model_b____abf007aebc

Treinado apenas com buy. Ajuste em todo pipeline, correções. 

{'count': 1889,
 'coverage_at_20': 0.2023,
 'coverage_at_5': 0.0564,
 'mean_average_precision': 0.539698906702632,
 'model_task': 'SupervisedModelTraining____mars_gym_model_b____abf007aebc',
 'mrr_at_10': 0.5348655961414034,
 'mrr_at_5': 0.5320981118757719,
 'ndcg_at_10': 0.5977063967210373,
 'ndcg_at_15': 0.5993016798713945,
 'ndcg_at_20': 0.599563538284374,
 'ndcg_at_5': 0.5908594578479752,
 'ndcg_at_50': 0.5996822486201165,
 'precision_at_1': 0.4722075172048703}

-> ML: 0.1948


# SupervisedModelTraining____mars_gym_model_b____b8d4996188

ML:  0.18399

{'count': 1876,
 'coverage_at_20': 0.1954,
 'coverage_at_5': 0.0567,
 'mean_average_precision': 0.5631568962608224,
 'model_task': 'SupervisedModelTraining____mars_gym_model_b____b8d4996188',
 'mrr_at_10': 0.5520712762717027,
 'mrr_at_5': 0.5374733475479744,
 'ndcg_at_10': 0.6591314029973595,
 'ndcg_at_15': 0.6775182279291796,
 'ndcg_at_20': 0.6888021048555724,
 'ndcg_at_5': 0.6215979154703808,
 'ndcg_at_50': 0.7067737012714879,
 'precision_at_1': 0.4525586353944563}
