
# Fatos

* De 30 dias para 6 meses houve melhora no modelo
* Usar apenas os daos de interacao buy reduz muito o score no ML
* Embeddings normalizados ajudam a melhorar a curva de loss mas reduzem um pouco a performance. A mascara tambem reduziu um pouco a peformance


# Historico


# SupervisedModelTraining____mars_gym_model_b____b92d68b8b7

Todos os dados e com sessão de view e buy no treinamento
submission_SupervisedModelTraining____mars_gym_model_b____b92d68b8b7_831bd03858_sub
-> 'ML': 0.20489	


# SupervisedModelTraining____mars_gym_model_b____2f2f2de57c

Apenas os dados de 30 dias e com sessão de view e buy no treinamento
submission_SupervisedModelTraining____mars_gym_model_b____2f2f2de57c_96f794e191_sub
-> 'ndcg_at_10': 0.7134313311016469
-> 'ML': 0.20115966524646176


# SupervisedModelTraining____mars_gym_model_b____c549ab3480

APenas os dados de 30 dias e apenas as interacoes de buy (mudou o test-dataset pois 
está usando apenas as interações de buy também)
SupervisedModelTraining____mars_gym_model_b____c549ab3480_f2846c540c_sub
-> 'ndcg_at_10': 0.5477472675988672
-> 'ML': 0.164470


# SupervisedModelTraining____mars_gym_model_b____31de201200

APenas os dados de 30 dias e todas as interações de session (mudou o test-dataset pois 
está usando apenas as interações de buy também)
SupervisedModelTraining____mars_gym_model_b____31de201200_c6dbff6d5f_sub
-> 'ndcg_at_10': 0.696523562786302,
-> 'ML': 0.2011596
