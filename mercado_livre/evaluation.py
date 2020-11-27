from mars_gym.evaluation.policy_estimator import PolicyEstimatorTraining
from mars_gym.torch.data import FasterBatchSampler, NoAutoCollationDataLoader
from mars_gym.utils.reflection import load_attr, get_attribute_names
from mars_gym.utils.utils import parallel_literal_eval, JsonEncoder
from mars_gym.utils.index_mapping import (
    create_index_mapping,
    create_index_mapping_from_arrays,
    transform_with_indexing,
    map_array,
)
import functools

from multiprocessing.pool import Pool
from mars_gym.evaluation.task import BaseEvaluationTask
from mercado_livre.data import PreProcessSessionTestDataset, SessionPrepareTestDataset, SessionPrepareLocalTestDataset
import abc
from typing import Type, Dict, List, Optional, Tuple, Union, Any, cast
from torch.utils.data import DataLoader
from mars_gym.torch.data import NoAutoCollationDataLoader, FasterBatchSampler
from torchbearer import Trial
from mercado_livre.data import SessionInteractionDataFrame
import gc
import luigi
import pandas as pd
import numpy as np
import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from mars_gym.cuda import CudaRepository
import torchbearer
from tqdm import tqdm
from mars_gym.data.dataset import (
    preprocess_interactions_data_frame,
    InteractionsDataset,
)
from mars_gym.utils.index_mapping import (
    transform_with_indexing,
)
from mars_gym.data.dataset import (
    preprocess_interactions_data_frame,
    InteractionsDataset,
)
from mars_gym.evaluation.metrics.rank import (
    mean_reciprocal_rank,
    average_precision,
    precision_at_k,
    ndcg_at_k,
    personalization_at_k,
    prediction_coverage_at_k,
)
from mars_gym.utils.utils import parallel_literal_eval, JsonEncoder
import pprint
import json
import luigi
import pandas as pd
import functools
import numpy as np
from tqdm import tqdm
import os
from multiprocessing.pool import Pool
from scipy import stats

ITEM_META_PATH = "/media/workspace/triplet_session/output/mercado_livre/dataset/item__processed.csv"


def ndcg_ml(r):
    pos = np.array(range(len(r))) + 1
    
    reclist_i = np.ones(len(r))
    reclist_i[0] = 12

    dcg  = r/np.log(1+pos)
    dcgi = reclist_i/np.log(1+pos)
    
    ndcg = np.sum(dcg)/np.sum(dcgi)

    return ndcg

def _sort_rank_list(score, index_mapping):
    # UNK, PAD, PAD
    score[0] = score[1] = score[2] = 0

    item_idx  = np.argsort(score)[::-1][:100]
    
    item_id   = [int(index_mapping[item]) for item in item_idx]
    
    return item_id

def _get_domain(arr, domain_map):
    return list(map(domain_map.get, arr))

def _get_moda(arr):
    try:
        return stats.mode(arr)[0][0]
    except:
        return 0 

def _get_count_moda(arr):
    try:
        return stats.mode(arr)[1][0]/len(arr)
    except:
        return 0     

def _create_relevance_list(sorted_actions, expected_action):
    return [1 if str(action) == str(expected_action) else 0 for action in sorted_actions]

def _create_relevance_list_domain(sorted_actions, expected_action):
    s = []
    for i in range(len(sorted_actions)):
        if str(sorted_actions[i]) == str(expected_action):
            s.append(i + 0)
        else:
            s.append(i + 100)
    return s

def _create_relevance_list_ml(reclist, domainlist, item_id, domain_id):
    s = []
    for i in range(len(reclist)):
        if str(reclist[i]) == str(item_id):
            value = 12
        elif str(domainlist[i]) == str(domain_id):
            value = 1
        else:
            value = 0
        s.append(value)
    return s    

def _sorte_by_domain_moda(reclist, relevance_list, percent_limit, limit):
    if percent_limit >= limit:
        return [x for _,x in sorted(zip(relevance_list, reclist), reverse=False)]
    else:
        return reclist


# PYTHONPATH="." luigi --module mercado_livre.evaluation MLEvaluationTask \
# --model-task-class "mars_gym.simulation.training.SupervisedModelTraining" \
# --model-task-id SupervisedModelTraining____mars_gym_model_b____e3ae64b091 \
# --normalize-file-path "226cbf7ae2_std_scaler.pkl" \
# --history-window 20 \
# --batch-size 1000 \
# --local-scheduler \
# --file "/media/workspace/triplet_session/output/mercado_livre/dataset/test_0.10_test=random_42_SessionInteractionDataFrame_____SessionID_226cbf7ae2.csv"


class MLEvaluationTask(BaseEvaluationTask):
    model_task_class: str = luigi.Parameter(
        default="mars_gym.simulation.training.SupervisedModelTraining"
    )
    model_task_id: str = luigi.Parameter()
    offpolicy_eval: bool = luigi.BoolParameter(default=False)
    task_hash: str = luigi.Parameter(default="sub")
    generator_workers: int = luigi.IntParameter(default=0)
    pin_memory: bool = luigi.BoolParameter(default=False)
    batch_size: int = luigi.IntParameter(default=100)
    device: str = luigi.ChoiceParameter(choices=["cpu", "cuda"], default="cuda")
    history_window: int = luigi.IntParameter(default=30)
    normalize_dense_features: int = luigi.Parameter(default="min_max")
    normalize_file_path: str = luigi.Parameter(default=None)
    local: bool = luigi.BoolParameter(default=False)
    most_popular: bool = luigi.BoolParameter(default=False)

    sample_size: int = luigi.Parameter(default=1000)
    percent_limit: float = luigi.FloatParameter(default=0.2)

    @property
    def task_name(self):
        return self.model_task_id + "_" + self.task_id.split("_")[-1] + "_sub_" + str(self.local)

    def requires(self):

        if self.local:
            return SessionPrepareLocalTestDataset(history_window=self.history_window)
        else:
            return SessionPrepareTestDataset(history_window=self.history_window)

    @property
    def torch_device(self) -> torch.device:
        if not hasattr(self, "_torch_device"):
            if self.device == "cuda":
                self._torch_device = torch.device(f"cuda:{self.device_id}")
            else:
                self._torch_device = torch.device("cpu")
        return self._torch_device

    @property
    def device_id(self):
        if not hasattr(self, "_device_id"):
            if self.device == "cuda":
                self._device_id = CudaRepository.get_avaliable_device()
            else:
                self._device_id = None
        return self._device_id

    def get_test_generator(self, df) -> Optional[DataLoader]:

        dataset = InteractionsDataset(
            data_frame=df,
            embeddings_for_metadata=self.model_training.embeddings_for_metadata,
            project_config=self.model_training.project_config,
            index_mapping=self.model_training.index_mapping
        )

        batch_sampler = FasterBatchSampler(
            dataset, self.batch_size, shuffle=False
        )

        return NoAutoCollationDataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=self.generator_workers,
            pin_memory=self.pin_memory if self.device == "cuda" else False,
        )

    def pos_process(self, rank_list):
        df_item     = pd.read_csv(ITEM_META_PATH, usecols=["item_id", "domain_id", "domain_idx"]).fillna(0)#.head(10)
        domain_map  = df_item[['item_id', 'domain_idx']].set_index("item_id").to_dict()["domain_idx"]

        with Pool(os.cpu_count()) as p:
            _map_domain = list(tqdm(
                p.map(functools.partial(_get_domain, domain_map=domain_map), rank_list),
                total=len(rank_list),
            ))  
            
        arr_moda = list(zip(list(rank_list), 
                            list(_map_domain),  
                            list(map(_get_moda, _map_domain)), 
                            list(map(_get_count_moda, _map_domain))))

        df_moda = pd.DataFrame(arr_moda, columns=["reclist", "domainlist", "domain_moda", "count"])

        df_moda['relevance_list'] = df_moda.apply(lambda row: 
                                                _create_relevance_list_domain(row['domainlist'], row['domain_moda']),  
                                                axis=1)
#_create_relevance_list_ml

        df_moda['reclist_2'] = df_moda.apply(lambda row: _sorte_by_domain_moda(
                        row['reclist'], 
                        row['relevance_list'], 
                        row['count'], self.percent_limit)[:10],  axis=1)
        

        df_moda['domainlist_2'] = df_moda.apply(lambda row: _sorte_by_domain_moda(
                        row['domainlist'], 
                        row['relevance_list'], 
                        row['count'], self.percent_limit)[:10],  axis=1)



        return df_moda

    def run(self):
        os.makedirs(self.output().path)
    
        df: pd.DataFrame = pd.read_parquet(self.input()[1].path)
    
        df = preprocess_interactions_data_frame(
            df, 
            self.model_training.project_config
        )

        data = SessionInteractionDataFrame(
                item_column="",
                normalize_dense_features=self.normalize_dense_features,
                normalize_file_path=self.normalize_file_path)
        
        data.transform_data_frame(df, "TEST_GENERATOR")

        # Index test dataset 
        df['Index'] = df['SessionID'].astype(int)
        df = df.sort_values("Index")

        df.to_csv(self.output().path+"/dataset.csv")

        transform_with_indexing(
            df, 
            self.model_training.index_mapping, 
            self.model_training.project_config
        )
        # 
        df.to_csv(self.output().path+"/dataset_indexed.csv")
        generator = self.get_test_generator(df)

        print(df.head())
        print(df.shape)

        # Gente Model
        model = self.model_training.get_trained_module()
        model.to(self.torch_device)
        model.eval()

        scores = []
        rank_list = []
        #with Pool(os.cpu_count()) as p:
        #    list(tqdm(p.starmap(_get_rank_list, )))

        reverse_index_mapping    = self.model_training.reverse_index_mapping['ItemID']
        reverse_index_mapping[1] = 0


        # Inference
        with torch.no_grad():
            for i, (x, _) in tqdm(enumerate(generator), total=len(generator)):
                input_params = x if isinstance(x, list) or isinstance(x, tuple) else [x]
                input_params = [t.to(self.torch_device) if isinstance(t, torch.Tensor) else t for t in input_params]

                scores_tensor: torch.Tensor  = model(*input_params)
                scores_batch = scores_tensor.detach().cpu().numpy()
                #scores.extend(scores_batch)

                # Test
                _sort_rank_list(scores_batch[0], index_mapping=reverse_index_mapping)

                with Pool(3) as p:
                    _rank_list = list(tqdm(
                        p.map(functools.partial(_sort_rank_list, index_mapping=reverse_index_mapping), scores_batch),
                        total=len(scores_batch),
                    ))
                    rank_list.extend(_rank_list)

                gc.collect()

        df_moda  = self.pos_process(rank_list)
        rank_list = df_moda['reclist_2'].values
        rank_list = np.array([np.array(r).astype(int) for r in rank_list])

        #from IPython import embed; embed()
        df_moda.to_csv(self.output().path+'/df_submission.csv', index=False)
        np.savetxt(self.output().path+'/submission_{}.csv'.format(self.task_name), rank_list, fmt='%i', delimiter=',') 


# PYTHONPATH="." luigi --module mercado_livre.evaluation EvaluationSubmission \
# --model-task-class "mars_gym.simulation.training.SupervisedModelTraining" \
# --model-task-id SupervisedModelTraining____mars_gym_model_b____e3ae64b091 \
# --normalize-file-path "226cbf7ae2_std_scaler.pkl" \
# --history-window 20 \
# --batch-size 1000 \
# --local-scheduler \
# --file "/media/workspace/triplet_session/output/mercado_livre/dataset/test_0.10_test=random_42_SessionInteractionDataFrame_____SessionID_226cbf7ae2.csv"

class EvaluationSubmission(luigi.Task):
    model_task_class: str = luigi.Parameter(
        default="mars_gym.simulation.training.SupervisedModelTraining"
    )
    model_task_id: str = luigi.Parameter()
    offpolicy_eval: bool = luigi.BoolParameter(default=False)
    task_hash: str = luigi.Parameter(default="sub")
    generator_workers: int = luigi.IntParameter(default=0)
    pin_memory: bool = luigi.BoolParameter(default=False)
    batch_size: int = luigi.IntParameter(default=100)
    device: str = luigi.ChoiceParameter(choices=["cpu", "cuda"], default="cuda")
    history_window: int = luigi.IntParameter(default=30)
    normalize_dense_features: int = luigi.Parameter(default="min_max")
    normalize_file_path: str = luigi.Parameter(default=None)
    local: bool = luigi.BoolParameter(default=False)
    sample_size: int = luigi.Parameter(default=1000)
    percent_limit: float = luigi.FloatParameter(default=0.4)
    most_popular: bool = luigi.BoolParameter(default=False)

    def requires(self):
        return MLEvaluationTask(model_task_class=self.model_task_class,
                                model_task_id=self.model_task_id,
                                normalize_dense_features=self.normalize_dense_features,
                                normalize_file_path=self.normalize_file_path,
                                batch_size=self.batch_size,
                                history_window=self.history_window,
                                local=self.local,
                                most_popular=self.most_popular,
                                sample_size=self.sample_size,
                                percent_limit=self.percent_limit), SessionPrepareLocalTestDataset(history_window=self.history_window)
    

    def output(self):
        return luigi.LocalTarget(os.path.join(self.input()[0].path, "metrics.json"))

    def run(self):
        df: pd.DataFrame = pd.read_parquet(self.input()[1][1].path)#.sample(n=self.sample_size, random_state=42, replace=True)#, usecols=['ItemID']).sample(n=self.sample_size, random_state=42, replace=True)
        arr_sub: pd.DataFrame = pd.read_csv(self.input()[0].path+'/submission_{}.csv'.format(self.requires()[0].task_name), header=None)
        df_sub: pd.DataFrame = pd.read_csv(self.input()[0].path+'/df_submission.csv')

        df['reclist']        = list(arr_sub.values)
        df['domainlist']     = list(df_sub.domainlist_2.apply(eval))
        df['relevance_list'] = df.apply(lambda row: _create_relevance_list(row['reclist'], row['ItemID']),  axis=1)
        df['relevance_list_ml'] = df.apply(lambda row: _create_relevance_list_ml(row['reclist'], row['domainlist'], row['ItemID'], row['domain_idx']),  axis=1)


        with Pool(os.cpu_count()) as p:
            print("Calculating average precision...")
            df["average_precision"] = list(
                tqdm(p.map(average_precision, df["relevance_list"]), total=len(df))
            )

            print("Calculating precision at 1...")
            df["precision_at_1"] = list(
                tqdm(
                    p.map(functools.partial(precision_at_k, k=1), df["relevance_list"]),
                    total=len(df),
                )
            )

            print("Calculating MRR at 5 ...")
            df["mrr_at_5"] = list(
                tqdm(
                    p.map(functools.partial(mean_reciprocal_rank, k=5), df["relevance_list"]),
                    total=len(df),
                )
            )


            print("Calculating MRR at 10 ...")
            df["mrr_at_10"] = list(
                tqdm(
                    p.map(functools.partial(mean_reciprocal_rank, k=10), df["relevance_list"]),
                    total=len(df),
                )
            )

            print("Calculating nDCG at 5...")
            df["ndcg_at_5"] = list(
                tqdm(
                    p.map(functools.partial(ndcg_at_k, k=5), df["relevance_list"]),
                    total=len(df),
                )
            )
            print("Calculating nDCG at 10...")
            df["ndcg_at_10"] = list(
                tqdm(
                    p.map(functools.partial(ndcg_at_k, k=10), df["relevance_list"]),
                    total=len(df),
                )
            )
            print("Calculating nDCG at 15...")
            df["ndcg_at_15"] = list(
                tqdm(
                    p.map(functools.partial(ndcg_at_k, k=15), df["relevance_list"]),
                    total=len(df),
                )
            )
            print("Calculating nDCG at 20...")
            df["ndcg_at_20"] = list(
                tqdm(
                    p.map(functools.partial(ndcg_at_k, k=20), df["relevance_list"]),
                    total=len(df),
                )
            )
            print("Calculating nDCG at 50...")
            df["ndcg_at_50"] = list(
                tqdm(
                    p.map(functools.partial(ndcg_at_k, k=50), df["relevance_list"]),
                    total=len(df),
                )
            )

            print("Calculating nDCGML...")
            df["ndcg_ml"] = list(
                tqdm(
                    p.map(functools.partial(ndcg_ml), df["relevance_list_ml"]),
                    total=len(df),
                )
            )
            

        metrics = {
            "model_task": self.model_task_id,
            "percent_limit": self.percent_limit,
            "count": len(df),
            "mean_average_precision": df["average_precision"].mean(),
            "precision_at_1": df["precision_at_1"].mean(),
            "mrr_at_5": df["mrr_at_5"].mean(),
            "mrr_at_10": df["mrr_at_10"].mean(),
            "ndcg_at_5": df["ndcg_at_5"].mean(),
            "ndcg_at_10": df["ndcg_at_10"].mean(),
            "ndcg_at_15": df["ndcg_at_15"].mean(),
            "ndcg_at_20": df["ndcg_at_20"].mean(),
            "ndcg_at_50": df["ndcg_at_50"].mean(),
            "ndcg_ml": df["ndcg_ml"].mean(),
        }
        pprint.pprint(metrics)
        
        df.to_csv(os.path.join(self.input()[0].path, "eval_dataset.csv"))
        with open(
            os.path.join(self.input()[0].path, "metrics.json"), "w"
        ) as metrics_file:
            json.dump(metrics, metrics_file, cls=JsonEncoder, indent=4)

