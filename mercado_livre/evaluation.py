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
from mars_gym.evaluation.task import BaseEvaluationTask
from mercado_livre.data import PreProcessSessionTestDataset, SessionPrepareTestDataset
import abc
from typing import Type, Dict, List, Optional, Tuple, Union, Any, cast
from torch.utils.data import DataLoader
from mars_gym.torch.data import NoAutoCollationDataLoader, FasterBatchSampler
from torchbearer import Trial

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

# PYTHONPATH="." luigi --module mercado_livre.evaluation MLEvaluationTask \
# --model-task-class "mars_gym.simulation.training.SupervisedModelTraining" \
# --model-task-id SupervisedModelTraining____mars_gym_model_b____b92d68b8b7 \
# --local-scheduler
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
    history_window: int = luigi.IntParameter(default=10)

    @property
    def task_name(self):
        return self.model_task_id + "_" + self.task_id.split("_")[-1] + "_sub"

    def requires(self):
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

    def run(self):
        os.makedirs(self.output().path)
        df: pd.DataFrame = pd.read_csv(self.input().path)

        # Index test dataset 
        df['Index'] = df['SessionID']
        print(df.head())

        df = preprocess_interactions_data_frame(
            df, 
            self.model_training.project_config
        )
        transform_with_indexing(
            df, 
            self.model_training.index_mapping, 
            self.model_training.project_config
        )
        df = df.sort_values("Index")

        print(df.head())
        print(df.shape)
        
        generator = self.get_test_generator(df)


        # Gente Model
        model = self.model_training.get_trained_module()
        model.to(self.torch_device)
        model.eval()

        scores = []
        rank_list = []
        #with Pool(os.cpu_count()) as p:
        #    list(tqdm(p.starmap(_get_rank_list, )))

        reverse_index_mapping = self.model_training.reverse_index_mapping['ItemID']
        reverse_index_mapping[1] = 0

        # Inference
        with torch.no_grad():
            for i, (x, _) in tqdm(enumerate(generator), total=len(generator)):
                
                input_params = x if isinstance(x, list) or isinstance(x, tuple) else [x]
                input_params = [t.to(self.torch_device) for t in input_params]

                scores_tensor: torch.Tensor  = model(*input_params)
                scores_batch = scores_tensor.detach().cpu().numpy()
                #scores.extend(scores_batch)


                for score in tqdm(scores_batch, total=len(scores_batch)):
                    item_idx  = np.argsort(score)[::-1][:10]
                    #from IPython import embed; embed()
                    item_id   = [int(reverse_index_mapping[item]) for item in item_idx]
                    rank_list.append(item_id)

                gc.collect()
        np.savetxt(self.output().path+'/submission_{}.csv'.format(self.task_name), np.array(rank_list).astype(int), fmt='%i', delimiter=',') 

                # scores_tensor: torch.Tensor  = model.recommendation_score(*input_params)
                # scores_batch: List[float] = scores_tensor.cpu().numpy().reshape(-1).tolist()
                # scores.extend(scores_batch)
                
                # for i in range(self.model_training.n_items):
                #     input_params[1] = torch.IntTensor([i] * self.batch_size).to(self.torch_device).long()
                #     scores_tensor: torch.Tensor  = model.recommendation_score(*input_params)
                #     scores_batch: List[float] = scores_tensor.cpu().numpy().reshape(-1).tolist()
                #     scores.extend(scores_batch)
        
        # trial = (
        #     Trial(
        #         model,
        #         criterion=lambda *args: torch.zeros(
        #             1, device=self.torch_device, requires_grad=True
        #         ),
        #     )
        #     .with_generators(val_generator=test_loader)
        #     .to(self.torch_device)
        #     .eval()
        # )

        # with torch.no_grad():
        #     model_output: Union[torch.Tensor, Tuple[torch.Tensor]] = trial.predict(
        #         verbose=0, data_key=torchbearer.VALIDATION_DATA
        #     )

        # scores_tensor: torch.Tensor = model_output if isinstance(
        #     model_output, torch.Tensor
        # ) else model_output[0][0]
        # scores: List[float] = scores_tensor.cpu().numpy().reshape(-1).tolist()
        

        # rank_list = []
        # #with Pool(os.cpu_count()) as p:
        # #    list(tqdm(p.starmap(_get_rank_list, )))

        # reverse_index_mapping = self.model_training.reverse_index_mapping['ItemID']

        # for score in tqdm(scores, total=len(scores)):
        #     item_idx  = np.argsort(score)[:10]
        #     item_id   = [int(reverse_index_mapping[item]) for item in item_idx]
        #     rank_list.append(item_id)


