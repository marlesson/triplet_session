
from mars_gym.simulation.training import SupervisedModelTraining
from yoochoose.loss import RelativeTripletLoss
import torch
import torch.nn as nn
import luigi
import numpy as np
from typing import Type, Dict, List, Optional, Tuple, Union, Any, cast
from mars_gym.utils.files import (
    get_index_mapping_path,
)
TORCH_LOSS_FUNCTIONS = dict(
    mse=nn.MSELoss,
    nll=nn.NLLLoss,
    bce=nn.BCELoss,
    mlm=nn.MultiLabelMarginLoss,
    relative_triplet=RelativeTripletLoss,    
)


class SupervisedTraining(SupervisedModelTraining):
    load_index_mapping_path: str = luigi.Parameter(default=None)

    @property
    def index_mapping_path(self) -> Optional[str]:
        if self.load_index_mapping_path:
            return get_index_mapping_path(self.load_index_mapping_path)
        return get_index_mapping_path(self.output().path)

class TripletTraining(SupervisedModelTraining):
    loss_function:  str = luigi.ChoiceParameter(choices=["relative_triplet"], default="relative_triplet")
    save_item_embedding_tsv: bool = luigi.BoolParameter(default=False)

    def after_fit(self):
        if self.save_item_embedding_tsv:
            print("save_item_embedding_tsv...")
            self.export_embs()

    def export_embs(self):
        module = self.get_trained_module()

        item_embeddings: np.ndarray = module.item_embeddings.weight.data.cpu().numpy()
        np.savetxt(self.output().path+"/item_embeddings.npy", item_embeddings, delimiter="\t")

    def _get_loss_function(self):
        return TORCH_LOSS_FUNCTIONS[self.loss_function](**self.loss_function_params)
