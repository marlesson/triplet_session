from typing import Dict, Any, List, Tuple, Union
import os
import luigi
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchbearer import Trial
from mars_gym.meta_config import ProjectConfig
from mars_gym.model.abstract import RecommenderModule
from mars_gym.model.bandit import BanditPolicy
from mars_gym.torch.init import lecun_normal_init

from numpy.random.mtrand import RandomState
import random

class SimpleLinearModel(RecommenderModule):
    def __init__(
        self,
        project_config: ProjectConfig,
        index_mapping: Dict[str, Dict[Any, int]],
        n_factors: int,
        path_item_embedding: str
    ):
        super().__init__(project_config, index_mapping)
        self.path_item_embedding = path_item_embedding

        if self.path_item_embedding:
            weights = np.loadtxt(self.path_item_embedding)
            self.item_embeddings = nn.Embedding.from_pretrained(torch.from_numpy(weights).float())              
        else:
            self.item_embeddings = nn.Embedding(self._n_items, n_factors)

        self.user_embeddings = nn.Embedding(self._n_users, n_factors)
        self.dayofweek_embeddings = nn.Embedding(7, n_factors)

        num_dense = n_factors * (1 + 10)

        self.dense = nn.Sequential(
            nn.Linear(num_dense, int(num_dense / 2)),
            nn.SELU(),
            nn.Linear(int(num_dense / 2), 1),
        )

        self.weight_init = lecun_normal_init
        self.apply(self.init_weights)

    def init_weights(self, module: nn.Module):
        if type(module) == nn.Linear:
            self.weight_init(module.weight)
            module.bias.data.fill_(0.1)
            
    def flatten(self, input):
        return input.view(input.size(0), -1)

    def forward(self, session_ids, item_ids, item_history_ids):
        # Item emb
        item_emb = self.item_embeddings(item_ids)

        # Item History embs
        interaction_item_emb = self.flatten(self.item_embeddings(item_history_ids))

        ## DayofWeek Emb
        #dayofweek_emb = self.dayofweek_embeddings(dayofweek.long())

        x = torch.cat(
            (item_emb, interaction_item_emb),
            dim=1,
        )

        out = torch.sigmoid(self.dense(x))
        return out

class TripletNet(RecommenderModule):
    def __init__(
        self,
        project_config: ProjectConfig,
        index_mapping: Dict[str, Dict[Any, int]],
        n_factors: int, use_normalize: bool,
        negative_random: float
    ):

        super().__init__(project_config, index_mapping)

        self.use_normalize   = use_normalize
        self.item_embeddings = nn.Embedding(self._n_items, n_factors)
        self.negative_random = negative_random

        self.weight_init = lecun_normal_init
        self.apply(self.init_weights)
        
    def init_weights(self, module: nn.Module):
        if type(module) == nn.Linear:
            self.weight_init(module.weight)
            module.bias.data.fill_(0.1)

    def normalize(self, x: torch.Tensor, dim: int = 1) -> torch.Tensor:
        if self.use_normalize:
            x = F.normalize(x, p=2, dim=dim)
        return x

    def get_harder_negative(self, item_ids: torch.Tensor, 
                            positive_item_ids: torch.Tensor, 
                            negative_item_list_idx: List[torch.Tensor]) -> Tuple[torch.Tensor]:

        batch_size = item_ids.size(0)

        anchors    = self.normalize(self.item_embeddings(item_ids.long()))  # (B, E)

        all_negative_items_embedding = self.normalize(
            self.item_embeddings(negative_item_list_idx), dim=2)  # (B, 100, E)

        distances_between_archor_and_negatives = (
            anchors.reshape(batch_size, 1, self.item_embeddings.embedding_dim)
            * all_negative_items_embedding
        ).sum(
            2
        )  # (B, 100)

        hardest_negative_items = torch.argmax(
            distances_between_archor_and_negatives, dim=1
        )  # (B,)

        negatives = all_negative_items_embedding[
            torch.arange(0, batch_size, device=item_ids.device), hardest_negative_items
        ]

        return negatives

    def select_negative_item_emb(self, item_ids: torch.Tensor, positive_item_ids: torch.Tensor,
                        negative_list_idx: List[torch.Tensor] = None):

        if random.random() < self.negative_random:
            negative_item_idx = negative_list_idx[:,0]
            negative_item_emb = self.normalize(self.item_embeddings(negative_item_idx.long()))
        else:
            negative_item_emb = self.get_harder_negative(item_ids, positive_item_ids, negative_list_idx)
        
        return negative_item_emb

    def similarity(self, itemA, itemB):
        return torch.cosine_similarity(itemA, itemB)

    def forward(self, item_ids: torch.Tensor, 
                      positive_item_ids: torch.Tensor,
                      negative_list_idx: List[torch.Tensor] = None) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        
        arch_item_emb = self.normalize(self.item_embeddings(item_ids.long()))
        positive_item_emb = self.normalize(self.item_embeddings(positive_item_ids.long()))

        if negative_list_idx is None:
            self.similarity(arch_item_emb, positive_item_emb)

        negative_item_emb = self.select_negative_item_emb(item_ids, positive_item_ids, negative_list_idx)

        return arch_item_emb, positive_item_emb, negative_item_emb

class RandomPolicy(BanditPolicy):
    def __init__(self, reward_model: nn.Module, seed: int = 42) -> None:
        super().__init__(None)
        self._rng = RandomState(seed)

    def _select_idx(
        self,
        arm_indices: List[int],
        arm_contexts: Tuple[np.ndarray, ...] = None,
        arm_scores: List[float] = None,
        pos: int = 0,
    ) -> Union[int, Tuple[int, float]]:

        n_arms = len(arm_indices)

        arm_probas = np.ones(n_arms) / n_arms

        action = self._rng.choice(n_arms, p=arm_probas)

        return action
