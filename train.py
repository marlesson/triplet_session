
from mars_gym.simulation.training import SupervisedModelTraining, DummyTraining
from loss import RelativeTripletLoss, ContrastiveLoss
import torch
import torch.nn as nn
import luigi
import numpy as np
from typing import Type, Dict, List, Optional, Tuple, Union, Any, cast
from mars_gym.utils.files import (
    get_index_mapping_path,
)
import pickle

from sklearn import manifold
from time import time
import os
import pandas as pd
from mars_gym.model.agent import BanditAgent
from torch.utils.data.dataset import Dataset, ChainDataset
import numpy as np
import itertools
from scipy.sparse import csr_matrix
from scipy.sparse import csr_matrix
from pandas.api.types import CategoricalDtype
from sklearn.metrics.pairwise import cosine_similarity
from plot import plot_tsne
from mars_gym.data.dataset import (
    preprocess_interactions_data_frame,
    preprocess_metadata_data_frame,
    literal_eval_array_columns,
    InteractionsDataset,
)
from mars_gym.utils.index_mapping import (
    transform_with_indexing,
)

TORCH_LOSS_FUNCTIONS = dict(
    mse=nn.MSELoss,
    nll=nn.NLLLoss,
    bce=nn.BCELoss,
    mlm=nn.MultiLabelMarginLoss,
    relative_triplet=RelativeTripletLoss,    
    contrastive_loss=ContrastiveLoss,
)


class RandomTraining(DummyTraining):
    '''
    Most Popular Model
    '''
    def fit(self, df_train: pd.DataFrame):
        pass

    def get_scores(self, agent: BanditAgent, ob_dataset: Dataset) -> List[float]:
        print("get_scores...")

        item_idx = ob_dataset._data_frame.ItemID.values
        scores   = list(np.random.rand(len(item_idx)))

        return scores

class MostPopularTraining(DummyTraining):
    '''
    Most Popular Model
    '''
    def fit(self, df_train: pd.DataFrame):
        print("fit...")

        self.item_counts = df_train.ItemID.value_counts()
        self.item_counts.loc[0] = 0

    def get_scores(self, agent: BanditAgent, ob_dataset: Dataset) -> List[float]:
        print("get_scores...")

        item_idx = ob_dataset._data_frame.ItemID.values
        scores   = self.item_counts.loc[item_idx].fillna(0).values
        return scores

class CoOccurrenceTraining(DummyTraining):
    '''
    Most Popular Model
    '''
    def fit(self, df_train: pd.DataFrame):
        print("fit...")
        
        item_idx = np.unique(df_train.ItemID.values)
        lists    = list(df_train.ItemIDHistory)
        cooc_matrix, to_id = self.create_co_occurences_matrix(item_idx, lists)

        self.columns_coocc = to_id
        self.cooc_matrix   = cooc_matrix


    def create_co_occurences_matrix(self, allowed_words, documents):
        word_to_id = dict(zip(allowed_words, range(len(allowed_words))))
        documents_as_ids = [np.sort([word_to_id[w] for w in doc if w in word_to_id]).astype('uint32') for doc in documents]
        row_ind, col_ind = zip(*itertools.chain(*[[(i, w) for w in doc] for i, doc in enumerate(documents_as_ids)]))
        data = np.ones(len(row_ind), dtype='uint32')  # use unsigned int for better memory utilization
        max_word_id = max(itertools.chain(*documents_as_ids)) + 1
        docs_words_matrix = csr_matrix((data, (row_ind, col_ind)), shape=(len(documents_as_ids), max_word_id))  # efficient arithmetic operations with CSR * CSR
        words_cooc_matrix = docs_words_matrix.T * docs_words_matrix  # multiplying docs_words_matrix with its transpose matrix would generate the co-occurences matrix
        words_cooc_matrix.setdiag(0)

        return words_cooc_matrix, word_to_id 
        
    def get_scores(self, agent: BanditAgent, ob_dataset: Dataset) -> List[float]:
        print("get_scores...")
        #

        last_items = list(ob_dataset._data_frame.ItemIDHistory.apply(lambda l: l[0]))
        next_items = list(ob_dataset._data_frame.ItemID.values)

        scores = []
        for last_item, next_item in zip(last_items, next_items):
            scores.append(self.get_score(last_item, next_item))

        return scores

    def get_score(self, item_a: int, item_b: int):
        try:
            item_a_idx = self.columns_coocc[item_a]
            item_b_idx = self.columns_coocc[item_b]

            return self.cooc_matrix[item_a_idx, item_b_idx]
        except:
            return 0

    # def run_evaluate_task(self) -> None:
    #     os.system(
    #         "PYTHONPATH=. luigi --module mars_gym.evaluation.task EvaluateTestSetPredictions "
    #         f"--model-task-class train.CoOccurrenceTraining --model-task-id {self.task_id} --only-new-interactions --only-exist-items --local-scheduler"
    #     )     

class IKNNTraining(DummyTraining):
    '''
    Most Popular Model
    '''
    def fit(self, df_train: pd.DataFrame):
        print("fit...")
        
        item_idx = np.unique(df_train.ItemID.values)
        lists    = list(df_train.ItemIDHistory)
        sparse_matrix = self.create_sparse_matrix(df_train)

        self.matrix_item_idx  = dict(zip(item_idx, list(range(len(item_idx)))))
        self.sparse_matrix    = sparse_matrix
        self.cos_matrix       = cosine_similarity(sparse_matrix)

    def create_sparse_matrix(self, df: pd.DataFrame):
                
        session_c = CategoricalDtype(sorted(df.SessionID.unique()), ordered=True)
        item_c    = CategoricalDtype(sorted(df.ItemID.unique()), ordered=True)

        col = df.SessionID.astype(session_c).cat.codes
        row = df.ItemID.astype(item_c).cat.codes
        sparse_matrix = csr_matrix((df["visit"], (row, col)), \
                                shape=(item_c.categories.size, session_c.categories.size))
        return sparse_matrix
        
    def get_scores(self, agent: BanditAgent, ob_dataset: Dataset) -> List[float]:
        print("get_scores...")

        last_items = list(ob_dataset._data_frame.ItemIDHistory.apply(lambda l: l[0]))
        next_items = list(ob_dataset._data_frame.ItemID.values)

        scores = []
        for last_item, next_item in zip(last_items, next_items):
            scores.append(self.get_score(last_item, next_item))

        return scores

    def get_score(self, item_a: int, item_b: int):
        
        try:
            #dot = self.sparse_matrix[self.matrix_item_idx[item_b]] * self.sparse_matrix[self.matrix_item_idx[item_a]].T
            #sim = np.array(dot.T.todense())[0]
            sim = self.cos_matrix[self.matrix_item_idx[item_b]][self.matrix_item_idx[item_a]]
            return sim#[0]#[self.matrix_item_idx[item_b]]
        except:
            return 0

class TripletPredTraining(DummyTraining):
    path_item_embedding:  str = luigi.Parameter()
    from_index_mapping:  str = luigi.Parameter()

    def load_embs(self):
        # Load extern index mapping
        if os.path.exists(self.from_index_mapping):
            with open(self.from_index_mapping, "rb") as f:
                index_mapping = pickle.load(f)    

        embs = np.loadtxt(self.path_item_embedding)

        return embs, index_mapping

    '''
    Most Popular Model
    '''
    def fit(self, df_train: pd.DataFrame):
        print("fit...")
        
        embs, index_mapping = self.load_embs()
        self._embs = embs
        self._from_index_mapping = index_mapping

        
    def get_scores(self, agent: BanditAgent, ob_dataset: Dataset) -> List[float]:
        print("get_scores...")

        last_items = list(ob_dataset._data_frame.ItemIDHistory.apply(lambda l: l[0]))
        next_items = list(ob_dataset._data_frame.ItemID.values)

        scores = []
        for last_item, next_item in zip(last_items, next_items):
            scores.append(self.get_score(last_item, next_item))

        return scores

    def get_score(self, item_a: int, item_b: int):
        try:
            _item_a = self._from_index_mapping[self.project_config.item_column.name][self.reverse_index_mapping[self.project_config.item_column.name][item_a]]
            _item_b = self._from_index_mapping[self.project_config.item_column.name][self.reverse_index_mapping[self.project_config.item_column.name][item_b]]

            sim = cosine_similarity([self._embs[_item_a]], [self._embs[_item_b]])[0][0]
            return sim
        except:
            return 0


class TripletTraining(SupervisedModelTraining):
    loss_function:  str = luigi.ChoiceParameter(choices=["relative_triplet", "contrastive_loss"], default="relative_triplet")
    save_item_embedding_tsv: bool = luigi.BoolParameter(default=False)

    def after_fit(self):
        if self.save_item_embedding_tsv:
            print("save_item_embedding_tsv...")
            self.export_embs()

    def export_embs(self):
        module = self.get_trained_module()

        item_embeddings: np.ndarray = module.item_embeddings.weight.data.cpu().numpy()
        np.savetxt(self.output().path+"/item_embeddings.npy", item_embeddings, delimiter="\t")

        self.export_tsne_file(item_embeddings, None)

    def export_tsne_file(self, embs, metadata):
        t0 = time()
        tsne = manifold.TSNE(n_components=2, init='random', random_state=0)
        Y = tsne.fit_transform(embs)
        t1 = time()
        print("circles in %.2g sec" % (t1 - t0))

        #metadata[self.tsne_column_plot].reset_index().index
        color = None
        
        plot_tsne(Y[:, 0], Y[:, 1], color).savefig(
            os.path.join(self.output().path, "item_embeddings.jpg"))

    @property
    def metadata_data_frame(self) -> Optional[pd.DataFrame]:
        if not hasattr(self, "_metadata_data_frame"):
            self._metadata_data_frame = (
                pd.read_csv(self.metadata_data_frame_path)
                if self.metadata_data_frame_path
                else None
            )

            self._metadata_data_frame['ItemID'] = self._metadata_data_frame['ItemID'].astype(str)
                        
            if self._metadata_data_frame is not None:
                literal_eval_array_columns(
                    self._metadata_data_frame, self.project_config.metadata_columns
                )

            transform_with_indexing(
                self._metadata_data_frame, self.index_mapping, self.project_config
            )
            self._metadata_data_frame.to_csv('_metadata_data_frame.csv')
        return self._metadata_data_frame

    @property
    def embeddings_for_metadata(self) -> Optional[Dict[str, np.ndarray]]:
        return self.metadata_data_frame

    def _get_loss_function(self):
        return TORCH_LOSS_FUNCTIONS[self.loss_function](**self.loss_function_params)

 