from typing import Tuple, List, Union, Optional, Dict, Any

import functools
import numpy as np
import pandas as pd
import random
from torch.utils.data import Dataset

from mars_gym.meta_config import ProjectConfig, IOType, Column
from mars_gym.utils.index_mapping import map_array
from mars_gym.utils.utils import parallel_literal_eval
from mars_gym.data.dataset import (
    InteractionsDataset,
    InteractionsWithNegativeItemGenerationDataset,
)

class MFWithBPRDataset(Dataset):
    def __init__(
        self,
        data_frame: pd.DataFrame,
        embeddings_for_metadata: Optional[Dict[Any, np.ndarray]],
        project_config: ProjectConfig,
        index_mapping: Dict[str, Dict[Any, int]],
        *args,
        **kwargs
    ) -> None:
        self._project_config = project_config
        self._index_mapping  = index_mapping 
        self._input_columns: List[Column] = project_config.input_columns
        if project_config.item_is_input:
            self._item_input_index = self._input_columns.index(
                project_config.item_column
            )

        input_column_names = [input_column.name for input_column in self._input_columns]
        auxiliar_output_column_names = [
            auxiliar_output_column.name
            for auxiliar_output_column in project_config.auxiliar_output_columns
        ]
        self._data_frame = data_frame[
            set(
                input_column_names
                + [project_config.output_column.name]
                + auxiliar_output_column_names
            ).intersection(data_frame.columns)
        ]
        self._embeddings_for_metadata = embeddings_for_metadata
        self._data_key = kwargs['data_key']
        self.num_item = max(self._index_mapping[self._project_config.item_column.name].values()) + 1
        self.__getitem__([1,2])

    def __len__(self) -> int:
        return self._data_frame.shape[0]

    def _convert_dtype(self, value: np.ndarray, type: IOType) -> np.ndarray:
        if type == IOType.INDEXABLE:
            return value.astype(np.int64)
        if type == IOType.NUMBER:
            return value.astype(np.float64)
        if type in (IOType.INT_ARRAY, IOType.INDEXABLE_ARRAY):
            return np.array([np.array(v, dtype=np.int64) for v in value])
        if type == IOType.FLOAT_ARRAY:
            return np.array([np.array(v, dtype=np.float64) for v in value])
        return value

    def __getitem__(
        self, indices: Union[int, List[int], slice]
    ) -> Tuple[Tuple[np.ndarray, ...], Union[np.ndarray, Tuple[np.ndarray, ...]]]:
        if isinstance(indices, int):
            indices = [indices]
        rows: pd.Series = self._data_frame.iloc[indices]

        inputs = tuple(
            self._convert_dtype(rows[column.name].values, column.type)
            for column in self._input_columns
        )
        if (
            self._project_config.item_is_input
            and self._embeddings_for_metadata is not None
        ):
            item_indices = inputs[self._item_input_index]
            inputs += tuple(
                self._embeddings_for_metadata[column.name][item_indices]
                for column in self._project_config.metadata_columns
            )

        output = self._convert_dtype(
            rows[self._project_config.output_column.name].values,
            self._project_config.output_column.type,
        )
        if self._project_config.auxiliar_output_columns:
            output = tuple([output]) + tuple(
                self._convert_dtype(rows[column.name].values, column.type)
                for column in self._project_config.auxiliar_output_columns
            )

        if self._data_key == 'test_data':
          return inputs, output
        else:
          column = self._input_columns[1]
          inputs += tuple([self._convert_dtype(np.array([np.random.randint(self.num_item) for i in range(len(rows))]), column.type)])            
        
        # print(inputs)
        return inputs, output
        
class TripletWithNegativeListDataset(InteractionsDataset):
    def __init__(self,
                data_frame: pd.DataFrame,
                embeddings_for_metadata: Optional[Dict[Any, np.ndarray]],
                project_config: ProjectConfig,
                index_mapping: Dict[str, Dict[Any, int]],
                *args,
                **kwargs) -> None:
        data_frame = data_frame[data_frame[project_config.output_column.name] > 0]
        super().__init__(data_frame, embeddings_for_metadata, project_config, index_mapping)
        #
        self.all_items = list(index_mapping[project_config.item_column.name].values())
        self.len_all_items = len(self.all_items)
        self._negative_proportion = 1
        #np.array(list(range(self._data_frame[self._input_columns[0]].max())) )

        self.positive_interactions = embeddings_for_metadata.set_index('ItemID')
        self.positive_interactions['sub_a_b_all'] =  self.positive_interactions.sub_a_b_all.apply(lambda x: [] if str(x) == 'nan' else x)
        
        self.__getitem__([1, 2])

    def __len__(self) -> int:
        return self._data_frame.shape[0]

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
        

    def _get_negatives(self, all_positive):
        sub_negative = self.setdiff(self.all_items, all_positive)
        np.random.shuffle(sub_negative)
        return sub_negative[:2000]

    def setdiff(self, ticks, new_ticks):
        #return np.setdiff1d(ticks, new_ticks)
        return list(set(ticks) - set(new_ticks))

    def __getitem__(self, indices: Union[int, List[int]]) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray],
                                                                   list]:
        if isinstance(indices, int):
            indices = [indices]

        rows: pd.Series = self._data_frame.iloc[indices]
        
        item_arch      = rows[self._input_columns[1].name].values
        item_positive  = rows[self._input_columns[2].name].values
        #all_positives  = rows[self._input_columns[3].name].values
        arch_all_positives  = self.positive_interactions.loc[item_arch].sub_a_b_all.values
        pos_all_positives   = self.positive_interactions.loc[item_positive].sub_a_b_all.values
        
        all_positives       = [np.unique(a + p)  for a, p in zip(arch_all_positives, pos_all_positives)]
        all_pos             = rows[self._project_config.auxiliar_output_columns[0].name].values
        output              = rows[self._project_config.output_column.name].values
        
        if self._project_config.auxiliar_output_columns:
            output = tuple(rows[auxiliar_output_column.name].values
                                            for auxiliar_output_column in self._project_config.auxiliar_output_columns)
        
        if self._negative_proportion > 0:
            
            item_negative = [self._get_negatives(all_positive) for all_positive in all_positives]
            min_item      = np.min([len(l) for l in item_negative])
            item_negative = np.array([l[:min_item] for l in item_negative])

            return (item_arch, item_positive, item_negative, all_pos),output#), output
        else:
            return (item_arch, item_positive, all_pos), output
