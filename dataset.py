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
        self.num_item = len(self._index_mapping[self._project_config.item_column.name])
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

        if self._data_key == 'val_data':
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
        self.__getitem__([1])

    def __len__(self) -> int:
        return self._data_frame.shape[0]

    def _get_negatives(self, all_positive):
        sub_negative = self.setdiff(self.all_items, all_positive)
        np.random.shuffle(sub_negative)
        return sub_negative

    def setdiff(self, ticks, new_ticks):
        #return np.setdiff1d(ticks, new_ticks)
        return list(set(ticks) - set(new_ticks))

    def __getitem__(self, indices: Union[int, List[int]]) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray],
                                                                   list]:
        if isinstance(indices, int):
            indices = [indices]

        rows: pd.Series = self._data_frame.iloc[indices]
        
        item_arch      = rows[self._input_columns[2].name].values
        item_positive  = rows[self._input_columns[3].name].values
        all_positives  = rows[self._input_columns[4].name].values
        output         = rows[self._project_config.output_column.name].values
        
        if self._project_config.auxiliar_output_columns:
            output = tuple(rows[auxiliar_output_column.name].values
                                            for auxiliar_output_column in self._project_config.auxiliar_output_columns)
        
        if self._negative_proportion > 0:
            
            min_items = np.min([self.len_all_items-2*len(all_positive) for all_positive in all_positives])
            
            item_negative = [self._get_negatives(all_positive) for all_positive in all_positives]
            min_item      = np.min([len(l) for l in item_negative])
            item_negative = np.array([l[:min_item] for l in item_negative])

            #item_negative = np.array(
            #    [self._get_negatives(all_positive)[:np.min([1000, min_items])]
            #    for all_positive in all_positives], dtype=np.int64)
            #from IPython import embed; embed()
            return (item_arch, item_positive, item_negative),output#), output
        else:
            return (item_arch, item_positive), output
