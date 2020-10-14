from mars_gym.data.dataset import (
    InteractionsDataset,
    InteractionsWithNegativeItemGenerationDataset,
)
from mars_gym.meta_config import *
from globo import data
import dataset

globo_interaction = ProjectConfig(
    base_dir=data.BASE_DIR,
    prepare_data_frames_task=data.SessionInteractionDataFrame,
    dataset_class=InteractionsDataset,
    user_column=Column("SessionID", IOType.INDEXABLE),
    item_column=Column("ItemID", IOType.INDEXABLE),
    timestamp_column_name="Timestamp",
    available_arms_column_name="AvailableItems",
    other_input_columns=[
        Column("ItemIDHistory", IOType.INDEXABLE_ARRAY, same_index_as="ItemID"),
    ],
    output_column=Column("visit", IOType.NUMBER),
    recommender_type=RecommenderType.USER_BASED_COLLABORATIVE_FILTERING,
)

globo_interaction_with_negative_sample = ProjectConfig(
    base_dir=data.BASE_DIR,
    prepare_data_frames_task=data.SessionInteractionDataFrame,
    dataset_class=InteractionsWithNegativeItemGenerationDataset,
    user_column=Column("SessionID", IOType.INDEXABLE),
    item_column=Column("ItemID", IOType.INDEXABLE),
    timestamp_column_name="Timestamp",
    available_arms_column_name="AvailableItems",
    other_input_columns=[
        Column("ItemIDHistory", IOType.INDEXABLE_ARRAY, same_index_as="ItemID"),
    ],
    output_column=Column("visit", IOType.NUMBER),
    recommender_type=RecommenderType.USER_BASED_COLLABORATIVE_FILTERING,
)


globo_rnn = ProjectConfig(
    base_dir=data.BASE_DIR,
    prepare_data_frames_task=data.SessionInteractionDataFrame,
    dataset_class=dataset.InteractionsDataset,
    user_column=Column("SessionID", IOType.INDEXABLE),
    item_column=Column("ItemID", IOType.INDEXABLE),
    timestamp_column_name="Timestamp",
    available_arms_column_name="AvailableItems",
    other_input_columns=[
        Column("ItemIDHistory", IOType.INDEXABLE_ARRAY, same_index_as="ItemID"),
    ],
    output_column=Column("ItemID", IOType.INDEXABLE),
    recommender_type=RecommenderType.USER_BASED_COLLABORATIVE_FILTERING,
)


globo_mf_bpr = ProjectConfig(
    base_dir=data.BASE_DIR,
    prepare_data_frames_task=data.SessionInteractionDataFrame,
    dataset_class=dataset.MFWithBPRDataset,
    user_column=Column("SessionID", IOType.INDEXABLE),
    item_column=Column("ItemID", IOType.INDEXABLE),
    timestamp_column_name="Timestamp",
    available_arms_column_name="AvailableItems",
    other_input_columns=[
        Column("ItemIDHistory", IOType.INDEXABLE_ARRAY, same_index_as="ItemID"),
    ],
    output_column=Column("visit", IOType.NUMBER),
    recommender_type=RecommenderType.USER_BASED_COLLABORATIVE_FILTERING,
)

globo_triplet = ProjectConfig(
    base_dir=data.BASE_DIR,
    prepare_data_frames_task=data.IntraSessionInteractionsDataFrame,
    dataset_class=dataset.TripletWithNegativeListDataset,
    user_column=Column("SessionID", IOType.INDEXABLE),
    item_column=Column("ItemID", IOType.INDEXABLE),
    timestamp_column_name="Timestamp",
    other_input_columns=[Column("ItemID_A", IOType.INDEXABLE, same_index_as="ItemID"), 
                        Column("ItemID_B", IOType.INDEXABLE, same_index_as="ItemID"), 
                        Column("sub_a_b", IOType.INDEXABLE_ARRAY, same_index_as="ItemID")],
    output_column=Column("visit", IOType.NUMBER),
    auxiliar_output_columns=[Column("relative_pos", IOType.NUMBER), 
                            Column("total_ocr", IOType.NUMBER), 
                            Column("prob", IOType.NUMBER)],
    recommender_type=RecommenderType.USER_BASED_COLLABORATIVE_FILTERING,
)  

