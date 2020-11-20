from mars_gym.data.dataset import (
    InteractionsDataset,
    InteractionsWithNegativeItemGenerationDataset,
)
from mars_gym.meta_config import *
from mercado_livre import data
import dataset
import warnings
warnings.filterwarnings("ignore")
mercado_livre_interaction = ProjectConfig(
    base_dir=data.BASE_DIR,
    prepare_data_frames_task=data.SessionInteractionDataFrame,
    dataset_class=InteractionsDataset,
    user_column=Column("SessionID", IOType.INDEXABLE),
    item_column=Column("ItemID", IOType.INDEXABLE),
    timestamp_column_name="Timestamp",
    available_arms_column_name="",
    other_input_columns=[
        Column("ItemIDHistory", IOType.INDEXABLE_ARRAY, same_index_as="ItemID"),
    ],
    output_column=Column("visit", IOType.NUMBER),
    recommender_type=RecommenderType.USER_BASED_COLLABORATIVE_FILTERING,
)

mercado_livre_interaction_with_negative_sample = ProjectConfig(
    base_dir=data.BASE_DIR,
    prepare_data_frames_task=data.SessionInteractionDataFrame,
    dataset_class=InteractionsWithNegativeItemGenerationDataset,
    user_column=Column("SessionID", IOType.INDEXABLE),
    item_column=Column("ItemID", IOType.INDEXABLE),
    timestamp_column_name="Timestamp",
    available_arms_column_name="",
    other_input_columns=[
        Column("ItemIDHistory", IOType.INDEXABLE_ARRAY, same_index_as="ItemID"),
    ],
    output_column=Column("visit", IOType.NUMBER),
    recommender_type=RecommenderType.USER_BASED_COLLABORATIVE_FILTERING,
)

mercado_livre_mf_bpr = ProjectConfig(
    base_dir=data.BASE_DIR,
    prepare_data_frames_task=data.SessionInteractionDataFrame,
    dataset_class=dataset.MFWithBPRDataset,
    user_column=Column("SessionID", IOType.INDEXABLE),
    item_column=Column("ItemID", IOType.INDEXABLE),
    timestamp_column_name="Timestamp",
    available_arms_column_name="",
    other_input_columns=[
        Column("ItemIDHistory", IOType.INDEXABLE_ARRAY, same_index_as="ItemID"),
    ],
    output_column=Column("visit", IOType.NUMBER),
    recommender_type=RecommenderType.USER_BASED_COLLABORATIVE_FILTERING,
)

mercado_livre_triplet =  ProjectConfig(
    base_dir=data.BASE_DIR,
    prepare_data_frames_task=data.IntraSessionInteractionsDataFrame,
    dataset_class=dataset.TripletWithNegativeListDataset,
    user_column=Column("SessionIDX", IOType.INDEXABLE),
    item_column=Column("ItemID", IOType.INDEXABLE),
    timestamp_column_name="Timestamp",
    other_input_columns=[Column("ItemID_B", IOType.INDEXABLE, same_index_as="ItemID")],
    metadata_columns=[Column("sub_a_b_all", IOType.INDEXABLE_ARRAY, same_index_as="ItemID")],
    output_column=Column("visit", IOType.NUMBER),
    auxiliar_output_columns=[Column("relative_pos", IOType.NUMBER), 
                            Column("total_ocr", IOType.NUMBER)],
    recommender_type=RecommenderType.USER_BASED_COLLABORATIVE_FILTERING,
)  

mercado_livre_rnn = ProjectConfig(
    base_dir=data.BASE_DIR,
    prepare_data_frames_task=data.SessionInteractionDataFrame,
    dataset_class=dataset.InteractionsDataset,
    user_column=Column("SessionID", IOType.INDEXABLE),
    item_column=Column("ItemID", IOType.INDEXABLE),
    timestamp_column_name="Timestamp",
    available_arms_column_name="",
    other_input_columns=[
        Column("ItemID_history", IOType.INDEXABLE_ARRAY, same_index_as="ItemID")
    ],
    output_column=Column("ItemID", IOType.INDEXABLE),
    recommender_type=RecommenderType.USER_BASED_COLLABORATIVE_FILTERING,
)

mercado_livre_narm = ProjectConfig(
    base_dir=data.BASE_DIR,
    prepare_data_frames_task=data.SessionInteractionDataFrame,
    dataset_class=dataset.InteractionsDataset,
    user_column=Column("SessionID", IOType.INDEXABLE),
    item_column=Column("ItemID", IOType.INDEXABLE),
    timestamp_column_name="Timestamp",
    available_arms_column_name="",
    other_input_columns=[
        Column("ItemID_history", IOType.INDEXABLE_ARRAY, same_index_as="ItemID"),
        Column("domain_idx_history", IOType.INDEXABLE_ARRAY),
        Column("timestamp_history", IOType.INT_ARRAY),
        Column("last_ItemID", IOType.INDEXABLE, same_index_as="ItemID"),
        Column("last_domain_idx", IOType.INDEXABLE, same_index_as="domain_idx_history"),        
        Column("last_ItemID_title", IOType.INT_ARRAY),
        Column("last_event_search", IOType.INT_ARRAY),
        Column("dense_features", IOType.FLOAT_ARRAY),
    ],
    output_column=Column("ItemID", IOType.INDEXABLE),
    recommender_type=RecommenderType.USER_BASED_COLLABORATIVE_FILTERING,
)

# mercado_livre_narm = ProjectConfig(
#     base_dir=data.BASE_DIR,
#     prepare_data_frames_task=data.SessionInteractionDataFrame,
#     dataset_class=dataset.InteractionsDataset,
#     user_column=Column("SessionID", IOType.INDEXABLE),
#     item_column=Column("ItemID", IOType.INDEXABLE),
#     timestamp_column_name="Timestamp",
#     available_arms_column_name="",
#     other_input_columns=[
#         Column("ItemID_history", IOType.INDEXABLE_ARRAY, same_index_as="ItemID"),
#         Column("domain_idx_history", IOType.INDEXABLE_ARRAY),
#         Column("timestamp_history", IOType.INT_ARRAY),
#         Column("last_ItemID", IOType.INDEXABLE, same_index_as="ItemID"),
#         Column("last_domain_idx", IOType.INDEXABLE, same_index_as="domain_idx_history"),        
#         Column("last_ItemID_title", IOType.INT_ARRAY),
#         Column("last_event_search", IOType.INT_ARRAY),
#         Column("dense_features", IOType.FLOAT_ARRAY),
#     ],
#     output_column=Column("ItemID", IOType.INDEXABLE),
#     recommender_type=RecommenderType.USER_BASED_COLLABORATIVE_FILTERING,
# )

mercado_livre_narm3 = ProjectConfig(
    base_dir=data.BASE_DIR,
    prepare_data_frames_task=data.SessionInteractionDataFrame,
    dataset_class=dataset.InteractionsDataset,
    user_column=Column("SessionID", IOType.INDEXABLE),
    item_column=Column("ItemID", IOType.INDEXABLE, same_index_as="ItemID_history"),
    timestamp_column_name="Timestamp",
    available_arms_column_name="",
    other_input_columns=[
        Column("ItemID_history", IOType.INDEXABLE_ARRAY),
        Column("domain_idx_history", IOType.INDEXABLE_ARRAY),
        Column("timestamp_history", IOType.INT_ARRAY),
        Column("last_ItemID", IOType.INDEXABLE, same_index_as="ItemID_history"),
        Column("last_domain_idx", IOType.INDEXABLE, same_index_as="domain_idx_history"),        
        Column("last_ItemID_title", IOType.INT_ARRAY),
        Column("last_event_search", IOType.INT_ARRAY),
        Column("dense_features", IOType.FLOAT_ARRAY),
    ],
    output_column=Column("ItemID", IOType.INDEXABLE, same_index_as="ItemID_history"),
    recommender_type=RecommenderType.USER_BASED_COLLABORATIVE_FILTERING,
)

mercado_livre_narm2 = ProjectConfig(
    base_dir=data.BASE_DIR,
    prepare_data_frames_task=data.SessionInteractionDataFrame,
    dataset_class=dataset.InteractionsDataset,
    user_column=Column("SessionID", IOType.INDEXABLE),
    item_column=Column("ItemID", IOType.INDEXABLE),
    timestamp_column_name="Timestamp",
    available_arms_column_name="",
    other_input_columns=[
        Column("ItemID_history", IOType.INDEXABLE_ARRAY, same_index_as="ItemID"),
        Column("step_history", IOType.INT_ARRAY)
    ],
    output_column=Column("ItemID", IOType.INDEXABLE),
    recommender_type=RecommenderType.USER_BASED_COLLABORATIVE_FILTERING,
)


mercado_livre_transformer = ProjectConfig(
    base_dir=data.BASE_DIR,
    prepare_data_frames_task=data.SessionInteractionDataFrame,
    dataset_class=dataset.InteractionsDataset,
    user_column=Column("SessionID", IOType.INDEXABLE),
    item_column=Column("ItemID", IOType.INDEXABLE),
    timestamp_column_name="Timestamp",
    available_arms_column_name="",
    other_input_columns=[
        Column("ItemID_history", IOType.INDEXABLE_ARRAY, same_index_as="ItemID"),
        Column("timestamp_history", IOType.INT_ARRAY),
        Column("category_idx_history", IOType.INDEXABLE_ARRAY),
        Column("domain_idx_history", IOType.INDEXABLE_ARRAY),
        Column("price_history", IOType.FLOAT_ARRAY),
        Column("last_event_search", IOType.INT_ARRAY),
    ],
    output_column=Column("ItemID", IOType.INDEXABLE),
    recommender_type=RecommenderType.USER_BASED_COLLABORATIVE_FILTERING,
)

mercado_livre_transformer2 = ProjectConfig(
    base_dir=data.BASE_DIR,
    prepare_data_frames_task=data.SessionInteractionDataFrame,
    dataset_class=dataset.InteractionsDataset,
    user_column=Column("SessionID", IOType.INDEXABLE),
    item_column=Column("ItemID", IOType.INDEXABLE),
    timestamp_column_name="Timestamp",
    available_arms_column_name="",
    other_input_columns=[
        Column("ItemID_history", IOType.INDEXABLE_ARRAY, same_index_as="ItemID"),
        Column("last_event_search", IOType.INT_ARRAY),
    ],
    output_column=Column("ItemID", IOType.INDEXABLE),
    recommender_type=RecommenderType.USER_BASED_COLLABORATIVE_FILTERING,
)
        # Column("last_ItemID", IOType.INDEXABLE, same_index_as="ItemID"),
        # Column("last_event_search", IOType.INT_ARRAY),
        # Column("last_event_type", IOType.INDEXABLE),        
        # Column("event_type_history", IOType.INDEXABLE_ARRAY),
        # Column("ItemID_history", IOType.INDEXABLE_ARRAY, same_index_as="ItemID"),
        # Column("timestamp_history", IOType.INT_ARRAY),
        # Column("category_id_history", IOType.INDEXABLE_ARRAY),
        # Column("domain_id_history", IOType.INDEXABLE_ARRAY),
        # Column("price_history", IOType.FLOAT_ARRAY),