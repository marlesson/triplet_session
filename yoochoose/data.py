import luigi
import pandas as pd
import numpy as np
import os
import pickle
list_sum = sum

from mars_gym.data.task import BasePrepareDataFrames, BasePySparkTask
from mars_gym.data.utils import DownloadDataset
from mars_gym.data.dataset import (
    InteractionsDataset,
    InteractionsWithNegativeItemGenerationDataset,
)
import random
from typing import Tuple, List, Union, Callable, Optional, Set, Dict, Any
from mars_gym.meta_config import *

from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import collect_set, collect_list, lit, sum, udf, concat_ws, col, count, abs, date_format, \
    from_utc_timestamp, expr, min, max
from pyspark.sql.functions import col, udf, size
from pyspark.sql.types import *
from pyspark.sql import functions as F
from pyspark.sql import Window
from pyspark.sql.functions import explode, posexplode
from itertools import chain
from datetime import datetime, timedelta

OUTPUT_PATH: str = os.environ[
    "OUTPUT_PATH"
] if "OUTPUT_PATH" in os.environ else os.path.join("output")
BASE_DIR: str = os.path.join(OUTPUT_PATH, "yoochoose")
DATASET_DIR: str = os.path.join(OUTPUT_PATH, "yoochoose", "dataset")

BASE_DATASET_FILE : str = os.path.join(OUTPUT_PATH, "yoochoose", "yoochoose", 'yoochoose-clicks.dat')

## AUX
pad_history = F.udf(
    lambda arr, size: list(reversed(([0] * (size - len(arr[:-1][:size])) + arr[:-1][:size]))), 
    ArrayType(IntegerType())
)

def sample_items(item, item_list, size_available_list):
    return random.sample(item_list, size_available_list - 1) + [item]

def udf_sample_items(item_list, size_available_list):
    return udf(lambda item: sample_items(item, item_list, size_available_list), ArrayType(IntegerType()))


def concat(type):
    def concat_(*args):
        return list(set(chain.from_iterable((arg if arg else [] for arg in args))))
    return udf(concat_, ArrayType(type))

#####


################################## Supervised ######################################

class SessionPrepareDataset(BasePySparkTask):
    sample_days: int = luigi.IntParameter(default=16)
    history_window: int = luigi.IntParameter(default=10)
    size_available_list: int = luigi.IntParameter(default=100)
    minimum_interactions: int = luigi.IntParameter(default=5)
    
    def output(self):
        return luigi.LocalTarget(os.path.join(DATASET_DIR, "dataset_prepared_sample={}_win={}_list={}_min_i={}.csv"\
                    .format(self.sample_days, self.history_window, self.size_available_list, self.minimum_interactions),))

    def add_history(self, df):
        
        w = Window.partitionBy('SessionID').orderBy('Timestamp')#.rangeBetween(Window.currentRow, 5)

        df = df.withColumn(
            'ItemIDHistory', F.collect_list('ItemID').over(w)
        ).where(size(col("ItemIDHistory")) >= 2)#\

        df = df.withColumn('ItemIDHistory', pad_history(df.ItemIDHistory, lit(self.history_window)))

        return df

    def filter(self, df):
        # filter date
        max_timestamp = df.select(max(col('Timestamp'))).collect()[0]['max(Timestamp)']
        init_timestamp = max_timestamp - timedelta(days = self.sample_days)
        df         = df.filter(col('Timestamp') >= init_timestamp).cache()

        # Filter minin interactions
        df_item    = df.groupBy("ItemID").count()
        df_item    = df_item.filter(col('count') >= self.minimum_interactions)

        # Filter session size
        df_session    = df.groupBy("SessionID").count()
        df_session    = df_session.filter(col('count') >= 2)

        df = df \
            .join(df_item, "ItemID", how="inner") \
            .join(df_session, "SessionID", how="inner")

        return df

    def add_available_items(self, df):
        all_items = list(df.select("ItemID").dropDuplicates().toPandas()["ItemID"])

        df = df.withColumn('AvailableItems', udf_sample_items(all_items, self.size_available_list)(col("ItemID")))

        return df

    def main(self, sc: SparkContext, *args):
        os.makedirs(DATASET_DIR, exist_ok=True)

        spark    = SparkSession(sc)
        df = spark.read.csv(BASE_DATASET_FILE, header=False, inferSchema=True)
        df = df.withColumnRenamed("_c0", "SessionID")\
            .withColumnRenamed("_c1", "Timestamp")\
            .withColumnRenamed("_c2", "ItemID")\
            .withColumnRenamed("_c3", "Category")\
            .orderBy(col('Timestamp')).select("SessionID", "ItemID", "Timestamp").filter(col('Timestamp') < '2017-10-16 24:59:59')
                    
        # Drop duplicate item in that same session
        df = df.dropDuplicates(['SessionID', 'ItemID'])

        df = self.filter(df)
        df = self.add_history(df)
        df = self.add_available_items(df)

        df = df.withColumn('visit',lit(1))

        df.toPandas().to_csv(self.output().path, index=False)

class SessionInteractionDataFrame(BasePrepareDataFrames):
    sample_days: int = luigi.IntParameter(default=16)
    history_window: int = luigi.IntParameter(default=10)
    size_available_list: int = luigi.IntParameter(default=100)
    days_test: int = luigi.IntParameter(default=1)
    index_mapping_path: str = luigi.Parameter(default=None)

    def requires(self):
        return SessionPrepareDataset(sample_days=self.sample_days, history_window=self.history_window, size_available_list=self.size_available_list)

    @property
    def timestamp_property(self) -> str:
        return "Timestamp"

    @property
    def item_property(self) -> str:
        return "ItemID"

    @property
    def dataset_dir(self) -> str:
        return DATASET_DIR

    @property
    def read_data_frame_path(self) -> pd.DataFrame:
        return self.input().path

    def read_data_frame(self) -> pd.DataFrame:
        df = pd.read_csv(self.read_data_frame_path)#.sample(10000)
        
        return df

    def transform_data_frame(self, df: pd.DataFrame, data_key: str) -> pd.DataFrame:
        return df

    def time_train_test_split(
        self, df: pd.DataFrame, test_size: float
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df[self.timestamp_property] = pd.to_datetime(df[self.timestamp_property])

        if self.timestamp_property:
            df = df.sort_values(self.timestamp_property)
        
        cutoff_date = df[self.timestamp_property].iloc[-1] - pd.Timedelta(days=self.days_test)

        return df[df[self.timestamp_property] < cutoff_date], df[df[self.timestamp_property] >= cutoff_date]


#################################  Triplet ##############################

def serach_positive(uid, df, max_deep = 1, deep = 0, list_pos = []):
    if uid not in list_pos:
        list_pos.append(uid)
        #print(list_pos)
       
        if deep >= max_deep:
            return df.loc[uid].sub_a_b
        else:
            l = [serach_positive(i, df, max_deep, deep+1, list_pos) for i in df.loc[uid].sub_a_b]
            l.append(df.loc[uid].sub_a_b)
            return list_sum(l, [])
    else:
        return []

class CreateIntraSessionInteractionDataset(BasePySparkTask):
    sample_days: int = luigi.IntParameter(default=16)
    history_window: int = luigi.IntParameter(default=10)
    size_available_list: int = luigi.IntParameter(default=100)
    minimum_interactions: int = luigi.IntParameter(default=5)
    max_itens_per_session: int = luigi.IntParameter(default=15)
    min_itens_interactions: int = luigi.IntParameter(default=3)
    max_relative_pos: int = luigi.IntParameter(default=3)
    pos_max_deep: int = luigi.IntParameter(default=1)

    # def requires(self):
    #     return SessionPrepareDataset(sample_days=self.sample_days, history_window=self.history_window, size_available_list=self.size_available_list)

    def output(self):
        return luigi.LocalTarget(os.path.join(DATASET_DIR, "indexed_intra_session_train_%d_w=%d_l=%d_m=%d_s=%d_i=%d_p=%d" % (self.sample_days, self.history_window, 
            self.size_available_list, self.minimum_interactions, self.max_itens_per_session, self.min_itens_interactions, self.max_relative_pos))),\
                luigi.LocalTarget(os.path.join(DATASET_DIR, "item_positive_interaction_%d_w=%d_l=%d_m=%d_s=%d_i=%d_p=%d.csv" % (self.sample_days, self.history_window, 
            self.size_available_list, self.minimum_interactions, self.max_itens_per_session, self.min_itens_interactions, self.max_relative_pos))),\
                luigi.LocalTarget(os.path.join(DATASET_DIR, "item_id_index_%d_w=%d_l=%d_m=%d_s=%d_i=%d_p=%d.csv" % (self.sample_days, self.history_window, 
            self.size_available_list, self.minimum_interactions, self.max_itens_per_session, self.min_itens_interactions, self.max_relative_pos))),\
                luigi.LocalTarget(os.path.join(DATASET_DIR, "session_index_%d_w=%d_l=%d_m=%d_s=%d_i=%d_p=%d.csv" % (self.sample_days, self.history_window, 
            self.size_available_list, self.minimum_interactions, self.max_itens_per_session, self.min_itens_interactions, self.max_relative_pos)))


    def get_df_tuple_probs(self, df):

        df_tuple_count  = df.groupby("ItemID_A","ItemID_B").count()
        df_count        = df.groupby("ItemID_A").count()\
                            .withColumnRenamed("count", "total")\
                            .withColumnRenamed("ItemID_A", "_ItemID_A")

        df_join         = df_tuple_count.join(df_count, df_tuple_count.ItemID_A == df_count._ItemID_A).cache()
        df_join         = df_join.withColumn("prob", col("count")/col("total"))

        df_join  = df_join.select("ItemID_A", 'ItemID_B', 'count', 'total', 'prob')\
                    .withColumnRenamed("ItemID_A", "_ItemID_A")\
                    .withColumnRenamed("ItemID_B", "_ItemID_B")\
                    .withColumnRenamed("count", "total_ocr_dupla")\
                    .withColumnRenamed("total", "total_ocr").cache()

        return df_join
    
    def add_positive_interactions(self, df):
        
        # Filter more then 1 ocurrence for positive interactions
        df = df.filter(col("total_ocr_dupla") >= 1)
    
        df = df\
            .groupby("ItemID_A")\
            .agg(F.collect_set("ItemID_B").alias("sub_a_b"))

        # df_b = df\
        #     .groupby("ItemID_B")\
        #     .agg(F.collect_set("ItemID_A").alias("sub_b"))

        # df = df.join(df_a, "ItemID_A").join(df_b, "ItemID_B").cache()

        # concat_int_arrays = concat(IntegerType())
        # df = df.withColumn("sub_a_b", concat_int_arrays("sub_a", "sub_b"))#.show(truncate=False)
        # return df
        df = df.withColumnRenamed("ItemID_A", "ItemID")
        #df = df.withColumn("ItemID_COPY",df.ItemID)

        df = df.toPandas().set_index('ItemID')
        print(df)

        sub_pos = []
        for i, row in df.iterrows():
            l = serach_positive(row.name, df, max_deep = self.pos_max_deep, deep=0, list_pos=[])
            sub_pos.append(list(np.unique(l)))
        
        df['sub_a_b_all'] = sub_pos

        return df

    def main(self, sc: SparkContext, *args):
        os.makedirs(DATASET_DIR, exist_ok=True)

        #parans
        min_itens_per_session  = 2
        max_itens_per_session  = self.max_itens_per_session
        min_itens_interactions = self.min_itens_interactions # Tupla interactions
        max_relative_pos       = self.max_relative_pos

        spark    = SparkSession(sc)
        df = spark.read.csv(BASE_DATASET_FILE, header=False, inferSchema=True)
        df = df.withColumnRenamed("_c0", "SessionID")\
            .withColumnRenamed("_c1", "Timestamp")\
            .withColumnRenamed("_c2", "ItemID")\
            .withColumnRenamed("_c3", "Category")\
            .orderBy(col('Timestamp')).select("SessionID", "ItemID", "Timestamp")#.filter(col('Timestamp') < '2017-10-16 24:59:59')
               
        # Drop duplicate item in that same session
        df       = df.dropDuplicates(['SessionID', 'ItemID'])

        # filter date
        max_timestamp = df.select(max(col('Timestamp'))).collect()[0]['max(Timestamp)']
        init_timestamp = max_timestamp - timedelta(days = self.sample_days)
        df         = df.filter(col('Timestamp') >= init_timestamp).cache()

        df       = df.groupby("SessionID").agg(
                    max("Timestamp").alias("Timestamp"),
                    collect_list("ItemID").alias("ItemIDs"),
                    count("ItemID").alias("total"))

        # Filter Interactions
        df = df.filter(df.total >= min_itens_per_session).cache()

        # Filter position in list
        df_pos = df.select(col('SessionID').alias('_SessionID'),
                                    posexplode(df.ItemIDs))

        # Explode A
        df = df.withColumn("ItemID_A", explode(df.ItemIDs))
        df = df.join(df_pos,
                    (df.SessionID == df_pos._SessionID) & (df.ItemID_A == df_pos.col))\
                .select('SessionID', 'Timestamp', 'ItemID_A', 'pos', 'ItemIDs')\
                .withColumnRenamed('pos', 'pos_A')

        # Explode B
        df = df.withColumn("ItemID_B", explode(df.ItemIDs))
        df = df.join(df_pos,
                    (df.SessionID == df_pos._SessionID) & (df.ItemID_B == df_pos.col))\
                .withColumnRenamed('pos', 'pos_B')

        df = df.withColumn("relative_pos", abs(df.pos_A - df.pos_B))

        # Filter  distincts
        df = df.select('SessionID', 'Timestamp', 'ItemID_A', 'pos_A', 
                        'ItemID_B', 'pos_B', 'relative_pos')\
                .distinct()\
                .filter(df.ItemID_A != df.ItemID_B).cache()

        # # Filter duplicates
        # udf_join = F.udf(lambda s,x,y : "_".join(sorted([str(s), str(x),str(y)])) , StringType())
        # df = df.withColumn('key', udf_join('SessionID', 'ItemID_A','ItemID_B'))
        # df = df.dropDuplicates(["key"])

        # Calculate and filter probs ocorrence
        df_probs = self.get_df_tuple_probs(df)
        df = df.join(df_probs, (df.ItemID_A == df_probs._ItemID_A) & (df.ItemID_B == df_probs._ItemID_B))

        # Add positive interactoes
        df_positive = self.add_positive_interactions(df)

        # Filter confidence
        df = df.filter(col("total_ocr_dupla") >= min_itens_interactions)\
               .filter(col("relative_pos") <= max_relative_pos)\
               .filter(col("pos_A") <= self.max_itens_per_session)
                   

        # df = df.select("SessionID", 'Timestamp', 'ItemID_A', 'pos_A',
        #                 'ItemID_B', 'pos_B', 'relative_pos', 
        #                 'total_ocr', 'total_ocr_dupla', 'prob', 'sub_a_b')\
        #         .dropDuplicates(['ItemID_A', 'ItemID_B', 'relative_pos']) # TODO is it right?
        df = df.select("SessionID", 'Timestamp', 'ItemID_A', 'ItemID_B', 'relative_pos', 
                        'total_ocr', 'total_ocr_dupla')\
                .dropDuplicates(['ItemID_A', 'ItemID_B', 'relative_pos']) # TODO is it right?

        df.select("ItemID_A").dropDuplicates().toPandas().to_csv(self.output()[2].path, index_label="item_idx")
        df.select("SessionID").dropDuplicates().toPandas().to_csv(self.output()[3].path, index_label="session_idx")
        df.write.parquet(self.output()[0].path)
        df_positive.to_csv(self.output()[1].path)

class IntraSessionInteractionsDataFrame(BasePrepareDataFrames):
    sample_days: int = luigi.IntParameter(default=16)
    max_itens_per_session: int = luigi.IntParameter(default=15)
    min_itens_interactions: int = luigi.IntParameter(default=3)
    max_relative_pos: int = luigi.IntParameter(default=3)
    days_test: int = luigi.IntParameter(default=1)
    pos_max_deep: int = luigi.IntParameter(default=1)    
    filter_first_interaction: bool = luigi.BoolParameter(default=False)

    def requires(self):
        return CreateIntraSessionInteractionDataset(
                        max_itens_per_session=self.max_itens_per_session,
                        sample_days=self.sample_days,
                        min_itens_interactions=self.min_itens_interactions,
                        max_relative_pos=self.max_relative_pos,
                        pos_max_deep=self.pos_max_deep)
                        
    @property
    def timestamp_property(self) -> str:
        return "Timestamp"

    @property
    def dataset_dir(self) -> str:
        return DATASET_DIR

    def read_data_frame(self) -> pd.DataFrame:
        df = pd.read_parquet(self.read_data_frame_path)#.sample(10000)

        # TODO
        if self.filter_first_interaction:
            df = df.groupby(['ItemID_A', 'ItemID_B']).head(1).reset_index(drop=True)
        
        #df["ItemID"]        = df.ItemID_A
        #df['sub_a_b']        = df['sub_a_b'].apply(list)
        df['available_arms'] = None
        df["visit"]          = 1
        
        df_session           = df[['SessionID']].drop_duplicates().reset_index().rename(columns={"index":'SessionIDX'})
        
        df = df.merge(df_session).drop(['SessionID'], axis=1)
        df = df.rename(columns={"ItemID_A":'ItemID'})

        return df

    @property
    def metadata_data_frame_path(self) -> Optional[str]:
        return self.input()[1].path
        
    @property
    def read_data_frame_path(self) -> pd.DataFrame:
        return self.input()[0].path

    def transform_data_frame(self, df: pd.DataFrame, data_key: str) -> pd.DataFrame:
        print(data_key)
        print(df.describe())
        
        return df

    def time_train_test_split(
        self, df: pd.DataFrame, test_size: float
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df[self.timestamp_property] = pd.to_datetime(df[self.timestamp_property])

        if self.timestamp_property:
            df = df.sort_values(self.timestamp_property)
        
        cutoff_date = df[self.timestamp_property].iloc[-1] - pd.Timedelta(days=self.days_test)

        return df[df[self.timestamp_property] < cutoff_date], df[df[self.timestamp_property] >= cutoff_date]

################################## Interactions ######################################

